"""
shared/crash_guard.py — keep one failing module from taking down the whole Suite, and make the
crashes we CAN'T prevent finally leave a trace.

WHY THIS EXISTS
  Heavy Suite modules (SD.UI: MediaPipe + engine; LoRA Health: torch/SVD) run native code in the
  Suite process. Two failure classes have been killing the entire app:

    1. Uncatchable NATIVE crashes (a ROCm/torch OOM segfault) — the process just dies, no Python
       traceback, nothing in any log. Python's try/except cannot catch these.
    2. Catchable PYTHON exceptions raised during a module's open/init — historically these propagated
       out of the Qt slot and, on some PySide6/Qt builds, aborted the app instead of failing gracefully.

  crash_guard addresses both as far as is possible:
    • faulthandler writes the Python stack to a crash log on a fatal native signal (SIGSEGV/SIGABRT).
      It can't stop a native crash, but it turns "app vanished, no clue" into a diagnosable stack — so
      the next OOM tells us exactly which module was mid-init. Paired with vram_guard (which PREVENTS
      the OOM) this closes the loop.
    • a global sys.excepthook logs the traceback and shows a dialog instead of dying silently.
    • guard_open() wraps a module's construction so a Python-level failure fails THAT module (logged +
      dialog + rollback) rather than the whole Suite.

  True process isolation of the native libs (running MediaPipe/torch in a child process so even a
  segfault can't reach the Suite) is the deeper follow-up; this delivers the containment + diagnostics
  that make the remaining native OOMs both rarer (via vram_guard) and, when they do happen, visible.

  stdlib + a lazy PySide6 import only inside the dialog helpers (so this is importable headless).
"""
from __future__ import annotations

import atexit
import datetime
import faulthandler
import os
import sys
import threading
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_SUITE_ROOT = os.path.dirname(_HERE)
_LOG_DIR = os.path.join(_SUITE_ROOT, "logs")
_CRASH_LOG = os.path.join(_LOG_DIR, "suite_crash.log")

_fault_fp = None            # kept-open handle faulthandler dumps into on a fatal signal
_installed = False



def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append(text: str) -> None:
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def install(show_dialog: bool = True) -> None:
    """Wire faulthandler + a global excepthook. Call ONCE at Suite startup (main.py), before the UI.

    faulthandler.enable(file=...) makes a native SIGSEGV/SIGABRT (the ROCm/torch OOM crash) dump the
    Python stack of every thread into logs/suite_crash.log on the way down — the traceback we've been
    missing. The excepthook catches ordinary unhandled Python exceptions, logs them, and (optionally)
    shows a dialog rather than letting Qt abort the process."""
    global _fault_fp, _installed
    if _installed:
        return
    _installed = True
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        _fault_fp = open(_CRASH_LOG, "a", encoding="utf-8", buffering=1)
        _fault_fp.write(f"\n===== Suite session started {_stamp()} (pid {os.getpid()}) =====\n")
        _fault_fp.flush()
        faulthandler.enable(file=_fault_fp, all_threads=True)
        atexit.register(_close)
    except Exception:
        pass

    prev = sys.excepthook

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return prev(exc_type, exc, tb)
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        _append(f"\n----- Unhandled exception {_stamp()} -----\n{text}")
        try:
            sys.stderr.write(text)
        except Exception:
            pass
        if show_dialog:
            _try_dialog("Suite Error",
                        "An unexpected error occurred. The Suite is still running.\n\n"
                        f"{exc_type.__name__}: {exc}\n\nDetails logged to:\n{_CRASH_LOG}")
        # deliberately do NOT re-raise / call prev: keep the event loop alive

    sys.excepthook = _hook

    # WORKER THREADS: sys.excepthook only covers the MAIN thread. An exception raised inside a QThread /
    # threading.Thread (scan workers, LoRA Health SVD, SD.UI render workers) is otherwise swallowed
    # silently — the job just stops with no trace. threading.excepthook (3.8+) is the only way to see it.
    def _thread_hook(args):
        if issubclass(args.exc_type, KeyboardInterrupt):
            return
        text = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        name = getattr(args.thread, "name", "?")
        _append(f"\n----- Unhandled exception in thread '{name}' {_stamp()} -----\n{text}")
        try:
            sys.stderr.write(text)
        except Exception:
            pass
        # no dialog here: worker threads must not touch Qt widgets (that itself crashes Qt)

    try:
        threading.excepthook = _thread_hook       # Python 3.8+
    except Exception:
        pass


def _close() -> None:
    global _fault_fp
    try:
        if _fault_fp:
            _fault_fp.write(f"===== Suite session ended cleanly {_stamp()} =====\n")
            _fault_fp.close()
    except Exception:
        pass
    _fault_fp = None


def _try_dialog(title: str, text: str) -> None:
    """Show a modal warning IF a Qt app is up; silently no-op otherwise (headless / pre-QApplication)."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is None:
            return
        QMessageBox.warning(None, title, text)
    except Exception:
        pass


def guard_open(module_name: str, factory, parent=None, on_error=None):
    """Construct a heavy module page under a safety net: run `factory()` and return its result; if it
    raises a Python exception, log it, show a dialog, run `on_error` (rollback), and return None instead
    of letting the exception escape the Qt slot and abort the Suite.

    NOTE: a NATIVE crash inside factory() (a ROCm OOM segfault) still can't be caught here — that's what
    vram_guard.preflight() is for (prevention) and what faulthandler logs (diagnosis). This catches the
    Python-level init failures so those, at least, only cost you the one module."""
    try:
        return factory()
    except BaseException as e:               # noqa: BLE001 — last line of defense for the whole app
        text = traceback.format_exc()
        _append(f"\n----- {module_name} failed to open {_stamp()} -----\n{text}")
        try:
            sys.stderr.write(text)
        except Exception:
            pass
        if on_error is not None:
            try:
                on_error()
            except Exception:
                pass
        _try_dialog(f"{module_name} failed to open",
                    f"{module_name} could not start and was not opened. The Suite is still running.\n\n"
                    f"{type(e).__name__}: {e}\n\nDetails logged to:\n{_CRASH_LOG}")
        return None
