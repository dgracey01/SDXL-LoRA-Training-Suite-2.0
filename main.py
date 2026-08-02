"""
main.py — Entry point for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.0)

Usage:
    python main.py              # normal launch
    python main.py --no-web     # skip QWebEngine (Tags + Calculator only)

IMPORTANT: AA_ShareOpenGLContexts MUST be set before QApplication() is created.
           Without it, QWebEngineView may render black or crash on Windows.
"""

import sys
import os
import ctypes

# ── pythonw.exe stream safety (MUST be early — before anything writes to stdout/stderr) ────────────
# Under pythonw (the GUI launch, no console) sys.stdout/stderr are None. Any library that writes to
# them — notably huggingface_hub's tqdm download bars — then dies with
# "'NoneType' object has no attribute 'write'", which is exactly what broke fresh WD14 tagger / model
# downloads on a clean clone. Point the missing streams at a discard sink and disable HF progress bars.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── ROCm attention (MUST precede any torch import, anywhere in the process) ───
# PyTorch gates its flash / mem-efficient attention kernels behind this flag on AMD. Unset, SDPA
# falls back to the `math` path: measured 210-486 ms per SDXL-1024 attention call vs ~6 ms with
# flash, i.e. 0.12 it/s instead of ~2 it/s — renders that feel CPU-bound.
# It is read when torch initialises, so setting it from a library module is too late: by the time
# shared/render_engine.py is imported (lazily, on first render) torch is already loaded and the flag
# has no effect. The entry point is the only place that reliably wins the race. Harmless on
# CUDA/CPU boxes.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# ── PyTorch HIP allocator: curb the fragmentation that causes ROCm hard-crash-on-OOM ──────────────
# The single-process Suite loads/frees big SDXL tensors repeatedly; a fragmented pool is the usual
# trigger of the native OOM that kills the whole process on this RDNA3 + ROCm-Windows box.
#   garbage_collection_threshold:0.6 — reclaim cached blocks earlier (before the pool is exhausted).
#   max_split_size_mb:512           — don't carve large blocks into small pieces that can't recombine.
# Deliberately NO expandable_segments: proven a no-op that HARD-CRASHED on this discrete card (it's a
# unified-memory/iGPU trick). Must be set before torch initialises → the entry point is the only place
# that wins the race. Set both the HIP (ROCm) and CUDA (fallback) names so any torch build reads it.
_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:512"
os.environ.setdefault("PYTORCH_ALLOC_CONF", _ALLOC_CONF)       # torch >= 2.9 canonical name
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", _ALLOC_CONF)   # ROCm-specific fallback for older builds

# ── MIOpen conv kernel-find: FAST mode (avoid the hires-fix 2.0x stall) ────────────────────────────
# The FIRST time MIOpen sees a large conv shape — e.g. the ~2048² UNet upsampling conv from a hires-fix
# 2.0x SECOND pass — its default find runs an EXHAUSTIVE kernel search that can hang the render for
# minutes (thread frozen inside _conv_forward, a full CPU core pegged). Verified 2026-08-02: a 2.0x
# render sat 25s+ on one such conv. FAST mode uses the find-db when present and an immediate heuristic /
# dynamic kernel otherwise (no exhaustive search), so the conv proceeds at once — a marginally slower
# kernel beats a multi-minute stall. Read by MIOpen at conv time; set before torch initialises to be safe.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

# ── Single-instance guard (Windows named mutex) ───────────────────────────────
# HARDENED: the old code read the error with a SEPARATE `windll.kernel32.GetLastError()` call — but ctypes
# can reset the thread's Win32 last-error between two foreign calls, so ERROR_ALREADY_EXISTS was
# intermittently missed and a SECOND instance slipped through (that's how stale-code duplicates were
# stacking up). WinDLL(use_last_error=True) snapshots the error right after CreateMutexW so
# ctypes.get_last_error() reads it reliably. The handle is kept in a module global for the process
# lifetime so the mutex isn't released early. bInitialOwner=False — existence detection doesn't need it.
_ERROR_ALREADY_EXISTS = 183
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateMutexW.restype  = ctypes.c_void_p
_k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
_MUTEX_NAME = "ZeroJarvis.LoraSuite.2.SingleInstance"
_mutex = _k32.CreateMutexW(None, 0, _MUTEX_NAME)
if _mutex and ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
    ctypes.windll.user32.MessageBoxW(
        0,
        "Lora Training Suite is already running.",
        "Already Running",
        0x30,  # MB_ICONWARNING
    )
    sys.exit(0)
# (if _mutex is NULL the mutex couldn't be created — proceed rather than block launch on a rare failure.)

# ── Windows taskbar: treat as its own app (not grouped under pythonw.exe) ─────
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
    "ZeroJarvis.LoraSuite.Launcher.2")

# ── Pre-QApplication setup ────────────────────────────────────────────────────
# Must happen before QApplication() — this attribute enables GPU sharing between
# the Qt main process and the Chromium GPU process used by QWebEngineView.
from PySide6.QtCore    import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui     import QIcon
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

from shared.theme     import apply_theme, VERSION
from shared import crash_guard
# faulthandler + global excepthook: a native ROCm/torch OOM (segfault) now dumps a Python stack to
# logs/suite_crash.log instead of vanishing, and an unhandled Python error shows a dialog instead of
# killing the Suite. Installed before the UI so it covers construction of every module.
crash_guard.install()

from launcher.main_window import Launcher

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def main():
    app = QApplication(sys.argv)

    # Fusion style = consistent dark theme on all platforms; eliminates OS chrome
    app.setStyle("Fusion")

    # Apply our global dark QSS stylesheet
    apply_theme(app)

    app.setApplicationName("Lora Training Suite")
    app.setApplicationVersion(VERSION)
    app.setOrganizationName("Zero / Jarvis")
    app.setWindowIcon(QIcon(os.path.join(_ASSETS, "launcher.ico")))

    window = Launcher()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
