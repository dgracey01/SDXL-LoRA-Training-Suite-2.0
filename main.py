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

# ── Single-instance guard (Windows named mutex) ───────────────────────────────
_MUTEX_NAME = "ZeroJarvis.LoraSuite.2.SingleInstance"
_mutex = ctypes.windll.kernel32.CreateMutexW(None, True, _MUTEX_NAME)
if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
    ctypes.windll.user32.MessageBoxW(
        0,
        "Lora Training Suite is already running.",
        "Already Running",
        0x30,  # MB_ICONWARNING
    )
    sys.exit(0)

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
