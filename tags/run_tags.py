"""
tags/run_tags.py — Standalone launcher for Tag Handler
Designed by: Zero  |  Built by: Jarvis (v2.0)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore    import Qt
from PySide6.QtWidgets import QApplication, QMainWindow
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

from shared.theme import apply_theme, VERSION
from tags.tag_handler_page import TagHandlerPage

app = QApplication(sys.argv)
app.setStyle("Fusion")
apply_theme(app)
app.setApplicationName("Tag Handler")
app.setApplicationVersion(VERSION)

win = QMainWindow()
win.setWindowTitle(f"Tag Handler  v{VERSION}")
win.resize(1400, 900)
win.setCentralWidget(TagHandlerPage(win))
win.show()

sys.exit(app.exec())
