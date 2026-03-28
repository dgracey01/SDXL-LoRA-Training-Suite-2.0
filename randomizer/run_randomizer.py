"""
randomizer/run_randomizer.py — Standalone launcher for Randomizer
Designed by: Zero  |  Built by: Jarvis (v2.0)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore    import Qt
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui     import QIcon
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

from shared.theme import apply_theme, VERSION
from randomizer.randomizer_page import RandomizerPage

_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

app = QApplication(sys.argv)
app.setStyle("Fusion")
apply_theme(app)
app.setApplicationName("Randomizer")
app.setApplicationVersion(VERSION)
app.setWindowIcon(QIcon(os.path.join(_ASSETS, "random.ico")))

win = QMainWindow()
win.setWindowTitle(f"Randomizer  v{VERSION}")
win.setWindowIcon(QIcon(os.path.join(_ASSETS, "random.ico")))
win.resize(1300, 850)
win.setCentralWidget(RandomizerPage(win))
win.show()

sys.exit(app.exec())
