"""
calculator/run_calculator.py — Standalone launcher for Calculator
Designed by: Zero  |  Built by: Jarvis (v2.0)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore    import Qt
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui     import QIcon
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

from shared.theme import apply_theme, VERSION
from calculator.calculator_page import CalculatorPage

_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

app = QApplication(sys.argv)
app.setStyle("Fusion")
apply_theme(app)
app.setApplicationName("LoRA Calculator")
app.setApplicationVersion(VERSION)
app.setWindowIcon(QIcon(os.path.join(_ASSETS, "calculator.ico")))

win = QMainWindow()
win.setWindowTitle(f"LoRA Calculator  v{VERSION}")
win.setWindowIcon(QIcon(os.path.join(_ASSETS, "calculator.ico")))
win.resize(900, 700)
win.setCentralWidget(CalculatorPage(win))
win.show()

sys.exit(app.exec())
