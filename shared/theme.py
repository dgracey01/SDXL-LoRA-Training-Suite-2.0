"""
shared/theme.py — Color constants and global QSS stylesheet for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.01)
"""

# ── Color palette (matches v1.x exactly) ─────────────────────────────────────
BG  = "#1a1a2e"   # app background
PAN = "#242424"   # panel / tab content background
CAR = "#16213e"   # card / input background
ACC = "#1f6feb"   # accent blue (buttons, borders, active)
GRN = "#2ea043"   # green  (online, good, success)
RED = "#da3633"   # red    (offline, error, delete)
MUT = "#444466"   # muted  (separator, inactive)
PRI = "#ffffff"   # primary text
SEC = "#a0a0a0"   # secondary text
AMB = "#EF9F27"   # amber  (warning)

VERSION = "2.01"
SIGNATURE = f"Designed by: Zero  |  Built by: Jarvis (v{VERSION})"

FONT       = "Consolas"
FONT_SM    = 9
FONT_MD    = 11
FONT_LG    = 13
FONT_XL    = 16
FONT_TITLE = 18

# ── Global QSS ────────────────────────────────────────────────────────────────
GLOBAL_QSS = f"""
/* ── Base ──────────────────────────────────────────────────────────────── */
QMainWindow, QDialog {{
    background-color: {BG};
    color: {PRI};
    font-family: {FONT};
    font-size: {FONT_MD}px;
}}

QWidget {{
    background-color: transparent;
    color: {PRI};
    font-family: {FONT};
    font-size: {FONT_MD}px;
}}

/* ── Scroll ─────────────────────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollBar:vertical {{
    background: {CAR};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {MUT};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACC};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {CAR};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {MUT};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {ACC};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Buttons ────────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {ACC};
    color: {PRI};
    border: none;
    border-radius: 6px;
    padding: 6px 16px;
    font-family: {FONT};
    font-size: {FONT_MD}px;
    font-weight: bold;
}}
QPushButton:hover  {{ background-color: #185FA5; }}
QPushButton:pressed {{ background-color: #0d4a8a; }}
QPushButton:disabled {{ background-color: {MUT}; color: {SEC}; }}

QPushButton[flat="true"] {{
    background-color: transparent;
    color: {SEC};
    border: 1px solid {MUT};
    font-weight: normal;
}}
QPushButton[flat="true"]:hover {{ border-color: {ACC}; color: {PRI}; }}

QPushButton[danger="true"] {{
    background-color: transparent;
    color: {RED};
    border: 1px solid {RED};
    font-weight: normal;
}}
QPushButton[danger="true"]:hover {{ background-color: {RED}; color: {PRI}; }}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {CAR};
    color: {PRI};
    border: 2px solid {MUT};
    border-radius: 6px;
    padding: 4px 8px;
    font-family: {FONT};
    font-size: {FONT_MD}px;
    selection-background-color: {ACC};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {ACC};
}}

/* ── ComboBox / OptionMenu ──────────────────────────────────────────────── */
QComboBox {{
    background-color: {CAR};
    color: {PRI};
    border: 2px solid {MUT};
    border-radius: 6px;
    padding: 4px 8px;
    font-family: {FONT};
    font-size: {FONT_MD}px;
    min-height: 28px;
}}
QComboBox:focus {{ border-color: {ACC}; }}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {SEC};
    width: 0; height: 0;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {CAR};
    color: {PRI};
    border: 1px solid {ACC};
    selection-background-color: {ACC};
    outline: none;
}}

/* ── CheckBox ───────────────────────────────────────────────────────────── */
QCheckBox {{
    color: {SEC};
    font-family: {FONT};
    font-size: {FONT_MD}px;
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {MUT};
    border-radius: 4px;
    background: {CAR};
}}
QCheckBox::indicator:checked {{
    background: {ACC};
    border-color: {ACC};
}}
QCheckBox::indicator:checked:hover {{ background: #185FA5; }}

/* ── Slider ─────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    background: {MUT};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACC};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {ACC}; border-radius: 2px; }}

/* ── Tab widget (internal tabs for Tags/Calculator sub-tabs) ────────────── */
QTabWidget::pane {{
    border: none;
    background: {PAN};
}}
QTabBar::tab {{
    background: {CAR};
    color: {SEC};
    border: none;
    padding: 8px 20px;
    font-family: {FONT};
    font-size: {FONT_MD}px;
    font-weight: bold;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {ACC};
    color: {PRI};
}}
QTabBar::tab:hover:!selected {{ background: {MUT}; color: {PRI}; }}

/* ── Frames / Cards ─────────────────────────────────────────────────────── */
QFrame[card="true"] {{
    background-color: {CAR};
    border-radius: 8px;
    border: 2px solid {MUT};
}}
QFrame[panel="true"] {{
    background-color: {PAN};
    border-radius: 0px;
}}

/* ── Labels ─────────────────────────────────────────────────────────────── */
QLabel {{ color: {PRI}; font-family: {FONT}; }}
QLabel[muted="true"]  {{ color: {MUT}; }}
QLabel[sec="true"]    {{ color: {SEC}; }}
QLabel[accent="true"] {{ color: {ACC}; font-weight: bold; }}
QLabel[good="true"]   {{ color: {GRN}; font-weight: bold; }}
QLabel[warn="true"]   {{ color: {AMB}; font-weight: bold; }}
QLabel[bad="true"]    {{ color: {RED}; font-weight: bold; }}

/* ── Progress bar ───────────────────────────────────────────────────────── */
QProgressBar {{
    background: {CAR};
    border: 1px solid {MUT};
    border-radius: 4px;
    height: 10px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {ACC};
    border-radius: 3px;
}}

/* ── Tooltip ────────────────────────────────────────────────────────────── */
QToolTip {{
    background: {CAR};
    color: {PRI};
    border: 1px solid {ACC};
    padding: 4px 8px;
    font-family: {FONT};
    font-size: {FONT_SM}px;
}}
"""


def apply_theme(app):
    """Call once after QApplication is created."""
    app.setStyleSheet(GLOBAL_QSS)


def make_label(parent, text="", size=FONT_MD, color=SEC, bold=False):
    """Helper — returns a styled QLabel."""
    from PySide6.QtWidgets import QLabel
    lbl = QLabel(text, parent)
    weight = "bold" if bold else "normal"
    lbl.setStyleSheet(
        f"color: {color}; font-family: {FONT}; font-size: {size}px; font-weight: {weight};"
    )
    return lbl


def card_frame(parent=None):
    """Returns a QFrame styled as a card."""
    from PySide6.QtWidgets import QFrame
    f = QFrame(parent)
    f.setProperty("card", True)
    f.setStyleSheet(f"QFrame {{ background:{CAR}; border-radius:8px; border:2px solid {MUT}; }}")
    return f
