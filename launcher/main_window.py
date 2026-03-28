"""
launcher/main_window.py — Main window for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.0)

Architecture:
  QMainWindow
    └── central QWidget  (QVBoxLayout, spacing=0)
          ├── Header QFrame (44px fixed height)
          │     ├── [Manage] [Tags] [Calculator]  nav tabs
          │     ├── dynamic app tab buttons (added on Open)
          │     └── zoom controls + signature label
          ├── Separator QFrame (1px)
          └── QStackedWidget  (fills remaining space)
                ├── 0  ManagePage   (always present)
                ├── …  BrowserView  (one per open web app)
                ├── N  TagHandlerPage  (lazy, first click)
                └── N+1 CalculatorPage (lazy, first click)
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore    import Qt, QUrl, QTimer, QSize, Signal, QObject
from PySide6.QtGui     import QFont, QCursor, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QFrame, QStackedWidget,
    QScrollArea, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSizePolicy, QInputDialog, QMessageBox,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore    import (
    QWebEnginePage, QWebEngineProfile, QWebEngineSettings,
)

from shared.theme  import (
    BG, PAN, CAR, ACC, GRN, RED, MUT, PRI, SEC,
    FONT, FONT_SM, FONT_MD, FONT_LG, VERSION, SIGNATURE,
)
from shared.config import (
    load_apps, save_apps, check_port, DEFAULT_APPS,
    load_hf_token, save_hf_token, apply_hf_token,
)

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(os.path.dirname(_HERE), "assets")


# ── Off-the-record WebEngine profile ─────────────────────────────────────────
def _make_otr_profile(parent: QObject) -> QWebEngineProfile:
    """
    Create a nameless (off-the-record) profile shared by all browser tabs.
    No cookies on disk, no cache on disk — equivalent of pywebview private_mode=True.
    """
    profile = QWebEngineProfile(parent)
    if not profile.isOffTheRecord():
        raise RuntimeError("WebEngine profile is not off-the-record — cookies may be written to disk.")

    s       = profile.settings()
    WebAttr = QWebEngineSettings.WebAttribute
    s.setAttribute(WebAttr.JavascriptEnabled,        True)
    s.setAttribute(WebAttr.LocalStorageEnabled,      True)
    s.setAttribute(WebAttr.WebGLEnabled,             True)
    s.setAttribute(WebAttr.FullScreenSupportEnabled, True)
    s.setAttribute(WebAttr.ScrollAnimatorEnabled,    True)
    s.setAttribute(WebAttr.JavascriptCanOpenWindows, True)
    s.setAttribute(WebAttr.ErrorPageEnabled,         True)
    return profile


# ── Browser page (intercepts popups) ─────────────────────────────────────────
class BrowserPage(QWebEnginePage):
    new_tab_requested = Signal(str)

    def __init__(self, profile: QWebEngineProfile, parent=None):
        super().__init__(profile, parent)

    def createWindow(self, window_type):
        # Redirect JS popups / target="_blank" into a new launcher tab
        self.new_tab_requested.emit("")
        return None


# ── Browser view (one per open app tab) ──────────────────────────────────────
class BrowserView(QWebEngineView):
    """
    Stays alive in the QStackedWidget even when hidden — page keeps its session.
    """

    def __init__(self, profile: QWebEngineProfile, url: str, parent=None):
        super().__init__(parent)
        self._url_str = url
        self._zoom    = 1.0

        page = BrowserPage(profile, self)
        self.setPage(page)
        self.load(QUrl(url))

    def set_zoom(self, factor: float) -> None:
        factor     = max(0.25, min(3.0, round(factor, 2)))
        self._zoom = factor
        self.setZoomFactor(factor)

    def get_zoom(self) -> float:
        return self._zoom


# ── Tab button widget ─────────────────────────────────────────────────────────
class TabButton(QWidget):
    """
    Header tab button. Optionally shows a status dot and a close ×.
    Active state is set via QSS dynamic property.
    """

    clicked         = Signal()
    close_requested = Signal()

    _STYLE_BTN = f"""
        QPushButton {{
            background-color: {CAR};
            color: {PRI};
            border: 2px solid {ACC};
            border-radius: 0px;
            font-family: {FONT};
            font-size: 11px;
            font-weight: bold;
            padding: 3px 10px;
            min-height: 28px;
        }}
        QPushButton:hover {{ background-color: #185FA5; }}
        QPushButton[active="true"] {{ background-color: {ACC}; color: {PRI}; }}
    """
    _STYLE_DOT_OFF = f"color: {MUT}; font-size: 10px; padding: 0 2px 0 6px;"
    _STYLE_DOT_ON  = f"color: {GRN}; font-size: 10px; padding: 0 2px 0 6px;"
    _STYLE_CLOSE   = f"""
        QPushButton {{
            background-color: transparent;
            color: {SEC};
            border: none;
            font-size: 14px;
            font-weight: bold;
            padding: 0px 4px;
            min-width: 20px; max-width: 20px;
            min-height: 28px;
        }}
        QPushButton:hover {{ background-color: {RED}; color: {PRI}; }}
    """

    def __init__(self, label: str,
                 show_dot: bool = False, show_close: bool = False,
                 parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self._dot = None
        if show_dot:
            self._dot = QLabel("●", self)
            self._dot.setStyleSheet(self._STYLE_DOT_OFF)
            row.addWidget(self._dot)

        self._btn = QPushButton(label, self)
        self._btn.setStyleSheet(self._STYLE_BTN)
        self._btn.setProperty("active", False)
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.clicked.connect(self.clicked)
        row.addWidget(self._btn)

        self._close_btn = None
        if show_close:
            self._close_btn = QPushButton("×", self)
            self._close_btn.setStyleSheet(self._STYLE_CLOSE)
            self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._close_btn.clicked.connect(self.close_requested)
            row.addWidget(self._close_btn)

    def set_active(self, active: bool) -> None:
        self._btn.setProperty("active", active)
        self._btn.style().unpolish(self._btn)
        self._btn.style().polish(self._btn)

    def set_dot_online(self, online: bool) -> None:
        if self._dot:
            self._dot.setStyleSheet(
                self._STYLE_DOT_ON if online else self._STYLE_DOT_OFF)


# ── App card (Manage page) ────────────────────────────────────────────────────
class AppCard(QFrame):
    """
    One card per app in the Manage page.
    Shows name, URL, status dot, Open / Start / Edit / Delete controls.
    Inline edit panel expands below on demand.
    """

    open_requested   = Signal(str)          # url
    edit_saved       = Signal(object, dict) # (self, data)
    delete_requested = Signal(object)       # self

    _CARD_BASE  = f"QFrame {{ background-color: {CAR}; border: 2px solid {MUT}; border-radius: 8px; }}"
    _CARD_LIVE  = f"QFrame {{ background-color: {CAR}; border: 2px solid {ACC}; border-radius: 8px; }}"

    def __init__(self, app: dict, parent=None):
        super().__init__(parent)
        self.app        = app
        self._online    = None
        self._edit_open = False

        self.setStyleSheet(self._CARD_BASE)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(0)

        # ── Main row ──────────────────────────────────────────────────────
        row = QHBoxLayout()
        row.setSpacing(8)

        self._dot = QLabel("●", self)
        self._dot.setStyleSheet(f"color: {MUT}; font-size: 14px;")
        row.addWidget(self._dot)

        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        self._name_lbl = QLabel(app["name"], self)
        self._name_lbl.setStyleSheet(
            f"color:{PRI}; font-family:{FONT}; font-size:{FONT_LG}px; font-weight:bold;")
        self._url_lbl = QLabel(app["url"], self)
        self._url_lbl.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;")
        info_col.addWidget(self._name_lbl)
        info_col.addWidget(self._url_lbl)
        row.addLayout(info_col, stretch=1)

        self._open_btn = QPushButton("Open", self)
        self._open_btn.setEnabled(False)
        self._open_btn.setFixedSize(QSize(80, 32))
        self._open_btn.clicked.connect(lambda: self.open_requested.emit(app["url"]))
        self._style_btn(self._open_btn, ACC)
        row.addWidget(self._open_btn)

        self._start_btn = None
        if app.get("cmd"):
            self._start_btn = QPushButton("Start", self)
            self._start_btn.setFixedSize(QSize(70, 32))
            self._start_btn.clicked.connect(self._start_app)
            self._style_btn(self._start_btn, GRN)
            row.addWidget(self._start_btn)

        edit_btn = QPushButton("✎", self)
        self._style_icon_btn(edit_btn)
        edit_btn.clicked.connect(self._toggle_edit)
        row.addWidget(edit_btn)

        del_btn = QPushButton("×", self)
        self._style_icon_btn(del_btn, danger=True)
        del_btn.clicked.connect(lambda: self.delete_requested.emit(self))
        row.addWidget(del_btn)

        outer.addLayout(row)

        # ── Inline edit panel ─────────────────────────────────────────────
        self._edit_panel = QWidget(self)
        self._edit_panel.setVisible(False)
        ep = QVBoxLayout(self._edit_panel)
        ep.setContentsMargins(0, 8, 0, 0)
        ep.setSpacing(4)

        self._ed_name = QLineEdit(app["name"], self._edit_panel)
        self._ed_name.setPlaceholderText("Name")
        ep.addWidget(self._ed_name)

        self._ed_url = QLineEdit(app["url"], self._edit_panel)
        self._ed_url.setPlaceholderText("http://...")
        ep.addWidget(self._ed_url)

        self._ed_cmd = QLineEdit(app.get("cmd", ""), self._edit_panel)
        self._ed_cmd.setPlaceholderText(
            "Launch command (optional): C:\\path\\to\\start.bat --args")
        ep.addWidget(self._ed_cmd)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        save_btn = QPushButton("Save", self._edit_panel)
        save_btn.setFixedSize(QSize(80, 28))
        self._style_btn(save_btn, ACC, small=True)
        save_btn.clicked.connect(self._save_edit)
        btn_row.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel", self._edit_panel)
        cancel_btn.setFixedSize(QSize(80, 28))
        self._style_btn(cancel_btn, MUT, small=True)
        cancel_btn.clicked.connect(self._toggle_edit)
        btn_row.addWidget(cancel_btn)

        ep.addLayout(btn_row)
        outer.addWidget(self._edit_panel)

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _style_btn(btn: QPushButton, color: str, small: bool = False) -> None:
        h = 28 if small else 32
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: {PRI};
                border: none;
                border-radius: 4px;
                font-family: {FONT};
                font-size: {FONT_MD}px;
                font-weight: bold;
                min-height: {h}px;
            }}
            QPushButton:hover   {{ background-color: rgba(255,255,255,20); }}
            QPushButton:disabled {{ background-color: {MUT}; color: {SEC}; }}
        """)

    @staticmethod
    def _style_icon_btn(btn: QPushButton, danger: bool = False) -> None:
        hover_bg = RED if danger else MUT
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {SEC};
                border: none;
                font-size: 16px;
                min-width: 28px; max-width: 28px;
                min-height: 28px; max-height: 28px;
            }}
            QPushButton:hover {{ background-color: {hover_bg}; color: {PRI}; }}
        """)
        btn.setFixedSize(QSize(28, 32))

    # ── Slots ──────────────────────────────────────────────────────────────
    def _toggle_edit(self):
        self._edit_open = not self._edit_open
        self._edit_panel.setVisible(self._edit_open)
        if self._edit_open:
            self._ed_name.setText(self.app["name"])
            self._ed_url.setText(self.app["url"])
            self._ed_cmd.setText(self.app.get("cmd", ""))

    def _save_edit(self):
        data = {
            "name": self._ed_name.text().strip(),
            "url":  self._ed_url.text().strip(),
            "cmd":  self._ed_cmd.text().strip(),
        }
        if data["name"] and data["url"]:
            self.edit_saved.emit(self, data)
            self._toggle_edit()

    def _start_app(self):
        import subprocess
        cmd = self.app.get("cmd", "")
        if not cmd:
            return
        exe = cmd.split()[0].strip('"')
        cwd = os.path.dirname(exe) if os.path.isfile(exe) else None
        subprocess.Popen(cmd, shell=True, cwd=cwd)
        if self._start_btn:
            self._start_btn.setEnabled(False)
            self._start_btn.setText("Starting…")

    # ── Public API ─────────────────────────────────────────────────────────
    def set_status(self, online: bool) -> None:
        if online == self._online:
            return
        self._online = online
        self._dot.setStyleSheet(
            f"color: {GRN if online else RED}; font-size: 14px;")
        self.setStyleSheet(self._CARD_LIVE if online else self._CARD_BASE)
        self._open_btn.setEnabled(online)
        if self._start_btn:
            self._start_btn.setEnabled(not online)
            self._start_btn.setText("Start")


# ── Manage page ───────────────────────────────────────────────────────────────
class ManagePage(QWidget):
    """Scrollable list of AppCards + add-app bar at the bottom."""

    open_app = Signal(str)   # url

    def __init__(self, apps: list[dict], parent=None):
        super().__init__(parent)
        self.apps   = apps
        self._cards: list[AppCard] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Scroll area ────────────────────────────────────────────────────
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {PAN}; }}")

        self._content = QWidget()
        self._content.setStyleSheet(f"background: {PAN};")
        self._card_layout = QVBoxLayout(self._content)
        self._card_layout.setContentsMargins(12, 12, 12, 8)
        self._card_layout.setSpacing(8)
        self._card_layout.addStretch()

        scroll.setWidget(self._content)
        root.addWidget(scroll, stretch=1)

        # ── Add-app bar ────────────────────────────────────────────────────
        add_bar = QFrame(self)
        add_bar.setStyleSheet(
            f"QFrame {{ background-color:{CAR}; border-radius:8px;"
            f"border:1px solid {MUT}; margin: 6px 12px; }}")
        add_row = QHBoxLayout(add_bar)
        add_row.setContentsMargins(8, 8, 8, 8)
        add_row.setSpacing(4)

        self._add_name = QLineEdit(self)
        self._add_name.setPlaceholderText("Name")
        self._add_name.setFixedWidth(120)
        add_row.addWidget(self._add_name)

        self._add_url = QLineEdit(self)
        self._add_url.setPlaceholderText("http://localhost:")
        self._add_url.setText("http://localhost:")
        add_row.addWidget(self._add_url, stretch=1)

        add_btn = QPushButton("+", self)
        add_btn.setFixedSize(QSize(34, 32))
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACC};
                color: {PRI};
                border: none;
                border-radius: 4px;
                font-family: {FONT};
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #185FA5; }}
        """)
        add_btn.clicked.connect(self._add_app)
        add_row.addWidget(add_btn)

        root.addWidget(add_bar)

        # ── HuggingFace token bar (single slim row) ────────────────────────
        hf_card = QFrame(self)
        hf_card.setFixedHeight(38)
        hf_card.setStyleSheet(
            f"QFrame {{ background-color:{CAR}; border-radius:6px;"
            f"border:1px solid {MUT}; margin: 0px 12px 8px 12px; }}")
        hf_row = QHBoxLayout(hf_card)
        hf_row.setContentsMargins(10, 0, 8, 0)
        hf_row.setSpacing(6)

        hf_title = QLabel("HF Token:", hf_card)
        hf_title.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" font-weight:bold; border:none; background:transparent;")
        hf_row.addWidget(hf_title)

        self._hf_edit = QLineEdit(hf_card)
        self._hf_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._hf_edit.setPlaceholderText("hf_••••••••••••••••••••••••••••••••••••")
        self._hf_edit.setFixedHeight(26)
        self._hf_edit.setStyleSheet(
            f"QLineEdit {{ background:{BG}; color:{PRI}; border:1px solid {MUT};"
            f" border-radius:4px; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" padding:2px 6px; }}"
            f"QLineEdit:focus {{ border-color:{ACC}; }}")
        hf_row.addWidget(self._hf_edit, stretch=1)

        show_btn = QPushButton("👁", hf_card)
        show_btn.setFixedSize(QSize(24, 24))
        show_btn.setCheckable(True)
        show_btn.setStyleSheet(
            f"QPushButton {{ background:{MUT}; color:{PRI}; border:none;"
            f" border-radius:3px; font-size:11px; }}"
            f"QPushButton:checked {{ background:{ACC}; }}")
        show_btn.toggled.connect(
            lambda v: self._hf_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if v else QLineEdit.EchoMode.Password))
        hf_row.addWidget(show_btn)

        save_hf_btn = QPushButton("Save", hf_card)
        save_hf_btn.setFixedSize(QSize(52, 24))
        save_hf_btn.setStyleSheet(
            f"QPushButton {{ background:{ACC}; color:{PRI}; border:none;"
            f" border-radius:3px; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" font-weight:bold; }}"
            f"QPushButton:hover {{ background:#185FA5; }}")
        save_hf_btn.clicked.connect(self._save_hf_token)
        hf_row.addWidget(save_hf_btn)

        clear_hf_btn = QPushButton("Clear", hf_card)
        clear_hf_btn.setFixedSize(QSize(44, 24))
        clear_hf_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{MUT}; border:1px solid {MUT};"
            f" border-radius:3px; font-family:{FONT}; font-size:{FONT_SM}px; }}"
            f"QPushButton:hover {{ border-color:{RED}; color:{RED}; }}")
        clear_hf_btn.clicked.connect(self._clear_hf_token)
        hf_row.addWidget(clear_hf_btn)

        self._hf_status = QLabel("○", hf_card)
        self._hf_status.setFixedWidth(14)
        self._hf_status.setStyleSheet(
            f"color:{MUT}; font-size:11px; border:none; background:transparent;")
        self._hf_status.setToolTip("No token set")
        hf_row.addWidget(self._hf_status)

        root.addWidget(hf_card)

        # Populate token field if already saved
        saved = load_hf_token()
        if saved:
            self._hf_edit.setText(saved)
            self._hf_status.setText("●")
            self._hf_status.setStyleSheet(
                f"color:{GRN}; font-size:11px; border:none; background:transparent;")
            self._hf_status.setToolTip("Token set")

        self.refresh_cards()

    def refresh_cards(self) -> None:
        for card in self._cards:
            self._card_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        for app in self.apps:
            card = AppCard(app)
            card.open_requested.connect(self.open_app)
            card.edit_saved.connect(self._on_edit_saved)
            card.delete_requested.connect(self._on_delete)
            self._card_layout.insertWidget(self._card_layout.count() - 1, card)
            self._cards.append(card)

    def poll_update(self, statuses: dict) -> None:
        for card in self._cards:
            url = card.app["url"]
            if url in statuses:
                card.set_status(statuses[url])

    def get_cards(self) -> list[AppCard]:
        return list(self._cards)

    def _add_app(self):
        name = self._add_name.text().strip()
        url  = self._add_url.text().strip()
        if not name or not url:
            return
        self.apps.append({"name": name, "url": url})
        save_apps(self.apps)
        self._add_name.clear()
        self._add_url.setText("http://localhost:")
        self.refresh_cards()

    def _on_edit_saved(self, card: AppCard, data: dict):
        card.app["name"] = data["name"]
        card.app["url"]  = data["url"]
        if data["cmd"]:
            card.app["cmd"] = data["cmd"]
        else:
            card.app.pop("cmd", None)
        save_apps(self.apps)
        self.refresh_cards()

    def _on_delete(self, card: AppCard):
        self.apps = [a for a in self.apps if a is not card.app]
        save_apps(self.apps)
        self.refresh_cards()

    def _save_hf_token(self):
        token = self._hf_edit.text().strip()
        if not token:
            return
        save_hf_token(token)
        ok = apply_hf_token(token)
        if ok:
            self._hf_status.setText("●  Token set")
            self._hf_status.setStyleSheet(
                f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f" border:none; background:transparent;")
        else:
            self._hf_status.setText("⚠  Login failed")
            self._hf_status.setStyleSheet(
                f"color:{AMB}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f" border:none; background:transparent;")

    def _clear_hf_token(self):
        self._hf_edit.clear()
        save_hf_token("")
        import os; os.environ.pop("HF_TOKEN", None)
        self._hf_status.setText("○  Not set")
        self._hf_status.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" border:none; background:transparent;")


# ── Poll worker (signals safe across threads) ─────────────────────────────────
class PollWorker(QObject):
    results_ready = Signal(dict)   # {url: bool}

    def __init__(self, apps: list[dict], parent=None):
        super().__init__(parent)
        self.apps      = apps
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="poll")

    def run(self) -> None:
        urls    = [a["url"] for a in self.apps]
        futures = {url: self._executor.submit(check_port, url) for url in urls}

        def collect():
            results = {url: f.result() for url, f in futures.items()}
            self.results_ready.emit(results)

        threading.Thread(target=collect, daemon=True).start()

    def update_apps(self, apps: list[dict]) -> None:
        self.apps = apps


# ── Main Launcher window ──────────────────────────────────────────────────────
class Launcher(QMainWindow):
    """
    Main application window for Lora Training Suite 2.0.

    State:
      apps            — list[dict] loaded from apps.json
      _profile        — shared OTR QWebEngineProfile
      _tab_data       — {url: {view, tab_btn, app}}
      _stack_index    — {url: int}
      _active_url     — currently visible browser tab URL or None
      _zoom           — current zoom level (float)
    """

    MANAGE_INDEX = 0

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Lora Training Suite v{VERSION}")
        self.setWindowIcon(QIcon(os.path.join(_ASSETS, "launcher.ico")))
        self.resize(1280, 900)
        self.setMinimumSize(900, 560)

        self.apps           = load_apps()
        self._profile       = _make_otr_profile(self)
        self._tab_data:     dict = {}
        self._stack_index:  dict = {}
        self._active_url:   str | None = None
        self._zoom:         float = 1.0

        # Lazy pages (created on first click)
        self._tags_page     = None
        self._calc_page     = None
        self._rand_page     = None
        self._tags_index:   int | None = None
        self._calc_index:   int | None = None
        self._rand_index:   int | None = None

        self._build_ui()
        self._start_polling()

        # Restore HuggingFace session if token was previously saved
        _saved_token = load_hf_token()
        if _saved_token:
            apply_hf_token(_saved_token)

    # ═══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root        = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        # ── Header ────────────────────────────────────────────────────────
        self._header = QFrame(self)
        self._header.setObjectName("header")
        self._header.setFixedHeight(44)
        self._header.setStyleSheet(f"QFrame#header {{ background-color: {CAR}; border: none; }}")
        hrow = QHBoxLayout(self._header)
        hrow.setContentsMargins(4, 4, 8, 4)
        hrow.setSpacing(0)

        # Nav tabs
        self._manage_tab = self._make_nav_tab("Manage",     self._show_manage)
        self._tags_tab   = self._make_nav_tab("Tags",       self._show_tags)
        self._calc_tab   = self._make_nav_tab("Calculator", self._show_calculator)
        self._rand_tab   = self._make_nav_tab("Randomizer", self._show_randomizer)

        for btn in (self._manage_tab, self._tags_tab, self._calc_tab, self._rand_tab):
            hrow.addWidget(btn)
            hrow.addSpacing(4)

        # Dynamic app tabs
        self._app_tabs_area = QHBoxLayout()
        self._app_tabs_area.setContentsMargins(0, 0, 0, 0)
        self._app_tabs_area.setSpacing(4)
        hrow.addLayout(self._app_tabs_area)

        hrow.addStretch(1)

        # Zoom controls
        self._zoom_lbl = QLabel("100%", self._header)
        self._zoom_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:11px;"
            f"min-width:46px; max-width:46px; qproperty-alignment:AlignCenter;")
        self._zoom_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self._zoom_lbl.mousePressEvent = lambda e: self._zoom_manual()

        btn_minus = self._make_zoom_btn("−", self._zoom_out)
        btn_plus  = self._make_zoom_btn("+", self._zoom_in)

        zoom_frame = QWidget(self._header)
        zoom_row   = QHBoxLayout(zoom_frame)
        zoom_row.setContentsMargins(0, 0, 0, 0)
        zoom_row.setSpacing(2)
        zoom_row.addWidget(btn_minus)
        zoom_row.addWidget(self._zoom_lbl)
        zoom_row.addWidget(btn_plus)
        hrow.addWidget(zoom_frame)
        hrow.addSpacing(8)

        # Signature
        sig = QLabel(SIGNATURE, self._header)
        sig.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;")
        hrow.addWidget(sig)

        root_layout.addWidget(self._header)

        # ── 1px separator ─────────────────────────────────────────────────
        sep = QFrame(self)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {MUT};")
        root_layout.addWidget(sep)

        # ── Content stack ──────────────────────────────────────────────────
        self._stack = QStackedWidget(self)
        root_layout.addWidget(self._stack, stretch=1)

        self._manage_page = ManagePage(self.apps)
        self._manage_page.open_app.connect(self._open_app_tab)
        self._stack.addWidget(self._manage_page)   # index 0

        self._show_manage()

    def _make_nav_tab(self, label: str, callback) -> TabButton:
        btn = TabButton(label, show_dot=False, show_close=False, parent=self._header)
        btn.clicked.connect(callback)
        return btn

    @staticmethod
    def _make_zoom_btn(label: str, callback) -> QPushButton:
        btn = QPushButton(label)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {CAR};
                color: {PRI};
                border: none;
                font-family: {FONT};
                font-size: 16px;
                font-weight: bold;
                min-width: 26px; max-width: 26px;
                min-height: 28px; max-height: 28px;
            }}
            QPushButton:hover {{ background-color: {MUT}; }}
        """)
        btn.clicked.connect(callback)
        return btn

    # ═══════════════════════════════════════════════════════════════════════════
    # Navigation
    # ═══════════════════════════════════════════════════════════════════════════

    def _deactivate_all(self):
        self._manage_tab.set_active(False)
        self._tags_tab.set_active(False)
        self._calc_tab.set_active(False)
        self._rand_tab.set_active(False)
        for entry in self._tab_data.values():
            entry["tab_btn"].set_active(False)

    def _show_manage(self):
        self._deactivate_all()
        self._manage_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self.MANAGE_INDEX)
        self._set_zoom_display(1.0)

    def _show_tags(self):
        if self._tags_page is None:
            from tags.tag_handler_page import TagHandlerPage
            self._tags_page  = TagHandlerPage()
            self._tags_index = self._stack.addWidget(self._tags_page)
        self._deactivate_all()
        self._tags_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self._tags_index)
        self._set_zoom_display(1.0)

    def _show_calculator(self):
        if self._calc_page is None:
            from calculator.calculator_page import CalculatorPage
            self._calc_page  = CalculatorPage()
            self._calc_index = self._stack.addWidget(self._calc_page)
        self._deactivate_all()
        self._calc_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self._calc_index)
        self._set_zoom_display(1.0)

    def _show_randomizer(self):
        if self._rand_page is None:
            from randomizer.randomizer_page import RandomizerPage
            self._rand_page  = RandomizerPage()
            self._rand_index = self._stack.addWidget(self._rand_page)
        self._deactivate_all()
        self._rand_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self._rand_index)
        self._set_zoom_display(1.0)

    def _open_app_tab(self, url: str):
        if url in self._tab_data:
            self._switch_to_tab(url)
            return

        view = BrowserView(self._profile, url)
        idx  = self._stack.addWidget(view)
        self._stack_index[url] = idx

        app     = next((a for a in self.apps if a["url"] == url), {"name": url, "url": url})
        tab_btn = TabButton(app["name"], show_dot=True, show_close=True,
                            parent=self._header)
        tab_btn.clicked.connect(lambda u=url: self._switch_to_tab(u))
        tab_btn.close_requested.connect(lambda u=url: self._close_tab(u))
        self._app_tabs_area.addWidget(tab_btn)

        self._tab_data[url] = {"view": view, "tab_btn": tab_btn, "app": app}
        self._switch_to_tab(url)

    def _switch_to_tab(self, url: str):
        if url not in self._tab_data:
            return
        self._deactivate_all()
        self._tab_data[url]["tab_btn"].set_active(True)
        self._active_url = url
        view: BrowserView = self._tab_data[url]["view"]
        self._set_zoom_display(view.get_zoom())
        self._stack.setCurrentIndex(self._stack_index[url])

    def _close_tab(self, url: str):
        if url not in self._tab_data:
            return
        entry             = self._tab_data.pop(url)
        view: BrowserView = entry["view"]
        tab_btn: TabButton = entry["tab_btn"]

        self._app_tabs_area.removeWidget(tab_btn)
        tab_btn.deleteLater()

        self._stack.removeWidget(view)
        self._stack_index.pop(url, None)
        view.setPage(None)
        view.deleteLater()

        self._rebuild_stack_index()
        if self._active_url == url:
            self._active_url = None
            self._show_manage()

    def _rebuild_stack_index(self):
        self._stack_index.clear()
        for url, entry in self._tab_data.items():
            idx = self._stack.indexOf(entry["view"])
            if idx >= 0:
                self._stack_index[url] = idx
        if self._tags_page:
            self._tags_index = self._stack.indexOf(self._tags_page)
        if self._calc_page:
            self._calc_index = self._stack.indexOf(self._calc_page)
        if self._rand_page:
            self._rand_index = self._stack.indexOf(self._rand_page)

    # ═══════════════════════════════════════════════════════════════════════════
    # Zoom
    # ═══════════════════════════════════════════════════════════════════════════

    def _zoom_in(self):
        self._apply_zoom(round(self._zoom + 0.01, 2))

    def _zoom_out(self):
        self._apply_zoom(round(self._zoom - 0.01, 2))

    def _zoom_manual(self):
        val, ok = QInputDialog.getInt(
            self, "Zoom", "Enter zoom % (25–300):",
            value=int(round(self._zoom * 100)),
            min=25, max=300, step=1)
        if ok:
            self._apply_zoom(val / 100.0)

    def _apply_zoom(self, level: float):
        level = round(max(0.25, min(3.0, level)), 2)
        self._set_zoom_display(level)
        if self._active_url and self._active_url in self._tab_data:
            self._tab_data[self._active_url]["view"].set_zoom(level)

    def _set_zoom_display(self, level: float):
        self._zoom = level
        self._zoom_lbl.setText(f"{int(round(level * 100))}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # Port polling
    # ═══════════════════════════════════════════════════════════════════════════

    def _start_polling(self):
        self._poller = PollWorker(self.apps)
        self._poller.results_ready.connect(self._on_poll_results)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(3000)
        self._poll_timer.timeout.connect(self._poller.run)
        self._poll_timer.start()
        self._poller.run()   # kick off immediately

    def _on_poll_results(self, statuses: dict):
        self._manage_page.poll_update(statuses)
        for url, online in statuses.items():
            if url in self._tab_data:
                self._tab_data[url]["tab_btn"].set_dot_online(online)
        self._poller.update_apps(self.apps)

    # ═══════════════════════════════════════════════════════════════════════════
    # Shutdown
    # ═══════════════════════════════════════════════════════════════════════════

    def closeEvent(self, event):
        self._poll_timer.stop()
        for url in list(self._tab_data.keys()):
            entry             = self._tab_data.pop(url)
            view: BrowserView = entry["view"]
            view.setPage(None)
            view.deleteLater()
        event.accept()
