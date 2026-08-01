"""
launcher/main_window.py — Main window for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.0)

Architecture:
  QMainWindow
    └── central QWidget  (QVBoxLayout, spacing=0)
          ├── Header QFrame (44px fixed height)
          │     ├── [Manage] nav tab
          │     ├── builtin tab buttons (hidden until app is opened)
          │     ├── dynamic WebUI tab buttons (added on Open)
          │     └── ⚙ Settings  zoom controls  signature label
          ├── Separator QFrame (1px)
          └── QStackedWidget  (fills remaining space)
                ├── 0  ManagePage   (always present)
                ├── …  BrowserView  (one per open web app)
                ├── N  TagHandlerPage  (lazy, first click)
                └── N+1 … other built-in app pages (lazy)
"""
from __future__ import annotations

import os
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore    import Qt, QUrl, QTimer, QSize, Signal, QObject
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFrame,
    QStackedWidget,
    QScrollArea,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QInputDialog,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QMenu,
    QFileDialog,
    QFormLayout,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore    import (
    QWebEnginePage, QWebEngineProfile, QWebEngineSettings,
)

from shared.theme import (
    BG,
    PAN,
    CAR,
    ACC,
    GRN,
    RED,
    MUT,
    PRI,
    SEC,
    FONT,
    FONT_SM,
    FONT_MD,
    VERSION,
    SIGNATURE,
)
from shared.config import (
    load_apps,
    save_apps,
    check_port,
    load_hf_token,
    save_hf_token,
    apply_hf_token,
)

_HERE             = os.path.dirname(os.path.abspath(__file__))
_ASSETS           = os.path.join(os.path.dirname(_HERE), "assets")

# SD.UI is an OPTIONAL local module (not shipped in the public repo — it needs a local diffusers
# engine + models). Only offer its launcher button when sdui/ is actually present, so a fresh clone
# doesn't get a button that crashes with ModuleNotFoundError. Local installs with sdui/ are unaffected.
_SDUI_PRESENT     = os.path.isdir(os.path.join(os.path.dirname(_HERE), "sdui"))
_CUSTOM_APPS_FILE = os.path.join(_HERE, "custom_apps.json")

# Accepted identifier-image formats. Includes .ico (program icons) and other
# common formats Qt can load — the box drops/pickers match against this list.
_IMG_EXTS   = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif',
               '.ico', '.jfif', '.tif', '.tiff')
_IMG_FILTER = ("Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif"
               " *.ico *.jfif *.tif *.tiff)")


def _load_custom_apps() -> list[dict]:
    import json
    try:
        if os.path.exists(_CUSTOM_APPS_FILE):
            with open(_CUSTOM_APPS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_custom_apps(apps: list[dict]) -> None:
    import json
    with open(_CUSTOM_APPS_FILE, "w", encoding="utf-8") as f:
        json.dump(apps, f, indent=2, ensure_ascii=False)


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
        self.new_tab_requested.emit("")
        return None


# ── Browser view (one per open app tab) ──────────────────────────────────────
class BrowserView(QWebEngineView):
    """Stays alive in the QStackedWidget even when hidden — page keeps its session."""

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
    Header tab button.  Optionally shows a status dot and a close ×.
    Active state is set via QSS dynamic property.
    """

    clicked           = Signal()
    close_requested   = Signal()
    refresh_requested = Signal()

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
                 enable_refresh: bool = False,
                 parent=None):
        super().__init__(parent)
        self._enable_refresh = enable_refresh
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

    def contextMenuEvent(self, event):
        if not self._enable_refresh:
            event.ignore()
            return
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_QSS)
        act_refresh = menu.addAction("↺  Refresh")
        chosen = menu.exec(event.globalPos())
        if chosen == act_refresh:
            self.refresh_requested.emit()


# ── App icon box (Manage page — Applications section) ────────────────────────
class AppIconBox(QFrame):
    """Clickable icon box for a built-in application — same size as WebUIBox."""

    clicked_app = Signal(str)   # emits app name

    # Same outer dimensions as WebUIBox
    _W    = 220
    _H    = 264
    _BAR_H = 44   # bottom name bar
    # Square icon area fills the rest: (_H - _BAR_H - 4 borders) × (_W - 4) = 216 × 216

    _STYLE = f"""
        QFrame {{
            background-color: {CAR};
            border: 2px solid {MUT};
            border-radius: 8px;
        }}
    """
    _STYLE_HOVER = f"""
        QFrame {{
            background-color: #152236;
            border: 2px solid {ACC};
            border-radius: 8px;
        }}
    """

    def __init__(self, name: str, icon_path: str, parent=None):
        super().__init__(parent)
        self.app_name = name
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(self._STYLE)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # ── Icon area (square) ─────────────────────────────────────────────
        icon_area_w = self._W - 4
        icon_area_h = self._H - self._BAR_H - 4  # = 216

        icon_lbl = QLabel(self)
        icon_lbl.setFixedSize(icon_area_w, icon_area_h)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet(
            f"background:{BG}; border:none; border-radius:6px 6px 0 0;")

        px = QPixmap(icon_path)
        if not px.isNull():
            px = px.scaled(icon_area_w - 16, icon_area_h - 16,
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            icon_lbl.setPixmap(px)
        else:
            icon_lbl.setText("?")
            icon_lbl.setStyleSheet(
                f"color:{MUT}; font-size:40px; background:{BG};"
                f" border:none; border-radius:6px 6px 0 0;")
        vbox.addWidget(icon_lbl)

        # ── Name bar ───────────────────────────────────────────────────────
        bar = QWidget(self)
        bar.setFixedHeight(self._BAR_H)
        bar.setStyleSheet(f"background:{PAN}; border:none; border-radius:0 0 6px 6px;")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 0, 8, 0)

        name_lbl = QLabel(name, bar)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet(
            f"color:{PRI}; font-family:{FONT}; font-size:13px;"
            f" font-weight:bold; background:transparent; border:none;")
        bar_layout.addWidget(name_lbl)
        vbox.addWidget(bar)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked_app.emit(self.app_name)
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.setStyleSheet(self._STYLE_HOVER)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self._STYLE)
        super().leaveEvent(event)


# ── Background image scaling ──────────────────────────────────────────────────
# QImage is thread-safe; QPixmap must only be created on the main thread.
_IMG_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="img_scale")


# Marshal a callable from a worker thread back onto the GUI thread. QTimer.singleShot
# cannot be used from a non-Qt worker thread (no event loop there → it never fires), which
# is why background-loaded box images silently failed to appear. Emitting through this
# queued signal runs the callback on the GUI thread, where QPixmap creation is also legal.
class _MainThreadInvoker(QObject):
    run = Signal(object)

    def __init__(self):
        super().__init__()
        self.run.connect(self._invoke, Qt.ConnectionType.QueuedConnection)

    @staticmethod
    def _invoke(fn):
        try:
            fn()
        except RuntimeError:
            pass


_MAIN_INVOKER = _MainThreadInvoker()


def _scale_qimage_bg(path: str, w: int, h: int):
    """Load, scale-to-fill, and center-crop a QImage off the main thread."""
    img = QImage(path)
    if img.isNull():
        return None
    scaled = img.scaled(w, h,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation)
    x = max(0, (scaled.width()  - w) // 2)
    y = max(0, (scaled.height() - h) // 2)
    return scaled.copy(x, y, w, h)


# ── WebUI entry box (Manage page — WebUI/Loader section) ─────────────────────
class WebUIBox(QFrame):
    """Card for a WebUI/Loader entry — supports drag-and-drop image, right-click menu."""

    open_requested   = Signal(str)    # url
    edit_requested   = Signal(object) # self — full edit (name/url/cmd changed)
    name_changed     = Signal(object) # self — rename only, no rebuild needed
    delete_requested = Signal(object) # self
    image_changed    = Signal(object) # self — image dropped/edited/cleared

    _W, _H  = 220, 264
    _BAR_H  = 44
    _IMG_H  = _W - 4   # = 216 — square image section

    _STYLE = f"""
        QFrame {{
            background-color: {CAR};
            border: 2px solid {MUT};
            border-radius: 8px;
        }}
    """
    _STYLE_ONLINE = f"""
        QFrame {{
            background-color: {CAR};
            border: 2px solid {ACC};
            border-radius: 8px;
        }}
    """

    def __init__(self, app: dict, parent=None):
        super().__init__(parent)
        self.app     = app
        self._online = None
        self.setFixedSize(self._W, self._H)
        self.setAcceptDrops(True)
        self.setStyleSheet(self._STYLE)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # ── Image area (square) ────────────────────────────────────────────
        self._img_lbl = QLabel(self)
        self._img_lbl.setFixedSize(self._W - 4, self._IMG_H)   # 216 × 216
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setStyleSheet(
            f"background:{BG}; border:none; border-radius:6px 6px 0 0;")
        self._refresh_image()
        vbox.addWidget(self._img_lbl)

        # ── Bottom bar ─────────────────────────────────────────────────────
        bar = QWidget(self)
        bar.setFixedHeight(self._BAR_H)
        bar.setStyleSheet(f"background:{PAN}; border:none; border-radius:0 0 6px 6px;")
        bar_row = QHBoxLayout(bar)
        bar_row.setContentsMargins(8, 0, 6, 0)
        bar_row.setSpacing(4)

        self._name_lbl = QLabel(app["name"], bar)
        self._name_lbl.setStyleSheet(
            f"color:{PRI}; font-family:{FONT}; font-size:12px;"
            f" font-weight:bold; background:transparent; border:none;")
        bar_row.addWidget(self._name_lbl, stretch=1)

        self._open_btn = QPushButton("Open", bar)
        self._open_btn.setFixedHeight(28)
        self._open_btn.setMinimumWidth(64)
        self._open_btn.setEnabled(False)
        self._open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._open_btn.setStyleSheet(f"""
            QPushButton {{
                background:{ACC}; color:{PRI}; border:none;
                border-radius:3px; font-size:11px; font-weight:bold;
            }}
            QPushButton:disabled {{ background:{MUT}; color:{SEC}; }}
            QPushButton:hover {{ background:#185FA5; }}
        """)
        self._open_btn.clicked.connect(
            lambda: self.open_requested.emit(self.app["url"]))
        bar_row.addWidget(self._open_btn)

        vbox.addWidget(bar)

        # ── Status dot (top-right overlay) ─────────────────────────────────
        self._dot = QLabel("●", self)
        self._dot.setFixedSize(16, 16)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dot.setStyleSheet(
            f"color:{MUT}; font-size:10px;"
            f" background:rgba(0,0,0,160); border-radius:8px; border:none;")
        self._dot.move(self._W - 20, 6)
        self._dot.raise_()

    # ── Image ──────────────────────────────────────────────────────────────
    def _refresh_image(self):
        img_path = self.app.get("image", "")
        if img_path and os.path.isfile(img_path):
            w, h = self._W - 4, self._IMG_H
            weak_self = weakref.ref(self)
            def _done(fut):
                self_ref = weak_self()
                if self_ref is None:
                    return
                result = fut.result()
                def _apply():
                    try:
                        if result is not None and not result.isNull():
                            self_ref._img_lbl.setPixmap(QPixmap.fromImage(result))
                            self_ref._img_lbl.setStyleSheet(
                                "border:none; border-radius:6px 6px 0 0;")
                        else:
                            self_ref._show_image_placeholder()
                    except RuntimeError:
                        pass
                _MAIN_INVOKER.run.emit(_apply)
            _IMG_EXECUTOR.submit(_scale_qimage_bg, img_path, w, h).add_done_callback(_done)
            return
        self._show_image_placeholder()

    def _show_image_placeholder(self):
        self._img_lbl.clear()
        self._img_lbl.setText("Drop image here")
        self._img_lbl.setStyleSheet(
            f"color:{MUT}; font-size:11px; background:{BG};"
            f" border:none; border-radius:6px 6px 0 0;")

    # ── Drag-and-drop ──────────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for u in event.mimeData().urls():
                if u.toLocalFile().lower().endswith(
                        _IMG_EXTS):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for u in event.mimeData().urls():
            path = u.toLocalFile()
            if path.lower().endswith(
                    _IMG_EXTS):
                self.app["image"] = path
                self._refresh_image()
                self.image_changed.emit(self)
                event.acceptProposedAction()
                return
        event.ignore()

    # ── Context menu ───────────────────────────────────────────────────────
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_QSS)
        act_edit   = menu.addAction("✎  Edit")
        act_rename = menu.addAction("✏  Rename")
        if self.app.get("cmd"):
            menu.addSeparator()
            act_start = menu.addAction("▶  Start")
        else:
            act_start = None
        menu.addSeparator()
        act_edit_img = menu.addAction("🖼  Edit Image")
        act_del_img  = menu.addAction("🗑  Delete Image")
        menu.addSeparator()
        act_delete   = menu.addAction("✕  Delete Entry")

        chosen = menu.exec(event.globalPos())
        if chosen == act_edit:
            self._open_edit_dialog()
        elif chosen == act_rename:
            self._rename()
        elif chosen == act_start and act_start is not None:
            self._start()
        elif chosen == act_edit_img:
            self._browse_image()
        elif chosen == act_del_img:
            self._delete_image()
        elif chosen == act_delete:
            self.delete_requested.emit(self)

    def _open_edit_dialog(self):
        dlg = WebUIEntryDialog(self.app, parent=self)
        if dlg.exec():
            data = dlg.get_data()
            self.app.update(data)
            self._name_lbl.setText(self.app["name"])
            self.edit_requested.emit(self)

    def _rename(self):
        new_name, ok = QInputDialog.getText(
            self, "Rename", "New name:", text=self.app["name"])
        if ok and new_name.strip():
            self.app["name"] = new_name.strip()
            self._name_lbl.setText(new_name.strip())
            self.name_changed.emit(self)

    def _start(self):
        import subprocess
        cmd = self.app.get("cmd", "")
        if not cmd:
            return
        exe = cmd.split()[0].strip('"')
        cwd = os.path.dirname(exe) if os.path.isfile(exe) else None
        subprocess.Popen(cmd, shell=True, cwd=cwd)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            _IMG_FILTER)
        if path:
            self.app["image"] = path
            self._refresh_image()
            self.image_changed.emit(self)

    def _delete_image(self):
        self.app.pop("image", None)
        self._refresh_image()
        self.image_changed.emit(self)

    # ── Status ─────────────────────────────────────────────────────────────
    def set_status(self, online: bool) -> None:
        if online == self._online:
            return
        self._online = online
        color = GRN if online else RED
        self._dot.setStyleSheet(
            f"color:{color}; font-size:10px;"
            f" background:rgba(0,0,0,160); border-radius:8px; border:none;")
        self.setStyleSheet(self._STYLE_ONLINE if online else self._STYLE)
        self._open_btn.setEnabled(online)


# ── "+" add box for WebUI section ─────────────────────────────────────────────
class AddWebUIBox(QFrame):
    """Clicking this box opens the Create WebUI dialog."""

    add_requested = Signal()

    _W, _H = 220, 264

    _STYLE = f"""
        QFrame {{
            background-color: {CAR};
            border: 2px dashed {MUT};
            border-radius: 8px;
        }}
    """
    _STYLE_HOVER = f"""
        QFrame {{
            background-color: #152236;
            border: 2px dashed {ACC};
            border-radius: 8px;
        }}
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(self._STYLE)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        plus = QLabel("+", self)
        plus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plus.setStyleSheet(
            f"color:{MUT}; font-size:64px; font-weight:bold;"
            f" background:transparent; border:none;")
        layout.addWidget(plus)

        lbl = QLabel("Add WebUI", self)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:12px;"
            f" background:transparent; border:none;")
        layout.addWidget(lbl)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.add_requested.emit()
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.setStyleSheet(self._STYLE_HOVER)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self._STYLE)
        super().leaveEvent(event)


# ── WebUI create / edit dialog ────────────────────────────────────────────────
class WebUIEntryDialog(QDialog):
    """Used for both creating new and editing existing WebUI entries."""

    _FIELD_STYLE = f"""
        QLineEdit {{
            background:{BG}; color:{PRI};
            border:1px solid {MUT}; border-radius:4px;
            font-family:{FONT}; font-size:{FONT_MD}px;
            padding:4px 8px; min-height:28px;
        }}
        QLineEdit:focus {{ border-color:{ACC}; }}
    """

    def __init__(self, app: dict | None = None, parent=None):
        super().__init__(parent)
        self.finished.connect(self.deleteLater)   # free after exec(); parented dialogs otherwise pile up
        is_edit = app is not None
        self.setWindowTitle("Edit WebUI Entry" if is_edit else "Create WebUI Entry")
        self.setMinimumWidth(380)
        self.setStyleSheet(f"QDialog {{ background:{BG}; }} QLabel {{ color:{PRI}; font-family:{FONT}; }}")

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._name = QLineEdit()
        self._name.setStyleSheet(self._FIELD_STYLE)
        self._name.setPlaceholderText("e.g.  SD.Next")
        if is_edit:
            self._name.setText(app.get("name", ""))
        form.addRow("Name:", self._name)

        self._url = QLineEdit()
        self._url.setStyleSheet(self._FIELD_STYLE)
        self._url.setPlaceholderText("http://localhost:7860")
        if is_edit:
            self._url.setText(app.get("url", ""))
        else:
            self._url.setText("http://localhost:")
        form.addRow("URL / IP:", self._url)

        self._cmd = QLineEdit()
        self._cmd.setStyleSheet(self._FIELD_STYLE)
        self._cmd.setPlaceholderText("C:\\path\\to\\start.bat  (optional)")
        if is_edit:
            self._cmd.setText(app.get("cmd", ""))
        form.addRow("Launch cmd:", self._cmd)

        layout.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Save" if is_edit else "Create")
        btns.accepted.connect(self._validate)
        btns.rejected.connect(self.reject)
        btns.setStyleSheet(f"""
            QPushButton {{
                background:{ACC}; color:{PRI}; border:none;
                border-radius:4px; font-family:{FONT};
                font-size:{FONT_MD}px; font-weight:bold;
                min-width:70px; min-height:28px; padding:0 12px;
            }}
            QPushButton:hover {{ background:#185FA5; }}
        """)
        layout.addWidget(btns)

    def _validate(self):
        if not self._name.text().strip():
            QMessageBox.warning(self, "Required", "Name is required.")
            return
        if not self._url.text().strip():
            QMessageBox.warning(self, "Required", "URL / IP address is required.")
            return
        self.accept()

    def get_data(self) -> dict:
        data: dict = {
            "name": self._name.text().strip(),
            "url":  self._url.text().strip(),
        }
        cmd = self._cmd.text().strip()
        if cmd:
            data["cmd"] = cmd
        return data


# ── Shared QSS for context menus (used by CustomAppBox & WebUIBox) ────────────
_MENU_QSS = f"""
    QMenu {{
        background:{CAR}; color:{PRI};
        border:1px solid {MUT};
        font-family:{FONT}; font-size:12px;
        padding:4px 0;
    }}
    QMenu::item {{ padding:5px 20px; }}
    QMenu::item:selected {{ background:{ACC}; }}
    QMenu::separator {{ background:{MUT}; height:1px; margin:4px 8px; }}
"""


# ── Custom external-app box ───────────────────────────────────────────────────
class CustomAppBox(QFrame):
    """
    User-added external application box in the Applications section.
    Left-click launches the app; right-click opens management menu.
    Supports drag-and-drop image (same as WebUIBox).
    """

    edit_requested   = Signal(object)  # self — cmd/name changed
    name_changed     = Signal(object)  # self — rename only, save without rebuild
    delete_requested = Signal(object)  # self
    image_changed    = Signal(object)  # self — image dropped/edited/cleared

    _W     = 220
    _H     = 264
    _BAR_H = 44
    _IMG_H = _W - 4   # = 216, square

    _STYLE = f"""
        QFrame {{
            background-color: {CAR};
            border: 2px solid {MUT};
            border-radius: 8px;
        }}
    """
    _STYLE_HOVER = f"""
        QFrame {{
            background-color: #152236;
            border: 2px solid {ACC};
            border-radius: 8px;
        }}
    """

    def __init__(self, app: dict, parent=None):
        super().__init__(parent)
        self.app = app
        self.setFixedSize(self._W, self._H)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(self._STYLE)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # ── Image area (square) ────────────────────────────────────────────
        self._img_lbl = QLabel(self)
        self._img_lbl.setFixedSize(self._W - 4, self._IMG_H)
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setStyleSheet(
            f"background:{BG}; border:none; border-radius:6px 6px 0 0;")
        self._refresh_image()
        vbox.addWidget(self._img_lbl)

        # ── Bottom bar ─────────────────────────────────────────────────────
        bar = QWidget(self)
        bar.setFixedHeight(self._BAR_H)
        bar.setStyleSheet(f"background:{PAN}; border:none; border-radius:0 0 6px 6px;")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 0, 8, 0)

        self._name_lbl = QLabel(app["name"], bar)
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_lbl.setStyleSheet(
            f"color:{PRI}; font-family:{FONT}; font-size:13px;"
            f" font-weight:bold; background:transparent; border:none;")
        bar_layout.addWidget(self._name_lbl)
        vbox.addWidget(bar)

    # ── Image ──────────────────────────────────────────────────────────────
    def _refresh_image(self):
        img_path = self.app.get("image", "")
        if img_path and os.path.isfile(img_path):
            w, h = self._W - 4, self._IMG_H
            weak_self = weakref.ref(self)
            def _done(fut):
                self_ref = weak_self()
                if self_ref is None:
                    return
                result = fut.result()
                def _apply():
                    try:
                        if result is not None and not result.isNull():
                            self_ref._img_lbl.setPixmap(QPixmap.fromImage(result))
                            self_ref._img_lbl.setStyleSheet(
                                "border:none; border-radius:6px 6px 0 0;")
                        else:
                            self_ref._show_image_placeholder()
                    except RuntimeError:
                        pass
                _MAIN_INVOKER.run.emit(_apply)
            _IMG_EXECUTOR.submit(_scale_qimage_bg, img_path, w, h).add_done_callback(_done)
            return
        self._show_image_placeholder()

    def _show_image_placeholder(self):
        self._img_lbl.clear()
        self._img_lbl.setText("Drop image here")
        self._img_lbl.setStyleSheet(
            f"color:{MUT}; font-size:11px; background:{BG};"
            f" border:none; border-radius:6px 6px 0 0;")

    # ── Drag-and-drop ──────────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for u in event.mimeData().urls():
                if u.toLocalFile().lower().endswith(
                        _IMG_EXTS):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for u in event.mimeData().urls():
            path = u.toLocalFile()
            if path.lower().endswith(
                    _IMG_EXTS):
                self.app["image"] = path
                self._refresh_image()
                self.image_changed.emit(self)
                event.acceptProposedAction()
                return
        event.ignore()

    # ── Mouse / hover ──────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._launch()
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.setStyleSheet(self._STYLE_HOVER)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self._STYLE)
        super().leaveEvent(event)

    # ── Context menu ───────────────────────────────────────────────────────
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_QSS)
        act_launch   = menu.addAction("▶  Launch")
        menu.addSeparator()
        act_edit     = menu.addAction("✎  Edit")
        act_rename   = menu.addAction("✏  Rename")
        menu.addSeparator()
        act_edit_img = menu.addAction("🖼  Edit Image")
        act_del_img  = menu.addAction("🗑  Delete Image")
        menu.addSeparator()
        act_delete   = menu.addAction("✕  Delete Entry")

        chosen = menu.exec(event.globalPos())
        if chosen == act_launch:
            self._launch()
        elif chosen == act_edit:
            self._open_edit_dialog()
        elif chosen == act_rename:
            self._rename()
        elif chosen == act_edit_img:
            self._browse_image()
        elif chosen == act_del_img:
            self._delete_image()
        elif chosen == act_delete:
            self.delete_requested.emit(self)

    # ── Actions ────────────────────────────────────────────────────────────
    def _launch(self):
        cmd = self.app.get("cmd", "")
        if not cmd:
            QMessageBox.warning(
                self, "No Command",
                f"No launch command set for \"{self.app['name']}\".\n"
                "Right-click → Edit to set one.")
            return
        import subprocess
        exe = cmd.split()[0].strip('"')
        cwd = os.path.dirname(exe) if os.path.isfile(exe) else None
        subprocess.Popen(cmd, shell=True, cwd=cwd)

    def _open_edit_dialog(self):
        dlg = CustomAppEntryDialog(self.app, parent=self)
        if dlg.exec():
            self.app.update(dlg.get_data())
            self._name_lbl.setText(self.app["name"])
            self.edit_requested.emit(self)

    def _rename(self):
        new_name, ok = QInputDialog.getText(
            self, "Rename", "New name:", text=self.app["name"])
        if ok and new_name.strip():
            self.app["name"] = new_name.strip()
            self._name_lbl.setText(new_name.strip())
            self.name_changed.emit(self)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            _IMG_FILTER)
        if path:
            self.app["image"] = path
            self._refresh_image()
            self.image_changed.emit(self)

    def _delete_image(self):
        self.app.pop("image", None)
        self._refresh_image()
        self.image_changed.emit(self)


# ── "+" add custom app box ────────────────────────────────────────────────────
class AddCustomAppBox(QFrame):
    """The '+' box for adding new custom application entries."""

    add_requested = Signal()

    _W, _H = 220, 264

    _STYLE = f"""
        QFrame {{
            background-color: {CAR};
            border: 2px dashed {MUT};
            border-radius: 8px;
        }}
    """
    _STYLE_HOVER = f"""
        QFrame {{
            background-color: #152236;
            border: 2px dashed {ACC};
            border-radius: 8px;
        }}
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(self._STYLE)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        plus = QLabel("+", self)
        plus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plus.setStyleSheet(
            f"color:{MUT}; font-size:64px; font-weight:bold;"
            f" background:transparent; border:none;")
        layout.addWidget(plus)

        lbl = QLabel("Add Application", self)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:12px;"
            f" background:transparent; border:none;")
        layout.addWidget(lbl)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.add_requested.emit()
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.setStyleSheet(self._STYLE_HOVER)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self._STYLE)
        super().leaveEvent(event)


# ── Custom app create / edit dialog ──────────────────────────────────────────
class CustomAppEntryDialog(QDialog):
    """Create/edit dialog for a custom external application."""

    _FIELD_STYLE = f"""
        QLineEdit {{
            background:{BG}; color:{PRI};
            border:1px solid {MUT}; border-radius:4px;
            font-family:{FONT}; font-size:{FONT_MD}px;
            padding:4px 8px; min-height:28px;
        }}
        QLineEdit:focus {{ border-color:{ACC}; }}
    """

    def __init__(self, app: dict | None = None, parent=None):
        super().__init__(parent)
        self.finished.connect(self.deleteLater)   # free after exec(); parented dialogs otherwise pile up
        is_edit = app is not None
        self.setWindowTitle("Edit Application" if is_edit else "Add Application")
        self.setMinimumWidth(420)
        self.setStyleSheet(
            f"QDialog {{ background:{BG}; }}"
            f" QLabel {{ color:{PRI}; font-family:{FONT}; }}")

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._name = QLineEdit()
        self._name.setStyleSheet(self._FIELD_STYLE)
        self._name.setPlaceholderText("e.g.  Photoshop")
        if is_edit:
            self._name.setText(app.get("name", ""))
        form.addRow("Name:", self._name)
        layout.addLayout(form)

        # Command row — field + Browse button
        cmd_lbl = QLabel("Command / Path:", self)
        cmd_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;")
        layout.addWidget(cmd_lbl)

        cmd_row = QHBoxLayout()
        cmd_row.setSpacing(4)

        self._cmd = QLineEdit()
        self._cmd.setStyleSheet(self._FIELD_STYLE)
        self._cmd.setPlaceholderText("C:\\path\\to\\app.exe  or  script.bat")
        if is_edit:
            self._cmd.setText(app.get("cmd", ""))
        cmd_row.addWidget(self._cmd, stretch=1)

        browse_btn = QPushButton("Browse…", self)
        browse_btn.setFixedHeight(36)
        browse_btn.setMinimumWidth(72)
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                background:{MUT}; color:{PRI}; border:none;
                border-radius:4px; font-family:{FONT};
                font-size:{FONT_SM}px; padding:0 8px;
            }}
            QPushButton:hover {{ background:{ACC}; }}
        """)
        browse_btn.clicked.connect(self._browse)
        cmd_row.addWidget(browse_btn)
        layout.addLayout(cmd_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Save" if is_edit else "Add")
        btns.accepted.connect(self._validate)
        btns.rejected.connect(self.reject)
        btns.setStyleSheet(f"""
            QPushButton {{
                background:{ACC}; color:{PRI}; border:none;
                border-radius:4px; font-family:{FONT};
                font-size:{FONT_MD}px; font-weight:bold;
                min-width:70px; min-height:28px; padding:0 12px;
            }}
            QPushButton:hover {{ background:#185FA5; }}
        """)
        layout.addWidget(btns)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Executable", "",
            "Executables (*.exe *.bat *.cmd *.sh *.py);;All files (*)")
        if path:
            self._cmd.setText(path)

    def _validate(self):
        if not self._name.text().strip():
            QMessageBox.warning(self, "Required", "Name is required.")
            return
        self.accept()

    def get_data(self) -> dict:
        data: dict = {"name": self._name.text().strip()}
        cmd = self._cmd.text().strip()
        if cmd:
            data["cmd"] = cmd
        return data


# ── Settings dialog ───────────────────────────────────────────────────────────
class SettingsDialog(QDialog):
    """Application settings — currently houses the HuggingFace token."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.finished.connect(self.deleteLater)   # free after exec(); parented dialogs otherwise pile up
        self.setWindowTitle("Settings")
        self.setMinimumWidth(440)
        self.setStyleSheet(f"""
            QDialog   {{ background:{BG}; }}
            QLabel    {{ color:{PRI}; font-family:{FONT}; font-size:{FONT_MD}px; }}
            QGroupBox {{ color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;
                         font-weight:bold; border:1px solid {MUT};
                         border-radius:6px; margin-top:8px; padding:8px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        # ── HuggingFace token ──────────────────────────────────────────────
        hf_frame = QFrame(self)
        hf_frame.setStyleSheet(
            f"QFrame {{ background:{CAR}; border:1px solid {MUT}; border-radius:6px; }}")
        hf_vbox = QVBoxLayout(hf_frame)
        hf_vbox.setContentsMargins(12, 10, 12, 10)
        hf_vbox.setSpacing(8)

        hf_title = QLabel("HuggingFace Token", hf_frame)
        hf_title.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" font-weight:bold; background:transparent; border:none;")
        hf_vbox.addWidget(hf_title)

        token_row = QHBoxLayout()
        token_row.setSpacing(4)

        self._token_edit = QLineEdit(hf_frame)
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_edit.setPlaceholderText("hf_••••••••••••••••••••••••••••••••••••")
        self._token_edit.setFixedHeight(30)
        self._token_edit.setStyleSheet(f"""
            QLineEdit {{
                background:{BG}; color:{PRI};
                border:1px solid {MUT}; border-radius:4px;
                font-family:{FONT}; font-size:{FONT_SM}px; padding:2px 8px;
            }}
            QLineEdit:focus {{ border-color:{ACC}; }}
        """)
        token_row.addWidget(self._token_edit, stretch=1)

        show_btn = QPushButton("👁", hf_frame)
        show_btn.setFixedSize(QSize(28, 30))
        show_btn.setCheckable(True)
        show_btn.setStyleSheet(f"""
            QPushButton {{
                background:{MUT}; color:{PRI}; border:none;
                border-radius:4px; font-size:13px;
            }}
            QPushButton:checked {{ background:{ACC}; }}
        """)
        show_btn.toggled.connect(
            lambda v: self._token_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if v else QLineEdit.EchoMode.Password))
        token_row.addWidget(show_btn)

        hf_vbox.addLayout(token_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        save_btn = QPushButton("Save Token", hf_frame)
        save_btn.setFixedHeight(28)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background:{ACC}; color:{PRI}; border:none; border-radius:4px;
                font-family:{FONT}; font-size:{FONT_SM}px; font-weight:bold; padding:0 14px;
            }}
            QPushButton:hover {{ background:#185FA5; }}
        """)
        save_btn.clicked.connect(self._save_token)
        btn_row.addWidget(save_btn)

        clear_btn = QPushButton("Clear", hf_frame)
        clear_btn.setFixedHeight(28)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{MUT};
                border:1px solid {MUT}; border-radius:4px;
                font-family:{FONT}; font-size:{FONT_SM}px; padding:0 14px;
            }}
            QPushButton:hover {{ border-color:{RED}; color:{RED}; }}
        """)
        clear_btn.clicked.connect(self._clear_token)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()

        self._hf_status = QLabel("", hf_frame)
        self._hf_status.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")
        btn_row.addWidget(self._hf_status)

        hf_vbox.addLayout(btn_row)
        outer.addWidget(hf_frame)

        # ── Close button ───────────────────────────────────────────────────
        outer.addStretch()
        close_btn = QPushButton("Close", self)
        close_btn.setFixedHeight(30)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background:{MUT}; color:{PRI}; border:none; border-radius:4px;
                font-family:{FONT}; font-size:{FONT_MD}px; font-weight:bold;
            }}
            QPushButton:hover {{ background:{ACC}; }}
        """)
        close_btn.clicked.connect(self.accept)
        outer.addWidget(close_btn)

        # Pre-fill saved token
        saved = load_hf_token()
        if saved:
            self._token_edit.setText(saved)
            self._hf_status.setText("● Token set")
            self._hf_status.setStyleSheet(
                f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f" background:transparent; border:none;")

    def _save_token(self):
        token = self._token_edit.text().strip()
        if not token:
            return
        save_hf_token(token)
        ok = apply_hf_token(token)
        if ok:
            self._hf_status.setText("✅ Token saved")
            self._hf_status.setStyleSheet(
                f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f" background:transparent; border:none;")
        else:
            self._hf_status.setText("⚠ Login failed")
            self._hf_status.setStyleSheet(
                f"color:{RED}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f" background:transparent; border:none;")

    def _clear_token(self):
        self._token_edit.clear()
        save_hf_token("")
        os.environ.pop("HF_TOKEN", None)
        self._hf_status.setText("○ Not set")
        self._hf_status.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")


# ── Manage page (visual redesign) ─────────────────────────────────────────────
class ManagePage(QWidget):
    """
    Two-section visual Manage page:
      1. Applications  — built-in app boxes + user-added custom app boxes + "+" box
      2. WebUI/Loader  — card boxes with drag-and-drop image + right-click menu
    """

    open_app     = Signal(str)  # url  — open WebUI browser tab
    open_builtin = Signal(str)  # name — open built-in app
    apps_changed = Signal(list) # webui list changed — notify Launcher/poller

    _BUILTIN_APPS = [
        ("Tags",         "tags.png"),
        ("Calculator",   "calculator.png"),
        ("Enhancer",     "Enhancer.png"),
        ("Face Swap",    "Face Swap.png"),
        ("Randomizer",   "Randomizer.png"),
        ("LoRA Health",  "health.png"),
        ("Model Merge",  "merge.png"),
        ("SD.UI",        "sdui.png"),
    ]

    _SECTION_LABEL = (
        f"color:{SEC}; font-family:{FONT}; font-size:10px;"
        f" font-weight:bold; letter-spacing:2px;"
        f" background:transparent;"
    )

    def __init__(self, apps: list[dict], parent=None):
        super().__init__(parent)
        self.apps               = apps
        self._custom_apps       = _load_custom_apps()
        self._webui_boxes:       list[WebUIBox]     = []
        self._custom_app_boxes:  list[CustomAppBox] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Outer scroll (vertical) ────────────────────────────────────────
        outer_scroll = QScrollArea(self)
        outer_scroll.setWidgetResizable(True)
        outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer_scroll.setStyleSheet(
            f"QScrollArea {{ border:none; background:{PAN}; }}")

        content = QWidget()
        content.setStyleSheet(f"background:{PAN};")
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(16, 16, 16, 20)
        vbox.setSpacing(12)

        # ── Applications section ───────────────────────────────────────────
        apps_label = QLabel("APPLICATIONS", content)
        apps_label.setStyleSheet(self._SECTION_LABEL)
        vbox.addWidget(apps_label)

        app_scroll = QScrollArea(content)
        app_scroll.setFixedHeight(AppIconBox._H + 24)
        app_scroll.setWidgetResizable(True)
        app_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        app_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        app_scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        self._app_row_widget = QWidget()
        self._app_row_widget.setStyleSheet(f"background:{PAN};")
        self._app_row = QHBoxLayout(self._app_row_widget)
        self._app_row.setContentsMargins(0, 4, 0, 4)
        self._app_row.setSpacing(16)

        # Effective built-in list: drop SD.UI when its local module isn't present (see _SDUI_PRESENT).
        self._builtin_apps = [(n, i) for (n, i) in self._BUILTIN_APPS
                              if n != "SD.UI" or _SDUI_PRESENT]
        for name, icon_file in self._builtin_apps:
            box = AppIconBox(name, os.path.join(_ASSETS, icon_file),
                             self._app_row_widget)
            box.clicked_app.connect(self.open_builtin)
            self._app_row.addWidget(box)

        # Custom app boxes + "+" are appended by _refresh_custom_apps()
        self._refresh_custom_apps()

        app_scroll.setWidget(self._app_row_widget)
        vbox.addWidget(app_scroll)

        # ── WebUI / Loader section ─────────────────────────────────────────
        web_label = QLabel("WEBUI / LOADER", content)
        web_label.setStyleSheet(self._SECTION_LABEL)
        vbox.addWidget(web_label)

        web_scroll = QScrollArea(content)
        web_scroll.setFixedHeight(WebUIBox._H + 24)
        web_scroll.setWidgetResizable(True)
        web_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        web_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        web_scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        self._webui_row_widget = QWidget()
        self._webui_row_widget.setStyleSheet(f"background:{PAN};")
        self._webui_layout = QHBoxLayout(self._webui_row_widget)
        self._webui_layout.setContentsMargins(0, 4, 0, 4)
        self._webui_layout.setSpacing(12)

        web_scroll.setWidget(self._webui_row_widget)
        vbox.addWidget(web_scroll)

        vbox.addStretch()
        outer_scroll.setWidget(content)
        root.addWidget(outer_scroll)

        self._refresh_webui()

    # ── WebUI card management ──────────────────────────────────────────────

    def _refresh_webui(self) -> None:
        while self._webui_layout.count():
            item = self._webui_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._webui_boxes.clear()

        for app in self.apps:
            box = WebUIBox(app, self._webui_row_widget)
            box.open_requested.connect(self.open_app)
            box.edit_requested.connect(self._on_webui_edit)
            box.name_changed.connect(self._on_name_changed)
            box.delete_requested.connect(self._on_webui_delete)
            box.image_changed.connect(self._on_image_changed)
            self._webui_layout.addWidget(box)
            self._webui_boxes.append(box)

        add_box = AddWebUIBox(self._webui_row_widget)
        add_box.add_requested.connect(self._create_webui)
        self._webui_layout.addWidget(add_box)
        self._webui_layout.addStretch()

    def poll_update(self, statuses: dict) -> None:
        for box in self._webui_boxes:
            url = box.app.get("url", "")
            if url in statuses:
                box.set_status(statuses[url])

    def _create_webui(self) -> None:
        dlg = WebUIEntryDialog(parent=self)
        if dlg.exec():
            self.apps.append(dlg.get_data())
            save_apps(self.apps)
            self.apps_changed.emit(self.apps)
            self._refresh_webui()

    def _on_webui_edit(self, box: WebUIBox) -> None:
        # box.app already updated in-place by the dialog inside WebUIBox
        save_apps(self.apps)
        self.apps_changed.emit(self.apps)
        self._refresh_webui()

    def _on_name_changed(self, box: WebUIBox) -> None:
        # In-place name update — just save, no rebuild needed
        save_apps(self.apps)
        self.apps_changed.emit(self.apps)

    def _on_webui_delete(self, box: WebUIBox) -> None:
        try:
            self.apps.remove(box.app)
        except ValueError:
            pass
        save_apps(self.apps)
        self.apps_changed.emit(self.apps)
        self._refresh_webui()

    def _on_image_changed(self, box: WebUIBox) -> None:
        # Image updated in-place — just save
        save_apps(self.apps)

    # ── Custom app management ──────────────────────────────────────────────

    def _refresh_custom_apps(self) -> None:
        """Remove custom boxes + trailing stretch/add-box, rebuild from self._custom_apps."""
        builtin_count = len(self._builtin_apps)
        # Strip everything appended after the 5 built-in boxes
        while self._app_row.count() > builtin_count:
            item = self._app_row.takeAt(builtin_count)
            w = item.widget()
            if w:
                w.deleteLater()
        self._custom_app_boxes.clear()

        for app in self._custom_apps:
            box = CustomAppBox(app, self._app_row_widget)
            box.edit_requested.connect(self._on_custom_edit)
            box.name_changed.connect(self._on_custom_name_changed)
            box.delete_requested.connect(self._on_custom_delete)
            box.image_changed.connect(self._on_custom_image_changed)
            self._app_row.addWidget(box)
            self._custom_app_boxes.append(box)

        add_box = AddCustomAppBox(self._app_row_widget)
        add_box.add_requested.connect(self._create_custom_app)
        self._app_row.addWidget(add_box)
        self._app_row.addStretch()

    def _create_custom_app(self) -> None:
        dlg = CustomAppEntryDialog(parent=self)
        if dlg.exec():
            self._custom_apps.append(dlg.get_data())
            _save_custom_apps(self._custom_apps)
            self._refresh_custom_apps()

    def _on_custom_edit(self, box: CustomAppBox) -> None:
        # box.app already updated in-place by the dialog inside CustomAppBox
        _save_custom_apps(self._custom_apps)
        self._refresh_custom_apps()

    def _on_custom_name_changed(self, box: CustomAppBox) -> None:
        _save_custom_apps(self._custom_apps)

    def _on_custom_delete(self, box: CustomAppBox) -> None:
        try:
            self._custom_apps.remove(box.app)
        except ValueError:
            pass
        _save_custom_apps(self._custom_apps)
        self._refresh_custom_apps()

    def _on_custom_image_changed(self, box: CustomAppBox) -> None:
        _save_custom_apps(self._custom_apps)


# ── Poll worker (signals safe across threads) ─────────────────────────────────
class PollWorker(QObject):
    results_ready = Signal(dict)   # {url: bool}

    def __init__(self, apps: list[dict], parent=None):
        super().__init__(parent)
        self.apps             = apps
        # Separate executors: checkers run in parallel; one collector waits for them.
        # Using two pools eliminates the per-cycle threading.Thread() churn.
        self._check_executor  = ThreadPoolExecutor(max_workers=4,
                                                   thread_name_prefix="poll_check")
        self._collect_executor = ThreadPoolExecutor(max_workers=1,
                                                    thread_name_prefix="poll_collect")

    def run(self) -> None:
        urls = [a["url"] for a in self.apps]
        if not urls:
            self.results_ready.emit({})
            return
        futures = {url: self._check_executor.submit(check_port, url) for url in urls}

        def collect():
            results = {url: f.result() for url, f in futures.items()}
            self.results_ready.emit(results)

        self._collect_executor.submit(collect)

    def update_apps(self, apps: list[dict]) -> None:
        self.apps = apps

    def shutdown(self) -> None:
        self._check_executor.shutdown(wait=False)
        self._collect_executor.shutdown(wait=False)


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
        self._face_page     = None
        self._enh_page      = None
        self._health_page   = None
        self._tags_index:   int | None = None
        self._calc_index:   int | None = None
        self._rand_index:   int | None = None
        self._face_index:   int | None = None
        self._enh_index:    int | None = None
        self._health_index: int | None = None
        self._merge_page:   None                = None
        self._merge_index:  int | None = None
        self._poser_page:   None                = None
        self._poser_index:  int | None = None
        self._poll_timer:   QTimer | None = None

        self._build_ui()
        self._start_polling()

        # Restore HuggingFace session if token was previously saved
        _saved_token = load_hf_token()
        if _saved_token:
            apply_hf_token(_saved_token)

    # ═══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ═══════════════════════════════════════════════════════════════════════════

    def _setup_menubar(self):
        mb = self.menuBar()
        mb.setStyleSheet(f"""
            QMenuBar {{
                background: {CAR};
                color: {PRI};
                font-family: {FONT};
                font-size: {FONT_SM}px;
                border-bottom: 1px solid {MUT};
                spacing: 2px;
                padding: 1px 4px;
            }}
            QMenuBar::item {{
                background: transparent;
                padding: 4px 10px;
                border-radius: 3px;
            }}
            QMenuBar::item:selected {{ background: {ACC}; }}
            QMenu {{
                background: {CAR};
                color: {PRI};
                border: 1px solid {MUT};
                font-family: {FONT};
                font-size: {FONT_SM}px;
                padding: 4px 0;
            }}
            QMenu::item {{ padding: 5px 20px; }}
            QMenu::item:selected {{ background: {ACC}; }}
            QMenu::separator {{
                background: {MUT};
                height: 1px;
                margin: 4px 8px;
            }}
        """)

        file_menu = mb.addMenu("File")

        act_settings = file_menu.addAction("Settings…")
        act_settings.triggered.connect(self._open_settings)

        file_menu.addSeparator()

        act_exit = file_menu.addAction("Exit")
        act_exit.triggered.connect(self.close)

    def _build_ui(self):
        self._setup_menubar()

        root        = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        # ── Header ────────────────────────────────────────────────────────
        self._header = QFrame(self)
        self._header.setObjectName("header")
        self._header.setFixedHeight(44)
        self._header.setStyleSheet(
            f"QFrame#header {{ background-color:{CAR}; border:none; }}")
        hrow = QHBoxLayout(self._header)
        hrow.setContentsMargins(4, 4, 8, 4)
        hrow.setSpacing(0)

        # Manage tab (always visible, no close)
        self._manage_tab = self._make_nav_tab("Manage", self._show_manage)
        hrow.addWidget(self._manage_tab)
        hrow.addSpacing(4)

        # Built-in app tabs — hidden by default, revealed when app is opened
        self._tags_tab = self._make_nav_tab(
            "Tags", self._show_tags, closeable=True)
        self._calc_tab = self._make_nav_tab(
            "Calculator", self._show_calculator, closeable=True)
        self._rand_tab = self._make_nav_tab(
            "Randomizer", self._show_randomizer, closeable=True)
        self._face_tab = self._make_nav_tab(
            "Face Swap", self._show_faceswap, closeable=True)
        self._enh_tab    = self._make_nav_tab(
            "Enhancer", self._show_enhancer, closeable=True)
        self._health_tab = self._make_nav_tab(
            "LoRA Health", self._show_health, closeable=True)
        self._merge_tab  = self._make_nav_tab(
            "Model Merge", self._show_merge, closeable=True)
        self._poser_tab  = self._make_nav_tab(
            "SD.UI", self._show_sdui, closeable=True, refresh=True)

        self._tags_tab.close_requested.connect(self._close_tags)
        self._calc_tab.close_requested.connect(self._close_calculator)
        self._rand_tab.close_requested.connect(self._close_randomizer)
        self._face_tab.close_requested.connect(self._close_faceswap)
        self._enh_tab.close_requested.connect(self._close_enhancer)
        self._health_tab.close_requested.connect(self._close_health)
        self._merge_tab.close_requested.connect(self._close_merge)
        self._poser_tab.close_requested.connect(self._close_sdui)
        self._poser_tab.refresh_requested.connect(self._restart_sdui)   # tab ↻ = reboot SD.UI in place

        for btn in (self._tags_tab, self._calc_tab, self._rand_tab,
                    self._face_tab, self._enh_tab, self._health_tab,
                    self._merge_tab, self._poser_tab):
            hrow.addWidget(btn)
            hrow.addSpacing(4)
            btn.setVisible(False)   # hidden until app is opened

        # Dynamic WebUI app tabs
        self._app_tabs_area = QHBoxLayout()
        self._app_tabs_area.setContentsMargins(0, 0, 0, 0)
        self._app_tabs_area.setSpacing(4)
        hrow.addLayout(self._app_tabs_area)

        hrow.addStretch(1)

        # Zoom controls
        btn_minus = self._make_zoom_btn("−", self._zoom_out)
        self._zoom_lbl = QLabel("100%", self._header)
        self._zoom_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:11px;"
            f" min-width:46px; max-width:46px;"
            f" qproperty-alignment:AlignCenter;")
        self._zoom_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self._zoom_lbl.mousePressEvent = lambda e: self._zoom_manual()
        btn_plus = self._make_zoom_btn("+", self._zoom_in)

        zoom_frame = QWidget(self._header)
        zoom_row   = QHBoxLayout(zoom_frame)
        zoom_row.setContentsMargins(0, 0, 0, 0)
        zoom_row.setSpacing(2)
        zoom_row.addWidget(btn_minus)
        zoom_row.addWidget(self._zoom_lbl)
        zoom_row.addWidget(btn_plus)
        hrow.addWidget(zoom_frame)
        # Signature
        sig = QLabel(SIGNATURE, self._header)
        sig.setStyleSheet(f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;")
        hrow.addWidget(sig)

        root_layout.addWidget(self._header)

        # ── 1px separator ─────────────────────────────────────────────────
        sep = QFrame(self)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color:{MUT};")
        root_layout.addWidget(sep)

        # ── Content stack ──────────────────────────────────────────────────
        self._stack = QStackedWidget(self)
        root_layout.addWidget(self._stack, stretch=1)

        self._manage_page = ManagePage(self.apps)
        self._manage_page.open_app.connect(self._open_app_tab)
        self._manage_page.open_builtin.connect(self._open_builtin)
        self._manage_page.apps_changed.connect(self._on_apps_changed)
        self._stack.addWidget(self._manage_page)   # index 0

        self._show_manage()

    def _make_nav_tab(self, label: str, callback,
                      closeable: bool = False, refresh: bool = False) -> TabButton:
        btn = TabButton(label, show_dot=closeable, show_close=closeable,
                        enable_refresh=refresh, parent=self._header)
        btn.clicked.connect(callback)
        return btn

    @staticmethod
    def _make_zoom_btn(label: str, callback) -> QPushButton:
        btn = QPushButton(label)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color:{CAR}; color:{PRI}; border:none;
                font-family:{FONT}; font-size:16px; font-weight:bold;
                min-width:26px; max-width:26px;
                min-height:28px; max-height:28px;
            }}
            QPushButton:hover {{ background-color:{MUT}; }}
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
        self._face_tab.set_active(False)
        self._enh_tab.set_active(False)
        self._health_tab.set_active(False)
        self._merge_tab.set_active(False)
        self._poser_tab.set_active(False)
        for entry in self._tab_data.values():
            entry["tab_btn"].set_active(False)

    def _show_manage(self):
        self._deactivate_all()
        self._manage_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self.MANAGE_INDEX)
        self._set_zoom_display(1.0)
        if self._poll_timer:
            self._poll_timer.setInterval(3_000)   # fast poll — dots visible on Manage page

    def _open_builtin(self, name: str) -> None:
        dispatch = {
            "Tags":        self._show_tags,
            "Calculator":  self._show_calculator,
            "Enhancer":    self._show_enhancer,
            "Face Swap":   self._show_faceswap,
            "Randomizer":  self._show_randomizer,
            "LoRA Health":  self._show_health,
            "Model Merge":  self._show_merge,
            "SD.UI":        self._show_sdui,
        }
        fn = dispatch.get(name)
        if fn:
            fn()

    def _show_tags(self):
        if self._tags_page is None:
            if not self._vram_preflight("Tagger", free_engine_ckpt=True):
                return
        self._tags_tab.setVisible(True)
        if self._tags_page is None:
            from shared import crash_guard
            def _make():
                from tags.tag_handler_page import TagHandlerPage
                return TagHandlerPage()
            page = crash_guard.guard_open("Tagger", _make, parent=self)
            if page is None:
                self._tags_tab.setVisible(False)
                self._show_manage()
                return
            self._tags_page  = page
            self._tags_index = self._stack.addWidget(self._tags_page)
            self._tags_tab.set_dot_online(True)
        self._deactivate_all()
        self._tags_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self._tags_index)
        self._set_zoom_display(1.0)

    def _show_calculator(self):
        self._calc_tab.setVisible(True)
        if self._calc_page is None:
            # CPU-only: no VRAM pre-flight, but still contain a Python init failure to this tab.
            from shared import crash_guard
            def _make():
                from calculator.calculator_page import CalculatorPage
                return CalculatorPage()
            page = crash_guard.guard_open("Calculator", _make, parent=self)
            if page is None:
                self._calc_tab.setVisible(False)
                self._show_manage()
                return
            self._calc_page  = page
            self._calc_index = self._stack.addWidget(self._calc_page)
            self._calc_tab.set_dot_online(True)
        self._deactivate_all()
        self._calc_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self._calc_index)
        self._set_zoom_display(1.0)

    def _show_randomizer(self):
        self._rand_tab.setVisible(True)
        if self._rand_page is None:
            # CPU-only: no VRAM pre-flight, but still contain a Python init failure to this tab.
            from shared import crash_guard
            def _make():
                from randomizer.randomizer_page import RandomizerPage
                return RandomizerPage()
            page = crash_guard.guard_open("Randomizer", _make, parent=self)
            if page is None:
                self._rand_tab.setVisible(False)
                self._show_manage()
                return
            self._rand_page  = page
            self._rand_index = self._stack.addWidget(self._rand_page)
            self._rand_tab.set_dot_online(True)
        self._deactivate_all()
        self._rand_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self._rand_index)
        self._set_zoom_display(1.0)

    def _show_faceswap(self):
        if self._face_page is None:
            if not self._vram_preflight("Face Swap", free_engine_ckpt=True):
                return
        self._face_tab.setVisible(True)
        if self._face_page is None:
            from shared import crash_guard
            def _make():
                from faces.face_swap_page import FaceSwapPage
                return FaceSwapPage()
            page = crash_guard.guard_open("Face Swap", _make, parent=self)
            if page is None:
                self._face_tab.setVisible(False)
                self._show_manage()
                return
            self._face_page  = page
            self._face_index = self._stack.addWidget(self._face_page)
            self._face_tab.set_dot_online(True)
        self._deactivate_all()
        self._face_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self._face_index)
        self._set_zoom_display(1.0)

    def _show_enhancer(self):
        if self._enh_page is None:
            if not self._vram_preflight("Enhancer", free_engine_ckpt=True):
                return
        self._enh_tab.setVisible(True)
        if self._enh_page is None:
            from shared import crash_guard
            def _make():
                from enhancer.enhancer_page import EnhancerPage
                return EnhancerPage()
            page = crash_guard.guard_open("Enhancer", _make, parent=self)
            if page is None:
                self._enh_tab.setVisible(False)
                self._show_manage()
                return
            self._enh_page  = page
            self._enh_index = self._stack.addWidget(self._enh_page)
            self._enh_tab.set_dot_online(True)
        self._deactivate_all()
        self._enh_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self._enh_index)
        self._set_zoom_display(1.0)

    def _vram_preflight(self, module: str, free_engine_ckpt: bool = False, need_gb: float = 3.0) -> bool:
        """Free reclaimable VRAM (LM Studio models Jarvis left resident; the engine checkpoint when the
        module doesn't need it) BEFORE opening a heavy GPU module, so its native libs don't OOM the card
        and crash the whole Suite. If VRAM still won't fit, ask the user to proceed or cancel.
        Returns True to open, False to abort. Never raises."""
        from PySide6.QtWidgets import QApplication
        try:
            from shared import vram_guard
        except Exception:
            return True                      # guard unavailable -> don't block the user
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rep = vram_guard.preflight(module, free_engine_ckpt=free_engine_ckpt, need_gb=need_gb)
        except Exception:
            rep = None
        finally:
            QApplication.restoreOverrideCursor()
        if rep and rep.get("fit") is False:
            r = QMessageBox.warning(
                self, f"Low VRAM — {module}", rep["message"],
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            return r == QMessageBox.StandardButton.Yes
        return True

    def _show_health(self):
        if self._health_page is None:
            # Health does NOT use the render engine, so free its checkpoint too for max headroom.
            if not self._vram_preflight("LoRA Health", free_engine_ckpt=True):
                return
        self._health_tab.setVisible(True)
        if self._health_page is None:
            import sys
            # Purge on (re)open so the tab always loads the latest Health code (charts / checks / scoring)
            # without a full Suite restart — matches the SD.UI reopen-purge. Safe: no live HealthPage
            # exists here (self._health_page is None), so nothing references the modules being dropped.
            for key in list(sys.modules.keys()):
                if key.startswith("health"):
                    del sys.modules[key]
            from shared import crash_guard
            def _make():
                from health.health_page import HealthPage
                return HealthPage()
            page = crash_guard.guard_open("LoRA Health", _make, parent=self)
            if page is None:                 # Python-level init failure -> fail the module, not the Suite
                self._health_tab.setVisible(False)
                self._show_manage()
                return
            self._health_page  = page
            self._health_index = self._stack.addWidget(self._health_page)
            self._health_tab.set_dot_online(True)
        self._deactivate_all()
        self._health_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self._health_index)
        self._set_zoom_display(1.0)

    def _open_app_tab(self, url: str):
        if url in self._tab_data:
            self._switch_to_tab(url)
            return

        view = BrowserView(self._profile, url)
        idx  = self._stack.addWidget(view)
        self._stack_index[url] = idx

        app     = next((a for a in self.apps if a["url"] == url),
                       {"name": url, "url": url})
        tab_btn = TabButton(app["name"], show_dot=True, show_close=True,
                            enable_refresh=True, parent=self._header)
        tab_btn.clicked.connect(lambda u=url: self._switch_to_tab(u))
        tab_btn.close_requested.connect(lambda u=url: self._close_tab(u))
        tab_btn.refresh_requested.connect(view.reload)
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
        if self._poll_timer:
            self._poll_timer.setInterval(10_000)  # slow poll — status dots not visible

    def _reclaim_after_close(self):
        """After a heavy module's page is torn down, hand its allocator cache + freed RAM back. Deferred so
        Qt's deleteLater and Python's gc have already run (empty_cache only frees once the tensors' refs are
        gone). Single-process reclaim — see shared/proc_memory. Cheap + safe when torch was never loaded."""
        from shared import proc_memory
        QTimer.singleShot(1200, proc_memory.reclaim)

    def _close_tab(self, url: str):
        if url not in self._tab_data:
            return
        entry              = self._tab_data.pop(url)
        view: BrowserView  = entry["view"]
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

    def _close_tags(self):
        if self._tags_page is None:
            return
        if getattr(self._tags_page, '_tags_changed', None):
            from PySide6.QtWidgets import QMessageBox
            r = QMessageBox.question(
                self, "Unsaved Changes",
                "Tag edits have not been saved.\nClose anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return
        self._stack.removeWidget(self._tags_page)
        self._tags_page.deleteLater()
        self._tags_page  = None
        self._tags_index = None
        self._tags_tab.set_active(False)
        self._tags_tab.set_dot_online(False)
        self._tags_tab.setVisible(False)
        import sys
        for key in list(sys.modules.keys()):
            if key.startswith("tags"):
                del sys.modules[key]
        self._rebuild_stack_index()
        self._show_manage()
        self._reclaim_after_close()

    def _close_calculator(self):
        if self._calc_page is None:
            return
        self._stack.removeWidget(self._calc_page)
        self._calc_page.deleteLater()
        self._calc_page  = None
        self._calc_index = None
        self._calc_tab.set_active(False)
        self._calc_tab.set_dot_online(False)
        self._calc_tab.setVisible(False)
        import sys
        for key in list(sys.modules.keys()):
            if key.startswith("calculator") or key == "shared.calc_engine":
                del sys.modules[key]
        self._rebuild_stack_index()
        self._show_manage()

    def _close_randomizer(self):
        if self._rand_page is None:
            return
        self._stack.removeWidget(self._rand_page)
        self._rand_page.deleteLater()
        self._rand_page  = None
        self._rand_index = None
        self._rand_tab.set_active(False)
        self._rand_tab.set_dot_online(False)
        self._rand_tab.setVisible(False)
        import sys
        for key in list(sys.modules.keys()):
            if key.startswith("randomizer"):
                del sys.modules[key]
        self._rebuild_stack_index()
        self._show_manage()
        self._reclaim_after_close()

    def _close_faceswap(self):
        if self._face_page is None:
            return
        self._stack.removeWidget(self._face_page)
        self._face_page.deleteLater()
        self._face_page  = None
        self._face_index = None
        self._face_tab.set_active(False)
        self._face_tab.set_dot_online(False)
        self._face_tab.setVisible(False)
        import sys
        for key in list(sys.modules.keys()):
            if key.startswith("faces"):
                del sys.modules[key]
        self._rebuild_stack_index()
        self._show_manage()
        self._reclaim_after_close()

    def _close_enhancer(self):
        if self._enh_page is None:
            return
        self._stack.removeWidget(self._enh_page)
        self._enh_page.deleteLater()
        self._enh_page  = None
        self._enh_index = None
        self._enh_tab.set_active(False)
        self._enh_tab.set_dot_online(False)
        self._enh_tab.setVisible(False)
        import sys
        for key in list(sys.modules.keys()):
            if key.startswith("enhancer"):
                del sys.modules[key]
        self._rebuild_stack_index()
        self._show_manage()
        self._reclaim_after_close()

    def _close_health(self):
        if self._health_page is None:
            return
        self._health_page.cleanup()
        self._stack.removeWidget(self._health_page)
        self._health_page.deleteLater()
        self._health_page  = None
        self._health_index = None
        self._health_tab.set_active(False)
        self._health_tab.set_dot_online(False)
        self._reclaim_after_close()

    def _show_sdui(self):
        if self._poser_page is None:
            # SD.UI NEEDS the engine checkpoint (it renders through it) -> keep it, only evict the
            # LM Studio models Jarvis leaves resident.
            if not self._vram_preflight("SD.UI", free_engine_ckpt=False):
                return
        self._poser_tab.setVisible(True)
        if self._poser_page is None:
            import sys
            # purge on (re)open too, so the tab always loads the latest SD.UI code even if the
            # close-time purge was missed — no full Suite restart needed for Python changes
            for key in list(sys.modules.keys()):
                if key.startswith("sdui") and not key.startswith("sdui.engine"):
                    del sys.modules[key]
            from shared import crash_guard
            def _make():
                from sdui.sdui_page import SDUIPage
                return SDUIPage()
            page = crash_guard.guard_open("SD.UI", _make, parent=self)
            if page is None:                 # Python-level init failure -> fail the module, not the Suite
                self._poser_tab.setVisible(False)
                self._show_manage()
                return
            self._poser_page  = page
            self._poser_index = self._stack.addWidget(self._poser_page)
            self._poser_tab.set_dot_online(True)
        self._deactivate_all()
        self._poser_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self._poser_index)
        self._set_zoom_display(1.0)

    def _restart_sdui(self):
        """Reboot the SD.UI tab in place: tear it down (purges + cleans up) then reopen it, which
        re-imports the sdui modules fresh and reloads the web poser. Loads SD.UI code changes
        without restarting the whole Suite; the engine subprocess (separate process) keeps running."""
        self._close_sdui()
        self._show_sdui()

    def _close_sdui(self):
        if self._poser_page is None:
            return
        if hasattr(self._poser_page, "cleanup"):
            self._poser_page.cleanup()
        self._stack.removeWidget(self._poser_page)
        self._poser_page.deleteLater()
        self._poser_page  = None
        self._poser_index = None
        self._poser_tab.set_active(False)
        self._poser_tab.set_dot_online(False)
        self._poser_tab.setVisible(False)
        import sys
        for key in list(sys.modules.keys()):
            if key.startswith("sdui"):
                del sys.modules[key]
        self._rebuild_stack_index()
        self._show_manage()
        self._reclaim_after_close()

    def _show_merge(self):
        if self._merge_page is None:
            # Merge loads full checkpoints into VRAM (torch) — free everything first.
            if not self._vram_preflight("Merge", free_engine_ckpt=True):
                return
        self._merge_tab.setVisible(True)
        if self._merge_page is None:
            from shared import crash_guard
            def _make():
                from merge.merge_page import MergePage
                return MergePage()
            page = crash_guard.guard_open("Merge", _make, parent=self)
            if page is None:
                self._merge_tab.setVisible(False)
                self._show_manage()
                return
            self._merge_page  = page
            self._merge_index = self._stack.addWidget(self._merge_page)
            self._merge_tab.set_dot_online(True)
        self._deactivate_all()
        self._merge_tab.set_active(True)
        self._active_url = None
        self._stack.setCurrentIndex(self._merge_index)
        self._set_zoom_display(1.0)

    def _close_merge(self):
        if self._merge_page is None:
            return
        self._merge_page.cleanup()
        self._stack.removeWidget(self._merge_page)
        self._merge_page.deleteLater()
        self._merge_page  = None
        self._merge_index = None
        self._merge_tab.set_active(False)
        self._merge_tab.set_dot_online(False)
        self._merge_tab.setVisible(False)
        import sys
        for key in list(sys.modules.keys()):
            if key.startswith("merge"):
                del sys.modules[key]
        self._rebuild_stack_index()
        self._show_manage()
        self._reclaim_after_close()

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
        if self._face_page:
            self._face_index = self._stack.indexOf(self._face_page)
        if self._enh_page:
            self._enh_index = self._stack.indexOf(self._enh_page)
        if self._health_page:
            self._health_index = self._stack.indexOf(self._health_page)
        if self._merge_page:
            self._merge_index = self._stack.indexOf(self._merge_page)

    # ═══════════════════════════════════════════════════════════════════════════
    # Settings
    # ═══════════════════════════════════════════════════════════════════════════

    def _open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()

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

    def _on_apps_changed(self, apps: list) -> None:
        self._poller.update_apps(apps)

    # ═══════════════════════════════════════════════════════════════════════════
    # Shutdown
    # ═══════════════════════════════════════════════════════════════════════════

    def changeEvent(self, event):
        # Idle CPU: pause the status poll while minimized (the dots aren't visible), and resume + refresh
        # immediately on restore. Cuts background CPU to ~zero when the Suite is tucked away.
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.WindowStateChange and self._poll_timer is not None:
            if self.isMinimized():
                self._poll_timer.stop()
            elif not self._poll_timer.isActive():
                self._poll_timer.start()
                self._poller.run()
        super().changeEvent(event)

    def closeEvent(self, event):
        self._poll_timer.stop()
        self._poller.shutdown()
        _IMG_EXECUTOR.shutdown(wait=False)
        for url in list(self._tab_data.keys()):
            entry             = self._tab_data.pop(url)
            view: BrowserView = entry["view"]
            view.setPage(None)
            view.deleteLater()
        # Release ML model memory and background workers
        try:
            from enhancer.enhancer_page     import release_enhancer_model
            from randomizer.randomizer_page  import release_randomizer_model
            from tags.tag_handler_page       import shutdown_image_loader
            release_enhancer_model()
            release_randomizer_model()
            shutdown_image_loader()
        except Exception:
            pass
        event.accept()
