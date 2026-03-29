"""
enhancer/enhancer_page.py — Image Enhancer page for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.0)

Controls:
  Model     — Realistic (RealESRGAN x4plus) / Anime (RealESRGAN x4plus anime 6B)
  Scale     — 1x  2x  4x  Custom
  Denoise   — cv2 fast NL-means denoising strength (0 = off)
  Deblur    — Unsharp mask strength (0 = off)
  Fix comp. — JPEG artifact removal strength (0 = off)
"""
from __future__ import annotations

import os
import sys
import io
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from PySide6.QtCore    import Qt, QThread, QObject, Signal
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFileDialog,
    QMessageBox, QProgressDialog, QApplication,
    QComboBox, QSlider, QSpinBox, QSizePolicy, QMenu,
)
from PySide6.QtGui import QCursor

from shared.theme import (
    BG, PAN, CAR, ACC, GRN, RED, MUT, PRI, SEC, AMB,
    FONT, FONT_SM, FONT_MD,
)
from shared.config import load_json, save_json
from randomizer.randomizer_page import BgViewerCard, _load_thumb

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
CFG_FILE     = os.path.join(_HERE, "enhancer_config.json")
MODELS_DIR   = Path(_HERE).parent / "tags" / "models"

THUMB_D    = 100
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
CARD_D     = 2000

ESR_MODELS = {
    "RealESRGAN_x4plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "~64 MB — photo/realistic upscale",
    ),
    "RealESRGAN_x4plus_anime_6B.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "~18 MB — anime/illustration upscale",
    ),
}

MODEL_OPTS   = ["Realistic", "Anime"]
SCALE_OPTS   = ["1x", "2x", "4x", "Custom"]

DEFAULTS = {
    "model":       "Realistic",
    "scale":       "4x",
    "scale_custom": 4,
    "denoise":     0,
    "deblur":      0,
    "fix_comp":    0,
    "saturation":  0,
    "contrast":    0,
    "last_input_dir":  "",
    "last_output_dir": "",
    "card_size":   CARD_D,
}


# ── UI helpers (shared style) ──────────────────────────────────────────────────

def _section(title: str) -> QLabel:
    lbl = QLabel(title)
    lbl.setStyleSheet(
        f"color:{ACC}; font-family:{FONT}; font-size:{FONT_MD}px;"
        f"font-weight:bold; padding:8px 0 4px 0; background:transparent;")
    return lbl


def _row_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
    lbl.setMinimumWidth(130)
    return lbl


def _val_label(val: int) -> QLabel:
    lbl = QLabel(str(val))
    lbl.setFixedWidth(30)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lbl.setStyleSheet(
        f"color:{PRI}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
    return lbl


def _slider(val: int, lo: int, hi: int) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    s.setStyleSheet(
        f"QSlider::groove:horizontal {{ height:4px; background:{MUT}; border-radius:2px; }}"
        f"QSlider::handle:horizontal {{ width:12px; height:12px; margin:-4px 0;"
        f"background:{ACC}; border-radius:6px; }}"
        f"QSlider::sub-page:horizontal {{ background:{ACC}; border-radius:2px; }}")
    return s


def _scale_btn(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setFixedHeight(28)
    b.setCheckable(True)
    b.setStyleSheet(
        f"QPushButton {{ background:{CAR}; color:{SEC}; border:1px solid {MUT};"
        f"border-radius:4px; padding:2px 10px; font-family:{FONT};"
        f"font-size:{FONT_SM}px; }}"
        f"QPushButton:checked {{ background:{ACC}; color:{PRI}; border:1px solid {ACC}; }}"
        f"QPushButton:hover:!checked {{ background:{MUT}; color:{PRI}; }}")
    return b


# ── Enhance worker ─────────────────────────────────────────────────────────────
class EnhanceWorker(QObject):
    finished = Signal(object)   # PIL Image
    error    = Signal(str)
    status   = Signal(str)

    def __init__(self, src: Image.Image, model_name: str, outscale: float,
                 denoise: int, deblur: int, fix_comp: int,
                 saturation: int = 0, contrast: int = 0,
                 target_w: int = 0, target_h: int = 0):
        super().__init__()
        self._src        = src
        self._model      = model_name
        self._outscale   = outscale
        self._target_w   = target_w
        self._target_h   = target_h
        self._denoise    = denoise
        self._deblur     = deblur
        self._fix_comp   = fix_comp
        self._saturation = saturation
        self._contrast   = contrast

    def run(self):
        try:
            import cv2

            arr = np.array(self._src.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            # ── Adjustments (applied at source resolution, before upscaling) ──

            # ── Denoise ───────────────────────────────────────────────────
            if self._denoise > 0:
                self.status.emit("Denoising…")
                h = max(1, self._denoise // 3)
                bgr = cv2.fastNlMeansDenoisingColored(
                    bgr, None, h=h, hColor=h,
                    templateWindowSize=7, searchWindowSize=21)

            # ── Fix compression (JPEG artifact removal) ────────────────────
            if self._fix_comp > 0:
                self.status.emit("Fixing compression artifacts…")
                h = max(1, self._fix_comp // 4)
                bgr = cv2.fastNlMeansDenoisingColored(
                    bgr, None, h=h, hColor=h,
                    templateWindowSize=5, searchWindowSize=15)

            # ── Deblur (unsharp mask) ─────────────────────────────────────
            if self._deblur > 0:
                self.status.emit("Sharpening…")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                radius = 1.5 + self._deblur * 0.03
                amount = 0.5 + self._deblur * 0.015
                pil = pil.filter(
                    ImageFilter.UnsharpMask(radius=radius,
                                            percent=int(amount * 100),
                                            threshold=2))
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            # ── Saturation / Contrast ─────────────────────────────────────
            if self._saturation != 0 or self._contrast != 0:
                from PIL import ImageEnhance
                pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if self._saturation != 0:
                    self.status.emit("Adjusting saturation…")
                    factor = 1.0 + self._saturation * 0.1
                    pil = ImageEnhance.Color(pil).enhance(max(0.0, factor))
                if self._contrast != 0:
                    self.status.emit("Adjusting contrast…")
                    factor = 1.0 + self._contrast * 0.1
                    pil = ImageEnhance.Contrast(pil).enhance(max(0.0, factor))
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            # ── RealESRGAN upscale ─────────────────────────────────────────
            do_upscale = (self._outscale != 1.0 or
                          (self._target_w > 0 and self._target_h > 0))
            if do_upscale:
                self.status.emit("Loading RealESRGAN model…")
                _stdout, _stderr = sys.stdout, sys.stderr
                if sys.stdout is None: sys.stdout = io.StringIO()
                if sys.stderr is None: sys.stderr = io.StringIO()
                try:
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from realesrgan import RealESRGANer

                    anime    = self._model == "Anime"
                    fname    = ("RealESRGAN_x4plus_anime_6B.pth" if anime
                                else "RealESRGAN_x4plus.pth")
                    n_blocks = 6 if anime else 23
                    mdl_path = str(MODELS_DIR / fname)

                    rrdb = RRDBNet(num_in_ch=3, num_out_ch=3,
                                   num_feat=64, num_block=n_blocks,
                                   num_grow_ch=32, scale=4)
                    upsampler = RealESRGANer(
                        scale=4, model_path=mdl_path, model=rrdb,
                        tile=0, tile_pad=10, pre_pad=0, half=False)

                    if self._target_w > 0 and self._target_h > 0:
                        # Custom dimensions: upscale enough then resize to exact target
                        src_w, src_h = self._src.size
                        scale_needed = max(self._target_w / src_w,
                                           self._target_h / src_h)
                        self.status.emit(
                            f"Upscaling to {self._target_w}×{self._target_h}…")
                        bgr, _ = upsampler.enhance(bgr, outscale=max(scale_needed, 1.0))
                        bgr = cv2.resize(bgr, (self._target_w, self._target_h),
                                         interpolation=cv2.INTER_LANCZOS4)
                    else:
                        self.status.emit(f"Upscaling {self._outscale}x…")
                        bgr, _ = upsampler.enhance(bgr, outscale=self._outscale)
                finally:
                    sys.stdout, sys.stderr = _stdout, _stderr

            result = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ── Main page ──────────────────────────────────────────────────────────────────
class EnhancerPage(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG};")

        self._cfg = load_json(CFG_FILE, DEFAULTS)

        self._src_pil:      Image.Image | None = None
        self._src_path:     str  = ""
        self._out_pil:      Image.Image | None = None
        self._preview_pil:  Image.Image | None = None
        self._viewing:      str  = "input"
        self._folder_imgs:  list[str] = []
        self._input_pills:  list[QPushButton] = []
        self._active_pill:  QPushButton | None = None
        self._thread:       QThread | None = None
        self._worker:       EnhanceWorker | None = None
        self._prog_dlg:     QProgressDialog | None = None
        self._ar_lock:      bool = True

        self._build_ui()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_toolbar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Left: image list panel (hidden until folder loaded)
        self._img_list_panel = self._build_img_list_panel()
        self._img_list_panel.setMaximumWidth(0)
        self._img_list_panel.setVisible(False)
        body.addWidget(self._img_list_panel)

        # Center: viewer
        body.addWidget(self._build_viewer_area(), stretch=1)

        # Right: settings
        body.addWidget(self._build_settings_panel())

        wrap = QWidget()
        wrap.setLayout(body)
        root.addWidget(wrap, stretch=1)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(54)
        bar.setStyleSheet(
            f"QFrame {{ background:{CAR}; border-bottom:1px solid {MUT}; }}")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 0, 12, 0)
        row.setSpacing(8)

        # Browse
        browse_btn = QPushButton("Browse ▾", bar)
        browse_btn.setFixedHeight(32)
        browse_btn.setStyleSheet(
            f"QPushButton {{ background:{ACC}; color:{PRI}; border:none;"
            f"border-radius:4px; padding:4px 12px; font-family:{FONT};"
            f"font-size:{FONT_MD}px; font-weight:bold; }}"
            f"QPushButton:hover {{ background:#185FA5; }}")
        browse_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        menu = QMenu(bar)
        menu.setStyleSheet(
            f"QMenu {{ background:{CAR}; color:{PRI}; border:1px solid {MUT}; }}"
            f"QMenu::item:selected {{ background:{ACC}; }}")
        menu.addAction("Browse Image",  self._browse_image)
        menu.addAction("Browse Folder", self._browse_folder)
        browse_btn.setMenu(menu)
        row.addWidget(browse_btn)

        self._path_lbl = QLabel("No image loaded", bar)
        self._path_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;")
        self._path_lbl.setMaximumWidth(360)
        row.addWidget(self._path_lbl)
        row.addStretch(1)

        def _btn(text, color, slot, enabled=True):
            b = QPushButton(text, bar)
            b.setEnabled(enabled)
            b.setFixedHeight(32)
            b.setStyleSheet(
                f"QPushButton {{ background:{color}; color:{PRI}; border:none;"
                f"border-radius:4px; padding:4px 12px; font-family:{FONT};"
                f"font-size:{FONT_MD}px; font-weight:bold; }}"
                f"QPushButton:disabled {{ background:{MUT}; color:{SEC}; }}"
                f"QPushButton:hover:!disabled {{ background:#185FA5; }}")
            b.clicked.connect(slot)
            return b

        self._apply_btn  = _btn("Enhance",      ACC, self._run_enhance,   False)
        self._toggle_btn = _btn("View Original", MUT, self._toggle_view)
        self._toggle_btn.setVisible(False)
        self._save_btn   = _btn("Save PNG",      ACC, self._save_output,   False)

        def _set_toggle(text: str):
            color      = ACC if text == "View Original" else GRN
            hover_clr  = "#185FA5" if text == "View Original" else "#237832"
            self._toggle_btn.setText(text)
            self._toggle_btn.setStyleSheet(
                f"QPushButton {{ background:{color}; color:{PRI}; border:none;"
                f"border-radius:4px; padding:4px 12px; font-family:{FONT};"
                f"font-size:{FONT_MD}px; font-weight:bold; }}"
                f"QPushButton:disabled {{ background:{MUT}; color:{SEC}; }}"
                f"QPushButton:hover:!disabled {{ background:{hover_clr}; }}")
        self._set_toggle = _set_toggle

        for w in (self._apply_btn, self._toggle_btn, self._save_btn):
            row.addWidget(w)

        row.addSpacing(12)
        self._status_lbl = QLabel("", bar)
        self._status_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;")
        self._status_lbl.setMinimumWidth(180)
        row.addWidget(self._status_lbl)

        return bar

    # ── Image list panel (left) ───────────────────────────────────────────────

    def _build_img_list_panel(self) -> QFrame:
        panel = QFrame()
        panel.setFixedWidth(THUMB_D + 16)
        panel.setStyleSheet(
            f"QFrame {{ background:{CAR}; border-right:1px solid {MUT}; }}")
        vl = QVBoxLayout(panel)
        vl.setContentsMargins(4, 4, 4, 4)
        vl.setSpacing(4)

        hdr = QFrame(panel)
        hdr.setFixedHeight(28)
        hdr.setStyleSheet(f"QFrame {{ background:{CAR}; border:none; }}")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(4, 0, 4, 0)

        lbl = QLabel("Images", hdr)
        lbl.setStyleSheet(
            f"color:{ACC}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f"font-weight:bold;")
        hl.addWidget(lbl, stretch=1)

        self._img_count = QLabel("0", hdr)
        self._img_count.setStyleSheet(
            f"background:{MUT}; color:{PRI}; font-family:{FONT};"
            f"font-size:{FONT_SM}px; border-radius:8px; padding:1px 5px;")
        hl.addWidget(self._img_count)
        vl.addWidget(hdr)

        sa = QScrollArea(panel)
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setStyleSheet(
            f"QScrollArea {{ background:{CAR}; border:none; }}"
            f"QScrollBar:vertical {{ background:{CAR}; width:6px; }}"
            f"QScrollBar::handle:vertical {{ background:{MUT}; border-radius:3px; }}")
        self._img_list_content = QWidget()
        self._img_list_content.setStyleSheet(f"background:{CAR};")
        self._img_list_vl = QVBoxLayout(self._img_list_content)
        self._img_list_vl.setContentsMargins(2, 2, 2, 2)
        self._img_list_vl.setSpacing(3)
        self._img_list_vl.addStretch(1)
        sa.setWidget(self._img_list_content)
        vl.addWidget(sa, stretch=1)
        return panel

    # ── Viewer (center) ───────────────────────────────────────────────────────

    def _build_viewer_area(self) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(False)
        sa.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sa.setStyleSheet(
            f"QScrollArea {{ background:{BG}; border:none; }}"
            f"QScrollBar:vertical, QScrollBar:horizontal"
            f" {{ background:{CAR}; width:8px; height:8px; }}"
            f"QScrollBar::handle {{ background:{MUT}; border-radius:4px; }}")
        self._viewer = BgViewerCard(self._cfg.get("card_size", CARD_D))
        sa.setWidget(self._viewer)
        return sa

    # ── Settings panel (right) ────────────────────────────────────────────────

    def _build_settings_panel(self) -> QFrame:
        panel = QFrame()
        panel.setFixedWidth(260)
        panel.setStyleSheet(
            f"QFrame {{ background:{CAR}; border-left:1px solid {MUT}; }}")

        sa = QScrollArea(panel)
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setStyleSheet(
            f"QScrollArea {{ background:{CAR}; border:none; }}"
            f"QScrollBar:vertical {{ background:{CAR}; width:6px; }}"
            f"QScrollBar::handle:vertical {{ background:{MUT}; border-radius:3px; }}")

        inner = QWidget()
        inner.setStyleSheet(f"background:{CAR};")
        vl = QVBoxLayout(inner)
        vl.setContentsMargins(14, 14, 14, 14)
        vl.setSpacing(0)

        def _spinbox_wh(val: int) -> QSpinBox:
            sb = QSpinBox()
            sb.setRange(1, 32768)
            sb.setValue(val)
            sb.setFixedWidth(72)
            sb.setStyleSheet(
                f"QSpinBox {{ background:{BG}; color:{PRI}; border:1px solid {MUT};"
                f"border-radius:4px; padding:3px 4px; font-family:{FONT};"
                f"font-size:{FONT_SM}px; }}"
                f"QSpinBox::up-button, QSpinBox::down-button {{ width:14px; }}")
            return sb

        # ── AI Model ──────────────────────────────────────────────────────
        vl.addWidget(_section("AI Model"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(MODEL_OPTS)
        self._model_combo.setCurrentText(self._cfg.get("model", "Realistic"))
        self._model_combo.setStyleSheet(
            f"QComboBox {{ background:{BG}; color:{PRI}; border:1px solid {MUT};"
            f"border-radius:4px; padding:4px 8px; font-family:{FONT};"
            f"font-size:{FONT_SM}px; }}"
            f"QComboBox::drop-down {{ border:none; width:20px; }}"
            f"QComboBox QAbstractItemView {{ background:{CAR}; color:{PRI};"
            f"selection-background-color:{ACC}; }}")
        self._model_combo.currentTextChanged.connect(
            lambda t: self._save_cfg("model", t))
        vl.addWidget(self._model_combo)
        vl.addSpacing(4)

        self._model_status = QLabel("", inner)
        self._model_status.setWordWrap(True)
        self._model_status.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f"background:transparent;")
        vl.addWidget(self._model_status)
        self._model_combo.currentTextChanged.connect(self._update_model_status)
        self._update_model_status(self._model_combo.currentText())

        # ── Minor Denoise ─────────────────────────────────────────────────
        vl.addSpacing(8)
        vl.addWidget(_section("Minor Denoise"))
        row_dn = QHBoxLayout()
        self._denoise_sl  = _slider(self._cfg.get("denoise", 0), 0, 100)
        self._denoise_val = _val_label(self._cfg.get("denoise", 0))
        self._denoise_sl.valueChanged.connect(
            lambda v: (self._denoise_val.setText(str(v)),
                       self._save_cfg("denoise", v)))
        self._denoise_sl.sliderReleased.connect(self._on_slider_released)
        row_dn.addWidget(self._denoise_sl)
        row_dn.addWidget(self._denoise_val)
        vl.addLayout(row_dn)

        # ── Minor Deblur ──────────────────────────────────────────────────
        vl.addSpacing(4)
        vl.addWidget(_section("Minor Deblur"))
        row_db = QHBoxLayout()
        self._deblur_sl  = _slider(self._cfg.get("deblur", 0), 0, 100)
        self._deblur_val = _val_label(self._cfg.get("deblur", 0))
        self._deblur_sl.valueChanged.connect(
            lambda v: (self._deblur_val.setText(str(v)),
                       self._save_cfg("deblur", v)))
        self._deblur_sl.sliderReleased.connect(self._on_slider_released)
        row_db.addWidget(self._deblur_sl)
        row_db.addWidget(self._deblur_val)
        vl.addLayout(row_db)

        # ── Fix Compression ───────────────────────────────────────────────
        vl.addSpacing(4)
        vl.addWidget(_section("Fix Compression"))
        row_fc = QHBoxLayout()
        self._fixcomp_sl  = _slider(self._cfg.get("fix_comp", 0), 0, 100)
        self._fixcomp_val = _val_label(self._cfg.get("fix_comp", 0))
        self._fixcomp_sl.valueChanged.connect(
            lambda v: (self._fixcomp_val.setText(str(v)),
                       self._save_cfg("fix_comp", v)))
        self._fixcomp_sl.sliderReleased.connect(self._on_slider_released)
        row_fc.addWidget(self._fixcomp_sl)
        row_fc.addWidget(self._fixcomp_val)
        vl.addLayout(row_fc)

        # ── Saturation ────────────────────────────────────────────────────
        vl.addSpacing(4)
        vl.addWidget(_section("Saturation"))
        row_sat = QHBoxLayout()
        self._sat_sl  = _slider(self._cfg.get("saturation", 0), -10, 10)
        self._sat_val = _val_label(self._cfg.get("saturation", 0))
        self._sat_sl.valueChanged.connect(
            lambda v: (self._sat_val.setText(str(v)),
                       self._save_cfg("saturation", v)))
        self._sat_sl.sliderReleased.connect(self._on_slider_released)
        row_sat.addWidget(self._sat_sl)
        row_sat.addWidget(self._sat_val)
        vl.addLayout(row_sat)

        # ── Contrast ──────────────────────────────────────────────────────
        vl.addSpacing(4)
        vl.addWidget(_section("Contrast"))
        row_con = QHBoxLayout()
        self._con_sl  = _slider(self._cfg.get("contrast", 0), -10, 10)
        self._con_val = _val_label(self._cfg.get("contrast", 0))
        self._con_sl.valueChanged.connect(
            lambda v: (self._con_val.setText(str(v)),
                       self._save_cfg("contrast", v)))
        self._con_sl.sliderReleased.connect(self._on_slider_released)
        row_con.addWidget(self._con_sl)
        row_con.addWidget(self._con_val)
        vl.addLayout(row_con)

        # ── Scale ─────────────────────────────────────────────────────────
        vl.addSpacing(8)
        vl.addWidget(_section("Scale"))

        scale_row = QHBoxLayout()
        scale_row.setSpacing(4)
        self._scale_btns = {}
        for opt in SCALE_OPTS:
            b = _scale_btn(opt)
            b.clicked.connect(lambda _, o=opt: self._on_scale_btn(o))
            self._scale_btns[opt] = b
            scale_row.addWidget(b)
        vl.addLayout(scale_row)
        vl.addSpacing(6)

        # Custom W × H (only visible when Custom is selected)
        self._custom_widget = QWidget()
        self._custom_widget.setStyleSheet("background:transparent;")
        cv = QVBoxLayout(self._custom_widget)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(4)

        w_row = QHBoxLayout()
        w_lbl = QLabel("W:", self._custom_widget)
        w_lbl.setFixedWidth(18)
        w_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        self._custom_w = _spinbox_wh(self._cfg.get("custom_w", 1024))
        self._custom_w.editingFinished.connect(self._on_custom_w_changed)
        w_row.addWidget(w_lbl)
        w_row.addWidget(self._custom_w)
        w_row.addStretch(1)

        h_row = QHBoxLayout()
        h_lbl = QLabel("H:", self._custom_widget)
        h_lbl.setFixedWidth(18)
        h_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        self._custom_h = _spinbox_wh(self._cfg.get("custom_h", 1024))
        self._custom_h.editingFinished.connect(self._on_custom_h_changed)
        h_row.addWidget(h_lbl)
        h_row.addWidget(self._custom_h)
        h_row.addStretch(1)
        self._ar_lock = True   # aspect ratio lock flag

        cv.addLayout(w_row)
        cv.addLayout(h_row)
        self._custom_widget.setVisible(False)
        vl.addWidget(self._custom_widget)

        # Restore scale selection
        saved_scale = self._cfg.get("scale", "4x")
        self._on_scale_btn(saved_scale if saved_scale in SCALE_OPTS else "4x")

        vl.addStretch(1)
        sa.setWidget(inner)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(sa)
        return panel

    # ── Scale button logic ────────────────────────────────────────────────────

    def _on_scale_btn(self, opt: str):
        for k, b in self._scale_btns.items():
            b.setChecked(k == opt)
        self._custom_widget.setVisible(opt == "Custom")
        self._save_cfg("scale", opt)

    def _current_outscale(self) -> float:
        sel = next((k for k, b in self._scale_btns.items() if b.isChecked()), "4x")
        if sel == "1x":  return 1.0
        if sel == "2x":  return 2.0
        if sel == "4x":  return 4.0
        # Custom — derive scale from W/H relative to source
        if self._src_pil is not None:
            sw, sh = self._src_pil.size
            tw, th = self._custom_w.value(), self._custom_h.value()
            return max(tw / sw, th / sh)
        return 4.0

    # ── Model status ──────────────────────────────────────────────────────────

    def _update_model_status(self, model_text: str):
        anime = model_text == "Anime"
        fname = ("RealESRGAN_x4plus_anime_6B.pth" if anime
                 else "RealESRGAN_x4plus.pth")
        exists = (MODELS_DIR / fname).exists()
        if exists:
            self._model_status.setText(f"✓ {fname}")
            self._model_status.setStyleSheet(
                f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f"background:transparent;")
        else:
            _, desc = ESR_MODELS[fname]
            self._model_status.setText(f"Not downloaded ({desc})")
            self._model_status.setStyleSheet(
                f"color:{AMB}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f"background:transparent;")

    def _ensure_model(self) -> bool:
        anime = self._model_combo.currentText() == "Anime"
        fname = ("RealESRGAN_x4plus_anime_6B.pth" if anime
                 else "RealESRGAN_x4plus.pth")
        path  = MODELS_DIR / fname
        if path.exists():
            return True
        url, desc = ESR_MODELS[fname]
        r = QMessageBox.question(
            self, "Model Not Found",
            f"{fname} not found.\n[{desc}]\n\nDownload now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes:
            return False
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            dlg = QProgressDialog(f"Downloading {fname}…", None, 0, 0, self)
            dlg.setWindowTitle("Download")
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            dlg.show()
            QApplication.processEvents()
            urllib.request.urlretrieve(url, str(path))
            dlg.close()
            self._update_model_status(self._model_combo.currentText())
            return True
        except Exception as e:
            QMessageBox.critical(self, "Download Failed", str(e))
            return False

    # ── Browse / load ─────────────────────────────────────────────────────────

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self._cfg.get("last_input_dir", ""),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif)")
        if not path:
            return
        self._cfg["last_input_dir"] = os.path.dirname(path)
        save_json(CFG_FILE, self._cfg)
        self._load_image(path)
        self._img_list_panel.setMaximumWidth(0)
        self._img_list_panel.setVisible(False)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder", self._cfg.get("last_input_dir", ""))
        if not folder:
            return
        self._cfg["last_input_dir"] = folder
        save_json(CFG_FILE, self._cfg)
        paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])
        if not paths:
            QMessageBox.warning(self, "No Images",
                                "No supported images found in that folder.")
            return
        self._folder_imgs = paths
        # Rebuild pills
        for p in self._input_pills:
            self._img_list_vl.removeWidget(p)
            p.deleteLater()
        self._input_pills.clear()
        self._active_pill = None
        for path in paths:
            pill = QPushButton(os.path.basename(path), self._img_list_content)
            pill.setFixedHeight(24)
            pill.setStyleSheet(self._pill_ss(False))
            pill.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            pill.clicked.connect(lambda _, p=pill, pa=path:
                                 self._select_pill(p, pa))
            self._img_list_vl.insertWidget(self._img_list_vl.count() - 1, pill)
            self._input_pills.append(pill)
        self._img_count.setText(str(len(paths)))
        self._img_list_panel.setMaximumWidth(THUMB_D + 16)
        self._img_list_panel.setVisible(True)
        if paths:
            self._select_pill(self._input_pills[0], paths[0])

    def _pill_ss(self, active: bool) -> str:
        bg = ACC if active else MUT
        return (f"QPushButton {{ background:{bg}; color:{PRI}; border:none;"
                f"border-radius:3px; text-align:left; padding:0 6px;"
                f"font-family:{FONT}; font-size:{FONT_SM}px; }}"
                f"QPushButton:hover {{ background:{ACC}; color:{PRI}; }}")

    def _select_pill(self, pill: QPushButton, path: str):
        if self._active_pill:
            self._active_pill.setStyleSheet(self._pill_ss(False))
        self._active_pill = pill
        pill.setStyleSheet(self._pill_ss(True))
        self._load_image(path)

    def _load_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))
            return
        self._src_pil     = img
        self._src_path    = path
        self._out_pil     = None
        self._preview_pil = None
        self._viewing     = "input"
        name = os.path.basename(path)
        self._path_lbl.setText(name[:50] + ("…" if len(name) > 50 else ""))
        self._viewer.set_image(img)
        w, h = img.size
        self._status_lbl.setText(f"{w} × {h} px")
        self._save_btn.setEnabled(False)
        self._toggle_btn.setVisible(False)
        self._apply_btn.setEnabled(True)
        # Populate custom W/H with current image dimensions (no AR callback)
        self._ar_lock = False
        self._custom_w.setValue(w)
        self._custom_h.setValue(h)
        self._ar_lock = True
        self._save_cfg("custom_w", w)
        self._save_cfg("custom_h", h)

    # ── Enhance ───────────────────────────────────────────────────────────────

    def _on_custom_w_changed(self):
        if not self._ar_lock or self._src_pil is None:
            self._save_cfg("custom_w", self._custom_w.value())
            return
        sw, sh = self._src_pil.size
        new_h = max(1, round(self._custom_w.value() * sh / sw))
        self._ar_lock = False
        self._custom_h.setValue(new_h)
        self._ar_lock = True
        self._save_cfg("custom_w", self._custom_w.value())
        self._save_cfg("custom_h", new_h)

    def _on_custom_h_changed(self):
        if not self._ar_lock or self._src_pil is None:
            self._save_cfg("custom_h", self._custom_h.value())
            return
        sw, sh = self._src_pil.size
        new_w = max(1, round(self._custom_h.value() * sw / sh))
        self._ar_lock = False
        self._custom_w.setValue(new_w)
        self._ar_lock = True
        self._save_cfg("custom_w", new_w)
        self._save_cfg("custom_h", self._custom_h.value())

    def _on_slider_released(self):
        """Preview adjustments on slider release.
        Only runs at 1x — at other scales the result is identical to final (no upscale needed).
        At 2x/4x/Custom the preview is skipped to avoid running RealESRGAN on every drag."""
        if self._src_pil is None:
            return
        sel = next((k for k, b in self._scale_btns.items() if b.isChecked()), "")
        if sel != "1x":
            self._status_lbl.setText("Preview available at 1x only — click Enhance to process")
            return
        if self._thread and self._thread.isRunning():
            return
        self._status_lbl.setText("Preview…")
        QApplication.processEvents()
        self._thread = QThread()
        self._worker = EnhanceWorker(
            self._src_pil, self._model_combo.currentText(), 1.0,
            self._denoise_sl.value(),
            self._deblur_sl.value(),
            self._fixcomp_sl.value(),
            saturation=self._sat_sl.value(),
            contrast=self._con_sl.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_preview_done)
        self._worker.error.connect(lambda e: self._status_lbl.setText(f"Preview error: {e}"))
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_preview_done(self, result: Image.Image):
        self._preview_pil = result
        self._viewing  = "preview"
        self._viewer.set_image(result)
        w, h = result.size
        sel = next((k for k, b in self._scale_btns.items() if b.isChecked()), "1x")
        note = "" if sel == "1x" else f"  (adjustments only — click Enhance for {sel} upscale)"
        self._status_lbl.setText(f"Preview  —  {w} × {h} px{note}")
        self._toggle_btn.setVisible(True)
        self._set_toggle("View Original")

    def _run_enhance(self):
        if self._src_pil is None:
            return
        sel = next((k for k, b in self._scale_btns.items() if b.isChecked()), "4x")
        custom = sel == "Custom"
        outscale = 1.0 if custom else self._current_outscale()
        target_w = self._custom_w.value() if custom else 0
        target_h = self._custom_h.value() if custom else 0
        needs_model = custom or outscale != 1.0
        if needs_model and not self._ensure_model():
            return

        self._apply_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._toggle_btn.setVisible(False)
        self._status_lbl.setText("Processing…")

        self._prog_dlg = QProgressDialog("Enhancing image…", None, 0, 0, self)
        self._prog_dlg.setWindowTitle("Enhancer")
        self._prog_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._prog_dlg.show()

        self._thread = QThread()
        self._worker = EnhanceWorker(
            self._src_pil,
            self._model_combo.currentText(),
            outscale,
            self._denoise_sl.value(),
            self._deblur_sl.value(),
            self._fixcomp_sl.value(),
            saturation=self._sat_sl.value(),
            contrast=self._con_sl.value(),
            target_w=target_w,
            target_h=target_h,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.status.connect(
            lambda s: self._prog_dlg.setLabelText(s) if self._prog_dlg else None)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_done(self, result: Image.Image):
        if self._prog_dlg:
            self._prog_dlg.close()
            self._prog_dlg = None
        self._out_pil  = result
        self._viewing  = "output"
        self._viewer.set_image(result)
        w, h = result.size
        self._status_lbl.setText(f"Done  —  {w} × {h} px")
        self._apply_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        self._toggle_btn.setVisible(True)
        self._set_toggle("View Original")

    def _on_error(self, msg: str):
        if self._prog_dlg:
            self._prog_dlg.close()
            self._prog_dlg = None
        self._status_lbl.setText("Error")
        self._apply_btn.setEnabled(self._src_pil is not None)
        QMessageBox.critical(self, "Enhance Error", msg)

    def _toggle_view(self):
        if self._viewing in ("output", "preview") and self._src_pil:
            self._viewer.set_image(self._src_pil)
            self._viewing = "input"
            # label depends on what we can go back to
            if self._out_pil:
                self._set_toggle("View Result")
            else:
                self._set_toggle("View Preview")
        elif self._viewing == "input":
            if self._out_pil:
                self._viewer.set_image(self._out_pil)
                self._viewing = "output"
                self._set_toggle("View Original")
            elif self._preview_pil:
                self._viewer.set_image(self._preview_pil)
                self._viewing = "preview"
                self._set_toggle("View Original")

    # ── Save ─────────────────────────────────────────────────────────────────

    def _save_output(self):
        if self._out_pil is None:
            return
        base = os.path.splitext(os.path.basename(self._src_path))[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image",
            os.path.join(self._cfg.get("last_output_dir", ""),
                         f"{base}-enhanced.png"),
            "PNG (*.png)")
        if not path:
            return
        self._cfg["last_output_dir"] = os.path.dirname(path)
        save_json(CFG_FILE, self._cfg)
        try:
            self._out_pil.save(path)
            self._status_lbl.setText(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # ── Config ────────────────────────────────────────────────────────────────

    def _save_cfg(self, key: str, value):
        self._cfg[key] = value
        save_json(CFG_FILE, self._cfg)
