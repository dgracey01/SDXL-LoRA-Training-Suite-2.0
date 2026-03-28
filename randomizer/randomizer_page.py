"""
randomizer/randomizer_page.py — Background Remover for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.0)

Models:
  Realism → BRIA RMBG-2.0   (briaai/RMBG-2.0)        ~885 MB
  Anime   → ToonOut          (joelseytre/toonout)      ~885 MB

Workflow:
  1. Browse an input image
  2. Select a random background from the right panel (optional)
  3. Click Remove BG → background removed, then composited over selected bg
  4. Save PNG  (RGBA transparent if no bg selected, RGB composite otherwise)
"""

import os
import threading
from pathlib import Path

from PIL import Image, ImageDraw
from PySide6.QtCore    import Qt, Signal, QObject, QThread
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QScrollArea, QSlider, QSpinBox, QFileDialog, QMessageBox,
    QProgressDialog, QApplication, QMenu, QCheckBox,
)
from PySide6.QtGui import QPixmap, QImage, QCursor, QAction

from shared.theme import (
    BG, PAN, CAR, ACC, GRN, RED, MUT, PRI, SEC, AMB,
    FONT, FONT_SM, FONT_MD, FONT_LG, VERSION,
)
from shared.config import load_json, save_json

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
CFG_FILE  = os.path.join(_HERE, "randomizer_config.json")
BG_FOLDER = os.path.join(_HERE, "random images")

# ── Constants ─────────────────────────────────────────────────────────────────
CARD_D      = 2000
CARD_D_MIN  =  400
CARD_D_MAX  = 3000
CARD_D_STEP =  100
THUMB_D     =  100
IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

MODELS = {
    "Realism": {
        "repo":  "briaai/RMBG-2.0",
        "check": "config.json",
        "size":  "~885 MB",
        "desc":  "BRIA RMBG-2.0 — photorealistic background removal",
    },
    "Anime": {
        "repo":  "joelseytre/toonout",
        "check": "config.json",
        "size":  "~885 MB",
        "desc":  "ToonOut — BiRefNet fine-tuned for anime/illustration",
    },
}

DEFAULTS = {
    "last_input_dir":  "",
    "last_output_dir": "",
    "mode":            "Realism",
    "card_size":       2000,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_checkerboard(size: int, tile: int = 20) -> Image.Image:
    img  = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    c1, c2 = (200, 200, 200), (155, 155, 155)
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            fill = c1 if (x // tile + y // tile) % 2 == 0 else c2
            draw.rectangle([x, y, x + tile - 1, y + tile - 1], fill=fill)
    return img


def _pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode == "RGBA":
        data = img.tobytes("raw", "RGBA")
        qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
    else:
        rgb  = img.convert("RGB")
        data = rgb.tobytes("raw", "RGB")
        qimg = QImage(data, rgb.width, rgb.height, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _load_thumb(path: str, d: int = THUMB_D) -> QPixmap:
    """Load an image, fit it inside a d×d square, return QPixmap."""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((d, d), Image.LANCZOS)
        canvas = Image.new("RGB", (d, d), (26, 26, 46))
        ox = (d - img.width)  // 2
        oy = (d - img.height) // 2
        canvas.paste(img, (ox, oy))
        return _pil_to_qpixmap(canvas)
    except Exception:
        canvas = Image.new("RGB", (d, d), (40, 40, 60))
        return _pil_to_qpixmap(canvas)


# ── Background thumbnail card ─────────────────────────────────────────────────
class BgThumbCard(QFrame):
    """100×100 selectable thumbnail — one per image in 'random images' folder."""

    selected_changed = Signal(object)   # emits self on click

    _SS_NORMAL   = (f"QFrame{{background:{CAR};border:2px solid {MUT};"
                    f"border-radius:4px;}}")
    _SS_SELECTED = (f"QFrame{{background:#0d2a4a;border:2px solid {ACC};"
                    f"border-radius:4px;}}")

    def __init__(self, img_path: str, parent=None):
        super().__init__(parent)
        self.img_path   = img_path
        self._selected  = False

        self.setFixedSize(THUMB_D + 8, THUMB_D + 8)
        self.setStyleSheet(self._SS_NORMAL)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setToolTip(os.path.basename(img_path))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(0)

        lbl = QLabel(self)
        lbl.setFixedSize(THUMB_D, THUMB_D)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setPixmap(_load_thumb(img_path))
        lay.addWidget(lbl)

    def set_selected(self, v: bool):
        self._selected = v
        self.setStyleSheet(self._SS_SELECTED if v else self._SS_NORMAL)

    def is_selected(self) -> bool:
        return self._selected

    def mousePressEvent(self, _):
        self.selected_changed.emit(self)


# ── Inference worker ──────────────────────────────────────────────────────────
class InferenceWorker(QObject):
    finished = Signal(object)   # PIL RGBA Image
    error    = Signal(str)
    status   = Signal(str)      # progress label updates

    def __init__(self, model_dir: str, pil_image: Image.Image):
        super().__init__()
        self._model_dir = model_dir
        self._pil       = pil_image

    def run(self):
        try:
            import sys
            import torch
            from torchvision import transforms

            torch.set_float32_matmul_precision("high")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Detect ToonOut-style: has a .pth weights file instead of
            # standard transformers model.safetensors / pytorch_model.bin.
            pth_files = list(Path(self._model_dir).glob("*.pth"))
            if pth_files:
                # ── Anime (ToonOut) ────────────────────────────────────────
                self.status.emit("Loading Anime model architecture…")
                from transformers import AutoModelForImageSegmentation
                model = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet", trust_remote_code=True)
                self.status.emit("Loading Anime weights…")
                state = torch.load(str(pth_files[0]),
                                   map_location="cpu", weights_only=True)
                # Strip DDP / torch.compile wrapper prefixes (module._orig_mod., etc.)
                for prefix in ("module._orig_mod.", "module.", "_orig_mod."):
                    if any(k.startswith(prefix) for k in state):
                        state = {k[len(prefix):]: v for k, v in state.items()}
                        break
                model.load_state_dict(state)
            else:
                # ── Realism (RMBG-2.0) ────────────────────────────────────
                # Bypass from_pretrained to avoid meta-tensor issues caused
                # by trunc_normal_ being called on uninitialised meta params
                # inside BiRefNet's __init__ (Swin-L backbone).
                # birefnet.py uses a relative import (from .BiRefNet_config …)
                # so both files must be loaded as a virtual package.
                import importlib.util, types
                self.status.emit("Loading Realism model…")
                _pkg = "_birefnet_pkg"
                _pkg_mod = types.ModuleType(_pkg)
                _pkg_mod.__path__ = [self._model_dir]
                _pkg_mod.__package__ = _pkg
                sys.modules[_pkg] = _pkg_mod
                try:
                    def _load(name, filename):
                        spec = importlib.util.spec_from_file_location(
                            f"{_pkg}.{name}",
                            os.path.join(self._model_dir, filename))
                        mod = importlib.util.module_from_spec(spec)
                        mod.__package__ = _pkg
                        sys.modules[f"{_pkg}.{name}"] = mod
                        spec.loader.exec_module(mod)
                        return mod

                    _load("BiRefNet_config", "BiRefNet_config.py")
                    _bm = _load("birefnet", "birefnet.py")

                    model = _bm.BiRefNet(bb_pretrained=False)
                    from safetensors.torch import load_file as _load_sf
                    sf = os.path.join(self._model_dir, "model.safetensors")
                    self.status.emit("Loading Realism weights…")
                    state = _load_sf(sf)
                    model.load_state_dict(state)
                finally:
                    for k in list(sys.modules):
                        if k.startswith(_pkg):
                            del sys.modules[k]

            self.status.emit("Preparing model…")
            model.float()   # ensure float32 — .pth weights may be saved as float16
            model.eval()
            model.to(device)

            self.status.emit("Running inference…")
            src  = self._pil.convert("RGB")
            W, H = src.size

            tf = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
            ])
            tensor = tf(src).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(tensor)

            self.status.emit("Compositing result…")
            raw         = preds[-1] if isinstance(preds, (list, tuple)) else preds
            mask_tensor = raw.sigmoid().squeeze().cpu()
            mask_pil    = transforms.ToPILImage()(mask_tensor).resize((W, H), Image.BILINEAR)

            result = src.convert("RGBA")
            result.putalpha(mask_pil)
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


# ── Image viewer (resizable, X/Y/Zoom sliders, checkerboard alpha) ────────────
class BgViewerCard(QFrame):

    def __init__(self, d: int = CARD_D, parent=None):
        super().__init__(parent)
        self.D = d
        self.setFixedSize(self.D + 8, self.D + 8)
        self.setStyleSheet(
            f"QFrame{{background:{CAR};border:3px solid {ACC};border-radius:6px;}}")

        self.offset_x  = 0.5
        self.offset_y  = 0.5
        self.card_zoom = 1.0
        self._pil      = None
        self._checker  = _make_checkerboard(self.D)
        self._dsx = self._dsy = 0
        self._dox = self._doy = 0.5

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        self._img_frame = QFrame(self)
        self._img_frame.setFixedSize(self.D, self.D)
        self._img_frame.setStyleSheet("background:transparent;border:none;")

        self._img_lbl = QLabel(self._img_frame)
        self._img_lbl.setFixedSize(self.D, self.D)
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        self._img_lbl.mousePressEvent = self._ds
        self._img_lbl.mouseMoveEvent  = self._dm

        # Zoom slider — top edge
        self._izs = QSlider(Qt.Orientation.Horizontal, self._img_frame)
        self._izs.setRange(-50, 50); self._izs.setValue(0)
        self._izs.setFixedWidth(self.D - 10); self._izs.setFixedHeight(12)
        self._izs.move(5, 2)
        self._izs.setStyleSheet(self._slider_ss())
        self._izs.valueChanged.connect(self._on_izoom)

        # Y slider — right edge, vertical
        self._ys = QSlider(Qt.Orientation.Vertical, self._img_frame)
        self._ys.setRange(0, 1000); self._ys.setValue(500)
        self._ys.setInvertedAppearance(True)
        self._ys.setFixedWidth(12); self._ys.setFixedHeight(self.D - 26)
        self._ys.move(self.D - 13, 18)
        self._ys.setStyleSheet(self._slider_ss())
        self._ys.valueChanged.connect(self._on_y)

        # X slider — bottom edge
        self._xs = QSlider(Qt.Orientation.Horizontal, self._img_frame)
        self._xs.setRange(0, 1000); self._xs.setValue(500)
        self._xs.setFixedWidth(self.D - 26); self._xs.setFixedHeight(12)
        self._xs.move(5, self.D - 13)
        self._xs.setStyleSheet(self._slider_ss())
        self._xs.valueChanged.connect(self._on_x)

        root.addWidget(self._img_frame)
        self._update_slider_states()
        self._render()

    def _slider_ss(self):
        return (
            f"QSlider::groove:horizontal{{background:#0d0d0d;height:4px;border-radius:2px;}}"
            f"QSlider::groove:vertical{{background:#0d0d0d;width:4px;border-radius:2px;}}"
            f"QSlider::handle:horizontal{{background:{ACC};width:10px;height:10px;"
            f"margin:-3px 0;border-radius:5px;}}"
            f"QSlider::handle:vertical{{background:{ACC};width:10px;height:10px;"
            f"margin:0 -3px;border-radius:5px;}}"
            f"QSlider::sub-page:horizontal{{background:{ACC};border-radius:2px;}}"
            f"QSlider::add-page:vertical{{background:{ACC};border-radius:2px;}}"
        )

    def _pan_ranges(self):
        if self._pil is None:
            return 0, 0
        D = self.D
        iw, ih = self._pil.size
        scale  = max(D / iw, D / ih) * self.card_zoom
        return max(0, round(iw * scale) - D), max(0, round(ih * scale) - D)

    def _update_slider_states(self):
        xt, yt = self._pan_ranges()
        self._xs.setEnabled(xt > 0)
        self._ys.setEnabled(yt > 0)

    def _render(self):
        D = self.D
        if self._pil is None:
            self._img_lbl.setPixmap(_pil_to_qpixmap(self._checker))
            return

        iw, ih = self._pil.size
        scale  = max(D / iw, D / ih) * self.card_zoom
        zw = max(1, round(iw * scale))
        zh = max(1, round(ih * scale))
        big = self._pil.resize((zw, zh), Image.LANCZOS)

        ox = round(self.offset_x * (zw - D)) if zw > D else -((D - zw) // 2)
        oy = round(self.offset_y * (zh - D)) if zh > D else -((D - zh) // 2)

        canvas = self._checker.copy().convert("RGBA")
        if big.mode == "RGBA":
            canvas.paste(big, (-ox, -oy), big)
        else:
            canvas.paste(big.convert("RGBA"), (-ox, -oy))
        self._img_lbl.setPixmap(_pil_to_qpixmap(canvas.convert("RGB")))

    def _on_izoom(self, v):
        self.card_zoom = max(0.3, 1.0 + float(v) / 10.0 * 0.4)
        self._update_slider_states(); self._render()

    def _ds(self, e):
        self._dsx = e.pos().x(); self._dsy = e.pos().y()
        self._dox = self.offset_x; self._doy = self.offset_y

    def _dm(self, e):
        xt, yt = self._pan_ranges()
        if xt > 0:
            self.offset_x = max(0.0, min(1.0,
                self._dox - (e.pos().x() - self._dsx) / max(1, xt)))
        if yt > 0:
            self.offset_y = max(0.0, min(1.0,
                self._doy - (e.pos().y() - self._dsy) / max(1, yt)))
        self._xs.blockSignals(True);  self._xs.setValue(int(self.offset_x * 1000)); self._xs.blockSignals(False)
        self._ys.blockSignals(True);  self._ys.setValue(int(self.offset_y * 1000)); self._ys.blockSignals(False)
        self._render()

    def _on_x(self, v): self.offset_x = v / 1000.0; self._render()
    def _on_y(self, v): self.offset_y = v / 1000.0; self._render()

    def resize_card(self, d: int):
        self.D = d
        self._checker = _make_checkerboard(d)
        self.setFixedSize(d + 8, d + 8)
        self._img_frame.setFixedSize(d, d)
        self._img_lbl.setFixedSize(d, d)
        self._izs.setFixedWidth(d - 10)
        self._ys.setFixedHeight(d - 26); self._ys.move(d - 13, 18)
        self._xs.setFixedWidth(d - 26);  self._xs.move(5, d - 13)
        self._update_slider_states(); self._render()

    def set_image(self, pil_image: Image.Image):
        self._pil = pil_image
        self._update_slider_states(); self._render()

    def clear(self):
        self._pil = None; self._render()


# ── Randomizer page ───────────────────────────────────────────────────────────
class RandomizerPage(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {PAN};")

        os.makedirs(BG_FOLDER, exist_ok=True)

        self._cfg            = load_json(CFG_FILE, DEFAULTS)
        self._src_pil        = None   # loaded input (RGBA)
        self._out_pil        = None   # inference result (RGBA, transparent bg)
        self._composited_pil = None   # _out_pil composited over selected bg (RGB)
        self._viewing        = "input"
        self._selected_bg    = None   # path of selected BgThumbCard, or None
        self._thumb_cards: list[BgThumbCard] = []
        self._thread         = None
        self._worker         = None
        self._folder_images: list[str] = []   # paths loaded from Browse Folder
        self._input_pills:   list[QPushButton] = []
        self._active_pill:   QPushButton | None = None
        self._src_path:      str = ""         # original input file path

        self._build_ui()
        self._load_bg_images()

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_toolbar())

        # Main content: [image list] + viewer + bg panel
        content = QWidget(self)
        content.setStyleSheet(f"background:{PAN};")
        hbox = QHBoxLayout(content)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)

        self._img_list_panel = self._build_img_list_panel()
        self._img_list_panel.setVisible(False)
        hbox.addWidget(self._img_list_panel)
        hbox.addWidget(self._build_viewer_area(), stretch=1)
        hbox.addWidget(self._build_bg_panel())

        root.addWidget(content, stretch=1)

    def _build_header(self) -> QFrame:
        header = QFrame(self)
        header.setFixedHeight(48)
        header.setStyleSheet(
            f"QFrame{{background:{CAR};border-bottom:1px solid {MUT};}}")
        row = QHBoxLayout(header)
        row.setContentsMargins(16, 0, 16, 0)
        row.setSpacing(8)

        title = QLabel("Randomizer", header)
        title.setStyleSheet(
            f"color:{ACC};font-family:{FONT};font-size:{FONT_LG}px;font-weight:bold;")
        row.addWidget(title)

        sub = QLabel("— Background Remover", header)
        sub.setStyleSheet(f"color:{MUT};font-family:{FONT};font-size:{FONT_MD}px;")
        row.addWidget(sub)
        row.addStretch()

        ver = QLabel(f"v{VERSION}", header)
        ver.setStyleSheet(f"color:{MUT};font-family:{FONT};font-size:{FONT_SM}px;")
        row.addWidget(ver)
        return header

    def _build_toolbar(self) -> QFrame:
        bar = QFrame(self)
        bar.setFixedHeight(54)
        bar.setStyleSheet(
            f"QFrame{{background:{CAR};border-bottom:1px solid {MUT};}}")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 0, 12, 0)
        row.setSpacing(8)

        browse_btn = self._btn("Browse ▾", ACC)
        browse_menu = QMenu(browse_btn)
        browse_menu.setStyleSheet(
            f"QMenu{{background:{CAR};color:{PRI};border:1px solid {MUT};"
            f"font-family:{FONT};font-size:{FONT_MD}px;}}"
            f"QMenu::item:selected{{background:{ACC};}}")
        act_image  = QAction("Browse Image",  browse_btn)
        act_folder = QAction("Browse Folder", browse_btn)
        act_image.triggered.connect(self._browse_input)
        act_folder.triggered.connect(self._browse_folder)
        browse_menu.addAction(act_image)
        browse_menu.addAction(act_folder)
        browse_btn.setMenu(browse_menu)
        row.addWidget(browse_btn)

        self._path_lbl = QLabel("No image loaded", bar)
        self._path_lbl.setStyleSheet(
            f"color:{MUT};font-family:{FONT};font-size:{FONT_SM}px;")
        self._path_lbl.setMaximumWidth(360)
        row.addWidget(self._path_lbl, stretch=1)

        row.addSpacing(8)
        row.addWidget(self._lbl("Mode:", bar))

        self._mode_combo = QComboBox(bar)
        self._mode_combo.addItems(list(MODELS.keys()))
        self._mode_combo.setCurrentText(self._cfg.get("mode", "Realism"))
        self._mode_combo.setFixedWidth(110)
        self._mode_combo.setToolTip(
            "Realism → BRIA RMBG-2.0\nAnime → ToonOut (BiRefNet fine-tune)")
        row.addWidget(self._mode_combo)

        row.addSpacing(8)

        self._remove_btn = self._btn("Remove BG", ACC)
        self._remove_btn.setEnabled(False)
        self._remove_btn.clicked.connect(self._run_removal)
        row.addWidget(self._remove_btn)

        self._toggle_btn = self._btn("View Original", MUT)
        self._toggle_btn.setVisible(False)
        self._toggle_btn.clicked.connect(self._toggle_view)
        row.addWidget(self._toggle_btn)

        self._save_btn = self._btn("Save PNG", GRN)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_output)
        row.addWidget(self._save_btn)

        self._rename_tags_cb = QCheckBox("Copy tag file", bar)
        self._rename_tags_cb.setToolTip(
            "If a .txt tag file exists next to the source image,\n"
            "copy it alongside the saved PNG with the same base name.")
        self._rename_tags_cb.setStyleSheet(
            f"color:{SEC};font-family:{FONT};font-size:{FONT_SM}px;")
        row.addWidget(self._rename_tags_cb)

        row.addSpacing(12)
        row.addWidget(self._lbl("Box:", bar))

        self._size_spin = QSpinBox(bar)
        self._size_spin.setRange(CARD_D_MIN, CARD_D_MAX)
        self._size_spin.setSingleStep(CARD_D_STEP)
        self._size_spin.setValue(int(self._cfg.get("card_size", CARD_D)))
        self._size_spin.setSuffix(" px")
        self._size_spin.setFixedWidth(90)
        self._size_spin.setStyleSheet(
            f"QSpinBox{{background:{CAR};color:{PRI};border:1px solid {MUT};"
            f"border-radius:4px;font-family:{FONT};font-size:{FONT_MD}px;padding:2px 6px;}}"
            f"QSpinBox::up-button,QSpinBox::down-button{{width:16px;background:{MUT};}}")
        self._size_spin.editingFinished.connect(self._on_size_changed)
        self._size_spin.valueChanged.connect(self._on_size_changed)
        row.addWidget(self._size_spin)

        row.addSpacing(12)

        self._status_lbl = QLabel("", bar)
        self._status_lbl.setStyleSheet(
            f"color:{SEC};font-family:{FONT};font-size:{FONT_SM}px;min-width:200px;")
        row.addWidget(self._status_lbl)

        return bar

    def _build_viewer_area(self) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{PAN};}}")

        init_d = int(self._cfg.get("card_size", CARD_D))
        self._viewer = BgViewerCard(d=init_d)
        scroll.setWidget(self._viewer)
        return scroll

    def _build_bg_panel(self) -> QFrame:
        """Right-side scrollable column of 100×100 background thumbnails."""
        panel = QFrame(self)
        panel.setFixedWidth(THUMB_D + 26)   # thumb + card padding + scrollbar
        panel.setStyleSheet(
            f"QFrame{{background:{CAR};border-left:1px solid {MUT};}}")

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # ── Header row ────────────────────────────────────────────────────
        hdr = QFrame(panel)
        hdr.setFixedHeight(36)
        hdr.setStyleSheet(
            f"QFrame{{background:{CAR};border-bottom:1px solid {MUT};}}")
        hrow = QHBoxLayout(hdr)
        hrow.setContentsMargins(6, 0, 4, 0)
        hrow.setSpacing(4)

        bg_lbl = QLabel("Backgrounds", hdr)
        bg_lbl.setStyleSheet(
            f"color:{ACC};font-family:{FONT};font-size:{FONT_SM}px;font-weight:bold;")
        hrow.addWidget(bg_lbl, stretch=1)

        self._bg_count_lbl = QLabel("0", hdr)
        self._bg_count_lbl.setStyleSheet(
            f"background:{MUT};color:{PRI};font-family:{FONT};font-size:{FONT_SM}px;"
            f"border-radius:8px;padding:1px 5px;")
        hrow.addWidget(self._bg_count_lbl)

        refresh_btn = QPushButton("↺", hdr)
        refresh_btn.setFixedSize(22, 22)
        refresh_btn.setToolTip("Refresh folder")
        refresh_btn.setStyleSheet(
            f"QPushButton{{background:{MUT};color:{PRI};border:none;"
            f"border-radius:4px;font-size:12px;font-weight:bold;}}"
            f"QPushButton:hover{{background:{ACC};}}")
        refresh_btn.clicked.connect(self._load_bg_images)
        hrow.addWidget(refresh_btn)

        vbox.addWidget(hdr)

        # ── Scroll area ───────────────────────────────────────────────────
        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{CAR};}}")

        self._bg_content = QWidget()
        self._bg_content.setStyleSheet(f"background:{CAR};")
        self._bg_layout = QVBoxLayout(self._bg_content)
        self._bg_layout.setContentsMargins(4, 4, 4, 4)
        self._bg_layout.setSpacing(4)
        self._bg_layout.addStretch()

        scroll.setWidget(self._bg_content)
        vbox.addWidget(scroll, stretch=1)

        # ── Empty hint ────────────────────────────────────────────────────
        self._bg_empty_lbl = QLabel(
            "Drop images into\n'random images'\nfolder & refresh", panel)
        self._bg_empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bg_empty_lbl.setStyleSheet(
            f"color:{MUT};font-family:{FONT};font-size:{FONT_SM}px;"
            f"background:transparent;padding:8px;")
        self._bg_empty_lbl.setWordWrap(True)
        self._bg_layout.insertWidget(0, self._bg_empty_lbl)

        return panel

    def _build_img_list_panel(self) -> QFrame:
        """Left panel — selectable pills for folder-browse mode."""
        PANEL_W = THUMB_D + 26   # same width as bg panel
        PILL_W  = THUMB_D + 8    # matches BgThumbCard width

        panel = QFrame(self)
        panel.setFixedWidth(PANEL_W)
        panel.setStyleSheet(
            f"QFrame{{background:{CAR};border-right:1px solid {MUT};}}")

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────
        hdr = QFrame(panel)
        hdr.setFixedHeight(36)
        hdr.setStyleSheet(
            f"QFrame{{background:{CAR};border-bottom:1px solid {MUT};}}")
        hrow = QHBoxLayout(hdr)
        hrow.setContentsMargins(6, 0, 4, 0)
        hrow.setSpacing(4)

        img_lbl = QLabel("Images", hdr)
        img_lbl.setStyleSheet(
            f"color:{ACC};font-family:{FONT};font-size:{FONT_SM}px;font-weight:bold;")
        hrow.addWidget(img_lbl, stretch=1)

        self._img_count_lbl = QLabel("0", hdr)
        self._img_count_lbl.setStyleSheet(
            f"background:{MUT};color:{PRI};font-family:{FONT};font-size:{FONT_SM}px;"
            f"border-radius:8px;padding:1px 5px;")
        hrow.addWidget(self._img_count_lbl)
        vbox.addWidget(hdr)

        # ── Scroll area ───────────────────────────────────────────────
        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{CAR};}}")

        self._img_list_content = QWidget()
        self._img_list_content.setStyleSheet(f"background:{CAR};")
        self._img_list_layout = QVBoxLayout(self._img_list_content)
        self._img_list_layout.setContentsMargins(4, 4, 4, 4)
        self._img_list_layout.setSpacing(3)
        self._img_list_layout.addStretch()

        scroll.setWidget(self._img_list_content)
        vbox.addWidget(scroll, stretch=1)

        self._pill_w = PILL_W
        return panel

    # =========================================================================
    # Widget helpers
    # =========================================================================

    def _btn(self, label: str, color: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setFixedHeight(36)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: {PRI};
                border: none;
                border-radius: 4px;
                font-family: {FONT};
                font-size: {FONT_MD}px;
                font-weight: bold;
                padding: 0 14px;
            }}
            QPushButton:hover    {{ background-color: #185FA5; }}
            QPushButton:disabled {{ background-color: {MUT}; color: {SEC}; }}
        """)
        return btn

    def _lbl(self, text: str, parent=None) -> QLabel:
        lbl = QLabel(text, parent)
        lbl.setStyleSheet(
            f"color:{SEC};font-family:{FONT};font-size:{FONT_MD}px;")
        return lbl

    # =========================================================================
    # Background image panel
    # =========================================================================

    def _load_bg_images(self):
        """Scan 'random images' folder and rebuild thumbnail column."""
        # Clear existing cards (keep stretch at end)
        for card in self._thumb_cards:
            self._bg_layout.removeWidget(card)
            card.deleteLater()
        self._thumb_cards.clear()
        self._selected_bg = None

        paths = sorted(
            p for p in Path(BG_FOLDER).iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )

        has_images = bool(paths)
        self._bg_empty_lbl.setVisible(not has_images)
        self._bg_count_lbl.setText(str(len(paths)))

        for p in paths:
            card = BgThumbCard(str(p), self._bg_content)
            card.selected_changed.connect(self._on_bg_selected)
            self._bg_layout.insertWidget(self._bg_layout.count() - 1, card)
            self._thumb_cards.append(card)

    def _on_bg_selected(self, card: BgThumbCard):
        """Toggle selection — clicking the active card deselects it."""
        if card.is_selected():
            card.set_selected(False)
            self._selected_bg = None
        else:
            for c in self._thumb_cards:
                c.set_selected(False)
            card.set_selected(True)
            self._selected_bg = card.img_path

        self._refresh_composite()

    # =========================================================================
    # Browse input
    # =========================================================================

    def _browse_input(self):
        last = self._cfg.get("last_input_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", last,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif)")
        if not path:
            return

        self._cfg["last_input_dir"] = os.path.dirname(path)
        save_json(CFG_FILE, self._cfg)

        try:
            self._src_pil  = Image.open(path).convert("RGBA")
            self._src_path = path
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self._out_pil        = None
        self._composited_pil = None
        self._viewing        = "input"

        fname = os.path.basename(path)
        self._path_lbl.setText(fname if len(fname) <= 50 else fname[:47] + "…")
        self._viewer.set_image(self._src_pil)
        self._status_lbl.setText(
            f"{self._src_pil.width} × {self._src_pil.height} px")

        self._remove_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._toggle_btn.setVisible(False)

        # Hide folder list — single image mode
        self._img_list_panel.setVisible(False)

    def _browse_folder(self):
        last = self._cfg.get("last_input_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", last)
        if not folder:
            return

        self._cfg["last_input_dir"] = folder
        save_json(CFG_FILE, self._cfg)

        paths = sorted(
            str(p) for p in Path(folder).iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
        if not paths:
            QMessageBox.warning(self, "No Images", "No supported images found in that folder.")
            return

        self._folder_images = paths
        self._load_img_list(paths)

        # Auto-select first image
        if self._input_pills:
            self._select_pill(self._input_pills[0], paths[0])

    def _load_img_list(self, paths: list[str]):
        """Rebuild the left pill list from a list of image paths."""
        # Clear existing pills
        for pill in self._input_pills:
            self._img_list_layout.removeWidget(pill)
            pill.deleteLater()
        self._input_pills.clear()
        self._active_pill = None

        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            pill = QPushButton(name, self._img_list_content)
            pill.setFixedWidth(self._pill_w)
            pill.setFixedHeight(26)
            pill.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            pill.setToolTip(os.path.basename(path))
            pill.setStyleSheet(self._pill_ss(False))
            pill.clicked.connect(lambda checked, p=path, b=pill: self._select_pill(b, p))
            self._img_list_layout.insertWidget(
                self._img_list_layout.count() - 1, pill)
            self._input_pills.append(pill)

        self._img_count_lbl.setText(str(len(paths)))
        self._img_list_panel.setVisible(True)

    def _pill_ss(self, active: bool) -> str:
        if active:
            return (f"QPushButton{{background:{ACC};color:{PRI};border:none;"
                    f"border-radius:4px;font-family:{FONT};font-size:{FONT_SM}px;"
                    f"text-align:left;padding:0 6px;}}"
                    f"QPushButton:hover{{background:{ACC};}}")
        return (f"QPushButton{{background:{MUT};color:{PRI};border:none;"
                f"border-radius:4px;font-family:{FONT};font-size:{FONT_SM}px;"
                f"text-align:left;padding:0 6px;}}"
                f"QPushButton:hover{{background:#2a4a6a;}}")

    def _select_pill(self, pill: QPushButton, path: str):
        """Select a pill and load its image into the viewer."""
        if self._active_pill:
            self._active_pill.setStyleSheet(self._pill_ss(False))
        self._active_pill = pill
        pill.setStyleSheet(self._pill_ss(True))

        try:
            self._src_pil  = Image.open(path).convert("RGBA")
            self._src_path = path
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self._out_pil        = None
        self._composited_pil = None
        self._viewing        = "input"

        fname = os.path.basename(path)
        self._path_lbl.setText(fname if len(fname) <= 50 else fname[:47] + "…")
        self._viewer.set_image(self._src_pil)
        self._status_lbl.setText(
            f"{self._src_pil.width} × {self._src_pil.height} px")

        self._remove_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._toggle_btn.setVisible(False)

    # =========================================================================
    # Model management
    # =========================================================================

    def _model_dir(self, mode: str) -> Path:
        return Path(_HERE) / "models" / mode

    def _model_ready(self, mode: str) -> bool:
        return (self._model_dir(mode) / MODELS[mode]["check"]).exists()

    def _ensure_model(self, mode: str) -> bool:
        if self._model_ready(mode):
            return True

        cfg = MODELS[mode]
        r = QMessageBox.question(
            self, "Download Model",
            f"The {mode} model is not downloaded yet.\n\n"
            f"  • {cfg['desc']}\n"
            f"  • Download size: {cfg['size']}\n\n"
            f"Download now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        return self._download_model(mode) if r == QMessageBox.StandardButton.Yes else False

    def _download_model(self, mode: str) -> bool:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            QMessageBox.critical(self, "Missing Package",
                                 "Install huggingface-hub:\n  pip install huggingface-hub")
            return False

        cfg  = MODELS[mode]
        dest = self._model_dir(mode)
        dest.mkdir(parents=True, exist_ok=True)

        dlg = QProgressDialog(
            f"Downloading {mode} model ({cfg['size']})…\nThis may take several minutes.",
            "Cancel", 0, 0, self)
        dlg.setWindowTitle("Model Download")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.show()
        QApplication.processEvents()

        failed = []

        def _do():
            try:
                snapshot_download(repo_id=cfg["repo"], local_dir=str(dest),
                                  ignore_patterns=["*.gguf"])
            except Exception as e:
                failed.append(str(e))

        t = threading.Thread(target=_do, daemon=True)
        t.start()

        import time
        while t.is_alive():
            if dlg.wasCanceled():
                break
            QApplication.processEvents()
            time.sleep(0.1)

        dlg.close()

        if failed:
            QMessageBox.warning(self, "Download Failed", failed[0])
            return False

        if not self._model_ready(mode):
            QMessageBox.warning(self, "Download Incomplete",
                                "Expected model files not found after download.")
            return False

        return True

    # =========================================================================
    # Inference
    # =========================================================================

    def _run_removal(self):
        if self._src_pil is None:
            return

        mode = self._mode_combo.currentText()
        self._cfg["mode"] = mode
        save_json(CFG_FILE, self._cfg)

        if not self._ensure_model(mode):
            return

        self._remove_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._toggle_btn.setVisible(False)
        self._status_lbl.setText("Processing…")
        QApplication.processEvents()

        self._progress_dlg = QProgressDialog("Initializing…", None, 0, 0, self)
        self._progress_dlg.setWindowTitle(f"Remove Background  ({mode})")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumWidth(340)
        self._progress_dlg.setMinimumDuration(0)
        self._progress_dlg.show()
        QApplication.processEvents()

        self._thread = QThread()
        self._worker = InferenceWorker(str(self._model_dir(mode)), self._src_pil)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.status.connect(self._progress_dlg.setLabelText)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_done(self, result: Image.Image):
        self._progress_dlg.close()
        self._out_pil = result
        self._viewing = "output"
        self._refresh_composite()
        self._remove_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        self._toggle_btn.setVisible(True)
        self._toggle_btn.setText("View Original")
        bg_note = f" + bg" if self._selected_bg else " (transparent)"
        self._status_lbl.setText(
            f"Done  ✓  —  {result.width} × {result.height} px{bg_note}")

    def _on_error(self, msg: str):
        self._progress_dlg.close()
        self._remove_btn.setEnabled(True)
        self._status_lbl.setText("Error")
        QMessageBox.critical(self, "Inference Error", msg)

    # =========================================================================
    # Composite / view helpers
    # =========================================================================

    def _refresh_composite(self):
        """
        Rebuild _composited_pil from _out_pil + selected bg.
        Updates viewer if currently showing the result.
        """
        if self._out_pil is None:
            return

        if self._selected_bg and os.path.exists(self._selected_bg):
            try:
                bg = Image.open(self._selected_bg).convert("RGB")
                bg = bg.resize(self._out_pil.size, Image.LANCZOS)
                comp = bg.convert("RGBA")
                comp.paste(self._out_pil, (0, 0), self._out_pil)
                self._composited_pil = comp.convert("RGB")
            except Exception:
                self._composited_pil = None
        else:
            self._composited_pil = None

        if self._viewing == "output":
            display = self._composited_pil if self._composited_pil else self._out_pil
            self._viewer.set_image(display)

    def _current_result(self) -> Image.Image | None:
        """Returns whichever result the user should see/save."""
        if self._out_pil is None:
            return None
        return self._composited_pil if self._composited_pil else self._out_pil

    def _on_size_changed(self):
        d = max(CARD_D_MIN, min(CARD_D_MAX,
                (self._size_spin.value() // CARD_D_STEP) * CARD_D_STEP))
        self._size_spin.blockSignals(True)
        self._size_spin.setValue(d)
        self._size_spin.blockSignals(False)
        self._cfg["card_size"] = d
        save_json(CFG_FILE, self._cfg)
        self._viewer.resize_card(d)

    def _toggle_view(self):
        if self._viewing == "output" and self._src_pil:
            self._viewing = "input"
            self._viewer.set_image(self._src_pil)
            self._toggle_btn.setText("View Result")
        elif self._viewing == "input":
            result = self._current_result()
            if result:
                self._viewing = "output"
                self._viewer.set_image(result)
                self._toggle_btn.setText("View Original")

    def _save_output(self):
        result = self._current_result()
        if result is None:
            return

        base      = os.path.splitext(os.path.basename(self._src_path))[0] if self._src_path else "output"
        default   = os.path.join(self._cfg.get("last_output_dir", ""), f"{base}-BG.png")
        path, _   = QFileDialog.getSaveFileName(self, "Save PNG", default, "PNG (*.png)")
        if not path:
            return

        self._cfg["last_output_dir"] = os.path.dirname(path)
        save_json(CFG_FILE, self._cfg)

        try:
            result.save(path, "PNG")
            self._status_lbl.setText(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return

        if self._rename_tags_cb.isChecked() and self._src_path:
            src_txt = os.path.splitext(self._src_path)[0] + ".txt"
            if os.path.exists(src_txt):
                import shutil
                dst_txt = os.path.splitext(path)[0] + ".txt"
                try:
                    shutil.copy2(src_txt, dst_txt)
                    self._status_lbl.setText(
                        f"Saved: {os.path.basename(path)}  +  {os.path.basename(dst_txt)}")
                except Exception as e:
                    QMessageBox.warning(self, "Tag Copy Error", str(e))
