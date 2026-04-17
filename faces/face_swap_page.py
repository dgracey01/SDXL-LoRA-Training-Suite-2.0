"""
faces/face_swap_page.py — Face Swap page for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.0)

Tabs (mirroring ReActor):  Face · Detection · Upscale · Tools · Settings
"""
from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from PySide6.QtCore    import Qt, QThread, QObject, Signal
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QScrollArea, QTabWidget,
    QFileDialog, QMessageBox, QProgressDialog, QApplication,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QMenu,
    QSlider, QSizePolicy,
)
from PySide6.QtGui import QCursor

from shared.theme import (
    BG, PAN, CAR, ACC, GRN, RED, MUT, PRI, SEC, AMB,
    FONT, FONT_SM, FONT_MD,
)
from shared.config import load_json, save_json
from randomizer.randomizer_page import BgViewerCard, _load_thumb

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
CFG_FILE     = os.path.join(_HERE, "faceswap_config.json")
FACES_DIR    = Path(_HERE) / "models"
SWAP_MODEL   = Path(_HERE) / "inswapper_128.onnx"
GFPGAN_MODEL = Path(_HERE) / "GFPGANv1.4.pth"
GFPGAN_URL   = ("https://github.com/TencentARC/GFPGAN/releases/download/"
                "v1.3.4/GFPGANv1.4.pth")

THUMB_D        = 100
IMAGE_EXTS     = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
FACE_MODEL_EXT = '.safetensors'
CARD_D         = 2000
CARD_D_MIN     = 400
CARD_D_MAX     = 3000
CARD_D_STEP    = 100

FACE_ORDERS    = ["Left → Right", "Right → Left", "Top → Bottom",
                  "Bottom → Top", "Large → Small", "Small → Large"]
FACE_ORDER_KEY = ["left-right", "right-left", "top-bottom",
                  "bottom-top", "large-small", "small-large"]
GENDER_OPTS    = ["Any", "Male only", "Female only"]
DEVICE_OPTS    = ["Auto", "CUDA", "DirectML", "CPU"]

DEFAULTS = {
    "last_input_dir":      "",
    "last_output_dir":     "",
    "faces_folder":        str(FACES_DIR),
    "card_size":           CARD_D,
    "batch_swap_out_folder": "",
    # Detection
    "det_thresh":       0.5,
    "det_maxnum":       10,
    "face_order":       "left-right",
    "source_face_idx":  0,
    "target_face_idx":  "0",
    "gender_source":    0,
    "gender_target":    0,
    # Upscale
    "enhance_face":     False,
    "gfpgan_weight":    0.5,
    "restore_first":    True,
    # Settings
    "device":           "Auto",
    # Tools
    "batch_src_folder": "",
    "batch_out_folder": str(FACES_DIR),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _providers(device: str) -> list[str]:
    if device == "CUDA":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "DirectML":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if device == "CPU":
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


def _sort_faces(faces: list, order: str) -> list:
    if order == "right-left":
        return sorted(faces, key=lambda f: -f.bbox[0])
    if order == "top-bottom":
        return sorted(faces, key=lambda f: f.bbox[1])
    if order == "bottom-top":
        return sorted(faces, key=lambda f: -f.bbox[1])
    if order == "large-small":
        return sorted(faces, key=lambda f: -(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    if order == "small-large":
        return sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return sorted(faces, key=lambda f: f.bbox[0])   # left-right (default)


def _filter_gender(faces: list, gender: int) -> list:
    if gender == 1:
        filtered = [f for f in faces if getattr(f, "gender", -1) == 1]
    elif gender == 2:
        filtered = [f for f in faces if getattr(f, "gender", -1) == 0]
    else:
        return faces
    return filtered or faces   # fall back to all if filter empties the list


def _read_face_meta(path: str) -> tuple[str, int]:
    try:
        from safetensors import safe_open
        with safe_open(path, framework="np") as f:
            gender = int(f.get_tensor("gender").flat[0])
            age    = int(f.get_tensor("age").flat[0])
        return ("M" if gender == 1 else "F"), age
    except Exception:
        return "?", 0


def save_reactor_face(face, path: str) -> None:
    import torch
    from safetensors.torch import save_file
    tensors = {}
    for key in ("bbox", "kps", "det_score", "landmark_3d_68", "pose",
                "landmark_2d_106", "embedding", "gender", "age"):
        val = face.get(key)
        if val is not None:
            tensors[key] = torch.tensor(np.array(val, dtype=np.float32))
    save_file(tensors, path)


def load_reactor_face(path: str):
    from safetensors import safe_open
    from insightface.app.common import Face
    face_data = {}
    with safe_open(path, framework="np") as f:
        for k in f.keys():
            face_data[k] = f.get_tensor(k)
    return Face(face_data)


# ── UI helpers ────────────────────────────────────────────────────────────────

def _tab_ss() -> str:
    return (f"QTabWidget::pane  {{ background:{PAN}; border:none; }}"
            f"QTabBar::tab      {{ background:{CAR}; color:{SEC}; padding:6px 18px;"
            f"                     margin-right:2px; font-family:{FONT};"
            f"                     font-size:{FONT_MD}px; }}"
            f"QTabBar::tab:selected        {{ background:{ACC}; color:{PRI}; }}"
            f"QTabBar::tab:hover:!selected {{ background:{MUT}; color:{PRI}; }}")


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
    lbl.setMinimumWidth(160)
    return lbl


def _combo(options: list[str], current: str, w: int = 180) -> QComboBox:
    cb = QComboBox()
    cb.addItems(options)
    if current in options:
        cb.setCurrentText(current)
    cb.setFixedWidth(w)
    cb.setStyleSheet(
        f"QComboBox {{ background:{CAR}; color:{PRI}; border:1px solid {MUT};"
        f"border-radius:4px; padding:3px 8px; font-family:{FONT};"
        f"font-size:{FONT_SM}px; }}"
        f"QComboBox::drop-down {{ border:none; width:20px; }}"
        f"QComboBox QAbstractItemView {{ background:{CAR}; color:{PRI};"
        f"selection-background-color:{ACC}; }}")
    return cb


def _spinbox(val: int, lo: int, hi: int, w: int = 80) -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(val)
    sb.setFixedWidth(w)
    sb.setStyleSheet(
        f"QSpinBox {{ background:{CAR}; color:{PRI}; border:1px solid {MUT};"
        f"border-radius:4px; padding:3px 4px; font-family:{FONT};"
        f"font-size:{FONT_SM}px; }}"
        f"QSpinBox::up-button, QSpinBox::down-button {{ width:14px; }}")
    return sb


def _dspinbox(val: float, lo: float, hi: float, step: float = 0.05,
              w: int = 80) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setDecimals(2)
    sb.setValue(val)
    sb.setFixedWidth(w)
    sb.setStyleSheet(
        f"QDoubleSpinBox {{ background:{CAR}; color:{PRI}; border:1px solid {MUT};"
        f"border-radius:4px; padding:3px 4px; font-family:{FONT};"
        f"font-size:{FONT_SM}px; }}"
        f"QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width:14px; }}")
    return sb


def _checkbox(text: str, checked: bool) -> QCheckBox:
    cb = QCheckBox(text)
    cb.setChecked(checked)
    cb.setStyleSheet(
        f"QCheckBox {{ color:{PRI}; font-family:{FONT}; font-size:{FONT_SM}px;"
        f"spacing:6px; background:transparent; }}"
        f"QCheckBox::indicator {{ width:15px; height:15px; border:1px solid {MUT};"
        f"border-radius:3px; background:{CAR}; }}"
        f"QCheckBox::indicator:checked {{ background:{ACC}; border:1px solid {ACC}; }}")
    return cb


def _settings_page() -> tuple[QWidget, QGridLayout]:
    page = QWidget()
    page.setStyleSheet(f"background:{PAN};")
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setStyleSheet(
        f"QScrollArea {{ background:{PAN}; border:none; }}"
        f"QScrollBar:vertical {{ background:{CAR}; width:6px; }}"
        f"QScrollBar::handle:vertical {{ background:{MUT}; border-radius:3px; }}")
    inner = QWidget()
    inner.setStyleSheet(f"background:{PAN};")
    grid = QGridLayout(inner)
    grid.setContentsMargins(24, 16, 24, 16)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(8)
    grid.setColumnStretch(2, 1)
    sa.setWidget(inner)
    vl = QVBoxLayout(page)
    vl.setContentsMargins(0, 0, 0, 0)
    vl.addWidget(sa)
    return page, grid


# ── Face image thumbnail card ─────────────────────────────────────────────────
class FaceThumbCard(QFrame):
    selected_changed = Signal(object)
    _SS_N = f"QFrame {{ background:{CAR}; border:2px solid {MUT}; border-radius:4px; }}"
    _SS_S = f"QFrame {{ background:#0d2a4a; border:2px solid {ACC}; border-radius:4px; }}"

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path     = path
        self.is_model = False
        self._sel     = False
        self.setFixedSize(THUMB_D + 8, THUMB_D + 8)
        self.setStyleSheet(self._SS_N)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lbl = QLabel(self)
        lbl.setFixedSize(THUMB_D, THUMB_D)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        px = _load_thumb(path, THUMB_D)
        if px:
            lbl.setPixmap(px)
        lay.addWidget(lbl)

    def set_selected(self, v: bool):
        self._sel = v
        self.setStyleSheet(self._SS_S if v else self._SS_N)

    def is_selected(self) -> bool:
        return self._sel

    def mousePressEvent(self, _):
        self.selected_changed.emit(self)


# ── ReActor face model card ───────────────────────────────────────────────────
class FaceModelCard(QFrame):
    selected_changed = Signal(object)
    _SS_N = f"QFrame {{ background:{CAR}; border:2px solid {MUT}; border-radius:4px; }}"
    _SS_S = f"QFrame {{ background:#1a2a0a; border:2px solid {GRN}; border-radius:4px; }}"

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path     = path
        self.is_model = True
        self._sel     = False
        self.setFixedSize(THUMB_D + 8, THUMB_D + 8)
        self.setStyleSheet(self._SS_N)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(3)

        badge = QLabel("FM", self)
        badge.setFixedSize(32, 32)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"background:{GRN}; color:{PRI}; border-radius:4px;"
            f"font-family:{FONT}; font-size:{FONT_MD}px; font-weight:bold;")
        lay.addWidget(badge, alignment=Qt.AlignmentFlag.AlignHCenter)

        name = Path(path).stem
        name_lbl = QLabel(name if len(name) <= 12 else name[:11] + "…", self)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet(
            f"color:{PRI}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        lay.addWidget(name_lbl)

        gender, age = _read_face_meta(path)
        meta_lbl = QLabel(f"{gender} · {age}y" if age > 0 else gender, self)
        meta_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meta_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        lay.addWidget(meta_lbl)
        lay.addStretch(1)

    def set_selected(self, v: bool):
        self._sel = v
        self.setStyleSheet(self._SS_S if v else self._SS_N)

    def is_selected(self) -> bool:
        return self._sel

    def mousePressEvent(self, _):
        self.selected_changed.emit(self)


# ── Face swap worker ──────────────────────────────────────────────────────────
class FaceSwapWorker(QObject):
    finished   = Signal(object)
    error      = Signal(str)
    status     = Signal(str)

    def __init__(self, target_img: Image.Image,
                 source_inputs: list,   # list of (pil_img_or_None, model_path_or_"")
                 model_path: str,
                 blend: bool = False,
                 gfpgan_restorer=None,
                 gfpgan_weight: float = 0.5,
                 det_thresh: float = 0.5,
                 det_maxnum: int = 10,
                 face_order: str = "left-right",
                 source_face_idx: int = 0,
                 target_face_idx: str = "0",
                 gender_source: int = 0,
                 gender_target: int = 0,
                 device: str = "Auto"):
        super().__init__()
        self._target        = target_img
        self._sources       = source_inputs
        self._model_path    = model_path
        self._blend         = blend
        self._gfpgan        = gfpgan_restorer
        self._gfpgan_weight = gfpgan_weight
        self._det_thresh    = det_thresh
        self._det_maxnum    = det_maxnum
        self._face_order    = face_order
        self._src_face_idx  = source_face_idx
        self._tgt_face_idx  = [int(x.strip()) for x in target_face_idx.split(",")
                               if x.strip().isdigit()]
        self._gender_src    = gender_source
        self._gender_tgt    = gender_target
        self._device        = device
        self.source_face    = None   # set during run(), read by main thread after done

    def run(self):
        try:
            import cv2
            import insightface
            from insightface.app import FaceAnalysis

            prov = _providers(self._device)

            _buffalo = Path.home() / ".insightface" / "models" / "buffalo_l"
            if not _buffalo.exists():
                self.status.emit("Downloading buffalo_l (~200 MB, first run)…")
            else:
                self.status.emit("Loading face analysis model…")

            app = FaceAnalysis(name="buffalo_l", providers=prov)
            app.prepare(ctx_id=0, det_size=(640, 640),
                        det_thresh=self._det_thresh)

            swapper = insightface.model_zoo.get_model(self._model_path,
                                                      providers=prov)

            def to_cv(img: Image.Image):
                return cv2.cvtColor(np.array(img.convert("RGB")),
                                    cv2.COLOR_RGB2BGR)

            # ── Source face(s) ────────────────────────────────────────────────
            raw_faces = []
            for i, (src_pil, model_path) in enumerate(self._sources):
                self.status.emit(f"Loading source {i+1}/{len(self._sources)}…")
                if model_path:
                    raw_faces.append(load_reactor_face(model_path))
                else:
                    src_cv    = to_cv(src_pil)
                    src_faces = app.get(src_cv)
                    src_faces = [f for f in src_faces
                                 if f.det_score >= self._det_thresh]
                    src_faces = _filter_gender(src_faces, self._gender_src)
                    src_faces = _sort_faces(src_faces, self._face_order)
                    if not src_faces:
                        self.error.emit(
                            f"No face detected in source image #{i+1}.\n"
                            "Try lowering the detection threshold in Detection tab.")
                        return
                    idx = min(self._src_face_idx, len(src_faces) - 1)
                    raw_faces.append(src_faces[idx])

            if self._blend and len(raw_faces) > 1:
                # Mean-blend embeddings into a synthetic Face
                self.status.emit(
                    f"Blending {len(raw_faces)} face embeddings (mean)…")
                from insightface.app.common import Face
                embeddings = np.array([f.embedding for f in raw_faces],
                                      dtype=np.float32)
                blended_emb = embeddings.mean(axis=0)
                norm = np.linalg.norm(blended_emb)
                if norm > 0:
                    blended_emb /= norm
                # Use first face's metadata (kps, bbox) as the container
                source_face = Face(dict(raw_faces[0]))
                source_face.embedding = blended_emb
            else:
                source_face = raw_faces[0]
            self.source_face = source_face

            # ── Target faces ──────────────────────────────────────────────────
            tgt_cv    = to_cv(self._target)
            self.status.emit("Detecting target faces…")
            tgt_faces = app.get(tgt_cv)
            tgt_faces = [f for f in tgt_faces
                         if f.det_score >= self._det_thresh]
            tgt_faces = _filter_gender(tgt_faces, self._gender_tgt)
            tgt_faces = _sort_faces(tgt_faces, self._face_order)
            if self._det_maxnum > 0:
                tgt_faces = tgt_faces[:self._det_maxnum]

            if not tgt_faces:
                self.error.emit(
                    "No face detected in the target image.\n"
                    "Try lowering the detection threshold in Detection tab.")
                return

            scores = ", ".join(f"{f.det_score:.2f}" for f in tgt_faces)
            low = any(f.det_score < 0.3 for f in tgt_faces)
            self.status.emit(
                f"Target: {len(tgt_faces)} face(s) — scores: {scores}"
                + ("  ⚠ low confidence" if low else ""))

            # ── Swap ──────────────────────────────────────────────────────────
            self.status.emit(f"Swapping {len(tgt_faces)} face(s)…")
            result = tgt_cv.copy()
            swapped = 0
            for i, face in enumerate(tgt_faces):
                if not self._tgt_face_idx or i in self._tgt_face_idx:
                    prev = result.copy()
                    result = swapper.get(result, face, source_face,
                                        paste_back=True)
                    if np.array_equal(result, prev):
                        self.status.emit(
                            f"⚠ Face #{i} swap produced no change "
                            f"(score {face.det_score:.2f} — try higher "
                            f"detection threshold for a cleaner result)")
                    else:
                        swapped += 1

            # ── GFPGAN enhancement ────────────────────────────────────────────
            if self._gfpgan is not None:
                self.status.emit("Enhancing face (GFPGAN)…")
                try:
                    _, _, enhanced = self._gfpgan.enhance(
                        result, has_aligned=False,
                        only_center_face=False, paste_back=True,
                        weight=self._gfpgan_weight)
                    result = enhanced
                except Exception as e:
                    self.status.emit(f"GFPGAN failed ({e}), using swap only")

            result_pil = Image.fromarray(
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            self.finished.emit(result_pil)

        except ImportError as e:
            self.error.emit(f"Missing package: {e}\n\npip install insightface")
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ── Face export worker ────────────────────────────────────────────────────────
class FaceExportWorker(QObject):
    finished = Signal(str)
    error    = Signal(str)
    status   = Signal(str)

    def __init__(self, img: Image.Image, save_path: str,
                 det_thresh: float = 0.5, device: str = "Auto"):
        super().__init__()
        self._img        = img
        self._save_path  = save_path
        self._det_thresh = det_thresh
        self._device     = device

    def run(self):
        try:
            import cv2
            from insightface.app import FaceAnalysis
            self.status.emit("Detecting face…")
            prov = _providers(self._device)
            app  = FaceAnalysis(name="buffalo_l", providers=prov)
            app.prepare(ctx_id=0, det_size=(640, 640))
            img_cv = cv2.cvtColor(np.array(self._img.convert("RGB")),
                                  cv2.COLOR_RGB2BGR)
            faces = [f for f in app.get(img_cv)
                     if f.det_score >= self._det_thresh]
            if not faces:
                self.error.emit(
                    "No face detected.\n"
                    f"Detection threshold: {self._det_thresh:.2f} — "
                    "try lowering it in the Detection tab.")
                return
            self.status.emit("Saving face model…")
            save_reactor_face(faces[0], self._save_path)
            from pathlib import Path as _Path
            if not _Path(self._save_path).is_file():
                self.error.emit(
                    f"Save appeared to succeed but file not found:\n{self._save_path}")
                return
            self.finished.emit(self._save_path)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ── Batch export worker ───────────────────────────────────────────────────────
class BatchExportWorker(QObject):
    progress = Signal(int, int, str)   # current, total, filename
    finished = Signal(int, int)        # saved, skipped
    error    = Signal(str)

    def __init__(self, src_folder: str, out_folder: str, device: str = "Auto"):
        super().__init__()
        self._src    = src_folder
        self._out    = out_folder
        self._device = device
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            import cv2
            from insightface.app import FaceAnalysis
            prov = _providers(self._device)
            app  = FaceAnalysis(name="buffalo_l", providers=prov)
            app.prepare(ctx_id=0, det_size=(640, 640))

            images = sorted([
                p for p in Path(self._src).iterdir()
                if p.suffix.lower() in IMAGE_EXTS
            ])
            total, saved, skipped = len(images), 0, 0
            Path(self._out).mkdir(parents=True, exist_ok=True)

            for i, img_path in enumerate(images):
                if self._cancel:
                    break
                self.progress.emit(i + 1, total, img_path.name)
                try:
                    img    = Image.open(img_path).convert("RGB")
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    faces  = app.get(img_cv)
                    if faces:
                        out = Path(self._out) / (img_path.stem + ".safetensors")
                        save_reactor_face(faces[0], str(out))
                        saved += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1

            self.finished.emit(saved, skipped)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ── Batch swap worker ─────────────────────────────────────────────────────────
class BatchSwapWorker(QObject):
    progress = Signal(int, int, str)   # current, total, filename
    finished = Signal(int, int)        # saved, skipped
    error    = Signal(str)
    status   = Signal(str)

    def __init__(self, image_paths: list[str],
                 source_inputs: list,
                 model_path: str,
                 out_folder: str,
                 blend: bool = False,
                 gfpgan_restorer=None,
                 gfpgan_weight: float = 0.5,
                 det_thresh: float = 0.5,
                 det_maxnum: int = 10,
                 face_order: str = "left-right",
                 source_face_idx: int = 0,
                 target_face_idx: str = "0",
                 gender_source: int = 0,
                 gender_target: int = 0,
                 device: str = "Auto",
                 copy_tags: bool = False):
        super().__init__()
        self._paths         = image_paths
        self._sources       = source_inputs
        self._model_path    = model_path
        self._out_folder    = out_folder
        self._blend         = blend
        self._gfpgan        = gfpgan_restorer
        self._gfpgan_weight = gfpgan_weight
        self._det_thresh    = det_thresh
        self._det_maxnum    = det_maxnum
        self._face_order    = face_order
        self._src_face_idx  = source_face_idx
        self._tgt_face_idx  = [int(x.strip()) for x in target_face_idx.split(",")
                               if x.strip().isdigit()]
        self._gender_src    = gender_source
        self._gender_tgt    = gender_target
        self._device        = device
        self._copy_tags     = copy_tags
        self._cancel        = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            import cv2
            import insightface
            from insightface.app import FaceAnalysis

            prov = _providers(self._device)

            _buffalo = Path.home() / ".insightface" / "models" / "buffalo_l"
            self.status.emit(
                "Downloading buffalo_l (~200 MB, first run)…"
                if not _buffalo.exists() else "Loading face analysis model…")

            app = FaceAnalysis(name="buffalo_l", providers=prov)
            app.prepare(ctx_id=0, det_size=(640, 640),
                        det_thresh=self._det_thresh)
            swapper = insightface.model_zoo.get_model(self._model_path,
                                                      providers=prov)

            def to_cv(img: Image.Image):
                return cv2.cvtColor(np.array(img.convert("RGB")),
                                    cv2.COLOR_RGB2BGR)

            # ── Build source face(s) once ──────────────────────────────────
            raw_faces = []
            for i, (src_pil, mdl_path) in enumerate(self._sources):
                self.status.emit(
                    f"Loading source face {i+1}/{len(self._sources)}…")
                if mdl_path:
                    raw_faces.append(load_reactor_face(mdl_path))
                else:
                    src_cv    = to_cv(src_pil)
                    src_faces = app.get(src_cv)
                    src_faces = [f for f in src_faces
                                 if f.det_score >= self._det_thresh]
                    src_faces = _filter_gender(src_faces, self._gender_src)
                    src_faces = _sort_faces(src_faces, self._face_order)
                    if not src_faces:
                        self.error.emit(
                            f"No face detected in source image #{i+1}.\n"
                            "Try lowering the detection threshold.")
                        return
                    raw_faces.append(
                        src_faces[min(self._src_face_idx,
                                      len(src_faces) - 1)])

            if self._blend and len(raw_faces) > 1:
                from insightface.app.common import Face
                embeddings = np.array([f.embedding for f in raw_faces],
                                      dtype=np.float32)
                blended = embeddings.mean(axis=0)
                norm = np.linalg.norm(blended)
                if norm > 0:
                    blended /= norm
                source_face = Face(dict(raw_faces[0]))
                source_face.embedding = blended
            else:
                source_face = raw_faces[0]

            # ── Process each image ─────────────────────────────────────────
            Path(self._out_folder).mkdir(parents=True, exist_ok=True)
            total, saved, skipped = len(self._paths), 0, 0

            for i, img_path in enumerate(self._paths):
                if self._cancel:
                    break
                name = Path(img_path).name
                self.progress.emit(i + 1, total, name)
                try:
                    img      = Image.open(img_path).convert("RGB")
                    tgt_cv   = to_cv(img)
                    tgt_faces = app.get(tgt_cv)
                    tgt_faces = [f for f in tgt_faces
                                 if f.det_score >= self._det_thresh]
                    tgt_faces = _filter_gender(tgt_faces, self._gender_tgt)
                    tgt_faces = _sort_faces(tgt_faces, self._face_order)
                    if self._det_maxnum > 0:
                        tgt_faces = tgt_faces[:self._det_maxnum]
                    if not tgt_faces:
                        skipped += 1
                        continue

                    result = tgt_cv.copy()
                    for j, face in enumerate(tgt_faces):
                        if not self._tgt_face_idx or j in self._tgt_face_idx:
                            result = swapper.get(result, face, source_face,
                                                 paste_back=True)

                    if self._gfpgan is not None:
                        try:
                            _, _, enhanced = self._gfpgan.enhance(
                                result, has_aligned=False,
                                only_center_face=False, paste_back=True,
                                weight=self._gfpgan_weight)
                            result = enhanced
                        except Exception:
                            pass

                    src_p    = Path(img_path)
                    out_name = src_p.stem + "-FS" + src_p.suffix
                    out_path = Path(self._out_folder) / out_name
                    Image.fromarray(
                        cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    ).save(str(out_path))

                    if self._copy_tags:
                        import shutil
                        src_txt = src_p.with_suffix(".txt")
                        if src_txt.exists():
                            dst_txt = Path(self._out_folder) / (src_p.stem + "-FS.txt")
                            shutil.copy2(str(src_txt), str(dst_txt))

                    saved += 1
                except Exception:
                    skipped += 1

            self.finished.emit(saved, skipped)

        except ImportError as e:
            self.error.emit(f"Missing package: {e}\n\npip install insightface")
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# Face Swap Page
# ══════════════════════════════════════════════════════════════════════════════
class FaceSwapPage(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg         = load_json(CFG_FILE, DEFAULTS)
        self._faces_dir   = Path(self._cfg.get("faces_folder", str(FACES_DIR)))

        self._src_pil:          Image.Image | None = None
        self._src_path:         str  = ""
        self._out_pil:          Image.Image | None = None
        self._last_source_face: object | None = None   # face used in last swap
        self._viewing:       str  = "input"
        self._face_cards:    list = []
        self._sel_faces:     list = []   # list of (path, is_model) tuples
        self._thread:        QThread | None = None
        self._worker:        FaceSwapWorker | None = None
        self._exp_thread:    QThread | None = None
        self._exp_worker:    FaceExportWorker | None = None
        self._batch_thread:  QThread | None = None
        self._batch_worker:  BatchExportWorker | None = None
        self._bswap_thread:  QThread | None = None
        self._bswap_worker:  BatchSwapWorker | None = None
        self._folder_images: list[str] = []
        self._input_pills:   list[QPushButton] = []
        self._active_pill:   QPushButton | None = None
        self._prog_dlg:      QProgressDialog | None = None

        self._build_ui()
        self._load_face_images()

    # ── Top-level layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_toolbar())

        tabs = QTabWidget()
        tabs.setStyleSheet(_tab_ss())
        tabs.addTab(self._build_main_tab(),      "Face")
        tabs.addTab(self._build_detection_tab(), "Detection")
        tabs.addTab(self._build_upscale_tab(),   "Upscale")
        tabs.addTab(self._build_tools_tab(),     "Tools")
        tabs.addTab(self._build_settings_tab(),  "Settings")
        root.addWidget(tabs, stretch=1)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(54)
        bar.setStyleSheet(
            f"QFrame {{ background:{CAR}; border-bottom:1px solid {MUT}; }}")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 0, 12, 0)
        row.setSpacing(8)

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
        menu.addAction("Browse Image",  self._browse_input)
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
                f"QPushButton:hover:!disabled {{ filter:brightness(1.15); }}")
            b.clicked.connect(slot)
            return b

        self._blend_chk = QCheckBox("Blend", bar)
        self._blend_chk.setToolTip(
            "Blend multiple selected source faces by averaging their embeddings (mean).\n"
            "Select two or more face images/models in the panel on the right.")
        self._blend_chk.setStyleSheet(
            f"QCheckBox {{ color:{PRI}; font-family:{FONT}; font-size:{FONT_MD}px; }}"
            f"QCheckBox::indicator {{ width:14px; height:14px; }}")
        self._blend_chk.stateChanged.connect(self._on_blend_toggled)
        row.addWidget(self._blend_chk)
        row.addSpacing(6)

        self._apply_btn      = _btn("Apply Swap",       ACC, self._run_swap,           False)
        self._bswap_btn      = _btn("Batch Swap All",   AMB, self._run_batch_swap,     False)
        self._bswap_btn.setVisible(False)

        self._copy_tags_chk = QCheckBox("Copy tags", bar)
        self._copy_tags_chk.setToolTip(
            "If a .txt tag file exists next to each source image,\n"
            "copy it to the output folder renamed with the -FS suffix.")
        self._copy_tags_chk.setStyleSheet(
            f"QCheckBox {{ color:{PRI}; font-family:{FONT}; font-size:{FONT_MD}px; }}"
            f"QCheckBox::indicator {{ width:14px; height:14px; }}")
        self._copy_tags_chk.setVisible(False)

        self._export_btn     = _btn("Export Face Model", GRN, self._export_face_model, False)
        self._toggle_btn     = _btn("View Original",     MUT, self._toggle_view)
        self._toggle_btn.setVisible(False)
        self._save_btn       = _btn("Save PNG",          ACC, self._save_output,        False)

        for w in (self._apply_btn, self._bswap_btn, self._copy_tags_chk,
                  self._export_btn, self._toggle_btn, self._save_btn):
            row.addWidget(w)

        row.addSpacing(12)
        row.addWidget(QLabel("Box:", bar))

        self._size_spin = QSpinBox(bar)
        self._size_spin.setRange(CARD_D_MIN, CARD_D_MAX)
        self._size_spin.setSingleStep(CARD_D_STEP)
        self._size_spin.setValue(self._cfg.get("card_size", CARD_D))
        self._size_spin.setFixedWidth(72)
        self._size_spin.setFixedHeight(30)
        self._size_spin.setStyleSheet(
            f"QSpinBox {{ background:{CAR}; color:{PRI}; border:1px solid {MUT};"
            f"border-radius:4px; padding:2px 4px; font-family:{FONT};"
            f"font-size:{FONT_SM}px; }}"
            f"QSpinBox::up-button, QSpinBox::down-button {{ width:14px; }}")
        self._size_spin.valueChanged.connect(self._on_size_changed)
        row.addWidget(self._size_spin)

        self._status_lbl = QLabel("", bar)
        self._status_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;")
        self._status_lbl.setMinimumWidth(200)
        row.addWidget(self._status_lbl)

        return bar

    # ── Tab: Main ─────────────────────────────────────────────────────────────

    def _build_main_tab(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet(f"background:{BG};")
        body = QHBoxLayout(page)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._img_list_panel = self._build_img_list_panel()
        self._img_list_panel.setMaximumWidth(0)
        self._img_list_panel.setVisible(False)
        body.addWidget(self._img_list_panel)
        body.addWidget(self._build_viewer_area(), stretch=1)
        body.addWidget(self._build_face_panel())
        return page

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

    def _build_face_panel(self) -> QFrame:
        panel = QFrame()
        panel.setFixedWidth(THUMB_D + 40)
        panel.setStyleSheet(
            f"QFrame {{ background:{CAR}; border-left:1px solid {MUT}; }}")
        vl = QVBoxLayout(panel)
        vl.setContentsMargins(4, 4, 4, 4)
        vl.setSpacing(4)

        hdr = QFrame(panel)
        hdr.setFixedHeight(36)
        hdr.setStyleSheet(f"QFrame {{ background:{CAR}; border:none; }}")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(4, 0, 4, 0)
        fb = QPushButton("Faces ▾", hdr)
        fb.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{ACC}; border:none;"
            f"font-family:{FONT}; font-size:{FONT_SM}px; font-weight:bold;"
            f"text-align:left; }}"
            f"QPushButton:hover {{ color:{PRI}; }}")
        fb.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        fb.setToolTip("Click to choose a faces folder")
        fb.clicked.connect(self._browse_faces_folder)
        fb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        hl.addWidget(fb, stretch=1)
        self._face_count = QLabel("0", hdr)
        self._face_count.setStyleSheet(
            f"background:{MUT}; color:{PRI}; font-family:{FONT};"
            f"font-size:{FONT_SM}px; border-radius:8px; padding:1px 5px;")
        hl.addWidget(self._face_count)
        rb = QPushButton("↺", hdr)
        rb.setFixedSize(22, 22)
        rb.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{MUT}; border:none; font-size:14px; }}"
            f"QPushButton:hover {{ color:{ACC}; }}")
        rb.clicked.connect(self._load_face_images)
        hl.addWidget(rb)
        vl.addWidget(hdr)

        self._desel_all_btn = QPushButton("Deselect All", panel)
        self._desel_all_btn.setFixedHeight(22)
        self._desel_all_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{MUT}; border:none;"
            f"font-family:{FONT}; font-size:{FONT_SM}px; }}"
            f"QPushButton:hover {{ color:{RED}; }}")
        self._desel_all_btn.clicked.connect(self._deselect_all_faces)
        vl.addWidget(self._desel_all_btn)

        sa = QScrollArea(panel)
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setStyleSheet(
            f"QScrollArea {{ background:{CAR}; border:none; }}"
            f"QScrollBar:vertical {{ background:{CAR}; width:6px; }}"
            f"QScrollBar::handle:vertical {{ background:{MUT}; border-radius:3px; }}")
        self._face_content = QWidget()
        self._face_content.setStyleSheet(f"background:{CAR};")
        self._face_vl = QVBoxLayout(self._face_content)
        self._face_vl.setContentsMargins(2, 2, 2, 2)
        self._face_vl.setSpacing(4)
        self._face_vl.addStretch(1)
        sa.setWidget(self._face_content)
        vl.addWidget(sa, stretch=1)

        self._face_hint = QLabel("Click 'Faces'\nto choose a folder", panel)
        self._face_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._face_hint.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        self._face_vl.insertWidget(0, self._face_hint)
        return panel

    def _build_img_list_panel(self) -> QFrame:
        panel = QFrame()
        panel.setFixedWidth(THUMB_D + 16)
        panel.setStyleSheet(
            f"QFrame {{ background:{CAR}; border-right:1px solid {MUT}; }}")
        vl = QVBoxLayout(panel)
        vl.setContentsMargins(4, 4, 4, 4)
        vl.setSpacing(4)
        hdr = QFrame(panel)
        hdr.setFixedHeight(36)
        hdr.setStyleSheet(f"QFrame {{ background:{CAR}; border:none; }}")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(4, 0, 4, 0)
        lbl = QLabel("Images", hdr)
        lbl.setStyleSheet(
            f"color:{ACC}; font-family:{FONT}; font-size:{FONT_SM}px; font-weight:bold;")
        hl.addWidget(lbl)
        hl.addStretch(1)
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

    # ── Tab: Detection ────────────────────────────────────────────────────────

    def _build_detection_tab(self) -> QWidget:
        page, g = _settings_page()
        r = 0

        g.addWidget(_section("Face Detection"), r, 0, 1, 3); r += 1

        g.addWidget(_row_label("Detection threshold"), r, 0)
        self._det_thresh = _dspinbox(self._cfg.get("det_thresh", 0.5),
                                     0.01, 0.99, 0.05)
        self._det_thresh.setToolTip(
            "Minimum confidence for a detected face to be accepted.\n"
            "Lower = detect more faces (including uncertain ones).\n"
            "Higher = only very confident detections.")
        self._det_thresh.valueChanged.connect(
            lambda v: self._save_cfg("det_thresh", v))
        g.addWidget(self._det_thresh, r, 1); r += 1

        g.addWidget(_row_label("Max faces"), r, 0)
        self._det_maxnum = _spinbox(self._cfg.get("det_maxnum", 10), 1, 20)
        self._det_maxnum.setToolTip("Maximum number of faces to process per image.")
        self._det_maxnum.valueChanged.connect(
            lambda v: self._save_cfg("det_maxnum", v))
        g.addWidget(self._det_maxnum, r, 1); r += 1

        g.addWidget(_section("Face Selection"), r, 0, 1, 3); r += 1

        g.addWidget(_row_label("Face ordering"), r, 0)
        order_key = self._cfg.get("face_order", "left-right")
        order_lbl = FACE_ORDERS[FACE_ORDER_KEY.index(order_key)] \
            if order_key in FACE_ORDER_KEY else FACE_ORDERS[0]
        self._face_order = _combo(FACE_ORDERS, order_lbl)
        self._face_order.setToolTip(
            "Sort detected faces before selecting source/target indices.")
        self._face_order.currentIndexChanged.connect(
            lambda i: self._save_cfg("face_order", FACE_ORDER_KEY[i]))
        g.addWidget(self._face_order, r, 1); r += 1

        g.addWidget(_row_label("Source face index"), r, 0)
        self._src_face_idx = _spinbox(self._cfg.get("source_face_idx", 0), 0, 9)
        self._src_face_idx.setToolTip(
            "Which detected face in the SOURCE image to use (0 = first).\n"
            "Only applies when source is an image, not a face model.")
        self._src_face_idx.valueChanged.connect(
            lambda v: self._save_cfg("source_face_idx", v))
        g.addWidget(self._src_face_idx, r, 1); r += 1

        g.addWidget(_row_label("Target face index"), r, 0)
        self._tgt_face_idx = QPushButton()   # styled line-edit-like
        from PySide6.QtWidgets import QLineEdit
        self._tgt_face_idx_edit = QLineEdit(
            self._cfg.get("target_face_idx", "0"))
        self._tgt_face_idx_edit.setFixedWidth(80)
        self._tgt_face_idx_edit.setStyleSheet(
            f"QLineEdit {{ background:{CAR}; color:{PRI}; border:1px solid {MUT};"
            f"border-radius:4px; padding:3px 6px; font-family:{FONT};"
            f"font-size:{FONT_SM}px; }}")
        self._tgt_face_idx_edit.setPlaceholderText("0,1,2…")
        self._tgt_face_idx_edit.setToolTip(
            "Comma-separated indices of target faces to replace.\n"
            "0 = first detected face. Leave as 0 to replace all.")
        self._tgt_face_idx_edit.textChanged.connect(
            lambda t: self._save_cfg("target_face_idx", t))
        g.addWidget(_row_label("Target face index"), r, 0)
        g.addWidget(self._tgt_face_idx_edit, r, 1); r += 1

        g.addWidget(_section("Gender Filter"), r, 0, 1, 3); r += 1

        g.addWidget(_row_label("Source gender"), r, 0)
        self._gender_src = _combo(GENDER_OPTS,
                                  GENDER_OPTS[self._cfg.get("gender_source", 0)])
        self._gender_src.setToolTip(
            "Only use faces matching this gender from the source image.")
        self._gender_src.currentIndexChanged.connect(
            lambda i: self._save_cfg("gender_source", i))
        g.addWidget(self._gender_src, r, 1); r += 1

        g.addWidget(_row_label("Target gender"), r, 0)
        self._gender_tgt = _combo(GENDER_OPTS,
                                  GENDER_OPTS[self._cfg.get("gender_target", 0)])
        self._gender_tgt.setToolTip(
            "Only replace faces matching this gender in the target image.")
        self._gender_tgt.currentIndexChanged.connect(
            lambda i: self._save_cfg("gender_target", i))
        g.addWidget(self._gender_tgt, r, 1); r += 1

        g.setRowStretch(r, 1)
        return page

    # ── Tab: Upscale ──────────────────────────────────────────────────────────

    def _build_upscale_tab(self) -> QWidget:
        page, g = _settings_page()
        r = 0

        g.addWidget(_section("Face Restoration (GFPGAN)"), r, 0, 1, 3); r += 1

        self._enhance_chk = _checkbox("Enable GFPGAN face enhancement",
                                      bool(self._cfg.get("enhance_face", False)))
        self._enhance_chk.setToolTip(
            "Run GFPGANv1.4 after the swap to restore and sharpen face details.\n"
            "First use downloads GFPGANv1.4.pth (~340 MB).")
        self._enhance_chk.stateChanged.connect(
            lambda s: self._save_cfg("enhance_face", bool(s)))
        g.addWidget(self._enhance_chk, r, 0, 1, 2); r += 1

        g.addWidget(_row_label("GFPGAN weight"), r, 0)
        self._gfpgan_weight = _dspinbox(
            self._cfg.get("gfpgan_weight", 0.5), 0.0, 1.0, 0.05)
        self._gfpgan_weight.setToolTip(
            "Blend weight for GFPGAN output.\n"
            "0.0 = swap only (no enhancement)  ·  1.0 = full GFPGAN output.")
        self._gfpgan_weight.valueChanged.connect(
            lambda v: self._save_cfg("gfpgan_weight", v))
        g.addWidget(self._gfpgan_weight, r, 1); r += 1

        g.addWidget(_row_label("Restore before upscale"), r, 0)
        self._restore_first = _checkbox("",
                                        bool(self._cfg.get("restore_first", True)))
        self._restore_first.setToolTip(
            "Apply GFPGAN before any upscaling pass.")
        self._restore_first.stateChanged.connect(
            lambda s: self._save_cfg("restore_first", bool(s)))
        g.addWidget(self._restore_first, r, 1); r += 1

        g.addWidget(_section("GFPGAN Model"), r, 0, 1, 3); r += 1

        status_color = GRN if GFPGAN_MODEL.exists() else AMB
        status_text  = (f"GFPGANv1.4.pth  ({GFPGAN_MODEL.stat().st_size // (1024*1024)} MB)"
                        if GFPGAN_MODEL.exists() else "Not downloaded")
        gfp_lbl = QLabel(status_text)
        gfp_lbl.setStyleSheet(
            f"color:{status_color}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f"background:transparent;")
        g.addWidget(gfp_lbl, r, 0, 1, 2); r += 1

        if not GFPGAN_MODEL.exists():
            dl_btn = QPushButton("Download GFPGANv1.4 (~340 MB)")
            dl_btn.setFixedHeight(30)
            dl_btn.setStyleSheet(
                f"QPushButton {{ background:{ACC}; color:{PRI}; border:none;"
                f"border-radius:4px; padding:4px 12px; font-family:{FONT};"
                f"font-size:{FONT_SM}px; }}"
                f"QPushButton:hover {{ background:#185FA5; }}")
            dl_btn.clicked.connect(lambda: self._ensure_gfpgan() and
                                   gfp_lbl.setText("Downloaded") or None)
            g.addWidget(dl_btn, r, 0, 1, 2); r += 1

        g.setRowStretch(r, 1)
        return page

    # ── Tab: Tools ────────────────────────────────────────────────────────────

    def _build_tools_tab(self) -> QWidget:
        page, g = _settings_page()
        r = 0

        g.addWidget(_section("Batch Face Model Export"), r, 0, 1, 3); r += 1

        info = QLabel(
            "Scan a folder of images, detect the primary face in each,\n"
            "and save a ReActor-compatible .safetensors model for each image.")
        info.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        info.setWordWrap(True)
        g.addWidget(info, r, 0, 1, 3); r += 1

        # Source folder
        g.addWidget(_row_label("Source folder"), r, 0)
        src_row = QHBoxLayout()
        self._batch_src_lbl = QLabel(
            self._cfg.get("batch_src_folder", "") or "Not selected", )
        self._batch_src_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        self._batch_src_lbl.setMaximumWidth(260)
        src_row.addWidget(self._batch_src_lbl)
        src_btn = QPushButton("Browse…")
        src_btn.setFixedHeight(26)
        src_btn.setStyleSheet(
            f"QPushButton {{ background:{MUT}; color:{PRI}; border:none;"
            f"border-radius:3px; padding:2px 10px; font-family:{FONT};"
            f"font-size:{FONT_SM}px; }}"
            f"QPushButton:hover {{ background:{ACC}; }}")
        src_btn.clicked.connect(self._browse_batch_src)
        src_row.addWidget(src_btn)
        src_row.addStretch(1)
        g.addLayout(src_row, r, 1, 1, 2); r += 1

        # Output folder
        g.addWidget(_row_label("Output folder"), r, 0)
        out_row = QHBoxLayout()
        self._batch_out_lbl = QLabel(
            self._cfg.get("batch_out_folder", str(FACES_DIR)))
        self._batch_out_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        self._batch_out_lbl.setMaximumWidth(260)
        out_row.addWidget(self._batch_out_lbl)
        out_btn = QPushButton("Browse…")
        out_btn.setFixedHeight(26)
        out_btn.setStyleSheet(src_btn.styleSheet())
        out_btn.clicked.connect(self._browse_batch_out)
        out_row.addWidget(out_btn)
        out_row.addStretch(1)
        g.addLayout(out_row, r, 1, 1, 2); r += 1

        # Run button + progress
        run_row = QHBoxLayout()
        self._batch_run_btn = QPushButton("Export All Faces")
        self._batch_run_btn.setFixedHeight(32)
        self._batch_run_btn.setStyleSheet(
            f"QPushButton {{ background:{GRN}; color:{PRI}; border:none;"
            f"border-radius:4px; padding:4px 16px; font-family:{FONT};"
            f"font-size:{FONT_MD}px; font-weight:bold; }}"
            f"QPushButton:disabled {{ background:{MUT}; color:{SEC}; }}"
            f"QPushButton:hover:!disabled {{ background:#2db84d; }}")
        self._batch_run_btn.clicked.connect(self._run_batch_export)
        run_row.addWidget(self._batch_run_btn)
        run_row.addStretch(1)
        g.addLayout(run_row, r, 0, 1, 3); r += 1

        self._batch_status = QLabel("")
        self._batch_status.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        g.addWidget(self._batch_status, r, 0, 1, 3); r += 1

        g.setRowStretch(r, 1)
        return page

    # ── Tab: Settings ─────────────────────────────────────────────────────────

    def _build_settings_tab(self) -> QWidget:
        page, g = _settings_page()
        r = 0

        g.addWidget(_section("Processing Device"), r, 0, 1, 3); r += 1

        g.addWidget(_row_label("Device"), r, 0)
        self._device_combo = _combo(
            DEVICE_OPTS, self._cfg.get("device", "Auto"))
        self._device_combo.setToolTip(
            "Auto: try CUDA -> DirectML -> CPU.\n"
            "CUDA: NVIDIA GPU.  DirectML: AMD/Intel GPU.\n"
            "CPU: fallback.")
        self._device_combo.currentTextChanged.connect(
            lambda t: self._save_cfg("device", t))
        g.addWidget(self._device_combo, r, 1); r += 1

        g.addWidget(_section("Swap Model"), r, 0, 1, 3); r += 1

        model_color = GRN if SWAP_MODEL.exists() else RED
        model_text  = (f"inswapper_128.onnx  "
                       f"({SWAP_MODEL.stat().st_size // (1024*1024)} MB)"
                       if SWAP_MODEL.exists()
                       else "inswapper_128.onnx  — NOT FOUND")
        model_lbl = QLabel(model_text)
        model_lbl.setStyleSheet(
            f"color:{model_color}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f"background:transparent;")
        g.addWidget(model_lbl, r, 0, 1, 3); r += 1

        if not SWAP_MODEL.exists():
            dl_btn = QPushButton("Download inswapper_128.onnx (~266 MB)")
            dl_btn.setFixedHeight(30)
            dl_btn.setStyleSheet(
                f"QPushButton {{ background:{ACC}; color:{PRI}; border:none;"
                f"border-radius:4px; padding:4px 12px; font-family:{FONT};"
                f"font-size:{FONT_SM}px; }}"
                f"QPushButton:hover {{ background:#185FA5; }}")
            dl_btn.clicked.connect(self._ensure_model)
            g.addWidget(dl_btn, r, 0, 1, 2); r += 1

        g.addWidget(_section("Faces Folder"), r, 0, 1, 3); r += 1

        folder_row = QHBoxLayout()
        self._folder_lbl = QLabel(str(self._faces_dir))
        self._folder_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        self._folder_lbl.setWordWrap(True)
        folder_row.addWidget(self._folder_lbl, stretch=1)
        chg_btn = QPushButton("Change…")
        chg_btn.setFixedHeight(26)
        chg_btn.setStyleSheet(
            f"QPushButton {{ background:{MUT}; color:{PRI}; border:none;"
            f"border-radius:3px; padding:2px 10px; font-family:{FONT};"
            f"font-size:{FONT_SM}px; }}"
            f"QPushButton:hover {{ background:{ACC}; }}")
        chg_btn.clicked.connect(self._browse_faces_folder)
        folder_row.addWidget(chg_btn)
        g.addLayout(folder_row, r, 0, 1, 3); r += 1

        g.setRowStretch(r, 1)
        return page

    # ── Face panel logic ──────────────────────────────────────────────────────

    def _browse_faces_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Faces Folder", str(self._faces_dir))
        if not folder:
            return
        self._faces_dir = Path(folder)
        self._cfg["faces_folder"] = folder
        save_json(CFG_FILE, self._cfg)
        self._load_face_images()
        if hasattr(self, "_folder_lbl"):
            self._folder_lbl.setText(folder)

    def _load_face_images(self):
        for card in self._face_cards:
            self._face_vl.removeWidget(card)
            card.deleteLater()
        self._face_cards.clear()
        self._sel_faces = []

        if not self._faces_dir.exists():
            self._face_hint.setVisible(True)
            self._face_count.setText("0")
            return

        all_paths = sorted([
            p for p in self._faces_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
            or p.suffix.lower() == FACE_MODEL_EXT
        ], key=lambda p: (p.suffix.lower() != FACE_MODEL_EXT, p.name.lower()))

        self._face_hint.setVisible(not all_paths)
        self._face_count.setText(str(len(all_paths)))

        for path in all_paths:
            if path.suffix.lower() == FACE_MODEL_EXT:
                card = FaceModelCard(str(path), self._face_content)
            else:
                card = FaceThumbCard(str(path), self._face_content)
            card.selected_changed.connect(self._on_face_selected)
            self._face_vl.insertWidget(self._face_vl.count() - 1, card)
            self._face_cards.append(card)

    def _deselect_all_faces(self):
        for card in self._face_cards:
            card.set_selected(False)
        self._sel_faces = []
        self._update_apply_btn()

    def _on_blend_toggled(self, state):
        # When blend is turned off, keep only the first selected face
        if not state and len(self._sel_faces) > 1:
            keep = self._sel_faces[0]
            for card in self._face_cards:
                if card.path != keep[0]:
                    card.set_selected(False)
            self._sel_faces = [keep]
        self._update_apply_btn()

    def _on_face_selected(self, card):
        if card.is_selected():
            # deselect
            card.set_selected(False)
            self._sel_faces = [(p, m) for p, m in self._sel_faces
                               if p != card.path]
        else:
            blend = self._blend_chk.isChecked()
            if not blend:
                # single-select: deselect all others first
                for c in self._face_cards:
                    c.set_selected(False)
                self._sel_faces = []
            card.set_selected(True)
            self._sel_faces.append((card.path, card.is_model))
        self._update_apply_btn()

    # ── Image browse / folder list ────────────────────────────────────────────

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self._cfg.get("last_input_dir", ""),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif)")
        if not path:
            return
        self._cfg["last_input_dir"] = os.path.dirname(path)
        save_json(CFG_FILE, self._cfg)
        self._folder_images = []
        self._bswap_btn.setVisible(False)
        self._copy_tags_chk.setVisible(False)
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
        self._folder_images = paths
        self._bswap_btn.setVisible(True)
        self._copy_tags_chk.setVisible(True)
        self._load_img_list(paths)
        if paths:
            self._select_pill(self._input_pills[0], paths[0])

    def _load_img_list(self, paths: list[str]):
        for pill in self._input_pills:
            self._img_list_vl.removeWidget(pill)
            pill.deleteLater()
        self._input_pills.clear()
        self._active_pill = None
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            pill = QPushButton(name, self._img_list_content)
            pill.setFixedSize(THUMB_D + 8, 26)
            pill.setStyleSheet(self._pill_ss(False))
            pill.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            pill.clicked.connect(lambda _, p=pill, pa=path:
                                 self._select_pill(p, pa))
            self._img_list_vl.insertWidget(self._img_list_vl.count() - 1, pill)
            self._input_pills.append(pill)
        self._img_count.setText(str(len(paths)))
        self._img_list_panel.setMaximumWidth(THUMB_D + 16)
        self._img_list_panel.setVisible(True)

    def _pill_ss(self, active: bool) -> str:
        bg = ACC if active else MUT
        hv = ACC if active else "#2a4a6a"
        return (f"QPushButton {{ background:{bg}; color:{PRI}; border:none;"
                f"border-radius:3px; text-align:left; padding:0 6px;"
                f"font-family:{FONT}; font-size:{FONT_SM}px; }}"
                f"QPushButton:hover {{ background:{hv}; }}")

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
        self._src_pil          = img
        self._src_path         = path
        self._out_pil          = None
        self._last_source_face = None
        self._viewing   = "input"
        name = os.path.basename(path)
        self._path_lbl.setText(name[:50] + ("…" if len(name) > 50 else ""))
        self._viewer.set_image(img)
        w, h = img.size
        self._status_lbl.setText(f"{w} × {h} px")
        self._save_btn.setEnabled(False)
        self._toggle_btn.setVisible(False)
        self._export_btn.setEnabled(True)
        self._update_apply_btn()

    def _update_apply_btn(self):
        ready = self._src_pil is not None and bool(self._sel_faces)
        self._apply_btn.setEnabled(ready)
        self._bswap_btn.setEnabled(bool(self._folder_images) and bool(self._sel_faces))

    def _save_cfg(self, key: str, value):
        self._cfg[key] = value
        save_json(CFG_FILE, self._cfg)

    # ── Face swap inference ───────────────────────────────────────────────────

    def _run_swap(self):
        if self._src_pil is None or not self._sel_faces:
            return
        if not self._ensure_model():
            return
        enhance = self._enhance_chk.isChecked()
        if enhance and not self._ensure_gfpgan():
            return

        blend = self._blend_chk.isChecked() and len(self._sel_faces) > 1

        # ── Load GFPGAN on the main thread (avoids QThread/OpenMP deadlock) ──
        gfpgan_restorer = None
        if enhance:
            try:
                import sys, io
                self._patch_torchvision_functional_tensor()
                from gfpgan import GFPGANer
                self._status_lbl.setText("Loading GFPGAN… (may download ~185 MB on first run)")
                QApplication.processEvents()
                # Guard against None stdout/stderr (no console / windowed mode)
                _stdout, _stderr = sys.stdout, sys.stderr
                if sys.stdout is None:
                    sys.stdout = io.StringIO()
                if sys.stderr is None:
                    sys.stderr = io.StringIO()
                try:
                    gfpgan_restorer = GFPGANer(
                        model_path=str(GFPGAN_MODEL),
                        upscale=1, arch="clean",
                        channel_multiplier=2,
                        bg_upsampler=None)
                finally:
                    sys.stdout, sys.stderr = _stdout, _stderr
            except Exception as e:
                QMessageBox.critical(self, "GFPGAN Error",
                                     f"Failed to load GFPGAN:\n{e}")
                return

        # Build source face inputs for the worker
        # Each entry: (face_img_or_None, face_model_path_or_"")
        source_inputs = []
        for path, is_model in self._sel_faces:
            if is_model:
                source_inputs.append((None, path))
            else:
                try:
                    img = Image.open(path).convert("RGB")
                    source_inputs.append((img, ""))
                except Exception as e:
                    QMessageBox.critical(self, "Face Load Error", str(e))
                    return

        self._apply_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._toggle_btn.setVisible(False)
        self._status_lbl.setText("Processing…")

        self._prog_dlg = QProgressDialog(
            "Applying face swap…", None, 0, 0, self)
        self._prog_dlg.setWindowTitle("Face Swap")
        self._prog_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._prog_dlg.show()

        self._thread = QThread()
        self._worker = FaceSwapWorker(
            self._src_pil,
            source_inputs,
            str(SWAP_MODEL),
            blend=blend,
            gfpgan_restorer=gfpgan_restorer,
            gfpgan_weight=self._gfpgan_weight.value(),
            det_thresh=self._det_thresh.value(),
            det_maxnum=self._det_maxnum.value(),
            face_order=FACE_ORDER_KEY[self._face_order.currentIndex()],
            source_face_idx=self._src_face_idx.value(),
            target_face_idx=self._tgt_face_idx_edit.text(),
            gender_source=self._gender_src.currentIndex(),
            gender_target=self._gender_tgt.currentIndex(),
            device=self._device_combo.currentText(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_swap_done)
        self._worker.error.connect(self._on_swap_error)
        self._worker.status.connect(
            lambda s: self._prog_dlg.setLabelText(s) if self._prog_dlg else None)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_swap_done(self, result: Image.Image):
        if self._prog_dlg:
            self._prog_dlg.close()
            self._prog_dlg = None
        # Capture the exact source face used (blended or single) for export
        if self._worker is not None and self._worker.source_face is not None:
            self._last_source_face = self._worker.source_face
        self._out_pil  = result
        self._viewing  = "output"
        self._viewer.set_image(result)
        w, h = result.size
        self._status_lbl.setText(f"Done  —  {w} x {h} px")
        self._apply_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        self._toggle_btn.setVisible(True)
        self._toggle_btn.setText("View Original")

    def _on_swap_error(self, msg: str):
        if self._prog_dlg:
            self._prog_dlg.close()
            self._prog_dlg = None
        self._status_lbl.setText("Error")
        self._update_apply_btn()
        QMessageBox.critical(self, "Face Swap Error", msg)

    def _toggle_view(self):
        if self._viewing == "output" and self._src_pil:
            self._viewer.set_image(self._src_pil)
            self._viewing = "input"
            self._toggle_btn.setText("View Result")
        elif self._viewing == "input" and self._out_pil:
            self._viewer.set_image(self._out_pil)
            self._viewing = "output"
            self._toggle_btn.setText("View Original")

    # ── Batch face swap ───────────────────────────────────────────────────────

    def _run_batch_swap(self):
        if not self._folder_images or not self._sel_faces:
            return
        if not self._ensure_model():
            return
        enhance = self._enhance_chk.isChecked()
        if enhance and not self._ensure_gfpgan():
            return

        # Ask for output folder
        src_folder = os.path.dirname(self._folder_images[0])
        default_out = self._cfg.get("batch_swap_out_folder", "") or \
                      os.path.join(src_folder, "swapped")
        out_folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", default_out)
        if not out_folder:
            return
        self._cfg["batch_swap_out_folder"] = out_folder
        save_json(CFG_FILE, self._cfg)

        blend = self._blend_chk.isChecked() and len(self._sel_faces) > 1

        # Load GFPGAN on main thread
        gfpgan_restorer = None
        if enhance:
            try:
                import sys, io
                self._patch_torchvision_functional_tensor()
                from gfpgan import GFPGANer
                self._status_lbl.setText("Loading GFPGAN… (may download ~185 MB on first run)")
                QApplication.processEvents()
                _stdout, _stderr = sys.stdout, sys.stderr
                if sys.stdout is None: sys.stdout = io.StringIO()
                if sys.stderr is None: sys.stderr = io.StringIO()
                try:
                    gfpgan_restorer = GFPGANer(
                        model_path=str(GFPGAN_MODEL),
                        upscale=1, arch="clean",
                        channel_multiplier=2,
                        bg_upsampler=None)
                finally:
                    sys.stdout, sys.stderr = _stdout, _stderr
            except Exception as e:
                QMessageBox.critical(self, "GFPGAN Error",
                                     f"Failed to load GFPGAN:\n{e}")
                return

        source_inputs = []
        for path, is_model in self._sel_faces:
            if is_model:
                source_inputs.append((None, path))
            else:
                try:
                    source_inputs.append((Image.open(path).convert("RGB"), ""))
                except Exception as e:
                    QMessageBox.critical(self, "Face Load Error", str(e))
                    return

        self._apply_btn.setEnabled(False)
        self._bswap_btn.setEnabled(False)
        self._status_lbl.setText(
            f"Batch swap — 0 / {len(self._folder_images)}")

        self._prog_dlg = QProgressDialog(
            f"Batch swapping 0 / {len(self._folder_images)}…",
            "Cancel", 0, len(self._folder_images), self)
        self._prog_dlg.setWindowTitle("Batch Face Swap")
        self._prog_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._prog_dlg.canceled.connect(self._cancel_batch_swap)
        self._prog_dlg.show()

        self._bswap_thread = QThread()
        self._bswap_worker = BatchSwapWorker(
            self._folder_images,
            source_inputs,
            str(SWAP_MODEL),
            out_folder,
            blend=blend,
            gfpgan_restorer=gfpgan_restorer,
            gfpgan_weight=self._gfpgan_weight.value(),
            det_thresh=self._det_thresh.value(),
            det_maxnum=self._det_maxnum.value(),
            face_order=FACE_ORDER_KEY[self._face_order.currentIndex()],
            source_face_idx=self._src_face_idx.value(),
            target_face_idx=self._tgt_face_idx_edit.text(),
            gender_source=self._gender_src.currentIndex(),
            gender_target=self._gender_tgt.currentIndex(),
            device=self._device_combo.currentText(),
            copy_tags=self._copy_tags_chk.isChecked(),
        )
        self._bswap_worker.moveToThread(self._bswap_thread)
        self._bswap_thread.started.connect(self._bswap_worker.run)
        self._bswap_worker.progress.connect(self._on_bswap_progress)
        self._bswap_worker.finished.connect(self._on_bswap_done)
        self._bswap_worker.error.connect(self._on_bswap_error)
        self._bswap_worker.status.connect(
            lambda s: self._prog_dlg.setLabelText(s) if self._prog_dlg else None)
        self._bswap_worker.finished.connect(self._bswap_thread.quit)
        self._bswap_worker.error.connect(self._bswap_thread.quit)
        self._bswap_thread.start()

    def _cancel_batch_swap(self):
        if self._bswap_worker:
            self._bswap_worker.cancel()

    def _on_bswap_progress(self, cur: int, total: int, name: str):
        self._status_lbl.setText(f"Batch swap — {cur} / {total}  {name}")
        if self._prog_dlg:
            self._prog_dlg.setValue(cur)
            self._prog_dlg.setLabelText(
                f"[{cur}/{total}] {name}")

    def _on_bswap_done(self, saved: int, skipped: int):
        if self._prog_dlg:
            self._prog_dlg.close()
            self._prog_dlg = None
        self._apply_btn.setEnabled(self._src_pil is not None and bool(self._sel_faces))
        self._bswap_btn.setEnabled(bool(self._folder_images) and bool(self._sel_faces))
        out = self._cfg.get("batch_swap_out_folder", "")
        self._status_lbl.setText(
            f"Batch done — {saved} saved, {skipped} skipped")
        QMessageBox.information(
            self, "Batch Swap Complete",
            f"{saved} image(s) saved to:\n{out}\n\n"
            f"{skipped} skipped (no face detected).")

    def _on_bswap_error(self, msg: str):
        if self._prog_dlg:
            self._prog_dlg.close()
            self._prog_dlg = None
        self._apply_btn.setEnabled(self._src_pil is not None and bool(self._sel_faces))
        self._bswap_btn.setEnabled(bool(self._folder_images) and bool(self._sel_faces))
        self._status_lbl.setText("Batch swap error")
        QMessageBox.critical(self, "Batch Swap Error", msg)

    # ── Export face model ─────────────────────────────────────────────────────

    def _export_face_model(self):
        if self._src_pil is None:
            return
        base = os.path.splitext(os.path.basename(self._src_path))[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Face Model",
            os.path.join(str(self._faces_dir), f"{base}.safetensors"),
            "ReActor Face Model (*.safetensors)")
        if not path:
            return

        # If a swap was just done, use the exact source face (blended or single)
        # that was used — no re-detection needed.
        if self._last_source_face is not None:
            self._status_lbl.setText("Saving face model…")
            try:
                save_reactor_face(self._last_source_face, path)
                self._on_export_done(path)
            except Exception as e:
                import traceback
                self._on_export_error(f"{e}\n{traceback.format_exc()}")
            return

        # No prior swap — detect face from the loaded image
        self._export_btn.setEnabled(False)
        self._status_lbl.setText("Exporting face model…")
        self._prog_dlg = QProgressDialog(
            "Detecting face and saving model…", None, 0, 0, self)
        self._prog_dlg.setWindowTitle("Export Face Model")
        self._prog_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._prog_dlg.show()

        self._exp_thread = QThread()
        self._exp_worker = FaceExportWorker(
            self._src_pil, path,
            det_thresh=self._det_thresh.value(),
            device=self._device_combo.currentText(),
        )
        self._exp_worker.moveToThread(self._exp_thread)
        self._exp_thread.started.connect(self._exp_worker.run)
        self._exp_worker.finished.connect(self._on_export_done)
        self._exp_worker.error.connect(self._on_export_error)
        self._exp_worker.status.connect(
            lambda s: self._prog_dlg.setLabelText(s) if self._prog_dlg else None)
        self._exp_worker.finished.connect(self._exp_thread.quit)
        self._exp_worker.error.connect(self._exp_thread.quit)
        self._exp_thread.start()

    def _on_export_done(self, saved_path: str):
        if self._prog_dlg:
            self._prog_dlg.close()
            self._prog_dlg = None
        self._export_btn.setEnabled(True)
        self._status_lbl.setText(f"Saved: {os.path.basename(saved_path)}")
        if Path(saved_path).parent.resolve() == self._faces_dir.resolve():
            self._load_face_images()
        QMessageBox.information(self, "Face Model Saved",
                                f"Saved:\n{saved_path}")

    def _on_export_error(self, msg: str):
        if self._prog_dlg:
            self._prog_dlg.close()
            self._prog_dlg = None
        self._export_btn.setEnabled(True)
        self._status_lbl.setText("Export failed")
        QMessageBox.critical(self, "Export Error", msg)

    # ── Batch export ──────────────────────────────────────────────────────────

    def _browse_batch_src(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Source Folder",
            self._cfg.get("batch_src_folder", ""))
        if not folder:
            return
        self._cfg["batch_src_folder"] = folder
        save_json(CFG_FILE, self._cfg)
        self._batch_src_lbl.setText(folder)

    def _browse_batch_out(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder",
            self._cfg.get("batch_out_folder", str(FACES_DIR)))
        if not folder:
            return
        self._cfg["batch_out_folder"] = folder
        save_json(CFG_FILE, self._cfg)
        self._batch_out_lbl.setText(folder)

    def _run_batch_export(self):
        src = self._cfg.get("batch_src_folder", "")
        out = self._cfg.get("batch_out_folder", str(FACES_DIR))
        if not src or not Path(src).exists():
            QMessageBox.warning(self, "No Source Folder",
                                "Please select a source folder first.")
            return

        self._batch_run_btn.setEnabled(False)
        self._batch_status.setText("Starting…")

        self._batch_thread = QThread()
        self._batch_worker = BatchExportWorker(
            src, out, self._device_combo.currentText())
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_done)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.error.connect(self._batch_thread.quit)
        self._batch_thread.start()

    def _on_batch_progress(self, cur: int, total: int, name: str):
        self._batch_status.setText(f"[{cur}/{total}] {name}")

    def _on_batch_done(self, saved: int, skipped: int):
        self._batch_run_btn.setEnabled(True)
        self._batch_status.setText(
            f"Done — {saved} model(s) saved, {skipped} skipped (no face detected)")
        self._load_face_images()   # refresh panel

    def _on_batch_error(self, msg: str):
        self._batch_run_btn.setEnabled(True)
        self._batch_status.setText("Error")
        QMessageBox.critical(self, "Batch Export Error", msg)

    # ── Model management — inswapper ──────────────────────────────────────────

    def _ensure_model(self) -> bool:
        if SWAP_MODEL.exists():
            return True
        r = QMessageBox.question(
            self, "Model Not Found",
            "inswapper_128.onnx not found.\n\n"
            f"Download (~266 MB)?\n\nOr place it manually in:\n{SWAP_MODEL.parent}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes:
            return False
        return self._download_model()

    def _download_model(self) -> bool:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            QMessageBox.critical(self, "Missing Package",
                                 "pip install huggingface-hub")
            return False
        dlg = QProgressDialog(
            "Downloading inswapper_128.onnx…", "Cancel", 0, 0, self)
        dlg.setWindowTitle("Downloading")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.show()
        QApplication.processEvents()
        result, done = [None], threading.Event()
        _SOURCES = [("ezioruan/inswapper_128.onnx", "inswapper_128.onnx"),
                    ("deepinsight/inswapper",        "inswapper_128.onnx")]

        def _do():
            for repo, fn in _SOURCES:
                try:
                    hf_hub_download(repo_id=repo, filename=fn,
                                    local_dir=str(SWAP_MODEL.parent))
                    result[0] = True
                    done.set()
                    return
                except Exception as e:
                    result[0] = str(e)
            done.set()

        threading.Thread(target=_do, daemon=True).start()
        while not done.is_set():
            if dlg.wasCanceled():
                done.set()
                dlg.close()
                return False
            QApplication.processEvents()
            done.wait(0.1)
        dlg.close()
        if result[0] is True and SWAP_MODEL.exists():
            return True
        QMessageBox.critical(self, "Download Failed", str(result[0]))
        return False

    # ── Model management — GFPGAN ─────────────────────────────────────────────

    @staticmethod
    def _patch_torchvision_functional_tensor():
        """Shim for torchvision >= 0.16 which removed functional_tensor.
        basicsr (GFPGAN dependency) still imports from it."""
        import sys, types
        if 'torchvision.transforms.functional_tensor' not in sys.modules:
            import torchvision.transforms.functional as _tvf
            _ft = types.ModuleType('torchvision.transforms.functional_tensor')
            for _attr in ('rgb_to_grayscale', 'adjust_brightness', 'adjust_contrast',
                          'adjust_hue', 'adjust_saturation', 'normalize',
                          'to_tensor', 'resize', 'pad', 'crop'):
                if hasattr(_tvf, _attr):
                    setattr(_ft, _attr, getattr(_tvf, _attr))
            sys.modules['torchvision.transforms.functional_tensor'] = _ft

    def _ensure_gfpgan(self) -> bool:
        if GFPGAN_MODEL.exists():
            return True
        r = QMessageBox.question(
            self, "GFPGAN Not Found",
            "GFPGANv1.4.pth not found.\n\nDownload (~340 MB)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes:
            return False
        return self._download_gfpgan()

    def _download_gfpgan(self) -> bool:
        dlg = QProgressDialog(
            "Downloading GFPGANv1.4.pth (~340 MB)…", "Cancel", 0, 0, self)
        dlg.setWindowTitle("Downloading GFPGAN")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.show()
        QApplication.processEvents()
        result, done = [None], threading.Event()

        def _do():
            try:
                urllib.request.urlretrieve(GFPGAN_URL, str(GFPGAN_MODEL))
                result[0] = True
            except Exception as e:
                result[0] = str(e)
            finally:
                done.set()

        threading.Thread(target=_do, daemon=True).start()
        while not done.is_set():
            if dlg.wasCanceled():
                done.set()
                dlg.close()
                GFPGAN_MODEL.unlink(missing_ok=True)
                return False
            QApplication.processEvents()
            done.wait(0.1)
        dlg.close()
        if result[0] is True and GFPGAN_MODEL.exists():
            return True
        QMessageBox.critical(self, "Download Failed", str(result[0]))
        return False

    # ── Size control / save output ────────────────────────────────────────────

    def _on_size_changed(self):
        v       = self._size_spin.value()
        rounded = round(v / CARD_D_STEP) * CARD_D_STEP
        if rounded != v:
            self._size_spin.blockSignals(True)
            self._size_spin.setValue(rounded)
            self._size_spin.blockSignals(False)
        self._cfg["card_size"] = rounded
        save_json(CFG_FILE, self._cfg)
        self._viewer.resize_card(rounded)

    def _save_output(self):
        if self._out_pil is None:
            return
        base = os.path.splitext(os.path.basename(self._src_path))[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image",
            os.path.join(self._cfg.get("last_output_dir", ""),
                         f"{base}-SWAP.png"),
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
