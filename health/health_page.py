"""
health/health_page.py — LoRA Health Analyzer for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.01)

Loads a .safetensors LoRA file and runs 8 structural checks:
  1. File Integrity   — hash metadata present
  2. NaN / Inf        — tensor value sanity
  3. Rank Consistency — shape agreement and metadata match
  4. Alpha/Rank Ratio — declared alpha relative to rank
  5. Rank Range       — rank within community-validated bounds
  6. Dead Layers      — layers with near-zero weights
  7. Overbaked        — layers with abnormally high magnitude (overtrained)
  8. Layer Balance    — hottest vs coldest layer ratio
"""
from __future__ import annotations

import os
import threading

from PySide6.QtCore    import Qt, QObject, Signal
from PySide6.QtGui     import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QComboBox,
    QScrollArea, QTabWidget, QFileDialog,
)

from shared.theme import (
    BG, PAN, CAR, ACC, GRN, RED, MUT, PRI, SEC, AMB,
    FONT, FONT_SM, FONT_MD, FONT_LG, FONT_XL,
    SIGNATURE,
)
from shared.config import HEALTH_CFG, load_json, save_json


# ── Community-sourced threshold presets ───────────────────────────────────────
# Values derived from: kohya-ss training guides, CivitAI wiki,
# LyCORIS/DyLoRA community benchmarks, and Ostris AI Toolkit documentation.
THRESHOLD_PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "sd15": {
        "Strict": {
            "rank_min":   8,    "rank_max":  32,
            "mag_dead":  1e-4,
            "mag_warn":   4.0,  "mag_fail":  8.0,
            "bal_warn":  100.0, "bal_fail":  500.0,
            "ratio_min":  0.25, "ratio_max": 1.0,
        },
        "Standard": {
            "rank_min":   4,    "rank_max":  64,
            "mag_dead":  1e-5,
            "mag_warn":   6.0,  "mag_fail": 12.0,
            "bal_warn":  200.0, "bal_fail": 1000.0,
            "ratio_min":  0.10, "ratio_max": 1.0,
        },
        "Relaxed": {
            "rank_min":   2,    "rank_max": 128,
            "mag_dead":  1e-6,
            "mag_warn":  10.0,  "mag_fail": 20.0,
            "bal_warn":  500.0, "bal_fail": 5000.0,
            "ratio_min":  0.05, "ratio_max": 2.0,
        },
    },
    "sdxl": {
        "Strict": {
            "rank_min":  16,    "rank_max": 128,
            "mag_dead":  1e-4,
            "mag_warn":   3.5,  "mag_fail":  7.0,
            "bal_warn":  100.0, "bal_fail":  500.0,
            "ratio_min":  0.25, "ratio_max": 1.0,
        },
        "Standard": {
            "rank_min":   8,    "rank_max": 256,
            "mag_dead":  1e-5,
            "mag_warn":   5.0,  "mag_fail": 10.0,
            "bal_warn":  200.0, "bal_fail": 1000.0,
            "ratio_min":  0.10, "ratio_max": 1.0,
        },
        "Relaxed": {
            "rank_min":   4,    "rank_max": 512,
            "mag_dead":  1e-6,
            "mag_warn":   8.0,  "mag_fail": 16.0,
            "bal_warn":  500.0, "bal_fail": 5000.0,
            "ratio_min":  0.05, "ratio_max": 2.0,
        },
    },
}

THRESHOLD_LABELS: dict[str, str] = {
    "rank_min":  "Min Rank",
    "rank_max":  "Max Rank",
    "mag_dead":  "Dead Layer Threshold",
    "mag_warn":  "Overcooked Warning",
    "mag_fail":  "Overcooked Fail",
    "bal_warn":  "Balance Ratio Warning",
    "bal_fail":  "Balance Ratio Fail",
    "ratio_min": "Alpha/Rank Min",
    "ratio_max": "Alpha/Rank Max",
}

DEFAULTS: dict = {
    "sd15_preset":    "Standard",
    "sdxl_preset":    "Standard",
    "sd15_overrides": {k: "" for k in THRESHOLD_LABELS},
    "sdxl_overrides": {k: "" for k in THRESHOLD_LABELS},
}


# ── Widget helpers ─────────────────────────────────────────────────────────────
def _lbl(text: str, color: str = SEC, size: int = FONT_MD, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{color}; font-family:{FONT}; font-size:{size}px;"
        f" font-weight:{'bold' if bold else 'normal'}; background:transparent;")
    return lbl


def _card_style() -> str:
    return f"QFrame {{ background:{CAR}; border-radius:8px; border:2px solid {MUT}; }}"


def _amber_edit_style() -> str:
    return (
        f"QLineEdit {{ background:{CAR}; color:{PRI}; border:2px solid {AMB};"
        f" border-radius:4px; padding:2px 6px; font-family:{FONT}; font-size:{FONT_MD}px; }}"
        f"QLineEdit:focus {{ border-color:#ffb347; }}"
    )


def _combo_style() -> str:
    return (
        f"QComboBox {{ background:{CAR}; color:{PRI}; border:2px solid {MUT};"
        f" border-radius:4px; padding:2px 6px; font-family:{FONT}; font-size:{FONT_MD}px;"
        f" min-height:26px; }}"
        f"QComboBox::drop-down {{ border:none; width:20px; }}"
        f"QComboBox QAbstractItemView {{ background:{CAR}; color:{PRI};"
        f" border:1px solid {ACC}; selection-background-color:{ACC}; }}"
    )


# ── Analysis worker ────────────────────────────────────────────────────────────
class _AnalysisWorker(QObject):
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, path: str, get_threshold_fn, model_type_override: str | None = None):
        super().__init__()
        self._path       = path
        self._get_fn     = get_threshold_fn
        self._type_override = model_type_override

    def run(self):
        try:
            result = _analyse(self._path, self._get_fn, self._type_override)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Core analysis ─────────────────────────────────────────────────────────────
def _detect_model_type(keys: list[str]) -> str:
    has_te2  = any("lora_te2_" in k for k in keys)
    up_count = sum(1 for k in keys if "lora_up" in k or "lora_B" in k)
    if has_te2 or up_count > 300:
        return "sdxl"
    return "sd15"


def _analyse(path: str, get_threshold, model_type_override: str | None = None) -> dict:
    """Load safetensors and run all 8 checks. Returns result dict."""
    try:
        from safetensors import safe_open
        import torch
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {e}. Install safetensors and torch.")

    tensors:  dict = {}
    metadata: dict = {}

    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            tensors[key] = f.get_tensor(key).float()

    model_type = model_type_override or _detect_model_type(list(tensors.keys()))
    checks: list[dict] = []

    # ── 1. File Integrity ─────────────────────────────────────────────────────
    has_hash   = "sshs_model_hash"  in metadata
    has_legacy = "sshs_legacy_hash" in metadata
    hash_parts = []
    if has_hash:   hash_parts.append(f"model_hash: {metadata['sshs_model_hash'][:16]}…")
    if has_legacy: hash_parts.append(f"legacy_hash: {metadata['sshs_legacy_hash'][:16]}…")
    if not has_hash and not has_legacy:
        hash_parts = ["No hash metadata — file may not have been saved by kohya/sd-scripts."]
    checks.append({
        "id": "integrity", "label": "File Integrity",
        "status": "pass" if (has_hash or has_legacy) else "warn",
        "detail": "  ".join(hash_parts),
    })

    # ── 2. NaN / Inf ──────────────────────────────────────────────────────────
    import torch
    nan_keys = [k for k, t in tensors.items() if torch.isnan(t).any()]
    inf_keys = [k for k, t in tensors.items() if torch.isinf(t).any()]
    if nan_keys or inf_keys:
        parts = []
        if nan_keys: parts.append(f"NaN in {len(nan_keys)} tensor(s)")
        if inf_keys: parts.append(f"Inf in {len(inf_keys)} tensor(s)")
        checks.append({"id": "nan_inf", "label": "NaN / Inf",
                        "status": "fail", "detail": ", ".join(parts)})
    else:
        checks.append({"id": "nan_inf", "label": "NaN / Inf",
                        "status": "pass", "detail": "All tensors are finite."})

    # ── Collect structural keys ───────────────────────────────────────────────
    up_keys   = [k for k in tensors if "lora_up"   in k or "lora_B"  in k]
    down_keys = [k for k in tensors if "lora_down" in k or "lora_A"  in k]

    # ── 3. Rank Consistency ───────────────────────────────────────────────────
    rank_errors:  list[str] = []
    actual_ranks: set[int]  = set()
    meta_rank:    int | None = None

    if "ss_network_dim" in metadata:
        try:
            meta_rank = int(metadata["ss_network_dim"])
        except (ValueError, TypeError):
            pass

    for dk in down_keys:
        t = tensors[dk]
        r = t.shape[0]
        actual_ranks.add(r)
        uk = dk.replace("lora_down", "lora_up").replace("lora_A", "lora_B")
        if uk in tensors:
            up_r = tensors[uk].shape[1]
            if r != up_r:
                rank_errors.append(f"{dk}: down={r} ≠ up={up_r}")

    if meta_rank is not None:
        for r in actual_ranks:
            if r != meta_rank:
                rank_errors.append(f"metadata rank={meta_rank} ≠ tensor rank={r}")

    actual_rank = next(iter(actual_ranks), None)
    if rank_errors:
        checks.append({"id": "rank_consistency", "label": "Rank Consistency",
                        "status": "fail", "detail": "; ".join(rank_errors[:3])})
    else:
        rank_str = f"rank={actual_rank}"
        if meta_rank:
            rank_str += f" (metadata confirms {meta_rank})"
        checks.append({"id": "rank_consistency", "label": "Rank Consistency",
                        "status": "pass", "detail": rank_str})

    # ── 4. Alpha/Rank Ratio ───────────────────────────────────────────────────
    alpha_keys     = [k for k in tensors if k.endswith(".alpha")]
    declared_alpha: float | None = None

    if "ss_network_alpha" in metadata:
        try:
            declared_alpha = float(metadata["ss_network_alpha"])
        except (ValueError, TypeError):
            pass
    if declared_alpha is None and alpha_keys:
        declared_alpha = float(tensors[alpha_keys[0]].item())

    ratio_min = get_threshold(model_type, "ratio_min")
    ratio_max = get_threshold(model_type, "ratio_max")

    if declared_alpha is not None and actual_rank:
        ratio     = declared_alpha / actual_rank
        ratio_str = f"alpha={declared_alpha:.4g} / rank={actual_rank} = {ratio:.4f}"
        if ratio < ratio_min:
            status = "warn"
            detail = f"{ratio_str} (below min {ratio_min})"
        elif ratio > ratio_max:
            status = "warn"
            detail = f"{ratio_str} (above max {ratio_max})"
        else:
            status = "pass"
            detail = ratio_str
        checks.append({"id": "alpha_ratio", "label": "Alpha/Rank Ratio",
                        "status": status, "detail": detail})
    else:
        checks.append({"id": "alpha_ratio", "label": "Alpha/Rank Ratio",
                        "status": "info", "detail": "Could not determine alpha or rank."})

    # ── 5. Rank Range ─────────────────────────────────────────────────────────
    rank_min_t = int(get_threshold(model_type, "rank_min"))
    rank_max_t = int(get_threshold(model_type, "rank_max"))

    if actual_rank is not None:
        if actual_rank < rank_min_t:
            checks.append({"id": "rank_range", "label": "Rank Range",
                            "status": "warn",
                            "detail": f"rank={actual_rank} below minimum {rank_min_t} for {model_type.upper()}"})
        elif actual_rank > rank_max_t:
            checks.append({"id": "rank_range", "label": "Rank Range",
                            "status": "warn",
                            "detail": f"rank={actual_rank} above recommended {rank_max_t} for {model_type.upper()}"})
        else:
            checks.append({"id": "rank_range", "label": "Rank Range",
                            "status": "pass",
                            "detail": f"rank={actual_rank} within [{rank_min_t}, {rank_max_t}] for {model_type.upper()}"})
    else:
        checks.append({"id": "rank_range", "label": "Rank Range",
                        "status": "info", "detail": "Could not determine rank."})

    # ── Per-layer magnitudes ──────────────────────────────────────────────────
    mag_dead = get_threshold(model_type, "mag_dead")
    mag_warn = get_threshold(model_type, "mag_warn")
    mag_fail = get_threshold(model_type, "mag_fail")

    up_mags:     list[float] = []
    dead_layers: list[str]   = []

    for uk in up_keys:
        mag = float(tensors[uk].abs().mean().item())
        up_mags.append(mag)
        if mag < mag_dead:
            dead_layers.append(uk)

    # ── 6. Dead Layers ────────────────────────────────────────────────────────
    if dead_layers:
        sample = f": {os.path.basename(dead_layers[0])}" if dead_layers else ""
        checks.append({"id": "dead_layers", "label": "Dead Layers",
                        "status": "warn",
                        "detail": f"{len(dead_layers)} layer(s) with mean_abs < {mag_dead:.2g}{sample}"})
    elif up_mags:
        checks.append({"id": "dead_layers", "label": "Dead Layers",
                        "status": "pass",
                        "detail": f"No dead layers (threshold: {mag_dead:.2g})"})
    else:
        checks.append({"id": "dead_layers", "label": "Dead Layers",
                        "status": "info", "detail": "No lora_up tensors found."})

    # ── 7. Overbaked ──────────────────────────────────────────────────────────
    if up_mags:
        mean_mag = sum(up_mags) / len(up_mags)
        if mean_mag > mag_fail:
            checks.append({"id": "overbaked", "label": "Overbaked",
                            "status": "fail",
                            "detail": f"mean magnitude={mean_mag:.4f} — exceeds fail threshold {mag_fail}"})
        elif mean_mag > mag_warn:
            checks.append({"id": "overbaked", "label": "Overbaked",
                            "status": "warn",
                            "detail": f"mean magnitude={mean_mag:.4f} — exceeds warning threshold {mag_warn}"})
        else:
            checks.append({"id": "overbaked", "label": "Overbaked",
                            "status": "pass",
                            "detail": f"mean magnitude={mean_mag:.4f} within normal range (warn>{mag_warn})"})
    else:
        checks.append({"id": "overbaked", "label": "Overbaked",
                        "status": "info", "detail": "No lora_up tensors found."})

    # ── 8. Layer Balance ──────────────────────────────────────────────────────
    bal_warn = get_threshold(model_type, "bal_warn")
    bal_fail = get_threshold(model_type, "bal_fail")

    active_mags = [m for m in up_mags if m >= mag_dead]
    if len(active_mags) >= 2:
        hottest = max(active_mags)
        coldest = min(active_mags)
        ratio   = hottest / coldest if coldest > 0 else float("inf")
        detail  = f"hottest/coldest = {hottest:.4f}/{coldest:.4f} = {ratio:.2f}x"
        if ratio > bal_fail:
            checks.append({"id": "layer_balance", "label": "Layer Balance",
                            "status": "fail", "detail": detail + f" (fail>{bal_fail}x)"})
        elif ratio > bal_warn:
            checks.append({"id": "layer_balance", "label": "Layer Balance",
                            "status": "warn", "detail": detail + f" (warn>{bal_warn}x)"})
        else:
            checks.append({"id": "layer_balance", "label": "Layer Balance",
                            "status": "pass", "detail": detail})
    else:
        checks.append({"id": "layer_balance", "label": "Layer Balance",
                        "status": "info",
                        "detail": "Insufficient active layers for balance check."})

    # ── Overall ───────────────────────────────────────────────────────────────
    statuses = [c["status"] for c in checks]
    if "fail" in statuses:
        overall = "fail"
    elif "warn" in statuses:
        overall = "warn"
    else:
        overall = "pass"

    # ── Metadata summary ──────────────────────────────────────────────────────
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    ratio_display = "?"
    if declared_alpha is not None and actual_rank:
        ratio_display = f"{declared_alpha / actual_rank:.4f}"

    meta_summary = {
        "filename":              os.path.basename(path),
        "size_mb":               f"{file_size_mb:.1f} MB",
        "model_type":            model_type.upper(),
        "rank":                  str(actual_rank) if actual_rank else "?",
        "alpha":                 f"{declared_alpha:.4g}" if declared_alpha is not None else "?",
        "ratio":                 ratio_display,
        "layers":                str(len(up_keys)),
        "dead":                  str(len(dead_layers)),
        "ss_base_model_version": metadata.get("ss_base_model_version", ""),
    }

    return {"overall": overall, "checks": checks, "meta": meta_summary,
            "model_type": model_type}


# ── Drop zone ─────────────────────────────────────────────────────────────────
class _DropZone(QFrame):
    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFixedHeight(54)
        self._set_idle()
        lbl = QLabel("or drag-and-drop a .safetensors file here", self)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(lbl)

    def _set_idle(self):
        self.setStyleSheet(
            f"QFrame {{ background:{BG}; border:2px dashed {MUT}; border-radius:6px; }}")

    def _set_active(self):
        self.setStyleSheet(
            f"QFrame {{ background:{CAR}; border:2px dashed {ACC}; border-radius:6px; }}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if path.lower().endswith(".safetensors"):
                event.acceptProposedAction()
                self._set_active()
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._set_idle()

    def dropEvent(self, event: QDropEvent):
        self._set_idle()
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(".safetensors"):
                self.file_dropped.emit(path)


# ── HealthPage ────────────────────────────────────────────────────────────────
class HealthPage(QWidget):
    """LoRA Health Analyzer — integrated page for Lora Training Suite 2.0."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{PAN};")

        self._cfg           = load_json(HEALTH_CFG, DEFAULTS)
        self._current_file: str | None           = None
        self._worker:       _AnalysisWorker | None = None
        self._thread:       threading.Thread | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────────────
        hdr = QFrame(self)
        hdr.setStyleSheet(f"background:{CAR}; border-bottom:2px solid {MUT};")
        hdr.setFixedHeight(56)
        hdr_row = QHBoxLayout(hdr)
        hdr_row.setContentsMargins(20, 0, 20, 0)
        hdr_row.setSpacing(0)

        title = QLabel("LoRA Health", hdr)
        title.setStyleSheet(
            f"color:{PRI}; font-family:{FONT}; font-size:{FONT_XL}px;"
            f" font-weight:bold; background:transparent;")
        hdr_row.addWidget(title)
        hdr_row.addStretch(1)

        sig = QLabel(SIGNATURE, hdr)
        sig.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        hdr_row.addWidget(sig)
        root.addWidget(hdr)

        # ── Tab widget ─────────────────────────────────────────────────────
        self._tabs = QTabWidget(self)
        self._tabs.setStyleSheet(
            f"QTabWidget::pane {{ border:none; background:{PAN}; }}"
            f"QTabBar::tab {{ background:{CAR}; color:{SEC}; border:none;"
            f" padding:8px 20px; font-family:{FONT}; font-size:{FONT_MD}px;"
            f" font-weight:bold; border-top-left-radius:6px;"
            f" border-top-right-radius:6px; margin-right:2px; }}"
            f"QTabBar::tab:selected {{ background:{ACC}; color:{PRI}; }}"
            f"QTabBar::tab:hover:!selected {{ background:{MUT}; color:{PRI}; }}"
        )
        root.addWidget(self._tabs, stretch=1)

        self._build_analyze_tab()
        self._build_thresholds_tab()
        self._build_help_tab()

    # ══════════════════════════════════════════════════════════════════════════
    # Analyze tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_analyze_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # ── File picker card ───────────────────────────────────────────────
        file_card = QFrame()
        file_card.setStyleSheet(_card_style())
        fl = QVBoxLayout(file_card)
        fl.setContentsMargins(16, 14, 16, 14)
        fl.setSpacing(10)

        fl.addWidget(_lbl("LoRA File", PRI, FONT_MD, bold=True))

        pick_row = QHBoxLayout()
        pick_row.setSpacing(8)
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("Browse or drag a .safetensors file here…")
        self._file_edit.setReadOnly(True)
        self._file_edit.setStyleSheet(
            f"QLineEdit {{ background:{BG}; color:{SEC}; border:2px solid {MUT};"
            f" border-radius:4px; padding:4px 8px; font-family:{FONT}; font-size:{FONT_MD}px; }}")
        pick_row.addWidget(self._file_edit, stretch=1)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedHeight(32)
        browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_btn.clicked.connect(self._on_browse)
        pick_row.addWidget(browse_btn)
        fl.addLayout(pick_row)

        drop_zone = _DropZone(self)
        drop_zone.file_dropped.connect(self._on_file_selected)
        fl.addWidget(drop_zone)

        # Model type + run row
        type_row = QHBoxLayout()
        type_row.setSpacing(10)
        type_row.addWidget(_lbl("Model Type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(["Auto-detect", "SD 1.5", "SDXL"])
        self._type_combo.setStyleSheet(_combo_style())
        self._type_combo.setFixedWidth(160)
        type_row.addWidget(self._type_combo)
        type_row.addStretch(1)

        self._run_btn = QPushButton("▶  Analyze")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setEnabled(False)
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.clicked.connect(self._on_run)
        type_row.addWidget(self._run_btn)
        fl.addLayout(type_row)

        layout.addWidget(file_card)

        # ── Results area ───────────────────────────────────────────────────
        self._results_widget = QWidget()
        self._results_widget.setStyleSheet("background:transparent;")
        self._results_layout = QVBoxLayout(self._results_widget)
        self._results_layout.setContentsMargins(0, 0, 0, 0)
        self._results_layout.setSpacing(12)
        layout.addWidget(self._results_widget)

        layout.addStretch(1)
        self._tabs.addTab(scroll, "Analyze")

    # ══════════════════════════════════════════════════════════════════════════
    # Thresholds tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_thresholds_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # ── Explanation card ───────────────────────────────────────────────
        exp_card = QFrame()
        exp_card.setStyleSheet(_card_style())
        el = QVBoxLayout(exp_card)
        el.setContentsMargins(16, 14, 16, 14)
        el.setSpacing(6)
        el.addWidget(_lbl("About These Thresholds", PRI, FONT_MD, bold=True))
        exp_text = QLabel(
            "Community-sourced defaults based on kohya-ss training guides, CivitAI wiki, "
            "LyCORIS benchmarks, and Ostris AI Toolkit documentation.\n\n"
            "Select a preset (Strict / Standard / Relaxed) per model type, or enter a value "
            "in the Manual Override column to pin a specific threshold. An override always "
            "takes priority over the preset — clear the field to return to preset behavior."
        )
        exp_text.setWordWrap(True)
        exp_text.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        el.addWidget(exp_text)
        layout.addWidget(exp_card)

        # ── Warning banner ─────────────────────────────────────────────────
        warn_card = QFrame()
        warn_card.setStyleSheet(
            f"QFrame {{ background:{CAR}; border-radius:8px; border:2px solid {AMB}; }}")
        wl = QHBoxLayout(warn_card)
        wl.setContentsMargins(14, 10, 14, 10)
        warn_lbl = QLabel(
            "⚠  Amber-bordered fields accept manual overrides. Changes are saved immediately.")
        warn_lbl.setStyleSheet(
            f"color:{AMB}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        warn_lbl.setWordWrap(True)
        wl.addWidget(warn_lbl)
        layout.addWidget(warn_card)

        # ── SD 1.5 section ─────────────────────────────────────────────────
        sd15_card, self._sd15_preset_combo, self._sd15_override_edits = \
            self._build_threshold_section("SD 1.5", "sd15")
        layout.addWidget(sd15_card)

        # ── SDXL section ───────────────────────────────────────────────────
        sdxl_card, self._sdxl_preset_combo, self._sdxl_override_edits = \
            self._build_threshold_section("SDXL", "sdxl")
        layout.addWidget(sdxl_card)

        layout.addStretch(1)
        self._tabs.addTab(scroll, "Thresholds")

    def _build_threshold_section(
        self, title: str, key: str
    ) -> tuple[QFrame, QComboBox, dict[str, QLineEdit]]:
        card = QFrame()
        card.setStyleSheet(_card_style())
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 14, 16, 16)
        cl.setSpacing(8)

        # Header row — title + preset selector
        hdr_row = QHBoxLayout()
        hdr_row.addWidget(_lbl(f"{title} Thresholds", PRI, FONT_LG, bold=True))
        hdr_row.addStretch(1)
        hdr_row.addWidget(_lbl("Preset:", SEC, FONT_SM))
        preset_combo = QComboBox()
        preset_combo.addItems(["Strict", "Standard", "Relaxed"])
        preset_combo.setCurrentText(self._cfg.get(f"{key}_preset", "Standard"))
        preset_combo.setStyleSheet(_combo_style())
        preset_combo.setFixedWidth(140)
        preset_combo.currentTextChanged.connect(
            lambda v, k=key: self._preset_changed(k, v))
        hdr_row.addWidget(preset_combo)
        cl.addLayout(hdr_row)

        # Column headers
        col_hdr = QFrame()
        col_hdr.setStyleSheet("background:transparent;")
        ch_row = QHBoxLayout(col_hdr)
        ch_row.setContentsMargins(8, 0, 8, 0)
        ch_row.addWidget(_lbl("Threshold", SEC, FONT_SM, bold=True), stretch=2)
        ch_row.addWidget(_lbl("Preset Default", SEC, FONT_SM, bold=True), stretch=1)
        ch_row.addWidget(_lbl("Manual Override", AMB, FONT_SM, bold=True), stretch=1)
        cl.addWidget(col_hdr)

        # Threshold rows
        edits: dict[str, QLineEdit] = {}
        overrides    = self._cfg.get(f"{key}_overrides", {})
        preset_name  = self._cfg.get(f"{key}_preset", "Standard")
        preset_vals  = THRESHOLD_PRESETS[key]

        for i, (tk, tlabel) in enumerate(THRESHOLD_LABELS.items()):
            row = QFrame()
            row.setStyleSheet(
                f"QFrame {{ background:{BG if i % 2 == 0 else CAR}; border-radius:4px; }}")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(8, 6, 8, 6)
            rl.setSpacing(8)

            rl.addWidget(_lbl(tlabel, PRI, FONT_SM), stretch=2)

            default_val = preset_vals.get(preset_name, preset_vals["Standard"])[tk]
            default_lbl = QLabel(f"{default_val:.6g}")
            default_lbl.setStyleSheet(
                f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
            rl.addWidget(default_lbl, stretch=1)

            ov_edit = QLineEdit(str(overrides.get(tk, "")))
            ov_edit.setPlaceholderText("override…")
            ov_edit.setStyleSheet(_amber_edit_style())
            ov_edit.setFixedWidth(110)
            ov_edit.textChanged.connect(
                lambda text, k=key, t=tk: self._override_changed(k, t, text))
            edits[tk] = ov_edit
            rl.addWidget(ov_edit, stretch=1)

            cl.addWidget(row)

        return card, preset_combo, edits

    # ══════════════════════════════════════════════════════════════════════════
    # Help tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_help_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        sections = [
            ("What Does LoRA Health Check?",
             "LoRA Health loads a .safetensors LoRA file and runs 8 structural checks without "
             "running inference. All checks are computed directly from tensor shapes, values, "
             "and embedded metadata."),
            ("1.  File Integrity",
             "Verifies that kohya/sd-scripts hash metadata (sshs_model_hash, sshs_legacy_hash) "
             "is present. Missing hashes may indicate the file was not saved by a standard "
             "trainer, but is otherwise harmless."),
            ("2.  NaN / Inf",
             "Scans every tensor for Not-a-Number and Infinity values. These are catastrophic — "
             "a LoRA with NaN or Inf values will produce corrupted or blank images."),
            ("3.  Rank Consistency",
             "Compares lora_down and lora_up tensor shapes to verify they agree on rank. "
             "Also checks against ss_network_dim in metadata if present."),
            ("4.  Alpha/Rank Ratio",
             "Computes declared_alpha / actual_rank. A ratio below 0.10 produces very weak "
             "effect; above 1.0 can cause color shifts and overexposure. Standard practice "
             "sets alpha = rank (ratio = 1.0) or alpha = rank/2 (ratio = 0.5)."),
            ("5.  Rank Range",
             "Checks whether the rank falls within community-validated bounds for the detected "
             "model type. Very low rank (<4) limits expressiveness; very high rank (>128 for "
             "SD1.5, >256 for SDXL) increases file size without meaningful quality gain."),
            ("6.  Dead Layers",
             "Identifies lora_up layers whose mean absolute value is below the dead threshold. "
             "Dead layers contribute nothing to the output and indicate undertrained modules "
             "or a training collapse."),
            ("7.  Overbaked",
             "Detects LoRAs trained to excess ('overcooked'). Mean lora_up magnitude above the "
             "warning threshold suggests aggressive training that may produce burned-in style "
             "with low prompt adherence. Above the fail threshold, the LoRA will likely "
             "saturate or destroy image quality."),
            ("8.  Layer Balance",
             "Computes max(lora_up mean_abs) / min(lora_up mean_abs) across all active layers. "
             "Important: a LoRA spans many different architectural modules (attention Q/K/V, "
             "cross-attention, MLP, text encoder, UNet at different depths) that naturally "
             "operate at very different weight scales. A ratio of 50x–500x is completely normal "
             "for a healthy LoRA — this is structural variation, not a problem. The check only "
             "warns above 200x and fails above 1000x (Standard preset), which indicates a "
             "catastrophic training collapse where one module is essentially dead while another "
             "is wildly overfit."),
            ("Threshold Presets",
             "Strict:    tight bounds — flags anything outside high-quality training range.\n"
             "Standard:  community-recommended defaults — practical for most LoRAs.\n"
             "Relaxed:   permissive — useful for experimental or artistic LoRAs.\n\n"
             "Set individual overrides in the Thresholds tab. Clear any override to return "
             "to the selected preset."),
            ("Model Type Detection",
             "Auto-detect checks for lora_te2_ keys (SDXL dual text encoder) and total "
             "lora_up layer count (>300 = SDXL). Use the Model Type dropdown to override "
             "if auto-detection is incorrect."),
        ]

        for title, body in sections:
            sec = QFrame()
            sec.setStyleSheet(_card_style())
            sl = QVBoxLayout(sec)
            sl.setContentsMargins(16, 12, 16, 12)
            sl.setSpacing(6)
            sl.addWidget(_lbl(title, ACC, FONT_MD, bold=True))
            body_lbl = QLabel(body)
            body_lbl.setWordWrap(True)
            body_lbl.setStyleSheet(
                f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
            sl.addWidget(body_lbl)
            layout.addWidget(sec)

        layout.addStretch(1)
        self._tabs.addTab(scroll, "Help")

    # ══════════════════════════════════════════════════════════════════════════
    # File selection
    # ══════════════════════════════════════════════════════════════════════════

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select LoRA File", "",
            "SafeTensors Files (*.safetensors);;All Files (*)")
        if path:
            self._on_file_selected(path)

    def _on_file_selected(self, path: str):
        self._current_file = path
        self._file_edit.setText(path)
        self._run_btn.setEnabled(True)

    # ══════════════════════════════════════════════════════════════════════════
    # Analysis
    # ══════════════════════════════════════════════════════════════════════════

    def _on_run(self):
        if not self._current_file or not os.path.isfile(self._current_file):
            return

        self._run_btn.setEnabled(False)
        self._run_btn.setText("Analyzing…")
        self._clear_results()

        type_sel = self._type_combo.currentText()
        forced_type: str | None = None
        if type_sel == "SD 1.5":
            forced_type = "sd15"
        elif type_sel == "SDXL":
            forced_type = "sdxl"

        if forced_type:
            get_fn = lambda mt, k: self.get_threshold(forced_type, k)
        else:
            get_fn = self.get_threshold

        self._worker = _AnalysisWorker(self._current_file, get_fn, forced_type)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def _on_analysis_done(self, result: dict):
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  Analyze")
        self._show_results(result)

    def _on_analysis_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  Analyze")
        err_lbl = _lbl(f"Error: {msg}", RED, FONT_MD)
        err_lbl.setWordWrap(True)
        self._results_layout.addWidget(err_lbl)

    def _clear_results(self):
        while self._results_layout.count():
            item = self._results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_results(self, result: dict):
        overall    = result["overall"]
        checks     = result["checks"]
        meta       = result["meta"]
        model_type = result["model_type"]

        # ── Overall badge ──────────────────────────────────────────────────
        badge_colors = {"pass": GRN,  "warn": AMB,   "fail": RED}
        badge_texts  = {"pass": "✓  PASS", "warn": "⚠  WARN", "fail": "✗  FAIL"}

        badge_row = QHBoxLayout()
        badge_row.setSpacing(16)

        badge = QLabel(badge_texts.get(overall, overall.upper()))
        badge.setStyleSheet(
            f"color:{PRI}; background:{badge_colors.get(overall, MUT)};"
            f" border-radius:6px; padding:6px 20px;"
            f" font-family:{FONT}; font-size:{FONT_LG}px; font-weight:bold;")
        badge_row.addWidget(badge)
        badge_row.addWidget(_lbl(f"Detected: {model_type.upper()}", SEC, FONT_MD))
        badge_row.addStretch(1)

        badge_w = QWidget()
        badge_w.setStyleSheet("background:transparent;")
        badge_w.setLayout(badge_row)
        self._results_layout.addWidget(badge_w)

        # ── Metadata card ──────────────────────────────────────────────────
        meta_card = QFrame()
        meta_card.setStyleSheet(_card_style())
        ml = QVBoxLayout(meta_card)
        ml.setContentsMargins(16, 12, 16, 12)
        ml.setSpacing(6)
        ml.addWidget(_lbl("File Metadata", PRI, FONT_MD, bold=True))

        fields = [
            ("File",      meta["filename"]),
            ("Size",      meta["size_mb"]),
            ("Model",     meta["model_type"]),
            ("Rank",      meta["rank"]),
            ("Alpha",     meta["alpha"]),
            ("α/r Ratio", meta["ratio"]),
            ("Layers",    meta["layers"]),
            ("Dead",      meta["dead"]),
        ]
        if meta.get("ss_base_model_version"):
            fields.append(("Base", meta["ss_base_model_version"]))

        grid = QGridLayout()
        grid.setSpacing(6)
        for i, (lk, lv) in enumerate(fields):
            col = (i % 2) * 2
            row = i // 2
            grid.addWidget(_lbl(f"{lk}:", SEC, FONT_SM), row, col)
            grid.addWidget(_lbl(lv, PRI, FONT_SM),       row, col + 1)
        ml.addLayout(grid)
        self._results_layout.addWidget(meta_card)

        # ── Check rows ─────────────────────────────────────────────────────
        checks_card = QFrame()
        checks_card.setStyleSheet(_card_style())
        cl = QVBoxLayout(checks_card)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(4)
        cl.addWidget(_lbl("Diagnostic Results", PRI, FONT_MD, bold=True))

        status_colors = {"pass": GRN, "warn": AMB, "fail": RED, "info": ACC}
        status_icons  = {"pass": "✓", "warn": "⚠", "fail": "✗", "info": "ℹ"}

        for i, check in enumerate(checks):
            row = QFrame()
            row.setStyleSheet(
                f"QFrame {{ background:{BG if i % 2 == 0 else CAR}; border-radius:4px; }}")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(8, 7, 8, 7)
            rl.setSpacing(10)

            sc = status_colors.get(check["status"], SEC)
            si = status_icons.get(check["status"], "?")

            icon_lbl = QLabel(si)
            icon_lbl.setFixedWidth(18)
            icon_lbl.setStyleSheet(
                f"color:{sc}; font-family:{FONT}; font-size:{FONT_MD}px;"
                f" font-weight:bold; background:transparent;")
            rl.addWidget(icon_lbl)

            name_lbl = _lbl(check["label"], PRI, FONT_MD, bold=True)
            name_lbl.setFixedWidth(160)
            rl.addWidget(name_lbl)

            detail_lbl = QLabel(check["detail"])
            detail_lbl.setWordWrap(True)
            detail_lbl.setStyleSheet(
                f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
            rl.addWidget(detail_lbl, stretch=1)

            cl.addWidget(row)

        self._results_layout.addWidget(checks_card)

    # ══════════════════════════════════════════════════════════════════════════
    # Threshold helpers
    # ══════════════════════════════════════════════════════════════════════════

    def get_threshold(self, model_type: str, key: str) -> float:
        """Return effective threshold: override if set and valid, else preset value."""
        ov = str(self._cfg.get(f"{model_type}_overrides", {}).get(key, "")).strip()
        if ov:
            try:
                v = float(ov)
                if v >= 0:
                    return v
            except ValueError:
                pass
        preset_name = self._cfg.get(f"{model_type}_preset", "Standard")
        presets     = THRESHOLD_PRESETS.get(model_type, THRESHOLD_PRESETS["sd15"])
        return presets.get(preset_name, presets["Standard"])[key]

    def _preset_changed(self, model_type: str, preset_name: str):
        self._cfg[f"{model_type}_preset"] = preset_name
        save_json(HEALTH_CFG, self._cfg)

    def _override_changed(self, model_type: str, threshold_key: str, text: str):
        overrides = self._cfg.setdefault(f"{model_type}_overrides", {})
        overrides[threshold_key] = text.strip()
        save_json(HEALTH_CFG, self._cfg)
