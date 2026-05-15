"""
health/health_page.py — LoRA Health Analyzer for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.01)

Loads a .safetensors LoRA file and runs 8 checks grouped by architectural module:

  File-level:
    1. File Integrity   — hash metadata present
    2. NaN / Inf        — tensor value sanity
    3. Rank Consistency — shape agreement and metadata match
    4. Alpha/Rank Ratio — declared alpha relative to rank
    5. Rank Range       — rank within community-validated bounds
    6. Overbaked        — global mean lora_up magnitude (overtrained signal)

  Per architectural module (UNet Cross-Attn, Self-Attn, Feedforward, Text Encoder):
    7. Dead Layers      — layers with near-zero weights within each module group
    8. Layer Balance    — hottest/coldest ratio within each module group
                         (comparing like-for-like — architecturally identical layers)
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


# ── Architectural module groups ────────────────────────────────────────────────
# Keys parsed from lora tensor names to classify each layer into its module role.
# Comparing layers within the same group is meaningful — they perform the same
# function at different network depths, so their magnitudes should be comparable.
MODULE_GROUPS = ["unet_cross_attn", "unet_self_attn", "unet_ff", "te"]

MODULE_LABELS: dict[str, str] = {
    "unet_cross_attn": "UNet Cross-Attention",
    "unet_self_attn":  "UNet Self-Attention",
    "unet_ff":         "UNet Feedforward",
    "te":              "Text Encoder",
    "unet_other":      "UNet Other",
}

_STATUS_ORDER = {"pass": 0, "info": 1, "warn": 2, "fail": 3}


def _max_status(a: str, b: str) -> str:
    return a if _STATUS_ORDER.get(a, 0) >= _STATUS_ORDER.get(b, 0) else b


def _classify_key(key: str) -> str:
    """Map a lora tensor key to an architectural module group."""
    k = key.lower()
    if "lora_unet_" in k:
        if "attn2" in k:
            return "unet_cross_attn"
        if "attn1" in k:
            return "unet_self_attn"
        if "ff_net" in k or "geglu" in k:
            return "unet_ff"
        return "unet_other"       # proj_in, proj_out, resnet layers
    return "te"                   # lora_te_, lora_te1_, lora_te2_


# ── Threshold presets ─────────────────────────────────────────────────────────
# Global thresholds: rank range, magnitude (overbaked), alpha/rank ratio.
# Per-module balance thresholds: hottest/coldest ratio within each module group.
# These are tighter than any global balance check because layers in the same
# group perform the same function — wide imbalance within a group is abnormal.
#
# Values derived from: kohya-ss guides, CivitAI wiki, LyCORIS benchmarks,
# Ostris AI Toolkit documentation, and community LoRA evaluation threads.

GLOBAL_THRESHOLD_LABELS: dict[str, str] = {
    "rank_min":  "Min Rank",
    "rank_max":  "Max Rank",
    "mag_dead":  "Dead Layer Threshold",
    "mag_warn":  "Overcooked Warning",
    "mag_fail":  "Overcooked Fail",
    "ratio_min": "Alpha/Rank Min",
    "ratio_max": "Alpha/Rank Max",
}

MODULE_BAL_LABELS: dict[str, str] = {
    "unet_cross_attn_bal_warn": "Cross-Attention  Warn",
    "unet_cross_attn_bal_fail": "Cross-Attention  Fail",
    "unet_self_attn_bal_warn":  "Self-Attention   Warn",
    "unet_self_attn_bal_fail":  "Self-Attention   Fail",
    "unet_ff_bal_warn":         "Feedforward      Warn",
    "unet_ff_bal_fail":         "Feedforward      Fail",
    "te_bal_warn":              "Text Encoder     Warn",
    "te_bal_fail":              "Text Encoder     Fail",
}

THRESHOLD_LABELS: dict[str, str] = {**GLOBAL_THRESHOLD_LABELS, **MODULE_BAL_LABELS}

# Overbaked (mag_warn / mag_fail):
#   lora-inspector community reference shows healthy LoRA global mean_abs ≈ 0.009–0.014.
#   Overtrained LoRAs typically show 3–10× that value before becoming visibly broken.
#   Source: rockerBOO/lora-inspector sample output from multiple community LoRAs.
#
# Layer balance (bal_warn / bal_fail):
#   Within a module group, per-layer mean_abs values realistically span ~0.002–0.06,
#   producing natural ratios of 20–50× even in healthy LoRAs. Cross-attention groups
#   also include to_q, to_k, to_v, to_out — some training configs intentionally leave
#   to_k/to_v near-zero, which looks like imbalance by design. Thresholds must be
#   permissive enough to ignore this normal variation and only flag pathological cases.
THRESHOLD_PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "sd15": {
        "Strict": {
            "rank_min":  8,    "rank_max":  32,
            "mag_dead":  1e-4, "mag_warn":  0.04,  "mag_fail": 0.08,
            "ratio_min": 0.25, "ratio_max": 1.0,
            "unet_cross_attn_bal_warn":  80.0, "unet_cross_attn_bal_fail": 300.0,
            "unet_self_attn_bal_warn":   60.0, "unet_self_attn_bal_fail":  200.0,
            "unet_ff_bal_warn":          60.0, "unet_ff_bal_fail":         200.0,
            "te_bal_warn":               40.0, "te_bal_fail":              150.0,
        },
        "Standard": {
            "rank_min":  4,    "rank_max":  64,
            "mag_dead":  1e-5, "mag_warn":  0.06,  "mag_fail": 0.12,
            "ratio_min": 0.10, "ratio_max": 1.0,
            "unet_cross_attn_bal_warn": 120.0, "unet_cross_attn_bal_fail": 500.0,
            "unet_self_attn_bal_warn":  100.0, "unet_self_attn_bal_fail":  400.0,
            "unet_ff_bal_warn":         100.0, "unet_ff_bal_fail":         400.0,
            "te_bal_warn":               60.0, "te_bal_fail":              250.0,
        },
        "Relaxed": {
            "rank_min":  2,    "rank_max": 128,
            "mag_dead":  1e-6, "mag_warn":  0.10,  "mag_fail": 0.20,
            "ratio_min": 0.05, "ratio_max": 2.0,
            "unet_cross_attn_bal_warn": 200.0, "unet_cross_attn_bal_fail": 1000.0,
            "unet_self_attn_bal_warn":  200.0, "unet_self_attn_bal_fail":  1000.0,
            "unet_ff_bal_warn":         200.0, "unet_ff_bal_fail":         1000.0,
            "te_bal_warn":              100.0, "te_bal_fail":               500.0,
        },
    },
    "sdxl": {
        "Strict": {
            "rank_min": 16,    "rank_max": 128,
            "mag_dead":  1e-4, "mag_warn":  0.04,  "mag_fail": 0.08,
            "ratio_min": 0.25, "ratio_max": 1.0,
            "unet_cross_attn_bal_warn":  80.0, "unet_cross_attn_bal_fail": 300.0,
            "unet_self_attn_bal_warn":   60.0, "unet_self_attn_bal_fail":  200.0,
            "unet_ff_bal_warn":          60.0, "unet_ff_bal_fail":         200.0,
            "te_bal_warn":               40.0, "te_bal_fail":              150.0,
        },
        "Standard": {
            "rank_min":  8,    "rank_max": 256,
            "mag_dead":  1e-5, "mag_warn":  0.06,  "mag_fail": 0.12,
            "ratio_min": 0.10, "ratio_max": 1.0,
            "unet_cross_attn_bal_warn": 120.0, "unet_cross_attn_bal_fail": 500.0,
            "unet_self_attn_bal_warn":  100.0, "unet_self_attn_bal_fail":  400.0,
            "unet_ff_bal_warn":         100.0, "unet_ff_bal_fail":         400.0,
            "te_bal_warn":               60.0, "te_bal_fail":              250.0,
        },
        "Relaxed": {
            "rank_min":  4,    "rank_max": 512,
            "mag_dead":  1e-6, "mag_warn":  0.10,  "mag_fail": 0.20,
            "ratio_min": 0.05, "ratio_max": 2.0,
            "unet_cross_attn_bal_warn": 200.0, "unet_cross_attn_bal_fail": 1000.0,
            "unet_self_attn_bal_warn":  200.0, "unet_self_attn_bal_fail":  1000.0,
            "unet_ff_bal_warn":         200.0, "unet_ff_bal_fail":         1000.0,
            "te_bal_warn":              100.0, "te_bal_fail":               500.0,
        },
    },
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


def _subsection_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{ACC}; font-family:{FONT}; font-size:{FONT_SM}px; font-weight:bold;"
        f" letter-spacing:1px; background:transparent;")
    return lbl


# ── Analysis worker ────────────────────────────────────────────────────────────
class _AnalysisWorker(QObject):
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, path: str, get_threshold_fn, model_type_override: str | None = None):
        super().__init__()
        self._path          = path
        self._get_fn        = get_threshold_fn
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
    """Load safetensors and run all checks. Returns result dict."""
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
            status, detail = "warn", f"{ratio_str} (below min {ratio_min})"
        elif ratio > ratio_max:
            status, detail = "warn", f"{ratio_str} (above max {ratio_max})"
        else:
            status, detail = "pass", ratio_str
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
            checks.append({"id": "rank_range", "label": "Rank Range", "status": "warn",
                            "detail": f"rank={actual_rank} below minimum {rank_min_t} for {model_type.upper()}"})
        elif actual_rank > rank_max_t:
            checks.append({"id": "rank_range", "label": "Rank Range", "status": "warn",
                            "detail": f"rank={actual_rank} above recommended {rank_max_t} for {model_type.upper()}"})
        else:
            checks.append({"id": "rank_range", "label": "Rank Range", "status": "pass",
                            "detail": f"rank={actual_rank} within [{rank_min_t}, {rank_max_t}] for {model_type.upper()}"})
    else:
        checks.append({"id": "rank_range", "label": "Rank Range",
                        "status": "info", "detail": "Could not determine rank."})

    # ── Group lora_up layers by architectural module ───────────────────────────
    mag_dead = get_threshold(model_type, "mag_dead")
    mag_warn = get_threshold(model_type, "mag_warn")
    mag_fail = get_threshold(model_type, "mag_fail")

    grouped: dict[str, list[tuple[str, float]]] = {}
    for uk in up_keys:
        mag = float(tensors[uk].abs().mean().item())
        mod = _classify_key(uk)
        grouped.setdefault(mod, []).append((uk, mag))

    # ── 6. Overbaked (global mean across all lora_up) ─────────────────────────
    all_mags = [m for layers in grouped.values() for _, m in layers]
    if all_mags:
        mean_mag = sum(all_mags) / len(all_mags)
        if mean_mag > mag_fail:
            checks.append({"id": "overbaked", "label": "Overbaked",
                            "status": "fail",
                            "detail": f"global mean magnitude={mean_mag:.4f} — exceeds fail threshold {mag_fail}"})
        elif mean_mag > mag_warn:
            checks.append({"id": "overbaked", "label": "Overbaked",
                            "status": "warn",
                            "detail": f"global mean magnitude={mean_mag:.4f} — exceeds warning threshold {mag_warn}"})
        else:
            checks.append({"id": "overbaked", "label": "Overbaked",
                            "status": "pass",
                            "detail": f"global mean magnitude={mean_mag:.4f} within normal range (warn>{mag_warn})"})
    else:
        checks.append({"id": "overbaked", "label": "Overbaked",
                        "status": "info", "detail": "No lora_up tensors found."})

    # ── 7-8. Per-module dead layers and balance ────────────────────────────────
    # Balance thresholds are per-module — layers of the same architectural role
    # should have similar magnitudes regardless of UNet depth.
    module_results: dict[str, dict] = {}
    display_order = MODULE_GROUPS + ["unet_other"]

    for mod in display_order:
        if mod not in grouped:
            continue
        layer_mags  = grouped[mod]
        mags        = [m for _, m in layer_mags]
        active_mags = [m for m in mags if m >= mag_dead]
        dead_count  = len(mags) - len(active_mags)
        mean_mag    = sum(mags) / len(mags) if mags else 0.0

        issues = []
        status = "pass"

        if dead_count > 0:
            status = _max_status(status, "warn")
            issues.append(f"{dead_count} dead layer(s)")

        balance_ratio: float | None = None
        if len(active_mags) >= 2:
            hottest = max(active_mags)
            coldest = min(active_mags)
            balance_ratio = hottest / coldest if coldest > 0 else float("inf")

            # Use per-module balance key; fall back to unet_self_attn for unet_other
            base = mod if mod in MODULE_GROUPS else "unet_self_attn"
            bal_w = get_threshold(model_type, f"{base}_bal_warn")
            bal_f = get_threshold(model_type, f"{base}_bal_fail")

            if balance_ratio > bal_f:
                status = _max_status(status, "fail")
                issues.append(f"balance {balance_ratio:.1f}x (fail>{bal_f:.0f}x)")
            elif balance_ratio > bal_w:
                status = _max_status(status, "warn")
                issues.append(f"balance {balance_ratio:.1f}x (warn>{bal_w:.0f}x)")

        module_results[mod] = {
            "label":       MODULE_LABELS.get(mod, mod),
            "count":       len(mags),
            "mean_abs":    mean_mag,
            "balance":     balance_ratio,
            "dead_count":  dead_count,
            "status":      status,
            "issues":      issues,
        }

    # ── Overall ───────────────────────────────────────────────────────────────
    all_statuses = [c["status"] for c in checks] + \
                   [g["status"] for g in module_results.values()]
    if "fail" in all_statuses:
        overall = "fail"
    elif "warn" in all_statuses:
        overall = "warn"
    else:
        overall = "pass"

    # ── File metadata summary ─────────────────────────────────────────────────
    ratio_display = "?"
    if declared_alpha is not None and actual_rank:
        ratio_display = f"{declared_alpha / actual_rank:.4f}"

    meta_summary = {
        "filename":              os.path.basename(path),
        "size_mb":               f"{os.path.getsize(path) / (1024 * 1024):.1f} MB",
        "model_type":            model_type.upper(),
        "rank":                  str(actual_rank) if actual_rank else "?",
        "alpha":                 f"{declared_alpha:.4g}" if declared_alpha is not None else "?",
        "ratio":                 ratio_display,
        "layers":                str(len(up_keys)),
        "ss_base_model_version": metadata.get("ss_base_model_version", ""),
    }

    return {
        "overall":       overall,
        "checks":        checks,
        "module_groups": module_results,
        "meta":          meta_summary,
        "model_type":    model_type,
    }


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
        self._current_file: str | None             = None
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

        # ── Tabs ───────────────────────────────────────────────────────────
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

        # Explanation
        exp_card = QFrame()
        exp_card.setStyleSheet(_card_style())
        el = QVBoxLayout(exp_card)
        el.setContentsMargins(16, 14, 16, 14)
        el.setSpacing(6)
        el.addWidget(_lbl("About These Thresholds", PRI, FONT_MD, bold=True))
        exp_text = QLabel(
            "Community-sourced defaults based on kohya-ss guides, CivitAI wiki, "
            "LyCORIS benchmarks, and Ostris AI Toolkit documentation.\n\n"
            "Global thresholds (rank, magnitude, alpha/rank) apply across the whole file. "
            "Per-module balance thresholds apply within each architectural group — "
            "comparing only layers that perform the same function, making the check "
            "structurally meaningful with appropriately tight bounds.\n\n"
            "Select a Strict / Standard / Relaxed preset per model type, or enter a value "
            "in the Manual Override column. An override always takes priority over the "
            "preset — clear the field to return to preset behavior."
        )
        exp_text.setWordWrap(True)
        exp_text.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        el.addWidget(exp_text)
        layout.addWidget(exp_card)

        # Warning banner
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

        sd15_card, self._sd15_preset_combo, self._sd15_override_edits = \
            self._build_threshold_section("SD 1.5", "sd15")
        layout.addWidget(sd15_card)

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

        # Header + preset selector
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

        edits: dict[str, QLineEdit] = {}
        overrides   = self._cfg.get(f"{key}_overrides", {})
        preset_name = self._cfg.get(f"{key}_preset", "Standard")
        preset_vals = THRESHOLD_PRESETS[key]

        # ── Global thresholds subsection ──────────────────────────────────
        cl.addWidget(_subsection_label("GLOBAL"))
        cl.addWidget(self._col_header_row())

        for i, (tk, tlabel) in enumerate(GLOBAL_THRESHOLD_LABELS.items()):
            cl.addWidget(self._threshold_row(
                i, tk, tlabel, key, overrides, preset_name, preset_vals, edits))

        # ── Per-module balance subsection ─────────────────────────────────
        cl.addSpacing(6)
        cl.addWidget(_subsection_label("LAYER BALANCE  (hottest / coldest within module group)"))
        cl.addWidget(self._col_header_row())

        # Pair rows: warn + fail side-by-side per module group
        mod_pairs = [
            ("unet_cross_attn", "Cross-Attention"),
            ("unet_self_attn",  "Self-Attention"),
            ("unet_ff",         "Feedforward"),
            ("te",              "Text Encoder"),
        ]
        row_idx = len(GLOBAL_THRESHOLD_LABELS)
        for mod_key, mod_label in mod_pairs:
            warn_tk = f"{mod_key}_bal_warn"
            fail_tk = f"{mod_key}_bal_fail"
            cl.addWidget(self._bal_pair_row(
                row_idx, mod_label, warn_tk, fail_tk,
                key, overrides, preset_name, preset_vals, edits))
            row_idx += 1

        return card, preset_combo, edits

    def _col_header_row(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        r = QHBoxLayout(w)
        r.setContentsMargins(8, 0, 8, 0)
        r.addWidget(_lbl("Threshold", SEC, FONT_SM, bold=True), stretch=2)
        r.addWidget(_lbl("Preset Default", SEC, FONT_SM, bold=True), stretch=1)
        r.addWidget(_lbl("Manual Override", AMB, FONT_SM, bold=True), stretch=1)
        return w

    def _threshold_row(self, idx: int, tk: str, tlabel: str, model_key: str,
                        overrides: dict, preset_name: str, preset_vals: dict,
                        edits: dict) -> QFrame:
        row = QFrame()
        row.setStyleSheet(
            f"QFrame {{ background:{BG if idx % 2 == 0 else CAR}; border-radius:4px; }}")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(8, 6, 8, 6)
        rl.setSpacing(8)

        rl.addWidget(_lbl(tlabel, PRI, FONT_SM), stretch=2)

        default_val = preset_vals.get(preset_name, preset_vals["Standard"])[tk]
        rl.addWidget(QLabel(f"{default_val:.6g}"), stretch=1)

        ov_edit = QLineEdit(str(overrides.get(tk, "")))
        ov_edit.setPlaceholderText("override…")
        ov_edit.setStyleSheet(_amber_edit_style())
        ov_edit.setFixedWidth(110)
        ov_edit.textChanged.connect(
            lambda text, k=model_key, t=tk: self._override_changed(k, t, text))
        edits[tk] = ov_edit
        rl.addWidget(ov_edit, stretch=1)
        return row

    def _bal_pair_row(self, idx: int, mod_label: str,
                       warn_tk: str, fail_tk: str, model_key: str,
                       overrides: dict, preset_name: str, preset_vals: dict,
                       edits: dict) -> QFrame:
        """Warn + Fail side-by-side in a single row for one module group."""
        row = QFrame()
        row.setStyleSheet(
            f"QFrame {{ background:{BG if idx % 2 == 0 else CAR}; border-radius:4px; }}")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(8, 6, 8, 6)
        rl.setSpacing(8)

        rl.addWidget(_lbl(mod_label, PRI, FONT_SM, bold=True), stretch=2)

        pv = preset_vals.get(preset_name, preset_vals["Standard"])

        # Warn
        warn_default = QLabel(f"warn: {pv[warn_tk]:.6g}")
        warn_default.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        rl.addWidget(warn_default, stretch=0)

        warn_edit = QLineEdit(str(overrides.get(warn_tk, "")))
        warn_edit.setPlaceholderText("warn…")
        warn_edit.setStyleSheet(_amber_edit_style())
        warn_edit.setFixedWidth(80)
        warn_edit.textChanged.connect(
            lambda text, k=model_key, t=warn_tk: self._override_changed(k, t, text))
        edits[warn_tk] = warn_edit
        rl.addWidget(warn_edit)

        # Fail
        fail_default = QLabel(f"fail: {pv[fail_tk]:.6g}")
        fail_default.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        rl.addWidget(fail_default, stretch=0)

        fail_edit = QLineEdit(str(overrides.get(fail_tk, "")))
        fail_edit.setPlaceholderText("fail…")
        fail_edit.setStyleSheet(_amber_edit_style())
        fail_edit.setFixedWidth(80)
        fail_edit.textChanged.connect(
            lambda text, k=model_key, t=fail_tk: self._override_changed(k, t, text))
        edits[fail_tk] = fail_edit
        rl.addWidget(fail_edit)

        return row

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
             "LoRA Health loads a .safetensors LoRA file and runs 8 checks without inference. "
             "Checks 1–6 are file-wide. Checks 7–8 (dead layers and layer balance) run "
             "per architectural module, so balance comparisons are structurally meaningful."),
            ("1.  File Integrity",
             "Verifies kohya/sd-scripts hash metadata (sshs_model_hash, sshs_legacy_hash) is "
             "present. Missing hashes are harmless but indicate the file was not produced by "
             "a standard kohya trainer."),
            ("2.  NaN / Inf",
             "Scans every tensor for Not-a-Number and Infinity values. These are catastrophic — "
             "a LoRA with NaN or Inf will produce corrupted or blank images."),
            ("3.  Rank Consistency",
             "Compares lora_down / lora_up tensor shapes to confirm rank agreement. Also checks "
             "the declared ss_network_dim metadata if present."),
            ("4.  Alpha/Rank Ratio",
             "Computes declared_alpha / actual_rank. Below 0.10 produces very weak effect; "
             "above 1.0 can cause colour shifts. Standard practice sets alpha = rank (ratio 1.0) "
             "or alpha = rank/2 (ratio 0.5)."),
            ("5.  Rank Range",
             "Validates rank against community bounds for the detected model type. Very low rank "
             "limits expressiveness; very high rank adds file size without quality gain."),
            ("6.  Overbaked",
             "Computes the global mean of all lora_up tensor mean-absolute values. Elevated "
             "values suggest aggressive or excessive training that may burn in style and reduce "
             "prompt responsiveness."),
            ("7.  Dead Layers  (per module)",
             "Within each architectural group, identifies lora_up layers whose mean absolute "
             "value is below the dead threshold. A dead layer contributes nothing to the output "
             "and indicates a collapsed or untrained module."),
            ("8.  Layer Balance  (per module)",
             "Computes max(mean_abs) / min(mean_abs) within each architectural group:\n\n"
             "  • UNet Cross-Attention  — attn2 layers (text conditioning)\n"
             "  • UNet Self-Attention   — attn1 layers (spatial relationships)\n"
             "  • UNet Feedforward      — ff_net layers (channel mixing)\n"
             "  • Text Encoder          — TE / TE1 / TE2 layers\n\n"
             "Layers in the same group perform the same function at different UNet depths, so "
             "wide magnitude imbalance within a group is genuinely abnormal — unlike a "
             "global comparison across architecturally incompatible modules.\n\n"
             "Standard warn/fail thresholds: 30x / 100x (much tighter than any global metric)."),
            ("Threshold Presets",
             "Strict:    tight bounds, flags anything outside high-quality training range.\n"
             "Standard:  community-recommended defaults, practical for most LoRAs.\n"
             "Relaxed:   permissive, useful for experimental or artistic LoRAs.\n\n"
             "Individual thresholds can be overridden in the Thresholds tab. "
             "Clear any override to return to the selected preset."),
            ("Model Type Detection",
             "Auto-detect checks for lora_te2_ keys (SDXL dual text encoder) and total "
             "lora_up layer count (>300 = SDXL). Use the Model Type dropdown if "
             "auto-detection is incorrect."),
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

        get_fn = (lambda mt, k: self.get_threshold(forced_type, k)) \
                 if forced_type else self.get_threshold

        self._worker = _AnalysisWorker(self._current_file, get_fn, forced_type)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def cleanup(self):
        """Disconnect any in-flight worker before the widget is deleted."""
        if self._worker is not None:
            try:
                self._worker.finished.disconnect()
                self._worker.error.disconnect()
            except RuntimeError:
                pass
            self._worker = None

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
        overall       = result["overall"]
        checks        = result["checks"]
        meta          = result["meta"]
        module_groups = result.get("module_groups", {})
        model_type    = result["model_type"]

        STATUS_COLORS = {"pass": GRN, "warn": AMB, "fail": RED, "info": ACC}
        STATUS_ICONS  = {"pass": "✓", "warn": "⚠", "fail": "✗", "info": "ℹ"}

        # ── Overall badge ──────────────────────────────────────────────────
        badge_colors = {"pass": GRN, "warn": AMB, "fail": RED}
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

        # ── File metadata card ─────────────────────────────────────────────
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
        ]
        if meta.get("ss_base_model_version"):
            fields.append(("Base", meta["ss_base_model_version"]))

        grid = QGridLayout()
        grid.setSpacing(6)
        for i, (lk, lv) in enumerate(fields):
            col = (i % 2) * 2
            row = i // 2
            grid.addWidget(_lbl(f"{lk}:", SEC, FONT_SM), row, col)
            grid.addWidget(_lbl(lv, PRI, FONT_SM), row, col + 1)
        ml.addLayout(grid)
        self._results_layout.addWidget(meta_card)

        # ── Structural checks card (file-level) ────────────────────────────
        struct_card = QFrame()
        struct_card.setStyleSheet(_card_style())
        sl = QVBoxLayout(struct_card)
        sl.setContentsMargins(16, 12, 16, 12)
        sl.setSpacing(4)
        sl.addWidget(_lbl("Structural Checks", PRI, FONT_MD, bold=True))

        for i, check in enumerate(checks):
            row = QFrame()
            row.setStyleSheet(
                f"QFrame {{ background:{BG if i % 2 == 0 else CAR}; border-radius:4px; }}")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(8, 7, 8, 7)
            rl.setSpacing(10)

            sc = STATUS_COLORS.get(check["status"], SEC)
            si = STATUS_ICONS.get(check["status"], "?")

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
            sl.addWidget(row)

        self._results_layout.addWidget(struct_card)

        # ── Module analysis card ───────────────────────────────────────────
        if module_groups:
            mod_card = QFrame()
            mod_card.setStyleSheet(_card_style())
            mcl = QVBoxLayout(mod_card)
            mcl.setContentsMargins(16, 12, 16, 12)
            mcl.setSpacing(4)
            mcl.addWidget(_lbl("Module Analysis", PRI, FONT_MD, bold=True))

            # Column header
            col_hdr = QWidget()
            col_hdr.setStyleSheet("background:transparent;")
            chr_ = QHBoxLayout(col_hdr)
            chr_.setContentsMargins(26, 0, 8, 0)
            chr_.addWidget(_lbl("Module", SEC, FONT_SM, bold=True), stretch=3)
            chr_.addWidget(_lbl("Layers", SEC, FONT_SM, bold=True), stretch=0)
            chr_.addWidget(_lbl("Mean abs", SEC, FONT_SM, bold=True), stretch=1)
            chr_.addWidget(_lbl("Balance", SEC, FONT_SM, bold=True), stretch=1)
            chr_.addWidget(_lbl("Notes", SEC, FONT_SM, bold=True), stretch=3)
            mcl.addWidget(col_hdr)

            display_order = MODULE_GROUPS + ["unet_other"]
            for i, mod in enumerate(display_order):
                if mod not in module_groups:
                    continue
                grp = module_groups[mod]

                row = QFrame()
                row.setStyleSheet(
                    f"QFrame {{ background:{BG if i % 2 == 0 else CAR}; border-radius:4px; }}")
                rl = QHBoxLayout(row)
                rl.setContentsMargins(8, 7, 8, 7)
                rl.setSpacing(10)

                sc = STATUS_COLORS.get(grp["status"], SEC)
                si = STATUS_ICONS.get(grp["status"], "?")
                icon_lbl = QLabel(si)
                icon_lbl.setFixedWidth(18)
                icon_lbl.setStyleSheet(
                    f"color:{sc}; font-family:{FONT}; font-size:{FONT_MD}px;"
                    f" font-weight:bold; background:transparent;")
                rl.addWidget(icon_lbl)

                rl.addWidget(_lbl(grp["label"], PRI, FONT_SM, bold=True), stretch=3)

                count_lbl = _lbl(str(grp["count"]), SEC, FONT_SM)
                count_lbl.setFixedWidth(40)
                rl.addWidget(count_lbl)

                rl.addWidget(_lbl(f"{grp['mean_abs']:.5f}", SEC, FONT_SM), stretch=1)

                if grp["balance"] is not None:
                    bal_str = f"{grp['balance']:.1f}x"
                    bal_color = RED if grp["status"] == "fail" else (
                        AMB if grp["status"] == "warn" else SEC)
                else:
                    bal_str, bal_color = "—", MUT
                rl.addWidget(_lbl(bal_str, bal_color, FONT_SM), stretch=1)

                notes = ", ".join(grp["issues"]) if grp["issues"] else "—"
                notes_lbl = QLabel(notes)
                notes_lbl.setWordWrap(True)
                notes_lbl.setStyleSheet(
                    f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;"
                    f" background:transparent;")
                rl.addWidget(notes_lbl, stretch=3)
                mcl.addWidget(row)

            self._results_layout.addWidget(mod_card)

    # ══════════════════════════════════════════════════════════════════════════
    # Threshold helpers
    # ══════════════════════════════════════════════════════════════════════════

    def get_threshold(self, model_type: str, key: str) -> float:
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
