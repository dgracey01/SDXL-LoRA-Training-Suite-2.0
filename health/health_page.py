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
import re
import statistics
import threading

from PySide6.QtCore    import Qt, QObject, Signal
from PySide6.QtGui     import QDragEnterEvent, QDropEvent, QWheelEvent, QPixmap
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QComboBox,
    QScrollArea, QTabWidget, QFileDialog, QMenu, QApplication,
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
    "sd15_preset":       "Standard",
    "sdxl_preset":       "Standard",
    "sd15_overrides":    {k: "" for k in THRESHOLD_LABELS},
    "sdxl_overrides":    {k: "" for k in THRESHOLD_LABELS},
    "model_type":        "Auto-detect",
    "trainer":           "Auto-detect",
    "batch_model_type":  "Auto-detect",
    "batch_trainer":     "Auto-detect",
    "batch_profile":     "Concept / Pose",
}


# ── Widget helpers ─────────────────────────────────────────────────────────────
def _lbl(text: str, color: str = SEC, size: int = FONT_MD, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{color}; font-family:{FONT}; font-size:{size}px;"
        f" font-weight:{'bold' if bold else 'normal'}; background:transparent; border:none;")
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

    def __init__(self, path: str, get_threshold_fn,
                 model_type_override: str | None = None,
                 trainer_override: str | None = None):
        super().__init__()
        self._path             = path
        self._get_fn           = get_threshold_fn
        self._type_override    = model_type_override
        self._trainer_override = trainer_override

    def run(self):
        try:
            result = _analyse(self._path, self._get_fn,
                              self._type_override, self._trainer_override)
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


def _detect_trainer(path: str, keys: list[str], metadata: dict) -> str:
    """Return 'aitoolkit' or 'kohya'. Checks config.yaml first, then tensor key heuristics."""
    folder = os.path.dirname(path)
    cfg_path = os.path.join(folder, "config.yaml")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as _f:
                content = _f.read()
            if "job: extension" in content or "diffusion_trainer" in content:
                return "aitoolkit"
        except Exception:
            pass

    # Heuristic: AI Toolkit produces conv LoRA keys, no kohya hash, no lora_te1_ prefix
    has_conv    = any("_conv.lora_down" in k or "_conv.lora_up" in k for k in keys)
    has_kohya_h = "sshs_model_hash" in metadata
    has_te1     = any("lora_te1_" in k for k in keys)
    if has_conv and not has_kohya_h and not has_te1:
        return "aitoolkit"

    return "kohya"


_LOG_STEP_RE    = re.compile(r'\|\s*(\d+)/(\d+)\s*\[.*?loss:\s*([\d.e+\-]+)\]')
_LOG_PREPROC_RE = re.compile(r'0/(\d+)\s*\[')
_LOG_SAVE_RE    = re.compile(r'^Saving at step (\d+)')
_CKPT_STEP_RE   = re.compile(r'_(\d{9})\.safetensors$')


def _get_checkpoint_step(filename: str) -> int | None:
    """Extract step number from AI Toolkit checkpoint filename, or None for the final file."""
    m = _CKPT_STEP_RE.search(filename)
    return int(m.group(1)) if m else None


def _parse_aitk_log(log_path: str) -> dict:
    """
    Parse an AI Toolkit log.txt.  Returns a dict with:
      image_count, total_steps, steps_per_image,
      q1_avg, q4_avg, late_cv, converged,
      checkpoint_losses {step: avg_loss},
      final_step, start_loss, end_loss.
    Returns {} on any failure.
    """
    try:
        steps: dict[int, float] = {}
        image_count:   int | None = None
        total_from_log: int | None = None
        save_steps:    list[int]  = []
        in_preprocess  = False

        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                # Image count — from the progress bar right after "Preprocessing image"
                if "Preprocessing image" in line:
                    in_preprocess = True
                if in_preprocess and image_count is None:
                    m = _LOG_PREPROC_RE.search(line)
                    if m and int(m.group(1)) > 10:   # filter 0/7 pipeline-load bars
                        image_count = int(m.group(1))
                        in_preprocess = False

                # Loss curve — last value written for each step wins (tqdm duplication)
                m = _LOG_STEP_RE.search(line)
                if m:
                    step  = int(m.group(1))
                    total = int(m.group(2))
                    loss  = float(m.group(3))
                    steps[step] = loss
                    if total_from_log is None:
                        total_from_log = total

                # Save events
                m = _LOG_SAVE_RE.match(line.strip())
                if m:
                    save_steps.append(int(m.group(1)))

        if not steps:
            return {}

        sorted_steps = sorted(steps.items())
        losses       = [l for _, l in sorted_steps]
        n            = len(losses)
        q            = max(1, n // 4)
        q4           = losses[3 * q:] or losses

        q1_avg  = sum(losses[:q]) / q
        q4_avg  = sum(q4) / len(q4)

        last500 = [l for s, l in sorted_steps if s >= max(steps) - 500]
        late_mean  = sum(last500) / len(last500) if last500 else 0.0
        late_stdev = statistics.stdev(last500) if len(last500) > 1 else 0.0
        late_cv    = late_stdev / late_mean if late_mean > 0 else 0.0

        def _window_avg(step: int, w: int = 50) -> float | None:
            nearby = [l for s, l in sorted_steps if abs(s - step) <= w]
            return sum(nearby) / len(nearby) if nearby else None

        final_step = max(steps)
        all_ckpt_steps = set(save_steps) | {final_step}
        checkpoint_losses = {s: _window_avg(s) for s in all_ckpt_steps}

        total_steps = total_from_log or final_step
        return {
            "image_count":       image_count,
            "total_steps":       total_steps,
            "steps_per_image":   total_steps / image_count if image_count else None,
            "start_loss":        losses[0],
            "end_loss":          losses[-1],
            "q1_avg":            q1_avg,
            "q4_avg":            q4_avg,
            "late_cv":           late_cv,
            "converged":         q4_avg < q1_avg * 0.97,
            "checkpoint_losses": checkpoint_losses,
            "final_step":        final_step,
        }
    except Exception:
        return {}


def _analyse(path: str, get_threshold, model_type_override: str | None = None,
             trainer_override: str | None = None) -> dict:
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

    all_keys   = list(tensors.keys())
    model_type = model_type_override or _detect_model_type(all_keys)
    trainer    = trainer_override    or _detect_trainer(path, all_keys, metadata)
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

            # AI Toolkit near-zeros to_k/to_v by design → cross-attn balance is a
            # structural artifact, not a training defect. Display the value but
            # do not flag it as warn/fail and do not count it toward the score.
            if trainer == "aitoolkit" and mod == "unet_cross_attn":
                issues.append(f"balance {balance_ratio:.1f}x (AI Toolkit — not scored)")
            else:
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

    _global_mag = (sum(all_mags) / len(all_mags)) if all_mags else 0.0
    _total_dead = sum(g["dead_count"] for g in module_results.values())
    # For AI Toolkit, exclude cross-attn balance from scoring — it's a structural artifact.
    _worst_bal  = max(
        (g["balance"] for mod_k, g in module_results.items()
         if g["balance"] is not None
         and not (trainer == "aitoolkit" and mod_k == "unet_cross_attn")),
        default=1.0,
    )

    # ── 9. Rank efficiency (SVD) ─────────────────────────────────────────────
    rank_efficiency = _compute_rank_efficiency(tensors, up_keys)

    # ── AI Toolkit log analysis ───────────────────────────────────────────────
    log_data: dict = {}
    if trainer == "aitoolkit":
        log_path = os.path.join(os.path.dirname(path), "log.txt")
        if os.path.isfile(log_path):
            log_data = _parse_aitk_log(log_path)
            if log_data:
                ckpt_step = _get_checkpoint_step(os.path.basename(path))
                if ckpt_step is None:
                    ckpt_step = log_data.get("final_step")
                log_data["this_checkpoint_step"] = ckpt_step
                log_data["this_checkpoint_loss"] = \
                    log_data.get("checkpoint_losses", {}).get(ckpt_step)

    # ── 10. Rank saturation ───────────────────────────────────────────────────
    mean_eff = rank_efficiency.get("mean_efficiency")
    if mean_eff is not None and mean_eff >= 0.88:
        plateaued = not log_data.get("converged", True)   # no log → assume unknown
        has_log   = bool(log_data)

        if has_log and plateaued:
            q1 = log_data.get("q1_avg", 0)
            q4 = log_data.get("q4_avg", 0)
            sat_detail = (
                f"Rank efficiency {mean_eff:.0%} with loss plateau confirmed "
                f"(Q1 {q1:.4f} → Q4 {q4:.4f}) — declared rank is a bottleneck. "
                f"Retrain at higher rank for better detail capture.")
            sat_status = "warn"
        elif mean_eff >= 0.93:
            sat_detail = (
                f"Rank efficiency {mean_eff:.0%} — nearly all rank dimensions in use. "
                f"Likely bottleneck; loss log unavailable to confirm plateau.")
            sat_status = "warn"
        else:
            sat_detail = (
                f"Rank efficiency {mean_eff:.0%} — rank dimensions heavily utilized. "
                f"Monitor loss trend; higher rank may improve detail capture.")
            sat_status = "info"

        checks.append({"id": "rank_saturation", "label": "Rank Saturation",
                       "status": sat_status, "detail": sat_detail})

        # Update overall to include the new check
        all_statuses = ([c["status"] for c in checks] +
                        [g["status"] for g in module_results.values()])
        overall = ("fail" if "fail" in all_statuses else
                   "warn" if "warn" in all_statuses else "pass")

    return {
        "overall":         overall,
        "checks":          checks,
        "module_groups":   module_results,
        "meta":            meta_summary,
        "model_type":      model_type,
        "trainer":         trainer,
        "log_data":        log_data,
        "rank_efficiency": rank_efficiency,
        "metrics": {
            "mean_mag":        _global_mag,
            "total_dead":      _total_dead,
            "worst_balance":   _worst_bal,
            "mean_efficiency": rank_efficiency.get("mean_efficiency"),
        },
    }


# ── Rank efficiency via SVD ───────────────────────────────────────────────────

def _compute_rank_efficiency(tensors: dict, up_keys: list[str],
                             threshold: float = 0.01) -> dict:
    """SVD-based rank efficiency per layer and per module group.

    For each lora_down [r, in], computes singular values and counts how many
    exceed 1% of the largest one.  effective_rank / declared_rank = efficiency.

    Returns {"mean_efficiency": float, "groups": {mod: float}} or {} on failure.
    Uses lora_down only (smallest matrix in the pair; sufficient for inter-
    checkpoint comparison where rank and architecture are identical).
    """
    import torch

    layer_effs: list[float]             = []
    group_effs: dict[str, list[float]]  = {}
    spectral_samples: list[tuple]       = []

    for uk in up_keys:
        dk = uk.replace("lora_up", "lora_down")
        if dk not in tensors:
            continue
        down = tensors[dk]
        if down.ndim != 2:
            continue
        r = down.shape[0]
        if r < 2:
            continue
        try:
            S     = torch.linalg.svdvals(down.float())
            s_max = S[0].item()
            if s_max == 0:
                continue
            eff_rank   = int((S > s_max * threshold).sum().item())
            efficiency = eff_rank / r
            mod = _classify_key(uk)
            group_effs.setdefault(mod, []).append(efficiency)
            layer_effs.append(efficiency)

            # Spectral energy quartiles — cumulative variance at 25/50/75% of rank dims
            energies = S.pow(2)
            total_e  = energies.sum().item()
            if total_e > 0:
                cum = (energies.cumsum(0) / total_e)
                q1i = max(0, r // 4 - 1)
                q2i = max(0, r // 2 - 1)
                q3i = max(0, (3 * r) // 4 - 1)
                spectral_samples.append((cum[q1i].item(), cum[q2i].item(), cum[q3i].item()))
        except Exception:
            continue

    if not layer_effs:
        return {}

    spectral = None
    if spectral_samples:
        def _avg(i): return sum(s[i] for s in spectral_samples) / len(spectral_samples)
        spectral = {"q1": _avg(0), "q2": _avg(1), "q3": _avg(2)}

    return {
        "mean_efficiency": sum(layer_effs) / len(layer_effs),
        "groups": {mod: sum(e) / len(e) for mod, e in group_effs.items()},
        "spectral": spectral,
    }


# ── Plain-language rank verdict ───────────────────────────────────────────────

def _rank_verdict(rank_efficiency: dict, log_data: dict | None = None,
                  declared_rank: int | None = None) -> tuple:
    """Return (verdict_text, color) or (None, None) if data unavailable.

    Decision tree:
      - Saturated (≥88%) + steep spectrum (Q2 ≥ 90%) → Balanced rank
      - Saturated (≥88%) + declared rank ≥ 128        → Balanced rank
        (at practical ceiling — 256 is not a meaningful recommendation)
      - Saturated (≥88%) + flat/unknown spectrum       → May benefit from higher rank
      - Mid-range (45–87%)                             → Balanced rank
      - Low utilization (<45%)                         → May benefit from lower rank
    """
    mean_eff = rank_efficiency.get("mean_efficiency")
    if mean_eff is None:
        return None, None
    spectral = rank_efficiency.get("spectral") or {}
    q2 = spectral.get("q2")
    if mean_eff >= 0.88:
        if (q2 is not None and q2 >= 0.90) or (declared_rank is not None and declared_rank >= 128):
            return "Balanced rank", GRN
        return "May benefit from higher rank", AMB
    if mean_eff >= 0.45:
        return "Balanced rank", GRN
    return "May benefit from lower rank", MUT


# ── Batch candidate scoring ────────────────────────────────────────────────────
def _score_result(result: dict, profile: str = "concept",
                  mag_warn: float = 0.06, mag_fail: float = 0.12,
                  checkpoint_step: int | None = None,
                  total_steps: int | None = None) -> float:
    """Penalty score for batch ranking — lower is better.
    Returns infinity for NaN/Inf failures or overbaked LoRAs in non-concept profiles."""
    for check in result["checks"]:
        if check["id"] == "nan_inf" and check["status"] == "fail":
            return float("inf")

    m        = result.get("metrics", {})
    mean_mag = m.get("mean_mag", 0.0)
    dead     = m.get("total_dead", 0)

    # step_ratio shared by identity / outfit / style
    step_ratio = 0.5
    if checkpoint_step is not None and total_steps and total_steps > 0:
        step_ratio = min(1.0, checkpoint_step / total_steps)

    mag_frac = min(1.0, mean_mag / mag_warn) if mag_warn > 0 else 0.0
    eff      = m.get("mean_efficiency")   # None if SVD was unavailable

    if profile == "identity":
        # Hard disqualify: overbaked LoRA overpowers other LoRAs
        if mean_mag >= mag_fail:
            return float("inf")
        # Prefer: latest checkpoint, highest safe magnitude, highest rank efficiency
        # step 0→100  ·  mag 0 (at warn)→40 (near zero)  ·  eff 0%→30  ·  dead 15 each
        eff_penalty = (1.0 - eff) * 30 if eff is not None else 15.0
        return (1.0 - step_ratio) * 100 + (1.0 - mag_frac) * 40 + dead * 15 + eff_penalty

    if profile == "outfit":
        # Hard disqualify: overbaked outfit bleeds into skin/hair/character
        if mean_mag >= mag_fail:
            return float("inf")
        # Prefer: moderate-to-late steps, magnitude sweet spot ~65% of warn, good efficiency
        sweet       = 0.65 * mag_warn
        mag_dev     = abs(mean_mag - sweet) / mag_warn if mag_warn > 0 else 0.0
        eff_penalty = (1.0 - eff) * 20 if eff is not None else 10.0
        return (1.0 - step_ratio) * 60 + mag_dev * 50 + dead * 12 + eff_penalty

    if profile == "style":
        # Hard disqualify: overbaked style flattens all outputs
        if mean_mag >= mag_fail:
            return float("inf")
        # Prefer: lower magnitude (subtle influence), mild step preference, decent efficiency
        eff_penalty = (1.0 - eff) * 15 if eff is not None else 7.5
        return (1.0 - step_ratio) * 25 + mag_frac * 60 + dead * 10 + eff_penalty

    # ── Concept / Pose (default) ──────────────────────────────────────────────
    # Concepts and poses are learned quickly; step count is not decisive.
    # Penalise fail/warn checks and magnitude continuously.
    score   = 0.0
    penalty = {"fail": 100, "warn": 20}
    boost   = {"overbaked": 2.0, "rank_consistency": 3.0}
    for check in result["checks"]:
        if check["id"] == "nan_inf":
            continue
        score += penalty.get(check["status"], 0) * boost.get(check["id"], 1.0)
    score += mean_mag * 500   # 0.06 warn ≈ +30 pts
    score += dead * 10
    worst  = m.get("worst_balance", 1.0) or 1.0
    if worst != float("inf"):
        score += max(0.0, worst - 50.0) * 0.3
    score += (1.0 - eff) * 25 if eff is not None else 0.0   # rank efficiency penalty
    return score


# ── Sample image strip ────────────────────────────────────────────────────────

def _read_sample_prompts(lora_folder: str) -> list[str]:
    """Return the ordered prompt list from AI Toolkit config.yaml, or [] if unavailable."""
    cfg_path = os.path.join(lora_folder, "config.yaml")
    if not os.path.isfile(cfg_path):
        return []
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        samples = (cfg.get("config", {})
                      .get("process", [{}])[0]
                      .get("sample", {})
                      .get("samples", []))
        return [str(s.get("prompt", "")) for s in samples if isinstance(s, dict)]
    except Exception:
        return []


def _sample_index(filename: str) -> int | None:
    """Extract the prompt index from an AI Toolkit sample filename (_N before ext)."""
    m = re.search(r"_(\d+)\.[^.]+$", filename)
    return int(m.group(1)) if m else None


class _SampleThumb(QFrame):
    """One thumbnail cell in the samples strip.
    Left-click opens full image; right-click shows the sample prompt."""

    def __init__(self, path: str, size: int, prompt: str = "", parent=None):
        super().__init__(parent)
        self._path   = path
        self._prompt = prompt
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame {{ background:{CAR}; border-radius:6px; border:1px solid {MUT}; }}"
            f"QFrame:hover {{ border-color:{ACC}; }}")
        self.setFixedSize(size + 10, size + 10)
        if prompt:
            self.setToolTip(prompt)

        v = QVBoxLayout(self)
        v.setContentsMargins(5, 5, 5, 5)
        v.setSpacing(0)

        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFixedSize(size, size)
        lbl.setStyleSheet("background:transparent; border:none;")
        px = QPixmap(path)
        if not px.isNull():
            px = px.scaled(size, size,
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        lbl.setPixmap(px)
        v.addWidget(lbl)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            os.startfile(self._path)
        super().mousePressEvent(e)

    def contextMenuEvent(self, e):
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background:{CAR}; color:{PRI}; border:1px solid {MUT};"
            f" font-family:{FONT}; font-size:{FONT_SM}px; }}"
            f"QMenu::item:selected {{ background:{ACC}; color:{BG}; }}")
        if self._prompt:
            prompt_action = menu.addAction(self._prompt)
            prompt_action.setEnabled(False)
            menu.addSeparator()
            copy_action = menu.addAction("Copy Prompt")
            copy_action.triggered.connect(
                lambda: QApplication.clipboard().setText(self._prompt))
        else:
            menu.addAction("No prompt metadata available").setEnabled(False)
        menu.exec(e.globalPos())


class _SampleStrip(QFrame):
    """Horizontally scrollable row of checkpoint sample images.
    Scroll wheel over the strip zooms all thumbnails uniformly."""

    _MIN = 80
    _MAX = 400
    _DEF = 160

    def __init__(self, samples_dir: str | None, parent=None):
        super().__init__(parent)
        self._samples_dir = samples_dir
        self._size        = self._DEF
        # Prompts indexed by position: loaded from config.yaml in the parent folder
        lora_folder = os.path.dirname(samples_dir) if samples_dir else None
        self._prompts: list[str] = _read_sample_prompts(lora_folder) if lora_folder else []

        self.setStyleSheet(
            f"QFrame {{ background:{BG}; border-top:1px solid {MUT}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 8, 16, 8)
        outer.setSpacing(4)

        self._header = _lbl("Samples", MUT, FONT_SM)
        outer.addWidget(self._header)

        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        self._scroll.setFixedHeight(self._DEF + 22)

        self._inner = QWidget()
        self._inner.setStyleSheet("background:transparent;")
        self._row = QHBoxLayout(self._inner)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(8)
        self._row.addStretch(1)
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll)

    def load_for_step(self, step: int | None, label: str = ""):
        while self._row.count() > 1:
            item = self._row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        hdr = f"Samples  —  {label}" if label else "Samples"
        if not self._samples_dir:
            self._header.setText(f"{hdr}  (no samples folder)")
            return

        paths = self._images_for_step(step)[:10]   # AI Toolkit max ~10 prompts
        if not paths:
            self._header.setText(f"{hdr}  (no images for this checkpoint)")
            return

        self._header.setText(hdr)
        for path in paths:
            idx    = _sample_index(os.path.basename(path))
            prompt = self._prompts[idx] if idx is not None and idx < len(self._prompts) else ""
            self._row.insertWidget(self._row.count() - 1,
                                   _SampleThumb(path, self._size, prompt))

    def _images_for_step(self, step: int | None) -> list[str]:
        if step is not None:
            pattern = f"__{step:09d}_"
        else:
            # Final LoRA — find the highest step present in the samples folder
            import re as _re
            best, pat = None, _re.compile(r"__(\d{9})_")
            for f in os.listdir(self._samples_dir):
                m = pat.search(f)
                if m:
                    n = int(m.group(1))
                    if best is None or n > best:
                        best = n
            if best is None:
                return []
            pattern = f"__{best:09d}_"

        return sorted(
            os.path.join(self._samples_dir, f)
            for f in os.listdir(self._samples_dir)
            if pattern in f and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )

    def _resize_thumbs(self):
        paths = [
            self._row.itemAt(i).widget()._path
            for i in range(self._row.count() - 1)
            if self._row.itemAt(i).widget() and
               isinstance(self._row.itemAt(i).widget(), _SampleThumb)
        ]
        while self._row.count() > 1:
            item = self._row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._scroll.setFixedHeight(self._size + 22)
        for path in paths:
            idx    = _sample_index(os.path.basename(path))
            prompt = self._prompts[idx] if idx is not None and idx < len(self._prompts) else ""
            self._row.insertWidget(self._row.count() - 1,
                                   _SampleThumb(path, self._size, prompt))

    def wheelEvent(self, e: QWheelEvent):
        step = 16
        if e.angleDelta().y() > 0:
            self._size = min(self._MAX, self._size + step)
        else:
            self._size = max(self._MIN, self._size - step)
        self._resize_thumbs()
        e.accept()


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


# ── Batch analysis worker ──────────────────────────────────────────────────────
class _BatchWorker(QObject):
    progress = Signal(int, int, str)  # index (1-based), total, current filename
    finished = Signal(list)           # list of {path, result, error}

    def __init__(self, paths: list, get_threshold_fn, trainer_override: str | None = None):
        super().__init__()
        self._paths            = paths
        self._get_fn           = get_threshold_fn
        self._trainer_override = trainer_override
        self._stop             = False

    def stop(self):
        self._stop = True

    def run(self):
        out   = []
        total = len(self._paths)
        for i, path in enumerate(self._paths):
            if self._stop:
                break
            self.progress.emit(i + 1, total, os.path.basename(path))
            try:
                out.append({"path": path,
                            "result": _analyse(path, self._get_fn,
                                               trainer_override=self._trainer_override),
                            "error": None})
            except Exception as exc:
                out.append({"path": path, "result": None, "error": str(exc)})
        self.finished.emit(out)


# ── HealthPage ────────────────────────────────────────────────────────────────
class HealthPage(QWidget):
    """LoRA Health Analyzer — integrated page for Lora Training Suite 2.0."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{PAN};")

        self._cfg           = load_json(HEALTH_CFG, DEFAULTS)
        self._current_file: str | None              = None
        self._worker:       _AnalysisWorker | None  = None
        self._thread:       threading.Thread | None = None
        self._batch_worker: _BatchWorker | None     = None
        self._batch_thread: threading.Thread | None = None
        self._sample_strip: _SampleStrip | None     = None

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
        self._build_batch_tab()
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
        self._type_combo.setCurrentText(self._cfg.get("model_type", "Auto-detect"))
        self._type_combo.setStyleSheet(_combo_style())
        self._type_combo.setFixedWidth(160)
        self._type_combo.currentTextChanged.connect(
            lambda v: self._save_combo("model_type", v))
        type_row.addWidget(self._type_combo)
        type_row.addSpacing(16)
        type_row.addWidget(_lbl("Trainer:"))
        self._trainer_combo = QComboBox()
        self._trainer_combo.addItems(["Auto-detect", "AI Toolkit", "Kohya / kohya-ss"])
        self._trainer_combo.setCurrentText(self._cfg.get("trainer", "Auto-detect"))
        self._trainer_combo.setStyleSheet(_combo_style())
        self._trainer_combo.setFixedWidth(180)
        self._trainer_combo.currentTextChanged.connect(
            lambda v: self._save_combo("trainer", v))
        type_row.addWidget(self._trainer_combo)
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
    # Batch Compare tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_batch_tab(self):
        from PySide6.QtWidgets import QProgressBar, QSizePolicy

        w = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # ── Folder picker card ─────────────────────────────────────────────
        folder_card = QFrame()
        folder_card.setStyleSheet(_card_style())
        fl = QVBoxLayout(folder_card)
        fl.setContentsMargins(16, 14, 16, 14)
        fl.setSpacing(10)

        fl.addWidget(_lbl("Batch Candidate Compare", PRI, FONT_MD, bold=True))
        fl.addWidget(_lbl(
            "Point to an output folder from a training run. Every .safetensors file "
            "found will be analysed and scored. The best candidate is highlighted.",
            SEC, FONT_SM))

        pick_row = QHBoxLayout()
        pick_row.setSpacing(8)
        self._batch_folder_edit = QLineEdit()
        self._batch_folder_edit.setPlaceholderText("Browse to training output folder…")
        self._batch_folder_edit.setReadOnly(True)
        self._batch_folder_edit.setStyleSheet(
            f"QLineEdit {{ background:{BG}; color:{SEC}; border:2px solid {MUT};"
            f" border-radius:4px; padding:4px 8px; font-family:{FONT}; font-size:{FONT_MD}px; }}")
        pick_row.addWidget(self._batch_folder_edit, stretch=1)

        batch_browse_btn = QPushButton("Browse…")
        batch_browse_btn.setFixedHeight(32)
        batch_browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        batch_browse_btn.clicked.connect(self._on_batch_browse)
        pick_row.addWidget(batch_browse_btn)
        fl.addLayout(pick_row)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)
        ctrl_row.addWidget(_lbl("Model Type:"))
        self._batch_type_combo = QComboBox()
        self._batch_type_combo.addItems(["Auto-detect", "SD 1.5", "SDXL"])
        self._batch_type_combo.setCurrentText(self._cfg.get("batch_model_type", "Auto-detect"))
        self._batch_type_combo.setStyleSheet(_combo_style())
        self._batch_type_combo.setFixedWidth(160)
        self._batch_type_combo.currentTextChanged.connect(
            lambda v: self._save_combo("batch_model_type", v))
        ctrl_row.addWidget(self._batch_type_combo)
        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(_lbl("Trainer:"))
        self._batch_trainer_combo = QComboBox()
        self._batch_trainer_combo.addItems(["Auto-detect", "AI Toolkit", "Kohya / kohya-ss"])
        self._batch_trainer_combo.setCurrentText(self._cfg.get("batch_trainer", "Auto-detect"))
        self._batch_trainer_combo.setStyleSheet(_combo_style())
        self._batch_trainer_combo.setFixedWidth(180)
        self._batch_trainer_combo.currentTextChanged.connect(
            lambda v: self._save_combo("batch_trainer", v))
        ctrl_row.addWidget(self._batch_trainer_combo)
        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(_lbl("Profile:"))
        self._batch_profile_combo = QComboBox()
        self._batch_profile_combo.addItems([
            "Concept / Pose", "Character / Identity", "Outfit / Costume", "Style / Art Direction"])
        self._batch_profile_combo.setCurrentText(self._cfg.get("batch_profile", "Concept / Pose"))
        self._batch_profile_combo.setStyleSheet(_combo_style())
        self._batch_profile_combo.setFixedWidth(200)
        self._batch_profile_combo.currentTextChanged.connect(
            lambda v: self._save_combo("batch_profile", v))
        ctrl_row.addWidget(self._batch_profile_combo)
        ctrl_row.addStretch(1)

        self._batch_run_btn = QPushButton("▶  Compare All")
        self._batch_run_btn.setFixedHeight(36)
        self._batch_run_btn.setEnabled(False)
        self._batch_run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._batch_run_btn.clicked.connect(self._on_batch_run)
        ctrl_row.addWidget(self._batch_run_btn)
        fl.addLayout(ctrl_row)
        layout.addWidget(folder_card)

        # ── Progress area ──────────────────────────────────────────────────
        prog_card = QFrame()
        prog_card.setStyleSheet(_card_style())
        prog_card.setVisible(False)
        pl = QVBoxLayout(prog_card)
        pl.setContentsMargins(16, 12, 16, 12)
        pl.setSpacing(8)

        prog_top = QHBoxLayout()
        self._batch_prog_lbl = _lbl("Preparing…", SEC, FONT_SM)
        prog_top.addWidget(self._batch_prog_lbl, stretch=1)
        self._batch_cancel_btn = QPushButton("Cancel")
        self._batch_cancel_btn.setFixedHeight(28)
        self._batch_cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._batch_cancel_btn.clicked.connect(self._on_batch_cancel)
        prog_top.addWidget(self._batch_cancel_btn)
        pl.addLayout(prog_top)

        self._batch_progress_bar = QProgressBar()
        self._batch_progress_bar.setRange(0, 100)
        self._batch_progress_bar.setValue(0)
        self._batch_progress_bar.setFixedHeight(10)
        self._batch_progress_bar.setTextVisible(False)
        self._batch_progress_bar.setStyleSheet(
            f"QProgressBar {{ background:{BG}; border:none; border-radius:5px; }}"
            f"QProgressBar::chunk {{ background:{ACC}; border-radius:5px; }}")
        pl.addWidget(self._batch_progress_bar)
        self._batch_prog_card = prog_card
        layout.addWidget(prog_card)

        # ── Results area ───────────────────────────────────────────────────
        self._batch_results_widget = QWidget()
        self._batch_results_widget.setStyleSheet("background:transparent;")
        self._batch_results_layout = QVBoxLayout(self._batch_results_widget)
        self._batch_results_layout.setContentsMargins(0, 0, 0, 0)
        self._batch_results_layout.setSpacing(12)
        layout.addWidget(self._batch_results_widget)

        layout.addStretch(1)
        self._tabs.addTab(scroll, "Batch Compare")

    # ── Batch slots ────────────────────────────────────────────────────────────

    def _on_batch_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Training Output Folder")
        if not folder:
            return
        self._batch_folder_edit.setText(folder)
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".safetensors")
        ]
        count = len(files)
        self._batch_run_btn.setEnabled(count > 0)
        if count == 0:
            self._batch_run_btn.setText("▶  Compare All")
            self._batch_folder_edit.setStyleSheet(
                self._batch_folder_edit.styleSheet().replace(MUT, RED))
        else:
            self._batch_run_btn.setText(f"▶  Compare All  ({count} files)")
            self._batch_folder_edit.setStyleSheet(
                f"QLineEdit {{ background:{BG}; color:{SEC}; border:2px solid {MUT};"
                f" border-radius:4px; padding:4px 8px; font-family:{FONT}; font-size:{FONT_MD}px; }}")

    def _on_batch_run(self):
        folder = self._batch_folder_edit.text()
        if not folder or not os.path.isdir(folder):
            return
        files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(".safetensors")]
        )
        if not files:
            return

        self._batch_run_btn.setEnabled(False)
        self._batch_run_btn.setText("Analyzing…")
        self._batch_prog_card.setVisible(True)
        self._batch_progress_bar.setValue(0)
        self._batch_prog_lbl.setText(f"Starting — {len(files)} files…")
        self._clear_batch_results()

        type_sel    = self._batch_type_combo.currentText()
        forced_type = {"SD 1.5": "sd15", "SDXL": "sdxl"}.get(type_sel)
        get_fn      = (lambda mt, k: self.get_threshold(forced_type, k)) \
                      if forced_type else self.get_threshold

        trainer_sel    = self._batch_trainer_combo.currentText()
        forced_trainer = {"AI Toolkit": "aitoolkit",
                          "Kohya / kohya-ss": "kohya"}.get(trainer_sel)

        self._batch_worker = _BatchWorker(files, get_fn, forced_trainer)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_thread = threading.Thread(target=self._batch_worker.run, daemon=True)
        self._batch_thread.start()

    def _on_batch_cancel(self):
        if self._batch_worker is not None:
            self._batch_worker.stop()
        self._batch_prog_card.setVisible(False)
        folder = self._batch_folder_edit.text()
        if folder and os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(".safetensors")]
            self._batch_run_btn.setText(f"▶  Compare All  ({len(files)} files)")
            self._batch_run_btn.setEnabled(len(files) > 0)
        else:
            self._batch_run_btn.setText("▶  Compare All")
            self._batch_run_btn.setEnabled(False)

    def _on_batch_progress(self, idx: int, total: int, filename: str):
        pct = int(idx / total * 100)
        self._batch_progress_bar.setValue(pct)
        self._batch_prog_lbl.setText(f"Analyzing {idx} of {total}:  {filename}")

    def _on_batch_finished(self, results: list):
        self._batch_prog_card.setVisible(False)
        folder = self._batch_folder_edit.text()
        files  = [f for f in os.listdir(folder) if f.lower().endswith(".safetensors")] \
                 if folder and os.path.isdir(folder) else []
        self._batch_run_btn.setText(f"▶  Compare All  ({len(files)} files)" if files else "▶  Compare All")
        self._batch_run_btn.setEnabled(bool(files))
        self._batch_worker = None
        self._show_batch_results(results)

    def _clear_batch_results(self):
        while self._batch_results_layout.count():
            item = self._batch_results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_batch_results(self, results: list):
        STATUS_COLORS = {"pass": GRN, "warn": AMB, "fail": RED, "info": ACC}
        STATUS_ICONS  = {"pass": "✓", "warn": "⚠", "fail": "✗", "info": "ℹ"}

        # Parse log.txt once for the folder (AI Toolkit training log)
        folder_log_data: dict = {}
        if results:
            first_path = results[0]["path"]
            log_path   = os.path.join(os.path.dirname(first_path), "log.txt")
            if os.path.isfile(log_path):
                folder_log_data = _parse_aitk_log(log_path)

        def _ckpt_loss(path: str) -> float | None:
            if not folder_log_data:
                return None
            step = _get_checkpoint_step(os.path.basename(path))
            if step is None:
                step = folder_log_data.get("final_step")
            return folder_log_data.get("checkpoint_losses", {}).get(step)

        # Score and sort — errors go to the bottom
        profile_sel = self._batch_profile_combo.currentText()
        _profile_map = {
            "Character / Identity":  "identity",
            "Outfit / Costume":      "outfit",
            "Style / Art Direction": "style",
        }
        profile_key = _profile_map.get(profile_sel, "concept")
        log_total   = folder_log_data.get("total_steps") if folder_log_data else None

        scored = []
        for entry in results:
            if entry["result"] is not None:
                r  = entry["result"]
                mt = r.get("model_type", "sdxl")
                mw = self.get_threshold(mt, "mag_warn")
                mf = self.get_threshold(mt, "mag_fail")
                ck = _get_checkpoint_step(os.path.basename(entry["path"]))
                s  = _score_result(r, profile=profile_key, mag_warn=mw, mag_fail=mf,
                                   checkpoint_step=ck, total_steps=log_total)
                scored.append((s, entry))
            else:
                scored.append((float("inf"), entry))
        scored.sort(key=lambda x: x[0])

        valid = [(s, e) for s, e in scored if e["result"] is not None and s < float("inf")]
        best  = valid[0] if valid else None

        # ── Sample strip ───────────────────────────────────────────────────
        folder      = self._batch_folder_edit.text() if results else ""
        samples_dir = os.path.join(folder, "samples") if folder else None
        if samples_dir and not os.path.isdir(samples_dir):
            samples_dir = None
        strip = _SampleStrip(samples_dir)
        self._sample_strip = strip

        # ── Winner banner ──────────────────────────────────────────────────
        if best:
            best_score, best_entry = best
            best_path   = best_entry["path"]
            best_result = best_entry["result"]
            best_meta   = best_result["meta"]

            winner_card = QFrame()
            winner_card.setStyleSheet(
                f"QFrame {{ background:{CAR}; border-radius:8px; border:2px solid {GRN}; }}")
            wl = QVBoxLayout(winner_card)
            wl.setContentsMargins(16, 14, 16, 14)
            wl.setSpacing(8)

            title_row = QHBoxLayout()
            star_lbl = QLabel("★  BEST CANDIDATE")
            star_lbl.setStyleSheet(
                f"color:{GRN}; font-family:{FONT}; font-size:{FONT_MD}px;"
                f" font-weight:bold; background:transparent; border:none;")
            title_row.addWidget(star_lbl)
            title_row.addStretch(1)

            overall = best_result["overall"]
            badge_colors = {"pass": GRN, "warn": AMB, "fail": RED}
            badge_texts  = {"pass": "✓  PASS", "warn": "⚠  WARN", "fail": "✗  FAIL"}
            badge = QLabel(badge_texts.get(overall, overall.upper()))
            badge.setStyleSheet(
                f"color:{PRI}; background:{badge_colors.get(overall, MUT)};"
                f" border-radius:4px; padding:4px 12px;"
                f" font-family:{FONT}; font-size:{FONT_SM}px; font-weight:bold;")
            title_row.addWidget(badge)
            wl.addLayout(title_row)

            fname_lbl = _lbl(best_meta["filename"], PRI, FONT_LG, bold=True)
            wl.addWidget(fname_lbl)

            detail_row = QHBoxLayout()
            detail_row.setSpacing(20)
            m = best_result.get("metrics", {})
            best_ckpt_loss = _ckpt_loss(best_path)
            loss_part = f"  ·  Loss: {best_ckpt_loss:.4f}" if best_ckpt_loss is not None else ""
            spi_part  = ""
            if folder_log_data.get("steps_per_image") is not None:
                spi_part = f"  ·  {folder_log_data['steps_per_image']:.0f} steps/img"
            best_eff  = m.get("mean_efficiency")
            eff_part  = f"  ·  Rank eff: {best_eff:.0%}" if best_eff is not None else ""
            detail_row.addWidget(_lbl(
                f"Score: {best_score:.1f}  ·  "
                f"Magnitude: {m.get('mean_mag', 0):.5f}  ·  "
                f"Dead layers: {m.get('total_dead', 0)}"
                f"{eff_part}  ·  "
                f"{best_meta['model_type']}  rank {best_meta['rank']}"
                f"{loss_part}{spi_part}",
                SEC, FONT_SM))
            detail_row.addStretch(1)

            copy_btn = QPushButton("Copy Path")
            copy_btn.setFixedHeight(30)
            copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            def _do_copy(p=best_path, btn=copy_btn):
                try:
                    import pyperclip as _pc
                    _pc.copy(p)
                    btn.setText("Copied ✓")
                except Exception:
                    from PySide6.QtWidgets import QApplication
                    QApplication.clipboard().setText(p)
                    btn.setText("Copied ✓")
            copy_btn.clicked.connect(_do_copy)
            detail_row.addWidget(copy_btn)

            open_btn = QPushButton("Open in Analyze ↗")
            open_btn.setFixedHeight(30)
            open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            open_btn.clicked.connect(lambda _=False, p=best_path: self._open_in_analyze(p))
            detail_row.addWidget(open_btn)
            wl.addLayout(detail_row)

            # Spectral energy breakdown — answers "how much data did rank constrain?"
            spectral = best_result.get("rank_efficiency", {}).get("spectral")
            if spectral:
                q1, q2, q3 = spectral["q1"], spectral["q2"], spectral["q3"]
                if q2 >= 0.90:
                    spec_color = GRN
                    spec_hint  = "steep spectrum — concept fits naturally in fewer dims"
                elif q2 >= 0.70:
                    spec_color = AMB
                    spec_hint  = "moderate spread — rank was a partial constraint"
                else:
                    spec_color = RED
                    spec_hint  = "flat spectrum — rank was a limiting factor"
                spec_row = QHBoxLayout()
                spec_row.setSpacing(8)
                spec_row.addWidget(_lbl(
                    f"Spectral energy:  top ¼ of dims → {q1:.0%}  ·  "
                    f"top ½ → {q2:.0%}  ·  top ¾ → {q3:.0%}",
                    SEC, FONT_SM))
                spec_row.addWidget(_lbl(f"({spec_hint})", spec_color, FONT_SM))
                spec_row.addStretch(1)
                wl.addLayout(spec_row)

            _best_rank = int(best_meta.get("rank") or 0) or None
            verdict_text, verdict_color = _rank_verdict(
                best_result.get("rank_efficiency", {}),
                best_result.get("log_data"),
                declared_rank=_best_rank,
            )
            if verdict_text:
                vrow = QHBoxLayout()
                vrow.setSpacing(8)
                vrow.addWidget(_lbl("Rank assessment:", SEC, FONT_SM))
                vrow.addWidget(_lbl(verdict_text, verdict_color, FONT_SM, bold=True))
                vrow.addStretch(1)
                wl.addLayout(vrow)

            self._batch_results_layout.addWidget(winner_card)

        # ── Results table ──────────────────────────────────────────────────
        table_card = QFrame()
        table_card.setStyleSheet(_card_style())
        tl = QVBoxLayout(table_card)
        tl.setContentsMargins(16, 12, 16, 12)
        tl.setSpacing(4)

        header_row = QWidget()
        header_row.setStyleSheet("background:transparent;")
        hr = QHBoxLayout(header_row)
        hr.setContentsMargins(8, 4, 8, 4)
        hr.setSpacing(8)
        show_loss_col = bool(folder_log_data)
        hr.addWidget(_lbl("#",          SEC, FONT_SM, bold=True), stretch=0)
        hr.addWidget(_lbl("File",       SEC, FONT_SM, bold=True), stretch=5)
        hr.addWidget(_lbl("Overall",    SEC, FONT_SM, bold=True), stretch=1)
        hr.addWidget(_lbl("Magnitude",  SEC, FONT_SM, bold=True), stretch=1)
        hr.addWidget(_lbl("Dead",       SEC, FONT_SM, bold=True), stretch=0)
        if show_loss_col:
            hr.addWidget(_lbl("Loss",   SEC, FONT_SM, bold=True), stretch=1)
        hr.addWidget(_lbl("Score ↓",    SEC, FONT_SM, bold=True), stretch=1)
        hr.addWidget(_lbl("",           SEC, FONT_SM),             stretch=1)
        tl.addWidget(header_row)

        for rank_idx, (score, entry) in enumerate(scored):
            is_winner = (rank_idx == 0 and best is not None and score < float("inf"))
            is_error  = entry["result"] is None

            row = QFrame()
            if is_winner:
                row.setStyleSheet(
                    f"QFrame {{ background:{BG}; border-radius:4px; border-left:3px solid {GRN}; }}")
            else:
                row.setStyleSheet(
                    f"QFrame {{ background:{BG if rank_idx % 2 == 0 else CAR}; border-radius:4px; }}")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(8, 7, 8, 7)
            rl.setSpacing(8)

            rank_lbl = QLabel(f"★" if is_winner else str(rank_idx + 1))
            rank_lbl.setFixedWidth(22)
            rank_lbl.setStyleSheet(
                f"color:{GRN if is_winner else MUT}; font-family:{FONT};"
                f" font-size:{FONT_SM}px; font-weight:bold; background:transparent; border:none;")
            rl.addWidget(rank_lbl)

            fname = os.path.basename(entry["path"])
            fname_btn = QPushButton(fname)
            fname_btn.setFlat(True)
            fname_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            fname_btn.setStyleSheet(
                f"QPushButton {{ color:{PRI}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f" background:transparent; border:none; text-align:left; padding:0; }}"
                f"QPushButton:hover {{ color:{ACC}; }}")
            _ck = _get_checkpoint_step(fname)
            _lbl_txt = os.path.splitext(fname)[0]
            fname_btn.clicked.connect(
                lambda _=False, s=_ck, lbl=_lbl_txt: strip.load_for_step(s, lbl))
            rl.addWidget(fname_btn, stretch=5)

            if is_error:
                err_lbl = _lbl(f"Error: {entry['error']}", RED, FONT_SM)
                err_lbl.setWordWrap(True)
                rl.addWidget(err_lbl, stretch=4)
            else:
                result  = entry["result"]
                overall = result["overall"]
                m       = result.get("metrics", {})

                badge_colors = {"pass": GRN, "warn": AMB, "fail": RED}
                badge_texts  = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
                ov_lbl = QLabel(badge_texts.get(overall, overall.upper()))
                ov_lbl.setStyleSheet(
                    f"color:{PRI}; background:{badge_colors.get(overall, MUT)};"
                    f" border-radius:3px; padding:2px 6px; font-family:{FONT};"
                    f" font-size:{FONT_SM}px; font-weight:bold; border:none;")
                rl.addWidget(ov_lbl, stretch=1)

                mag_val = m.get("mean_mag", 0.0)
                mag_color = RED if overall == "fail" else (AMB if overall == "warn" else GRN)
                rl.addWidget(_lbl(f"{mag_val:.5f}", mag_color, FONT_SM), stretch=1)

                dead_val = m.get("total_dead", 0)
                dead_color = AMB if dead_val > 0 else SEC
                dead_lbl = QLabel(str(dead_val))
                dead_lbl.setFixedWidth(30)
                dead_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                dead_lbl.setStyleSheet(
                    f"color:{dead_color}; font-family:{FONT}; font-size:{FONT_SM}px;"
                    f" background:transparent; border:none;")
                rl.addWidget(dead_lbl)

                if show_loss_col:
                    ckpt_lv = _ckpt_loss(entry["path"])
                    all_losses = [_ckpt_loss(e["path"]) for e in results
                                  if e["result"] is not None]
                    all_losses = [v for v in all_losses if v is not None]
                    best_lv = min(all_losses) if all_losses else None
                    if ckpt_lv is not None:
                        is_best_l = best_lv is not None and abs(ckpt_lv - best_lv) < 0.001
                        loss_txt  = f"{ckpt_lv:.4f}{'  *' if is_best_l else ''}"
                        lc        = GRN if is_best_l else SEC
                    else:
                        loss_txt, lc = "—", MUT
                    rl.addWidget(_lbl(loss_txt, lc, FONT_SM), stretch=1)

                score_txt = "∞ (disqualified)" if score == float("inf") else f"{score:.1f}"
                rl.addWidget(_lbl(score_txt, MUT if score == float("inf") else SEC, FONT_SM), stretch=1)

                open_row_btn = QPushButton("Open ↗")
                open_row_btn.setFixedHeight(24)
                open_row_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                open_row_btn.setStyleSheet(
                    f"QPushButton {{ background:{BG}; color:{ACC}; border:1px solid {MUT};"
                    f" border-radius:3px; font-family:{FONT}; font-size:{FONT_SM}px; padding:1px 6px; }}"
                    f"QPushButton:hover {{ border-color:{ACC}; }}")
                open_row_btn.clicked.connect(
                    lambda _=False, p=entry["path"]: self._open_in_analyze(p))
                rl.addWidget(open_row_btn, stretch=1)

            tl.addWidget(row)

        note_lbl = _lbl("Score: lower is better  ·  ∞ = NaN/Inf detected (file unusable)", MUT, FONT_SM)
        tl.addWidget(note_lbl)
        self._batch_results_layout.addWidget(table_card)

        # ── Sample image strip ─────────────────────────────────────────────
        self._batch_results_layout.addWidget(strip)
        if best:
            best_ck  = _get_checkpoint_step(os.path.basename(best[1]["path"]))
            best_lbl = os.path.splitext(os.path.basename(best[1]["path"]))[0]
            strip.load_for_step(best_ck, best_lbl)
        else:
            strip.load_for_step(None)

    def _open_in_analyze(self, path: str):
        """Load a file into the Analyze tab and switch to it."""
        self._on_file_selected(path)
        self._tabs.setCurrentIndex(0)

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
            ("Module Analysis",
             "Checks 7 and 8 (Dead Layers and Layer Balance) run per architectural module, not "
             "globally. The LoRA network is split into 4 groups based on tensor key names:\n\n"
             "  UNet Cross-Attention (attn2) — text conditioning layers. These receive the "
             "prompt embeddings and control what the image depicts. In AI-Toolkit training, "
             "to_k and to_v projections are intentionally left near-zero — the trainer focuses "
             "weight on to_q and to_out. This is by design, not a defect. The checker compares "
             "layers within cross-attention only, so this normal variation does not affect the "
             "self-attention or feedforward results.\n\n"
             "  UNet Self-Attention (attn1) — spatial relationship layers. These handle how "
             "image regions relate to each other (structure, composition). All four projections "
             "(to_q, to_k, to_v, to_out) are expected to be active.\n\n"
             "  UNet Feedforward (ff_net / geglu) — channel mixing layers located between "
             "attention blocks. Responsible for non-linear feature transformation.\n\n"
             "  Text Encoder (lora_te*, lora_te1*, lora_te2*) — modifies how the model "
             "interprets text tokens. SDXL has two encoders (CLIP-L and OpenCLIP-G); "
             "both are grouped here.\n\n"
             "Comparing layers across groups is meaningless — they operate in different "
             "dimensional spaces with different magnitude scales. Per-group comparison makes "
             "the balance thresholds structurally tight and the dead-layer check accurate."),
            ("7.  Dead Layers  (per module)",
             "Within each architectural group, identifies lora_up layers whose mean absolute "
             "value is below the dead threshold. A dead layer contributes nothing to the output "
             "and indicates a collapsed or untrained module."),
            ("8.  Layer Balance  (per module)",
             "Computes max(mean_abs) / min(mean_abs) within each architectural group. "
             "Layers in the same group perform the same function at different UNet depths, so "
             "wide magnitude imbalance within a group is genuinely abnormal — unlike a "
             "global comparison across architecturally incompatible modules.\n\n"
             "Note: cross-attention balance is evaluated only across active layers. Near-zero "
             "to_k / to_v (AI-Toolkit default) are counted as dead, not as imbalance — "
             "preventing false balance failures on otherwise healthy AI-Toolkit LoRAs."),
            ("Batch Compare",
             "The Batch Compare tab analyses every .safetensors file in a folder and "
             "ranks them so you can pick the best checkpoint from a training run.\n\n"
             "How to use:\n"
             "  1. Click Browse and select your training output folder.\n"
             "  2. Set Model Type and Trainer (or leave both on Auto-detect).\n"
             "  3. Click Compare All. A progress bar tracks each file as it is analysed.\n"
             "  4. When complete, the Best Candidate banner shows the top-ranked file "
             "with a Copy Path button and an Open in Analyze button.\n"
             "  5. The full ranking table lists every file with its overall status, "
             "magnitude, dead layer count, checkpoint loss (if log.txt is present), "
             "and score.\n"
             "  6. Click Open ↗ on any row to load that file into the Analyze tab for "
             "the full per-module breakdown.\n\n"
             "Loss column (AI Toolkit):\n"
             "  If a log.txt file is found in the folder, the tool parses the full loss "
             "curve and shows a ±50-step window average for each checkpoint. The "
             "checkpoint with the lowest average loss is highlighted green. This is a "
             "separate signal from the tensor health score — a checkpoint can have a "
             "healthy score but have been saved at a noisy spike in the loss curve.\n\n"
             "Scoring (lower is better):\n"
             "  · NaN / Inf present → score ∞, disqualified (never selected as best).\n"
             "  · Each fail check adds 100 pts; each warn adds 20 pts.\n"
             "    Overbaked and Rank Consistency failures are weighted more heavily.\n"
             "  · Global mean magnitude contributes a continuous penalty "
             "(magnitude × 500), so two files with the same check statuses are "
             "separated by how far into the warn/fail zone their magnitude sits.\n"
             "  · Dead layers add 10 pts each.\n"
             "  · Extreme layer balance (above 50×) adds a small continuous penalty "
             "(cross-attention balance is excluded for AI Toolkit LoRAs).\n\n"
             "The score is a relative ranking tool, not an absolute quality number. "
             "Two files both scoring 0 are equally healthy — use the Analyze tab to "
             "inspect the module details and make the final call."),
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
            ("Training Log  (AI Toolkit)",
             "When Trainer is set to AI Toolkit (or auto-detected as such) and a "
             "log.txt file is found alongside the LoRA file, the tool parses the "
             "training log and displays a summary card below the module analysis.\n\n"
             "Dataset / Steps / Steps per image:\n"
             "  Image count is read from the latent preprocessing progress bar in the "
             "log. Steps per image = total_steps / image_count. Healthy range is "
             "roughly 80–400 steps per image for a character LoRA — below 80 may be "
             "undertrained; above 400 risks memorisation.\n\n"
             "Loss trend  (Q1 avg vs Q4 avg):\n"
             "  The log contains a loss value at every training step. The tool computes "
             "the mean loss for the first and last quarters of training. If Q4 < Q1 by "
             "more than 3%, training is labelled 'still learning' (green) — the model "
             "was actively improving at the end. If Q4 ≈ Q1, the run plateaued early "
             "(amber). Late-stage noise is reported as a coefficient of variation % — "
             "high CV (50–80%) is normal for this training setup and does not indicate "
             "a problem.\n\n"
             "Loss at this checkpoint:\n"
             "  A ±50-step window average is computed around each checkpoint's save "
             "step. This smooths out per-step noise and shows whether the checkpoint "
             "was captured at a loss valley or a transient spike. The checkpoint with "
             "the lowest window average is marked with * and shown in green. This is "
             "visible in both the Analyze tab card and the Batch Compare loss column."),
            ("Training Software",
             "The Trainer selector adjusts the scoring rules to match how the training "
             "tool structures its LoRA weights.\n\n"
             "AI Toolkit:  Cross-attention balance is not scored. AI Toolkit intentionally "
             "leaves to_k and to_v projections near-zero — this produces a mechanically "
             "high hottest/coldest ratio in cross-attention that is not a defect. The "
             "balance value is still shown for reference but does not contribute to the "
             "score or the overall pass/warn/fail status. Dead layers and magnitude are "
             "scored normally.\n\n"
             "Kohya / kohya-ss:  All checks and balance thresholds apply as configured.\n\n"
             "Auto-detect:  The tool checks for a config.yaml in the same folder as the "
             "LoRA file (AI Toolkit leaves one there). If absent, it falls back to tensor "
             "key heuristics — conv LoRA keys with no kohya hash metadata → AI Toolkit. "
             "Override if the detection is incorrect."),
            ("Rank Saturation",
             "Rank saturation is the opposite problem to overbaking — the declared rank "
             "was too small for the dataset, not too large.\n\n"
             "Detected when rank efficiency (from SVD) reaches 88% or higher, meaning "
             "nearly every rank dimension is carrying signal and the model had no spare "
             "capacity. The optimizer cannot improve further no matter how many steps "
             "are run — detail is lost because there is nowhere to store it.\n\n"
             "Severity:\n"
             "  Warn + loss plateau confirmed:  rank was definitely the bottleneck. "
             "The loss stopped improving while rank dimensions were fully occupied.\n"
             "  Warn (no log):  efficiency ≥ 93% — very likely saturated.\n"
             "  Info:  efficiency 88–92% — possible saturation; worth monitoring.\n\n"
             "Rank saturation does not affect Batch Compare rankings within a single "
             "training run since all checkpoints share the same rank setting and will "
             "all receive the same flag. It is a training configuration diagnostic, "
             "not a checkpoint quality differentiator.\n\n"
             "Resolution: retrain the subject at a higher rank (e.g. double it) and "
             "compare rank efficiency of both runs. If the higher-rank run shows 55–75% "
             "efficiency, the new rank is well-sized for your dataset."),
            ("Recommendation Profile",
             "The Profile selector changes how the Batch Compare ranking score is computed. "
             "Select the profile that matches what you trained.\n\n"
             "Concept / Pose (default):  General-purpose. Penalises fail/warn checks and "
             "magnitude continuously. Step count is not decisive — concepts and poses are "
             "learned quickly and an earlier checkpoint is often just as good.\n\n"
             "Character / Identity:  For character LoRAs where face and appearance must be "
             "captured. Overbaked files are hard-disqualified (they overpower other LoRAs). "
             "Among safe candidates, later checkpoints rank higher (more training = stronger "
             "identity), with a secondary bonus for higher magnitude within the safe range.\n\n"
             "Outfit / Costume:  For clothing or accessory LoRAs that will be layered over a "
             "character. Overbaked files are disqualified (outfit LoRA bleeding into skin/hair "
             "is a common problem). Magnitude is scored against a sweet spot of ~65% of the "
             "warn threshold — just enough to capture detail without overpowering.\n\n"
             "Style / Art Direction:  For artistic style or medium LoRAs. Lower magnitude "
             "is preferred so the style influences rather than dominates. Overbaked files "
             "are disqualified. Step count has only a minor effect — style tends to stabilise "
             "at moderate training levels.\n\n"
             "All profile selections are saved between sessions."),
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
                f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f" background:transparent; border:none;")
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

        trainer_sel = self._trainer_combo.currentText()
        forced_trainer: str | None = None
        if trainer_sel == "AI Toolkit":
            forced_trainer = "aitoolkit"
        elif trainer_sel == "Kohya / kohya-ss":
            forced_trainer = "kohya"

        get_fn = (lambda mt, k: self.get_threshold(forced_type, k)) \
                 if forced_type else self.get_threshold

        self._worker = _AnalysisWorker(self._current_file, get_fn,
                                       forced_type, forced_trainer)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def cleanup(self):
        """Disconnect any in-flight workers before the widget is deleted."""
        if self._worker is not None:
            try:
                self._worker.finished.disconnect()
                self._worker.error.disconnect()
            except RuntimeError:
                pass
            self._worker = None
        if self._batch_worker is not None:
            self._batch_worker.stop()
            try:
                self._batch_worker.progress.disconnect()
                self._batch_worker.finished.disconnect()
            except RuntimeError:
                pass
            self._batch_worker = None

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
        trainer       = result.get("trainer", "kohya")

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
        trainer_labels = {"aitoolkit": "AI Toolkit", "kohya": "Kohya"}
        trainer_label  = trainer_labels.get(trainer, trainer)
        badge_row.addWidget(badge)
        badge_row.addWidget(_lbl(
            f"Detected: {model_type.upper()}  ·  Trainer: {trainer_label}", SEC, FONT_MD))
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

        # Layout: col 0=left-label, col 1=left-value, col 2=spacer, col 3=right-label, col 4=right-value
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 2)   # left value — expands to fill left half
        grid.setColumnStretch(2, 3)   # spacer between pairs
        grid.setColumnStretch(4, 2)   # right value
        for i, (lk, lv) in enumerate(fields):
            col_base = (i % 2) * 3   # 0 for left pair, 3 for right pair
            row = i // 2
            grid.addWidget(_lbl(f"{lk}:", SEC, FONT_SM), row, col_base)
            grid.addWidget(_lbl(lv, PRI, FONT_SM), row, col_base + 1)
        ml.addLayout(grid)

        _declared_rank = int(meta.get("rank") or 0) or None
        verdict_text, verdict_color = _rank_verdict(
            result.get("rank_efficiency", {}),
            result.get("log_data"),
            declared_rank=_declared_rank,
        )
        if verdict_text:
            vrow = QHBoxLayout()
            vrow.setSpacing(8)
            vrow.addWidget(_lbl("Rank assessment:", SEC, FONT_SM))
            vrow.addWidget(_lbl(verdict_text, verdict_color, FONT_SM, bold=True))
            vrow.addStretch(1)
            ml.addLayout(vrow)

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
            chr_.addWidget(_lbl("Eff %", SEC, FONT_SM, bold=True), stretch=1)
            chr_.addWidget(_lbl("Notes", SEC, FONT_SM, bold=True), stretch=3)
            mcl.addWidget(col_hdr)

            rank_eff_groups = result.get("rank_efficiency", {}).get("groups", {})
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

                grp_eff = rank_eff_groups.get(mod)
                eff_str   = f"{grp_eff:.0%}" if grp_eff is not None else "—"
                eff_color = (GRN if grp_eff is not None and grp_eff >= 0.6 else
                             AMB if grp_eff is not None and grp_eff >= 0.35 else
                             RED if grp_eff is not None else MUT)
                rl.addWidget(_lbl(eff_str, eff_color, FONT_SM), stretch=1)

                notes = ", ".join(grp["issues"]) if grp["issues"] else "—"
                notes_lbl = QLabel(notes)
                notes_lbl.setWordWrap(True)
                notes_lbl.setStyleSheet(
                    f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;"
                    f" background:transparent;")
                rl.addWidget(notes_lbl, stretch=3)
                mcl.addWidget(row)

            self._results_layout.addWidget(mod_card)

        # ── Training Log card (AI Toolkit only) ────────────────────────────
        log_data = result.get("log_data", {})
        if log_data:
            log_card = QFrame()
            log_card.setStyleSheet(_card_style())
            ll = QVBoxLayout(log_card)
            ll.setContentsMargins(16, 12, 16, 12)
            ll.setSpacing(6)
            ll.addWidget(_lbl("Training Log  (AI Toolkit)", PRI, FONT_MD, bold=True))

            img_count   = log_data.get("image_count")
            total_steps = log_data.get("total_steps")
            spi         = log_data.get("steps_per_image")
            if spi is not None:
                spi_flag = ("  — may be undertrained" if spi < 80 else
                            "  — overfit risk"        if spi > 400 else "")
                spi_str = f"{spi:.0f}{spi_flag}"
            else:
                spi_str = "?"
            row1 = QHBoxLayout()
            row1.addWidget(_lbl(
                f"Dataset: {img_count or '?'} images     "
                f"Steps: {total_steps or '?'}     "
                f"Steps / image: {spi_str}",
                SEC, FONT_SM))
            row1.addStretch(1)
            ll.addLayout(row1)

            q1 = log_data.get("q1_avg")
            q4 = log_data.get("q4_avg")
            cv = log_data.get("late_cv")
            converged = log_data.get("converged", False)
            if q1 is not None and q4 is not None:
                trend_color = GRN if converged else AMB
                cv_str = f"     Late noise: {cv * 100:.0f}% CV" if cv is not None else ""
                trend_lbl = QLabel(
                    f"Loss trend  Q1: {q1:.4f}  Q4: {q4:.4f}  "
                    f"({'still learning' if converged else 'plateaued'})"
                    + cv_str)
                trend_lbl.setStyleSheet(
                    f"color:{trend_color}; font-family:{FONT}; font-size:{FONT_SM}px;"
                    f" background:transparent;")
                ll.addWidget(trend_lbl)

            ckpt_loss = log_data.get("this_checkpoint_loss")
            ckpt_step = log_data.get("this_checkpoint_step")
            all_ckpt  = {s: v for s, v in log_data.get("checkpoint_losses", {}).items()
                         if v is not None}
            best_loss = min(all_ckpt.values()) if all_ckpt else None

            if ckpt_loss is not None:
                is_best   = best_loss is not None and abs(ckpt_loss - best_loss) < 0.001
                note      = "  (lowest among checkpoints)" if is_best else ""
                loss_lbl  = QLabel(
                    f"Loss at step {ckpt_step} (50-step window avg): {ckpt_loss:.4f}{note}")
                loss_lbl.setStyleSheet(
                    f"color:{GRN if is_best else SEC}; font-family:{FONT};"
                    f" font-size:{FONT_SM}px; background:transparent;")
                ll.addWidget(loss_lbl)

                if len(all_ckpt) > 1:
                    ll.addWidget(_lbl("All checkpoints:", MUT, FONT_SM))
                    tbl_row = QHBoxLayout()
                    tbl_row.setSpacing(16)
                    for s, lv in sorted(all_ckpt.items()):
                        is_b = best_loss is not None and abs(lv - best_loss) < 0.001
                        is_t = (s == ckpt_step)
                        c    = GRN if is_b else (PRI if is_t else MUT)
                        tbl_row.addWidget(_lbl(
                            f"step {s}: {lv:.4f}{'  *' if is_b else ''}",
                            c, FONT_SM))
                    tbl_row.addStretch(1)
                    ll.addLayout(tbl_row)

            self._results_layout.addWidget(log_card)

    # ══════════════════════════════════════════════════════════════════════════
    # Threshold helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _save_combo(self, key: str, value: str):
        self._cfg[key] = value
        save_json(HEALTH_CFG, self._cfg)

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
