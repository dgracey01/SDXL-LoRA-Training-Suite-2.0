"""
Model Merge page — integrated into Lora Training Suite 2.0.
Tabs: Checkpoint Merge | LoRA Merge | Bake LoRA | Help
"""
from __future__ import annotations

import os
import threading

from PySide6.QtCore    import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QLineEdit, QPushButton, QComboBox, QSlider, QDoubleSpinBox,
    QSpinBox, QProgressBar, QFileDialog, QScrollArea,
    QTabWidget, QMessageBox, QSizePolicy, QInputDialog, QCheckBox,
)

from shared.theme  import (
    BG, PAN, CAR, ACC, GRN, RED, MUT, PRI, SEC, AMB,
    FONT, FONT_SM, FONT_MD, FONT_LG, FONT_XL, SIGNATURE,
)
from shared.config import load_json, save_json
import os as _os

_HERE      = _os.path.dirname(_os.path.abspath(__file__))
_MERGE_CFG = _os.path.join(_HERE, "merge_config.json")

_CFG_DEFAULTS = {
    "ckpt_model_a":     "",
    "ckpt_model_b":     "",
    "ckpt_model_c":     "",
    "ckpt_vae":         "",
    "ckpt_method":      "Weighted Sum",
    "ckpt_selective":   False,
    "ckpt_te_alpha":    0.00,
    "ckpt_ca_alpha":    0.00,
    "ckpt_sa_alpha":    0.75,
    "ckpt_ff_alpha":    0.75,
    "ckpt_other_alpha": 0.00,
    "ckpt_alpha":     0.5,
    "ckpt_beta":      0.7,
    "ckpt_density":   0.9,
    "ckpt_precision": "fp16",
    "ckpt_output":    "",
    "lora_a":         "",
    "lora_b":         "",
    "lora_weight_a":  1.0,
    "lora_weight_b":  1.0,
    "lora_rank":      0,
    "lora_output":    "",
    "bake_ckpt":      "",
    "bake_lora":      "",
    "bake_mult":      1.0,
    "bake_precision": "fp16",
    "bake_output":    "",
    "device":         "auto",
    # LoRA Merge slots
    "lora_type_a":    "Character",
    "lora_type_b":    "Character",
    "lora_c":         "", "lora_type_c": "Character", "lora_weight_c": 0.75,
    "lora_d":         "", "lora_type_d": "Character", "lora_weight_d": 0.75,
    # Extract LoRA
    "extract_tuned":  "", "extract_base": "", "extract_rank": 32,
    "extract_conv":   True, "extract_output": "",
    # Bake LoRA slots
    "bake_lora_1":    "", "bake_type_1": "Character", "bake_ratio_1": 0.85,
    "bake_lora_2":    "", "bake_type_2": "Character", "bake_ratio_2": 0.85,
    "bake_lora_3":    "", "bake_type_3": "Character", "bake_ratio_3": 0.85,
    "bake_lora_4":    "", "bake_type_4": "Character", "bake_ratio_4": 0.85,
}

# ── LoRA type presets (ported from kohya_rocm/gui.py) ─────────────────────────

_LORA_TYPES = ["Character", "Pose", "Detail", "Style", "Concept"]

_BAKE_PRESET  = {"Character": 0.85, "Pose": 0.88, "Detail": 0.75, "Style": 0.75, "Concept": 0.70}
_MERGE_PRESET = {"Character": 0.75, "Pose": 0.78, "Detail": 0.65, "Style": 0.60, "Concept": 0.50}

_BAKE_MAX   = {"Character": 1.00, "Pose": 0.95, "Detail": 0.85, "Style": 0.85, "Concept": 0.80}
_MERGE_MAX  = {"Character": 0.90, "Pose": 0.92, "Detail": 0.80, "Style": 0.80, "Concept": 0.70}

# Combined-total thresholds
_CAUTION_AT = 1.5
_WARN_AT    = 2.0

# ── Checkpoint method detailed help ──────────────────────────────────────────

_CK_METHOD_HELP: dict[str, str] = {
    "Weighted Sum": (
        "LINEAR BLEND\n\n"
        "result = (1 − α) · A  +  α · B\n\n"
        "The simplest and most predictable method. Every parameter is a weighted "
        "average — nothing is discarded, nothing is amplified.\n\n"
        "α = 0.25  subtle B touch, A dominates\n"
        "α = 0.50  equal contribution from both\n"
        "α = 0.70  B's aesthetic takes over while A holds the structure\n\n"
        "Iterate in 0.05–0.10 steps from a baseline until output quality peaks.\n\n"
        "Does not require Model C."
    ),
    "Slerp": (
        "SPHERICAL INTERPOLATION\n\n"
        "Interpolates along the surface of a hypersphere instead of a straight line. "
        "Produces smoother transitions than Weighted Sum — the result feels less like "
        "an average and more like a natural blend.\n\n"
        "α = 0.30  style injection, A stays dominant\n"
        "α = 0.50  smooth midpoint blend\n\n"
        "Best when A and B are stylistically close — two anime checkpoints, two "
        "realism models, or models trained on similar datasets.\n\n"
        "Does not require Model C."
    ),
    "Add Difference": (
        "DELTA INJECTION\n\n"
        "result = A  +  α · (B − C)\n\n"
        "B − C isolates exactly what B learned relative to the original base. "
        "That delta is then injected into A — which can be any model, not necessarily "
        "the same base B was trained from.\n\n"
        "α = 1.00  full delta injection (community default for concept grafting)\n"
        "α = 0.50  half-strength, softer effect\n\n"
        "Common uses: graft a concept, style, or quality refiner onto a different "
        "base. Apply aesthetic adaptations without retraining.\n\n"
        "Requires Model C — the original base B was fine-tuned from."
    ),
    "TIES": (
        "TRIM · ELECT SIGN · DISJOINT MERGE\n\n"
        "Merges two fine-tunes by:\n"
        "1.  Trimming low-magnitude deltas  (Beta controls keep-rate)\n"
        "2.  Resolving sign conflicts by majority vote\n"
        "3.  Averaging only deltas that agree on sign\n\n"
        "Reduces interference between unrelated parameters — significantly better "
        "than Weighted Sum when A and B were both fine-tuned from the same base C "
        "for different subjects or styles.\n\n"
        "Beta = 0.50  keep top 50 % of delta weights  (recommended default)\n\n"
        "Requires Model C — the shared base both A and B were fine-tuned from."
    ),
    "DARE": (
        "DROP AND RESCALE\n\n"
        "Randomly zeros (1 − density) of each model's delta, then rescales by "
        "1/density to preserve expected magnitude. Random pruning decorrelates "
        "the two delta streams, reducing the chance that conflicting weights cancel "
        "each other out.\n\n"
        "Density = 0.75  keep 75 % of weights  (community sweet spot)\n"
        "Higher density → closer to a straight average\n"
        "Lower density → more aggressive pruning, higher variance\n\n"
        "Best when A and B have overlapping fine-tune targets that would otherwise "
        "interfere destructively.\n\n"
        "Requires Model C."
    ),
    "DARE+TIES": (
        "DARE PREPROCESSING  +  TIES SIGN-ELECT\n\n"
        "Combines both interference-reduction strategies:\n"
        "1.  DARE randomly prunes and rescales each delta\n"
        "2.  TIES resolves remaining sign conflicts by majority vote\n\n"
        "The strongest option when merging two fine-tunes that differ significantly — "
        "e.g. a character-focused model and a style-focused model both derived from "
        "the same SDXL or Illustrious base.\n\n"
        "Beta = 0.50, Density = 0.75 are the recommended starting values.\n\n"
        "Requires Model C."
    ),
}

# ── Checkpoint merge presets ──────────────────────────────────────────────────
# beta/density stored for all entries — graying is visual only, values always set
_CK_PRESETS_BUILTIN: dict[str, dict] = {
    # beta/density stored for all entries so applying a preset always sets all fields;
    # irrelevant fields are visually dimmed and disabled by _update_ck_param_state.
    "Weighted Sum – Conservative": {"method": "Weighted Sum",   "alpha": 0.25, "beta": 0.50, "density": 0.75},
    "Weighted Sum – Equal":        {"method": "Weighted Sum",   "alpha": 0.50, "beta": 0.50, "density": 0.75},
    "Weighted Sum – Style Push":   {"method": "Weighted Sum",   "alpha": 0.70, "beta": 0.50, "density": 0.75},
    "Slerp – Neutral":             {"method": "Slerp",          "alpha": 0.50, "beta": 0.50, "density": 0.75},
    "Slerp – Style Inject":        {"method": "Slerp",          "alpha": 0.30, "beta": 0.50, "density": 0.75},
    "Add Difference – Concept Graft": {"method": "Add Difference", "alpha": 1.00, "beta": 0.50, "density": 0.75},
    "TIES – Conservative":         {"method": "TIES",           "alpha": 0.50, "beta": 0.50, "density": 0.75},
    "DARE – Standard":             {"method": "DARE",           "alpha": 0.50, "beta": 0.50, "density": 0.75},
    "DARE+TIES – Full":            {"method": "DARE+TIES",      "alpha": 0.50, "beta": 0.50, "density": 0.75},
    # ── Selective merge presets ──────────────────────────────────────────────
    "Model A Thinks, Model B Renders": {
        "method": "Weighted Sum", "alpha": 0.35, "beta": 0.50, "density": 0.75,
        "selective": True,
        "te_alpha": 0.00, "ca_alpha": 0.00, "sa_alpha": 0.75, "ff_alpha": 0.75, "other_alpha": 0.00,
    },
    "Selective – Soft TE Blend": {
        "method": "Weighted Sum", "alpha": 0.25, "beta": 0.50, "density": 0.75,
        "selective": True,
        "te_alpha": 0.20, "ca_alpha": 0.10, "sa_alpha": 0.30, "ff_alpha": 0.30, "other_alpha": 0.00,
    },
    "Selective – Style Only": {
        "method": "Weighted Sum", "alpha": 0.25, "beta": 0.50, "density": 0.75,
        "selective": True,
        "te_alpha": 0.00, "ca_alpha": 0.00, "sa_alpha": 0.10, "ff_alpha": 0.40, "other_alpha": 0.00,
    },
}

_CK_SELECTIVE_HELP = (
    "SELECTIVE MERGE MODE\n\n"
    "Applies a different blend ratio to each architectural group.\n"
    "α 0.00 = 100% Model A  ·  α 1.00 = 100% Model B\n\n"
    "Text Encoders — vocabulary, token meaning, tag vs. caption\n"
    "conditioning. Low alpha preserves Model A's prompt understanding.\n\n"
    "Cross-Attention — how text guidance drives image content. The CA\n"
    "K/V projections were trained on a specific TE's output. Keep CA α\n"
    "within ~0.10 of TE α — large divergence breaks text conditioning.\n"
    "At 0.00 or 1.00 they must match exactly.\n\n"
    "Self-Attention — spatial structure, composition, subject layout.\n\n"
    "Feedforward — rendering quality, texture, color, fine detail.\n\n"
    "Other — VAE, norms, noise schedule. Usually keep at 0.00.\n\n"
    "⚠  Locked to Weighted Sum. Three-model methods (TIES, DARE,\n"
    "Add Difference) treat α=0 as Model C — not Model A."
)

_CK_SELECTIVE_PRESET_HELP = {
    "Model A Thinks, Model B Renders": (
        "MODEL A THINKS, MODEL B RENDERS\n\n"
        "Model A's full text conditioning system — both text encoders\n"
        "and cross-attention layers — drives Model B's visual rendering\n"
        "engine (self-attention and feedforward layers).\n\n"
        "Use this when Model A has superior prompt understanding (dual\n"
        "tag + caption training, domain-specific vocabulary) and Model B\n"
        "has the visual quality or style you want to inherit.\n\n"
        "The TE and cross-attention are kept as a matched pair at 0.00\n"
        "— swapping only one would break the coupling between them.\n\n"
        "TE α=0.00  ·  Cross-Attn α=0.00\n"
        "Self-Attn α=0.75  ·  Feedforward α=0.75  ·  Other α=0.00"
    ),
    "Selective – Soft TE Blend": (
        "SOFT TEXT ENCODER BLEND\n\n"
        "Low-risk introduction of Model A's vocabulary into Model B.\n"
        "Text encoders blend at 20%, cross-attention at 10% — enough\n"
        "for tag vocabulary leakage without breaking the TE / cross-\n"
        "attention coupling.\n\n"
        "Safe starting point before committing to a full conditioning\n"
        "transplant. If output reads well, increase TE and cross-\n"
        "attention alphas in 0.05 steps toward Model A Thinks.\n\n"
        "TE α=0.20  ·  Cross-Attn α=0.10\n"
        "Self-Attn α=0.30  ·  Feedforward α=0.30  ·  Other α=0.00"
    ),
    "Selective – Style Only": (
        "STYLE INJECTION — CONDITIONING PRESERVED\n\n"
        "Keeps Model A's text conditioning completely intact while\n"
        "borrowing Model B's rendering aesthetic. Only feedforward\n"
        "layers — which carry texture, color, and rendering style —\n"
        "are significantly blended from Model B.\n\n"
        "Self-attention is kept close to Model A to preserve subject\n"
        "structure. Use when you want Model B's look but need Model A's\n"
        "prompt responsiveness to remain in control.\n\n"
        "TE α=0.00  ·  Cross-Attn α=0.00\n"
        "Self-Attn α=0.10  ·  Feedforward α=0.40  ·  Other α=0.00"
    ),
}


def _ratio_warning(types: list[str], ratios: list[float],
                   max_map: dict, warn_lbl: QLabel):
    """Update a warning QLabel based on per-type ceilings and combined total."""
    active = [(t, r) for t, r in zip(types, ratios) if r > 0.0]
    if not active:
        warn_lbl.setText("")
        return

    total  = sum(r for _, r in active)
    lines  = []

    for t, r in active:
        cap = max_map.get(t, 0.75)
        if r > cap:
            lines.append(f"  {t} at {r:.2f} exceeds ceiling for this type ({cap:.2f})")

    if total > _WARN_AT:
        header_color = RED
        header = f"Overload — combined {total:.2f}:  reduce ratios to avoid incoherent output"
    elif total > _CAUTION_AT:
        header_color = "#e6a817"
        header = f"Caution — combined {total:.2f}:  test a sample before committing"
    else:
        header_color = GRN
        header = f"OK — combined {total:.2f}:  within safe range"

    warn_lbl.setText(header + ("" if not lines else "\n" + "\n".join(lines)))
    warn_lbl.setStyleSheet(
        f"color:{header_color}; font-family:{FONT}; font-size:{FONT_SM}px;"
        f" background:transparent; border:none;")


# ── Widget helpers ─────────────────────────────────────────────────────────────

def _lbl(text: str, color: str = SEC, size: int = FONT_MD, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{color}; font-family:{FONT}; font-size:{size}px;"
        f" font-weight:{'bold' if bold else 'normal'}; background:transparent; border:none;")
    return lbl


def _card() -> QFrame:
    f = QFrame()
    f.setStyleSheet(f"QFrame {{ background:{CAR}; border-radius:8px; border:2px solid {MUT}; }}")
    return f


def _field_style() -> str:
    return (
        f"QLineEdit {{ background:{BG}; color:{PRI}; border:1px solid {MUT};"
        f" border-radius:4px; padding:4px 8px; font-family:{FONT}; font-size:{FONT_MD}px; }}"
        f"QLineEdit:focus {{ border-color:{ACC}; }}"
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


def _btn_style(color: str = ACC) -> str:
    return (
        f"QPushButton {{ background:{color}; color:{PRI}; border:none;"
        f" border-radius:4px; font-family:{FONT}; font-size:{FONT_MD}px;"
        f" font-weight:bold; min-height:30px; padding:0 16px; }}"
        f"QPushButton:hover {{ background:#185FA5; }}"
        f"QPushButton:disabled {{ background:{MUT}; color:{SEC}; }}"
    )


def _browse_btn() -> QPushButton:
    btn = QPushButton("Browse")
    btn.setStyleSheet(_btn_style())
    btn.setFixedWidth(80)
    return btn


def _file_row(label: str, field: QLineEdit, btn: QPushButton) -> QVBoxLayout:
    vbox = QVBoxLayout()
    vbox.setSpacing(4)
    vbox.addWidget(_lbl(label, PRI, FONT_MD, bold=False))
    hbox = QHBoxLayout()
    hbox.setSpacing(6)
    hbox.addWidget(field)
    hbox.addWidget(btn)
    vbox.addLayout(hbox)
    return vbox


def _progress_row() -> tuple[QProgressBar, QLabel]:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(False)
    bar.setFixedHeight(8)
    bar.setStyleSheet(
        f"QProgressBar {{ background:{BG}; border:none; border-radius:4px; }}"
        f"QProgressBar::chunk {{ background:{ACC}; border-radius:4px; }}"
    )
    status = _lbl("Idle", SEC, FONT_SM)
    return bar, status


def _vram_info_text() -> str:
    """Build a one-line VRAM summary for display."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "No CUDA GPU detected — CPU mode only"
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        name       = props.name
        total_gb   = total / (1024 ** 3)
        free_gb    = free  / (1024 ** 3)
        return f"{name}  |  {free_gb:.1f} GB free / {total_gb:.1f} GB total"
    except Exception:
        return "GPU info unavailable"


# ── Worker signal carrier ──────────────────────────────────────────────────────

class _Worker(QObject):
    progress     = Signal(int, int, str)
    finished     = Signal()
    error        = Signal(str)
    health_done  = Signal(dict)   # checkpoint health: 4 indicator results
    lora_health  = Signal(dict)   # full LoRA analysis result (extract tab)


# ── MergePage ─────────────────────────────────────────────────────────────────

class MergePage(QWidget):
    """Model Merge — checkpoint merge, LoRA merge, LoRA bake-in."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{PAN};")
        self._cfg    = load_json(_MERGE_CFG, _CFG_DEFAULTS)
        self._thread: threading.Thread | None = None
        self._worker: _Worker | None          = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────────────
        hdr = QFrame(self)
        hdr.setStyleSheet(f"background:{CAR}; border-bottom:2px solid {MUT};")
        hdr.setFixedHeight(56)
        hrow = QHBoxLayout(hdr)
        hrow.setContentsMargins(20, 0, 20, 0)
        title = QLabel("Model Merge", hdr)
        title.setStyleSheet(
            f"color:{PRI}; font-family:{FONT}; font-size:{FONT_XL}px;"
            f" font-weight:bold; background:transparent;")
        hrow.addWidget(title)
        hrow.addStretch(1)
        sig = QLabel(SIGNATURE, hdr)
        sig.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        hrow.addWidget(sig)
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

        self._build_device_bar()
        self._build_checkpoint_tab()
        self._build_lora_tab()
        self._build_bake_tab()
        self._build_extract_tab()
        self._build_help_tab()

    # ══════════════════════════════════════════════════════════════════════════
    # Device / VRAM bar  (sits between header and tab strip)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_device_bar(self):
        bar = QFrame(self)
        bar.setStyleSheet(
            f"QFrame {{ background:{BG}; border-bottom:1px solid {MUT}; border-top:none; }}")
        bar.setFixedHeight(38)
        row = QHBoxLayout(bar)
        row.setContentsMargins(20, 0, 20, 0)
        row.setSpacing(12)

        row.addWidget(_lbl("Compute Device:", SEC, FONT_SM))

        self._device_combo = QComboBox()
        self._device_combo.setStyleSheet(_combo_style())
        self._device_combo.setFixedWidth(120)
        for opt in ("Auto", "GPU (CUDA)", "CPU"):
            self._device_combo.addItem(opt)
        saved = self._cfg.get("device", "auto")
        idx_map = {"auto": 0, "cuda": 1, "cpu": 2}
        self._device_combo.setCurrentIndex(idx_map.get(saved, 0))
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        row.addWidget(self._device_combo)

        self._vram_lbl = _lbl(_vram_info_text(), MUT, FONT_SM)
        row.addWidget(self._vram_lbl)
        row.addStretch(1)

        # Insert bar between header and tabs
        layout = self.layout()
        layout.insertWidget(1, bar)

    def _on_device_changed(self, _idx: int):
        dev_map = {0: "auto", 1: "cuda", 2: "cpu"}
        dev = dev_map.get(self._device_combo.currentIndex(), "auto")
        self._cfg["device"] = dev
        save_json(_MERGE_CFG, self._cfg)
        self._vram_lbl.setText(_vram_info_text())

    def _current_device(self) -> str:
        dev_map = {0: "auto", 1: "cuda", 2: "cpu"}
        return dev_map.get(self._device_combo.currentIndex(), "auto")

    # ── Health indicator helpers ───────────────────────────────────────────────

    @staticmethod
    def _mk_health_lbl(text: str) -> QLabel:
        lbl = QLabel(f"● {text}")
        lbl.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")
        return lbl

    @staticmethod
    def _set_health_lbl(lbl: QLabel, status: str, detail: str):
        color = {
            "pass": GRN, "warn": AMB, "fail": RED,
        }.get(status, MUT)
        lbl.setStyleSheet(
            f"color:{color}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")
        lbl.setToolTip(detail)

    def _apply_health_result(self, health: dict,
                              lbl_key:       QLabel, lbl_mag:       QLabel,
                              lbl_vae:       QLabel, lbl_alpha:     QLabel,
                              lbl_confirmed: QLabel | None = None):
        for key, lbl in (
            ("key_count",        lbl_key),
            ("magnitude_drift",  lbl_mag),
            ("vae_integrity",    lbl_vae),
            ("alpha_consistency", lbl_alpha),
            ("merge_confirmed",  lbl_confirmed),
        ):
            if lbl is not None and key in health:
                self._set_health_lbl(lbl, health[key]["status"], health[key]["detail"])

    def _on_ck_health_done(self, health: dict):
        self._apply_health_result(
            health, self._ck_h_key, self._ck_h_mag,
            self._ck_h_vae, self._ck_h_alpha, self._ck_h_confirmed)

    def _on_bk_health_done(self, health: dict):
        self._apply_health_result(
            health, self._bk_h_key, self._bk_h_mag,
            self._bk_h_vae, self._bk_h_alpha, self._bk_h_confirmed)

    def _on_ex_lora_health(self, result: dict):
        """Display LoRA health analysis in the Extract tab results area."""
        STATUS_COLORS = {"pass": GRN, "warn": AMB, "fail": RED, "info": ACC}
        STATUS_ICONS  = {"pass": "✓", "warn": "⚠", "fail": "✗", "info": "ℹ"}

        # Clear previous results
        while self._ex_health_layout.count():
            item = self._ex_health_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        overall = result.get("overall", "fail")
        checks  = result.get("checks", [])
        meta    = result.get("meta", {})
        model_type = result.get("model_type", "?")

        badge_colors = {"pass": GRN, "warn": AMB, "fail": RED}
        badge_texts  = {"pass": "✓  PASS", "warn": "⚠  WARN", "fail": "✗  FAIL"}

        badge_row = QHBoxLayout()
        badge_row.setSpacing(12)
        badge = QLabel(badge_texts.get(overall, overall.upper()))
        badge.setStyleSheet(
            f"color:{PRI}; background:{badge_colors.get(overall, MUT)};"
            f" border-radius:5px; padding:4px 14px;"
            f" font-family:{FONT}; font-size:{FONT_MD}px; font-weight:bold;")
        badge_row.addWidget(badge)
        badge_row.addWidget(_lbl(
            f"Detected: {model_type.upper()}  ·  "
            f"Rank: {meta.get('rank','?')}  ·  Layers: {meta.get('layers','?')}  ·  "
            f"Size: {meta.get('size_mb','?')}",
            SEC, FONT_SM))
        badge_row.addStretch(1)
        bw = QWidget(); bw.setStyleSheet("background:transparent;"); bw.setLayout(badge_row)
        self._ex_health_layout.addWidget(bw)

        checks_card = QFrame()
        checks_card.setStyleSheet(
            f"QFrame {{ background:{CAR}; border-radius:6px; border:1px solid {MUT}; }}")
        cl = QVBoxLayout(checks_card)
        cl.setContentsMargins(12, 10, 12, 10)
        cl.setSpacing(3)

        for i, check in enumerate(checks):
            row = QFrame()
            row.setStyleSheet(
                f"QFrame {{ background:{BG if i%2==0 else CAR}; border-radius:3px; }}")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(6, 5, 6, 5)
            rl.setSpacing(8)
            sc = STATUS_COLORS.get(check["status"], SEC)
            si = STATUS_ICONS.get(check["status"], "?")
            icon = QLabel(si)
            icon.setFixedWidth(16)
            icon.setStyleSheet(
                f"color:{sc}; font-family:{FONT}; font-size:{FONT_MD}px;"
                f" font-weight:bold; background:transparent;")
            rl.addWidget(icon)
            name = _lbl(check["label"], PRI, FONT_SM, bold=True)
            name.setFixedWidth(150)
            rl.addWidget(name)
            det = QLabel(check["detail"])
            det.setWordWrap(True)
            det.setStyleSheet(
                f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
            rl.addWidget(det, stretch=1)
            cl.addWidget(row)

        self._ex_health_layout.addWidget(checks_card)
        self._ex_health_card.setVisible(True)

    def _update_ck_param_state(self):
        method        = self._ck_method.currentText()
        needs_beta    = method in {"TIES", "DARE+TIES"}
        needs_density = method in {"DARE", "DARE+TIES"}
        active_col    = SEC
        dim_col       = MUT
        spin_active   = (
            f"QDoubleSpinBox {{ background:{BG}; color:{PRI}; border:1px solid {MUT};"
            f" border-radius:4px; padding:2px 6px; font-family:{FONT}; font-size:{FONT_MD}px; }}")
        spin_dim = (
            f"QDoubleSpinBox {{ background:{BG}; color:{MUT}; border:1px solid {BG};"
            f" border-radius:4px; padding:2px 6px; font-family:{FONT}; font-size:{FONT_MD}px; }}")

        self._ck_beta.setEnabled(needs_beta)
        self._ck_beta.setStyleSheet(spin_active if needs_beta else spin_dim)
        self._ck_beta_lbl.setStyleSheet(
            f"color:{active_col if needs_beta else dim_col}; font-family:{FONT};"
            f" font-size:{FONT_MD}px; background:transparent; border:none;")

        self._ck_density.setEnabled(needs_density)
        self._ck_density.setStyleSheet(spin_active if needs_density else spin_dim)
        self._ck_density_lbl.setStyleSheet(
            f"color:{active_col if needs_density else dim_col}; font-family:{FONT};"
            f" font-size:{FONT_MD}px; background:transparent; border:none;")

        needs_c = method in {"Add Difference", "TIES", "DARE", "DARE+TIES"}
        self._ck_c.setEnabled(needs_c)
        self._ck_c_btn.setEnabled(needs_c)
        dim_field = (
            f"QLineEdit {{ background:{BG}; color:{MUT}; border:1px solid {BG};"
            f" border-radius:4px; padding:4px 8px; font-family:{FONT}; font-size:{FONT_MD}px; }}")
        self._ck_c.setStyleSheet(_field_style() if needs_c else dim_field)

    def _reload_ck_preset_combo(self):
        """Rebuild preset dropdown. All presets visible; method is in the name."""
        combo = self._ck_preset
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("── select ──")

        def _sep(text: str):
            combo.addItem(text)
            item = combo.model().item(combo.count() - 1)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled
                                       & ~Qt.ItemFlag.ItemIsSelectable)
            item.setForeground(combo.palette().mid())

        _sep("── Built-in ──")
        for name in _CK_PRESETS_BUILTIN:
            combo.addItem(name)

        user = self._cfg.get("user_presets", {})
        if user:
            _sep("── My Presets ──")
            for name in user:
                combo.addItem(name)

        combo.blockSignals(False)
        combo.setCurrentIndex(0)
        if hasattr(self, "_ck_preset_del"):
            self._ck_preset_del.setEnabled(False)

    def _apply_ck_preset(self, index: int):
        if index <= 0:
            self._ck_preset_del.setEnabled(False)
            return

        name = self._ck_preset.itemText(index)
        p = _CK_PRESETS_BUILTIN.get(name) or self._cfg.get("user_presets", {}).get(name)
        if p is None:
            self._ck_preset_del.setEnabled(False)
            return

        # Apply method + values — preset is a complete setup
        self._ck_method.blockSignals(True)
        self._ck_method.setCurrentText(p["method"])
        self._ck_method.blockSignals(False)
        self._ck_alpha.setValue(p["alpha"])
        self._ck_beta.setValue(p["beta"])
        self._ck_density.setValue(p["density"])
        self._update_ck_param_state()

        is_selective = bool(p.get("selective", False))
        self._ck_selective_chk.blockSignals(True)
        self._ck_selective_chk.setChecked(is_selective)
        self._ck_selective_chk.blockSignals(False)
        self._ck_sel_panel.setVisible(is_selective)
        self._ck_method.setEnabled(not is_selective)
        if is_selective:
            self._ck_te_alpha.setValue(p.get("te_alpha", 0.0))
            self._ck_ca_alpha.setValue(p.get("ca_alpha", 0.0))
            self._ck_sa_alpha.setValue(p.get("sa_alpha", 0.75))
            self._ck_ff_alpha.setValue(p.get("ff_alpha", 0.75))
            self._ck_other_alpha.setValue(p.get("other_alpha", 0.0))
            self._ck_method_help.setText(
                _CK_SELECTIVE_PRESET_HELP.get(name, _CK_SELECTIVE_HELP))
        else:
            self._ck_method_help.setText(_CK_METHOD_HELP.get(p["method"], ""))

        is_user = name in self._cfg.get("user_presets", {})
        self._ck_preset_del.setEnabled(is_user)

    def _save_ck_user_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in _CK_PRESETS_BUILTIN:
            QMessageBox.warning(self, "Name Taken",
                                f"'{name}' is a built-in preset and cannot be overwritten.")
            return
        user = self._cfg.setdefault("user_presets", {})
        user[name] = {
            "method":  self._ck_method.currentText(),
            "alpha":   self._ck_alpha.value(),
            "beta":    self._ck_beta.value(),
            "density": self._ck_density.value(),
        }
        save_json(_MERGE_CFG, self._cfg)
        self._reload_ck_preset_combo()
        # Select the newly saved preset
        idx = self._ck_preset.findText(name)
        if idx >= 0:
            self._ck_preset.setCurrentIndex(idx)

    def _delete_ck_user_preset(self):
        name = self._ck_preset.currentText()
        user = self._cfg.get("user_presets", {})
        if name not in user:
            return
        if QMessageBox.question(
                self, "Delete Preset", f"Delete preset '{name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        del user[name]
        save_json(_MERGE_CFG, self._cfg)
        self._reload_ck_preset_combo()

    # ══════════════════════════════════════════════════════════════════════════
    # Checkpoint Merge tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_checkpoint_tab(self):
        from .merge_methods import METHODS, THREE_MODEL_METHODS, METHOD_DESCRIPTIONS

        w      = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # ── Input files card ───────────────────────────────────────────────
        inputs = _card()
        il = QVBoxLayout(inputs)
        il.setContentsMargins(16, 14, 16, 14)
        il.setSpacing(12)
        il.addWidget(_lbl("Input Models", PRI, FONT_MD, bold=True))

        self._ck_a = QLineEdit(self._cfg.get("ckpt_model_a", ""))
        self._ck_a.setStyleSheet(_field_style())
        self._ck_a.setPlaceholderText("Model A  (.safetensors)")
        btn_a = _browse_btn()
        btn_a.clicked.connect(lambda: self._browse_model(self._ck_a, "ckpt_model_a"))
        il.addLayout(_file_row("Model A", self._ck_a, btn_a))

        self._ck_b = QLineEdit(self._cfg.get("ckpt_model_b", ""))
        self._ck_b.setStyleSheet(_field_style())
        self._ck_b.setPlaceholderText("Model B  (.safetensors)")
        btn_b = _browse_btn()
        btn_b.clicked.connect(lambda: self._browse_model(self._ck_b, "ckpt_model_b"))
        il.addLayout(_file_row("Model B", self._ck_b, btn_b))

        self._ck_c = QLineEdit(self._cfg.get("ckpt_model_c", ""))
        self._ck_c.setStyleSheet(_field_style())
        self._ck_c.setPlaceholderText("Optional — required for Add Difference, TIES, DARE, DARE+TIES")
        self._ck_c_btn = _browse_btn()
        self._ck_c_btn.clicked.connect(lambda: self._browse_model(self._ck_c, "ckpt_model_c"))
        il.addLayout(_file_row(
            "Model C — Original Baseline Model  (e.g. SDXL, Illustrious, Pony)",
            self._ck_c, self._ck_c_btn))

        self._ck_vae = QLineEdit(self._cfg.get("ckpt_vae", ""))
        self._ck_vae.setStyleSheet(_field_style())
        self._ck_vae.setPlaceholderText("Optional — copies VAE weights from this file into merged output")
        btn_vae = _browse_btn()
        btn_vae.clicked.connect(lambda: self._browse_model(self._ck_vae, "ckpt_vae"))
        il.addLayout(_file_row("Replace VAE  (optional)", self._ck_vae, btn_vae))

        layout.addWidget(inputs)

        # ── Method & parameters card ───────────────────────────────────────
        params = _card()
        pl = QVBoxLayout(params)
        pl.setContentsMargins(16, 14, 16, 14)
        pl.setSpacing(12)
        pl.addWidget(_lbl("Method & Parameters", PRI, FONT_MD, bold=True))

        # Two-column: left = controls, right = method explanation
        cols = QHBoxLayout()
        cols.setSpacing(16)

        # ── Left column ───────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(10)

        # Method
        self._ck_method = QComboBox()
        self._ck_method.setStyleSheet(_combo_style())
        self._ck_method.setFixedWidth(180)
        for m in METHODS:
            self._ck_method.addItem(m)
        self._ck_method.setCurrentText(self._cfg.get("ckpt_method", "Weighted Sum"))
        self._ck_method.currentTextChanged.connect(
            lambda t: (self._update_ck_param_state(),
                       self._ck_method_help.setText(_CK_METHOD_HELP.get(t, ""))))
        method_row = QHBoxLayout()
        method_row.setSpacing(6)
        method_row.addWidget(_lbl("Method", SEC, FONT_SM))
        method_row.addWidget(self._ck_method)
        method_row.addStretch()
        left.addLayout(method_row)

        # Preset
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(_lbl("Preset", SEC, FONT_SM))
        self._ck_preset = QComboBox()
        self._ck_preset.setStyleSheet(_combo_style())
        self._ck_preset.setFixedWidth(200)
        preset_row.addWidget(self._ck_preset)
        self._ck_preset_save = QPushButton("Save")
        self._ck_preset_save.setStyleSheet(_btn_style())
        self._ck_preset_save.setFixedWidth(52)
        self._ck_preset_save.setToolTip("Save current values as a preset")
        preset_row.addWidget(self._ck_preset_save)
        self._ck_preset_del = QPushButton("Delete")
        self._ck_preset_del.setStyleSheet(_btn_style(RED))
        self._ck_preset_del.setFixedWidth(60)
        self._ck_preset_del.setEnabled(False)
        self._ck_preset_del.setToolTip("Delete selected user preset")
        preset_row.addWidget(self._ck_preset_del)
        preset_row.addStretch()
        left.addLayout(preset_row)
        left.addWidget(_lbl("Presets override alpha / beta / density", MUT, FONT_SM))

        self._reload_ck_preset_combo()

        spin_style = (
            f"QDoubleSpinBox {{ background:{BG}; color:{PRI}; border:1px solid {MUT};"
            f" border-radius:4px; padding:2px 6px; font-family:{FONT}; font-size:{FONT_MD}px; }}")

        # Alpha
        al = QHBoxLayout()
        al.addWidget(_lbl("Alpha"))
        al.addSpacing(8)
        self._ck_alpha = QDoubleSpinBox()
        self._ck_alpha.setRange(0.0, 1.0)
        self._ck_alpha.setSingleStep(0.05)
        self._ck_alpha.setDecimals(2)
        self._ck_alpha.setValue(self._cfg.get("ckpt_alpha", 0.5))
        self._ck_alpha.setStyleSheet(spin_style)
        self._ck_alpha.setFixedWidth(80)
        al.addWidget(self._ck_alpha)
        al.addStretch()
        left.addLayout(al)
        _ah = _lbl("Blend ratio — 0.00 = pure Model A, 1.00 = pure Model B", MUT, FONT_SM)
        _ah.setContentsMargins(4, 0, 0, 2)
        left.addWidget(_ah)

        # Beta
        bl = QHBoxLayout()
        self._ck_beta_lbl = _lbl("Beta")
        bl.addWidget(self._ck_beta_lbl)
        bl.addSpacing(8)
        self._ck_beta = QDoubleSpinBox()
        self._ck_beta.setRange(0.0, 1.0)
        self._ck_beta.setSingleStep(0.05)
        self._ck_beta.setDecimals(2)
        self._ck_beta.setValue(self._cfg.get("ckpt_beta", 0.5))
        self._ck_beta.setStyleSheet(spin_style)
        self._ck_beta.setFixedWidth(80)
        bl.addWidget(self._ck_beta)
        bl.addStretch()
        left.addLayout(bl)
        _bh = _lbl("Increase to keep more delta weights — decrease for stronger trimming", MUT, FONT_SM)
        _bh.setContentsMargins(4, 0, 0, 2)
        left.addWidget(_bh)

        # Density
        dl = QHBoxLayout()
        self._ck_density_lbl = _lbl("Density")
        dl.addWidget(self._ck_density_lbl)
        dl.addSpacing(8)
        self._ck_density = QDoubleSpinBox()
        self._ck_density.setRange(0.0, 1.0)
        self._ck_density.setSingleStep(0.05)
        self._ck_density.setDecimals(2)
        self._ck_density.setValue(self._cfg.get("ckpt_density", 0.75))
        self._ck_density.setStyleSheet(spin_style)
        self._ck_density.setFixedWidth(80)
        dl.addWidget(self._ck_density)
        dl.addStretch()
        left.addLayout(dl)
        _dh = _lbl("Increase for less random pruning — decrease for stronger parameter decorrelation", MUT, FONT_SM)
        _dh.setContentsMargins(4, 0, 0, 2)
        left.addWidget(_dh)

        # Precision
        prec_l = QHBoxLayout()
        prec_l.addWidget(_lbl("Precision", SEC, FONT_SM))
        prec_l.addSpacing(8)
        self._ck_prec = QComboBox()
        self._ck_prec.setStyleSheet(_combo_style())
        for p in ("fp16", "bf16", "fp32"):
            self._ck_prec.addItem(p)
        self._ck_prec.setCurrentText(self._cfg.get("ckpt_precision", "fp16"))
        self._ck_prec.setFixedWidth(100)
        prec_l.addWidget(self._ck_prec)
        prec_l.addStretch()
        left.addLayout(prec_l)

        # ── Selective merge mode ──────────────────────────────────────────
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color:{MUT}; background:{MUT};")
        div.setFixedHeight(1)
        left.addWidget(div)

        self._ck_selective_chk = QCheckBox("Selective Mode  (per-group alpha)")
        self._ck_selective_chk.setChecked(self._cfg.get("ckpt_selective", False))
        self._ck_selective_chk.setStyleSheet(
            f"QCheckBox {{ color:{ACC}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" font-weight:bold; background:transparent; }}"
            f"QCheckBox::indicator {{ width:14px; height:14px; }}"
            f"QCheckBox::indicator:unchecked {{ border:2px solid {MUT};"
            f" border-radius:3px; background:{BG}; }}"
            f"QCheckBox::indicator:checked {{ border:2px solid {ACC};"
            f" border-radius:3px; background:{ACC}; }}")
        left.addWidget(self._ck_selective_chk)

        # Selective alpha panel (shown only when checkbox is on)
        self._ck_sel_panel = QWidget()
        self._ck_sel_panel.setStyleSheet("background:transparent;")
        sel_grid = QVBoxLayout(self._ck_sel_panel)
        sel_grid.setContentsMargins(0, 4, 0, 0)
        sel_grid.setSpacing(6)

        sel_spin_style = (
            f"QDoubleSpinBox {{ background:{BG}; color:{PRI}; border:1px solid {ACC};"
            f" border-radius:4px; padding:2px 6px; font-family:{FONT}; font-size:{FONT_SM}px; }}")

        def _sel_row(label: str, cfg_key: str, default: float, hint: str = ""):
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = _lbl(label, SEC, FONT_SM)
            lbl.setFixedWidth(160)
            row.addWidget(lbl)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(self._cfg.get(cfg_key, default))
            spin.setStyleSheet(sel_spin_style)
            spin.setFixedWidth(72)
            row.addWidget(spin)
            row.addStretch()
            sel_grid.addLayout(row)
            if hint:
                h = _lbl(hint, MUT, FONT_SM)
                h.setContentsMargins(4, 0, 0, 2)
                sel_grid.addWidget(h)
            return spin

        self._ck_te_alpha = _sel_row(
            "Text Encoders α", "ckpt_te_alpha", 0.00,
            "Increase to blend Model B's vocabulary and prompt understanding — keep close to Cross-Attention α")
        self._ck_ca_alpha = _sel_row(
            "Cross-Attention α", "ckpt_ca_alpha", 0.00,
            "Keep within ~0.10 of Text Encoders α — large divergence breaks text conditioning")
        self._ck_sa_alpha = _sel_row(
            "Self-Attention α", "ckpt_sa_alpha", 0.75,
            "Increase to adopt Model B's composition and subject layout")
        self._ck_ff_alpha = _sel_row(
            "Feedforward α", "ckpt_ff_alpha", 0.75,
            "Increase to adopt Model B's rendering quality, texture, and color")
        self._ck_other_alpha = _sel_row(
            "Other (VAE, norms) α", "ckpt_other_alpha", 0.00,
            "Increase to blend VAE and norms — usually keep at 0.00")

        left.addWidget(self._ck_sel_panel)
        left.addStretch()

        def _on_selective_toggled(checked: bool):
            self._ck_sel_panel.setVisible(checked)
            if checked:
                self._ck_method.blockSignals(True)
                self._ck_method.setCurrentText("Weighted Sum")
                self._ck_method.blockSignals(False)
                self._ck_method.setEnabled(False)
                self._ck_method_help.setText(_CK_SELECTIVE_HELP)
            else:
                self._ck_method.setEnabled(True)
                self._ck_method_help.setText(
                    _CK_METHOD_HELP.get(self._ck_method.currentText(), ""))
            self._update_ck_param_state()

        self._ck_selective_chk.toggled.connect(_on_selective_toggled)
        if self._cfg.get("ckpt_selective", False):
            self._ck_method.setEnabled(False)

        # ── Right column — method explanation ─────────────────────────────
        right = QFrame()
        right.setStyleSheet(
            f"QFrame {{ background:{BG}; border-radius:6px; border:1px solid {MUT}; }}")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(14, 12, 14, 12)
        self._ck_method_help = QLabel(
            _CK_METHOD_HELP.get(self._ck_method.currentText(), ""))
        self._ck_method_help.setWordWrap(True)
        self._ck_method_help.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._ck_method_help.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")
        rl.addWidget(self._ck_method_help)

        cols.addLayout(left, stretch=0)
        cols.addWidget(right, stretch=1)
        pl.addLayout(cols)

        self._ck_preset.currentIndexChanged.connect(self._apply_ck_preset)
        self._ck_preset_save.clicked.connect(self._save_ck_user_preset)
        self._ck_preset_del.clicked.connect(self._delete_ck_user_preset)
        self._update_ck_param_state()

        layout.addWidget(params)

        # ── Output card ────────────────────────────────────────────────────
        out_card = _card()
        ol = QVBoxLayout(out_card)
        ol.setContentsMargins(16, 14, 16, 14)
        ol.setSpacing(12)
        ol.addWidget(_lbl("Output", PRI, FONT_MD, bold=True))
        self._ck_out = QLineEdit(self._cfg.get("ckpt_output", ""))
        self._ck_out.setStyleSheet(_field_style())
        self._ck_out.setPlaceholderText("Output path  (.safetensors)")
        btn_out = _browse_btn()
        btn_out.setText("Save As")
        btn_out.setFixedWidth(90)
        btn_out.clicked.connect(lambda: self._save_as(self._ck_out, "ckpt_output"))
        ol.addLayout(_file_row("Output File", self._ck_out, btn_out))
        layout.addWidget(out_card)

        # ── Run card ───────────────────────────────────────────────────────
        run_card = _card()
        rl = QVBoxLayout(run_card)
        rl.setContentsMargins(16, 14, 16, 14)
        rl.setSpacing(8)

        self._ck_bar, self._ck_status = _progress_row()
        rl.addWidget(self._ck_bar)
        rl.addWidget(self._ck_status)

        self._ck_run = QPushButton("Merge Checkpoints")
        self._ck_run.setStyleSheet(_btn_style())
        self._ck_run.clicked.connect(self._run_checkpoint_merge)
        rl.addWidget(self._ck_run)

        # ── Health check indicators (auto-populated after merge) ──────────
        h_row = QHBoxLayout()
        h_row.setSpacing(20)
        self._ck_h_key       = self._mk_health_lbl("Key count delta")
        self._ck_h_mag       = self._mk_health_lbl("Weight magnitude drift")
        self._ck_h_vae       = self._mk_health_lbl("VAE integrity")
        self._ck_h_alpha     = self._mk_health_lbl("Alpha key consistency")
        self._ck_h_confirmed = self._mk_health_lbl("Merge Confirmed")
        for lbl in (self._ck_h_key, self._ck_h_mag, self._ck_h_vae,
                    self._ck_h_alpha, self._ck_h_confirmed):
            h_row.addWidget(lbl)
        h_row.addStretch(1)
        rl.addLayout(h_row)

        layout.addWidget(run_card)
        layout.addStretch(1)

        self._tabs.addTab(scroll, "Checkpoint Merge")

    # ══════════════════════════════════════════════════════════════════════════
    # LoRA Merge tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_lora_tab(self):
        w      = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # ── Input LoRAs card ───────────────────────────────────────────────
        inputs = _card()
        il = QVBoxLayout(inputs)
        il.setContentsMargins(16, 14, 16, 14)
        il.setSpacing(10)
        il.addWidget(_lbl("Input LoRAs", PRI, FONT_MD, bold=True))

        spin_style = (
            f"QDoubleSpinBox {{ background:{BG}; color:{PRI}; border:1px solid {MUT};"
            f" border-radius:4px; padding:2px 6px; font-family:{FONT}; font-size:{FONT_MD}px; }}")
        slot_frame_style = (
            f"QFrame {{ background:{BG}; border-radius:6px; border:1px solid {MUT}; }}")

        def _slot_frame(label: str, path_key: str, type_key: str, weight_key: str
                        ) -> tuple[QFrame, QLineEdit, QComboBox, QDoubleSpinBox]:
            frame = QFrame()
            frame.setStyleSheet(slot_frame_style)
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(10, 8, 10, 8)
            fl.setSpacing(6)

            fl.addWidget(_lbl(label, ACC, FONT_SM, bold=True))

            path_row = QHBoxLayout()
            path_row.setSpacing(4)
            path_edit = QLineEdit(self._cfg.get(path_key, ""))
            path_edit.setStyleSheet(_field_style())
            path_edit.setPlaceholderText(".safetensors")
            path_row.addWidget(path_edit, stretch=1)
            btn = _browse_btn()
            btn.clicked.connect(
                lambda _=False, f=path_edit, k=path_key: self._browse_lora(f, k))
            path_row.addWidget(btn)
            fl.addLayout(path_row)

            tw_row = QHBoxLayout()
            tw_row.setSpacing(6)
            type_combo = QComboBox()
            type_combo.setStyleSheet(_combo_style())
            for t in _LORA_TYPES:
                type_combo.addItem(t)
            type_combo.setCurrentText(self._cfg.get(type_key, "Character"))
            tw_row.addWidget(type_combo, stretch=1)
            w_spin = QDoubleSpinBox()
            w_spin.setRange(0.0, 2.0)
            w_spin.setSingleStep(0.05)
            w_spin.setDecimals(2)
            w_spin.setValue(self._cfg.get(weight_key,
                                          _MERGE_PRESET.get("Character", 0.75)))
            w_spin.setStyleSheet(spin_style)
            w_spin.setFixedWidth(72)
            w_spin.setToolTip("Merge weight")
            tw_row.addWidget(w_spin)
            fl.addLayout(tw_row)

            return frame, path_edit, type_combo, w_spin

        # Row 1: A + B side by side
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        fa, self._lo_a, self._lo_ta, self._lo_wa = _slot_frame(
            "LoRA A", "lora_a", "lora_type_a", "lora_weight_a")
        fb, self._lo_b, self._lo_tb, self._lo_wb = _slot_frame(
            "LoRA B", "lora_b", "lora_type_b", "lora_weight_b")
        row1.addWidget(fa)
        row1.addWidget(fb)
        il.addLayout(row1)

        # Row 2: C + D side by side (optional)
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        fc, self._lo_c, self._lo_tc, self._lo_wc = _slot_frame(
            "LoRA C  (optional)", "lora_c", "lora_type_c", "lora_weight_c")
        fd, self._lo_d, self._lo_td, self._lo_wd = _slot_frame(
            "LoRA D  (optional)", "lora_d", "lora_type_d", "lora_weight_d")
        row2.addWidget(fc)
        row2.addWidget(fd)
        il.addLayout(row2)

        # Warning bar
        self._lo_warn = QLabel("")
        self._lo_warn.setWordWrap(True)
        self._lo_warn.setStyleSheet(
            f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")
        il.addWidget(self._lo_warn)
        layout.addWidget(inputs)

        # Wire up type→ratio presets and live warning
        _lo_slots = [
            (self._lo_ta, self._lo_wa),
            (self._lo_tb, self._lo_wb),
            (self._lo_tc, self._lo_wc),
            (self._lo_td, self._lo_wd),
        ]

        def _update_lo_warn():
            _ratio_warning(
                [c.currentText() for c, _ in _lo_slots],
                [s.value()        for _, s in _lo_slots],
                _MERGE_MAX, self._lo_warn)

        def _type_changed_lo(combo, spin, _text):
            spin.setValue(_MERGE_PRESET.get(combo.currentText(), 0.65))
            _update_lo_warn()

        for combo, spin in _lo_slots:
            combo.currentTextChanged.connect(
                lambda t, c=combo, s=spin: _type_changed_lo(c, s, t))
            spin.valueChanged.connect(lambda _: _update_lo_warn())
        _update_lo_warn()

        # ── Output ────────────────────────────────────────────────────────
        out_card = _card()
        ol = QVBoxLayout(out_card)
        ol.setContentsMargins(16, 14, 16, 14)
        ol.setSpacing(10)
        ol.addWidget(_lbl("Output", PRI, FONT_MD, bold=True))
        self._lo_out = QLineEdit(self._cfg.get("lora_output", ""))
        self._lo_out.setStyleSheet(_field_style())
        self._lo_out.setPlaceholderText("Output LoRA (.safetensors)")
        btn_lo = _browse_btn()
        btn_lo.setText("Save As")
        btn_lo.setFixedWidth(90)
        btn_lo.clicked.connect(lambda: self._save_as(self._lo_out, "lora_output"))
        ol.addLayout(_file_row("Output File", self._lo_out, btn_lo))
        layout.addWidget(out_card)

        # ── Run ───────────────────────────────────────────────────────────
        run_card = _card()
        rl = QVBoxLayout(run_card)
        rl.setContentsMargins(16, 14, 16, 14)
        rl.setSpacing(8)
        self._lo_bar, self._lo_status = _progress_row()
        rl.addWidget(self._lo_bar)
        rl.addWidget(self._lo_status)
        self._lo_run = QPushButton("Merge LoRAs")
        self._lo_run.setStyleSheet(_btn_style())
        self._lo_run.clicked.connect(self._run_lora_merge)
        rl.addWidget(self._lo_run)
        layout.addWidget(run_card)
        layout.addStretch(1)

        self._tabs.addTab(scroll, "LoRA Merge")

    # ══════════════════════════════════════════════════════════════════════════
    # Bake LoRA tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_bake_tab(self):
        w      = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        info = _lbl(
            "Bake up to 4 LoRA files permanently into a checkpoint in a single pass.  "
            "Select each LoRA's type to get a suggested ratio and live combined-load warning.",
            SEC, FONT_SM)
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── Checkpoint ──────────────────────────────────────────────────────
        ck_card = _card()
        cl = QVBoxLayout(ck_card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(10)
        cl.addWidget(_lbl("Base Checkpoint", PRI, FONT_MD, bold=True))
        self._bk_ckpt = QLineEdit(self._cfg.get("bake_ckpt", ""))
        self._bk_ckpt.setStyleSheet(_field_style())
        self._bk_ckpt.setPlaceholderText("Base Checkpoint (.safetensors)")
        btn_bc = _browse_btn()
        btn_bc.clicked.connect(lambda: self._browse_model(self._bk_ckpt, "bake_ckpt"))
        cl.addLayout(_file_row("", self._bk_ckpt, btn_bc))

        pr = QHBoxLayout()
        pr.addWidget(_lbl("Output Precision", SEC, FONT_SM))
        pr.addStretch()
        self._bk_prec = QComboBox()
        self._bk_prec.setStyleSheet(_combo_style())
        for p in ("fp16", "bf16", "fp32"):
            self._bk_prec.addItem(p)
        self._bk_prec.setCurrentText(self._cfg.get("bake_precision", "fp16"))
        self._bk_prec.setFixedWidth(100)
        pr.addWidget(self._bk_prec)
        cl.addLayout(pr)
        layout.addWidget(ck_card)

        # ── LoRA slots ───────────────────────────────────────────────────────
        loras_card = _card()
        ll = QVBoxLayout(loras_card)
        ll.setContentsMargins(16, 14, 16, 14)
        ll.setSpacing(10)
        ll.addWidget(_lbl("LoRAs  (type → ratio auto-fills)", PRI, FONT_MD, bold=True))

        spin_style = (
            f"QDoubleSpinBox {{ background:{BG}; color:{PRI}; border:1px solid {MUT};"
            f" border-radius:4px; padding:2px 6px; font-family:{FONT}; font-size:{FONT_MD}px; }}")

        self._bk_loras: list[tuple[QLineEdit, QComboBox, QDoubleSpinBox]] = []

        slot_frame_style = (
            f"QFrame {{ background:{BG}; border-radius:6px; border:1px solid {MUT}; }}")

        def _bk_slot(label: str, path_key: str, type_key: str, ratio_key: str) -> QFrame:
            frame = QFrame()
            frame.setStyleSheet(slot_frame_style)
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(10, 8, 10, 8)
            fl.setSpacing(6)

            fl.addWidget(_lbl(label, ACC, FONT_SM, bold=True))

            path_row = QHBoxLayout()
            path_row.setSpacing(4)
            path_edit = QLineEdit(self._cfg.get(path_key, ""))
            path_edit.setStyleSheet(_field_style())
            path_edit.setPlaceholderText(".safetensors")
            path_row.addWidget(path_edit, stretch=1)
            btn = _browse_btn()
            btn.clicked.connect(
                lambda _=False, f=path_edit, k=path_key: self._browse_lora(f, k))
            path_row.addWidget(btn)
            fl.addLayout(path_row)

            tw_row = QHBoxLayout()
            tw_row.setSpacing(6)
            type_combo = QComboBox()
            type_combo.setStyleSheet(_combo_style())
            for t in _LORA_TYPES:
                type_combo.addItem(t)
            type_combo.setCurrentText(self._cfg.get(type_key, "Character"))
            tw_row.addWidget(type_combo, stretch=1)
            ratio_spin = QDoubleSpinBox()
            ratio_spin.setRange(0.0, 2.0)
            ratio_spin.setSingleStep(0.05)
            ratio_spin.setDecimals(2)
            ratio_spin.setValue(self._cfg.get(ratio_key, _BAKE_PRESET.get("Character", 0.85)))
            ratio_spin.setStyleSheet(spin_style)
            ratio_spin.setFixedWidth(72)
            ratio_spin.setToolTip("Bake ratio")
            tw_row.addWidget(ratio_spin)
            fl.addLayout(tw_row)

            self._bk_loras.append((path_edit, type_combo, ratio_spin))
            return frame

        row1 = QHBoxLayout()
        row1.setSpacing(10)
        row1.addWidget(_bk_slot("LoRA 1",             "bake_lora_1", "bake_type_1", "bake_ratio_1"))
        row1.addWidget(_bk_slot("LoRA 2  (optional)", "bake_lora_2", "bake_type_2", "bake_ratio_2"))
        ll.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)
        row2.addWidget(_bk_slot("LoRA 3  (optional)", "bake_lora_3", "bake_type_3", "bake_ratio_3"))
        row2.addWidget(_bk_slot("LoRA 4  (optional)", "bake_lora_4", "bake_type_4", "bake_ratio_4"))
        ll.addLayout(row2)

        # Warning bar
        self._bk_warn = QLabel("")
        self._bk_warn.setWordWrap(True)
        self._bk_warn.setStyleSheet(
            f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")
        ll.addWidget(self._bk_warn)
        layout.addWidget(loras_card)

        def _update_bk_warn():
            types  = [combo.currentText() for _, combo, _ in self._bk_loras]
            ratios = [spin.value()        for _, _,     spin in self._bk_loras]
            _ratio_warning(types, ratios, _BAKE_MAX, self._bk_warn)

        def _type_changed_bk(combo, spin, _text):
            spin.setValue(_BAKE_PRESET.get(combo.currentText(), 0.85))
            _update_bk_warn()

        for _, combo, spin in self._bk_loras:
            combo.currentTextChanged.connect(
                lambda t, c=combo, s=spin: _type_changed_bk(c, s, t))
            spin.valueChanged.connect(lambda _: _update_bk_warn())
        _update_bk_warn()

        # ── Output ──────────────────────────────────────────────────────────
        out_card = _card()
        ol = QVBoxLayout(out_card)
        ol.setContentsMargins(16, 14, 16, 14)
        ol.setSpacing(10)
        ol.addWidget(_lbl("Output", PRI, FONT_MD, bold=True))
        self._bk_out = QLineEdit(self._cfg.get("bake_output", ""))
        self._bk_out.setStyleSheet(_field_style())
        self._bk_out.setPlaceholderText("Output Checkpoint (.safetensors)")
        btn_bo = _browse_btn()
        btn_bo.setText("Save As")
        btn_bo.setFixedWidth(90)
        btn_bo.clicked.connect(lambda: self._save_as(self._bk_out, "bake_output"))
        ol.addLayout(_file_row("Output File", self._bk_out, btn_bo))
        layout.addWidget(out_card)

        # ── Run ─────────────────────────────────────────────────────────────
        run_card = _card()
        rl = QVBoxLayout(run_card)
        rl.setContentsMargins(16, 14, 16, 14)
        rl.setSpacing(8)
        self._bk_bar, self._bk_status = _progress_row()
        rl.addWidget(self._bk_bar)
        rl.addWidget(self._bk_status)
        self._bk_run = QPushButton("Bake LoRAs into Checkpoint")
        self._bk_run.setStyleSheet(_btn_style())
        self._bk_run.clicked.connect(self._run_bake)
        rl.addWidget(self._bk_run)

        bk_h_row = QHBoxLayout()
        bk_h_row.setSpacing(20)
        self._bk_h_key       = self._mk_health_lbl("Key count delta")
        self._bk_h_mag       = self._mk_health_lbl("Weight magnitude drift")
        self._bk_h_vae       = self._mk_health_lbl("VAE integrity")
        self._bk_h_alpha     = self._mk_health_lbl("Alpha key consistency")
        self._bk_h_confirmed = self._mk_health_lbl("Merge Confirmed")
        for lbl in (self._bk_h_key, self._bk_h_mag, self._bk_h_vae,
                    self._bk_h_alpha, self._bk_h_confirmed):
            bk_h_row.addWidget(lbl)
        bk_h_row.addStretch(1)
        rl.addLayout(bk_h_row)

        layout.addWidget(run_card)
        layout.addStretch(1)

        self._tabs.addTab(scroll, "Bake LoRA")

    # ══════════════════════════════════════════════════════════════════════════
    # Help tab
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # Extract LoRA tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_extract_tab(self):
        _RANK_HELP = (
            "RANK  (SVD decomposition depth)\n\n"
            "Controls how faithfully the extracted LoRA reconstructs the original "
            "fine-tuning.  Higher rank = more parameters = larger file = more accurate.\n\n"
            "16 — compact.  Captures dominant style and subject features.  "
            "Good when you only need the broad strokes of a fine-tune.\n\n"
            "32 — balanced default.  Captures most meaningful training signal "
            "with a reasonable file size.  Start here.\n\n"
            "64 — high fidelity.  Recommended when the fine-tune had strong "
            "character detail, complex textures, or multiple concepts.\n\n"
            "128 — near-lossless.  File size approaches the full delta.  "
            "Only worth it if rank 64 still shows visible quality loss.\n\n"
            "Rule of thumb: if the extracted LoRA produces blurry or drifted "
            "results at rank 32, double the rank and compare."
        )
        _CONV_HELP = (
            "CONVOLUTIONAL LAYERS\n\n"
            "When checked, extraction includes Conv2d layers — the spatial "
            "convolutions in the UNet down/up blocks — in addition to the "
            "Linear layers used by attention (Q, K, V projections).\n\n"
            "Include (recommended)\n"
            "Captures the full fine-tuning signal including spatial and "
            "structural features.  Results in more keys in the LoRA file "
            "but significantly better reconstruction fidelity, especially "
            "for DreamBooth or full fine-tunes that modified composition.\n\n"
            "Exclude\n"
            "Attention-only LoRA.  Smaller file, slightly faster to apply.  "
            "Acceptable when the fine-tune was purely stylistic and you can "
            "confirm quality is preserved without the conv keys."
        )

        w      = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        info = _lbl(
            "Extract the difference between a fine-tuned checkpoint and its original "
            "base as a portable LoRA file.  Useful for DreamBooth outputs, full "
            "fine-tunes, or recovering a LoRA from a baked model.",
            SEC, FONT_SM)
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── Input models ──────────────────────────────────────────────────
        inputs = _card()
        il = QVBoxLayout(inputs)
        il.setContentsMargins(16, 14, 16, 14)
        il.setSpacing(12)
        il.addWidget(_lbl("Input Models", PRI, FONT_MD, bold=True))

        self._ex_tuned = QLineEdit(self._cfg.get("extract_tuned", ""))
        self._ex_tuned.setStyleSheet(_field_style())
        self._ex_tuned.setPlaceholderText("Fine-tuned checkpoint  (.safetensors)")
        btn_et = _browse_btn()
        btn_et.clicked.connect(lambda: self._browse_model(self._ex_tuned, "extract_tuned"))
        il.addLayout(_file_row("Model A — Fine-tuned", self._ex_tuned, btn_et))

        self._ex_base = QLineEdit(self._cfg.get("extract_base", ""))
        self._ex_base.setStyleSheet(_field_style())
        self._ex_base.setPlaceholderText("Original base checkpoint  (.safetensors)")
        btn_eb = _browse_btn()
        btn_eb.clicked.connect(lambda: self._browse_model(self._ex_base, "extract_base"))
        il.addLayout(_file_row(
            "Model B — Original Baseline Model  (e.g. SDXL, Illustrious, Pony)",
            self._ex_base, btn_eb))
        layout.addWidget(inputs)

        # ── Parameters card (two-column) ──────────────────────────────────
        params = _card()
        pl = QVBoxLayout(params)
        pl.setContentsMargins(16, 14, 16, 14)
        pl.setSpacing(12)
        pl.addWidget(_lbl("Extraction Parameters", PRI, FONT_MD, bold=True))

        cols = QHBoxLayout()
        cols.setSpacing(16)

        # Left controls
        left = QVBoxLayout()
        left.setSpacing(12)

        rank_row = QHBoxLayout()
        rank_row.setSpacing(8)
        rank_row.addWidget(_lbl("Rank", SEC, FONT_SM))
        self._ex_rank = QComboBox()
        self._ex_rank.setStyleSheet(_combo_style())
        self._ex_rank.setFixedWidth(90)
        for r in (16, 32, 64, 128):
            self._ex_rank.addItem(str(r))
        saved_rank = str(self._cfg.get("extract_rank", 32))
        idx = self._ex_rank.findText(saved_rank)
        self._ex_rank.setCurrentIndex(idx if idx >= 0 else 1)
        rank_row.addWidget(self._ex_rank)
        rank_row.addStretch()
        left.addLayout(rank_row)

        self._ex_conv = QCheckBox("Include Conv layers")
        self._ex_conv.setChecked(self._cfg.get("extract_conv", True))
        self._ex_conv.setStyleSheet(
            f"QCheckBox {{ color:{PRI}; font-family:{FONT}; font-size:{FONT_MD}px; "
            f"background:transparent; }}"
            f"QCheckBox::indicator {{ width:16px; height:16px; }}"
            f"QCheckBox::indicator:unchecked {{ border:2px solid {MUT}; "
            f"border-radius:3px; background:{BG}; }}"
            f"QCheckBox::indicator:checked {{ border:2px solid {ACC}; "
            f"border-radius:3px; background:{ACC}; }}")
        left.addWidget(self._ex_conv)
        left.addStretch()

        # Right help panel
        right = QFrame()
        right.setStyleSheet(
            f"QFrame {{ background:{BG}; border-radius:6px; border:1px solid {MUT}; }}")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(14, 12, 14, 12)
        rl.setSpacing(14)

        rank_lbl = QLabel(_RANK_HELP)
        rank_lbl.setWordWrap(True)
        rank_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        rank_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")
        rl.addWidget(rank_lbl)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color:{MUT}; background:{MUT};")
        div.setFixedHeight(1)
        rl.addWidget(div)

        conv_lbl = QLabel(_CONV_HELP)
        conv_lbl.setWordWrap(True)
        conv_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        conv_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")
        rl.addWidget(conv_lbl)

        cols.addLayout(left, stretch=0)
        cols.addWidget(right, stretch=1)
        pl.addLayout(cols)
        layout.addWidget(params)

        # ── Output ────────────────────────────────────────────────────────
        out_card = _card()
        ol = QVBoxLayout(out_card)
        ol.setContentsMargins(16, 14, 16, 14)
        ol.setSpacing(10)
        ol.addWidget(_lbl("Output", PRI, FONT_MD, bold=True))
        self._ex_out = QLineEdit(self._cfg.get("extract_output", ""))
        self._ex_out.setStyleSheet(_field_style())
        self._ex_out.setPlaceholderText("Output LoRA (.safetensors)")
        btn_eo = _browse_btn()
        btn_eo.setText("Save As")
        btn_eo.setFixedWidth(90)
        btn_eo.clicked.connect(lambda: self._save_as(self._ex_out, "extract_output"))
        ol.addLayout(_file_row("Output File", self._ex_out, btn_eo))
        layout.addWidget(out_card)

        # ── Run ───────────────────────────────────────────────────────────
        run_card = _card()
        rl2 = QVBoxLayout(run_card)
        rl2.setContentsMargins(16, 14, 16, 14)
        rl2.setSpacing(8)
        self._ex_bar, self._ex_status = _progress_row()
        rl2.addWidget(self._ex_bar)
        rl2.addWidget(self._ex_status)
        self._ex_run = QPushButton("Extract LoRA")
        self._ex_run.setStyleSheet(_btn_style())
        self._ex_run.clicked.connect(self._run_extract)
        rl2.addWidget(self._ex_run)
        layout.addWidget(run_card)

        # ── LoRA Health results card (hidden until extract completes) ──────
        self._ex_health_card = _card()
        hc = QVBoxLayout(self._ex_health_card)
        hc.setContentsMargins(16, 12, 16, 12)
        hc.setSpacing(8)
        hc.addWidget(_lbl("Extracted LoRA Health", PRI, FONT_MD, bold=True))
        self._ex_health_layout = QVBoxLayout()
        self._ex_health_layout.setSpacing(6)
        hc.addLayout(self._ex_health_layout)
        self._ex_health_card.setVisible(False)
        layout.addWidget(self._ex_health_card)

        layout.addStretch(1)

        self._tabs.addTab(scroll, "Extract LoRA")

    def _run_extract(self):
        if self._is_busy():
            QMessageBox.warning(self, "Busy", "A merge operation is already running.")
            return

        tuned = self._ex_tuned.text().strip()
        base  = self._ex_base.text().strip()
        out   = self._ex_out.text().strip()

        if not tuned or not os.path.isfile(tuned):
            QMessageBox.warning(self, "Missing", "Fine-tuned model not found.")
            return
        if not base or not os.path.isfile(base):
            QMessageBox.warning(self, "Missing", "Base model not found.")
            return
        if not out:
            QMessageBox.warning(self, "Missing", "Output path is required.")
            return

        self._save_extract_cfg()

        from .merge_engine import extract_lora
        rank      = int(self._ex_rank.currentText())
        conv      = self._ex_conv.isChecked()
        dev       = self._current_device()

        self._ex_run.setEnabled(False)
        self._ex_bar.setValue(0)
        self._ex_status.setText("Starting…")

        worker = _Worker()
        worker.progress.connect(
            lambda i, t, k: self._on_progress(self._ex_bar, self._ex_status, i, t, k))
        worker.finished.connect(
            lambda: self._on_finished(self._ex_run, self._ex_status, self._ex_bar,
                                      f"Saved: {out}"))
        worker.error.connect(
            lambda e: self._on_error(self._ex_run, self._ex_status, e))
        worker.lora_health.connect(self._on_ex_lora_health)
        self._worker = worker

        def _run():
            try:
                extract_lora(tuned, base, out, rank, conv, dev,
                             progress_fn=lambda i, t, k: worker.progress.emit(i, t, k))
                # Auto LoRA health check
                try:
                    from health.health_page import _analyse, THRESHOLD_PRESETS
                    def _thr(mt: str, k: str) -> float:
                        presets = THRESHOLD_PRESETS.get(mt, THRESHOLD_PRESETS["sd15"])
                        return presets.get("Standard", presets[next(iter(presets))])[k]
                    result = _analyse(out, _thr)
                    worker.lora_health.emit(result)
                except Exception:
                    pass
                worker.finished.emit()
            except Exception as exc:
                worker.error.emit(str(exc))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _save_extract_cfg(self):
        self._cfg.update({
            "extract_tuned":  self._ex_tuned.text().strip(),
            "extract_base":   self._ex_base.text().strip(),
            "extract_rank":   int(self._ex_rank.currentText()),
            "extract_conv":   self._ex_conv.isChecked(),
            "extract_output": self._ex_out.text().strip(),
        })
        save_json(_MERGE_CFG, self._cfg)

    def _build_help_tab(self):
        from .merge_methods import METHOD_DESCRIPTIONS

        w      = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        methods_card = _card()
        ml = QVBoxLayout(methods_card)
        ml.setContentsMargins(16, 14, 16, 14)
        ml.setSpacing(10)
        ml.addWidget(_lbl("Merge Methods", PRI, FONT_MD, bold=True))

        for name, desc in METHOD_DESCRIPTIONS.items():
            row = QFrame()
            row.setStyleSheet(f"QFrame {{ background:{BG}; border-radius:4px; border:none; }}")
            rv  = QVBoxLayout(row)
            rv.setContentsMargins(10, 8, 10, 8)
            rv.setSpacing(4)
            rv.addWidget(_lbl(name, ACC, FONT_MD, bold=True))
            d = _lbl(desc, SEC, FONT_SM)
            d.setWordWrap(True)
            rv.addWidget(d)
            ml.addWidget(row)

        layout.addWidget(methods_card)

        ops_card = _card()
        ol = QVBoxLayout(ops_card)
        ol.setContentsMargins(16, 14, 16, 14)
        ol.setSpacing(10)
        ol.addWidget(_lbl("Operations", PRI, FONT_MD, bold=True))

        ops_text = [
            ("Checkpoint Merge",
             "Merges two full model checkpoints (.safetensors) key-by-key.  "
             "Two-model methods (Weighted Sum, Slerp) only need A and B.  "
             "Three-model methods (Add Difference, TIES, DARE, DARE+TIES) also "
             "require a base model C — typically the original base checkpoint that both "
             "A and B were fine-tuned from."),
            ("LoRA Merge",
             "Merges two LoRA adapters by expanding each to its full delta matrix "
             "(up @ down × alpha/dim), summing with per-file weights, then "
             "re-compressing via SVD back into LoRA format.  "
             "Output rank 0 auto-selects the max rank of the inputs."),
            ("Bake LoRA",
             "Permanently adds a LoRA's deltas into a checkpoint.  "
             "Equivalent to running inference with that LoRA at the given multiplier, "
             "but the result is a standalone checkpoint that needs no LoRA file.  "
             "Key mapping is heuristic (lora_unet_* → unet.*); unmapped layers "
             "are silently skipped."),
        ]
        for title, body in ops_text:
            row = QFrame()
            row.setStyleSheet(f"QFrame {{ background:{BG}; border-radius:4px; border:none; }}")
            rv  = QVBoxLayout(row)
            rv.setContentsMargins(10, 8, 10, 8)
            rv.setSpacing(4)
            rv.addWidget(_lbl(title, ACC, FONT_MD, bold=True))
            b = _lbl(body, SEC, FONT_SM)
            b.setWordWrap(True)
            rv.addWidget(b)
            ol.addWidget(row)

        layout.addWidget(ops_card)
        layout.addStretch(1)

        self._tabs.addTab(scroll, "Help")

    # ══════════════════════════════════════════════════════════════════════════
    # File dialog helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_model(self, field: QLineEdit, cfg_key: str):
        start = os.path.dirname(field.text()) if field.text() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", start, "Safetensors (*.safetensors)")
        if path:
            field.setText(path)
            self._cfg[cfg_key] = path
            save_json(_MERGE_CFG, self._cfg)

    def _browse_lora(self, field: QLineEdit, cfg_key: str):
        self._browse_model(field, cfg_key)

    def _save_as(self, field: QLineEdit, cfg_key: str):
        start = os.path.dirname(field.text()) if field.text() else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As", start, "Safetensors (*.safetensors)")
        if path:
            if not path.endswith(".safetensors"):
                path += ".safetensors"
            field.setText(path)
            self._cfg[cfg_key] = path
            save_json(_MERGE_CFG, self._cfg)

    # ══════════════════════════════════════════════════════════════════════════
    # Run logic
    # ══════════════════════════════════════════════════════════════════════════

    def _is_busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_checkpoint_merge(self):
        if self._is_busy():
            QMessageBox.warning(self, "Busy", "A merge operation is already running.")
            return

        a   = self._ck_a.text().strip()
        b   = self._ck_b.text().strip()
        c   = self._ck_c.text().strip() or None
        vae = self._ck_vae.text().strip() or None
        out = self._ck_out.text().strip()

        if not a or not os.path.isfile(a):
            QMessageBox.warning(self, "Missing", "Model A file not found.")
            return
        if not b or not os.path.isfile(b):
            QMessageBox.warning(self, "Missing", "Model B file not found.")
            return
        if c and not os.path.isfile(c):
            QMessageBox.warning(self, "Missing", "Model C file not found.")
            return
        if vae and not os.path.isfile(vae):
            QMessageBox.warning(self, "Missing", "Replace VAE file not found.")
            return
        if not out:
            QMessageBox.warning(self, "Missing", "Output path is required.")
            return

        self._save_ckpt_cfg()

        from .merge_engine import merge_checkpoints, check_checkpoint_health
        method  = self._ck_method.currentText()
        alpha   = self._ck_alpha.value()
        beta    = self._ck_beta.value()
        density = self._ck_density.value()
        prec    = self._ck_prec.currentText()
        dev     = self._current_device()
        group_alphas = {
            "te":    self._ck_te_alpha.value(),
            "ca":    self._ck_ca_alpha.value(),
            "sa":    self._ck_sa_alpha.value(),
            "ff":    self._ck_ff_alpha.value(),
            "other": self._ck_other_alpha.value(),
        } if self._ck_selective_chk.isChecked() else None

        self._ck_run.setEnabled(False)
        self._ck_bar.setValue(0)
        self._ck_status.setText("Starting…")
        # Reset health indicators to neutral
        for lbl in (self._ck_h_key, self._ck_h_mag, self._ck_h_vae,
                    self._ck_h_alpha, self._ck_h_confirmed):
            lbl.setStyleSheet(
                f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f" background:transparent; border:none;")
            lbl.setToolTip("")

        worker = _Worker()
        worker.progress.connect(
            lambda i, t, k: self._on_progress(self._ck_bar, self._ck_status, i, t, k))
        worker.finished.connect(
            lambda: self._on_finished(self._ck_run, self._ck_status, self._ck_bar,
                                       f"Saved: {out}"))
        worker.error.connect(
            lambda e: self._on_error(self._ck_run, self._ck_status, e))
        worker.health_done.connect(self._on_ck_health_done)
        self._worker = worker

        def _run():
            try:
                stats = merge_checkpoints(
                    a, b, c, method, alpha, beta, density, out, prec, dev,
                    vae_path=vae, group_alphas=group_alphas,
                    progress_fn=lambda i, t, k: worker.progress.emit(i, t, k),
                )
                # Auto health check using Model A as reference
                try:
                    health = check_checkpoint_health(out, a, merge_stats=stats)
                    worker.health_done.emit(health)
                except Exception:
                    pass
                worker.finished.emit()
            except Exception as exc:
                worker.error.emit(str(exc))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _run_lora_merge(self):
        if self._is_busy():
            QMessageBox.warning(self, "Busy", "A merge operation is already running.")
            return

        # Collect all active slots (A required, B required, C/D optional)
        slot_defs = [
            (self._lo_a, self._lo_wa, "LoRA A", True),
            (self._lo_b, self._lo_wb, "LoRA B", True),
            (self._lo_c, self._lo_wc, "LoRA C", False),
            (self._lo_d, self._lo_wd, "LoRA D", False),
        ]
        paths:   list[str]   = []
        weights: list[float] = []
        for field, spin, label, required in slot_defs:
            p = field.text().strip()
            if p:
                if not os.path.isfile(p):
                    QMessageBox.warning(self, "Missing", f"{label} not found:\n{p}")
                    return
                paths.append(p)
                weights.append(spin.value())
            elif required:
                QMessageBox.warning(self, "Missing", f"{label} is required.")
                return

        out = self._lo_out.text().strip()
        if not out:
            QMessageBox.warning(self, "Missing", "Output path is required.")
            return

        self._save_lora_cfg()

        from .merge_engine import merge_loras
        dev = self._current_device()

        self._lo_run.setEnabled(False)
        self._lo_bar.setValue(0)
        self._lo_status.setText("Starting…")

        worker = _Worker()
        worker.progress.connect(
            lambda i, t, k: self._on_progress(self._lo_bar, self._lo_status, i, t, k))
        worker.finished.connect(
            lambda: self._on_finished(self._lo_run, self._lo_status, self._lo_bar,
                                       f"Saved: {out}"))
        worker.error.connect(
            lambda e: self._on_error(self._lo_run, self._lo_status, e))
        self._worker = worker

        def _run():
            try:
                merge_loras(
                    paths, weights, out, output_rank=0, device=dev,
                    progress_fn=lambda i, t, k: worker.progress.emit(i, t, k),
                )
                worker.finished.emit()
            except Exception as exc:
                worker.error.emit(str(exc))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _run_bake(self):
        if self._is_busy():
            QMessageBox.warning(self, "Busy", "A merge operation is already running.")
            return

        ckpt = self._bk_ckpt.text().strip()
        out  = self._bk_out.text().strip()

        if not ckpt or not os.path.isfile(ckpt):
            QMessageBox.warning(self, "Missing", "Checkpoint not found.")
            return
        if not out:
            QMessageBox.warning(self, "Missing", "Output path is required.")
            return

        # Collect active LoRA slots
        entries: list[tuple[str, float]] = []
        for path_edit, _combo, ratio_spin in self._bk_loras:
            p = path_edit.text().strip()
            if p:
                if not os.path.isfile(p):
                    QMessageBox.warning(self, "Missing", f"LoRA not found:\n{p}")
                    return
                entries.append((p, ratio_spin.value()))

        if not entries:
            QMessageBox.warning(self, "Missing", "Add at least one LoRA file.")
            return

        self._save_bake_cfg()

        from .merge_engine import bake_loras
        prec = self._bk_prec.currentText()
        dev  = self._current_device()

        self._bk_run.setEnabled(False)
        self._bk_bar.setValue(0)
        self._bk_status.setText("Starting…")
        for lbl in (self._bk_h_key, self._bk_h_mag, self._bk_h_vae,
                    self._bk_h_alpha, self._bk_h_confirmed):
            lbl.setStyleSheet(
                f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f" background:transparent; border:none;")
            lbl.setToolTip("")

        from .merge_engine import check_checkpoint_health

        worker = _Worker()
        worker.progress.connect(
            lambda i, t, k: self._on_progress(self._bk_bar, self._bk_status, i, t, k))
        worker.finished.connect(
            lambda: self._on_finished(self._bk_run, self._bk_status, self._bk_bar,
                                       f"Saved: {out}"))
        worker.error.connect(
            lambda e: self._on_error(self._bk_run, self._bk_status, e))
        worker.health_done.connect(self._on_bk_health_done)
        self._worker = worker

        def _run():
            try:
                stats = bake_loras(
                    ckpt, entries, out, prec, dev,
                    progress_fn=lambda i, t, k: worker.progress.emit(i, t, k),
                )
                try:
                    health = check_checkpoint_health(out, ckpt, merge_stats=stats)
                    worker.health_done.emit(health)
                except Exception:
                    pass
                worker.finished.emit()
            except Exception as exc:
                worker.error.emit(str(exc))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    # ── Progress / finish / error callbacks ───────────────────────────────────

    def _on_progress(self, bar: QProgressBar, status: QLabel,
                     i: int, total: int, key: str):
        pct = int(100 * i / total) if total > 0 else 0
        bar.setValue(pct)
        status.setText(f"{i}/{total}  {key[:60]}")

    def _on_finished(self, btn: QPushButton, status: QLabel,
                     bar: QProgressBar, msg: str):
        bar.setValue(100)
        status.setText(msg)
        status.setStyleSheet(
            f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")
        btn.setEnabled(True)

    def _on_error(self, btn: QPushButton, status: QLabel, error: str):
        status.setText(f"Error: {error[:120]}")
        status.setStyleSheet(
            f"color:{RED}; font-family:{FONT}; font-size:{FONT_SM}px;"
            f" background:transparent; border:none;")
        btn.setEnabled(True)

    # ── Config save helpers ────────────────────────────────────────────────────

    def _save_ckpt_cfg(self):
        self._cfg.update({
            "ckpt_model_a":     self._ck_a.text().strip(),
            "ckpt_model_b":     self._ck_b.text().strip(),
            "ckpt_model_c":     self._ck_c.text().strip(),
            "ckpt_vae":         self._ck_vae.text().strip(),
            "ckpt_method":      self._ck_method.currentText(),
            "ckpt_selective":   self._ck_selective_chk.isChecked(),
            "ckpt_te_alpha":    self._ck_te_alpha.value(),
            "ckpt_ca_alpha":    self._ck_ca_alpha.value(),
            "ckpt_sa_alpha":    self._ck_sa_alpha.value(),
            "ckpt_ff_alpha":    self._ck_ff_alpha.value(),
            "ckpt_other_alpha": self._ck_other_alpha.value(),
            "ckpt_alpha":       self._ck_alpha.value(),
            "ckpt_beta":        self._ck_beta.value(),
            "ckpt_density":     self._ck_density.value(),
            "ckpt_precision":   self._ck_prec.currentText(),
            "ckpt_output":      self._ck_out.text().strip(),
        })
        save_json(_MERGE_CFG, self._cfg)

    def _save_lora_cfg(self):
        self._cfg.update({
            "lora_a":        self._lo_a.text().strip(),
            "lora_b":        self._lo_b.text().strip(),
            "lora_c":        self._lo_c.text().strip(),
            "lora_d":        self._lo_d.text().strip(),
            "lora_type_a":   self._lo_ta.currentText(),
            "lora_type_b":   self._lo_tb.currentText(),
            "lora_type_c":   self._lo_tc.currentText(),
            "lora_type_d":   self._lo_td.currentText(),
            "lora_weight_a": self._lo_wa.value(),
            "lora_weight_b": self._lo_wb.value(),
            "lora_weight_c": self._lo_wc.value(),
            "lora_weight_d": self._lo_wd.value(),
            "lora_output":   self._lo_out.text().strip(),
        })
        save_json(_MERGE_CFG, self._cfg)

    def _save_bake_cfg(self):
        data: dict = {
            "bake_ckpt":      self._bk_ckpt.text().strip(),
            "bake_precision": self._bk_prec.currentText(),
            "bake_output":    self._bk_out.text().strip(),
        }
        for i, (path_edit, type_combo, ratio_spin) in enumerate(self._bk_loras, start=1):
            data[f"bake_lora_{i}"]  = path_edit.text().strip()
            data[f"bake_type_{i}"]  = type_combo.currentText()
            data[f"bake_ratio_{i}"] = ratio_spin.value()
        self._cfg.update(data)
        save_json(_MERGE_CFG, self._cfg)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def cleanup(self):
        """Called before the widget is removed from the stack."""
        if self._worker is not None:
            try:
                self._worker.progress.disconnect()
                self._worker.finished.disconnect()
                self._worker.error.disconnect()
            except RuntimeError:
                pass
