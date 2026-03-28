"""
calculator/calculator_page.py — LoRA Training Calculator for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.0)

Full v1 feature port: 6 tabs — AI-Toolkit, Kohya, CivitAI, Logs, TOS, Help
TOS-based calculation, YAML export, training log, per-type TOS presets.
"""
from __future__ import annotations

import os
import re

from PySide6.QtCore    import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox,
    QScrollArea, QSizePolicy, QTabWidget, QSplitter,
    QFileDialog, QApplication,
)

from shared.theme  import (
    BG, PAN, CAR, ACC, GRN, RED, MUT, PRI, SEC, AMB,
    FONT, FONT_SM, FONT_MD, FONT_LG, FONT_XL,
    VERSION, SIGNATURE,
)
from shared.config  import CALC_CFG, load_json, save_json
from shared.calc_engine import (
    LORA_TYPES, TOS_RANGES, RANK_CAP_OPTIONS, SHARED_HELP, HELP_CONTENT,
    get_images, calc_at, calc_ko, calc_cv,
    get_averages, init_log, add_log_entry, delete_log_entry, load_log,
    count_dataset_files, build_at_yaml, get_bucket_resolutions,
)

DEFAULTS: dict = {
    "last_dataset_dir":       "",
    "last_export_dir":        "",
    "last_model_dir":         "",
    "last_kohya_prompts_dir": "",
    "aitoolkit_path":         "",
    "base_model_path":        "",
    "default_caption":        "Created by Zero",
    "version":                "1.0",
    "sample_prompts":         ["", "", ""],
    "kohya_sample_prompts":   ["", "", ""],
    "in_network":             False,
    "local_drive":            "X",
    "remote_drive":           "I",
    "bucket_512":             False,
    "bucket_256":             False,
    "cache_latents":          True,
    "lora_type":              "character",
    "dataset_files":          "0",
    "tagged":                 False,
    "rank_cap":               "auto",
    "batch_size":             "2",
    "at_factor":              "1",
    "ko_factor":              "1",
    "cv_factor":              "1",
    "tos_presets":  {lt: "Recommended" for lt in LORA_TYPES},
    "tos_overrides": {lt: "" for lt in LORA_TYPES},
}


# ── Widget helpers ─────────────────────────────────────────────────────────────
def _lbl(text: str, color: str = SEC, size: int = FONT_SM, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{color}; font-family:{FONT}; font-size:{size}px;"
        f"font-weight:{'bold' if bold else 'normal'}; background:transparent;")
    return lbl


def _edit_style() -> str:
    return (
        f"QLineEdit {{ background:{CAR}; color:{PRI}; border:2px solid {MUT};"
        f"border-radius:4px; padding:2px 6px; font-family:{FONT}; font-size:{FONT_MD}px; }}"
        f"QLineEdit:focus {{ border-color:{ACC}; }}"
    )


def _combo_style() -> str:
    return (
        f"QComboBox {{ background:{CAR}; color:{PRI}; border:2px solid {MUT};"
        f"border-radius:4px; padding:2px 6px; font-family:{FONT}; font-size:{FONT_MD}px;"
        f"min-height:26px; }}"
        f"QComboBox::drop-down {{ border:none; width:20px; }}"
        f"QComboBox QAbstractItemView {{ background:{CAR}; color:{PRI};"
        f"border:1px solid {ACC}; selection-background-color:{ACC}; }}"
    )


def _check_style() -> str:
    return (
        f"QCheckBox {{ color:{SEC}; font-family:{FONT}; font-size:{FONT_MD}px; spacing:4px; }}"
        f"QCheckBox::indicator {{ width:18px; height:18px; border:2px solid {MUT};"
        f"border-radius:3px; background:{CAR}; }}"
        f"QCheckBox::indicator:checked {{ background:{ACC}; border-color:{ACC}; }}"
    )


def _btn(label: str, color: str = ACC, width: int | None = None) -> QPushButton:
    b = QPushButton(label)
    b.setStyleSheet(f"""
        QPushButton {{
            background:{color}; color:{PRI}; border:none; border-radius:4px;
            padding:4px 12px; font-family:{FONT}; font-size:{FONT_MD}px; font-weight:bold;
        }}
        QPushButton:hover {{ background:rgba(255,255,255,25); }}
        QPushButton:disabled {{ background:{MUT}; color:{SEC}; }}
    """)
    if width:
        b.setFixedWidth(width)
    return b


def _outline_btn(label: str, color: str = ACC, width: int | None = None) -> QPushButton:
    b = QPushButton(label)
    b.setStyleSheet(f"""
        QPushButton {{
            background:{CAR}; color:{color}; border:2px solid {color};
            border-radius:4px; padding:4px 10px;
            font-family:{FONT}; font-size:{FONT_MD}px; font-weight:bold;
        }}
        QPushButton:hover {{ background:{color}; color:{PRI}; }}
    """)
    if width:
        b.setFixedWidth(width)
    return b


def _card_style(alt: bool = False) -> str:
    bg = CAR if not alt else PAN
    return f"QFrame {{ background:{bg}; border-radius:6px; }}"


# ── CalculatorPage ─────────────────────────────────────────────────────────────
class CalculatorPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg = load_json(CALC_CFG, DEFAULTS)
        if not isinstance(self._cfg.get("tos_presets"), dict):
            self._cfg["tos_presets"] = DEFAULTS["tos_presets"].copy()
        if not isinstance(self._cfg.get("tos_overrides"), dict):
            self._cfg["tos_overrides"] = DEFAULTS["tos_overrides"].copy()

        self._dataset_path  = ""
        self._at_data: dict = {}

        init_log()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        root.addWidget(self._build_inputs())

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane  {{ background:{PAN}; border:none; }}
            QTabBar::tab      {{ background:{CAR}; color:{SEC}; padding:6px 16px;
                                 margin-right:2px; font-family:{FONT}; font-size:{FONT_MD}px; }}
            QTabBar::tab:selected       {{ background:{ACC}; color:{PRI}; }}
            QTabBar::tab:hover:!selected {{ background:{MUT}; color:{PRI}; }}
        """)

        self._build_at_tab()
        self._build_ko_tab()
        self._build_cv_tab()
        self._build_log_tab()
        self._build_tos_tab()
        self._build_help_tab()

        root.addWidget(self._tabs, stretch=1)
        QTimer.singleShot(100, self._calc)

    # ── Header ─────────────────────────────────────────────────────────────────
    def _build_header(self) -> QFrame:
        hdr = QFrame()
        hdr.setStyleSheet(f"background:{CAR}; border-bottom:1px solid {MUT};")
        row = QHBoxLayout(hdr)
        row.setContentsMargins(12, 8, 12, 8)
        title = QLabel("LoRA Training Calculator")
        title.setStyleSheet(
            f"color:{PRI}; font-family:{FONT}; font-size:{FONT_XL}px; font-weight:bold;")
        row.addWidget(title)
        row.addStretch(1)
        sub = QLabel("SDXL · 1280p")
        sub.setStyleSheet(f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;")
        row.addWidget(sub)
        row.addSpacing(16)
        sig = QLabel(SIGNATURE)
        sig.setStyleSheet(f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;")
        row.addWidget(sig)
        return hdr

    # ── Shared inputs bar ──────────────────────────────────────────────────────
    def _build_inputs(self) -> QFrame:
        bar = QFrame()
        bar.setStyleSheet(f"QFrame {{ background:{PAN}; }}")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)

        def _col(label_text: str, widget: QWidget) -> QVBoxLayout:
            col = QVBoxLayout()
            col.setSpacing(2)
            lbl = QLabel(label_text)
            lbl.setStyleSheet(
                f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
            col.addWidget(lbl)
            col.addWidget(widget)
            return col

        # LoRA Type
        self._lt_combo = QComboBox()
        self._lt_combo.addItems(LORA_TYPES)
        self._lt_combo.setCurrentText(self._cfg.get("lora_type", "character"))
        self._lt_combo.setStyleSheet(_combo_style())
        self._lt_combo.setFixedWidth(120)
        self._lt_combo.currentTextChanged.connect(self._calc)
        layout.addLayout(_col("LoRA Type", self._lt_combo))

        # Dataset: Browse + file count
        ds_w = QWidget()
        ds_w.setStyleSheet("background:transparent;")
        ds_row = QHBoxLayout(ds_w)
        ds_row.setContentsMargins(0, 0, 0, 0)
        ds_row.setSpacing(4)
        browse = _outline_btn("📁 Browse", ACC)
        browse.setFixedHeight(28)
        browse.clicked.connect(self._browse_dataset)
        ds_row.addWidget(browse)
        self._files_edit = QLineEdit(str(self._cfg.get("dataset_files", "0")))
        self._files_edit.setFixedWidth(70)
        self._files_edit.setStyleSheet(_edit_style())
        self._files_edit.textChanged.connect(self._calc)
        ds_row.addWidget(self._files_edit)
        layout.addLayout(_col("Dataset", ds_w))

        # Tagged
        self._tagged_cb = QCheckBox()
        self._tagged_cb.setChecked(bool(self._cfg.get("tagged", False)))
        self._tagged_cb.setStyleSheet(_check_style())
        self._tagged_cb.stateChanged.connect(self._calc)
        layout.addLayout(_col("Tagged", self._tagged_cb))

        # Total Images (ACC border)
        self._total_img_lbl = QLabel("0")
        self._total_img_lbl.setStyleSheet(
            f"color:{PRI}; font-family:{FONT}; font-size:13px; font-weight:bold;"
            f"background:{CAR}; border:2px solid {ACC}; border-radius:4px;"
            f"padding:2px 8px; min-width:60px;")
        self._total_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(_col("Total Images", self._total_img_lbl))

        # Rank Cap
        self._rc_combo = QComboBox()
        self._rc_combo.addItems(RANK_CAP_OPTIONS)
        self._rc_combo.setCurrentText(str(self._cfg.get("rank_cap", "auto")))
        self._rc_combo.setStyleSheet(_combo_style())
        self._rc_combo.setFixedWidth(100)
        self._rc_combo.currentTextChanged.connect(self._calc)
        layout.addLayout(_col("Rank Cap", self._rc_combo))

        # Batch Size
        self._batch_edit = QLineEdit(str(self._cfg.get("batch_size", "2")))
        self._batch_edit.setFixedWidth(60)
        self._batch_edit.setStyleSheet(_edit_style())
        self._batch_edit.textChanged.connect(self._calc)
        layout.addLayout(_col("Batch Size", self._batch_edit))

        layout.addStretch(1)
        return bar

    # ── Helper: results grid ───────────────────────────────────────────────────
    def _make_results(self, fields: list[tuple[str, str]]) -> tuple[QFrame, dict]:
        frame = QFrame()
        frame.setStyleSheet(_card_style())
        grid = QGridLayout(frame)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(5, 1)

        lbls: dict[str, QLabel] = {}
        for i, (disp, key) in enumerate(fields):
            row, pair = divmod(i, 2)
            bc = pair * 3

            name = QLabel(disp + ":")
            name.setStyleSheet(
                f"background:transparent; color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px;")
            grid.addWidget(name, row, bc)

            val = QLabel("—")
            val.setStyleSheet(
                f"background:transparent; color:{PRI}; font-family:{FONT};"
                f"font-size:{FONT_MD}px; font-weight:bold;")
            val.setMinimumWidth(110)
            lbls[key] = val
            grid.addWidget(val, row, bc + 1)

            cp = QPushButton("⎘")
            cp.setStyleSheet(
                f"QPushButton {{ background:transparent; color:{MUT}; border:none; font-size:10px; }}"
                f"QPushButton:hover {{ color:{ACC}; }}")
            cp.setFixedSize(18, 18)
            cp.clicked.connect(lambda _, v=val: (
                QApplication.clipboard().setText(v.text()) if v.text() != "—" else None))
            grid.addWidget(cp, row, bc + 2)

        return frame, lbls

    # ── Helper: factor row (Manual Factor | Suggested Factor | ETA) ────────────
    def _make_factor(self, label: str = "Manual Factor (R2)") -> tuple[QFrame, QLineEdit, QLabel, QLabel]:
        frame = QFrame()
        frame.setStyleSheet(_card_style())
        row = QHBoxLayout(frame)
        row.setContentsMargins(12, 6, 12, 8)
        row.setSpacing(0)

        def _section(title: str, w: QWidget):
            col = QVBoxLayout()
            col.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet(
                f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
            col.addWidget(t)
            col.addWidget(w)
            return col

        fe = QLineEdit("1")
        fe.setFixedWidth(80)
        fe.setStyleSheet(_edit_style())
        fe.textChanged.connect(self._calc)
        row.addLayout(_section(label, fe))
        row.addSpacing(24)

        sl = QLabel("—")
        sl.setStyleSheet(
            f"color:{AMB}; font-family:{FONT}; font-size:13px; font-weight:bold; background:transparent;")
        row.addLayout(_section("Suggested Factor", sl))
        row.addSpacing(24)

        el = QLabel("—")
        el.setStyleSheet(
            f"color:{ACC}; font-family:{FONT}; font-size:13px; font-weight:bold; background:transparent;")
        row.addLayout(_section("ETA", el))
        row.addStretch(1)

        return frame, fe, sl, el

    # ── Helper: status row (ETS Signal | TOS/img | Warning) ────────────────────
    def _make_status(self) -> tuple[QFrame, QLabel, QLabel, QLabel]:
        frame = QFrame()
        frame.setStyleSheet(_card_style())
        row = QHBoxLayout(frame)
        row.setContentsMargins(12, 6, 12, 8)
        row.setSpacing(0)

        def _section(title: str, w: QWidget):
            col = QVBoxLayout()
            col.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet(
                f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
            col.addWidget(t)
            col.addWidget(w)
            return col

        ets = QLabel("—")
        ets.setStyleSheet(
            f"color:{GRN}; font-family:{FONT}; font-size:14px; font-weight:bold; background:transparent;")
        row.addLayout(_section("ETS Signal", ets))
        row.addSpacing(24)

        tos = QLabel("—")
        tos.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_MD}px; background:transparent;")
        row.addLayout(_section("TOS / img", tos))
        row.addSpacing(24)

        warn = QLabel("")
        warn.setStyleSheet(
            f"color:{RED}; font-family:{FONT}; font-size:{FONT_MD}px; background:transparent;")
        row.addLayout(_section("Warning", warn))
        row.addStretch(1)

        return frame, ets, tos, warn

    # ── AI-Toolkit tab ─────────────────────────────────────────────────────────
    def _build_at_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        f, self._at_fe, self._at_sl, self._at_el = self._make_factor("Manual Factor (R2)")
        layout.addWidget(f)

        rf, self._at_v = self._make_results([
            ("Steps",           "steps"),  ("Learning Rate",   "lr"),
            ("Weight Decay",    "weight_decay"), ("Caption Dropout", "caption_dropout"),
            ("Grad Accum",      "grad_accum"),   ("Eff Batch",       "eff_batch"),
            ("Linear Rank",     "linear_rank"),  ("Conv Rank",       "conv_rank"),
            ("Max Saves",       "max_saves"),    ("Save Every",      "save_every"),
        ])
        layout.addWidget(rf)

        sf, self._at_ets, self._at_tos, self._at_warn = self._make_status()
        layout.addWidget(sf)

        layout.addWidget(self._build_export_panel(), stretch=1)
        self._tabs.addTab(w, "AI-Toolkit")

    def _build_export_panel(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(_card_style())
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        hdr = QFrame()
        hdr.setStyleSheet(f"QFrame {{ background:{PAN}; border-radius:4px; }}")
        hdr.setFixedHeight(38)
        hdr_row = QHBoxLayout(hdr)
        hdr_row.setContentsMargins(10, 4, 8, 4)
        es_lbl = QLabel("Export Settings")
        es_lbl.setStyleSheet(
            f"color:{ACC}; font-family:{FONT}; font-size:13px; font-weight:bold;")
        hdr_row.addWidget(es_lbl)
        hdr_row.addStretch(1)
        self._exp_status = QLabel("")
        self._exp_status.setStyleSheet(
            f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px;")
        hdr_row.addWidget(self._exp_status)
        export_btn = _btn("Export", ACC, width=90)
        export_btn.clicked.connect(self.do_export)
        hdr_row.addWidget(export_btn)
        create_btn = _outline_btn("Create Job", ACC, width=110)
        create_btn.clicked.connect(self.do_create_job)
        hdr_row.addWidget(create_btn)
        layout.addWidget(hdr)

        # Row 1 — LoRA Name | Default Caption | Version | Trigger Word
        r1 = QWidget()
        r1.setStyleSheet("background:transparent;")
        r1_row = QHBoxLayout(r1)
        r1_row.setContentsMargins(2, 0, 2, 0)
        r1_row.setSpacing(8)

        def _field_col(label_text: str, widget: QWidget, fixed_w: int | None = None):
            col = QVBoxLayout()
            col.setSpacing(2)
            lbl = QLabel(label_text)
            lbl.setStyleSheet(
                f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
            col.addWidget(lbl)
            if fixed_w:
                widget.setFixedWidth(fixed_w)
            col.addWidget(widget)
            return col

        self._lora_name_edit = QLineEdit()
        self._lora_name_edit.setPlaceholderText("my_lora")
        self._lora_name_edit.setStyleSheet(_edit_style())
        r1_row.addLayout(_field_col("LoRA Name", self._lora_name_edit), stretch=2)

        self._caption_edit = QLineEdit(self._cfg.get("default_caption", "Created by Zero"))
        self._caption_edit.setStyleSheet(_edit_style())
        r1_row.addLayout(_field_col("Default Caption", self._caption_edit), stretch=2)

        self._version_edit = QLineEdit(str(self._cfg.get("version", "1.0")))
        self._version_edit.setStyleSheet(_edit_style())
        r1_row.addLayout(_field_col("Version", self._version_edit, fixed_w=70))

        self._trigger_edit = QLineEdit()
        self._trigger_edit.setPlaceholderText("trigger word")
        self._trigger_edit.setStyleSheet(_edit_style())
        r1_row.addLayout(_field_col("Trigger Word", self._trigger_edit), stretch=2)
        layout.addWidget(r1)

        # Bucket resolutions row
        rb = QWidget()
        rb.setStyleSheet("background:transparent;")
        rb_row = QHBoxLayout(rb)
        rb_row.setContentsMargins(2, 0, 2, 0)
        rb_row.setSpacing(4)
        bres = QLabel("Bucket Resolutions")
        bres.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        rb_row.addWidget(bres)
        rb_row.addSpacing(8)
        for res in ["768", "1024", "1280", "1536"]:
            rl = QLabel(res)
            rl.setStyleSheet(
                f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px;"
                f"font-weight:bold; background:transparent;")
            rl.setFixedWidth(36)
            rl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rb_row.addWidget(rl)
        self._bucket_512_cb = QCheckBox("512")
        self._bucket_512_cb.setChecked(bool(self._cfg.get("bucket_512", False)))
        self._bucket_512_cb.setStyleSheet(_check_style())
        rb_row.addWidget(self._bucket_512_cb)
        self._bucket_256_cb = QCheckBox("256")
        self._bucket_256_cb.setChecked(bool(self._cfg.get("bucket_256", False)))
        self._bucket_256_cb.setStyleSheet(_check_style())
        rb_row.addWidget(self._bucket_256_cb)
        rb_row.addSpacing(12)
        self._cache_latents_cb = QCheckBox("Cache Latents")
        self._cache_latents_cb.setChecked(bool(self._cfg.get("cache_latents", True)))
        self._cache_latents_cb.setStyleSheet(_check_style())
        rb_row.addWidget(self._cache_latents_cb)
        rb_row.addStretch(1)
        layout.addWidget(rb)

        # Row 2 — AT Folder | Base Model | In Network?
        r2 = QWidget()
        r2.setStyleSheet("background:transparent;")
        r2_row = QHBoxLayout(r2)
        r2_row.setContentsMargins(2, 0, 2, 0)
        r2_row.setSpacing(8)

        def _path_col(label_text: str, path_edit: QLineEdit, browse_cb):
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(_lbl(label_text))
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addWidget(path_edit, stretch=1)
            pb = QPushButton("📁")
            pb.setFixedSize(28, 26)
            pb.setStyleSheet(
                f"QPushButton {{ background:{CAR}; color:{PRI}; border:2px solid {ACC};"
                f"border-radius:4px; font-size:13px; }}"
                f"QPushButton:hover {{ background:{ACC}; }}")
            pb.clicked.connect(browse_cb)
            row.addWidget(pb)
            col.addLayout(row)
            return col

        self._at_path_edit = QLineEdit(self._cfg.get("aitoolkit_path", ""))
        self._at_path_edit.setStyleSheet(_edit_style())
        r2_row.addLayout(_path_col("AI-Toolkit Folder", self._at_path_edit, self._browse_aitoolkit), stretch=2)

        self._model_path_edit = QLineEdit(self._cfg.get("base_model_path", ""))
        self._model_path_edit.setStyleSheet(_edit_style())
        r2_row.addLayout(_path_col("Base Model", self._model_path_edit, self._browse_model), stretch=2)

        # In Network?
        net_col = QVBoxLayout()
        net_col.setSpacing(2)
        net_col.addWidget(_lbl("In Network?"))
        self._network_cb = QCheckBox()
        self._network_cb.setChecked(bool(self._cfg.get("in_network", False)))
        self._network_cb.setStyleSheet(_check_style())
        self._network_cb.stateChanged.connect(self._toggle_network)
        net_col.addWidget(self._network_cb)
        r2_row.addLayout(net_col)

        # Drive letters (show/hide)
        self._net_frame = QWidget()
        self._net_frame.setStyleSheet("background:transparent;")
        nf_row = QHBoxLayout(self._net_frame)
        nf_row.setContentsMargins(0, 0, 0, 0)
        nf_row.setSpacing(6)

        for label_text, cfg_key, attr in [
            ("Local",  "local_drive",  "_local_drive_edit"),
            ("Remote", "remote_drive", "_remote_drive_edit"),
        ]:
            d_col = QVBoxLayout()
            d_col.setSpacing(2)
            d_col.addWidget(_lbl(label_text))
            de = QLineEdit(str(self._cfg.get(cfg_key, "X")))
            de.setFixedWidth(36)
            de.setAlignment(Qt.AlignmentFlag.AlignCenter)
            de.setStyleSheet(_edit_style())
            setattr(self, attr, de)
            d_col.addWidget(de)
            nf_row.addLayout(d_col)

        r2_row.addWidget(self._net_frame)
        self._net_frame.setVisible(bool(self._cfg.get("in_network", False)))
        layout.addWidget(r2)

        # Sample prompts
        ph = QWidget()
        ph.setStyleSheet("background:transparent;")
        ph_row = QHBoxLayout(ph)
        ph_row.setContentsMargins(2, 2, 2, 0)
        ph_row.setSpacing(10)
        ph_row.addWidget(_lbl("Sample Prompts"))
        add_btn = _outline_btn("Add Prompt", ACC, width=110)
        add_btn.clicked.connect(self._add_at_prompt)
        ph_row.addWidget(add_btn)
        ph_row.addStretch(1)
        layout.addWidget(ph)

        self._at_prompts_widget = QWidget()
        self._at_prompts_widget.setStyleSheet("background:transparent;")
        self._at_prompts_layout = QVBoxLayout(self._at_prompts_widget)
        self._at_prompts_layout.setContentsMargins(0, 0, 0, 0)
        self._at_prompts_layout.setSpacing(4)
        self._at_prompts_layout.addStretch(1)

        prompts_scroll = QScrollArea()
        prompts_scroll.setWidgetResizable(True)
        prompts_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        prompts_scroll.setStyleSheet(f"QScrollArea {{ border:none; background:transparent; }}")
        prompts_scroll.setFixedHeight(90)
        prompts_scroll.setWidget(self._at_prompts_widget)
        layout.addWidget(prompts_scroll)

        self._at_prompt_edits: list[QLineEdit] = []
        for p in self._cfg.get("sample_prompts", ["", "", ""])[:3]:
            self._add_at_prompt_row(str(p))

        return frame

    def _add_at_prompt_row(self, text: str = ""):
        edit = QLineEdit(text)
        edit.setPlaceholderText("Enter sample prompt...")
        edit.setStyleSheet(_edit_style())
        self._at_prompt_edits.append(edit)
        # Insert before the stretch
        idx = max(0, self._at_prompts_layout.count() - 1)
        self._at_prompts_layout.insertWidget(idx, edit)

    def _add_at_prompt(self):
        self._add_at_prompt_row()

    # ── Kohya tab ──────────────────────────────────────────────────────────────
    def _build_ko_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        f, self._ko_fe, self._ko_sl, self._ko_el = self._make_factor()
        layout.addWidget(f)

        rf, self._ko_v = self._make_results([
            ("Repeats",       "repeats"),    ("Epochs",       "epochs"),
            ("Steps",         "steps"),      ("UNet LR",      "lr"),
            ("Text Enc LR",   "te_lr"),      ("LR Scheduler", "scheduler"),
            ("Optimizer",     "optimizer"),  ("Weight Decay", "weight_decay"),
            ("Grad Accum",    "grad_accum"), ("Eff Batch",    "eff_batch"),
            ("Linear Rank",   "linear_rank"),("Conv Rank",    "conv_rank"),
            ("Network Alpha", "net_alpha"),  ("Noise Offset", "noise_offset"),
            ("Min SNR Gamma", "min_snr"),    ("Save Every (ep)", "save_every"),
        ])
        layout.addWidget(rf)

        sf, self._ko_ets, self._ko_tos, self._ko_warn = self._make_status()
        layout.addWidget(sf)

        layout.addWidget(self._build_ko_prompts_panel(), stretch=1)
        self._tabs.addTab(w, "Kohya")

    def _build_ko_prompts_panel(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(_card_style())
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        hdr = QFrame()
        hdr.setStyleSheet(f"QFrame {{ background:{PAN}; border-radius:4px; }}")
        hdr.setFixedHeight(38)
        hdr_row = QHBoxLayout(hdr)
        hdr_row.setContentsMargins(10, 4, 8, 4)
        hdr_row.addWidget(_lbl("Sample Prompts", ACC, FONT_MD, bold=True))
        hdr_row.addStretch(1)
        self._ko_exp_status = QLabel("")
        self._ko_exp_status.setStyleSheet(
            f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px;")
        hdr_row.addWidget(self._ko_exp_status)
        exp_btn = _btn("Export", ACC, width=90)
        exp_btn.clicked.connect(self.do_ko_export)
        hdr_row.addWidget(exp_btn)
        layout.addWidget(hdr)

        # Note
        note = QFrame()
        note.setStyleSheet(f"QFrame {{ background:{PAN}; border-radius:4px; }}")
        note_lbl = QLabel(
            "One prompt per line. Point Kohya's Sample prompts field to the exported .txt file. "
            "Each prompt generates one sample image per cycle. No hard limit — keep to 3–5.")
        note_lbl.setStyleSheet(
            f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        note_lbl.setWordWrap(True)
        note_lbl.setContentsMargins(10, 6, 10, 6)
        n_layout = QVBoxLayout(note)
        n_layout.setContentsMargins(0, 0, 0, 0)
        n_layout.addWidget(note_lbl)
        layout.addWidget(note)

        # Add prompt
        ph = QWidget()
        ph.setStyleSheet("background:transparent;")
        ph_row = QHBoxLayout(ph)
        ph_row.setContentsMargins(2, 0, 2, 0)
        ph_row.setSpacing(10)
        ph_row.addWidget(_lbl("Prompts"))
        add_btn = _outline_btn("Add Prompt", ACC, width=110)
        add_btn.clicked.connect(self._add_ko_prompt)
        ph_row.addWidget(add_btn)
        ph_row.addStretch(1)
        layout.addWidget(ph)

        self._ko_prompts_widget = QWidget()
        self._ko_prompts_widget.setStyleSheet("background:transparent;")
        self._ko_prompts_layout = QVBoxLayout(self._ko_prompts_widget)
        self._ko_prompts_layout.setContentsMargins(0, 0, 0, 0)
        self._ko_prompts_layout.setSpacing(4)
        self._ko_prompts_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:transparent; }}")
        scroll.setWidget(self._ko_prompts_widget)
        layout.addWidget(scroll, stretch=1)

        self._ko_prompt_edits: list[QLineEdit] = []
        for p in self._cfg.get("kohya_sample_prompts", ["", "", ""])[:3]:
            self._add_ko_prompt_row(str(p))

        return frame

    def _add_ko_prompt_row(self, text: str = ""):
        edit = QLineEdit(text)
        edit.setPlaceholderText("Enter sample prompt...")
        edit.setStyleSheet(_edit_style())
        self._ko_prompt_edits.append(edit)
        idx = max(0, self._ko_prompts_layout.count() - 1)
        self._ko_prompts_layout.insertWidget(idx, edit)

    def _add_ko_prompt(self):
        self._add_ko_prompt_row()

    # ── CivitAI tab ────────────────────────────────────────────────────────────
    def _build_cv_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # CivitAI factor row (Manual Factor | Suggested Factor | Buzz Cost | CivitAI Type)
        cv_top = QFrame()
        cv_top.setStyleSheet(_card_style())
        cv_top_row = QHBoxLayout(cv_top)
        cv_top_row.setContentsMargins(12, 6, 12, 8)
        cv_top_row.setSpacing(0)

        def _cv_section(title: str, w_inner: QWidget):
            col = QVBoxLayout()
            col.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet(
                f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
            col.addWidget(t)
            col.addWidget(w_inner)
            return col

        self._cv_fe = QLineEdit("1")
        self._cv_fe.setFixedWidth(80)
        self._cv_fe.setStyleSheet(_edit_style())
        self._cv_fe.textChanged.connect(self._calc)
        cv_top_row.addLayout(_cv_section("Manual Factor", self._cv_fe))
        cv_top_row.addSpacing(24)

        self._cv_sl = QLabel("—")
        self._cv_sl.setStyleSheet(
            f"color:{AMB}; font-family:{FONT}; font-size:13px; font-weight:bold; background:transparent;")
        cv_top_row.addLayout(_cv_section("Suggested Factor", self._cv_sl))
        cv_top_row.addSpacing(24)

        self._cv_buzz = QLabel("—")
        self._cv_buzz.setStyleSheet(
            f"color:#FFD700; font-family:{FONT}; font-size:13px; font-weight:bold; background:transparent;")
        cv_top_row.addLayout(_cv_section("Buzz Cost (est.)", self._cv_buzz))
        cv_top_row.addSpacing(24)

        self._cv_type = QLabel("—")
        self._cv_type.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:13px; font-weight:bold; background:transparent;")
        cv_top_row.addLayout(_cv_section("CivitAI Type", self._cv_type))
        cv_top_row.addStretch(1)
        layout.addWidget(cv_top)

        rf, self._cv_v = self._make_results([
            ("Repeats",       "repeats"),    ("Epochs",       "epochs"),
            ("Steps",         "steps"),      ("UNet LR",      "lr"),
            ("Text Enc LR",   "te_lr"),      ("LR Scheduler", "scheduler"),
            ("LR Cycles",     "lr_cycles"),  ("Min SNR Gamma","min_snr"),
            ("Weight Decay",  "weight_decay"),("Network Dim", "net_dim"),
            ("Network Alpha", "net_alpha"),  ("Noise Offset", "noise_offset"),
            ("Optimizer",     "optimizer"),  ("ETS Signal",   "ets"),
        ])
        layout.addWidget(rf)

        # CivitAI-specific status (Cap Status | Dataset Status | TOS/img)
        cv_st = QFrame()
        cv_st.setStyleSheet(_card_style())
        cv_st_row = QHBoxLayout(cv_st)
        cv_st_row.setContentsMargins(12, 6, 12, 8)
        cv_st_row.setSpacing(0)

        def _cv_status_section(title: str, w_inner: QWidget):
            col = QVBoxLayout()
            col.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet(
                f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
            col.addWidget(t)
            col.addWidget(w_inner)
            return col

        self._cv_cap_lbl = QLabel("—")
        self._cv_cap_lbl.setStyleSheet(
            f"color:{GRN}; font-family:{FONT}; font-size:13px; font-weight:bold; background:transparent;")
        cv_st_row.addLayout(_cv_status_section("Cap Status", self._cv_cap_lbl))
        cv_st_row.addSpacing(24)

        self._cv_img_lbl = QLabel("—")
        self._cv_img_lbl.setStyleSheet(
            f"color:{GRN}; font-family:{FONT}; font-size:13px; font-weight:bold; background:transparent;")
        cv_st_row.addLayout(_cv_status_section("Dataset Status", self._cv_img_lbl))
        cv_st_row.addSpacing(24)

        self._cv_tos_lbl = QLabel("—")
        self._cv_tos_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_MD}px; background:transparent;")
        cv_st_row.addLayout(_cv_status_section("TOS / img", self._cv_tos_lbl))
        cv_st_row.addStretch(1)
        layout.addWidget(cv_st)

        layout.addStretch(1)
        self._tabs.addTab(w, "CivitAI")

    # ── Logs tab ───────────────────────────────────────────────────────────────
    def _build_log_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Log a run
        log_entry = QFrame()
        log_entry.setStyleSheet(_card_style())
        le_layout = QGridLayout(log_entry)
        le_layout.setContentsMargins(12, 8, 12, 8)
        le_layout.setSpacing(8)

        le_layout.addWidget(_lbl("Log a Training Run", PRI, FONT_MD, bold=True), 0, 0, 1, 8)
        for i, txt in enumerate(["Framework", "Steps/Hour", "Sec/Iter"]):
            le_layout.addWidget(_lbl(txt), 1, i * 2, 1, 2)

        self._log_fw = QComboBox()
        self._log_fw.addItems(["aitoolkit", "kohya"])
        self._log_fw.setStyleSheet(_combo_style())
        self._log_fw.setFixedWidth(110)
        le_layout.addWidget(self._log_fw, 2, 0, 1, 2)

        self._log_sph = QLineEdit()
        self._log_sph.setPlaceholderText("260")
        self._log_sph.setFixedWidth(90)
        self._log_sph.setStyleSheet(_edit_style())
        le_layout.addWidget(self._log_sph, 2, 2, 1, 2)

        self._log_sit = QLineEdit()
        self._log_sit.setPlaceholderText("13.8")
        self._log_sit.setFixedWidth(80)
        self._log_sit.setStyleSheet(_edit_style())
        le_layout.addWidget(self._log_sit, 2, 4, 1, 2)

        log_btn = _btn("Log Run", ACC, width=90)
        log_btn.clicked.connect(self.do_log)
        le_layout.addWidget(log_btn, 2, 6)

        self._log_status = QLabel("")
        self._log_status.setStyleSheet(
            f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        le_layout.addWidget(self._log_status, 2, 7)
        layout.addWidget(log_entry)

        # Parse from console
        parse_frame = QFrame()
        parse_frame.setStyleSheet("background:transparent;")
        parse_row = QHBoxLayout(parse_frame)
        parse_row.setContentsMargins(0, 0, 0, 0)
        parse_row.setSpacing(8)

        for side, fw_key, parse_cb, stat_attr, parse_attr in [
            ("AI-Toolkit", "aitoolkit", self._parse_at,    "_at_pstat",  "_at_parse_edit"),
            ("Kohya",      "kohya",     self._parse_kohya, "_ko_pstat",  "_ko_parse_edit"),
        ]:
            pf = QFrame()
            pf.setStyleSheet(_card_style())
            pf_row = QHBoxLayout(pf)
            pf_row.setContentsMargins(8, 6, 8, 6)
            pf_row.setSpacing(6)
            pe = QLineEdit()
            pe.setPlaceholderText(f"Paste {side} console output...")
            pe.setStyleSheet(_edit_style())
            setattr(self, parse_attr, pe)
            pf_row.addWidget(pe, stretch=1)
            pb = _btn("Parse & Log", ACC, width=100)
            pb.clicked.connect(parse_cb)
            pf_row.addWidget(pb)
            sl = QLabel("")
            sl.setStyleSheet(
                f"color:{GRN}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
            sl.setFixedWidth(80)
            setattr(self, stat_attr, sl)
            pf_row.addWidget(sl)
            parse_row.addWidget(pf, stretch=1)

        layout.addWidget(parse_frame)

        # Two-column log display
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setStyleSheet(f"QSplitter::handle {{ background:{MUT}; width:1px; }}")
        split.setHandleWidth(1)

        self._at_log_frame, at_avg_hdr = self._make_log_column(ACC)
        self._at_log_avg_lbl = at_avg_hdr
        split.addWidget(self._at_log_frame)

        self._ko_log_frame, ko_avg_hdr = self._make_log_column("#185FA5")
        self._ko_log_avg_lbl = ko_avg_hdr
        split.addWidget(self._ko_log_frame)

        layout.addWidget(split, stretch=1)
        self._tabs.addTab(w, "Logs")

        self.refresh_logs()

    def _make_log_column(self, header_color: str) -> tuple[QFrame, QLabel]:
        frame = QFrame()
        frame.setStyleSheet(_card_style())
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        hdr = QFrame()
        hdr.setStyleSheet(f"QFrame {{ background:{header_color}; border-radius:4px; }}")
        hdr.setFixedHeight(30)
        hdr_row = QHBoxLayout(hdr)
        hdr_row.setContentsMargins(8, 4, 8, 4)
        avg_lbl = QLabel("avg: —")
        avg_lbl.setStyleSheet(
            f"color:white; font-family:{FONT}; font-size:{FONT_SM}px; font-weight:bold;")
        hdr_row.addWidget(avg_lbl)
        hdr_row.addStretch(1)
        cols_lbl = QLabel("Steps/h   S/it")
        cols_lbl.setStyleSheet(
            f"color:white; font-family:{FONT}; font-size:{FONT_SM}px;")
        hdr_row.addWidget(cols_lbl)
        layout.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{CAR}; }}")
        entries_w = QWidget()
        entries_w.setStyleSheet(f"background:{CAR};")
        entries_layout = QVBoxLayout(entries_w)
        entries_layout.setContentsMargins(0, 2, 0, 2)
        entries_layout.setSpacing(2)
        entries_layout.addStretch(1)
        scroll.setWidget(entries_w)
        layout.addWidget(scroll, stretch=1)

        # Store references so refresh_logs can repopulate
        frame._scroll   = scroll
        frame._entries_layout = entries_layout
        frame._entries_widget = entries_w

        return frame, avg_lbl

    def refresh_logs(self):
        log = load_log()
        for frame, fw, avg_lbl, name in [
            (self._at_log_frame, "aitoolkit", self._at_log_avg_lbl, "AI-Toolkit"),
            (self._ko_log_frame, "kohya",     self._ko_log_avg_lbl, "Kohya"),
        ]:
            entries_layout: QVBoxLayout = frame._entries_layout
            # Clear existing widgets (keep the stretch)
            while entries_layout.count() > 1:
                item = entries_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            entries = log.get(fw, [])
            if not entries:
                empty = _lbl("No entries yet")
                empty.setContentsMargins(8, 8, 8, 8)
                entries_layout.insertWidget(0, empty)
            else:
                for i, entry in enumerate(reversed(entries)):
                    idx = len(entries) - 1 - i
                    bg  = PAN if i % 2 == 0 else CAR
                    row = QFrame()
                    row.setStyleSheet(f"QFrame {{ background:{bg}; border-radius:3px; }}")
                    row.setFixedHeight(28)
                    row_layout = QHBoxLayout(row)
                    row_layout.setContentsMargins(6, 2, 4, 2)
                    row_layout.setSpacing(4)

                    date = entry.get("date", "—")
                    sph  = round(entry["steps_per_hour"], 1)
                    sit  = round(entry["sec_per_iter"], 2)
                    tag  = "📌 " if "Preloaded" in date else ""
                    date_lbl = QLabel(f"{tag}{date[:16]}")
                    date_lbl.setStyleSheet(
                        f"color:{MUT}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
                    row_layout.addWidget(date_lbl, stretch=1)

                    vals_lbl = QLabel(f"{sph:>7}   {sit:>5}")
                    vals_lbl.setStyleSheet(
                        f"color:{PRI}; font-family:{FONT}; font-size:{FONT_SM}px;"
                        f"font-weight:bold; background:transparent;")
                    row_layout.addWidget(vals_lbl)

                    if "Preloaded" not in date:
                        del_btn = QPushButton("✕")
                        del_btn.setFixedSize(18, 18)
                        del_btn.setStyleSheet(
                            f"QPushButton {{ background:transparent; color:{MUT}; border:none; font-size:10px; }}"
                            f"QPushButton:hover {{ color:{RED}; }}")
                        del_btn.clicked.connect(
                            lambda _, f=fw, ix=idx: self._del_log_entry(f, ix))
                        row_layout.addWidget(del_btn)

                    entries_layout.insertWidget(i, row)

            sph_avg, sit_avg = get_averages(fw)
            cnt = len(entries)
            if sph_avg:
                avg_lbl.setText(
                    f"{name} ({cnt}/20)  avg: {round(sph_avg,1)} s/h  {round(sit_avg,2)} s/it")
            else:
                avg_lbl.setText(f"{name} (0/20)  avg: —")

    def _del_log_entry(self, fw: str, idx: int):
        delete_log_entry(fw, idx)
        self.refresh_logs()
        self._calc()

    def do_log(self):
        try:
            add_log_entry(self._log_fw.currentText(),
                          float(self._log_sph.text()),
                          float(self._log_sit.text()))
            self._log_status.setText("✓ Logged")
            self._log_sph.clear()
            self._log_sit.clear()
            self.refresh_logs()
            self._calc()
            QTimer.singleShot(3000, lambda: self._log_status.setText(""))
        except Exception:
            self._log_status.setText("⚠ Invalid")
            QTimer.singleShot(3000, lambda: self._log_status.setText(""))

    def _parse_at(self):
        raw = self._at_parse_edit.text().strip()
        m_step = re.search(r'\|\s*(\d+)/', raw)
        m_time = re.search(r'\[(\d+):(\d+):(\d+)<', raw)
        if m_step and m_time:
            steps   = int(m_step.group(1))
            elapsed = (int(m_time.group(1)) * 3600
                       + int(m_time.group(2)) * 60
                       + int(m_time.group(3)))
            if steps > 0 and elapsed > 0:
                sph = round(steps / elapsed * 3600, 1)
                sit = round(elapsed / steps, 3)
                add_log_entry("aitoolkit", sph, sit)
                self._at_parse_edit.clear()
                self._at_pstat.setText("✓ Logged")
                QTimer.singleShot(3000, lambda: self._at_pstat.setText(""))
                self.refresh_logs()
                self._calc()
                return
        self._at_pstat.setText("⚠ Not found")
        QTimer.singleShot(3000, lambda: self._at_pstat.setText(""))

    def _parse_kohya(self):
        raw = self._ko_parse_edit.text().strip()
        m = re.search(r'(\d+\.?\d*)s/it', raw)
        if m:
            sit = float(m.group(1))
            sph = round(3600 / sit, 1)
            add_log_entry("kohya", sph, sit)
            self._ko_parse_edit.clear()
            self._ko_pstat.setText("✓ Logged")
            QTimer.singleShot(3000, lambda: self._ko_pstat.setText(""))
            self.refresh_logs()
            self._calc()
            return
        m = re.search(r'(\d+\.?\d*)it/s', raw)
        if m:
            its = float(m.group(1))
            sit = round(1 / its, 3)
            sph = round(3600 * its, 1)
            add_log_entry("kohya", sph, sit)
            self._ko_parse_edit.clear()
            self._ko_pstat.setText("✓ Logged")
            QTimer.singleShot(3000, lambda: self._ko_pstat.setText(""))
            self.refresh_logs()
            self._calc()
            return
        self._ko_pstat.setText("⚠ Not found")
        QTimer.singleShot(3000, lambda: self._ko_pstat.setText(""))

    # ── TOS tab ────────────────────────────────────────────────────────────────
    def _build_tos_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")

        inner = QWidget()
        inner.setStyleSheet(f"background:{PAN};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Target Optimization Signal (TOS)")
        title.setStyleSheet(
            f"color:{ACC}; font-family:{FONT}; font-size:13px; font-weight:bold; background:transparent;")
        layout.addWidget(title)

        explanation = (
            "TOS — Target Optimization Signal — defines how many effective gradient updates each image "
            "in your dataset receives across the entire training run. It is the core exposure metric "
            "that drives step and repeat calculations across all three trainers.\n\n"
            "How it was derived: TOS values were established through community research and validated "
            "against real training outcomes across multiple LoRA types. Each type requires a different "
            "level of exposure to learn its target concept without underfitting or overfitting. "
            "Character LoRAs need high exposure to lock in consistent identity across varied poses and "
            "lighting. Style LoRAs need the highest exposure to capture subtle rendering aesthetics. "
            "Concept LoRAs need moderate exposure — enough to teach the visual concept without burning "
            "in irrelevant details. Outfit and Pose LoRAs sit between these.\n\n"
            "The table below shows the validated Min, Mid and Max values for each type. Min represents "
            "a conservative baseline. Mid is a balanced target for typical datasets. Max is the "
            "community high-end recommendation and is the default used by the calculator.\n\n"
            "Select a preset per type using the Preset column. If you need a value outside the "
            "validated range, enter it directly in the Manual Override column. "
            "⚠ A value in Manual Override always takes priority over the preset and applies across "
            "all three trainers immediately. Clear the override to return to preset behavior."
        )
        exp_frame = QFrame()
        exp_frame.setStyleSheet(_card_style())
        exp_lbl = QLabel(explanation)
        exp_lbl.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        exp_lbl.setWordWrap(True)
        exp_lbl.setContentsMargins(12, 10, 12, 10)
        ef_layout = QVBoxLayout(exp_frame)
        ef_layout.setContentsMargins(0, 0, 0, 0)
        ef_layout.addWidget(exp_lbl)
        layout.addWidget(exp_frame)

        # Warning banner
        warn_frame = QFrame()
        warn_frame.setStyleSheet(
            f"QFrame {{ background:#3a2a00; border-radius:6px; border:1px solid {AMB}; }}")
        warn_lbl = QLabel(
            f"⚠  Manual Override supersedes the Preset and applies across AI-Toolkit, Kohya and CivitAI. "
            f"Clear the field to restore preset behavior.")
        warn_lbl.setStyleSheet(
            f"color:{AMB}; font-family:{FONT}; font-size:{FONT_SM}px; font-weight:bold;"
            f"background:transparent;")
        warn_lbl.setWordWrap(True)
        warn_lbl.setContentsMargins(12, 8, 12, 8)
        wf_layout = QVBoxLayout(warn_frame)
        wf_layout.setContentsMargins(0, 0, 0, 0)
        wf_layout.addWidget(warn_lbl)
        layout.addWidget(warn_frame)

        # Table header
        hdr = QFrame()
        hdr.setStyleSheet(f"QFrame {{ background:{ACC}; border-radius:6px; }}")
        hdr.setFixedHeight(32)
        hdr_row = QHBoxLayout(hdr)
        hdr_row.setContentsMargins(8, 4, 8, 4)
        for txt, w_px in [("LoRA Type", 130), ("Min", 60), ("Mid", 60), ("Max", 60),
                           ("Preset", 160), ("Manual Override", 160)]:
            col_lbl = QLabel(txt)
            col_lbl.setStyleSheet(
                f"color:white; font-family:{FONT}; font-size:{FONT_SM}px; font-weight:bold;")
            col_lbl.setFixedWidth(w_px)
            col_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hdr_row.addWidget(col_lbl)
        hdr_row.addStretch(1)
        layout.addWidget(hdr)

        # Table rows
        self._tos_preset_combos:  dict[str, QComboBox]  = {}
        self._tos_override_edits: dict[str, QLineEdit]  = {}
        presets   = self._cfg.get("tos_presets",  {})
        overrides = self._cfg.get("tos_overrides", {})

        for i, lt in enumerate(LORA_TYPES):
            r    = TOS_RANGES[lt]
            row  = QFrame()
            row.setStyleSheet(f"QFrame {{ background:{CAR if i%2==0 else PAN}; border-radius:6px; }}")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 4, 8, 4)
            row_layout.setSpacing(0)

            type_lbl = QLabel(lt.capitalize())
            type_lbl.setStyleSheet(
                f"color:{PRI}; font-family:{FONT}; font-size:{FONT_MD}px; background:transparent;")
            type_lbl.setFixedWidth(130)
            row_layout.addWidget(type_lbl)

            for val_str, w_px, color in [
                (str(r["min"]), 60, SEC),
                (str(r["mid"]), 60, SEC),
                (str(r["max"]), 60, PRI),
            ]:
                v_lbl = QLabel(val_str)
                v_lbl.setStyleSheet(
                    f"color:{color}; font-family:{FONT}; font-size:{FONT_MD}px;"
                    f"{'font-weight:bold;' if color==PRI else ''} background:transparent;")
                v_lbl.setFixedWidth(w_px)
                v_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                row_layout.addWidget(v_lbl)

            preset_combo = QComboBox()
            preset_combo.addItems(["Conservative", "Balanced", "Recommended"])
            preset_combo.setCurrentText(presets.get(lt, "Recommended"))
            preset_combo.setStyleSheet(_combo_style())
            preset_combo.setFixedWidth(150)
            preset_combo.currentTextChanged.connect(lambda _, l=lt: self._tos_changed(l))
            self._tos_preset_combos[lt] = preset_combo
            row_layout.addSpacing(6)
            row_layout.addWidget(preset_combo)

            override_edit = QLineEdit(str(overrides.get(lt, "")))
            override_edit.setPlaceholderText("e.g. 175")
            override_edit.setStyleSheet(
                f"QLineEdit {{ background:{CAR}; color:{PRI}; border:2px solid {AMB};"
                f"border-radius:4px; padding:2px 6px; font-family:{FONT}; font-size:{FONT_MD}px; }}"
                f"QLineEdit:focus {{ border-color:#ffb347; }}")
            override_edit.setFixedWidth(120)
            override_edit.textChanged.connect(lambda _, l=lt: self._tos_changed(l))
            self._tos_override_edits[lt] = override_edit
            row_layout.addSpacing(6)
            row_layout.addWidget(override_edit)
            row_layout.addStretch(1)

            layout.addWidget(row)

        layout.addStretch(1)
        scroll.setWidget(inner)

        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._tabs.addTab(w, "TOS")

    def _tos_changed(self, lt: str):
        self._cfg["tos_presets"]  = {l: self._tos_preset_combos[l].currentText()  for l in LORA_TYPES}
        self._cfg["tos_overrides"] = {l: self._tos_override_edits[l].text()         for l in LORA_TYPES}
        save_json(CALC_CFG, self._cfg)
        self._calc()

    def get_tos(self, lt: str) -> float:
        overrides = self._cfg.get("tos_overrides", {})
        ov = str(overrides.get(lt, "")).strip()
        if ov:
            try:
                v = float(ov)
                if v > 0:
                    return v
            except Exception:
                pass
        presets = self._cfg.get("tos_presets", {})
        preset  = str(presets.get(lt, "Recommended")).lower()
        r = TOS_RANGES.get(lt, {"min": 50, "mid": 100, "max": 150})
        if preset == "conservative":
            return r["min"]
        elif preset == "balanced":
            return r["mid"]
        return r["max"]

    # ── Help tab ───────────────────────────────────────────────────────────────
    def _build_help_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{PAN};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Section selector
        sel_frame = QFrame()
        sel_frame.setStyleSheet(_card_style())
        sel_row = QHBoxLayout(sel_frame)
        sel_row.setContentsMargins(12, 8, 12, 8)
        sel_row.setSpacing(8)
        sel_row.addWidget(_lbl("Section"))
        self._help_combo = QComboBox()
        self._help_combo.addItems(["AI-Toolkit", "Kohya", "CivitAI"])
        self._help_combo.setStyleSheet(_combo_style())
        self._help_combo.setFixedWidth(140)
        self._help_combo.currentTextChanged.connect(self._refresh_help)
        sel_row.addWidget(self._help_combo)
        sel_row.addStretch(1)
        layout.addWidget(sel_frame)

        # Cards scroll area
        self._help_scroll = QScrollArea()
        self._help_scroll.setWidgetResizable(True)
        self._help_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._help_scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PAN}; }}")
        layout.addWidget(self._help_scroll, stretch=1)

        self._tabs.addTab(w, "Help")
        self._refresh_help()

    def _refresh_help(self):
        inner = QWidget()
        inner.setStyleSheet(f"background:{PAN};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(2)
        sep.setStyleSheet(f"background:{ACC}; border-radius:1px;")
        layout.addWidget(sep)

        # Shared help items
        for param, desc in SHARED_HELP:
            card = self._make_help_card(param, desc, title_color=AMB)
            layout.addWidget(card)

        sep2 = QFrame()
        sep2.setFixedHeight(2)
        sep2.setStyleSheet(f"background:{MUT}; border-radius:1px;")
        layout.addWidget(sep2)

        # Section-specific items
        section = self._help_combo.currentText()
        for param, desc in HELP_CONTENT.get(section, []):
            card = self._make_help_card(param, desc, title_color=ACC)
            layout.addWidget(card)

        layout.addStretch(1)
        self._help_scroll.setWidget(inner)

    def _make_help_card(self, param: str, desc: str, title_color: str = ACC) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_card_style())
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 8, 12, 8)
        card_layout.setSpacing(2)

        title = QLabel(param)
        title.setStyleSheet(
            f"color:{title_color}; font-family:{FONT}; font-size:{FONT_MD}px;"
            f"font-weight:bold; background:transparent;")
        title.setFixedWidth(160)
        card_layout.addWidget(title)

        body = QLabel(desc)
        body.setStyleSheet(
            f"color:{SEC}; font-family:{FONT}; font-size:{FONT_SM}px; background:transparent;")
        body.setWordWrap(True)
        card_layout.addWidget(body)

        return card

    # ── Calculation ────────────────────────────────────────────────────────────
    def _calc(self):
        try:
            lt   = self._lt_combo.currentText()
            rc   = self._rc_combo.currentText()
            imgs = get_images(self._files_edit.text(), self._tagged_cb.isChecked())
            try:
                batch = max(1, int(self._batch_edit.text()))
            except Exception:
                batch = 2
            self._total_img_lbl.setText(str(imgs))
            tos = self.get_tos(lt)

            # AI-Toolkit
            try:
                af = float(self._at_fe.text())
            except Exception:
                af = 1.0
            sph_at, _ = get_averages("aitoolkit")
            at = calc_at(lt, imgs, af, rc, sph_at, batch, tos=tos)
            if at:
                self._at_data = at
                for k, v in self._at_v.items():
                    v.setText(str(at.get(k, "—")))
                self._at_ets.setText(at["ets"])
                self._at_ets.setStyleSheet(
                    f"color:{at['ets_color']}; font-family:{FONT}; font-size:14px;"
                    f"font-weight:bold; background:transparent;")
                self._at_tos.setText(f"{at['actual_tos']} / {at['target_tos']}")
                self._at_sl.setText(str(at["suggested_factor"]))
                self._at_el.setText(at["eta"] if at["eta"] else "— (log runs to calibrate)")
                self._at_warn.setText(at["img_warning"])

            # Kohya
            try:
                kf = float(self._ko_fe.text())
            except Exception:
                kf = 1.0
            sph_ko, _ = get_averages("kohya")
            ko = calc_ko(lt, imgs, kf, rc, sph_ko, batch, tos=tos)
            if ko:
                for k, v in self._ko_v.items():
                    v.setText(str(ko.get(k, "—")))
                self._ko_ets.setText(ko["ets"])
                self._ko_ets.setStyleSheet(
                    f"color:{ko['ets_color']}; font-family:{FONT}; font-size:14px;"
                    f"font-weight:bold; background:transparent;")
                self._ko_tos.setText(f"{ko['actual_tos']} / {ko['target_tos']}")
                self._ko_sl.setText(str(ko["suggested_factor"]))
                self._ko_el.setText(ko["eta"] if ko["eta"] else "— (log runs to calibrate)")
                self._ko_warn.setText(ko["img_warning"])

            # CivitAI
            try:
                cf = float(self._cv_fe.text())
            except Exception:
                cf = 1.0
            cv = calc_cv(lt, imgs, cf, rc, tos=tos)
            if cv:
                for k, v in self._cv_v.items():
                    v.setText(str(cv.get(k, "—")))
                self._cv_sl.setText(str(cv["suggested_factor"]))
                self._cv_buzz.setText(f"{cv['buzz']} Buzz")
                self._cv_type.setText(cv["civitai_type"])
                self._cv_cap_lbl.setText(cv["cap_warning"])
                self._cv_cap_lbl.setStyleSheet(
                    f"color:{cv['cap_color']}; font-family:{FONT}; font-size:13px;"
                    f"font-weight:bold; background:transparent;")
                self._cv_img_lbl.setText(cv["img_warning"])
                self._cv_img_lbl.setStyleSheet(
                    f"color:{cv['img_warning_color']}; font-family:{FONT}; font-size:13px;"
                    f"font-weight:bold; background:transparent;")
                self._cv_tos_lbl.setText(f"{cv['actual_tos']} / {cv['target_tos']}")

        except Exception:
            pass

    # ── Browse actions ─────────────────────────────────────────────────────────
    def _browse_dataset(self):
        last = self._cfg.get("last_dataset_dir", "") or ""
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", last or "/")
        if folder:
            self._dataset_path = folder
            self._cfg["last_dataset_dir"] = os.path.dirname(folder)
            count = count_dataset_files(folder)
            self._files_edit.setText(str(count))
            # Auto-populate LoRA name
            self._lora_name_edit.setText(os.path.basename(folder))
            self._calc()

    def _browse_aitoolkit(self):
        last = self._cfg.get("aitoolkit_path", "") or ""
        folder = QFileDialog.getExistingDirectory(self, "Select AI-Toolkit Folder", last or "/")
        if folder:
            self._at_path_edit.setText(folder)
            self._cfg["aitoolkit_path"] = folder

    def _browse_model(self):
        last = self._cfg.get("last_model_dir", "") or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Base Model", last or "/",
            "Model files (*.safetensors *.ckpt);;All files (*.*)")
        if path:
            self._model_path_edit.setText(path)
            self._cfg["base_model_path"] = path
            self._cfg["last_model_dir"]  = os.path.dirname(path)

    def _toggle_network(self):
        visible = self._network_cb.isChecked()
        self._net_frame.setVisible(visible)
        self._cfg["in_network"] = visible
        save_json(CALC_CFG, self._cfg)

    # ── Export ─────────────────────────────────────────────────────────────────
    def _save_export_cfg(self):
        self._cfg["default_caption"] = self._caption_edit.text()
        self._cfg["version"]          = self._version_edit.text()
        self._cfg["sample_prompts"]   = [e.text() for e in self._at_prompt_edits[:3]]
        self._cfg["in_network"]       = self._network_cb.isChecked()
        self._cfg["local_drive"]      = self._local_drive_edit.text().strip().upper()
        self._cfg["remote_drive"]     = self._remote_drive_edit.text().strip().upper()
        self._cfg["bucket_512"]       = self._bucket_512_cb.isChecked()
        self._cfg["bucket_256"]       = self._bucket_256_cb.isChecked()
        self._cfg["cache_latents"]    = self._cache_latents_cb.isChecked()
        save_json(CALC_CFG, self._cfg)

    def _set_exp_status(self, msg: str, delay: int = 4000):
        self._exp_status.setText(msg)
        QTimer.singleShot(delay, lambda: self._exp_status.setText(""))

    def _build_yaml_config(self) -> tuple[dict | None, str | None]:
        lora_name  = self._lora_name_edit.text().strip()
        at_path    = self._at_path_edit.text().strip()
        model_path = self._model_path_edit.text().strip()

        if not lora_name:
            return None, "⚠ LoRA Name required"
        if not model_path:
            return None, "⚠ Base Model path required"
        if not self._at_data:
            return None, "⚠ No calc data — enter image count first"

        def remap(path: str) -> str:
            if self._network_cb.isChecked():
                loc = self._local_drive_edit.text().strip().upper()
                rem = self._remote_drive_edit.text().strip().upper()
                if loc and rem and path.upper().startswith(f"{loc}:"):
                    path = f"{rem}:{path[2:]}"
            return path.replace("/", "\\")

        d = self._at_data
        steps       = d.get("steps", 3000)
        grad        = d.get("grad_accum", 6)
        lr_val      = float(d.get("lr", "0.000036"))
        wd          = d.get("weight_decay", 0.009)
        cd          = d.get("caption_dropout", 0.15)
        lrank       = d.get("linear_rank", 64)
        crank       = d.get("conv_rank", 32)
        save_every  = d.get("save_every", 500)
        max_saves   = d.get("max_saves", 4)

        if self._dataset_path:
            dataset_folder = remap(self._dataset_path)
        elif at_path:
            dataset_folder = remap(at_path + "\\datasets\\my_dataset")
        else:
            dataset_folder = "I:\\AI-Toolkit\\datasets\\my_dataset"

        if "\\" in dataset_folder:
            parts = dataset_folder.rsplit("\\", 1)
            dataset_folder = parts[0] + "/" + parts[1] if len(parts) == 2 else dataset_folder

        model_path_yml  = remap(model_path)
        training_folder = remap(os.path.join(at_path, "output")) if at_path else "I:\\AI-Toolkit\\output"
        caption         = self._caption_edit.text()
        version         = self._version_edit.text() or "1.0"
        trigger_txt     = self._trigger_edit.text().strip() or None
        prompts = [{"prompt": e.text()} for e in self._at_prompt_edits if e.text().strip()]
        if not prompts:
            prompts = [{"prompt": "Your prompt here"}]

        try:
            batch = max(1, int(self._batch_edit.text()))
        except Exception:
            batch = 2

        buckets = get_bucket_resolutions(
            self._bucket_512_cb.isChecked(), self._bucket_256_cb.isChecked())

        config = {
            "job": "extension",
            "config": {
                "name": lora_name,
                "process": [{
                    "type": "diffusion_trainer",
                    "training_folder": training_folder,
                    "sqlite_db_path": "./aitk_db.db",
                    "device": "cuda",
                    "trigger_word": trigger_txt,
                    "performance_log_every": 10,
                    "network": {
                        "type": "lora", "linear": lrank, "linear_alpha": lrank,
                        "conv": crank, "conv_alpha": crank,
                        "lokr_full_rank": True, "lokr_factor": -1,
                        "network_kwargs": {"ignore_if_contains": []},
                    },
                    "save": {
                        "dtype": "fp16", "save_every": save_every,
                        "max_step_saves_to_keep": max_saves,
                        "save_format": "diffusers", "push_to_hub": False,
                    },
                    "datasets": [{
                        "folder_path": dataset_folder,
                        "mask_path": None, "mask_min_value": 0.1,
                        "default_caption": caption, "caption_ext": "txt",
                        "caption_dropout_rate": cd,
                        "cache_latents_to_disk": self._cache_latents_cb.isChecked(),
                        "is_reg": False, "network_weight": 1,
                        "resolution": buckets,
                        "controls": [], "shrink_video_to_frames": True,
                        "num_frames": 1, "do_i2v": True, "flip_x": True, "flip_y": False,
                    }],
                    "train": {
                        "batch_size": batch, "bypass_guidance_embedding": False,
                        "steps": steps, "gradient_accumulation": grad,
                        "train_unet": True, "train_text_encoder": False,
                        "gradient_checkpointing": True, "noise_scheduler": "ddpm",
                        "optimizer": "adamw8bit", "timestep_type": "sigmoid",
                        "content_or_style": "balanced",
                        "optimizer_params": {"weight_decay": wd},
                        "unload_text_encoder": False, "cache_text_embeddings": False,
                        "lr": lr_val,
                        "ema_config": {"use_ema": False, "ema_decay": 0.99},
                        "skip_first_sample": False, "force_first_sample": False,
                        "disable_sampling": False, "dtype": "bf16",
                        "diff_output_preservation": False,
                        "diff_output_preservation_multiplier": 1,
                        "diff_output_preservation_class": "person",
                        "switch_boundary_every": 1, "loss_type": "mse",
                    },
                    "model": {
                        "name_or_path": model_path_yml, "quantize": False,
                        "qtype": "qfloat8", "quantize_te": False, "qtype_te": "qfloat8",
                        "arch": "sdxl", "low_vram": False, "model_kwargs": {},
                    },
                    "sample": {
                        "sampler": "ddpm", "sample_every": save_every,
                        "width": 1024, "height": 1024, "samples": prompts,
                        "neg": "", "seed": 42, "walk_seed": True,
                        "guidance_scale": 6, "sample_steps": 25, "num_frames": 1, "fps": 1,
                    },
                }],
            },
            "meta": {"name": lora_name, "version": version},
        }
        return config, None

    def do_export(self):
        self._save_export_cfg()
        config, err = self._build_yaml_config()
        if err:
            self._set_exp_status(err, 3000)
            return
        lora_name = self._lora_name_edit.text().strip()
        last = self._cfg.get("last_export_dir", "") or os.path.expanduser("~/Desktop")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save YAML Config", os.path.join(last, f"{lora_name}.yml"),
            "YAML files (*.yml *.yaml);;All files (*.*)")
        if not path:
            return
        self._cfg["last_export_dir"] = os.path.dirname(path)
        save_json(CALC_CFG, self._cfg)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(build_at_yaml(config))
            self._set_exp_status(f"✓ Saved: {os.path.basename(path)}", 5000)
        except Exception as e:
            self._set_exp_status(f"⚠ Error: {str(e)[:50]}")

    def do_create_job(self):
        self._save_export_cfg()
        config, err = self._build_yaml_config()
        if err:
            self._set_exp_status(err, 3000)
            return
        try:
            yml = build_at_yaml(config)
            QApplication.clipboard().setText(yml)
            self._set_exp_status("✓ Copied — paste into AI-Toolkit Show Advanced", 6000)
        except Exception as e:
            self._set_exp_status(f"⚠ Error: {str(e)[:50]}")

    def do_ko_export(self):
        self._cfg["kohya_sample_prompts"] = [e.text() for e in self._ko_prompt_edits[:3]]
        save_json(CALC_CFG, self._cfg)
        prompts = [e.text().strip() for e in self._ko_prompt_edits if e.text().strip()]
        if not prompts:
            self._ko_exp_status.setText("⚠ No prompts to export")
            QTimer.singleShot(3000, lambda: self._ko_exp_status.setText(""))
            return
        last = self._cfg.get("last_kohya_prompts_dir", "") or os.path.expanduser("~/Desktop")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Kohya Sample Prompts",
            os.path.join(last, "sample_prompts.txt"),
            "Text files (*.txt);;All files (*.*)")
        if not path:
            return
        self._cfg["last_kohya_prompts_dir"] = os.path.dirname(path)
        save_json(CALC_CFG, self._cfg)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(prompts) + "\n")
            self._ko_exp_status.setText(f"✓ Saved: {os.path.basename(path)}")
            QTimer.singleShot(5000, lambda: self._ko_exp_status.setText(""))
        except Exception as e:
            self._ko_exp_status.setText(f"⚠ Error: {str(e)[:40]}")
            QTimer.singleShot(4000, lambda: self._ko_exp_status.setText(""))

    def closeEvent(self, event):
        save_json(CALC_CFG, self._cfg)
        event.accept()
