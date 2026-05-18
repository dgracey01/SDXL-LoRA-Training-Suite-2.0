"""
shared/calc_engine.py — LoRA training math and YAML generation for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.0)

Pure Python — no Qt dependencies. All calculation logic lives here so it can be
unit-tested independently of the UI.
"""

import math
import os
import re
import json
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
SUITE_ROOT = os.path.dirname(_HERE)
LOG_FILE   = os.path.join(SUITE_ROOT, "calculator", "training_log.json")


# ── SDXL resolution map ───────────────────────────────────────────────────────
# (display_label, width, height, ratio_label)
RESOLUTIONS: list[tuple[str, int, int, str]] = [
    ("1024 × 1024", 1024, 1024, "1:1"),
    ("1152 × 896",  1152,  896, "4:3"),
    ("1216 × 832",  1216,  832, "3:2"),
    ("1344 × 768",  1344,  768, "16:9"),
    ("1536 × 640",  1536,  640, "12:5"),
    ("896 × 1152",   896, 1152, "3:4"),
    ("768 × 1344",   768, 1344, "9:16"),
    ("768 × 768",    768,  768, "SD HD"),
    ("512 × 512",    512,  512, "SD 1.5"),
]

RES_LABELS = [r[0] for r in RESOLUTIONS]

# ── Network architectures ─────────────────────────────────────────────────────
NETWORK_TYPES  = ["LoRA", "LyCORIS / LoKr", "LyCORIS / LoHa", "DoRA"]
OPTIMIZERS     = ["AdamW8bit", "AdamW", "Prodigy", "DAdaptAdam", "Lion8bit", "SGDNesterov"]
SCHEDULERS     = ["cosine_with_restarts", "cosine", "linear", "constant", "polynomial"]
PRECISIONS     = ["bf16", "fp16", "fp32"]
SAVE_FORMATS   = ["safetensors", "ckpt"]

# ── VRAM estimates per optimizer (GB, rough rule-of-thumb for SDXL) ───────────
_VRAM_BASE: dict[str, float] = {
    "AdamW8bit":   10.0,
    "AdamW":       14.0,
    "Prodigy":     12.0,
    "DAdaptAdam":  12.0,
    "Lion8bit":    10.0,
    "SGDNesterov":  9.0,
}

# ── Steps/time per batch for common batch sizes (seconds, A100 rough estimate) ─
_SEC_PER_STEP_PER_BS: float = 1.8   # seconds per step at batch=1, resolution=1024
_BS_SCALE_FACTOR:     float = 0.85  # each doubling of BS saves this fraction of time


# ── V1 TOS-based constants ────────────────────────────────────────────────────
LORA_TYPES  = ["character", "concept", "style", "outfit", "pose"]

TOS = {
    "character": 150, "concept": 48, "style": 240, "outfit": 113, "pose": 130,
}

TOS_RANGES = {
    "character": {"min": 56,  "mid": 103, "max": 150},
    "concept":   {"min": 16,  "mid": 32,  "max": 48},
    "style":     {"min": 120, "mid": 180, "max": 240},
    "outfit":    {"min": 68,  "mid": 90,  "max": 113},
    "pose":      {"min": 65,  "mid": 98,  "max": 130},
}

MIN_IMAGES  = {"character": 50, "concept": 40, "style": 50, "outfit": 50, "pose": 50}
CIVITAI_MAX = {"character": 500, "concept": 700, "style": 300, "outfit": 700, "pose": 600}

WD = {"character": 0.01,  "concept": 0.01,  "style": 0.01,  "outfit": 0.01,  "pose": 0.01}
CD = {"character": 0.05,  "concept": 0.05,  "style": 0.10,  "outfit": 0.08,  "pose": 0.10}

MAX_LOG         = 20
IMAGE_EXTS      = {'.jpg', '.jpeg', '.png', '.txt'}
RANK_CAP_OPTIONS = ["auto", "min", "outfit", "default", "large", "xlarge", "max"]

PRELOADED_AT = [
    {"date": "Preloaded", "steps_per_hour": 187.4,  "sec_per_iter": 19.2},
    {"date": "Preloaded", "steps_per_hour": 262.2,  "sec_per_iter": 13.73},
    {"date": "Preloaded", "steps_per_hour": 262.2,  "sec_per_iter": 13.73},
    {"date": "Preloaded", "steps_per_hour": 264.7,  "sec_per_iter": 13.6},
    {"date": "Preloaded", "steps_per_hour": 133.3,  "sec_per_iter": 27.0},
    {"date": "Preloaded", "steps_per_hour": 254.13, "sec_per_iter": 14.17},
    {"date": "Preloaded", "steps_per_hour": 251.24, "sec_per_iter": 14.33},
    {"date": "Preloaded", "steps_per_hour": 262.17, "sec_per_iter": 13.73},
    {"date": "Preloaded", "steps_per_hour": 287.82, "sec_per_iter": 12.51},
    {"date": "Preloaded", "steps_per_hour": 259.21, "sec_per_iter": 13.89},
    {"date": "Preloaded", "steps_per_hour": 261.23, "sec_per_iter": 13.78},
    {"date": "Preloaded", "steps_per_hour": 204.18, "sec_per_iter": 17.63},
    {"date": "Preloaded", "steps_per_hour": 259.06, "sec_per_iter": 13.90},
    {"date": "Preloaded", "steps_per_hour": 261.78, "sec_per_iter": 13.75},
    {"date": "Preloaded", "steps_per_hour": 235.46, "sec_per_iter": 15.29},
    {"date": "Preloaded", "steps_per_hour": 303.71, "sec_per_iter": 11.85},
    {"date": "Preloaded", "steps_per_hour": 294.71, "sec_per_iter": 12.22},
    {"date": "Preloaded", "steps_per_hour": 204.30, "sec_per_iter": 17.62},
    {"date": "Preloaded", "steps_per_hour": 181.75, "sec_per_iter": 19.81},
    {"date": "Preloaded", "steps_per_hour": 183.29, "sec_per_iter": 19.64},
]

SHARED_HELP = [
    ("Batch Size",    "Maximum number of images to be loaded at once. Highly dependent on your GPU's VRAM."),
    ("Dataset Files", "The total number of files in the training folder. Only .jpg, .jpeg, .png and .txt files are counted. Other formats (.webp, .bmp, .tiff, etc.) are not supported by AI-Toolkit and are excluded."),
    ("Tagged",        "Check if Dataset is already tagged. When ON, Total Images = Dataset Files ÷ 2."),
    ("Rank Cap",      "Controls Linear Rank calculation. Auto = type-aware (recommended). Presets: min=16, outfit=20, default=32, large=64, xlarge=128, max=256."),
]

HELP_CONTENT = {
    "AI-Toolkit": [
        ("LoRA Type",          "Dropdown with 5 types: character, concept, style, outfit, pose. Controls all auto-calculated parameters including LR, weight decay, caption dropout, grad accum, TOS target."),
        ("Steps",              "Total training steps. Sole training control in ai-toolkit — repeats and epochs are not reliably enforced. Formula: (Images × TOS) ÷ Effective Batch × Factor."),
        ("Learning Rate",      "UNet learning rate. Character/Concept: dataset-size curve — ≤50=5e-5, ≤200=7e-5, >200=1e-4. Style/Outfit/Pose: flat 1e-4."),
        ("Weight Decay",       "AdamW8bit regularization. Flat 0.01 for all types — matches the optimizer's own default (weight_decay=1e-2). Type-specific values were spreadsheet rounding errors."),
        ("Caption Dropout",    "Rate at which captions are randomly dropped during training: Character/Concept=0.05, Outfit=0.08, Style/Pose=0.10."),
        ("Grad Accum",         "Gradient accumulation. Inverted curve — smaller datasets get lower accum for more frequent updates: ≤100=1, ≤300=2, ≤700=3, >700=4."),
        ("Eff Batch",          "Effective batch = Batch Size × Grad Accum. This is the actual gradient update frequency."),
        ("Linear Rank",        "LoRA network dimension. Auto (type-aware): character/concept=32 flat; style=32 (≤200 imgs) / 64 (>200 imgs); outfit/pose=16."),
        ("Conv Rank",          "Convolutional rank. Derived from Linear Rank: Character/Concept/Style=50%, Outfit=40%, Pose=35%. Clamped 2–32."),
        ("ETS Signal",         "Effective Training Signal. Compares actual TOS/image against target with 5% tolerance. Good/Underfit/Overfit."),
        ("Suggested Factor",   "The R2 factor value needed to bring ETS back to Good. Copy into Manual Factor field."),
        ("Manual Factor (R2)", "Step count multiplier. Default=1. Increase to add steps when ETS shows Underfit. Decrease when Overfit."),
        ("Max Saves",          "Maximum checkpoint saves. Minimum 2 to always have an intermediate checkpoint."),
        ("Save Every",         "Steps between each checkpoint save."),
        ("ETA",                "Estimated training time based on your logged average steps/hour. Log training runs in the Logs tab to calibrate."),
        ("TOS",                "Target Optimization Signal per image. Character=150, Concept=48, Style=240, Outfit=113, Pose=130."),
        ("Export Settings",    "Two buttons control job creation. Export — saves a .yml config file to a location you choose (remembers last save location). Create Job — generates the YAML and copies it directly to your clipboard. Switch to ai-toolkit Web UI, click New Training Job, click Show Advanced, paste the clipboard contents into the YAML text area, then click Create Job."),
        ("In Network?",        "Check this box if ai-toolkit is running on a different PC via mapped network drive. Local = drive letter on your PC (e.g. X). Remote = drive letter on the ai-toolkit machine (e.g. I). All paths are automatically remapped on export."),
    ],
    "Kohya": [
        ("Repeats",          "Times each image is seen per epoch. Auto: dataset-size curve MAX(1,MIN(15,800/images))."),
        ("Epochs",           "Full training passes. Smooth power 0.55 curve: Character 50img=5ep/500img=20ep, Concept 40img=2ep/1000img=20ep, Style 50img=8ep/300img=20ep."),
        ("Steps",            "Total steps = Repeats × Epochs × Images ÷ Batch. Informational in Kohya — controlled by Repeats and Epochs."),
        ("UNet LR",          "UNet learning rate. Character/Concept: ≤50=5e-5, ≤200=7e-5, >200=1e-4. Style/Outfit/Pose: flat 1e-4. Text Encoder LR = UNet LR ÷ 10."),
        ("Text Enc LR",      "Text Encoder LR = UNet LR ÷ 10. The 10:1 ratio is community confirmed for SDXL Kohya training."),
        ("LR Scheduler",     "Pose always uses linear. All others use cosine_with_restarts with Adafactor/AdamW8bit, cosine with Prodigy/Automagic."),
        ("Optimizer",        "AdamW8bit recommended for 24GB VRAM. Reduces VRAM 25-30% with negligible quality loss."),
        ("Network Alpha",    "Set equal to Linear Rank (1:1 ratio). Effective LR = LR × (Alpha ÷ Dim). At 1:1 the LR is applied at full strength."),
        ("Min SNR Gamma",    "Fixed at 5 for all types. Community unanimous recommendation for SDXL Kohya training."),
        ("Noise Offset",     "Fixed at 0.1 for Kohya. Introduces true darkness and highlights into the model."),
        ("Grad Accum",       "Same inverted curve as AI-Toolkit, capped at 4: ≤100=1, ≤300=2, ≤700=3, >700=4."),
        ("Save Every (ep)",  "Save checkpoint every N epochs. ≤10 epochs=every 1, ≤20 epochs=every 2, >20 epochs=every 5."),
        ("Suggested Factor", "Exact factor value needed to hit target TOS. Copy into Manual Factor."),
    ],
    "CivitAI": [
        ("CivitAI Type",   "CivitAI only supports Character, Concept and Style. Pose and Outfit are submitted as Concept."),
        ("Repeats",        "Enter this value into CivitAI Num Repeats field."),
        ("Epochs",         "Enter this value into CivitAI Epochs field. Hard cap at 20."),
        ("Steps",          "Informational only. CivitAI calculates automatically: Repeats × Epochs × Images ÷ 4."),
        ("UNet LR",        "Enter into CivitAI Unet LR. Character=0.0003, Concept=0.0005, Style=0.0003, Outfit/Pose=0.0004."),
        ("Text Enc LR",    "Enter into CivitAI Text Encoder LR. Always 1/10th of UNet LR."),
        ("LR Scheduler",   "Pose=linear. Others=cosine_with_restarts (Adafactor/AdamW8bit) or cosine (Prodigy/Automagic)."),
        ("LR Cycles",      "Enter into CivitAI LR Scheduler Cycles. Linear=1, all others=3."),
        ("Min SNR Gamma",  "Enter 5 into CivitAI Min SNR Gamma. Fixed for all types."),
        ("Weight Decay",   "Enter into CivitAI Weight Decay field."),
        ("Network Dim",    "Enter into CivitAI Network Dim field. Same as Linear Rank."),
        ("Network Alpha",  "Enter into CivitAI Network Alpha. Set equal to Network Dim for full LR strength."),
        ("Noise Offset",   "Enter into CivitAI Noise Offset. Character=0.1, Concept=0.08, Style=0.06, Outfit=0.09, Pose=0.04."),
        ("Optimizer",      "Enter into CivitAI Optimizer. Adafactor=default. Use Kohya engine."),
        ("Cap Status",     "Within caps = steps ≤10,000 and epochs ≤20. Step/Epoch/Both Caps = TOS not fully achievable."),
        ("Dataset Status", "Good Count = within safe image range. Low Count = below minimum. Exceeds Cap = above max for full TOS."),
        ("Buzz Cost",      "Estimated CivitAI Buzz cost. Formula: 78.6 + (Steps × 0.1156). Custom checkpoint adds ~1000 Buzz."),
        ("CivitAI Limits", "Max 1000 images. Max 20 epochs. Max 10,000 steps. Safe ceilings: Character=500, Concept=700, Style=300, Outfit=700, Pose=600."),
    ],
}


# ── Dataclass representing all user-editable training parameters ───────────────
@dataclass
class TrainingParams:
    # Dataset
    dataset_folder:   str   = ""
    image_count:      int   = 0
    repeats:          int   = 10

    # Model
    model_name:       str   = "stabilityai/stable-diffusion-xl-base-1.0"
    resolution_label: str   = "1024 × 1024"

    # Training
    batch_size:       int   = 1
    epochs:           int   = 10
    learning_rate:    str   = "4e-4"
    network_type:     str   = "LoRA"
    network_dim:      int   = 32
    network_alpha:    int   = 16
    optimizer:        str   = "AdamW8bit"
    scheduler:        str   = "cosine_with_restarts"
    warmup_ratio:     float = 0.05
    mixed_precision:  str   = "bf16"
    save_precision:   str   = "bf16"
    gradient_checkpointing: bool = True
    shuffle_caption:  bool  = True

    # Export
    output_name:      str   = "my_lora"
    output_dir:       str   = ""
    save_every_n_epochs: int = 1
    bucket_512:       bool  = False
    bucket_256:       bool  = False
    cache_latents:    bool  = True

    # Caption / trigger
    caption_extension:  str  = ".txt"
    trigger_word:       str  = ""
    keep_tokens:        int  = 1

    # AI-Toolkit specific
    aitk_project_name:   str       = ""
    aitk_sample_prompts: list[str] = field(default_factory=list)


# ── Results dataclass ─────────────────────────────────────────────────────────
@dataclass
class TrainingEstimate:
    images_total:     int   = 0
    steps_per_epoch:  int   = 0
    total_steps:      int   = 0
    warmup_steps:     int   = 0
    estimated_vram:   float = 0.0
    estimated_sec:    float = 0.0
    estimated_days:   int   = 0
    estimated_hours:  int   = 0
    estimated_mins:   int   = 0
    latent_cache_gb:  float = 0.0


# ── Core calculation (epoch-based, for legacy use) ────────────────────────────
def calculate(p: TrainingParams) -> TrainingEstimate:
    r = TrainingEstimate()
    r.images_total    = p.image_count * p.repeats
    if r.images_total == 0 or p.batch_size == 0:
        return r
    r.steps_per_epoch = math.ceil(r.images_total / p.batch_size)
    r.total_steps     = r.steps_per_epoch * p.epochs
    r.warmup_steps    = max(1, round(r.total_steps * p.warmup_ratio))
    base_vram         = _VRAM_BASE.get(p.optimizer, 12.0)
    dim_factor        = 1.0 + (p.network_dim / 128) * 0.5
    r.estimated_vram  = round(base_vram * dim_factor, 1)
    bs_speedup        = _BS_SCALE_FACTOR ** math.log2(max(1, p.batch_size))
    sec_per_step      = _SEC_PER_STEP_PER_BS * bs_speedup
    total_sec         = r.total_steps * sec_per_step
    r.estimated_sec   = total_sec
    r.estimated_days  = int(total_sec // 86400)
    r.estimated_hours = int((total_sec % 86400) // 3600)
    r.estimated_mins  = int((total_sec % 3600)  // 60)
    res_entry = next((rv for rv in RESOLUTIONS if rv[0] == p.resolution_label), None)
    if res_entry:
        w, h = res_entry[1], res_entry[2]
        latent_mb_per_image = (w // 8) * (h // 8) * 4 * 2 / (1024 * 1024)
        r.latent_cache_gb = round(p.image_count * latent_mb_per_image / 1024, 2)
    return r


# ── Bucket resolution list ────────────────────────────────────────────────────
def get_bucket_resolutions(bucket_512: bool, bucket_256: bool) -> list[int]:
    lower = []
    if bucket_256:
        lower.append(256)
    if bucket_512:
        lower.append(512)
    return lower + [1280]


# ── Simple YAML export (sd_trainer style) ────────────────────────────────────
def build_aitk_yaml(p: TrainingParams, est: TrainingEstimate) -> str:
    job_name = p.aitk_project_name or p.output_name or "my_lora"
    config = {
        "job": "extension",
        "config": {
            "name": job_name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": p.output_dir or "./output",
                "device": "cuda:0",
                "network": {"type": "lora", "linear": p.network_dim, "linear_alpha": p.network_alpha},
                "save": {"dtype": p.save_precision, "save_every": p.save_every_n_epochs, "max_step_saves_to_keep": 4},
                "datasets": [{
                    "folder_path": p.dataset_folder,
                    "caption_ext": p.caption_extension,
                    "caption_dropout_rate": 0.05,
                    "shuffle_tokens": p.shuffle_caption,
                    "cache_latents_to_disk": p.cache_latents,
                    "resolution": get_bucket_resolutions(p.bucket_512, p.bucket_256),
                }],
                "train": {
                    "batch_size": p.batch_size, "steps": est.total_steps,
                    "gradient_accumulation_steps": 1,
                    "train_unet": True, "train_text_encoder": False,
                    "gradient_checkpointing": p.gradient_checkpointing,
                    "noise_scheduler": "flowmatch",
                    "optimizer": p.optimizer.lower().replace("8bit", "_8bit"),
                    "lr": p.learning_rate,
                    "ema_config": {"use_ema": True, "ema_decay": 0.99},
                    "dtype": p.mixed_precision,
                },
                "model": {"name_or_path": p.model_name, "is_flux": False, "quantize": False},
                "sample": {
                    "sampler": "flowmatch", "sample_every": p.save_every_n_epochs,
                    "width": 1024, "height": 1024,
                    "prompts": p.aitk_sample_prompts or [f"{p.trigger_word} test sample" if p.trigger_word else "test sample"],
                    "neg": "", "seed": 42, "walk_seed": True,
                    "guidance_scale": 4.0, "sample_steps": 20,
                },
            }],
        },
        "meta": {"name": "[file prefix]", "version": "1.0"},
    }
    return yaml.dump(config, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ── Kohya TOML export ─────────────────────────────────────────────────────────
def build_kohya_toml(p: TrainingParams, est: TrainingEstimate) -> str:
    lines = [
        "[general]", "enable_bucket = true", "",
        "[[datasets]]",
        f"resolution = {max(next((rv[1] for rv in RESOLUTIONS if rv[0] == p.resolution_label), 1024), next((rv[2] for rv in RESOLUTIONS if rv[0] == p.resolution_label), 1024))}",
        f"batch_size = {p.batch_size}", "",
        "  [[datasets.subsets]]",
        f'  image_dir = "{p.dataset_folder}"',
        f'  caption_extension = "{p.caption_extension}"',
        f"  num_repeats = {p.repeats}",
        f"  keep_n_tokens = {p.keep_tokens}",
    ]
    return "\n".join(lines)


# ── V1 TOS-based math ─────────────────────────────────────────────────────────

def get_images(files: str, tagged: bool) -> int:
    try:
        n = int(files)
        return n // 2 if tagged else n
    except Exception:
        return 0


def lr_at(lt: str, imgs: int) -> float:
    if lt in ("character", "concept"):
        if imgs <= 50:    return 5e-5
        elif imgs <= 200: return 7e-5
        return 1e-4
    return 1e-4


def grad_at(lt: str, imgs: int, batch: int = 2) -> int:
    # Inverted: small datasets need more frequent updates (low accum), not larger batches
    if imgs <= 100:   return 1
    elif imgs <= 300: return 2
    elif imgs <= 700: return 3
    return 4


def grad_ko(lt: str, imgs: int, batch: int = 2) -> int:
    if imgs <= 100:   return 1
    elif imgs <= 300: return 2
    elif imgs <= 700: return 3
    return 4


def lin_rank(imgs: int, rc: str = "auto", lt: str = "character") -> int:
    presets = {"min": 16, "outfit": 20, "default": 32, "large": 64, "xlarge": 128, "max": 256}
    if rc.lower() in presets:
        return presets[rc.lower()]
    if lt in ("character", "concept"):
        return 32                          # flat — community sweet spot regardless of dataset size
    elif lt == "style":
        return 32 if imgs <= 200 else 64   # style benefits from higher rank at scale
    elif lt == "outfit":
        return 16                          # garment details captured efficiently at low rank
    else:                                  # pose
        return 16                          # geometric patterns need minimal rank


def conv_rank(lt: str, lr: int) -> int:
    r = {"character": 0.5, "concept": 0.5, "style": 0.5, "outfit": 0.4, "pose": 0.35}.get(lt, 0.5)
    return max(2, min(32, round(lr * r)))


def epochs_smooth(lt: str, imgs: int) -> int:
    a = {
        "character": (50, 5,  500,  20),
        "concept":   (40, 2,  1000, 20),
        "style":     (50, 8,  300,  20),
        "outfit":    (50, 4,  700,  20),
        "pose":      (50, 4,  600,  20),
    }.get(lt, (50, 5, 500, 20))
    mi, me, ma, xe = a
    ratio = max(0, min(1, (imgs - mi) / (ma - mi)))
    return min(xe, max(me, round(me + (xe - me) * (ratio ** 0.55))))


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def fmt_eta(hours: float) -> str:
    if not hours or hours <= 0:
        return "—"
    d = int(hours // 24)
    h = int(hours % 24)
    m = int((hours % 1) * 60)
    return f"{d}d {h}h {m}m" if d > 0 else f"{h}h {m}m"


def ets_check(actual: float, target: float) -> tuple[str, str]:
    if actual < target * 0.95:
        return "Underfit", "#E24B4A"
    elif actual > target * 1.05:
        return "Overfit", "#EF9F27"
    return "Good", "#639922"


def calc_at(lt: str, imgs: int, factor: float, rc: str,
            sph: Optional[float], batch: int = 2,
            tos: Optional[float] = None) -> dict:
    if imgs <= 0:
        return {}
    if tos is None:
        tos = TOS.get(lt, 100)
    g      = grad_at(lt, imgs, batch)
    eb     = batch * g
    steps  = round((imgs * tos / eb) * factor)
    lr_val = lr_at(lt, imgs)
    lrank  = lin_rank(imgs, rc, lt)
    crank  = conv_rank(lt, lrank)
    actual = round((steps * eb) / imgs, 1)
    ets, col = ets_check(actual, tos)
    ms  = min(12, max(2, math.ceil(steps / 1000)))
    se  = round(steps / ms) if ms else steps
    eta = fmt_eta(steps / sph) if sph and sph > 0 else None
    return {
        "steps": steps, "lr": f"{lr_val:.6f}",
        "weight_decay": WD.get(lt, 0.008), "caption_dropout": CD.get(lt, 0.12),
        "grad_accum": g, "eff_batch": eb,
        "linear_rank": lrank, "conv_rank": crank,
        "actual_tos": actual, "target_tos": tos,
        "ets": ets, "ets_color": col,
        "max_saves": ms, "save_every": se,
        "suggested_factor": round(tos / actual, 2) if actual else 1.0,
        "eta": eta,
        "img_warning": "⚠ Low image count" if imgs < MIN_IMAGES.get(lt, 50) else "",
    }


def calc_ko(lt: str, imgs: int, factor: float, rc: str,
            sph: Optional[float], batch: int = 2,
            tos: Optional[float] = None) -> dict:
    if imgs <= 0:
        return {}
    if tos is None:
        tos = TOS.get(lt, 100)
    g      = grad_ko(lt, imgs, batch)
    eb     = batch * g
    steps  = round((imgs * tos / eb) * factor)
    lr_val = lr_at(lt, imgs)
    te     = round(lr_val / 10, 6)
    lrank  = lin_rank(imgs, rc, lt)
    crank  = conv_rank(lt, lrank)
    ep     = epochs_smooth(lt, imgs)
    rep    = max(1, min(15, round(800 / imgs)))
    sch    = "linear" if lt == "pose" else "cosine_with_restarts"
    actual = round((steps * eb) / imgs, 1)
    ets, col = ets_check(actual, tos)
    se  = 1 if ep <= 10 else (2 if ep <= 20 else 5)
    ms  = min(12, math.ceil(ep / se))
    eta = fmt_eta(steps / sph) if sph and sph > 0 else None
    return {
        "steps": steps, "repeats": rep, "epochs": ep,
        "lr": f"{lr_val:.6f}", "te_lr": f"{te:.6f}",
        "scheduler": sch, "optimizer": "AdamW8bit",
        "weight_decay": WD.get(lt, 0.008), "caption_dropout": CD.get(lt, 0.12),
        "grad_accum": g, "eff_batch": eb,
        "linear_rank": lrank, "conv_rank": crank,
        "net_alpha": lrank, "noise_offset": 0.1, "min_snr": 5,
        "actual_tos": actual, "target_tos": tos,
        "ets": ets, "ets_color": col,
        "save_every": se, "max_saves": ms,
        "suggested_factor": round(tos / actual, 2) if actual else 1.0,
        "eta": eta,
        "img_warning": "⚠ Low image count" if imgs < MIN_IMAGES.get(lt, 50) else "",
    }


def calc_cv(lt: str, imgs: int, factor: float, rc: str,
            tos: Optional[float] = None) -> dict:
    if imgs <= 0:
        return {}
    if tos is None:
        tos = TOS.get(lt, 100)
    eb     = 4
    ep     = epochs_smooth(lt, imgs)
    rep    = max(1, round_half_up(((imgs * tos / 4) * 4) / (imgs * ep) * factor))
    steps  = round(rep * ep * imgs / 4)
    lr_val = {"character": 0.0003, "concept": 0.0005, "style": 0.0003,
              "outfit": 0.0004, "pose": 0.0004}.get(lt, 0.0003)
    te     = round(lr_val / 10, 6)
    lrank  = lin_rank(imgs, rc, lt)
    sch    = "linear" if lt == "pose" else "cosine_with_restarts"
    no     = {"character": 0.1, "concept": 0.08, "style": 0.06,
              "outfit": 0.09, "pose": 0.04}.get(lt, 0.1)
    actual = round((steps * eb) / imgs, 1)
    ets, col = ets_check(actual, tos)
    cw, cc = "Within caps", "#639922"
    if steps > 10000 and ep > 20:
        cw, cc = "Both Caps", "#E24B4A"
    elif steps > 10000:
        cw, cc = "Step Cap",  "#EF9F27"
    elif ep > 20:
        cw, cc = "Epoch Cap", "#EF9F27"
    mi = MIN_IMAGES.get(lt, 50)
    mx = CIVITAI_MAX.get(lt, 500)
    if imgs < mi:
        iw, ic = "⚠ Low Count",    "#E24B4A"
    elif imgs > mx:
        iw, ic = "⚠ Exceeds Cap",  "#EF9F27"
    else:
        iw, ic = "✓ Good Count",   "#639922"
    return {
        "repeats": rep, "epochs": ep, "steps": steps,
        "lr": f"{lr_val:.5f}", "te_lr": f"{te:.6f}",
        "scheduler": sch, "lr_cycles": 1 if sch == "linear" else 3,
        "min_snr": 5, "weight_decay": WD.get(lt, 0.008),
        "net_dim": lrank, "net_alpha": lrank,
        "noise_offset": no, "optimizer": "Adafactor",
        "actual_tos": actual, "target_tos": tos,
        "ets": ets, "ets_color": col,
        "suggested_factor": round(tos / actual, 2) if actual else 1.0,
        "cap_warning": cw, "cap_color": cc,
        "buzz": round(78.6 + (steps * 0.1156)),
        "img_warning": iw, "img_warning_color": ic,
        "civitai_type": "Concept" if lt in ["pose", "outfit"] else lt.capitalize(),
    }


def count_dataset_files(folder: str) -> int:
    try:
        return sum(
            1
            for root, _, files in os.walk(folder)
            for f in files
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
    except Exception:
        return 0


# ── Training log I/O ──────────────────────────────────────────────────────────

def load_log() -> dict:
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"aitoolkit": list(PRELOADED_AT[-MAX_LOG:]), "kohya": []}


def save_log(log: dict) -> None:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def init_log() -> None:
    if not os.path.exists(LOG_FILE):
        save_log({"aitoolkit": list(PRELOADED_AT[-MAX_LOG:]), "kohya": []})


def add_log_entry(fw: str, sph: float, sit: float) -> None:
    log = load_log()
    log[fw].append({
        "date":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        "steps_per_hour": sph,
        "sec_per_iter":   sit,
    })
    if len(log[fw]) > MAX_LOG:
        log[fw] = log[fw][-MAX_LOG:]
    save_log(log)


def delete_log_entry(fw: str, idx: int) -> None:
    log = load_log()
    if 0 <= idx < len(log[fw]):
        log[fw].pop(idx)
    save_log(log)


def get_averages(fw: str) -> tuple[Optional[float], Optional[float]]:
    entries = load_log().get(fw, [])
    if not entries:
        return None, None
    return (
        sum(e["steps_per_hour"] for e in entries) / len(entries),
        sum(e["sec_per_iter"]   for e in entries) / len(entries),
    )


# ── Clean YAML builder for AI-Toolkit (v1-compatible) ────────────────────────

def build_at_yaml(config: dict) -> str:
    """Produce clean YAML without !!type tags, matching ai-toolkit format."""
    class CleanDumper(yaml.Dumper):
        pass

    CleanDumper.add_representer(
        bool,      lambda d, v: d.represent_scalar("tag:yaml.org,2002:bool",  "true" if v else "false"))
    CleanDumper.add_representer(
        type(None), lambda d, v: d.represent_scalar("tag:yaml.org,2002:null",  "null"))
    CleanDumper.add_representer(
        float,     lambda d, v: d.represent_scalar("tag:yaml.org,2002:float", str(v)))
    CleanDumper.add_representer(
        int,       lambda d, v: d.represent_scalar("tag:yaml.org,2002:int",   str(v)))

    def str_representer(d, v):
        if "\\" in v:
            return d.represent_scalar("tag:yaml.org,2002:str", v, style='"')
        return d.represent_scalar("tag:yaml.org,2002:str", v)

    CleanDumper.add_representer(str, str_representer)
    CleanDumper.ignore_aliases = lambda *a: True

    result = yaml.dump(
        config, Dumper=CleanDumper, allow_unicode=True,
        sort_keys=False, default_flow_style=False, width=10000)
    return "---\n" + result
