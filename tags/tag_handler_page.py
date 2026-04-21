"""
tags/tag_handler_page.py — Tag Handler embedded tab for Lora Training Suite 2.0
Designed by: Dagoberto (Zero) Gracey  |  Built by: Jarvis (v2.0)

Full PySide6 port of ultimate_tag_handler.py (v1.6 CustomTkinter → PySide6).
Translated as close to the original as possible.
"""

import os, sys, json, math, threading, shutil, zipfile, csv, re, subprocess
import importlib, random, datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from PySide6.QtCore    import (Qt, QTimer, QThread, QObject, Signal,
                                QRect, QPoint, QSize, QEvent)
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox,
    QSlider, QProgressBar, QScrollArea, QTabWidget, QSplitter,
    QFileDialog, QSizePolicy, QMessageBox, QInputDialog,
    QPlainTextEdit, QDialog, QProgressDialog, QApplication,
    QMenu, QSpacerItem, QSpinBox,
)
from PySide6.QtGui import QPixmap, QImage, QFont, QCursor, QKeySequence, QShortcut, QDrag
from PySide6.QtCore import QMimeData

from shared.theme import (
    BG, PAN, CAR, ACC, GRN, RED, MUT, PRI, SEC, AMB,
    FONT, FONT_SM, FONT_MD, FONT_LG, FONT_XL, VERSION, SIGNATURE,
)
from shared.config import load_json, save_json

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
CFG_FILE   = os.path.join(_HERE, "uth_config.json")
PROF_FILE  = os.path.join(_HERE, "uth_profiles.json")
TAGS_MASTER = os.path.join(_HERE, "tags_data.json")

# ── Layout ────────────────────────────────────────────────────────────────────
DEFAULT_CARD   = 240
MIN_CARD       = 180
MAX_CARD       = 520
COLS           = 6
CARDS_PER_PAGE = 18
PREP_BATCH     = 30   # cards loaded per scroll-triggered batch in Prepare tab
EDITOR_BATCH   = 15   # cards loaded per scroll-triggered batch in Editor tab
THUMB_SIZE     = 340
MAX_PILL_POOL  = 64

# ── Data constants ─────────────────────────────────────────────────────────────
RESOLUTIONS = [
    ("1024 × 1024", 1024, 1024, "1:1"),
    ("1152 × 896",  1152,  896, "4:3"),
    ("1216 × 832",  1216,  832, "3:2"),
    ("1344 × 768",  1344,  768, "16:9"),
    ("1536 × 640",  1536,  640, "12:5"),
    ("896 × 1152",   896, 1152, "3:4"),
    ("768 × 1344",   768, 1344, "9:16"),
    ("768 × 768",    768,  768, "SD HD"),
    ("512 × 512",    512,  512, "SD1.5"),
]
RES_LABELS    = [r[0] for r in RESOLUTIONS]
RES_MAP       = {r[0]: (r[1], r[2], r[3]) for r in RESOLUTIONS}
DEFAULT_RES   = "1024 × 1024"
RATIO_LABELS  = [r[3] for r in RESOLUTIONS]
RATIO_MAP     = {r[3]: (r[1], r[2]) for r in RESOLUTIONS}
RATIO_BY_SIZE = {(r[1], r[2]): r[3] for r in RESOLUTIONS}
LORA_TYPES    = ["Character", "Concept", "Style", "Outfit", "Pose"]
NEG_KEYS      = {"Character":"character_neg","Concept":"concept_neg",
                 "Style":"style_neg","Outfit":"outfit_neg","Pose":"pose_neg"}
IMAGE_EXTS    = {'.jpg','.jpeg','.png','.webp','.bmp','.tiff','.tif','.gif','.avif'}
WD_MODELS     = ["WD14 ConvNextV2","WD EVA02-Large"]
OUTPUT_MODES  = ["Tags","Captions","Captions (Fast)","Hybrid","Hybrid (GPU)","Hybrid (Fast)"]
CAP_TYPES     = ["Descriptive","Training Prompt","MidJourney","Booru-like Tags","Formal","Casual"]
CAP_LENGTHS   = ["Short","Medium","Long","Any"]
JOY_MODEL_ID  = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
MOONDREAM_MODEL_ID  = "vikhyatk/moondream2"
MOONDREAM_REVISION  = "2024-08-26"
def _check_joycaption_model(path: str):
    """Verify every safetensors shard is present and non-truncated.

    Returns (True, "") on success, or (False, human-readable error) on any
    problem.  Uses only the file header + last tensor read per shard so it
    is fast (no full data scan) but still catches:
      - missing files
      - zero / impossibly-small files
      - header/data mismatch (re-written header over a shorter payload)
      - truncated downloads (last tensor offset beyond EOF)
    """
    import os as _os, json as _json
    required = ["config.json", "tokenizer.json",
                "tokenizer_config.json", "model.safetensors.index.json"]
    for _f in required:
        if not _os.path.isfile(_os.path.join(path, _f)):
            return False, f"Model file missing: {_f}"

    _idx = _os.path.join(path, "model.safetensors.index.json")
    try:
        with open(_idx) as _fh:
            _wmap = _json.load(_fh)["weight_map"]
    except Exception as _e:
        return False, f"Cannot read model index: {_e}"

    # Build expected key-count per shard
    _shard_keys: dict = {}
    for _k, _s in _wmap.items():
        _shard_keys.setdefault(_s, []).append(_k)

    try:
        from safetensors import safe_open as _sf_open
    except ImportError:
        return False, "safetensors package not installed"

    for _shard, _exp_keys in sorted(_shard_keys.items()):
        _sp = _os.path.join(path, _shard)
        if not _os.path.isfile(_sp):
            return False, f"Missing shard: {_shard}"
        _sz = _os.path.getsize(_sp)
        if _sz < 4096:
            return False, f"Shard suspiciously small ({_sz} bytes): {_shard}"
        try:
            with _sf_open(_sp, framework="pt", device="cpu") as _sf:
                _actual = list(_sf.keys())
                if len(_actual) != len(_exp_keys):
                    return False, (
                        f"Shard {_shard}: index expects {len(_exp_keys)} tensors "
                        f"but file header has {len(_actual)}")
                # Read the last tensor — catches truncated payloads where the
                # header is intact but the data region is cut short.
                _sf.get_tensor(_actual[-1])
        except Exception as _e:
            return False, f"Shard {_shard} is corrupt or truncated: {_e}"

    return True, ""


def _joy_prompt(cap_type: str, length: str) -> str:
    """Generate the instruction prompt for JoyCaption given type and length."""
    l = {"Any": "", "Short": "short ", "Medium": "medium-length ",
         "Long": "long "}.get(length, f"{length} ")
    _P = {
        "Descriptive":           f"Write a {l}descriptive caption for this image in a formal tone.",
        "Descriptive (Informal)": f"Write a {l}descriptive caption for this image in a casual, informal tone.",
        "Training Prompt":       f"Write a {l}stable diffusion training prompt for this image.",
        "MidJourney":            f"Write a {l}MidJourney prompt for this image.",
        "Booru tag list":        f"Write a {l}list of Booru-style tags for this image.",
        "Art Critic":            f"Analyze this image like an art critic and write a {l}critique.",
        "Product Listing":       f"Write a {l}product listing description for this image.",
        "Social Media Post":     f"Write a {l}social media caption for this image.",
    }
    return _P.get(cap_type, f"Write a {l}caption for this image.")
DEVICES      = ["Auto-detect","Force CUDA","Force CPU"]
EXIST_MODES  = ["Skip","Append","Overwrite"]
PROC_MODES   = ["Enabled","Disabled"]
CONV_FMTS    = ["PNG","JPEG"]

WD14_REPOS = {
    "WD14 ConvNextV2": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "WD EVA02-Large":  "SmilingWolf/wd-eva02-large-tagger-v3",
}
# timm-based models: pure PyTorch inference, GPU-accelerated via ROCm/CUDA
WD14_TIMM_REPOS = {
    "WD EVA02-Large (GPU)": "SmilingWolf/wd-eva02-large-tagger-v3",
    "WD SwinV2 v3 (GPU)":   "SmilingWolf/wd-swinv2-tagger-v3",
}
WD14_IMG_SIZE   = 448
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
WD14_RATING_MAP = {
    "rating:general":      "general",
    "rating:sensitive":    "sensitive",
    "rating:questionable": "questionable",
    "rating:explicit":     "explicit",
}

DEFAULT_PROFILE = {
    "trigger_word":"","desired_tags":"","negative_tags":"",
    "output_mode":"Tags","threshold":0.35,"char_threshold":0.85,
    "ratings":{"general":True,"sensitive":True,"questionable":True,"explicit":True},
    "caption_type":"Descriptive","caption_length":"Medium",
    "extra_instructions":"","device":"Auto-detect","batch_size":4,
}

HELP = {
    "Prepare":[
        ("Dataset Folder","The folder containing your training images. All supported formats are detected automatically (.jpg .jpeg .png .webp .bmp .tiff). Browse to select."),
        ("Format Conversion","Converts all images to PNG or JPEG. PNG is lossless and recommended for training. JPEG produces smaller files with quality loss. If multiple images share the same base name but differ in extension (e.g. Photo.jpg, Photo.webp, Photo.bmp), they are renamed Photo.png, Photo-1.png, Photo-2.png — never treated as duplicates. The paired .txt tag file is renamed to match the new image name. Always convert before adjusting images in the grid."),
        ("Resolution","Target resolution for the final output. SDXL trains natively at 1024×1024. Non-square buckets preserve subject framing for portrait/landscape images."),
        ("SDXL Bucketing","SDXL was trained using multi-aspect ratio bucketing. Native buckets: 1024×1024 (1:1), 1152×896 (4:3), 1216×832 (3:2), 1344×768 (16:9), 1536×640 (12:5), 896×1152 (3:4), 768×1344 (9:16). Cropping to the natural bucket preserves framing and uses the full resolution budget."),
        ("Manual Resolution","Selecting 'Manual' from the Resolution dropdown unlocks per-image aspect ratio assignment. Each card shows a ratio dropdown. Ratios and pan/zoom positions are saved per-image as you page through."),
        ("Enhanced Scaling","Enabled: automatically upscales small images using Real-ESRGAN AI and downscales large images using cv2.INTER_AREA. Disabled: crop only, no scaling applied."),
        ("Image Cards","Each card is a fixed square — the card border is the crop frame. Whatever is visible inside the border is exactly what gets written to disk when you press Process."),
        ("Card Border Colors","BLUE: image is already square. RED: image is non-square — use drag and sliders to frame your subject. AMBER: filename collision duplicate."),
        ("Drag to Pan","Click and drag anywhere inside the card to reposition the image within the crop square."),
        ("X / Y Sliders","Fine-tune after dragging. X slider nudges left/right. Y slider nudges up/down. All positions are saved per-image as you page through."),
        ("Process / Apply","Permanently resizes and crops all images. This is the point of no return. The only way to recover is to restore from your Safeguard ZIP."),
        ("Safeguard ZIP","Creates a timestamped ZIP of your entire dataset saved next to the dataset folder. Make one before running Process, before tagging, and before any large batch edit."),
    ],
    "Tagger":[
        ("Trigger Tag","Always placed FIRST in every .txt file. The base SDXL model learns to associate this word with everything in your dataset. Convention: prefix with 'z0' — e.g. 'z0charname'."),
        ("Desired Tags","Tags merged into EVERY image's .txt file regardless of tagger output."),
        ("Negative Tags","Tags stripped from ALL tagger output. Your blacklist."),
        ("WD14 Ensemble","Runs both WD14 ConvNextV2 and WD EVA02-Large and unions their tag sets. Recommended default for all datasets."),
        ("JoyCaption Beta One","Large vision-language model (LLaVA, SiGLIP2 + LLaMA 3.1 8B) generating natural language captions. Requires 8GB+ VRAM. ~18GB download on first use. Loaded directly to VRAM — does not require 18GB of system RAM."),
        ("Caption Type","Descriptive (formal/informal), Training Prompt, MidJourney, Booru tag list, Art Critic, Product Listing, Social Media Post."),
        ("Caption Length","Any, Short, Medium, Long. Controls the verbosity of the generated caption."),
        ("Extra Instructions","Optional text appended to the caption prompt. Use to steer the output — e.g. 'Focus on the outfit and accessories.' or 'Do not mention the background.'"),
        ("Confidence Threshold","Minimum confidence (0.0–1.0) for a WD14 tag to be included. Default 0.35."),
        ("Output Mode","Tags: WD14 only. Captions: JoyCaption only. Hybrid: WD14 tags + JoyCaption caption in one file. Captions (Fast) / Hybrid (Fast): Moondream2 instead of JoyCaption — faster, lower VRAM."),
        ("Existing Files","Skip: leave existing .txt untouched. Append: add new tags. Overwrite: replace content."),
        ("Device","Auto-detect: uses CUDA if available (NVIDIA or AMD ROCm), CPU fallback. Force CUDA: always use GPU. Force CPU: always use CPU."),
        ("Batch Size","Images processed simultaneously. Higher = faster but more VRAM. Default 4."),
    ],
    "Tag Editor":[
        ("Tag Viewer","Collapsible panel showing all unique tags across the dataset with frequency counts. Click any tag to filter gallery."),
        ("Tag Frequency Cloud","All tags as pills sorted by frequency. Number = images containing that tag."),
        ("Tag Selection","Click any cloud pill to select it (turns blue with ✓). Selected tags filter the gallery."),
        ("Don't Sort","When checked, removing a tag pill updates only that card — the cloud and gallery are frozen. Use during rapid cleanup."),
        ("Actions Menu","Add tags to all, Add tag to selection, Remove tags, Replace tags."),
        ("Tag Pills","Each tag shown as a rounded pill. Click a pill to remove that tag from that image."),
        ("Add Tags","Text field at bottom of each card. Type and press Enter or click + to add."),
        ("Undo","Ctrl+Z reverses the last tag operation. Up to 20 undo steps."),
        ("Hover Actions","Hover over a card to reveal: eye-slash (disable image) and trash (delete image)."),
    ],
    "Randomizer":[
        ("Overview","Background remover for LoRA datasets. Removes the background from a subject image and composites it over a randomly selected background. Prevents the model from learning a repeated background as part of the subject."),
        ("Mode","Realism — uses BRIA RMBG-2.0 (BiRefNet), best for photos and realistic subjects. Anime — uses ToonOut (BiRefNet fine-tuned), best for illustrations and anime art."),
        ("Download Models","Models are downloaded on first use (~845 MB each). A HuggingFace account with license acceptance is required for Realism (RMBG-2.0). Set your HF token under Manage before downloading."),
        ("Browse Image","Load a single subject image. Supports PNG, JPG, WEBP, BMP, TIFF."),
        ("Remove BG","Runs the selected model. A progress dialog shows each step: loading model, loading weights, running inference, compositing. Result is shown in the viewer."),
        ("Background Panel","Right column — 100×100 thumbnails loaded from the 'random images' folder inside the Randomizer module. Click a thumbnail to composite the subject over that background. Click again to deselect (transparent output)."),
        ("Random Images Folder","Place your background images in: Lora Training Suite 2.0\\randomizer\\random images\\. Recommended resolution: 2000×2000 px or larger. Use the ↺ refresh button to reload after adding files."),
        ("Viewer","2000×2000 canvas with Zoom, X, and Y sliders. Drag to pan. Checkerboard pattern indicates transparent areas."),
        ("Box Size","Resize the viewer canvas (400–3000 px, 100 px steps)."),
        ("View Original / Output","Toggle button switches between the original loaded image and the background-removed result."),
        ("Save PNG","Saves the current result. If a background is selected, saves the composited RGB image. If no background is selected, saves transparent RGBA PNG."),
    ],
    "Batch":[
        ("Find & Create","Scans the dataset for images with no paired .txt file and creates empty ones. Run this before Batch Add if your dataset has no tag files yet."),
        ("Batch Add Tags","Adds the tags you type (comma-separated) to every .txt file in the dataset. If a .txt file does not exist it is created automatically. Duplicate tags are skipped. Minimum workflow: set your trigger word, type one additional tag, click Add to All — every image gets a new .txt with both tags."),
        ("Shuffle Tags","Randomizes the tag order in every .txt file. The trigger word is always pinned to position 1. Random tag order trains better than alphabetical — it prevents the model from over-weighting tags that always appear together in the same position."),
        ("Freq Report","Builds a comma-separated list of every tag ordered by frequency. Saved as tags_report.txt."),
        ("Save Final ZIP","Creates the finished dataset ZIP: all images, .txt tag files, and tags_report.txt. Deletes the thumbnail cache and safeguard ZIP."),
        ("Results Font Size","The A — slider — A control adjusts the font size of the log output (8–20pt)."),
    ],
    "Model Credits":[
        ("WD14 ConvNextV2 Tagger v3","By SmilingWolf. License: Apache 2.0 — free for commercial and personal use. Attribution: include license notice with any distribution. Source: https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3"),
        ("WD EVA02-Large Tagger v3","By SmilingWolf. License: Apache 2.0 — free for commercial and personal use. Attribution: include license notice with any distribution. Source: https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3"),
        ("JoyCaption Beta One","By fancyfeast / fpgaminer (John David Pressman). License: Apache 2.0 + Meta Llama 3.1 Community License. Commercial use permitted. REQUIRED: Any product or documentation must display 'Built with Llama' and include the notice: 'Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.' Architecture: LLaVA with SiGLIP2 so400m vision encoder + LLaMA 3.1 8B Instruct. Source: https://github.com/fpgaminer/joycaption  |  https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava"),
        ("Moondream2","By Vikhyat Koparekar et al. License: Apache 2.0 — free for commercial and personal use. Compact 1.86B parameter vision-language model. Used for Captions (Fast) and Hybrid (Fast) output modes. Source: https://huggingface.co/vikhyatk/moondream2"),
        ("BRIA RMBG-2.0  ⚠ NON-COMMERCIAL","By BRIA AI. License: CC BY-NC 4.0 — NON-COMMERCIAL USE ONLY. Personal, academic, and non-profit use is permitted. Commercial use requires a separate paid license from BRIA AI. If you distribute or monetize this tool you must obtain a commercial license or replace this model. Contact: https://bria.ai  |  Source: https://huggingface.co/briaai/RMBG-2.0"),
        ("BiRefNet","By ZhengPeng7 et al. License: MIT — free for commercial and personal use. Architecture used by both RMBG-2.0 and ToonOut. Paper: 'Bilateral Reference for High-Resolution Dichotomous Image Segmentation', CAAI AI Research, 2024. Source: https://github.com/ZhengPeng7/BiRefNet  |  https://huggingface.co/ZhengPeng7/BiRefNet"),
        ("ToonOut","By Joël Seytre and Matteo Muratori. License: MIT — free for commercial and personal use. BiRefNet fine-tuned for anime and illustration background removal. Paper: 'ToonOut: Fine-tuned Background Removal for Anime Characters', arXiv:2509.06839, 2025. Source: https://huggingface.co/joelseytre/toonout"),
        ("Real-ESRGAN","By Xintao Wang et al. License: BSD 3-Clause — free for commercial and personal use. Authors' names may not be used to endorse derivative products without written permission. Paper: 'Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data', ICCVW 2021. Source: https://github.com/xinntao/Real-ESRGAN"),
    ],
}

# ── Config I/O ─────────────────────────────────────────────────────────────────
def _load_cfg():
    try:
        if os.path.exists(CFG_FILE):
            with open(CFG_FILE,'r') as f: return json.load(f)
    except: pass
    return {"last_dataset":"","resolution":DEFAULT_RES,"convert_format":"PNG",
            "jpeg_quality":95,"process_mode":"Enabled","scaling_mode":"Enabled",
            "trigger_word":""}

def _save_cfg(cfg):
    try:
        with open(CFG_FILE,'w') as f: json.dump(cfg,f,indent=2)
    except: pass

def _load_profiles():
    tags_master = _load_tags_master()
    base = {}
    for t in LORA_TYPES:
        p = dict(DEFAULT_PROFILE)
        neg_key = NEG_KEYS.get(t, "")
        if neg_key and neg_key in tags_master:
            p["negative_tags"] = tags_master[neg_key]
        base[t] = p
    try:
        if os.path.exists(PROF_FILE):
            with open(PROF_FILE, 'r') as f:
                saved = json.load(f)
            base.update(saved)
    except:
        pass
    return base

def _save_profiles(p):
    try:
        with open(PROF_FILE,'w') as f: json.dump(p,f,indent=2)
    except: pass

def _load_tags_master():
    try:
        if os.path.exists(TAGS_MASTER):
            with open(TAGS_MASTER,'r',encoding='utf-8') as f: return json.load(f)
    except: pass
    return {}

# ── File helpers ───────────────────────────────────────────────────────────────
def _get_images(folder):
    try:
        return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
    except: return []

def _txt_path(img_path):
    return os.path.splitext(img_path)[0]+'.txt'

# ── In-memory tag cache ────────────────────────────────────────────────────────
_tag_cache: dict = None
_tag_dirty: set  = None
_pil_prefetch: dict      = {}
_pil_prefetch_lock       = threading.Lock()

def _line_is_tags(line: str) -> bool:
    """True when a line looks like comma-separated tags (short tokens)."""
    if ',' not in line:
        return False
    frags = [f.strip() for f in line.split(',') if f.strip()]
    if not frags:
        return False
    avg_words = sum(len(f.split()) for f in frags) / len(frags)
    return avg_words <= 3.0


def _read_tags(img_path):
    if _tag_cache is not None and img_path in _tag_cache:
        return list(_tag_cache[img_path])
    tp = _txt_path(img_path)
    try:
        if os.path.exists(tp):
            with open(tp, 'r', encoding='utf-8') as f:
                raw = f.read()
            lines = [l.strip() for l in raw.split('\n') if l.strip()]
            if not lines:
                return []
            items = []
            # Line 0: comma-split only if it looks like tags (short tokens)
            # If avg fragment is >3 words it is a caption sentence — keep whole
            if _line_is_tags(lines[0]):
                items.extend([t.strip() for t in lines[0].split(',') if t.strip()])
            else:
                items.append(lines[0])
            # Line 1+: always captions, never comma-split
            for line in lines[1:]:
                items.append(line)
            return items
    except: pass
    return []

def _write_tags(img_path, items):
    if _tag_cache is not None:
        _tag_cache[img_path] = list(items)
        if _tag_dirty is not None:
            _tag_dirty.add(img_path)
        return
    tags, caps = _split_tags_captions(items)
    content = ', '.join(tags)
    if caps:
        content = (content + '\n' if content else '') + caps[0]
    tp = _txt_path(img_path)
    try:
        with open(tp, 'w', encoding='utf-8') as f:
            f.write(content)
    except: pass

def _init_tag_cache(paths):
    global _tag_cache, _tag_dirty
    _tag_cache = {}
    _tag_dirty = set()
    for p in paths:
        items = []
        tp = _txt_path(p)
        try:
            if os.path.exists(tp):
                with open(tp, 'r', encoding='utf-8') as f:
                    raw = f.read()
                lines = [l.strip() for l in raw.split('\n') if l.strip()]
                if lines:
                    if _line_is_tags(lines[0]):
                        items.extend([t.strip() for t in lines[0].split(',') if t.strip()])
                    else:
                        items.append(lines[0])
                    for line in lines[1:]:
                        items.append(line)
        except:
            pass
        _tag_cache[p] = items

def _flush_tag_cache():
    global _tag_dirty
    if _tag_cache is None or _tag_dirty is None: return 0
    count = 0
    for img_path in list(_tag_dirty):
        tp = _txt_path(img_path)
        try:
            items = _tag_cache.get(img_path, [])
            tags, caps = _split_tags_captions(items)
            content = ', '.join(tags)
            if caps:
                content = (content + '\n' if content else '') + caps[0]
            with open(tp, 'w', encoding='utf-8') as f:
                f.write(content)
            count += 1
        except: pass
    _tag_dirty.clear()
    return count

# ── Background image loader ────────────────────────────────────────────────────
# Shared thread pool used by PrepareCard to load full-resolution images off the
# main thread.  4 workers is enough to stay ahead of the scroll speed without
# saturating the disk.
_IMAGE_LOAD_EXECUTOR: ThreadPoolExecutor | None = None


def _get_image_executor() -> ThreadPoolExecutor:
    global _IMAGE_LOAD_EXECUTOR
    if _IMAGE_LOAD_EXECUTOR is None:
        _IMAGE_LOAD_EXECUTOR = ThreadPoolExecutor(max_workers=4,
                                                  thread_name_prefix="img_load")
    return _IMAGE_LOAD_EXECUTOR


def shutdown_image_loader() -> None:
    """Call on app close to cleanly stop pending image-load tasks."""
    global _IMAGE_LOAD_EXECUTOR
    if _IMAGE_LOAD_EXECUTOR is not None:
        _IMAGE_LOAD_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _IMAGE_LOAD_EXECUTOR = None


# ── Thumbnail cache ────────────────────────────────────────────────────────────
_TEMP_DIR = os.path.join(os.path.dirname(_HERE), "temp")

def _thumb_cache_dir(img_path: str) -> str:
    """Return temp/<dataset_name>/ for the given image — never inside the dataset."""
    dataset_name = os.path.basename(os.path.dirname(img_path))
    return os.path.join(_TEMP_DIR, dataset_name)

def _load_or_cache_thumb(img_path):
    if not img_path or not os.path.exists(img_path):
        return Image.new("RGB",(THUMB_SIZE,THUMB_SIZE),CAR[1:] if CAR.startswith('#') else "1a1a1a")
    with _pil_prefetch_lock:
        pre = _pil_prefetch.pop(img_path,None)
    if pre is not None:
        return pre
    cache_dir = _thumb_cache_dir(img_path)
    stem      = os.path.splitext(os.path.basename(img_path))[0]
    try:    mtime = int(os.path.getmtime(img_path))
    except: mtime = 0
    cache_file = os.path.join(cache_dir,f"{stem}_{mtime}.webp")
    if os.path.exists(cache_file):
        try: return Image.open(cache_file).convert("RGB")
        except: pass
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((THUMB_SIZE,THUMB_SIZE),Image.LANCZOS)
        w,h  = img.size; side = min(w,h)
        img  = img.crop(((w-side)//2,(h-side)//2,(w+side)//2,(h+side)//2))
        img  = img.resize((THUMB_SIZE,THUMB_SIZE),Image.LANCZOS)
    except:
        img = Image.new("RGB",(THUMB_SIZE,THUMB_SIZE),(26,26,46))
    try:
        os.makedirs(cache_dir,exist_ok=True)
        img.save(cache_file,"WEBP",quality=85)
        for f in os.listdir(cache_dir):
            if f.startswith(stem+"_") and f.endswith(".webp") and f!=os.path.basename(cache_file):
                try: os.remove(os.path.join(cache_dir,f))
                except: pass
    except: pass
    return img

def _is_caption(text):
    t = text.strip()
    words = t.split()
    if not words: return False
    if any(c in t for c in '.!?') and len(words)>=3: return True
    return len(words)>=5

def _split_tags_captions(items):
    tags     = [x for x in items if not _is_caption(x)]
    captions = [x for x in items if _is_caption(x)]
    return tags,captions

def _image_status(img_path):
    try:
        with Image.open(img_path) as im:
            w,h = im.size
            return "normal" if w==h else "nonsquare"
    except: return "normal"

def _pil_to_qpixmap(pil_img):
    data = np.asarray(pil_img.convert("RGB"))
    h,w,ch = data.shape
    qi = QImage(data.tobytes(),w,h,ch*w,QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)

def _lazy_install(pip_name, import_name=None, status_cb=None):
    import_name = import_name or pip_name
    try: importlib.import_module(import_name); return True
    except ImportError: pass
    if status_cb: status_cb(f"Installing {pip_name}…")
    try:
        subprocess.check_call(
            [sys.executable,"-m","pip","install","--quiet",
             "--disable-pip-version-check",pip_name],
            creationflags=0x08000000)
        importlib.invalidate_caches()
        importlib.import_module(import_name)
        return True
    except Exception as ex:
        print(f"[UTH] Could not install {pip_name}: {ex}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  FLOW LAYOUT  (standard Qt flow-wrap layout)
# ══════════════════════════════════════════════════════════════════════════════
class FlowLayout(QLayout):
    def __init__(self, parent=None, h_spacing=4, v_spacing=4):
        super().__init__(parent)
        self._items = []
        self._h = h_spacing
        self._v = v_spacing

    def addItem(self, item): self._items.append(item)
    def count(self):         return len(self._items)
    def itemAt(self,i):      return self._items[i] if 0<=i<len(self._items) else None
    def takeAt(self,i):      return self._items.pop(i) if 0<=i<len(self._items) else None
    def hasHeightForWidth(self): return True
    def heightForWidth(self,w):  return self._layout(QRect(0,0,w,0),True)
    def setGeometry(self,r):     super().setGeometry(r); self._layout(r,False)
    def sizeHint(self):          return self.minimumSize()
    def minimumSize(self):
        sz = QSize()
        for it in self._items: sz = sz.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return sz + QSize(m.left()+m.right(), m.top()+m.bottom())

    def _layout(self,rect,test):
        m = self.contentsMargins()
        x = rect.x()+m.left(); y = rect.y()+m.top()
        lh = 0
        for it in self._items:
            w = it.widget()
            if w and not w.isVisible(): continue
            sw = it.sizeHint()
            nx = x + sw.width() + self._h
            if nx - self._h > rect.right()-m.right() and lh>0:
                x = rect.x()+m.left()
                y += lh + self._v
                nx = x + sw.width() + self._h
                lh = 0
            if not test:
                it.setGeometry(QRect(QPoint(x,y),sw))
            x = nx
            lh = max(lh, sw.height())
        return y + lh - rect.y()


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTED BUTTON  (replaces CTkSegmentedButton)
# ══════════════════════════════════════════════════════════════════════════════
class SegmentedButton(QWidget):
    valueChanged = Signal(str)

    def __init__(self, parent=None, values=None):
        super().__init__(parent)
        values = values or []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.setSpacing(2)
        self._buttons = {}
        self._current = values[0] if values else ""
        for v in values:
            btn = QPushButton(v, self)
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.setStyleSheet(self._style(False))
            btn.clicked.connect(lambda _,val=v: self._select(val))
            lay.addWidget(btn)
            self._buttons[v] = btn
        if values: self._select(values[0], emit=False)

    def _style(self, active):
        bg = ACC if active else CAR
        return (f"QPushButton{{background:{bg};color:{PRI};border:none;"
                f"border-radius:4px;padding:4px 14px;"
                f"font-family:{FONT};font-size:{FONT_MD}px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{'#185FA5' if active else MUT};}}")

    def _select(self, val, emit=True):
        self._current = val
        for v,btn in self._buttons.items():
            btn.setChecked(v==val)
            btn.setStyleSheet(self._style(v==val))
        if emit: self.valueChanged.emit(val)

    def get(self): return self._current
    def set(self, val):
        if val in self._buttons: self._select(val, emit=False)


# ══════════════════════════════════════════════════════════════════════════════
#  PREPARE CARD
# ══════════════════════════════════════════════════════════════════════════════
class PrepareCard(QFrame):
    D = DEFAULT_CARD
    _image_ready = Signal(object)   # PIL.Image — emitted from bg thread

    def __init__(self, parent, img_path, status="normal",
                 saved_offset=None, saved_zoom=1.0, tw=None, th=None,
                 on_deleted=None):
        super().__init__(parent)
        bc = {"normal": ACC, "nonsquare": RED, "duplicate": AMB}.get(status, ACC)
        self.setStyleSheet(
            f"PrepareCard{{background:{CAR};border:3px solid {bc};"
            f"border-radius:6px;}}")
        self.setObjectName("PrepareCard")
        self.setFixedWidth(self.D + 8)   # 4px margin each side
        self.img_path  = img_path
        self.status    = status
        self.on_deleted = on_deleted
        self.offset_x  = saved_offset[0] if saved_offset else 0.5
        self.offset_y  = saved_offset[1] if saved_offset else 0.5
        self.card_zoom = float(saved_zoom) if saved_zoom else 1.0
        self._tw       = tw if tw else self.D
        self._th       = th if th else self.D
        self._dsx = self._dsy = 0
        self._dox = self._doy = 0.5
        self._pixmap   = None

        # Placeholder until the background load completes
        self._pil = Image.new("RGB", (self.D, self.D), (22, 33, 62))

        root = QVBoxLayout(self)
        root.setContentsMargins(4,4,4,4)
        root.setSpacing(2)

        # Header row
        hdr = QWidget(self)
        hdr.setStyleSheet("background:transparent;")
        hlay = QHBoxLayout(hdr)
        hlay.setContentsMargins(2,0,2,0); hlay.setSpacing(4)

        del_btn = QPushButton("DEL", hdr)
        del_btn.setFixedSize(26, 14)
        del_btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{MUT};border:none;"
            f"font-family:{FONT};font-size:8px;font-weight:bold;padding:0;}}"
            f"QPushButton:hover{{color:{RED};}}")
        del_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        del_btn.setToolTip("Delete image from dataset")
        del_btn.clicked.connect(self._delete_image)
        hlay.addWidget(del_btn)

        fname = os.path.basename(img_path)
        fname_s = fname if len(fname)<=30 else fname[:27]+"…"
        nm = QLabel(fname_s, hdr)
        nm.setStyleSheet(f"color:{MUT};font-family:{FONT};font-size:9px;background:transparent;")
        hlay.addWidget(nm)
        if status=="duplicate":
            dup = QLabel(" ⚠ DUP", hdr)
            dup.setStyleSheet(f"color:{AMB};font-weight:bold;font-size:8px;background:transparent;")
            hlay.addWidget(dup)
        hlay.addStretch()

        # Per-card ratio dropdown (hidden until Manual mode)
        init_ratio = RATIO_BY_SIZE.get((self._tw,self._th),"1:1")
        self._ratio_combo = QComboBox(hdr)
        self._ratio_combo.addItems(RATIO_LABELS)
        self._ratio_combo.setCurrentText(init_ratio)
        self._ratio_combo.setFixedSize(62,20)
        self._ratio_combo.setStyleSheet(
            f"QComboBox{{background:{CAR};color:{PRI};border:1px solid {MUT};"
            f"border-radius:3px;font-size:9px;padding:1px 4px;}}"
            f"QComboBox QAbstractItemView{{background:{CAR};color:{PRI};"
            f"border:1px solid {ACC};selection-background-color:{ACC};}}")
        self._ratio_combo.currentTextChanged.connect(self._on_card_res)
        self._ratio_combo.setVisible(False)
        hlay.addWidget(self._ratio_combo)
        root.addWidget(hdr)

        # Image frame (fixed D×D)
        self._img_frame = QFrame(self)
        self._img_frame.setFixedSize(self.D,self.D)
        self._img_frame.setStyleSheet("background:transparent;border:none;")

        self._img_lbl = QLabel(self._img_frame)
        self._img_lbl.setFixedSize(self.D,self.D)
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        self._img_lbl.mousePressEvent   = self._ds
        self._img_lbl.mouseMoveEvent    = self._dm

        # Zoom slider (top edge)
        _izv = max(-50, min(50, int(round((self.card_zoom-1.0)/0.4*10))))
        self._izs = QSlider(Qt.Orientation.Horizontal, self._img_frame)
        self._izs.setRange(-50,50); self._izs.setValue(_izv)
        self._izs.setFixedWidth(self.D-10); self._izs.setFixedHeight(12)
        self._izs.move(5,2)
        self._izs.setStyleSheet(self._slider_ss())
        self._izs.valueChanged.connect(self._on_izoom)

        # Y slider (right edge, vertical)
        self._ys = QSlider(Qt.Orientation.Vertical, self._img_frame)
        self._ys.setRange(0,1000); self._ys.setValue(int(self.offset_y*1000))
        self._ys.setInvertedAppearance(True)
        self._ys.setFixedWidth(12); self._ys.setFixedHeight(self.D-26)
        self._ys.move(self.D-13,18)
        self._ys.setStyleSheet(self._slider_ss())
        self._ys.valueChanged.connect(self._on_y)

        # X slider (bottom edge)
        self._xs = QSlider(Qt.Orientation.Horizontal, self._img_frame)
        self._xs.setRange(0,1000); self._xs.setValue(int(self.offset_x*1000))
        self._xs.setFixedWidth(self.D-26); self._xs.setFixedHeight(12)
        self._xs.move(5,self.D-13)
        self._xs.setStyleSheet(self._slider_ss())
        self._xs.valueChanged.connect(self._on_x)

        root.addWidget(self._img_frame)
        self._update_slider_states()
        self._render()   # renders placeholder immediately

        # Load the real image on a background thread; update card when ready
        self._image_ready.connect(self._on_image_loaded)
        _get_image_executor().submit(self._bg_load_image)

    def _bg_load_image(self):
        """Background thread: load full image and emit _image_ready."""
        try:
            raw = Image.open(self.img_path)
            if raw.mode == "RGBA":
                bg = Image.new("RGB", raw.size, (22, 33, 62))
                bg.paste(raw, mask=raw.split()[3])
                pil = bg
            else:
                pil = raw.convert("RGB")
        except Exception:
            pil = Image.new("RGB", (self.D, self.D), (22, 33, 62))
        self._image_ready.emit(pil)

    def _on_image_loaded(self, pil: Image.Image):
        """Main thread: replace placeholder with the real image and re-render."""
        try:
            self._pil = pil
            self._update_slider_states()
            self._render()
        except RuntimeError:
            pass   # card was deleted before the load finished

    def _slider_ss(self):
        return (f"QSlider::groove:horizontal{{background:#0d0d0d;height:4px;border-radius:2px;}}"
                f"QSlider::groove:vertical{{background:#0d0d0d;width:4px;border-radius:2px;}}"
                f"QSlider::handle:horizontal{{background:{ACC};width:10px;height:10px;margin:-3px 0;border-radius:5px;}}"
                f"QSlider::handle:vertical{{background:{ACC};width:10px;height:10px;margin:0 -3px;border-radius:5px;}}"
                f"QSlider::sub-page:horizontal{{background:{ACC};border-radius:2px;}}"
                f"QSlider::add-page:vertical{{background:{ACC};border-radius:2px;}}")

    def _delete_image(self):
        if QMessageBox.question(
                self, "Delete Image",
                f"Permanently delete:\n{os.path.basename(self.img_path)}\n"
                f"and its .txt file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            try: os.remove(self.img_path)
            except: pass
            tp = _txt_path(self.img_path)
            if os.path.exists(tp):
                try: os.remove(tp)
                except: pass
            if self.on_deleted:
                self.on_deleted(self.img_path)

    @property
    def _inner_rect(self):
        D,tw,th = self.D,self._tw,self._th
        if tw>=th:
            rw,rh = D, max(4,round(D*th/tw))
        else:
            rw,rh = max(4,round(D*tw/th)), D
        return (D-rw)//2,(D-rh)//2,rw,rh

    def _pan_ranges(self):
        _,_,rw,rh = self._inner_rect
        iw,ih = self._pil.size
        scale = max(rw/iw,rh/ih)*self.card_zoom
        return max(0,round(iw*scale)-rw), max(0,round(ih*scale)-rh)

    def _update_slider_states(self):
        xt,yt = self._pan_ranges()
        self._xs.setEnabled(xt>0)
        self._ys.setEnabled(yt>0)

    def _render(self):
        D = self.D
        rx0,ry0,rw,rh = self._inner_rect
        iw,ih = self._pil.size
        scale = max(rw/iw,rh/ih)*self.card_zoom
        zw = max(1,round(iw*scale)); zh = max(1,round(ih*scale))
        big = self._pil.resize((zw,zh),Image.LANCZOS)
        ox = round(self.offset_x*(zw-rw)) if zw>rw else -((rw-zw)//2)
        oy = round(self.offset_y*(zh-rh)) if zh>rh else -((rh-zh)//2)
        canvas = Image.new("RGB",(D,D),(26,26,46))
        canvas.paste(big,(rx0-ox,ry0-oy))
        if rw<D or rh<D:
            ov = Image.new("RGBA",(D,D),(0,0,0,0))
            dr = ImageDraw.Draw(ov)
            sh = (0,0,0,128)
            if ry0>0:      dr.rectangle([0,0,D-1,ry0-1],fill=sh)
            if ry0+rh<D:   dr.rectangle([0,ry0+rh,D-1,D-1],fill=sh)
            if rx0>0:      dr.rectangle([0,ry0,rx0-1,ry0+rh-1],fill=sh)
            if rx0+rw<D:   dr.rectangle([rx0+rw,ry0,D-1,ry0+rh-1],fill=sh)
            canvas = Image.alpha_composite(canvas.convert("RGBA"),ov).convert("RGB")
            ImageDraw.Draw(canvas).rectangle(
                [rx0,ry0,rx0+rw-1,ry0+rh-1],outline=(218,54,51),width=2)
        self._img_lbl.setPixmap(_pil_to_qpixmap(canvas))

    def set_resolution(self,tw,th):
        self._tw=tw; self._th=th
        self._update_slider_states(); self._render()

    def _on_izoom(self,v):
        self.card_zoom = max(0.3,1.0+float(v)/10.0*0.4)
        self._update_slider_states(); self._render()

    def _ds(self,e):
        self._dsx=e.pos().x(); self._dsy=e.pos().y()
        self._dox=self.offset_x; self._doy=self.offset_y

    def _dm(self,e):
        xt,yt = self._pan_ranges()
        if xt>0:
            self.offset_x = max(0.0,min(1.0,
                self._dox-(e.pos().x()-self._dsx)/max(1,xt)))
        if yt>0:
            self.offset_y = max(0.0,min(1.0,
                self._doy-(e.pos().y()-self._dsy)/max(1,yt)))
        self._xs.blockSignals(True); self._xs.setValue(int(self.offset_x*1000)); self._xs.blockSignals(False)
        self._ys.blockSignals(True); self._ys.setValue(int(self.offset_y*1000)); self._ys.blockSignals(False)
        self._render()

    def _on_x(self,v): self.offset_x=v/1000.0; self._render()
    def _on_y(self,v): self.offset_y=v/1000.0; self._render()

    def get_offset(self):     return (self.offset_x,self.offset_y)
    def get_zoom(self):       return self.card_zoom
    def get_resolution(self): return (self._tw,self._th)

    def set_manual_mode(self,enabled):
        self._ratio_combo.setVisible(enabled)

    def _on_card_res(self,ratio):
        tw,th = RATIO_MAP.get(ratio,(1024,1024))
        self._tw=tw; self._th=th
        self._update_slider_states(); self._render()


# ══════════════════════════════════════════════════════════════════════════════
#  TAG CLOUD
# ══════════════════════════════════════════════════════════════════════════════
class TagCloud(QFrame):
    def __init__(self, parent=None, on_filter=None, on_remove=None,
                 on_replace=None, on_add=None, on_add_custom=None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame{{background:{PAN};border-radius:6px;border:none;}}")
        self.on_filter     = on_filter
        self.on_remove     = on_remove
        self.on_replace    = on_replace
        self.on_add        = on_add
        self.on_add_custom = on_add_custom
        self.all_tags   = {}
        self.selected   = set()
        self.collapsed  = False
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8,6,8,4); root.setSpacing(4)

        # Header
        hdr = QWidget(self); hdr.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(0,0,0,0); hl.setSpacing(6)
        ttl = QLabel("Tag Viewer",hdr)
        ttl.setStyleSheet(f"color:{ACC};font-weight:bold;font-size:13px;background:transparent;")
        hl.addWidget(ttl)
        self._img_badge = QLabel("🖼 0",hdr)
        self._img_badge.setStyleSheet(
            f"background:{CAR};color:{ACC};font-weight:bold;font-size:11px;"
            f"border-radius:10px;padding:1px 6px;")
        hl.addWidget(self._img_badge)
        self._sel_badge = QLabel("",hdr)
        self._sel_badge.setStyleSheet(
            f"background:#185FA5;color:{PRI};font-weight:bold;font-size:11px;"
            f"border-radius:10px;padding:1px 6px;")
        self._sel_badge.setVisible(False)
        hl.addWidget(self._sel_badge)
        hl.addStretch()
        collapse_btn = QPushButton("∧",hdr)
        collapse_btn.setFixedSize(28,22)
        collapse_btn.setStyleSheet(f"background:transparent;color:{SEC};border:none;font-size:13px;")
        collapse_btn.clicked.connect(self._toggle)
        hl.addWidget(collapse_btn)
        root.addWidget(hdr)

        # Actions row
        self._ctrl = QWidget(self); self._ctrl.setStyleSheet("background:transparent;")
        ctrl_v = QVBoxLayout(self._ctrl); ctrl_v.setContentsMargins(0,0,0,0); ctrl_v.setSpacing(3)

        act_row = QWidget(self._ctrl); act_row.setStyleSheet("background:transparent;")
        al = QHBoxLayout(act_row); al.setContentsMargins(0,0,0,0); al.setSpacing(4)
        self._act_btn = QPushButton("Actions ▼", act_row)
        self._act_btn.setFixedHeight(28)
        self._act_btn.setStyleSheet(
            f"background:{ACC};color:{PRI};border:none;border-radius:4px;"
            f"font-weight:bold;padding:4px 10px;")
        self._act_btn.clicked.connect(self._show_actions)
        al.addWidget(self._act_btn)
        self._desel_btn = QPushButton("Select ▼", act_row)
        self._desel_btn.setFixedHeight(28)
        self._desel_btn.setStyleSheet(
            f"background:{CAR};color:{SEC};border:1px solid {MUT};border-radius:4px;padding:4px 8px;")
        self._desel_btn.clicked.connect(self._show_select_menu)
        al.addWidget(self._desel_btn)
        al.addStretch()
        ctrl_v.addWidget(act_row)

        # Search row
        srch_row = QWidget(self._ctrl); srch_row.setStyleSheet("background:transparent;")
        cl = QHBoxLayout(srch_row); cl.setContentsMargins(0,0,0,0); cl.setSpacing(4)
        srch_ico = QLabel("🔍", srch_row)
        srch_ico.setStyleSheet(f"color:{MUT};font-size:14px;background:transparent;")
        cl.addWidget(srch_ico)
        self._search = QLineEdit(srch_row)
        self._search.setPlaceholderText("Search tags...")
        self._search.setFixedHeight(28)
        self._search.setStyleSheet(f"background:{CAR};color:{PRI};border:1px solid {MUT};border-radius:4px;padding:2px 6px;")
        self._search.textChanged.connect(lambda _: self._refresh())
        cl.addWidget(self._search, 1)
        clr = QPushButton("×", srch_row)
        clr.setFixedSize(28, 28)
        clr.setStyleSheet(f"background:transparent;color:{MUT};border:none;font-size:15px;")
        clr.clicked.connect(lambda: self._search.clear())
        cl.addWidget(clr)
        ctrl_v.addWidget(srch_row)

        root.addWidget(self._ctrl)

        # Pill scroll area — flow layout (horizontal mode)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFixedHeight(220)
        self._scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        self._pill_widget = QWidget()
        self._pill_widget.setStyleSheet(f"background:{PAN};")
        self._pill_layout = FlowLayout(self._pill_widget,h_spacing=4,v_spacing=4)
        self._pill_widget.setLayout(self._pill_layout)
        self._scroll.setWidget(self._pill_widget)
        root.addWidget(self._scroll)

        # Pill scroll area — vertical layout (vertical mode, one pill per line)
        self._scroll_v = QScrollArea(self)
        self._scroll_v.setWidgetResizable(True)
        self._scroll_v.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        self._pill_v_widget = QWidget()
        self._pill_v_widget.setStyleSheet(f"background:{PAN};")
        self._pill_v_layout = QVBoxLayout(self._pill_v_widget)
        self._pill_v_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._pill_v_layout.setContentsMargins(4, 4, 4, 4)
        self._pill_v_layout.setSpacing(3)
        self._scroll_v.setWidget(self._pill_v_widget)
        self._scroll_v.setVisible(False)
        root.addWidget(self._scroll_v)

        self._single_col = False

    def _toggle(self):
        self.collapsed = not self.collapsed
        self._ctrl.setVisible(not self.collapsed)
        # Only show the active scroll area when expanded
        if not self.collapsed:
            self._scroll.setVisible(not self._single_col)
            self._scroll_v.setVisible(self._single_col)
        else:
            self._scroll.setVisible(False)
            self._scroll_v.setVisible(False)

    def load(self,tags_freq,total_images):
        self.all_tags = tags_freq
        self._img_badge.setText(f"🖼 {total_images}")
        self._refresh()

    def set_single_column(self, single: bool):
        """Switch between flow layout (horizontal mode) and single-column layout (vertical mode)."""
        if getattr(self, '_single_col', False) == single:
            return
        self._single_col = single
        self._scroll.setVisible(not single)
        self._scroll_v.setVisible(single)
        self._refresh()

    def _refresh(self):
        # Clear active pill container
        if self._single_col:
            while self._pill_v_layout.count():
                it = self._pill_v_layout.takeAt(0)
                if it.widget(): it.widget().deleteLater()
            active_widget = self._pill_v_widget
        else:
            while self._pill_layout.count():
                it = self._pill_layout.takeAt(0)
                if it.widget(): it.widget().deleteLater()
            active_widget = self._pill_widget

        search = self._search.text().lower()
        tags = [(t,c) for t,c in sorted(self.all_tags.items(),key=lambda x:-x[1])
                if search in t.lower()]
        if not tags:
            lbl = QLabel("No tags found.")
            lbl.setStyleSheet(f"color:{MUT};font-size:11px;padding:8px;background:transparent;")
            if self._single_col:
                self._pill_v_layout.addWidget(lbl)
            else:
                self._pill_layout.addWidget(lbl)
            return
        self._pill_map = {}
        for tag,count in tags:
            sel = tag in self.selected
            txt = f" {tag}  {count} "
            fg  = ACC if sel else CAR
            bc2 = ACC if sel else MUT
            btn = QPushButton(txt, active_widget)
            btn.setFixedHeight(28)
            btn.setStyleSheet(
                f"QPushButton{{background:{fg};color:{PRI};border:1px solid {bc2};"
                f"border-radius:10px;font-size:11px;font-weight:bold;padding:2px 6px;}}"
                f"QPushButton:hover{{background:#185FA5;border-color:{ACC};}}")
            if self._single_col:
                btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _,t=tag: self._toggle_tag(t))
            if self._single_col:
                self._pill_v_layout.addWidget(btn)
            else:
                self._pill_layout.addWidget(btn)
            self._pill_map[tag] = btn
        QTimer.singleShot(0, lambda: active_widget.adjustSize())

    def _toggle_tag(self,tag):
        if tag in self.selected: self.selected.discard(tag)
        else:                     self.selected.add(tag)
        n = len(self.selected)
        self._sel_badge.setText(f"🏷 {n}" if n else "")
        self._sel_badge.setVisible(bool(n))
        self._desel_btn.setStyleSheet(
            f"background:{CAR};color:{RED if n else SEC};border:1px solid {RED if n else MUT};border-radius:4px;padding:4px 8px;")

        if True:
            btn = self._pill_map.get(tag)
            if btn and btn.isVisible():
                sel = tag in self.selected
                txt = f" {tag}  {self.all_tags.get(tag,'')} "
                fg  = ACC if sel else CAR
                bc2 = ACC if sel else MUT
                btn.setText(txt)
                btn.setStyleSheet(
                    f"QPushButton{{background:{fg};color:{PRI};border:1px solid {bc2};"
                    f"border-radius:10px;font-size:11px;font-weight:bold;padding:2px 6px;}}"
                    f"QPushButton:hover{{background:#185FA5;border-color:{ACC};}}")
        else:
            self._refresh()
        if self.on_filter: self.on_filter(self.selected)

    def _deselect_all(self):
        self.selected.clear()
        self._sel_badge.setText(""); self._sel_badge.setVisible(False)
        self._desel_btn.setStyleSheet(
            f"background:{CAR};color:{SEC};border:1px solid {MUT};border-radius:4px;padding:4px 8px;")
        self._refresh()
        if self.on_filter: self.on_filter(set())

    def _select_all(self):
        self.selected = set(self.all_tags.keys())
        n = len(self.selected)
        self._sel_badge.setText(f"🏷 {n}" if n else "")
        self._sel_badge.setVisible(bool(n))
        self._desel_btn.setStyleSheet(
            f"background:{CAR};color:{RED};border:1px solid {RED};border-radius:4px;padding:4px 8px;")
        self._refresh()
        if self.on_filter: self.on_filter(self.selected)

    def _show_select_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu{{background:{CAR};color:{PRI};border:1px solid {MUT};}}"
            f"QMenu::item:selected{{background:{ACC};}}")
        menu.addAction("Deselect All", self._deselect_all)
        menu.addAction("Select All",   self._select_all)
        btn = self._desel_btn
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    def _show_actions(self):
        if not self.selected:
            QMessageBox.information(self,"Actions","Select tags in the cloud first."); return
        n = len(self.selected)
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu{{background:{CAR};color:{PRI};border:1px solid {MUT};}}"
            f"QMenu::item:selected{{background:{ACC};}}")
        menu.addAction(f"➕  Add tags to all ({n})",
                       lambda: self.on_add and self.on_add(self.selected.copy()))
        menu.addAction("✏  Add tag to selection",
                       lambda: self.on_add_custom and self.on_add_custom())
        menu.addAction(f"🗑  Remove tags ({n})",
                       lambda: self.on_remove and self.on_remove(self.selected.copy()))
        menu.addAction(f"↔  Replace tags ({n})",
                       lambda: self.on_replace and self.on_replace(self.selected.copy()))
        menu.exec(self._act_btn.mapToGlobal(
            QPoint(0,self._act_btn.height())))

    def highlight(self,selected_tags):
        self.selected = set(selected_tags)
        self._refresh()


# ══════════════════════════════════════════════════════════════════════════════
#  THUMB LABEL  (hover detection for EditorCard)
# ══════════════════════════════════════════════════════════════════════════════
class _ThumbLabel(QLabel):
    hovered = Signal(bool)
    def enterEvent(self,e): self.hovered.emit(True); super().enterEvent(e)
    def leaveEvent(self,e):
        QTimer.singleShot(80,self._check)
        super().leaveEvent(e)
    def _check(self):
        pos = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(pos): self.hovered.emit(False)


# ══════════════════════════════════════════════════════════════════════════════
#  DRAG-AND-DROP TAG PILLS
# ══════════════════════════════════════════════════════════════════════════════
class DraggableTagPill(QPushButton):
    """Tag pill that initiates a drag on significant mouse movement."""

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_start = e.pos()
            self._dragging   = False
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if not getattr(self, '_drag_start', None):
            return
        if (e.pos() - self._drag_start).manhattanLength() < 8:
            return
        self._dragging = True
        mime = QMimeData()
        mime.setText(self.property("tag_name"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.setHotSpot(e.pos())
        drag.exec(Qt.DropAction.MoveAction)

    def mouseReleaseEvent(self, e):
        if getattr(self, '_dragging', False):
            self._dragging   = False
            self._drag_start = None
            return          # swallow — no click/delete
        super().mouseReleaseEvent(e)


class DroppablePillWidget(QWidget):
    """Pill container that accepts tag drops and emits the new insertion index."""
    tag_reordered = Signal(str, int)   # (tag_name, new_index)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        tag = e.mimeData().text()
        idx = self._drop_index(e.position().toPoint())
        e.acceptProposedAction()
        self.tag_reordered.emit(tag, idx)

    def _drop_index(self, pos):
        layout = self.layout()
        if not layout:
            return 0
        n = layout.count()
        for i in range(n):
            item = layout.itemAt(i)
            if not item or not item.widget():
                continue
            geo = item.widget().geometry()
            # Drop point is above this pill's row → insert before it
            if pos.y() < geo.top():
                return i
            # Same row: insert before if left of centre
            if geo.top() <= pos.y() <= geo.bottom():
                if pos.x() < geo.x() + geo.width() // 2:
                    return i
        return n


# ══════════════════════════════════════════════════════════════════════════════
#  EDITOR CARD
# ══════════════════════════════════════════════════════════════════════════════
class EditorCard(QFrame):
    def __init__(self, parent, img_path, on_tags_changed=None,
                 on_deleted=None, highlighted_tags=None, view_mode="Tag",
                 trigger_word=""):
        super().__init__(parent)
        self.setStyleSheet(
            f"EditorCard{{background:{CAR};border:2px solid {ACC};border-radius:6px;}}")
        self.setObjectName("EditorCard")
        self.img_path        = img_path
        self.on_tags_changed = on_tags_changed
        self.on_deleted      = on_deleted
        self.highlighted     = highlighted_tags or set()
        self._view_mode      = view_mode
        self._captions       = []
        self._trigger_word   = trigger_word.lower()
        self._hover_frame    = None

        root = QVBoxLayout(self)
        root.setContentsMargins(2,2,2,4); root.setSpacing(0)

        # Thumbnail
        self.thumb_lbl = _ThumbLabel(self)
        self.thumb_lbl.setFixedSize(THUMB_SIZE,THUMB_SIZE)
        self.thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_lbl.setStyleSheet("background:transparent;border:none;")
        self.thumb_lbl.hovered.connect(self._on_hover)
        root.addWidget(self.thumb_lbl)

        # Tag pill scroll area
        self._tag_scroll = QScrollArea(self)
        self._tag_scroll.setWidgetResizable(True)
        self._tag_scroll.setFixedHeight(160)
        self._tag_scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        self._pill_widget = DroppablePillWidget()
        self._pill_widget.setStyleSheet(f"background:{CAR};")
        self._pill_layout = FlowLayout(self._pill_widget,h_spacing=3,v_spacing=3)
        self._pill_layout.setContentsMargins(4,4,4,4)
        self._pill_widget.setLayout(self._pill_layout)
        self._pill_widget.tag_reordered.connect(self._on_tag_reordered)
        self._tag_scroll.setWidget(self._pill_widget)
        root.addWidget(self._tag_scroll)

        # Caption section
        self._cap_frame = QFrame(self)
        self._cap_frame.setStyleSheet(f"background:{CAR};border:none;")
        cap_vl = QVBoxLayout(self._cap_frame)
        cap_vl.setContentsMargins(2,2,2,4); cap_vl.setSpacing(2)
        cap_hdr_w = QWidget(self._cap_frame)
        cap_hdr_w.setStyleSheet(f"background:{PAN};")
        cap_hdr_w.setFixedHeight(26)
        cap_hl = QHBoxLayout(cap_hdr_w)
        cap_hl.setContentsMargins(6,0,4,0); cap_hl.setSpacing(4)
        cap_lbl = QLabel("CAPTION",cap_hdr_w)
        cap_lbl.setStyleSheet(f"color:{SEC};font-weight:bold;font-size:10px;background:transparent;")
        cap_hl.addWidget(cap_lbl); cap_hl.addStretch()
        save_cap_btn = QPushButton("✓",cap_hdr_w)
        save_cap_btn.setFixedSize(28,20)
        save_cap_btn.setStyleSheet(f"background:transparent;color:{GRN};border:none;font-weight:bold;font-size:12px;")
        save_cap_btn.clicked.connect(self._save_caption)
        cap_hl.addWidget(save_cap_btn)
        cap_vl.addWidget(cap_hdr_w)
        self._cap_text = QPlainTextEdit(self._cap_frame)
        self._cap_text.setFixedHeight(120)
        self._cap_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._cap_text.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._cap_text.setStyleSheet(
            f"background:{PAN};color:{PRI};border:none;font-size:11px;padding:2px;")
        self._cap_text.focusOutEvent = lambda e: (self._save_caption(), QPlainTextEdit.focusOutEvent(self._cap_text,e))
        cap_vl.addWidget(self._cap_text)
        root.addWidget(self._cap_frame)

        # Add tag row
        self._add_row = QWidget(self)
        self._add_row.setStyleSheet("background:transparent;")
        add_hl = QHBoxLayout(self._add_row)
        add_hl.setContentsMargins(4,2,4,4); add_hl.setSpacing(4)
        self._add_entry = QLineEdit(self._add_row)
        self._add_entry.setPlaceholderText("Add tags...")
        self._add_entry.setFixedHeight(30)
        self._add_entry.setStyleSheet(
            f"background:{PAN};color:{PRI};border:1px solid {MUT};border-radius:4px;padding:2px 6px;")
        self._add_entry.returnPressed.connect(self._add_tags)
        add_hl.addWidget(self._add_entry,1)
        plus_btn = QPushButton("+",self._add_row)
        plus_btn.setFixedSize(30,30)
        plus_btn.setStyleSheet(
            f"background:{ACC};color:{PRI};border:none;border-radius:4px;font-weight:bold;font-size:15px;")
        plus_btn.clicked.connect(self._add_tags)
        add_hl.addWidget(plus_btn)
        root.addWidget(self._add_row)

        if img_path:
            self._load_thumb()
            self.refresh_pills()

    def _load_thumb(self):
        if not self.img_path: return
        img = _load_or_cache_thumb(self.img_path)
        self.thumb_lbl.setPixmap(_pil_to_qpixmap(img))

    def refresh_pills(self):
        all_items = _read_tags(self.img_path)
        tags,captions = _split_tags_captions(all_items)
        self._captions = captions

        self._cap_text.blockSignals(True)
        self._cap_text.setPlainText(captions[0] if captions else "")
        self._cap_text.blockSignals(False)

        # Clear pills
        while self._pill_layout.count():
            it = self._pill_layout.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self._pill_map = {}

        trig_low = self._trigger_word
        for tag in tags:
            hl  = tag.lower() in {tg.lower() for tg in self.highlighted}
            is_trig = bool(trig_low) and tag.lower()==trig_low
            fg  = ACC if hl else "#3a3a3a"
            bc2 = ACC if (hl or is_trig) else SEC
            btn = DraggableTagPill(f" {tag.upper()} ", self._pill_widget)
            btn.setProperty("tag_name", tag)
            btn.setFixedHeight(28)
            btn.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            btn.setStyleSheet(
                f"QPushButton{{background:{fg};color:{PRI};border:1px solid {bc2};"
                f"border-radius:4px;font-size:12px;font-weight:bold;padding:2px 4px;}}"
                f"QPushButton:hover{{background:{RED};border-color:{RED};}}")
            btn.clicked.connect(lambda _,tg=tag: self._remove_tag(tg))
            self._pill_layout.addWidget(btn)
            self._pill_map[tag.lower()] = btn

        QTimer.singleShot(0,lambda: self._pill_widget.adjustSize())
        self._refresh_sections()

    def _refresh_sections(self):
        mode = self._view_mode
        show_tags = mode in ("Tag", "Both")
        show_cap  = mode in ("Caption", "Both")
        self._tag_scroll.setVisible(show_tags)
        self._cap_frame.setVisible(show_cap)
        self._add_row.setVisible(show_tags)

    def _remove_tag(self,tag):
        all_items = _read_tags(self.img_path)
        tags,caps = _split_tags_captions(all_items)
        tags = [tg for tg in tags if tg.lower()!=tag.lower()]
        _write_tags(self.img_path,tags+caps)
        btn = self._pill_map.pop(tag.lower(),None)
        if btn:
            self._pill_layout.removeWidget(btn)
            btn.deleteLater()
            QTimer.singleShot(0,lambda: self._pill_widget.adjustSize())
        if self.on_tags_changed: self.on_tags_changed(self.img_path)

    def _add_tags(self):
        raw = self._add_entry.text().strip()
        if not raw: return
        new  = [t.strip() for t in raw.split(',') if t.strip()]
        all_items = _read_tags(self.img_path)
        tags,caps = _split_tags_captions(all_items)
        curr_low = {t.lower() for t in tags}
        before = len(tags)
        for t in new:
            if t.lower() not in curr_low:
                tags.append(t); curr_low.add(t.lower())
        if len(tags)==before:
            self._add_entry.clear(); return
        _write_tags(self.img_path,tags+caps)
        self._add_entry.clear()
        self.refresh_pills()
        if self.on_tags_changed: self.on_tags_changed(self.img_path)

    def _save_caption(self,event=None):
        text = self._cap_text.toPlainText().strip()
        all_items = _read_tags(self.img_path)
        tags,_ = _split_tags_captions(all_items)
        if text:
            _write_tags(self.img_path,tags+[text]); self._captions=[text]
        else:
            _write_tags(self.img_path,tags); self._captions=[]
        if self.on_tags_changed: self.on_tags_changed(self.img_path)

    def _on_tag_reordered(self, tag: str, drop_idx: int):
        all_items = _read_tags(self.img_path)
        tags, caps = _split_tags_captions(all_items)
        # Find original position (case-insensitive)
        tag_low = tag.lower()
        orig_idx = next((i for i,t in enumerate(tags) if t.lower()==tag_low), None)
        if orig_idx is None or orig_idx == drop_idx:
            return
        tags.insert(drop_idx, tags.pop(orig_idx))
        _write_tags(self.img_path, tags + caps)
        self.refresh_pills()
        if self.on_tags_changed: self.on_tags_changed(self.img_path)

    def set_highlighted(self,tags):
        self.highlighted=tags; self.refresh_pills()

    def set_view_mode(self,mode):
        self._view_mode=mode; self._refresh_sections()

    def _on_hover(self,entered):
        if entered: self._show_hover()
        else:       self._hide_hover()

    def _show_hover(self):
        if self._hover_frame: return
        self._hover_frame = QFrame(self.thumb_lbl)
        self._hover_frame.setStyleSheet("background:transparent;border:none;")
        hfl = QHBoxLayout(self._hover_frame)
        hfl.setContentsMargins(0,0,0,0); hfl.setSpacing(4)
        for text,tip,color,cmd in [("OFF","Disable image (hides from tagger)",AMB,self._disable_image),
                                    ("DEL","Delete image from dataset",RED,self._delete_image)]:
            btn = QPushButton(text,self._hover_frame)
            btn.setFixedSize(34,24)
            btn.setToolTip(tip)
            btn.setStyleSheet(
                f"QPushButton{{background:{CAR};color:{color};border-radius:4px;"
                f"font-family:{FONT};font-size:9px;font-weight:bold;border:none;}}"
                f"QPushButton:hover{{background:{color};color:{PRI};}}")
            btn.clicked.connect(cmd)
            hfl.addWidget(btn)
        self._hover_frame.adjustSize()
        self._hover_frame.move(
            self.thumb_lbl.width()-self._hover_frame.width()-4,4)
        self._hover_frame.show()
        self._hover_frame.raise_()

    def _hide_hover(self):
        if self._hover_frame:
            self._hover_frame.deleteLater()
            self._hover_frame=None

    def _disable_image(self):
        tp = _txt_path(self.img_path)
        if os.path.exists(tp):
            try: os.rename(tp,tp+".disabled")
            except: pass
        if _tag_cache is not None: _tag_cache.pop(self.img_path,None)
        if _tag_dirty is not None: _tag_dirty.discard(self.img_path)
        self.refresh_pills()

    def _delete_image(self):
        if QMessageBox.question(self,"Delete Image",
                f"Permanently delete:\n{os.path.basename(self.img_path)}\nand its .txt file?",
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No
                ) == QMessageBox.StandardButton.Yes:
            try: os.remove(self.img_path)
            except: pass
            tp = _txt_path(self.img_path)
            if os.path.exists(tp):
                try: os.remove(tp)
                except: pass
            if self.on_deleted: self.on_deleted(self.img_path)

    def recycle(self,img_path,highlighted_tags=None,view_mode="Tag",trigger_word=""):
        if self._hover_frame:
            try: self._hover_frame.deleteLater()
            except: pass
            self._hover_frame=None
        self.img_path      = img_path
        self.highlighted   = highlighted_tags or set()
        self._view_mode    = view_mode
        self._trigger_word = trigger_word.lower()
        self._load_thumb()
        self.refresh_pills()

    def hide(self): self.setVisible(False)
    def show(self): self.setVisible(True)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRA CONSTANTS  (Part 2 — TagHandlerPage)
# ══════════════════════════════════════════════════════════════════════════════
SCALE_MODES   = ["Normal", "RealESRGAN 4x (Photo)", "RealESRGAN 4x (Anime)"]

# JoyCaption option lists (Part 1 has CAP_TYPES / CAP_LENGTHS for the cloud
#  display names — these match the v1 tagger panel exactly)
JOY_CAP_TYPES  = [
    "Descriptive", "Descriptive (Informal)", "Training Prompt",
    "MidJourney", "Booru tag list", "Art Critic", "Product Listing",
    "Social Media Post",
]
JOY_CAP_LENGTHS = ["Any", "Short", "Medium", "Long"]

# Output modes for the tagger panel (v1 naming)
_TAGGER_OUTPUT_MODES = ["Append", "Overwrite", "Skip"]

# Tagger engine options
_ENGINES = ["WD14 Ensemble", "WD14 Single", "JoyCaption"]


# ══════════════════════════════════════════════════════════════════════════════
#  _UIBridge  — thread-safe callable dispatcher (replaces Tk's after(0, fn))
# ══════════════════════════════════════════════════════════════════════════════
class _UIBridge(QObject):
    _call_sig = Signal(object)

    def __init__(self):
        super().__init__()
        self._call_sig.connect(lambda fn: fn())

    def call(self, fn):
        self._call_sig.emit(fn)


# ══════════════════════════════════════════════════════════════════════════════
#  TagHandlerPage
# ══════════════════════════════════════════════════════════════════════════════
class TagHandlerPage(QWidget):
    """Root widget for the Tag Handler tab — full PySide6 port of v1.6."""

    # ── widget helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _mk_btn(parent, text, color=None, min_w=80, checkable=False):
        b = QPushButton(text, parent)
        b.setCheckable(checkable)
        b.setMinimumWidth(min_w)
        b.setStyleSheet(
            f"QPushButton{{background:{color or ACC};color:#fff;"
            f"border:none;border-radius:4px;padding:4px 10px;}}"
            f"QPushButton:hover{{background:{PRI};}}"
            f"QPushButton:checked{{background:{GRN};}}")
        return b

    @staticmethod
    def _mk_combo(parent, items, width=160):
        c = QComboBox(parent)
        c.addItems(items)
        c.setFixedWidth(width)
        c.setStyleSheet(
            f"QComboBox{{background:{PAN};color:{SEC};border:1px solid {MUT};"
            f"border-radius:4px;padding:2px 6px;}}"
            f"QComboBox QAbstractItemView{{background:{PAN};color:{SEC};}}")
        return c

    @staticmethod
    def _mk_label(parent, text, bold=False, color=None):
        lbl = QLabel(text, parent)
        f = lbl.font()
        if bold:
            f.setBold(True)
        lbl.setFont(f)
        lbl.setStyleSheet(f"color:{color or SEC};")
        return lbl

    @staticmethod
    def _mk_entry(parent, width=200, placeholder=""):
        e = QLineEdit(parent)
        e.setFixedWidth(width)
        e.setPlaceholderText(placeholder)
        e.setStyleSheet(
            f"QLineEdit{{background:{PAN};color:{SEC};border:1px solid {MUT};"
            f"border-radius:4px;padding:2px 6px;}}")
        return e

    @staticmethod
    def _mk_slider(parent, lo=0, hi=100, val=50):
        s = QSlider(Qt.Horizontal, parent)
        s.setRange(lo, hi)
        s.setValue(val)
        s.setStyleSheet(
            f"QSlider::groove:horizontal{{height:4px;background:{MUT};}}"
            f"QSlider::handle:horizontal{{background:{ACC};width:14px;height:14px;"
            f"margin:-5px 0;border-radius:7px;}}")
        return s

    @staticmethod
    def _mk_progress(parent):
        pb = QProgressBar(parent)
        pb.setRange(0, 1000)
        pb.setValue(0)
        pb.setTextVisible(False)
        pb.setFixedHeight(6)
        pb.setStyleSheet(
            f"QProgressBar{{background:{PAN};border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{ACC};border-radius:3px;}}")
        return pb

    @staticmethod
    def _mk_text_edit(parent, height=120, read_only=False):
        t = QPlainTextEdit(parent)
        t.setFixedHeight(height)
        t.setReadOnly(read_only)
        t.setStyleSheet(
            f"QPlainTextEdit{{background:{PAN};color:{SEC};border:1px solid {MUT};"
            f"border-radius:4px;padding:4px;}}")
        return t

    @staticmethod
    def _set_progress(pb: QProgressBar, v: float):
        pb.setValue(max(0, min(1000, int(v * 1000))))

    # ── init ─────────────────────────────────────────────────────────────────
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG};")

        self._cfg      = _load_cfg()
        self._bridge   = _UIBridge()
        self._profiles = _load_profiles()

        # dataset state
        self._dataset_folder: str   = self._cfg.get("last_dataset", "")
        self._images:    list[str]  = []      # full paths
        self._tag_freq:  dict[str, int] = {}
        self._undo_stack: list      = []

        # editor state
        self._editor_loaded:  int  = 0
        self._editor_visible: list = []
        self._editor_cards: list[EditorCard] = []
        self._tags_changed: set[str] = set()
        self._flush_timer  = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_tags_changed)

        # prepare state
        self._prep_card_widgets: list[PrepareCard] = []
        self._crop_offsets: dict = {}   # img_path → (ox, oy, zoom, tw, th)
        self._prep_loaded: int = 0      # how many cards have been appended so far

        # tagger state
        self._tagger_thread_obj = None

        # build UI
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._build_tabs(root)

        # keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+S"), self, self._apply_edits)

        # auto-load last dataset
        if self._dataset_folder and os.path.isdir(self._dataset_folder):
            QTimer.singleShot(200, self._load_dataset)

        # first-run model check
        QTimer.singleShot(600, self._check_first_run)

    # ── topbar ───────────────────────────────────────────────────────────────
    def _build_topbar(self, root: QVBoxLayout):
        pass  # topbar removed — dataset + controls live inside the Prepare tab

    # ── tabs ─────────────────────────────────────────────────────────────────
    def _build_tabs(self, root: QVBoxLayout):
        self._tabs = QTabWidget(self)
        self._tabs.setStyleSheet(
            f"QTabWidget::pane{{border:1px solid {MUT};border-radius:4px;background:{BG};}}"
            f"QTabBar::tab{{background:{PAN};color:{MUT};padding:6px 18px;"
            f"border-radius:4px 4px 0 0;}}"
            f"QTabBar::tab:selected{{background:{ACC};color:#fff;}}"
            f"QTabBar::tab:hover{{background:{PRI};}}")

        self._prepare_tab = QWidget()
        self._tagger_tab  = QWidget()
        self._editor_tab  = QWidget()
        self._batch_tab   = QWidget()
        self._help_tab    = QWidget()

        self._tabs.addTab(self._prepare_tab, "Prepare")
        self._tabs.addTab(self._tagger_tab,  "Tagger")
        self._tabs.addTab(self._editor_tab,  "Editor")
        self._tabs.addTab(self._batch_tab,   "Batch")
        self._tabs.addTab(self._help_tab,    "Help")

        self._build_prepare()
        self._build_tagger()
        self._build_editor()
        self._build_batch()
        self._build_help()

        self._tabs.currentChanged.connect(self._on_tab_change)
        root.addWidget(self._tabs, stretch=1)

    def _on_tab_change(self, idx: int):
        pass

    # ══════════════════════════════════════════════════════════════════════════
    #  PREPARE TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_prepare(self):
        v = QVBoxLayout(self._prepare_tab)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        # ── dataset row ──────────────────────────────────────────────────────
        ds_row = QFrame(self._prepare_tab)
        ds_row.setStyleSheet(f"background:{PAN};border-radius:4px;")
        dh = QHBoxLayout(ds_row)
        dh.setContentsMargins(8, 4, 8, 4)
        dh.setSpacing(8)

        btn_browse = self._mk_btn(ds_row, "Browse Dataset", min_w=110)
        btn_browse.clicked.connect(self._browse_dataset)
        dh.addWidget(btn_browse)

        self._prep_status = self._mk_label(ds_row, "No dataset loaded.", color=MUT)
        dh.addWidget(self._prep_status)
        dh.addStretch()
        v.addWidget(ds_row)

        # ── controls row ─────────────────────────────────────────────────────
        ctrl = QFrame(self._prepare_tab)
        ctrl.setStyleSheet(f"background:{PAN};border-radius:4px;")
        ch = QHBoxLayout(ctrl)
        ch.setContentsMargins(8, 4, 8, 4)
        ch.setSpacing(8)

        ch.addWidget(self._mk_label(ctrl, "Resolution:"))
        self._res_combo = self._mk_combo(ctrl, RES_LABELS + ["Manual"], width=130)
        cur_res = self._cfg.get("resolution", DEFAULT_RES)
        idx = self._res_combo.findText(cur_res)
        if idx >= 0:
            self._res_combo.setCurrentIndex(idx)
        self._res_combo.currentTextChanged.connect(self._update_res_indicator)
        ch.addWidget(self._res_combo)

        self._res_ratio_lbl = self._mk_label(ctrl, "1:1", color=MUT)
        ch.addWidget(self._res_ratio_lbl)

        ch.addWidget(self._mk_label(ctrl, "Scale:"))
        self._scale_combo = self._mk_combo(ctrl, SCALE_MODES, width=120)
        ch.addWidget(self._scale_combo)

        ch.addWidget(self._mk_label(ctrl, "Format:"))
        btn_convert = self._mk_btn(ctrl, "Convert PNG/JPG", min_w=110)
        btn_convert.clicked.connect(self._convert_formats)
        ch.addWidget(btn_convert)

        ch.addStretch()

        btn_clear = self._mk_btn(ctrl, "Clear Cache", min_w=90)
        btn_clear.clicked.connect(self._clear_thumb_cache)
        ch.addWidget(btn_clear)

        btn_backup = self._mk_btn(ctrl, "Backup", min_w=80)
        btn_backup.clicked.connect(self._backup_dataset)
        ch.addWidget(btn_backup)

        ch.addWidget(self._mk_label(ctrl, "Cols:"))
        self._prep_cols_spin = QSpinBox(ctrl)
        self._prep_cols_spin.setRange(1, 10)
        self._prep_cols_spin.setValue(4)
        self._prep_cols_spin.setFixedWidth(50)
        self._prep_cols_spin.setStyleSheet(
            f"QSpinBox{{background:{PAN};color:{SEC};border:1px solid {MUT};"
            f"border-radius:4px;padding:2px 4px;}}")
        self._prep_cols_spin.valueChanged.connect(self._render_prepare_page)
        ch.addWidget(self._prep_cols_spin)

        btn_process = self._mk_btn(ctrl, "Process", color=GRN, min_w=90)
        btn_process.clicked.connect(self._process_images)
        ch.addWidget(btn_process)

        v.addWidget(ctrl)

        # progress
        self._prep_progress = self._mk_progress(self._prepare_tab)
        v.addWidget(self._prep_progress)

        # scroll area for cards
        self._prep_scroll = QScrollArea(self._prepare_tab)
        self._prep_scroll.setWidgetResizable(True)
        self._prep_scroll.setStyleSheet(f"QScrollArea{{border:none;background:{BG};}}")
        self._prep_inner  = QWidget()
        self._prep_inner.setStyleSheet(f"background:{BG};")
        self._prep_grid   = QGridLayout(self._prep_inner)
        self._prep_grid.setSpacing(6)
        self._prep_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._prep_scroll.setWidget(self._prep_inner)
        self._prep_scroll.verticalScrollBar().valueChanged.connect(
            self._on_prep_scroll)
        v.addWidget(self._prep_scroll, stretch=1)


        # init ratio label
        self._update_res_indicator(self._res_combo.currentText())

    def _render_prepare_page(self):
        """Clear all cards and load the first batch (infinite-scroll reset)."""
        # snapshot any currently visible cards before clearing
        for card in self._prep_card_widgets:
            ox, oy = card.get_offset()
            tw_c, th_c = card.get_resolution()
            self._crop_offsets[card.img_path] = (ox, oy, card.get_zoom(), tw_c, th_c)

        for card in self._prep_card_widgets:
            card.setParent(None)
            card.deleteLater()
        self._prep_card_widgets.clear()

        while self._prep_grid.count():
            item = self._prep_grid.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        self._prep_loaded = 0
        if not self._images:
            return
        self._append_prepare_cards(0, min(PREP_BATCH, len(self._images)))

    def _append_prepare_cards(self, start: int, end: int):
        """Append cards for self._images[start:end] to the grid."""
        cols      = self._prep_cols_spin.value()
        res_txt   = self._res_combo.currentText()
        is_manual = (res_txt == "Manual")
        tw, th, _ = RES_MAP.get(res_txt, (1024, 1024, "1:1"))

        for i, img_path in enumerate(self._images[start:end], start=start):
            saved    = self._crop_offsets.get(img_path)
            s_offset = (saved[0], saved[1]) if saved else None
            s_zoom   = saved[2] if saved else 1.0
            if is_manual and saved and len(saved) > 3:
                _tw, _th = saved[3], saved[4]
            else:
                _tw, _th = tw, th
            card = PrepareCard(self._prep_inner, img_path,
                               saved_offset=s_offset, saved_zoom=s_zoom,
                               tw=_tw, th=_th,
                               on_deleted=self._on_prepare_image_deleted)
            if is_manual:
                card.set_manual_mode(True)
            self._prep_card_widgets.append(card)
            row, col = divmod(i, cols)
            self._prep_grid.addWidget(card, row, col)

        self._prep_loaded = end
        total = len(self._images)
        if end < total:
            self._prep_status.setText(
                f"{end} of {total} shown — scroll down to load more")
        else:
            self._prep_status.setText(
                f"{total} images loaded from {os.path.basename(self._dataset_folder)}")

    def _on_prep_scroll(self, value: int):
        """Load next batch when scrolled to 85% of the current content."""
        if self._prep_loaded >= len(self._images):
            return
        sb = self._prep_scroll.verticalScrollBar()
        if sb.maximum() > 0 and value >= sb.maximum() * 0.85:
            nxt = min(self._prep_loaded + PREP_BATCH, len(self._images))
            self._append_prepare_cards(self._prep_loaded, nxt)


    # ══════════════════════════════════════════════════════════════════════════
    #  TAGGER TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tagger(self):  # noqa: C901
        v = QVBoxLayout(self._tagger_tab)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        scroll = QScrollArea(self._tagger_tab)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{BG};}}")
        inner = QWidget()
        inner.setStyleSheet(f"background:{BG};")
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(4, 4, 4, 4)
        iv.setSpacing(6)
        scroll.setWidget(inner)
        v.addWidget(scroll, stretch=1)

        # ── Trigger Tag ──────────────────────────────────────────────────────
        trig_frame = QFrame(inner)
        trig_frame.setStyleSheet(
            f"QFrame{{background:{CAR};border:2px solid {ACC};border-radius:6px;}}")
        th = QHBoxLayout(trig_frame)
        th.setContentsMargins(12, 8, 12, 8)
        th.setSpacing(8)
        th.addWidget(self._mk_label(trig_frame, "⭐  Trigger Tag", bold=True, color=ACC))
        self._trigger_entry = QLineEdit(trig_frame)
        self._trigger_entry.setPlaceholderText(
            'Add a trigger word, e.g. "z0charname" (required)')
        self._trigger_entry.setText(self._cfg.get("trigger_word", ""))
        self._trigger_entry.setStyleSheet(
            f"QLineEdit{{background:{PAN};color:{PRI};border:none;"
            f"border-radius:4px;padding:4px 8px;font-family:{FONT};font-size:{FONT_MD}px;}}")
        th.addWidget(self._trigger_entry, stretch=1)
        iv.addWidget(trig_frame)

        # ── Desired Tags + Profile management ────────────────────────────────
        des_frame = QFrame(inner)
        des_frame.setStyleSheet(f"QFrame{{background:{CAR};border-radius:6px;}}")
        dv = QVBoxLayout(des_frame)
        dv.setContentsMargins(12, 8, 12, 8)
        dv.setSpacing(4)

        des_hdr = QWidget(des_frame)
        des_hdr.setStyleSheet("background:transparent;")
        dh = QHBoxLayout(des_hdr)
        dh.setContentsMargins(0, 0, 0, 0)
        dh.setSpacing(4)
        dh.addWidget(self._mk_label(des_hdr,
            "Desired Tags  (forced into every image)", bold=True))
        dh.addStretch()

        btn_prof_action = QPushButton("💾 Profile ▾", des_hdr)
        btn_prof_action.setFixedHeight(26)
        btn_prof_action.setStyleSheet(
            f"QPushButton{{background:{ACC};color:{PRI};border:none;border-radius:4px;"
            f"font-size:10px;padding:2px 8px;font-weight:bold;}}"
            f"QPushButton:hover{{background:#185FA5;}}")
        btn_prof_action.clicked.connect(self._show_profile_menu)
        dh.addWidget(btn_prof_action)

        btn_reset_prof = QPushButton("↺ Reset", des_hdr)
        btn_reset_prof.setFixedHeight(26)
        btn_reset_prof.setStyleSheet(
            f"QPushButton{{background:transparent;color:{RED};border:1px solid {RED};"
            f"border-radius:4px;font-size:10px;padding:2px 8px;}}"
            f"QPushButton:hover{{background:{RED};color:{PRI};}}")
        btn_reset_prof.clicked.connect(self._reset_tagger_profile)
        dh.addWidget(btn_reset_prof)

        btn_reload_master = QPushButton("↺ Reload Master", des_hdr)
        btn_reload_master.setFixedHeight(26)
        btn_reload_master.setStyleSheet(
            f"QPushButton{{background:transparent;color:{AMB};border:1px solid {AMB};"
            f"border-radius:4px;font-size:10px;padding:2px 8px;}}"
            f"QPushButton:hover{{background:{AMB};color:{PRI};}}")
        btn_reload_master.clicked.connect(self._reload_neg_master)
        dh.addWidget(btn_reload_master)

        dh.addWidget(self._mk_label(des_hdr, "Load:", color=MUT))
        self._neg_profile_combo = QComboBox(des_hdr)
        self._neg_profile_combo.addItems(self._get_profile_names())
        self._neg_profile_combo.setFixedWidth(130)
        self._neg_profile_combo.setStyleSheet(
            f"QComboBox{{background:{PAN};color:{PRI};border:1px solid {ACC};"
            f"border-radius:4px;font-size:10px;padding:1px 4px;}}"
            f"QComboBox QAbstractItemView{{background:{PAN};color:{PRI};"
            f"border:1px solid {ACC};selection-background-color:{ACC};}}")
        self._neg_profile_combo.currentTextChanged.connect(self._load_tagger_profile)
        dh.addWidget(self._neg_profile_combo)

        dv.addWidget(des_hdr)
        self._desired_edit = self._mk_text_edit(des_frame, height=80)
        dv.addWidget(self._desired_edit)
        iv.addWidget(des_frame)

        # ── Negative Tags ────────────────────────────────────────────────────
        neg_frame = QFrame(inner)
        neg_frame.setStyleSheet(f"QFrame{{background:{CAR};border-radius:6px;}}")
        nv = QVBoxLayout(neg_frame)
        nv.setContentsMargins(12, 8, 12, 8)
        nv.setSpacing(4)
        nv.addWidget(self._mk_label(neg_frame,
            "Negative Tags  (stripped from all output)", bold=True))
        self._negative_edit = self._mk_text_edit(neg_frame, height=120)
        nv.addWidget(self._negative_edit)
        iv.addWidget(neg_frame)

        # ── Engine row: Output Mode / Existing Files / Device / Batch Size ───
        eng_frame = QFrame(inner)
        eng_frame.setStyleSheet(f"QFrame{{background:{CAR};border-radius:6px;}}")
        eg = QGridLayout(eng_frame)
        eg.setContentsMargins(12, 8, 12, 8)
        eg.setSpacing(6)

        for i, txt in enumerate(["Output Mode", "Existing Files", "Device", "Batch Size"]):
            eg.addWidget(self._mk_label(eng_frame, txt, color=MUT), 0, i * 2,
                         alignment=Qt.AlignmentFlag.AlignLeft)

        self._output_combo = self._mk_combo(eng_frame, OUTPUT_MODES, width=140)
        self._output_combo.currentTextChanged.connect(self._on_output_change)
        eg.addWidget(self._output_combo, 1, 0, 1, 2)

        self._tagger_exist_combo = self._mk_combo(eng_frame, EXIST_MODES, width=100)
        eg.addWidget(self._tagger_exist_combo, 1, 2, 1, 2)

        self._device_combo = self._mk_combo(eng_frame, DEVICES, width=130)
        saved_device = self._cfg.get("tagger_device", "Auto-detect")
        idx = self._device_combo.findText(saved_device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)
        self._device_combo.currentTextChanged.connect(self._on_device_changed)
        eg.addWidget(self._device_combo, 1, 4, 1, 2)

        batch_row = QWidget(eng_frame)
        batch_row.setStyleSheet("background:transparent;")
        br = QHBoxLayout(batch_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(4)
        self._batch_slider = self._mk_slider(batch_row, lo=1, hi=16, val=4)
        self._batch_lbl = self._mk_label(batch_row, "4", bold=True)
        self._batch_slider.valueChanged.connect(
            lambda v: self._batch_lbl.setText(str(v)))
        br.addWidget(self._batch_slider)
        br.addWidget(self._batch_lbl)
        eg.addWidget(batch_row, 1, 6, 1, 2)

        iv.addWidget(eng_frame)

        # ── WD14 Settings panel ───────────────────────────────────────────────
        self._wd14_frame = QFrame(inner)
        self._wd14_frame.setStyleSheet(
            f"QFrame{{background:{CAR};border:none;border-radius:6px;}}")
        wv = QVBoxLayout(self._wd14_frame)
        wv.setContentsMargins(12, 8, 12, 8)
        wv.setSpacing(6)
        wv.addWidget(self._mk_label(self._wd14_frame,
            "WD14 Settings", bold=True, color=ACC))

        wd_inner = QWidget(self._wd14_frame)
        wd_inner.setStyleSheet("background:transparent;")
        wg = QGridLayout(wd_inner)
        wg.setContentsMargins(0, 0, 0, 0)
        wg.setSpacing(6)

        wg.addWidget(self._mk_label(wd_inner, "Threshold", bold=True, color=ACC), 0, 0)
        thresh_row = QWidget(wd_inner)
        thresh_row.setStyleSheet("background:transparent;")
        tr_h = QHBoxLayout(thresh_row)
        tr_h.setContentsMargins(0, 0, 0, 0)
        tr_h.setSpacing(4)
        self._thresh_slider = self._mk_slider(thresh_row, lo=10, hi=90, val=35)
        self._thresh_lbl = self._mk_label(thresh_row, "0.35", bold=True)
        self._thresh_slider.valueChanged.connect(
            lambda v: self._thresh_lbl.setText(f"{v/100:.2f}"))
        tr_h.addWidget(self._thresh_slider)
        tr_h.addWidget(self._thresh_lbl)
        wg.addWidget(thresh_row, 1, 0)

        wg.addWidget(self._mk_label(wd_inner, "Character Threshold", bold=True, color=ACC), 0, 3)
        char_row = QWidget(wd_inner)
        char_row.setStyleSheet("background:transparent;")
        cr_h = QHBoxLayout(char_row)
        cr_h.setContentsMargins(0, 0, 0, 0)
        cr_h.setSpacing(4)
        self._char_thresh_slider = self._mk_slider(char_row, lo=50, hi=100, val=85)
        self._char_thresh_lbl = self._mk_label(char_row, "0.85", bold=True)
        self._char_thresh_slider.valueChanged.connect(
            lambda v: self._char_thresh_lbl.setText(f"{v/100:.2f}"))
        cr_h.addWidget(self._char_thresh_slider)
        cr_h.addWidget(self._char_thresh_lbl)
        wg.addWidget(char_row, 1, 3)

        wg.addWidget(self._mk_label(wd_inner, "Ratings", color=MUT), 0, 6)
        self._rat_checks: dict[str, QCheckBox] = {}
        for j, r in enumerate(["General", "Sensitive", "Questionable", "Explicit"]):
            chk = QCheckBox(r, wd_inner)
            chk.setChecked(True)
            chk.setStyleSheet(
                f"QCheckBox{{color:{SEC};font-size:{FONT_MD}px;}}"
                f"QCheckBox::indicator{{width:16px;height:16px;border:2px solid {MUT};"
                f"border-radius:3px;background:{CAR};}}"
                f"QCheckBox::indicator:checked{{background:{ACC};border-color:{ACC};}}")
            self._rat_checks[r.lower()] = chk
            wg.addWidget(chk, 1, 6 + j)

        self._rm_underscore_chk = QCheckBox("Remove Underscores", wd_inner)
        self._rm_underscore_chk.setChecked(True)
        self._rm_underscore_chk.setStyleSheet(
            f"QCheckBox{{color:{SEC};font-size:{FONT_MD}px;}}"
            f"QCheckBox::indicator{{width:16px;height:16px;border:2px solid {MUT};"
            f"border-radius:3px;background:{CAR};}}"
            f"QCheckBox::indicator:checked{{background:{ACC};border-color:{ACC};}}")
        wg.addWidget(self._rm_underscore_chk, 1, 10)

        wv.addWidget(wd_inner)
        iv.addWidget(self._wd14_frame)

        # ── JoyCaption Settings panel ─────────────────────────────────────────
        self._joy_frame = QFrame(inner)
        self._joy_frame.setStyleSheet(
            f"QFrame{{background:{CAR};border:none;border-radius:6px;}}")
        jv = QVBoxLayout(self._joy_frame)
        jv.setContentsMargins(12, 8, 12, 8)
        jv.setSpacing(6)
        jv.addWidget(self._mk_label(self._joy_frame,
            "JoyCaption Alpha Two Settings", bold=True, color=ACC))

        joy_inner = QWidget(self._joy_frame)
        joy_inner.setStyleSheet("background:transparent;")
        jg = QGridLayout(joy_inner)
        jg.setContentsMargins(0, 0, 0, 0)
        jg.setSpacing(6)

        for i, txt in enumerate(["Caption Type", "Caption Length"]):
            jg.addWidget(self._mk_label(joy_inner, txt, color=MUT), 0, i * 2)

        self._joy_type_combo = self._mk_combo(joy_inner, JOY_CAP_TYPES, width=220)
        jg.addWidget(self._joy_type_combo, 1, 0)

        self._joy_len_combo = self._mk_combo(joy_inner, JOY_CAP_LENGTHS, width=140)
        jg.addWidget(self._joy_len_combo, 1, 2)

        jg.addWidget(self._mk_label(joy_inner,
            "Extra Instructions (optional)", color=MUT), 0, 6)
        self._extra_entry = self._mk_entry(joy_inner, width=260)
        jg.addWidget(self._extra_entry, 1, 6)

        _max_threads = max(1, os.cpu_count() or 4)
        jg.addWidget(self._mk_label(joy_inner, "CPU Threads", color=MUT), 2, 0)
        jt_row = QWidget(joy_inner)
        jt_row.setStyleSheet("background:transparent;")
        jt_h = QHBoxLayout(jt_row)
        jt_h.setContentsMargins(0, 0, 0, 0)
        jt_h.setSpacing(4)
        self._joy_threads_slider = self._mk_slider(
            jt_row, lo=1, hi=_max_threads, val=_max_threads)
        self._joy_threads_lbl = self._mk_label(jt_row, str(_max_threads), bold=True)
        self._joy_threads_slider.valueChanged.connect(
            lambda v: self._joy_threads_lbl.setText(str(v)))
        jt_h.addWidget(self._joy_threads_slider)
        jt_h.addWidget(self._joy_threads_lbl)
        jg.addWidget(jt_row, 3, 0)
        jg.addWidget(self._mk_label(
            joy_inner, f"(max {_max_threads})", color=MUT), 3, 2)

        jv.addWidget(joy_inner)
        self._joy_frame.setVisible(False)
        iv.addWidget(self._joy_frame)

        # ── Run row ──────────────────────────────────────────────────────────
        run_frame = QFrame(inner)
        run_frame.setStyleSheet(f"QFrame{{background:{CAR};border-radius:6px;}}")
        rh = QHBoxLayout(run_frame)
        rh.setContentsMargins(12, 8, 12, 8)
        rh.setSpacing(8)
        self._btn_run_tagger = QPushButton("▶  Run Tagger", run_frame)
        self._btn_run_tagger.setMinimumWidth(140)
        self._btn_run_tagger.setStyleSheet(
            f"QPushButton{{background:{ACC};color:{PRI};border:none;border-radius:6px;"
            f"font-size:{FONT_LG}px;font-family:{FONT};font-weight:bold;padding:6px 16px;}}"
            f"QPushButton:hover{{background:#185FA5;}}"
            f"QPushButton:disabled{{background:{MUT};color:{SEC};}}")
        self._btn_run_tagger.clicked.connect(self._run_tagger)
        rh.addWidget(self._btn_run_tagger)
        self._tag_progress = self._mk_progress(run_frame)
        rh.addWidget(self._tag_progress, stretch=1)
        self._tag_status = self._mk_label(run_frame, "Ready.", color=SEC)
        rh.addWidget(self._tag_status)
        iv.addWidget(run_frame)

        # ── Tagger log ────────────────────────────────────────────────────────
        self._tagger_log = QPlainTextEdit(inner)
        self._tagger_log.setReadOnly(True)
        self._tagger_log.setMinimumHeight(120)
        self._tagger_log.setMaximumHeight(220)
        self._tagger_log.setPlaceholderText("Tagger output will appear here…")
        self._tagger_log.setStyleSheet(
            f"QPlainTextEdit {{"
            f"  background:#0d0d0d; color:#c8c8c8;"
            f"  border:1px solid {MUT}; border-radius:4px;"
            f"  font-family:'Consolas','Courier New',monospace; font-size:11px;"
            f"  padding:4px 6px;"
            f"}}"
            f"QScrollBar:vertical {{ background:{CAR}; width:6px; }}"
            f"QScrollBar::handle:vertical {{ background:{MUT}; border-radius:3px; }}")
        iv.addWidget(self._tagger_log)

        iv.addStretch()

        # initial panel visibility — also seeds correct output mode list for saved device
        self._refresh_output_modes(self._device_combo.currentText())

    def _on_device_changed(self, val: str):
        self._cfg["tagger_device"] = val
        _save_cfg(self._cfg)
        self._refresh_output_modes(val)

    def _refresh_output_modes(self, device: str):
        """Swap Output Mode options to match the selected device."""
        use_gpu = "cuda" in device.lower() or device.lower() == "auto-detect"
        if use_gpu:
            modes = ["Tags", "Captions", "Captions (Fast)",
                     "Hybrid (GPU)", "Hybrid", "Hybrid (Fast)"]
        else:
            modes = ["Tags", "Captions", "Captions (Fast)",
                     "Hybrid", "Hybrid (Fast)"]
        cur = self._output_combo.currentText()
        self._output_combo.blockSignals(True)
        self._output_combo.clear()
        self._output_combo.addItems(modes)
        # keep current selection if still valid, else default to first item
        idx = self._output_combo.findText(cur)
        self._output_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._output_combo.blockSignals(False)
        self._on_output_change(self._output_combo.currentText())

    def _on_output_change(self, val: str):
        show_wd  = val in ("Tags", "Hybrid", "Hybrid (GPU)", "Hybrid (Fast)")
        show_joy = val in ("Captions", "Hybrid", "Hybrid (GPU)",
                           "Captions (Fast)", "Hybrid (Fast)")
        self._wd14_frame.setVisible(show_wd)
        self._joy_frame.setVisible(show_joy)

    def _show_profile_menu(self):
        btn = self.sender()
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu{{background:{PAN};color:{PRI};border:1px solid {MUT};}}"
            f"QMenu::item:selected{{background:{ACC};}}")
        menu.addAction("💾 Save Profile", self._save_tagger_profile)
        menu.addAction("🗑 Delete Profile", self._delete_tagger_profile)
        if btn:
            menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))
        else:
            menu.exec(QCursor.pos())

    def _get_profile_names(self) -> list:
        base   = ["Blank"] + LORA_TYPES
        custom = ([k for k in self._profiles.keys() if k not in LORA_TYPES]
                  if isinstance(self._profiles, dict) else [])
        return base + custom

    def _update_res_indicator(self, text: str = ""):
        txt = text or self._res_combo.currentText()
        if txt == "Manual":
            self._res_ratio_lbl.setText("Manual")
            for card in self._prep_card_widgets:
                if card.isVisible():
                    card.set_manual_mode(True)
        else:
            _, _, ratio = RES_MAP.get(txt, (1024, 1024, "1:1"))
            self._res_ratio_lbl.setText(ratio)
            tw, th, _ = RES_MAP.get(txt, (1024, 1024, "1:1"))
            for card in self._prep_card_widgets:
                if card.isVisible():
                    card.set_manual_mode(False)
                    card.set_resolution(tw, th)

    # ── Profile methods ───────────────────────────────────────────────────────
    def _collect_tagger_state(self) -> dict:
        return {
            "trigger_word":       self._trigger_entry.text().strip(),
            "desired_tags":       self._desired_edit.toPlainText().strip(),
            "negative_tags":      self._negative_edit.toPlainText().strip(),
            "output_mode":        self._output_combo.currentText(),
            "threshold":          self._thresh_slider.value() / 100,
            "char_threshold":     self._char_thresh_slider.value() / 100,
            "batch_size":         self._batch_slider.value(),
            "ratings":            {k: chk.isChecked()
                                   for k, chk in self._rat_checks.items()},
            "rm_underscores":     self._rm_underscore_chk.isChecked(),
            "caption_type":       self._joy_type_combo.currentText(),
            "caption_length":     self._joy_len_combo.currentText(),
            "extra_instructions": self._extra_entry.text().strip(),
        }

    def _load_tagger_profile(self, name: str = ""):
        if name == "Blank":
            self._negative_edit.setPlainText("")
            return
        p = (self._profiles.get(name, {})
             if isinstance(self._profiles, dict) else {})
        if not p:
            return
        self._trigger_entry.setText(p.get("trigger_word", ""))
        self._desired_edit.setPlainText(p.get("desired_tags", ""))
        self._negative_edit.setPlainText(p.get("negative_tags", ""))
        idx = self._output_combo.findText(p.get("output_mode", "Tags"))
        if idx >= 0:
            self._output_combo.setCurrentIndex(idx)
        self._thresh_slider.setValue(int(p.get("threshold", 0.35) * 100))
        self._char_thresh_slider.setValue(int(p.get("char_threshold", 0.85) * 100))
        self._batch_slider.setValue(int(p.get("batch_size", 4)))
        ratings = p.get("ratings", {})
        for k, chk in self._rat_checks.items():
            chk.setChecked(ratings.get(k, True))
        self._rm_underscore_chk.setChecked(p.get("rm_underscores", True))
        idx = self._joy_type_combo.findText(p.get("caption_type", "Descriptive"))
        if idx >= 0:
            self._joy_type_combo.setCurrentIndex(idx)
        idx = self._joy_len_combo.findText(p.get("caption_length", "any"))
        if idx >= 0:
            self._joy_len_combo.setCurrentIndex(idx)
        self._extra_entry.setText(p.get("extra_instructions", ""))

    def _save_tagger_profile(self):
        current = self._neg_profile_combo.currentText()
        if current in LORA_TYPES or current == "Blank":
            name, ok = QInputDialog.getText(
                self, "Save Profile As",
                f"'{current}' is a base preset — enter a new profile name:")
            if not ok or not name.strip():
                return
            name = name.strip()
            if name in LORA_TYPES:
                QMessageBox.warning(self, "Protected Name",
                    f"'{name}' is a protected base preset. Choose a different name.")
                return
        else:
            name = current
        if not isinstance(self._profiles, dict):
            self._profiles = {}
        self._profiles[name] = self._collect_tagger_state()
        _save_profiles(self._profiles)
        # refresh dropdown
        self._neg_profile_combo.blockSignals(True)
        self._neg_profile_combo.clear()
        self._neg_profile_combo.addItems(self._get_profile_names())
        idx = self._neg_profile_combo.findText(name)
        if idx >= 0:
            self._neg_profile_combo.setCurrentIndex(idx)
        self._neg_profile_combo.blockSignals(False)
        self._tag_status.setText(f"Profile '{name}' saved.")

    def _delete_tagger_profile(self):
        name = self._neg_profile_combo.currentText()
        if name in LORA_TYPES or name == "Blank":
            QMessageBox.warning(self, "Cannot Delete",
                f"'{name}' is a built-in preset and cannot be deleted.")
            return
        if QMessageBox.question(
                self, "Delete Profile",
                f"Permanently delete profile '{name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                ) != QMessageBox.StandardButton.Yes:
            return
        if isinstance(self._profiles, dict):
            self._profiles.pop(name, None)
        _save_profiles(self._profiles)
        self._neg_profile_combo.blockSignals(True)
        self._neg_profile_combo.clear()
        self._neg_profile_combo.addItems(self._get_profile_names())
        self._neg_profile_combo.setCurrentText("Blank")
        self._neg_profile_combo.blockSignals(False)
        self._tag_status.setText(f"Profile '{name}' deleted.")

    def _reset_tagger_profile(self):
        name = self._neg_profile_combo.currentText()
        if name == "Blank":
            self._negative_edit.setPlainText("")
            return
        if QMessageBox.question(
                self, "Reset Profile", f"Reset '{name}' to defaults?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                ) != QMessageBox.StandardButton.Yes:
            return
        if isinstance(self._profiles, dict) and name not in LORA_TYPES:
            self._profiles.pop(name, None)
            _save_profiles(self._profiles)
            self._neg_profile_combo.blockSignals(True)
            self._neg_profile_combo.clear()
            self._neg_profile_combo.addItems(self._get_profile_names())
            self._neg_profile_combo.setCurrentText("Blank")
            self._neg_profile_combo.blockSignals(False)
        self._negative_edit.setPlainText("")
        self._desired_edit.setPlainText("")

    def _reload_neg_master(self):
        name = self._neg_profile_combo.currentText()
        neg_key = NEG_KEYS.get(name, "")
        if not neg_key:
            QMessageBox.information(self, "No Master List",
                f"'{name}' has no master negative list.")
            return
        tags_master = _load_tags_master()
        master_text = tags_master.get(neg_key, "")
        if not master_text:
            QMessageBox.warning(self, "Not Found",
                f"Master list key '{neg_key}' not found in tags_data.json.")
            return
        current = self._negative_edit.toPlainText().strip()
        if current:
            if QMessageBox.question(
                    self, "Reload Master List",
                    "Are you sure? Your session edits will be lost.\n"
                    "The master list will be restored.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    ) != QMessageBox.StandardButton.Yes:
                return
        self._negative_edit.setPlainText(master_text)

    # ══════════════════════════════════════════════════════════════════════════
    #  EDITOR TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_editor(self):
        v = QVBoxLayout(self._editor_tab)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        self._ed_main_vbox = v

        # ── top controls ─────────────────────────────────────────────────────
        ctrl = QFrame(self._editor_tab)
        ctrl.setStyleSheet(f"background:{PAN};border-radius:4px;")
        ch = QHBoxLayout(ctrl)
        ch.setContentsMargins(8, 4, 8, 4)
        ch.setSpacing(8)

        ch.addWidget(self._mk_label(ctrl, "View:"))
        self._view_seg = SegmentedButton(ctrl, values=["Tag", "Caption", "Both"])
        self._view_seg.valueChanged.connect(self._on_label_type_change)
        self._view_seg.set("Tag")
        ch.addWidget(self._view_seg)

        ch.addWidget(self._mk_label(ctrl, "  Trigger:"))
        self._editor_trigger_entry = self._mk_entry(ctrl, width=150,
                                                     placeholder="trigger word")
        ch.addWidget(self._editor_trigger_entry)

        ch.addStretch()

        # Layout dropdown ("Horizontal" / "Vertical")
        self._layout_combo = QComboBox(ctrl)
        self._layout_combo.addItems(["Horizontal", "Vertical"])
        self._layout_combo.setCurrentText("Vertical")
        self._layout_combo.setFixedHeight(26)
        self._layout_combo.setFixedWidth(110)
        self._layout_combo.setStyleSheet(
            f"QComboBox{{background:{CAR};color:{PRI};border:1px solid {MUT};"
            f"border-radius:4px;padding:2px 6px;font-size:12px;}}"
            f"QComboBox::drop-down{{border:none;}}"
            f"QComboBox QAbstractItemView{{background:{CAR};color:{PRI};"
            f"selection-background-color:{ACC};}}")
        self._layout_combo.currentTextChanged.connect(self._on_editor_layout_changed)
        ch.addWidget(self._layout_combo)

        # "Don't Sort" toggle button (replaces checkbox)
        self._dont_sort_btn = QPushButton("Don't Sort", ctrl)
        self._dont_sort_btn.setCheckable(True)
        self._dont_sort_btn.setFixedHeight(26)
        self._dont_sort_btn.setStyleSheet(
            f"QPushButton{{background:{CAR};color:{SEC};border:1px solid {MUT};"
            f"border-radius:4px;font-size:12px;padding:0 10px;}}"
            f"QPushButton:checked{{background:{ACC};color:{PRI};border-color:{ACC};}}"
            f"QPushButton:hover:!checked{{border-color:{ACC};}}")
        ch.addWidget(self._dont_sort_btn)

        self._unsaved_lbl = self._mk_label(ctrl, "", color=AMB)
        ch.addWidget(self._unsaved_lbl)

        btn_save_all = self._mk_btn(ctrl, "Save All", color=GRN, min_w=90)
        btn_save_all.clicked.connect(self._apply_edits)
        ch.addWidget(btn_save_all)

        btn_undo = self._mk_btn(ctrl, "Undo", min_w=60)
        btn_undo.clicked.connect(self._undo)
        ch.addWidget(btn_undo)

        v.addWidget(ctrl)

        # ── shared widgets (parented later by _apply_editor_layout) ──────────
        self._tag_cloud = TagCloud(
            None,
            on_add=self._batch_add_from_cloud,
            on_remove=self._batch_remove_from_cloud_set,
            on_replace=self._batch_replace_prompt,
            on_filter=self._on_cloud_filter_set,
        )

        self._editor_scroll = QScrollArea()
        self._editor_scroll.setWidgetResizable(True)
        self._editor_scroll.setStyleSheet(f"QScrollArea{{border:none;background:{BG};}}")
        self._editor_inner = QWidget()
        self._editor_inner.setStyleSheet(f"background:{BG};")
        self._editor_grid  = QGridLayout(self._editor_inner)
        self._editor_grid.setSpacing(6)
        self._editor_scroll.setWidget(self._editor_inner)
        self._editor_scroll.verticalScrollBar().valueChanged.connect(
            self._on_editor_scroll)

        # Build initial layout (Horizontal)
        self._ed_body = None
        self._apply_editor_layout("Vertical")


    def _apply_editor_layout(self, mode: str):
        """Switch between Horizontal and Vertical editor layout."""
        # Remove old body from main vbox and detach shared widgets
        if self._ed_body is not None:
            self._ed_main_vbox.removeWidget(self._ed_body)
            self._tag_cloud.setParent(None)
            self._editor_scroll.setParent(None)
            self._ed_body.deleteLater()
            self._ed_body = None

        if mode == "Horizontal":
            body = QWidget(self._editor_tab)
            bv = QVBoxLayout(body)
            bv.setContentsMargins(0, 0, 0, 0)
            bv.setSpacing(6)
            self._tag_cloud.setMinimumHeight(100)
            self._tag_cloud.setMaximumHeight(200)
            bv.addWidget(self._tag_cloud)
            bv.addWidget(self._editor_scroll, stretch=1)
            self._tag_cloud.set_single_column(False)

        else:  # Vertical
            body = QSplitter(Qt.Orientation.Horizontal, self._editor_tab)

            left = QWidget(body)
            lv = QVBoxLayout(left)
            lv.setContentsMargins(0, 0, 0, 0)
            lv.setSpacing(0)
            lv.addWidget(self._editor_scroll)
            body.addWidget(left)

            self._tag_cloud.setMinimumHeight(0)
            self._tag_cloud.setMaximumHeight(16777215)
            self._tag_cloud.setMinimumWidth(100)
            body.addWidget(self._tag_cloud)
            body.setSizes([900, 170])
            body.setCollapsible(1, True)
            self._tag_cloud.set_single_column(True)

        self._ed_body = body
        # Insert at index 1: after ctrl bar, before pagination nav
        self._ed_main_vbox.insertWidget(1, body, stretch=1)

    def _on_editor_layout_changed(self, mode: str):
        self._apply_editor_layout(mode)
        self._render_editor_page()

    def _render_editor_page(self):
        """Reset the editor grid and load the first batch (infinite-scroll reset)."""
        # hide all pooled cards and clear grid
        for card in self._editor_cards:
            card.hide()
        while self._editor_grid.count():
            item = self._editor_grid.takeAt(0)
            if item.widget():
                item.widget().hide()

        if not self._images:
            self._editor_visible = []
            self._editor_loaded  = 0
            return

        highlighted = self._tag_cloud.selected
        dont_sort   = self._dont_sort_btn.isChecked()
        if highlighted and not dont_sort:
            visible = [p for p in self._images
                       if highlighted.issubset(
                           {t.strip().lower() for t in _read_tags(p)})]
        else:
            visible = self._images

        self._editor_visible = visible
        self._editor_loaded  = 0
        self._append_editor_cards(0, min(EDITOR_BATCH, len(visible)))

    def _append_editor_cards(self, start: int, end: int):
        """Append editor cards for _editor_visible[start:end] to the grid."""
        cols         = 5 if self._layout_combo.currentText() == "Vertical" else COLS
        highlighted  = self._tag_cloud.selected
        view_mode    = self._view_seg.get()
        trigger_word = self._editor_trigger_entry.text().strip()
        visible      = self._editor_visible

        # grow pool as needed
        while len(self._editor_cards) < end:
            card = EditorCard(self._editor_inner, "",
                              on_tags_changed=self._on_tags_changed,
                              on_deleted=self._on_image_deleted)
            self._editor_cards.append(card)

        for i in range(start, end):
            card = self._editor_cards[i]
            card.recycle(visible[i],
                         highlighted_tags=highlighted,
                         view_mode=view_mode,
                         trigger_word=trigger_word)
            card.show()
            row, col = divmod(i, cols)
            self._editor_grid.addWidget(card, row, col)

        self._editor_loaded = end
        total = len(visible)
        if end < total:
            self._tag_status.setText(
                f"{end} of {total} shown — scroll down to load more")
        else:
            self._tag_status.setText(f"{total} images ready.")

        QTimer.singleShot(100, self._prefetch_next_thumbs)

    def _on_editor_scroll(self, value: int):
        """Load next batch when scrolled to 85% of current content."""
        if self._editor_loaded >= len(self._editor_visible):
            return
        sb = self._editor_scroll.verticalScrollBar()
        if sb.maximum() > 0 and value >= sb.maximum() * 0.85:
            nxt = min(self._editor_loaded + EDITOR_BATCH, len(self._editor_visible))
            self._append_editor_cards(self._editor_loaded, nxt)

    def _on_cloud_filter_set(self, selected: set):
        """Called by TagCloud when selection changes — filter and re-render."""
        self._render_editor_page()

    def _on_tags_changed(self, img_path: str):
        self._tags_changed.add(img_path)
        self._update_unsaved_label()
        self._flush_timer.start(2000)

    def _flush_tags_changed(self):
        if not self._tags_changed:
            return
        self._push_undo()
        dont_sort = self._dont_sort_btn.isChecked()
        for card in self._editor_cards:
            if card.isVisible() and card.img_path in self._tags_changed:
                all_items = _read_tags(card.img_path)
                tags, caps = _split_tags_captions(all_items)
                if not dont_sort:
                    tags = sorted(tags)
                _write_tags(card.img_path, tags + caps)
                self._tags_changed.discard(card.img_path)
        _flush_tag_cache()
        self._update_unsaved_label()
        self._rebuild_tag_freq()

    def _on_label_type_change(self, val: str):
        self._render_editor_page()

    def _on_image_deleted(self, img_path: str):
        if img_path in self._images:
            self._images.remove(img_path)
        self._tags_changed.discard(img_path)
        self._rebuild_tag_freq()
        self._render_editor_page()

    def _on_prepare_image_deleted(self, img_path: str):
        if img_path in self._images:
            self._images.remove(img_path)
        self._crop_offsets.pop(img_path, None)
        self._tags_changed.discard(img_path)
        self._rebuild_tag_freq()
        self._render_prepare_page()

    def _on_dont_sort_toggle(self, checked: bool):
        pass  # applied at save time

    def _apply_edits(self):
        self._push_undo()
        dont_sort = self._dont_sort_btn.isChecked()
        trig = self._trigger_entry.text().strip().lower()
        for card in self._editor_cards:
            if card.isVisible():
                all_items = _read_tags(card.img_path)
                tags, caps = _split_tags_captions(all_items)
                # Split trigger from rest
                trigger_tags = [t for t in tags if t.strip().lower() == trig] if trig else []
                other_tags   = [t for t in tags if t.strip().lower() != trig] if trig else tags
                if not dont_sort:
                    other_tags = sorted(other_tags)
                _write_tags(card.img_path, trigger_tags + other_tags + caps)
        _flush_tag_cache()
        self._tags_changed.clear()
        self._update_unsaved_label()
        self._rebuild_tag_freq()
        self._show_save_toast()

    def _show_save_toast(self):
        toast = QLabel("✓  Saved!", self._editor_tab)
        toast.setStyleSheet(
            f"background:{GRN};color:{PRI};font-family:{FONT};"
            f"font-size:{FONT_MD}px;font-weight:bold;"
            f"border-radius:8px;padding:10px 24px;")
        toast.adjustSize()
        # centre over the editor tab
        pw, ph = self._editor_tab.width(), self._editor_tab.height()
        toast.move((pw - toast.width()) // 2, (ph - toast.height()) // 2)
        toast.raise_()
        toast.show()
        QTimer.singleShot(1500, toast.deleteLater)

    def _update_unsaved_label(self):
        n = len(self._tags_changed)
        self._unsaved_lbl.setText(f"({n} unsaved)" if n else "")

    def _prewarm_thumb_cache(self):
        for p in self._images[:40]:
            _load_or_cache_thumb(p)

    def _prefetch_next_thumbs(self):
        nxt = self._editor_loaded
        for p in self._editor_visible[nxt:nxt + EDITOR_BATCH]:
            _load_or_cache_thumb(p)

    # ══════════════════════════════════════════════════════════════════════════
    #  BATCH TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_batch(self):
        v = QVBoxLayout(self._batch_tab)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        # ── find missing ──────────────────────────────────────────────────────
        sec1 = QFrame(self._batch_tab)
        sec1.setStyleSheet(f"background:{PAN};border-radius:4px;")
        s1h = QHBoxLayout(sec1)
        s1h.setContentsMargins(8, 6, 8, 6)
        s1h.setSpacing(8)
        s1h.addWidget(self._mk_label(sec1, "Missing .txt files:", bold=True))
        btn_find = self._mk_btn(sec1, "Find && Create", min_w=120)
        btn_find.clicked.connect(self._find_missing)
        s1h.addWidget(btn_find)
        s1h.addStretch()
        v.addWidget(sec1)

        # ── sort tags ─────────────────────────────────────────────────────────
        sec2 = QFrame(self._batch_tab)
        sec2.setStyleSheet(f"background:{PAN};border-radius:4px;")
        s2h = QHBoxLayout(sec2)
        s2h.setContentsMargins(8, 6, 8, 6)
        s2h.setSpacing(8)
        s2h.addWidget(self._mk_label(sec2, "Randomize tag order:", bold=True))
        btn_sort = self._mk_btn(sec2, "Shuffle Tags", min_w=100)
        btn_sort.clicked.connect(self._sort_tags)
        s2h.addWidget(btn_sort)
        s2h.addStretch()
        v.addWidget(sec2)

        # ── add tags ──────────────────────────────────────────────────────────
        sec3 = QFrame(self._batch_tab)
        sec3.setStyleSheet(f"background:{PAN};border-radius:4px;")
        s3v = QVBoxLayout(sec3)
        s3v.setContentsMargins(8, 6, 8, 6)
        s3v.setSpacing(6)
        s3h_row = QHBoxLayout()
        s3h_row.addWidget(self._mk_label(sec3, "Batch add tags:", bold=True))
        self._batch_add_edit = self._mk_text_edit(sec3, height=50)
        btn_batch_add = self._mk_btn(sec3, "Add to All", min_w=100)
        btn_batch_add.clicked.connect(self._batch_add_custom_to_selection)
        s3h_row.addWidget(btn_batch_add)
        s3h_row.addStretch()
        s3v.addLayout(s3h_row)
        s3v.addWidget(self._batch_add_edit)
        v.addWidget(sec3)

        # ── remove tags ───────────────────────────────────────────────────────
        sec4 = QFrame(self._batch_tab)
        sec4.setStyleSheet(f"background:{PAN};border-radius:4px;")
        s4v = QVBoxLayout(sec4)
        s4v.setContentsMargins(8, 6, 8, 6)
        s4v.setSpacing(6)
        s4h_row = QHBoxLayout()
        s4h_row.addWidget(self._mk_label(sec4, "Batch remove tags:", bold=True))
        self._batch_rm_edit = self._mk_text_edit(sec4, height=50)
        btn_batch_rm = self._mk_btn(sec4, "Remove from All", color=RED, min_w=130)
        btn_batch_rm.clicked.connect(
            lambda: self._batch_remove_from_cloud_set(
                {t.strip() for t in self._batch_rm_edit.toPlainText().split(",") if t.strip()}))
        s4h_row.addWidget(btn_batch_rm)
        s4h_row.addStretch()
        s4v.addLayout(s4h_row)
        s4v.addWidget(self._batch_rm_edit)
        v.addWidget(sec4)

        # ── freq report + export ──────────────────────────────────────────────
        sec6 = QFrame(self._batch_tab)
        sec6.setStyleSheet(f"background:{PAN};border-radius:4px;")
        s6h = QHBoxLayout(sec6)
        s6h.setContentsMargins(8, 6, 8, 6)
        s6h.setSpacing(8)
        s6h.addWidget(self._mk_label(sec6, "Tag frequency:", bold=True))
        btn_list = self._mk_btn(sec6, "Show List", min_w=100)
        btn_list.clicked.connect(self._show_tag_list)
        btn_csv = self._mk_btn(sec6, "Export CSV", min_w=100)
        btn_csv.clicked.connect(self._export_freq_csv)
        btn_zip = self._mk_btn(sec6, "Save Final ZIP", min_w=120)
        btn_zip.clicked.connect(self._save_zip)
        s6h.addWidget(btn_list)
        s6h.addWidget(btn_csv)
        s6h.addWidget(btn_zip)
        s6h.addStretch()
        v.addWidget(sec6)

        # font size slider for log
        font_row = QWidget(self._batch_tab)
        font_row.setStyleSheet("background:transparent;")
        fh = QHBoxLayout(font_row)
        fh.setContentsMargins(0, 0, 0, 0)
        fh.setSpacing(6)
        fh.addWidget(self._mk_label(font_row, "A", color=MUT))
        self._batch_font_slider = self._mk_slider(font_row, lo=8, hi=20, val=11)
        self._batch_font_slider.setFixedWidth(120)
        self._batch_font_slider.valueChanged.connect(self._on_batch_font_change)
        fh.addWidget(self._batch_font_slider)
        fh.addWidget(self._mk_label(font_row, "A", bold=True))
        fh.addStretch()
        v.addWidget(font_row)

        # batch log
        self._batch_log_box = self._mk_text_edit(
            self._batch_tab, height=160, read_only=True)
        v.addWidget(self._batch_log_box)

        self._batch_progress = self._mk_progress(self._batch_tab)
        self._batch_status   = self._mk_label(self._batch_tab, "", color=MUT)
        v.addWidget(self._batch_progress)
        v.addWidget(self._batch_status)

    def _batch_log(self, msg: str):
        self._batch_log_box.appendPlainText(msg)
        sb = self._batch_log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_batch_font_change(self, pt: int):
        f = self._batch_log_box.font()
        f.setPointSize(pt)
        self._batch_log_box.setFont(f)

    # ══════════════════════════════════════════════════════════════════════════
    #  HELP TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_help(self):
        v = QVBoxLayout(self._help_tab)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(0)
        self._help_text = QPlainTextEdit(self._help_tab)
        self._help_text.setReadOnly(True)
        self._help_text.setStyleSheet(
            f"QPlainTextEdit{{background:{PAN};color:{SEC};border:none;padding:12px;}}")
        v.addWidget(self._help_text)
        self._render_help()

    def _render_help(self):
        lines = []
        for section, items in HELP.items():
            lines.append(f"\n{'═'*60}")
            lines.append(f"  {section.upper()}")
            lines.append(f"{'═'*60}")
            for title, desc in items:
                lines.append(f"\n  {title}")
                lines.append(f"    {desc}")
        self._help_text.setPlainText("\n".join(lines).strip())

    # ══════════════════════════════════════════════════════════════════════════
    #  DATASET METHODS
    # ══════════════════════════════════════════════════════════════════════════
    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", self._dataset_folder)
        if path:
            self._dataset_folder = path
            self._load_dataset()

    def _load_dataset(self):
        path = self._dataset_folder
        if not path or not os.path.isdir(path):
            return

        self._dataset_folder = path
        self._cfg["last_dataset"] = path
        _save_cfg(self._cfg)

        self._images = _get_images(path)

        if not self._images:
            QMessageBox.warning(self, "No images",
                                f"No supported images found in:\n{path}\n\n"
                                "Supported formats: JPG, PNG, WEBP, BMP, TIFF, GIF, AVIF")
            return

        _init_tag_cache(self._images)
        self._rebuild_tag_freq()

        total = len(self._images)
        self._prep_status.setText(
            f"Loaded {total} images from {os.path.basename(path)}")
        self._tag_status.setText(f"{total} images ready.")

        self._render_prepare_page()
        self._render_editor_page()
        QTimer.singleShot(300, self._prewarm_thumb_cache)

    def _rebuild_tag_freq(self):
        freq: dict[str, int] = {}
        for img in self._images:
            tags, _ = _split_tags_captions(_read_tags(img))
            for tag in tags:
                freq[tag] = freq.get(tag, 0) + 1
        self._tag_freq = freq
        if hasattr(self, "_tag_cloud"):
            self._tag_cloud.load(freq, len(self._images))

    # ══════════════════════════════════════════════════════════════════════════
    #  UNDO
    # ══════════════════════════════════════════════════════════════════════════
    def _push_undo(self):
        snapshot: dict[str, str] = {}
        for img in self._images:
            txt = _txt_path(img)
            if os.path.isfile(txt):
                try:
                    with open(txt, "r", encoding="utf-8") as f:
                        snapshot[txt] = f.read()
                except Exception:
                    pass
        if snapshot:
            self._undo_stack.append(snapshot)
            if len(self._undo_stack) > 20:
                self._undo_stack.pop(0)

    def _undo(self):
        if not self._undo_stack:
            QMessageBox.information(self, "Undo", "Nothing to undo.")
            return
        snapshot = self._undo_stack.pop()
        for txt_p, content in snapshot.items():
            try:
                with open(txt_p, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception:
                pass
        # Discard any dirty cache entries — don't let them overwrite restored files
        if _tag_dirty is not None:
            _tag_dirty.clear()
        _init_tag_cache(self._images)
        self._tags_changed.clear()
        self._update_unsaved_label()
        self._rebuild_tag_freq()
        self._render_editor_page()

    # ══════════════════════════════════════════════════════════════════════════
    #  CONVERT FORMATS
    # ══════════════════════════════════════════════════════════════════════════
    def _convert_formats(self):
        if not self._images:
            QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return
        fmt, ok = QInputDialog.getItem(self, "Convert Formats",
                                        "Convert all images to:",
                                        CONV_FMTS, 0, False)
        if not ok:
            return
        quality = 100
        converted = 0
        errors    = 0
        ext = ".png" if fmt == "PNG" else ".jpg"
        for img_path in list(self._images):
            try:
                img = Image.open(img_path).convert("RGB")
                folder = os.path.dirname(img_path)
                base   = os.path.splitext(os.path.basename(img_path))[0]
                out    = os.path.join(folder, base + ext)

                # If target already exists and is a different source file,
                # find a free numbered name (photo-1.png, photo-2.png, …)
                if out != img_path and os.path.exists(out):
                    counter = 1
                    while True:
                        candidate = os.path.join(folder, f"{base}-{counter}{ext}")
                        if not os.path.exists(candidate):
                            out = candidate
                            break
                        counter += 1

                if fmt == "PNG":
                    img.save(out, "PNG")
                else:
                    img.save(out, "JPEG", quality=quality)

                if out != img_path and os.path.exists(img_path):
                    old_txt = _txt_path(img_path)
                    new_txt = _txt_path(out)
                    if os.path.isfile(old_txt) and old_txt != new_txt:
                        shutil.copy2(old_txt, new_txt)
                    os.remove(img_path)
                converted += 1
            except Exception as e:
                errors += 1
                self._batch_log(f"Convert error: {os.path.basename(img_path)}: {e}")
        self._load_dataset()
        QMessageBox.information(self, "Convert Complete",
                                f"Converted {converted} images. Errors: {errors}.")

    # ══════════════════════════════════════════════════════════════════════════
    #  PROCESS IMAGES
    # ══════════════════════════════════════════════════════════════════════════
    def _process_images(self):
        if not self._images:
            QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return

        res_key    = self._res_combo.currentText()
        is_manual  = (res_key == "Manual")
        tw, th, _  = RES_MAP.get(res_key, (1024, 1024, "1:1"))
        scale_mode = self._scale_combo.currentText()
        quality    = 100
        use_esr    = scale_mode.startswith("RealESRGAN")
        esr_anime  = scale_mode == "RealESRGAN 4x (Anime)"

        # snapshot all card states before processing
        for card in self._prep_card_widgets:
            ox, oy = card.get_offset()
            tw_c, th_c = card.get_resolution()
            self._crop_offsets[card.img_path] = (ox, oy, card.get_zoom(), tw_c, th_c)

        resample = Image.LANCZOS

        total  = len(self._images)
        done   = 0
        errors = 0

        self._prep_status.setText(f"Processing 0/{total}…")
        self._set_progress(self._prep_progress, 0.0)
        QApplication.processEvents()

        for img_path in list(self._images):
            try:
                img = Image.open(img_path).convert("RGB")
                iw, ih = img.size

                if use_esr:
                    try:
                        import cv2
                        import numpy as _np
                        from basicsr.archs.rrdbnet_arch import RRDBNet
                        from realesrgan import RealESRGANer
                        cache_key = "_realesrgan_anime" if esr_anime else "_realesrgan_photo"
                        if not hasattr(self, cache_key) or getattr(self, cache_key) is None:
                            if esr_anime:
                                fname    = "RealESRGAN_x4plus_anime_6B.pth"
                                n_blocks = 6
                            else:
                                fname    = "RealESRGAN_x4plus.pth"
                                n_blocks = 23
                            if not self._ensure_esr_model(fname):
                                break   # user declined download — abort processing
                            mdl_path = os.path.join(_HERE, "models", fname)
                            _rrdb = RRDBNet(num_in_ch=3, num_out_ch=3,
                                            num_feat=64, num_block=n_blocks,
                                            num_grow_ch=32, scale=4)
                            setattr(self, cache_key, RealESRGANer(
                                scale=4, model_path=mdl_path, model=_rrdb,
                                tile=0, tile_pad=10, pre_pad=0, half=False))
                        esr_model = getattr(self, cache_key)
                        arr     = _np.array(img)
                        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        out_bgr, _ = esr_model.enhance(arr_bgr, outscale=4)
                        img = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
                        iw, ih = img.size
                    except Exception as esr_err:
                        self._batch_log(f"ESR error on {os.path.basename(img_path)}: {esr_err}")

                # get per-image offset/zoom/resolution from saved state
                state     = self._crop_offsets.get(img_path, (0.5, 0.5, 1.0, tw, th))
                ox        = state[0]
                oy        = state[1]
                card_zoom = state[2]
                img_tw    = state[3] if is_manual else tw
                img_th    = state[4] if is_manual else th

                # scale so image fills target rect, then apply zoom
                scale = max(img_tw / iw, img_th / ih) * card_zoom
                sw = max(img_tw, int(iw * scale))
                sh = max(img_th, int(ih * scale))
                img = img.resize((sw, sh), resample)

                # offset-aware crop (ox/oy are 0..1 pan fractions)
                cox = int(ox * max(0, sw - img_tw))
                coy = int(oy * max(0, sh - img_th))
                img = img.crop((cox, coy, cox + img_tw, coy + img_th))

                ext = os.path.splitext(img_path)[1].lower()
                if ext in (".jpg", ".jpeg"):
                    img.save(img_path, "JPEG", quality=quality)
                elif ext == ".webp":
                    img.save(img_path, "WEBP", quality=quality)
                else:
                    img.save(img_path, "PNG")

                done += 1
            except Exception as e:
                errors += 1
                self._batch_log(
                    f"Error: {os.path.basename(img_path)}: {e}")

            self._set_progress(self._prep_progress, done / total)
            self._prep_status.setText(f"Processing {done}/{total}…")
            QApplication.processEvents()

        self._crop_offsets.clear()
        _flush_tag_cache()
        self._render_prepare_page()
        self._prep_status.setText(f"Done. {done} processed, {errors} errors.")
        self._set_progress(self._prep_progress, 1.0 if not errors else 0.5)

    # ══════════════════════════════════════════════════════════════════════════
    #  BACKUP / RESTORE
    # ══════════════════════════════════════════════════════════════════════════
    def _show_backup_menu(self):
        btn = self.sender()
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu{{background:{PAN};color:{SEC};border:1px solid {MUT};}}"
            f"QMenu::item:selected{{background:{ACC};}}")
        menu.addAction("Backup Dataset → ZIP", self._backup_dataset)
        menu.addAction("Restore Images from ZIP…", self._restore_images_zip)
        if btn:
            menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))
        else:
            menu.exec(self.cursor().pos())

    def _clear_thumb_cache(self):
        if not self._dataset_folder:
            QMessageBox.warning(self, "No Dataset", "Load a dataset first.")
            return
        dataset_name = os.path.basename(self._dataset_folder)
        cache_dir = Path(_TEMP_DIR) / dataset_name
        if cache_dir.exists():
            shutil.rmtree(str(cache_dir))
            QMessageBox.information(self, "Cache Cleared",
                                    f"Thumbnail cache for '{dataset_name}' cleared.")
        else:
            QMessageBox.information(self, "Cache Cleared", "No cache found for this dataset.")

    def _backup_dataset(self):
        if not self._images:
            QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"backup_{os.path.basename(self._dataset_folder)}_{ts}.zip"
        default_path = os.path.join(os.path.dirname(self._dataset_folder), default_name)
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Backup ZIP", default_path, "ZIP files (*.zip)")
        if not out_path:
            return
        total = len(self._images)
        done  = 0
        try:
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for img_path in self._images:
                    zf.write(img_path, os.path.basename(img_path))
                    txt = _txt_path(img_path)
                    if os.path.isfile(txt):
                        zf.write(txt, os.path.basename(txt))
                    done += 1
                    self._set_progress(self._prep_progress, done / total)
                    QApplication.processEvents()
            self._prep_status.setText(
                f"Backup saved: {os.path.basename(out_path)}")
            self._set_progress(self._prep_progress, 1.0)
        except Exception as e:
            QMessageBox.critical(self, "Backup Error", str(e))

    def _restore_images_zip(self):
        zip_path, _ = QFileDialog.getOpenFileName(
            self, "Select Backup ZIP", "", "ZIP files (*.zip)")
        if not zip_path:
            return
        dest = QFileDialog.getExistingDirectory(self, "Restore to Folder")
        if not dest:
            return
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest)
            QMessageBox.information(self, "Restore Complete",
                                    f"Restored to {dest}")
        except Exception as e:
            QMessageBox.critical(self, "Restore Error", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    #  TAGGER  (threaded)
    # ══════════════════════════════════════════════════════════════════════════
    def _run_tagger(self):
        if not self._images:
            QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return
        if self._tagger_thread_obj and self._tagger_thread_obj.isRunning():
            QMessageBox.information(self, "Busy", "Tagger is already running.")
            return

        # save trigger word to config
        self._cfg["trigger_word"] = self._trigger_entry.text().strip()
        _save_cfg(self._cfg)

        self._btn_run_tagger.setEnabled(False)
        self._tag_status.setText("Starting…")
        self._set_progress(self._tag_progress, 0.0)
        self._tagger_log.clear()
        self._tagger_log.appendPlainText(
            f"Device: {self._device_combo.currentText()}  |  "
            f"Mode: {self._output_combo.currentText()}  |  "
            f"Images: {len(self._images)}")

        output_mode    = self._output_combo.currentText()
        # engine is derived from output mode
        if output_mode == "Captions":
            engine = "JoyCaption"
        elif output_mode in ("Captions (Fast)", "Hybrid (Fast)"):
            engine = "WD14 Single"   # single EVA02 pass for Hybrid (Fast)
        elif output_mode == "Hybrid (GPU)":
            engine = "WD14 GPU"
        else:
            engine = "WD14 Ensemble"
        # Hybrid (Fast) uses EVA02-Large (index 1) as its single WD14 model
        if output_mode == "Hybrid (Fast)":
            model1 = list(WD14_REPOS.keys())[1]   # "WD EVA02-Large"
        else:
            model1 = list(WD14_REPOS.keys())[0]
        model2         = list(WD14_REPOS.keys())[1]
        threshold      = self._thresh_slider.value() / 100
        char_thresh    = self._char_thresh_slider.value() / 100
        device         = self._device_combo.currentText()
        exist_mode     = self._tagger_exist_combo.currentText()
        trigger        = self._trigger_entry.text().strip()
        desired        = self._desired_edit.toPlainText().strip()
        negative       = self._negative_edit.toPlainText().strip()
        batch_size     = self._batch_slider.value()
        ratings        = {k: chk.isChecked() for k, chk in self._rat_checks.items()}
        rm_underscores = self._rm_underscore_chk.isChecked()
        extra          = self._extra_entry.text().strip()
        joy_threads    = self._joy_threads_slider.value()

        self._tagger_thread_obj = _TaggerThread(
            images         = list(self._images),
            engine         = engine,
            model1         = model1,
            model2         = model2,
            threshold      = threshold,
            char_thresh    = char_thresh,
            device         = device,
            exist_mode     = exist_mode,
            output_mode    = output_mode,
            trigger        = trigger,
            desired        = desired,
            negative       = negative,
            batch_size     = batch_size,
            ratings        = ratings,
            rm_underscores = rm_underscores,
            joy_path       = os.path.join(_HERE, "models", "joycaption"),
            moondream_path = os.path.join(_HERE, "models", "moondream") if os.path.isdir(os.path.join(_HERE, "models", "moondream")) else MOONDREAM_MODEL_ID,
            joy_type       = self._joy_type_combo.currentText(),
            joy_length     = self._joy_len_combo.currentText(),
            extra          = extra,
            joy_threads    = joy_threads,
            app_dir        = _HERE,
        )
        self._tagger_thread_obj.progress.connect(self._on_tagger_progress)
        self._tagger_thread_obj.done.connect(self._on_tagger_done)
        self._tagger_thread_obj.start()

    def _on_tagger_progress(self, value: float, status: str):
        self._set_progress(self._tag_progress, value)
        self._tag_status.setText(status)
        if status and status != self._tagger_log.toPlainText().split("\n")[-1]:
            self._tagger_log.appendPlainText(status)

    def _on_tagger_done(self, ok: bool, msg: str):
        self._btn_run_tagger.setEnabled(True)
        self._tag_status.setText(msg)
        self._set_progress(self._tag_progress, 1.0 if ok else 0.0)
        marker = "✓ " if ok else "✗ "
        self._tagger_log.appendPlainText(f"\n{marker}{msg}")
        # Clear dirty set BEFORE reinit — prevents stale cache from overwriting
        # what the tagger just wrote to disk.
        if _tag_dirty is not None:
            _tag_dirty.clear()
        _init_tag_cache(self._images)   # reload fresh from disk
        self._rebuild_tag_freq()
        self._render_editor_page()
        if not ok:
            QMessageBox.critical(self, "Tagger Error", msg)

    # ══════════════════════════════════════════════════════════════════════════
    #  BATCH OPERATIONS
    # ══════════════════════════════════════════════════════════════════════════
    def _batch_add_from_cloud(self, tags):
        """Add a set of tags (from TagCloud action) to all images."""
        tags = list(tags) if not isinstance(tags, list) else tags
        if not tags:
            return
        self._push_undo()
        count = 0
        for img in self._images:
            existing = _read_tags(img)
            existing_low = {t.lower() for t in existing}
            new_tags = [t for t in tags if t.lower() not in existing_low]
            if new_tags:
                _write_tags(img, existing + new_tags)
                count += 1
        _flush_tag_cache()
        self._rebuild_tag_freq()
        self._render_editor_page()
        self._batch_log(f"Added {tags} to {count} files.")

    def _batch_add_custom_to_selection(self):
        raw = self._batch_add_edit.toPlainText().strip()
        if not raw:
            return
        tags = [t.strip() for t in raw.split(",") if t.strip()]
        if not tags:
            return
        self._batch_add_from_cloud(tags)

    def _batch_remove_from_cloud_set(self, tags):
        """Remove a set of tags (from TagCloud action or batch rm box)."""
        tags_set = set(tags) if not isinstance(tags, set) else tags
        if not tags_set:
            return
        self._push_undo()
        count = 0
        for img in self._images:
            existing = _read_tags(img)
            cleaned  = [t for t in existing if t not in tags_set]
            if len(cleaned) != len(existing):
                _write_tags(img, cleaned)
                count += 1
        _flush_tag_cache()
        self._rebuild_tag_freq()
        self._tag_cloud._deselect_all()
        self._render_editor_page()
        self._batch_log(f"Removed {tags_set} from {count} files.")

    def _batch_replace_prompt(self, tags):
        """Prompt user for replacement tag(s) then replace across dataset."""
        tags_list = list(tags) if not isinstance(tags, list) else tags
        if not tags_list:
            return
        n = len(tags_list)
        label = (f"Replacing {n} tag(s) across all images.\nNew tag:"
                 if n > 1 else
                 f"Replacing: '{tags_list[0]}'\nNew tag:")
        replacement, ok = QInputDialog.getText(self, "Replace Tags", label)
        if ok and replacement.strip():
            for src in tags_list:
                self._batch_replace_from_cloud(src, replacement.strip())

    def _batch_replace_from_cloud(self, src: str, dst: str):
        src = src.strip()
        dst = dst.strip()
        if not src:
            return
        self._push_undo()
        count = 0
        for img in self._images:
            existing = _read_tags(img)
            # Replace src with dst, then deduplicate preserving first occurrence
            replaced = [dst if t == src else t for t in existing]
            seen = {}
            deduped = [seen.setdefault(t, t) for t in replaced if t not in seen]
            if deduped != existing:
                _write_tags(img, deduped)
                count += 1
        _flush_tag_cache()
        self._rebuild_tag_freq()
        self._render_editor_page()
        self._batch_log(f"Replaced '{src}' → '{dst}' in {count} files.")

    def _find_missing(self):
        if not self._images:
            QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return
        created = 0
        for img in self._images:
            txt = _txt_path(img)
            if not os.path.isfile(txt):
                try:
                    open(txt, "w").close()
                    created += 1
                except Exception:
                    pass
        self._batch_log(f"Created {created} missing .txt files.")
        self._rebuild_tag_freq()

    def _sort_tags(self):
        if not self._images:
            QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return
        self._push_undo()
        trigger = self._trigger_entry.text().strip().lower()
        count = 0
        for img in self._images:
            tags, caps = _split_tags_captions(_read_tags(img))
            if tags:
                rest = [t for t in tags if t.lower() != trigger]
                random.shuffle(rest)
                if trigger:
                    trig_actual = next((t for t in tags if t.lower() == trigger), trigger)
                    rest = [trig_actual] + rest
                _write_tags(img, rest + caps)
                count += 1
        _flush_tag_cache()
        self._render_editor_page()
        self._batch_log(f"Shuffled tags in {count} files.")

    def _show_tag_list(self):
        if not self._tag_freq:
            QMessageBox.warning(self, "No data", "Load a dataset first.")
            return

        trigger = self._trigger_entry.text().strip().lower()
        sorted_tags = sorted(self._tag_freq.items(), key=lambda x: -x[1])

        # Trigger word first, then remaining by frequency
        ordered = []
        if trigger and trigger in self._tag_freq:
            ordered.append(trigger)
        for tag, _ in sorted_tags:
            if tag.lower() != trigger:
                ordered.append(tag)

        tag_list = ", ".join(ordered)

        dlg = QDialog(self)
        dlg.setWindowTitle("Tag List — sorted by frequency")
        dlg.resize(700, 400)
        dlg.setStyleSheet(f"background:{BG};")
        vl = QVBoxLayout(dlg)
        vl.setContentsMargins(12, 12, 12, 12)
        vl.setSpacing(8)

        info = QLabel(
            f"{len(ordered)} unique tags  ·  trigger word first, then by frequency",
            dlg)
        info.setStyleSheet(
            f"color:{SEC};font-family:{FONT};font-size:{FONT_SM}px;background:transparent;")
        vl.addWidget(info)

        txt = QPlainTextEdit(dlg)
        txt.setPlainText(tag_list)
        txt.setReadOnly(True)
        txt.setStyleSheet(
            f"QPlainTextEdit{{background:{CAR};color:{PRI};border:1px solid {MUT};"
            f"border-radius:4px;font-family:{FONT};font-size:{FONT_SM}px;padding:6px;}}")
        vl.addWidget(txt, stretch=1)

        btn_copy = QPushButton("Copy to Clipboard", dlg)
        btn_copy.setFixedHeight(30)
        btn_copy.setStyleSheet(
            f"QPushButton{{background:{ACC};color:{PRI};border:none;border-radius:4px;"
            f"font-family:{FONT};font-size:{FONT_SM}px;font-weight:bold;padding:4px 16px;}}"
            f"QPushButton:hover{{background:#185FA5;}}")
        btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(tag_list))
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_copy)
        vl.addLayout(btn_row)

        dlg.exec()

    def _export_freq_csv(self):
        if not self._tag_freq:
            QMessageBox.warning(self, "No data", "Load a dataset first.")
            return
        # also write tags_report.txt alongside the dataset
        report_path = os.path.join(self._dataset_folder, "tags_report.txt")
        try:
            sorted_tags = sorted(self._tag_freq.items(), key=lambda x: -x[1])
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(", ".join(t for t, _ in sorted_tags))
            self._batch_log(f"Freq report saved: {report_path}")
        except Exception as e:
            self._batch_log(f"Report error: {e}")

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Tag Frequency CSV", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["tag", "count"])
                for tag, cnt in sorted(self._tag_freq.items(), key=lambda x: -x[1]):
                    w.writerow([tag, cnt])
            self._batch_log(f"Exported CSV: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _save_zip(self):
        """Save Final ZIP — images + txt + tags_report."""
        if not self._images:
            QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return
        # generate freq report first
        report_path = os.path.join(self._dataset_folder, "tags_report.txt")
        try:
            sorted_tags = sorted(self._tag_freq.items(), key=lambda x: -x[1])
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(", ".join(t for t, _ in sorted_tags))
        except Exception:
            pass

        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.join(os.path.dirname(self._dataset_folder),
                               f"final_{os.path.basename(self._dataset_folder)}_{ts}.zip")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Final ZIP", default, "ZIP files (*.zip)")
        if not out_path:
            return
        total = len(self._images)
        done  = 0
        try:
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for img_path in self._images:
                    zf.write(img_path, os.path.basename(img_path))
                    txt = _txt_path(img_path)
                    if os.path.isfile(txt):
                        zf.write(txt, os.path.basename(txt))
                    done += 1
                    self._set_progress(self._batch_progress, done / total)
                    QApplication.processEvents()
                if os.path.isfile(report_path):
                    zf.write(report_path, "tags_report.txt")
            self._batch_status.setText(
                f"ZIP saved: {os.path.basename(out_path)}")
            self._set_progress(self._batch_progress, 1.0)
            self._batch_log(f"Final ZIP: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "ZIP Error", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    #  FIRST-RUN / MODEL DOWNLOADS
    # ══════════════════════════════════════════════════════════════════════════
    def _check_first_run(self):
        models_dir = Path(_HERE) / "models"
        models_dir.mkdir(exist_ok=True)
        missing_wd14 = not any(models_dir.rglob("*.onnx"))
        missing_esr  = any(
            not (models_dir / fname).exists() for fname in ESR_MODELS)
        if missing_wd14 or missing_esr:
            parts = []
            if missing_wd14:
                parts.append("• WD14 tagger models (~500 MB)")
            if missing_esr:
                for fname, (_, desc) in ESR_MODELS.items():
                    if not (models_dir / fname).exists():
                        parts.append(f"• {fname}  [{desc}]")
            r = QMessageBox.question(
                self, "Download Models",
                "The following models are missing:\n\n"
                + "\n".join(parts)
                + "\n\nDownload all now? You can skip and they will be "
                  "downloaded on first use.",
                QMessageBox.Yes | QMessageBox.No)
            if r == QMessageBox.Yes:
                self._run_model_downloads()

    def _run_model_downloads(self):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            QMessageBox.critical(self, "Missing Package",
                                 "Install huggingface-hub:\n  pip install huggingface-hub")
            return

        import urllib.request

        models_dir = Path(_HERE) / "models"
        models_dir.mkdir(exist_ok=True)

        # each WD14 model needs 2 files → 2 steps each
        total = len(WD14_REPOS) * 2 + len(ESR_MODELS)
        dlg = QProgressDialog("Preparing downloads…", "Cancel", 0, total, self)
        dlg.setWindowTitle("Model Download")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()

        failed = []
        step   = 0

        # ── WD14 models (HuggingFace, file-by-file) ───────────────────────────
        for name, repo in WD14_REPOS.items():
            dest = models_dir / name
            dest.mkdir(exist_ok=True)
            for filename in ("model.onnx", "selected_tags.csv"):
                if dlg.wasCanceled():
                    break
                out = dest / filename
                dlg.setLabelText(f"Downloading {name} — {filename}…")
                dlg.setValue(step)
                QApplication.processEvents()
                if not out.exists():
                    try:
                        hf_hub_download(repo_id=repo, filename=filename,
                                        local_dir=str(dest))
                    except Exception as e:
                        failed.append(f"{name}/{filename}: {e}")
                step += 1

        # ── RealESRGAN models (GitHub releases) ───────────────────────────────
        for fname, (url, desc) in ESR_MODELS.items():
            if dlg.wasCanceled():
                break
            out_path = models_dir / fname
            if out_path.exists():
                step += 1
                continue
            dlg.setLabelText(f"Downloading {fname}  ({desc})…")
            dlg.setValue(step)
            QApplication.processEvents()
            try:
                urllib.request.urlretrieve(url, str(out_path))
            except Exception as e:
                failed.append(f"{fname}: {e}")
            step += 1

        dlg.setValue(total)
        dlg.close()

        if failed:
            QMessageBox.warning(self, "Download Issues",
                                "Some models failed:\n" + "\n".join(failed))
        else:
            QMessageBox.information(self, "Done",
                                    "All models downloaded successfully.")

    def _ensure_esr_model(self, fname: str) -> bool:
        """Return True if model exists; prompt to download if not."""
        models_dir = Path(_HERE) / "models"
        out_path   = models_dir / fname
        if out_path.exists():
            return True
        url, desc = ESR_MODELS[fname]
        r = QMessageBox.question(
            self, "Model Not Found",
            f"{fname} not found.\n[{desc}]\n\nDownload now?",
            QMessageBox.Yes | QMessageBox.No)
        if r != QMessageBox.Yes:
            return False
        import urllib.request
        try:
            models_dir.mkdir(exist_ok=True)
            dlg = QProgressDialog(f"Downloading {fname}…", None, 0, 0, self)
            dlg.setWindowModality(Qt.WindowModal)
            dlg.show()
            QApplication.processEvents()
            urllib.request.urlretrieve(url, str(out_path))
            dlg.close()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Download Failed", str(e))
            return False


# ══════════════════════════════════════════════════════════════════════════════
#  _TaggerThread — WD14 / JoyCaption in background QThread
# ══════════════════════════════════════════════════════════════════════════════
class _TaggerThread(QThread):
    progress = Signal(float, str)
    done     = Signal(bool, str)

    def __init__(self, *, images, engine, model1, model2, threshold,
                 char_thresh, device, exist_mode, output_mode,
                 trigger, desired, negative,
                 batch_size=4, ratings=None, rm_underscores=True,
                 joy_path=None, moondream_path=None,
                 joy_type="Descriptive", joy_length="any",
                 extra="", joy_threads=4, app_dir):
        super().__init__()
        self.images         = images
        self.engine         = engine
        self.model1         = model1
        self.model2         = model2
        self.threshold      = threshold
        self.char_thresh    = char_thresh
        self.device         = device
        self.exist_mode     = exist_mode
        self.output_mode    = output_mode
        self.trigger        = trigger
        self.desired        = [t.strip() for t in desired.split(",") if t.strip()]
        self.negative       = {t.strip().lower() for t in negative.split(",") if t.strip()}
        self.batch_size     = batch_size
        self.ratings        = ratings or {
            "general": True, "sensitive": True,
            "questionable": True, "explicit": True}
        self.rm_underscores = rm_underscores
        self.joy_path       = joy_path if (joy_path and os.path.isdir(joy_path)) else JOY_MODEL_ID
        self.moondream_path = moondream_path if (moondream_path and os.path.isdir(moondream_path)) else MOONDREAM_MODEL_ID
        self.joy_type       = joy_type
        self.joy_length     = joy_length
        self.extra          = extra
        self.joy_threads    = joy_threads
        self.app_dir        = app_dir

    # ── helpers ───────────────────────────────────────────────────────────────
    def _should_skip(self, img_path: str) -> bool:
        if self.exist_mode != "Skip":
            return False
        txt = _txt_path(img_path)
        return os.path.isfile(txt) and os.path.getsize(txt) > 0

    def _joy_should_skip(self, img_path: str) -> bool:
        """JoyCaption-specific skip: only skip if the .txt already contains a
        caption (sentence text).  A file that only has WD14 comma-separated
        tags is NOT considered captioned — JoyCaption will append to it."""
        if self.exist_mode != "Skip":
            return False
        txt = _txt_path(img_path)
        if not os.path.isfile(txt) or os.path.getsize(txt) == 0:
            return False
        try:
            with open(txt, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.read().split("\n") if l.strip()]
            if not lines:
                return False
            # If the only content is a single tags line, don't skip —
            # JoyCaption will append the caption below it.
            if len(lines) == 1 and _line_is_tags(lines[0]):
                return False
            # A second line or a non-tag first line means a caption exists.
            return True
        except Exception:
            return False

    def _write(self, img_path: str, tags: list[str], caption: str = ""):
        """Write result respecting exist_mode and output_mode."""
        txt = _txt_path(img_path)

        existing_tags: list[str] = []
        existing_cap:  str       = ""
        if self.exist_mode == "Append" and os.path.isfile(txt):
            try:
                # Always read from disk — tagger writes bypass the in-memory
                # cache, so _read_tags would return stale data (e.g. Hybrid
                # phase-2 needs the tags phase-1 just wrote to disk).
                with open(txt, "r", encoding="utf-8") as f:
                    raw = f.read()
                disk_lines = [l.strip() for l in raw.split('\n') if l.strip()]
                if disk_lines:
                    if _line_is_tags(disk_lines[0]):
                        existing_tags = [t.strip() for t in disk_lines[0].split(',') if t.strip()]
                        if len(disk_lines) > 1:
                            existing_cap = "\n".join(disk_lines[1:])
                    else:
                        existing_cap = "\n".join(disk_lines)
            except Exception:
                pass

        # merge tags
        all_tags = list(dict.fromkeys(existing_tags + tags))
        # apply desired / negative
        all_tags = [t for t in all_tags if t not in self.negative]
        for dt in reversed(self.desired):
            if dt and dt not in all_tags:
                all_tags.insert(0, dt)
        if self.trigger and self.trigger not in all_tags:
            all_tags.insert(0, self.trigger)
        elif self.trigger and self.trigger in all_tags:
            all_tags = [self.trigger] + [t for t in all_tags if t != self.trigger]

        final_cap = caption or existing_cap

        # build output string based on output_mode
        mode = self.output_mode
        if mode == "Tags":
            content = ", ".join(all_tags)
        elif mode in ("Captions", "Captions (Fast)"):
            # preserve existing tags when appending
            content = ", ".join(all_tags) if all_tags else ""
            if final_cap:
                content = (content + "\n" if content else "") + final_cap
        elif mode in ("Hybrid", "Hybrid (GPU)", "Hybrid (Fast)"):
            content = ", ".join(all_tags)
            if final_cap:
                content = (content + "\n" if content else "") + final_cap
        else:
            content = ", ".join(all_tags)

        try:
            with open(txt, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass

    def run(self):
        try:
            # Hybrid (GPU) uses timm/PyTorch for WD14 (GPU-accelerated).
            # Plain Hybrid uses ONNX WD14 (CPU only via ORT).
            use_gpu = "cuda" in self.device.lower() or self.device.lower() == "auto-detect"

            if self.output_mode == "Captions":
                self._run_joycaption()
            elif self.output_mode == "Captions (Fast)":
                self._run_moondream()
            elif self.output_mode == "Hybrid (GPU)":
                self._run_wd14_timm(emit_done=False)
                saved_exist = self.exist_mode
                self.exist_mode = "Append"
                self._run_joycaption()
                self.exist_mode = saved_exist
            elif self.output_mode == "Hybrid":
                self._run_wd14(emit_done=False)
                saved_exist = self.exist_mode
                self.exist_mode = "Append"
                self._run_joycaption()
                self.exist_mode = saved_exist
            elif self.output_mode == "Hybrid (Fast)":
                self._run_wd14(emit_done=False)
                saved_exist = self.exist_mode
                self.exist_mode = "Append"
                self._run_moondream()
                self.exist_mode = saved_exist
            else:
                # Tags mode — timm on GPU, ONNX on CPU
                if use_gpu:
                    self._run_wd14_timm()
                else:
                    self._run_wd14()
        except Exception as e:
            import traceback
            self.done.emit(False, f"Fatal: {e}\n{traceback.format_exc()}")

    # ── WD14 ─────────────────────────────────────────────────────────────────
    def _run_wd14(self, emit_done=True):
        # Add PyTorch's CUDA DLLs to the Windows DLL search path so that
        # onnxruntime-gpu can find cudart/cublas/cuDNN without a system install.
        # NOTE: must store the handle — if dropped, CPython immediately calls
        # RemoveDllDirectory and the path vanishes before onnxruntime imports.
        _dll_handles = []
        try:
            import torch as _torch
            _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
            if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
                _dll_handles.append(os.add_dll_directory(_torch_lib))
        except Exception:
            pass
        try:
            import onnxruntime as ort
        except Exception as _ort_err:
            import traceback as _tb, ctypes
            _err_detail = f"onnxruntime not installed. Run INSTALL.bat and choose a GPU option.\nDetail: {_ort_err}\n\n{_tb.format_exc()}"
            # Try loading onnxruntime DLLs individually via ctypes to get the real Windows error
            try:
                import onnxruntime as _ort_mod_path
            except Exception:
                _ort_mod_path = None
            _ctypes_log = []
            try:
                import importlib.util as _ilu
                _spec = _ilu.find_spec("onnxruntime")
                _ort_pkg = os.path.dirname(_spec.origin) if _spec else None
                if _ort_pkg:
                    _capi_dir = os.path.join(_ort_pkg, "capi")
                    for _fn in sorted(os.listdir(_capi_dir)):
                        if _fn.endswith(".dll") or _fn.endswith(".pyd"):
                            try:
                                ctypes.WinDLL(os.path.join(_capi_dir, _fn))
                                _ctypes_log.append(f"  OK:   {_fn}")
                            except OSError as _ce:
                                _ctypes_log.append(f"  FAIL: {_fn} => {_ce}")
            except Exception as _ce2:
                _ctypes_log.append(f"  ctypes probe error: {_ce2}")
            if _ctypes_log:
                _err_detail += "\n\nDLL probe:\n" + "\n".join(_ctypes_log)
            try:
                _log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ort_error.log")
                with open(_log_path, "w", encoding="utf-8") as _lf:
                    _lf.write(_err_detail)
            except Exception:
                pass
            self.done.emit(False, _err_detail)
            return

        models_to_use = ([self.model1, self.model2]
                         if self.engine == "WD14 Ensemble"
                         else [self.model1])

        sessions = []
        for mname in models_to_use:
            model_dir  = Path(self.app_dir) / "models" / mname
            onnx_files = list(model_dir.glob("*.onnx"))
            if not onnx_files:
                self.done.emit(False,
                    f"Model '{mname}': no .onnx file in {model_dir}.\n"
                    f"Use Tagger → Download Models.")
                return
            csv_files  = list(model_dir.glob("*.csv"))
            if not csv_files:
                self.done.emit(False,
                    f"Model '{mname}': no .csv tags file in {model_dir}.")
                return

            dev_lower = self.device.lower()
            if "cuda" in dev_lower:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "directml" in dev_lower:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            try:
                sess = ort.InferenceSession(str(onnx_files[0]), providers=providers)
                active = sess.get_providers()[0] if sess.get_providers() else "unknown"
                if active != providers[0]:
                    # ORT CUDAExecutionProvider not available — expected with ROCm.
                    # WD14 runs fine on CPU (fast enough). JoyCaption uses PyTorch GPU.
                    self.progress.emit(0.0, f"{mname}: CPU (ORT CUDAExecutionProvider unavailable)")
                else:
                    self.progress.emit(0.0, f"{mname}: using {active}")
            except Exception as e:
                self.done.emit(False, f"ONNX load failed for {mname}: {e}")
                return

            tag_names:   list[str] = []
            general_idx: list[int] = []
            char_idx:    list[int] = []
            rating_idx:  list[int] = []
            with open(csv_files[0], newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tag_names.append(row.get("name", ""))
                    cat = int(row.get("category", 0))
                    idx = len(tag_names) - 1
                    if cat == 0:
                        general_idx.append(idx)
                    elif cat == 4:
                        char_idx.append(idx)
                    elif cat == 9:
                        rating_idx.append(idx)

            sessions.append((sess, tag_names, general_idx, char_idx, rating_idx))

        import numpy as _np

        total  = len(self.images)
        tagged = 0
        errors = 0
        bs     = max(1, self.batch_size)

        def _prep(img_path):
            """Load and preprocess one image → float32 CHW array, or None on error."""
            img = Image.open(img_path).convert("RGBA")
            bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.alpha_composite(img)
            img = bg.convert("RGB").resize(
                (WD14_IMG_SIZE, WD14_IMG_SIZE), Image.LANCZOS)
            arr = _np.array(img, dtype=_np.float32)[:, :, ::-1]   # RGB→BGR
            return arr

        # Chunk images into batches
        for batch_start in range(0, total, bs):
            batch_paths = self.images[batch_start:batch_start + bs]

            # Separate skips from images to process
            to_process = []   # (original_index, path, arr)
            for j, img_path in enumerate(batch_paths):
                abs_i = batch_start + j
                self.progress.emit(abs_i / total,
                                   f"Tagging {abs_i+1}/{total}…")
                if self._should_skip(img_path):
                    tagged += 1
                    continue
                try:
                    to_process.append((abs_i, img_path, _prep(img_path)))
                except Exception as e:
                    errors += 1
                    self.progress.emit(abs_i / total,
                        f"Error loading {os.path.basename(img_path)}: {e}")

            if not to_process:
                continue

            paths_batch = [x[1] for x in to_process]
            arrs_batch  = _np.stack([x[2] for x in to_process], axis=0)  # (N,H,W,C)

            try:
                # Run inference once per session for the whole batch,
                # cache results keyed by image index within to_process
                # session_preds[sess_idx][img_idx] = preds array
                session_preds = []
                for sess, tag_names, gen_idx, ch_idx, rat_idx in sessions:
                    inp_name  = sess.get_inputs()[0].name
                    all_preds = sess.run(None, {inp_name: arrs_batch})[0]  # (N, tags)
                    session_preds.append(all_preds)

                for k, (abs_i, img_path, _) in enumerate(to_process):
                    try:
                        # ratings filter — check across all sessions
                        skip_img = False
                        for s_idx, (sess, tag_names, gen_idx, ch_idx, rat_idx) \
                                in enumerate(sessions):
                            preds = session_preds[s_idx][k]
                            dom_rating, dom_prob = "general", 0.0
                            for idx in rat_idx:
                                rkey = WD14_RATING_MAP.get(
                                    tag_names[idx] if idx < len(tag_names) else "")
                                if rkey and idx < len(preds) and \
                                        float(preds[idx]) > dom_prob:
                                    dom_rating, dom_prob = rkey, float(preds[idx])
                            if not self.ratings.get(dom_rating, True):
                                skip_img = True
                                break
                        if skip_img:
                            tagged += 1
                            continue

                        # tag extraction — reuse same cached preds
                        tag_union: set[str] = set()
                        for s_idx, (sess, tag_names, gen_idx, ch_idx, rat_idx) \
                                in enumerate(sessions):
                            preds = session_preds[s_idx][k]
                            for idx in gen_idx:
                                if idx < len(preds) and preds[idx] > self.threshold:
                                    name = tag_names[idx]
                                    tag_union.add(
                                        name.replace("_", " ")
                                        if self.rm_underscores else name)
                            for idx in ch_idx:
                                if idx < len(preds) and preds[idx] > self.char_thresh:
                                    name = tag_names[idx]
                                    tag_union.add(
                                        name.replace("_", " ")
                                        if self.rm_underscores else name)

                        self._write(img_path, list(tag_union))
                        tagged += 1
                    except Exception as e2:
                        errors += 1
                        self.progress.emit(abs_i / total,
                            f"Error: {os.path.basename(img_path)}: {e2}")

            except Exception as e:
                # Batch inference failed (e.g. OOM) — fall back to per-image
                for abs_i, img_path, arr in to_process:
                    try:
                        single    = _np.expand_dims(arr, 0)
                        skip_img  = False
                        for sess, tag_names, gen_idx, ch_idx, rat_idx in sessions:
                            inp_name = sess.get_inputs()[0].name
                            preds    = sess.run(None, {inp_name: single})[0][0]
                            dom_rating, dom_prob = "general", 0.0
                            for idx in rat_idx:
                                rkey = WD14_RATING_MAP.get(
                                    tag_names[idx] if idx < len(tag_names) else "")
                                if rkey and idx < len(preds) and \
                                        float(preds[idx]) > dom_prob:
                                    dom_rating, dom_prob = rkey, float(preds[idx])
                            if not self.ratings.get(dom_rating, True):
                                skip_img = True
                                break
                        if skip_img:
                            tagged += 1
                            continue
                        tag_union: set[str] = set()
                        for sess, tag_names, gen_idx, ch_idx, rat_idx in sessions:
                            inp_name = sess.get_inputs()[0].name
                            preds    = sess.run(None, {inp_name: single})[0][0]
                            for idx in gen_idx:
                                if idx < len(preds) and preds[idx] > self.threshold:
                                    name = tag_names[idx]
                                    tag_union.add(
                                        name.replace("_", " ")
                                        if self.rm_underscores else name)
                            for idx in ch_idx:
                                if idx < len(preds) and preds[idx] > self.char_thresh:
                                    name = tag_names[idx]
                                    tag_union.add(
                                        name.replace("_", " ")
                                        if self.rm_underscores else name)
                        self._write(img_path, list(tag_union))
                        tagged += 1
                    except Exception as e2:
                        errors += 1
                        self.progress.emit(abs_i / total,
                            f"Error: {os.path.basename(img_path)}: {e2}")

        self.progress.emit(1.0, "Done.")
        if emit_done:
            self.done.emit(True,
                f"Tagged {tagged}/{total} images. Errors: {errors}.")

    # ── WD14 timm (GPU) ──────────────────────────────────────────────────────
    def _run_wd14_timm(self, emit_done=True):
        """Pure PyTorch WD14 inference via timm — GPU-accelerated (ROCm/CUDA)."""
        try:
            import torch
        except ImportError:
            self.done.emit(False, "torch not installed. Run INSTALL.bat and choose a GPU option.")
            return
        try:
            import timm
        except ImportError:
            self.done.emit(False,
                "timm not installed. Run: pip install timm")
            return
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            self.done.emit(False,
                "huggingface_hub not installed. Run: pip install huggingface_hub")
            return

        # ── resolve device ────────────────────────────────────────────────────
        dev_lower = self.device.lower()
        if "cuda" in dev_lower or dev_lower == "auto-detect":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        elif "cpu" in dev_lower:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # float16 halves VRAM usage with negligible accuracy loss for tagging.
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        mname   = "WD EVA02-Large (GPU)"
        repo_id = WD14_TIMM_REPOS[mname]

        # ── pythonw.exe has no console: stdout/stderr are None.
        #    tqdm/huggingface_hub write to them during download — guard here.
        import io as _io
        _saved_out, _saved_err = sys.stdout, sys.stderr
        if sys.stdout is None: sys.stdout = _io.StringIO()
        if sys.stderr is None: sys.stderr = _io.StringIO()
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            from huggingface_hub import disable_progress_bars as _dpb
            _dpb()
        except Exception:
            pass

        # ── load model ────────────────────────────────────────────────────────
        self.progress.emit(0.0, f"Loading {mname}…")
        try:
            model = timm.create_model("hf-hub:" + repo_id, pretrained=True)
            model.eval().to(device=device, dtype=dtype)
            self.progress.emit(0.0, f"{mname}: using {device} ({dtype})")
        except Exception as e:
            sys.stdout, sys.stderr = _saved_out, _saved_err
            self.done.emit(False, f"timm model load failed for {mname}: {e}")
            return
        finally:
            sys.stdout, sys.stderr = _saved_out, _saved_err

        # ── load tags CSV from HF hub ─────────────────────────────────────────
        if sys.stdout is None: sys.stdout = _io.StringIO()
        if sys.stderr is None: sys.stderr = _io.StringIO()
        try:
            csv_path = hf_hub_download(repo_id, "selected_tags.csv")
        except Exception as e:
            self.done.emit(False, f"Could not download tags CSV for {mname}: {e}")
            return
        finally:
            sys.stdout, sys.stderr = _saved_out, _saved_err

        tag_names:   list[str] = []
        general_idx: list[int] = []
        char_idx:    list[int] = []
        rating_idx:  list[int] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag_names.append(row.get("name", ""))
                cat = int(row.get("category", 0))
                idx = len(tag_names) - 1
                if cat == 0:   general_idx.append(idx)
                elif cat == 4: char_idx.append(idx)
                elif cat == 9: rating_idx.append(idx)

        # ── build transform from model's own data config ──────────────────────
        data_cfg  = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_cfg, is_training=False)

        # ── GPU pre-flight check ─────────────────────────────────────────────
        # Run a dummy forward pass before touching real images. If the GPU
        # doesn't support the required op variant, fall back to CPU cleanly.
        if device.type == "cuda":
            try:
                _dummy = torch.zeros(1, 3, WD14_IMG_SIZE, WD14_IMG_SIZE,
                                     device=device, dtype=dtype)
                with torch.inference_mode():
                    model(_dummy)
                del _dummy
            except RuntimeError as _e:
                if "CUBLAS" in str(_e) or "cublas" in str(_e).lower():
                    self.progress.emit(0.0,
                        f"{mname}: GPU pre-flight failed — switching to CPU")
                    device = torch.device("cpu")
                    dtype  = torch.float32
                    model  = model.to(device=device, dtype=dtype)
                    self.progress.emit(0.0, f"{mname}: using cpu (float32)")
                else:
                    self.done.emit(False, f"GPU pre-flight failed: {_e}")
                    return

        def _prep(img_path):
            img = Image.open(img_path).convert("RGBA")
            bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.alpha_composite(img)
            img = bg.convert("RGB").resize(
                (WD14_IMG_SIZE, WD14_IMG_SIZE), Image.LANCZOS)
            return transform(img)   # CHW float tensor

        total  = len(self.images)
        tagged = 0
        errors = 0
        bs     = max(1, self.batch_size)

        with torch.inference_mode():
            for batch_start in range(0, total, bs):
                batch_paths = self.images[batch_start:batch_start + bs]

                to_process = []
                for j, img_path in enumerate(batch_paths):
                    abs_i = batch_start + j
                    self.progress.emit(abs_i / total, f"Tagging {abs_i+1}/{total}…")
                    if self._should_skip(img_path):
                        tagged += 1
                        continue
                    try:
                        to_process.append((abs_i, img_path, _prep(img_path)))
                    except Exception as e:
                        errors += 1
                        self.progress.emit(abs_i / total,
                            f"Error loading {os.path.basename(img_path)}: {e}")

                if not to_process:
                    continue

                try:
                    batch_t = torch.stack([x[2] for x in to_process]).to(device=device, dtype=dtype)
                    preds_batch = torch.sigmoid(model(batch_t)).cpu().float().numpy()  # (N, tags)

                    for k, (abs_i, img_path, _) in enumerate(to_process):
                        try:
                            preds = preds_batch[k]

                            # ratings filter
                            dom_rating, dom_prob = "general", 0.0
                            for idx in rating_idx:
                                rkey = WD14_RATING_MAP.get(
                                    tag_names[idx] if idx < len(tag_names) else "")
                                if rkey and idx < len(preds) and float(preds[idx]) > dom_prob:
                                    dom_rating, dom_prob = rkey, float(preds[idx])
                            if not self.ratings.get(dom_rating, True):
                                tagged += 1
                                continue

                            tag_union: set[str] = set()
                            for idx in general_idx:
                                if idx < len(preds) and preds[idx] > self.threshold:
                                    name = tag_names[idx]
                                    tag_union.add(
                                        name.replace("_", " ")
                                        if self.rm_underscores else name)
                            for idx in char_idx:
                                if idx < len(preds) and preds[idx] > self.char_thresh:
                                    name = tag_names[idx]
                                    tag_union.add(
                                        name.replace("_", " ")
                                        if self.rm_underscores else name)

                            self._write(img_path, list(tag_union))
                            tagged += 1
                        except Exception as e2:
                            errors += 1
                            self.progress.emit(abs_i / total,
                                f"Error: {os.path.basename(img_path)}: {e2}")

                except Exception as e:
                    # batch failed — fall back to per-image
                    for abs_i, img_path, tensor in to_process:
                        try:
                            preds = torch.sigmoid(
                                model(tensor.unsqueeze(0).to(device=device, dtype=dtype))
                            )[0].cpu().float().numpy()

                            dom_rating, dom_prob = "general", 0.0
                            for idx in rating_idx:
                                rkey = WD14_RATING_MAP.get(
                                    tag_names[idx] if idx < len(tag_names) else "")
                                if rkey and idx < len(preds) and float(preds[idx]) > dom_prob:
                                    dom_rating, dom_prob = rkey, float(preds[idx])
                            if not self.ratings.get(dom_rating, True):
                                tagged += 1
                                continue

                            tag_union: set[str] = set()
                            for idx in general_idx:
                                if idx < len(preds) and preds[idx] > self.threshold:
                                    name = tag_names[idx]
                                    tag_union.add(
                                        name.replace("_", " ")
                                        if self.rm_underscores else name)
                            for idx in char_idx:
                                if idx < len(preds) and preds[idx] > self.char_thresh:
                                    name = tag_names[idx]
                                    tag_union.add(
                                        name.replace("_", " ")
                                        if self.rm_underscores else name)
                            self._write(img_path, list(tag_union))
                            tagged += 1
                        except Exception as e2:
                            errors += 1
                            self.progress.emit(abs_i / total,
                                f"Error: {os.path.basename(img_path)}: {e2}")

        self.progress.emit(1.0, "Done.")
        if emit_done:
            self.done.emit(True,
                f"Tagged {tagged}/{total} images. Errors: {errors}.")

    # ── JoyCaption ───────────────────────────────────────────────────────────
    def _run_joycaption(self):
        import traceback as _tb
        _joy_log = os.path.join(os.path.dirname(os.path.dirname(__file__)), "joy_crash.log")
        def _jlog(msg):
            try:
                with open(_joy_log, "a", encoding="utf-8") as _f:
                    _f.write(msg + "\n")
            except Exception:
                pass
        _jlog("=== JoyCaption run started ===")
        self.progress.emit(0.0, "Loading JoyCaption model…")
        _jlog("Calling _load_joycaption()")
        try:
            model, proc, device, load_err = self._load_joycaption()
        except Exception as _je:
            _jlog(f"_load_joycaption raised: {_je}\n{_tb.format_exc()}")
            self.done.emit(False, f"Failed to load JoyCaption model.\n{_je}")
            return
        _jlog(f"_load_joycaption returned: model={model is not None}, device={device}, err={load_err!r}")
        if model is None:
            self.done.emit(False, f"Failed to load JoyCaption model.\n{load_err}")
            return

        self.progress.emit(0.0, f"JoyCaption loaded on {device}")

        total     = len(self.images)
        captioned = 0
        skipped   = 0
        errors    = 0

        for i, img_path in enumerate(self.images):
            self.progress.emit(i / total, f"Captioning {i+1}/{total}…")
            if self._joy_should_skip(img_path):
                skipped += 1
                continue
            try:
                img     = Image.open(img_path).convert("RGB")
                caption = self._joy_infer_one(img, model, proc, device)
                self._write(img_path, [], caption=caption)
                captioned += 1
            except Exception as e:
                errors += 1
                self.progress.emit(i / total,
                    f"Error: {os.path.basename(img_path)}: {e}")

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self.progress.emit(1.0, "Done.")
        parts = [f"Captioned {captioned}/{total} images"]
        if skipped:
            parts.append(f"skipped {skipped} (caption exists)")
        if errors:
            parts.append(f"errors: {errors}")
        self.done.emit(True, ".  ".join(parts) + ".")

    def _load_joycaption(self):
        # Integrity check first — catches corrupt/truncated downloads before
        # we spend time importing torch and loading the model.
        _ok, _err = _check_joycaption_model(self.joy_path)
        if not _ok:
            return None, None, None, (
                f"JoyCaption model integrity check failed:\n{_err}\n\n"
                f"Re-download the model to fix this.")

        # pythonw.exe has no console — stdout/stderr are None.
        # transformers/huggingface_hub write tqdm progress during download,
        # which segfaults when stdout is None. Guard before ANY HF import.
        import io as _io
        _saved_out, _saved_err = sys.stdout, sys.stderr
        if sys.stdout is None: sys.stdout = _io.StringIO()
        if sys.stderr is None: sys.stderr = _io.StringIO()
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        try:
            from huggingface_hub import disable_progress_bars as _dpb
            _dpb()
        except Exception:
            pass

        _joy_log = os.path.join(os.path.dirname(os.path.dirname(__file__)), "joy_crash.log")
        def _jlog(msg):
            try:
                with open(_joy_log, "a", encoding="utf-8") as _f:
                    _f.write(msg + "\n")
            except Exception:
                pass
        try:
            _jlog("importing torch")
            import torch
            _jlog(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
            # Only stub C10D on ROCm — NVIDIA has proper implementation
            _is_rocm = getattr(torch.version, 'hip', None) is not None
            _jlog(f"is_rocm={_is_rocm}")
            if _is_rocm:
                self._stub_c10d()
                _jlog("_stub_c10d done (ROCm)")
            try:                # disable dynamo compilation — not needed for
                torch._dynamo.config.disable = True   # inference
            except Exception:
                pass
            _jlog("importing transformers")
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            _jlog("transformers imported")

            _jlog(f"loading processor from {self.joy_path}")
            proc = AutoProcessor.from_pretrained(self.joy_path)
            _jlog("processor loaded")

            has_cuda = torch.cuda.is_available()
            _jlog(f"has_cuda={has_cuda}")

            dtype = torch.bfloat16 if has_cuda else torch.float32
            _jlog(f"dtype={dtype}")

            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            # Load via mmap (low RAM usage), then move to CUDA through pinned memory.
            # Pinning is a CPU op (no GPU timeout), DMA from pinned→CUDA is fast.
            # This avoids Windows TDR caused by GPU waiting on HDD page faults.
            if has_cuda:
                # 1. from_pretrained builds the correct model structure via mmap
                #    (no weight data accessed, no page faults).
                # 2. to_empty("cuda:0") replaces every mmap-backed CPU tensor with
                #    an uninitialised CUDA tensor of the same shape, freeing all
                # Load directly to CUDA via device_map — accelerate handles
                # weight streaming so the 17.8 GB model never fully occupies RAM.
                # This also correctly initialises all buffers (inv_freq, etc.)
                # without needing manual remapping or reinitialisation.
                _jlog("loading model → cuda:0 via device_map (accelerate)")
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.joy_path,
                    torch_dtype=dtype,
                    device_map="cuda:0",
                    low_cpu_mem_usage=True,
                )
                _jlog(f"model loaded, free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB")
            else:
                _jlog("loading model to CPU (mmap)")
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.joy_path, torch_dtype=dtype, low_cpu_mem_usage=True)
                _jlog("model loaded (CPU)")

            model.eval()
            dev = next(model.parameters()).device
            _jlog(f"model.eval() done, device={dev}")
            return model, proc, dev, ""
        except BaseException as e:
            import traceback
            err = f"{e}\n{traceback.format_exc()}"
            _jlog(f"EXCEPTION in _load_joycaption: {err}")
            self.progress.emit(0.0, f"JoyCaption load error: {e}")
            return None, None, None, err
        finally:
            sys.stdout, sys.stderr = _saved_out, _saved_err

    def _joy_infer_one(self, img, model, proc, device):
        import torch
        prompt = _joy_prompt(self.joy_type, self.joy_length)
        extra  = getattr(self, 'extra', '').strip()
        if extra:
            prompt = f"{prompt} {extra}"
        img_tok_id  = model.config.image_token_index
        image_token = proc.tokenizer.convert_ids_to_tokens(img_tok_id)
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user",   "content": f"{image_token}\n{prompt}"},
        ]
        chat_text = proc.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=chat_text, images=[img], return_tensors="pt")
        model_dtype = next(model.parameters()).dtype
        inputs = {
            k: (v.to(device=device, dtype=model_dtype)
                if hasattr(v, "to") and hasattr(v, "is_floating_point") and v.is_floating_point()
                else v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        input_len = inputs["input_ids"].shape[1]
        decoded = proc.decode(output[0][input_len:], skip_special_tokens=True)
        del inputs, output
        return decoded.strip()

    # ── Moondream2 (Fast caption) ─────────────────────────────────────────────
    def _run_moondream(self):
        self.progress.emit(0.0, "Loading Moondream2 model…")
        model, tokenizer, device, load_err = self._load_moondream()
        if model is None:
            self.done.emit(False, f"Failed to load Moondream2 model.\n{load_err}")
            return

        self.progress.emit(0.0, f"Moondream2 loaded on {device}")

        total     = len(self.images)
        captioned = 0
        skipped   = 0
        errors    = 0

        for i, img_path in enumerate(self.images):
            self.progress.emit(i / total, f"Captioning {i+1}/{total}…")
            if self._joy_should_skip(img_path):
                skipped += 1
                continue
            try:
                img     = Image.open(img_path).convert("RGB")
                caption = self._moondream_infer_one(img, model, tokenizer, device)
                self._write(img_path, [], caption=caption)
                captioned += 1
            except Exception as e:
                errors += 1
                self.progress.emit(i / total,
                    f"Error: {os.path.basename(img_path)}: {e}")

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self.progress.emit(1.0, "Done.")
        parts = [f"Captioned {captioned}/{total} images"]
        if skipped:
            parts.append(f"skipped {skipped} (caption exists)")
        if errors:
            parts.append(f"errors: {errors}")
        self.done.emit(True, ".  ".join(parts) + ".")

    @staticmethod
    def _stub_c10d():
        """Patch ROCm/Windows torch for inference.

        Call AFTER `import torch`, BEFORE `from transformers import ...`.

        On ROCm Windows, torch._C._distributed_c10d is inaccessible via
        Python's import path (torch._C is a .pyd, not a package), so any
        code that does `from torch._C._distributed_c10d import X` fails.
        This affects both torch._dynamo (load time) and model.chat() (runtime).

        Fix: replace the missing/empty stub with an _AutoStub whose __getattr__
        creates a placeholder class for every requested symbol.  This lets
        torch/distributed/distributed_c10d.py import normally so that
        torch.distributed.is_available() / is_initialized() work correctly.
        """
        import sys as _sys, types as _types

        class _StubMeta(type):
            """Metaclass for stub classes.
            Handles *class-level* attribute access (e.g. ProcessGroup.BackendType)
            by auto-creating a child stub class on demand.
            Special cases:
              __members__  → {} (Enum iteration)
              other dunders → AttributeError (let importlib/inspect use defaults)
            """
            def __getattr__(cls, name):
                if name == '__members__':
                    return {}           # Enum-style: empty member dict
                if name.startswith('__') and name.endswith('__'):
                    raise AttributeError(name)
                child = _StubMeta(f"{cls.__name__}.{name}", (), {})
                setattr(cls, name, child)
                return child

        def _stub_cls(name):
            """Create a self-expanding stub class (class attrs auto-created)."""
            return _StubMeta(name, (), {})

        # ── meta-path finder: auto-stub any submodule of an _AutoStub ─────
        # When `import torch.distributed.tensor.parallel` is attempted and
        # torch.distributed.tensor is already an _AutoStub, this finder
        # intercepts it, creates a child _AutoStub, and returns a no-op spec.
        # This prevents whack-a-mole for every sub-package of stubbed modules.
        class _AutoStubLoader:
            def create_module(self, spec):
                return _sys.modules.get(spec.name)
            def exec_module(self, module):
                pass

        class _AutoStubFinder:
            _loader = _AutoStubLoader()
            def find_spec(self, fullname, path, target=None):
                if '.' not in fullname or fullname in _sys.modules:
                    return None
                parent = fullname.rsplit('.', 1)[0]
                if not isinstance(_sys.modules.get(parent), _AutoStub):
                    return None
                stub = _AutoStub(fullname)
                _sys.modules[fullname] = stub
                import importlib.machinery as _imm
                spec = _imm.ModuleSpec(
                    fullname, self._loader, origin='<auto-stub>')
                spec.submodule_search_locations = []
                return spec

        if not any(type(f).__name__ == '_AutoStubFinder'
                   for f in _sys.meta_path):
            _sys.meta_path.append(_AutoStubFinder())

        class _AutoStub(_types.ModuleType):
            """Returns a new stub class for any attribute not already set.
            __path__ = [] marks this as a package so Python calls meta_path
            finders for sub-imports instead of raising 'is not a package'.
            Dunder attributes raise AttributeError so inspect/importlib see
            normal module metadata (e.g. __file__ must be a string, not a class)."""
            def __init__(self, name):
                super().__init__(name)
                self.__dict__['__path__'] = []   # register as package
            def __getattr__(self, name):
                if name.startswith('__') and name.endswith('__'):
                    raise AttributeError(name)
                val = _stub_cls(name)
                self.__dict__[name] = val   # cache — no repeated __getattr__
                return val

        def _ensure_stub(key, **attrs):
            if key not in _sys.modules:
                m = _types.ModuleType(key)
                for k, v in attrs.items():
                    setattr(m, k, v)
                _sys.modules[key] = m

        import torch as _torch

        # ── torch.compiler.disable → no-op ───────────────────────────────
        # torchvision/tv_tensors/__init__.py applies @torch.compiler.disable
        # at import time.  torch.compiler.disable() does `import torch._dynamo`
        # which cascades into torch.distributed.tensor → _functional_collectives
        # → C-level operator registration that fails on ROCm Windows.
        # Replace with a no-op decorator; since we also disable dynamo globally,
        # there is no compilation to exclude from anyway.
        try:
            import torch.compiler as _tc
            if not getattr(_tc, '_rocm_patched', False):
                def _noop_disable(fn=None, recursive=True):
                    return fn if fn is not None else (lambda f: f)
                _tc.disable = _noop_disable
                _tc._rocm_patched = True
        except Exception:
            pass

        # ── torch._C._distributed_c10d ────────────────────────────────────
        # Two access patterns need patching:
        #   (a) `from torch._C._distributed_c10d import X`  → sys.modules lookup
        #   (b) `torch._C._distributed_c10d`                → attribute on torch._C
        # sys.modules handles (a). Setting the attribute handles (b).
        _c10d_key = 'torch._C._distributed_c10d'
        if not hasattr(_sys.modules.get(_c10d_key), '_DistributedBackendOptions'):
            _sys.modules[_c10d_key] = _AutoStub(_c10d_key)
        if not hasattr(_torch._C, '_distributed_c10d'):
            _torch._C._distributed_c10d = _sys.modules[_c10d_key]

        # ── torch._dynamo → fsdp chain (load-time) ────────────────────────
        # Pre-stub with _AutoStub so any `from ... import X` works, not just
        # the original _fsdp_param_group.
        if 'torch.distributed.fsdp._fully_shard' not in _sys.modules:
            _sys.modules['torch.distributed.fsdp._fully_shard'] = _AutoStub(
                'torch.distributed.fsdp._fully_shard')

        # ── torch.distributed.tensor / _functional_collectives ────────────
        # torch._dynamo.trace_rules imports torch.distributed.tensor to build
        # its object-rule map.  tensor/__init__ imports _functional_collectives,
        # whose module-level code registers _c10d_functional::all_reduce — an
        # operator that doesn't exist in the ROCm dispatch table → RuntimeError.
        # Pre-stubbing both prevents their module code from ever executing.
        for _key in (
            'torch.distributed._functional_collectives',
            'torch.distributed._functional_collectives_impl',
            'torch.distributed.tensor',
            'torch.distributed.device_mesh',
        ):
            if _key not in _sys.modules:
                _sys.modules[_key] = _AutoStub(_key)

        # ── fake_pg (load-time) ───────────────────────────────────────────
        # Pre-stub so the real file (subclasses dist.Store, absent on ROCm)
        # is never executed.
        _ensure_stub('torch.testing._internal.distributed.fake_pg',
                     FakeProcessGroup=_stub_cls('FakeProcessGroup'),
                     FakeStore=_stub_cls('FakeStore'))
        _ensure_stub('torch.distributed.fake_pg',
                     FakeProcessGroup=_stub_cls('FakeProcessGroup'),
                     FakeStore=_stub_cls('FakeStore'))

        # ── torch.distributed — auto-stub missing C10d symbols ───────────
        # On ROCm Windows, torch.distributed.is_available() returns False so
        # __init__.py never imports Store/FileStore/HashStore/_remote_device
        # etc. from the C extension.  Instead of listing every missing name,
        # install a module-level __getattr__ (PEP 562) that auto-stubs any
        # missing attribute via the _AutoStub for _distributed_c10d.
        def _install_getattr(mod_name):
            """Install auto-stub __getattr__ on an already-loaded real module."""
            try:
                mod = _sys.modules.get(mod_name)
                if mod is None:
                    import importlib as _il
                    mod = _il.import_module(mod_name)
                if hasattr(mod, '__getattr__'):
                    return
                _c10d = _sys.modules[_c10d_key]
                def _getattr(name, _c10d=_c10d, _mod=mod):
                    if name.startswith('__') and name.endswith('__'):
                        raise AttributeError(name)
                    val = getattr(_c10d, name)
                    setattr(_mod, name, val)
                    return val
                mod.__getattr__ = _getattr
            except Exception:
                pass

        _install_getattr('torch.distributed')
        _install_getattr('torch.distributed.device_mesh')

    def _load_moondream(self):
        import io as _io
        _saved_out, _saved_err = sys.stdout, sys.stderr
        if sys.stdout is None: sys.stdout = _io.StringIO()
        if sys.stderr is None: sys.stderr = _io.StringIO()
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        try:
            from huggingface_hub import disable_progress_bars as _dpb
            _dpb()
        except Exception:
            pass

        try:
            import torch
            self._stub_c10d()   # guard against ROCm distributed import chain
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.moondream_path,
                revision=MOONDREAM_REVISION,
                trust_remote_code=True,
            )

            has_cuda = torch.cuda.is_available()
            dtype    = torch.float16 if has_cuda else torch.float32

            try:
                import accelerate  # noqa: F401
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.moondream_path,
                        revision=MOONDREAM_REVISION,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        device_map={"": 0},
                    )
                except (RuntimeError, Exception):
                    model = AutoModelForCausalLM.from_pretrained(
                        self.moondream_path,
                        revision=MOONDREAM_REVISION,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        device_map="auto",
                    )
            except ImportError:
                model = AutoModelForCausalLM.from_pretrained(
                    self.moondream_path,
                    revision=MOONDREAM_REVISION,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
                if has_cuda:
                    model = model.cuda()

            model.eval()
            dev = next(model.parameters()).device
            return model, tokenizer, dev, ""
        except Exception as e:
            import traceback
            err = f"{e}\n{traceback.format_exc()}"
            self.progress.emit(0.0, f"Moondream2 load error: {e}")
            return None, None, None, err
        finally:
            sys.stdout, sys.stderr = _saved_out, _saved_err

    def _moondream_infer_one(self, img, model, tokenizer, device):
        """Inference for Moondream2.
        Newer revisions (2025+): model.caption(img, length='long')
        Older revisions (2024-08-26): model.encode_image + answer_question
        """
        # Use encode_image + answer_question — works across all revisions
        enc = model.encode_image(img)
        return model.answer_question(
            enc, "Describe this image in detail.", tokenizer).strip()
