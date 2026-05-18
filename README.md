# Lora Training Suite v2.0

A desktop application for managing SDXL LoRA training workflows — built with PySide6.

Designed by **Zero** | Built by **Jarvis**

---

## Features

### Tag Handler
- Browse and edit image datasets with a card-based gallery
- Auto-tag with **WD14 Ensemble** (ConvNextV2 + EVA02-Large)
- Generate natural language captions with **JoyCaption Beta One** (formal, informal, training prompt, booru, art critic, and more)
- Fast captioning with **Moondream2** (lower VRAM, quicker turnaround)
- Hybrid mode: WD14 tags + caption in one file
- Extra Instructions field: steer caption output per run (e.g. "Focus on outfit and accessories.")
- Upscale images with **Real-ESRGAN** (photo and anime modes)
- Batch tag operations: add, remove, replace, shuffle, sort
- Caption Find & Replace preserves scroll position — the gallery does not jump back to the top after each replacement
- Tag frequency chart and profile system
- Model integrity check on startup — detects corrupt or incomplete downloads before loading

### Randomizer
- Background removal with **BRIA RMBG-2.0** (realism) and **ToonOut** (anime)
- Pose and expression randomization
- Copy matching `.txt` tag files alongside saved outputs

### LoRA Calculator
- TOS-based step calculations for AI-Toolkit and Kohya
- YAML config export
- Training log with time estimates

### Enhancer
- Upscale images with **Real-ESRGAN x4+** (Realistic or Anime model)
- Scale modes: 1x, 2x, 4x, or Custom dimensions (aspect-ratio locked)
- Adjustment sliders: Minor Denoise, Minor Deblur, Fix Compression, Saturation (−10 to +10), Contrast (−10 to +10)
- **All adjustments are applied at the original source resolution before upscaling.**
  This means the upscaler receives a pre-corrected image, producing cleaner results.
  At 1x scale, the preview is identical to the final output — what you see is what you get.
- Live preview on slider release (1x only — higher scales require clicking Enhance)

### Face Swap
- Swap faces using **INSwapper** with optional **GFPGAN** face restoration
- Multi-face blending: select multiple source faces and enable **Blend** to merge them (mean embedding)
- Supports face image files and pre-built face safetensor models
- Detection threshold, max faces, face order, and gender filter controls
- **Batch Swap:** browse a folder to load all images, then click **Batch Swap All** to apply the selected face/model to every image in one pass
  - Output files are saved with a `-FS` suffix (e.g. `photo.jpg` → `photo-FS.jpg`)
  - Enable **Copy tags** to copy matching `.txt` tag files to the output folder, also renamed with `-FS`
  - **Intended workflow — real person anonymization:** select multiple generic AI faces in the panel, enable **Blend**, then run Batch Swap All. The blended face is a unique synthetic identity that does not correspond to any real person, allowing the dataset to be used or published on platforms that prohibit real-person imagery.

### LoRA Health
- Load any `.safetensors` LoRA file and run structural checks without inference
- **File Integrity** — verifies kohya hash metadata is present
- **NaN / Inf** — scans every tensor for corrupted values
- **Rank Consistency** — shape agreement between lora_down / lora_up and metadata
- **Alpha/Rank Ratio** — checks declared alpha relative to rank against community bounds
- **Rank Range** — validates rank is within recommended range per model type (SD1.5 / SDXL)
- **Overbaked** — detects overtrained LoRAs via elevated global lora_up mean magnitude
- **Module Analysis** — breaks down Dead Layers and Layer Balance per architectural group:
  - UNet Cross-Attention (`attn2`) · UNet Self-Attention (`attn1`) · UNet Feedforward (`ff_net`) · Text Encoder (`lora_te*`)
  - Compares like-for-like layers within each group, so a near-zero `to_k`/`to_v` in cross-attention (normal for AI-Toolkit training) doesn't pollute the self-attention or feedforward result
- **Batch Compare** — point to a training output folder to rank all `.safetensors` candidates at once:
  - Runs all 8 checks on every file in the background with a live progress bar
  - Scores each candidate (lower = better): NaN/Inf → disqualified; penalty points for fail/warn checks, overbaked magnitude, dead layers, and layer imbalance
  - Highlights the best candidate with a Copy Path button
  - **Open in Analyze ↗** on any row loads that file into the single-file tab for full module inspection
- Auto-detects SD 1.5 vs SDXL; manual override via dropdown
- File metadata panel: filename, model type, size, rank, alpha, a/r ratio, layer count, base model
- Drag-and-drop file input
- Configurable thresholds — Strict / Standard / Relaxed presets per model type, with per-threshold manual overrides (amber fields, same pattern as Calculator TOS)

### Launcher
- Embedded Chromium browser (no Chrome/Edge dependency)
- Open any local WebUI or external URL in a tab
- Supports: SD.next, AI-Toolkit, Kohya, ComfyUI, and any `http://` or `https://` address
- Per-tab zoom, off-the-record browsing (no cookies on disk)

---

## Requirements

- Windows 10/11
- Python 3.10+ (3.11 recommended)
- NVIDIA GPU recommended for tagging and captioning (8 GB+ VRAM for JoyCaption)
- JoyCaption Beta One loads directly to VRAM via accelerate — does not require 18 GB of system RAM

---

## Installation

1. Clone or download this repository
2. Double-click **`INSTALL.bat`**
3. Follow the prompts — choose your GPU type when asked
4. Once complete, launch with **`run.bat`**

> Models are downloaded automatically on first use (WD14, Real-ESRGAN, RMBG-2.0, ToonOut).
> JoyCaption Beta One (~18 GB) must be downloaded manually before first caption run — see below.

### Downloading JoyCaption Beta One

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="fancyfeast/llama-joycaption-beta-one-hf-llava",
    local_dir=r"<install_path>\tags\models\joycaption",
    local_dir_use_symlinks=False,
)
```

Run this once from the `.venv` Python after installation. The suite checks model integrity on every caption run and will report corrupt or incomplete downloads with a clear error.

---

## Models Used

| Model | Size | License | Use |
|-------|------|---------|-----|
| WD14 ConvNextV2 | ~600 MB | Apache 2.0 | Image tagging |
| WD EVA02-Large | ~600 MB | Apache 2.0 | Image tagging |
| JoyCaption Beta One | ~18 GB | Apache 2.0 + Llama 3.1 | Captioning |
| Moondream2 | ~1.9 GB | Apache 2.0 | Fast captioning |
| BRIA RMBG-2.0 | ~885 MB | CC BY-NC 4.0 ⚠ Non-commercial | Background removal (realism) |
| ToonOut | ~885 MB | MIT | Background removal (anime) |
| Real-ESRGAN x4+ | ~64 MB | BSD-3 | Upscaling (photo) |
| Real-ESRGAN x4+ Anime | ~18 MB | BSD-3 | Upscaling (anime) |

> **BRIA RMBG-2.0** is non-commercial only. Commercial use requires a separate license from [bria.ai](https://bria.ai).

---

## Project Structure

```
Lora Training Suite 2.0/
├── main.py                  # Entry point
├── run.bat                  # Launch script
├── INSTALL.bat              # Installer
├── assets/                  # Icons
├── shared/                  # Theme, config, calc engine
├── launcher/                # Main window + embedded browser
├── tags/                    # Tag Handler page
├── calculator/              # LoRA Calculator page
├── randomizer/              # Randomizer / background removal page
├── faces/                   # Face Swap page
├── enhancer/                # Enhancer / upscaling page
└── health/                  # LoRA Health analyzer
```

---

## License

This project is for personal and educational use.
Third-party model licenses apply — see the **Models Used** table above.
