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
- **Rank Consistency** — shape agreement between lora_down / lora_up and metadata. Mixed-rank LoCon/LoHa LoRAs (linear + a smaller conv rank, e.g. linear=96, conv=32) display both ranks. `ss_network_dim` is the **linear** rank, so the metadata is matched only against the dominant (linear) tensor rank — a smaller conv rank is expected and no longer triggers a false "metadata ≠ tensor rank" failure (affected both kohya and AI Toolkit LoCon LoRAs)
- **Alpha/Rank Ratio** — checks declared alpha relative to rank against community bounds
- **Rank Range** — validates rank is within recommended range per model type (SD1.5 / SDXL)
- **Overbaked** — detects overtrained LoRAs via elevated global lora_up mean magnitude
- **Module Analysis** — breaks down Dead Layers and Layer Balance per architectural group:
  - UNet Cross-Attention (`attn2`) · UNet Self-Attention (`attn1`) · UNet Feedforward (`ff_net`) · Text Encoder (`lora_te*`)
  - Compares like-for-like layers within each group, so a near-zero `to_k`/`to_v` in cross-attention (normal for AI-Toolkit training) doesn't pollute the self-attention or feedforward result
- **Structural Zeros** — SDXL conditions on TE1's *penultimate* hidden state, so kohya's LoRA modules on TE1's final layer (`te1…layers_11`) never receive gradient and stay at their zero init. These are reported as an info note and **excluded from the Dead Layers count** — they are a kohya packaging artifact (AI Toolkit doesn't emit them), not a training defect
- **Training Software selector** — AI Toolkit / Kohya / Auto-detect:
  - Auto-detect reads `config.yaml` in the same folder (AI Toolkit leaves one there), then falls back to tensor key heuristics
  - **AI Toolkit mode:** cross-attention balance ratio is excluded from scoring — it is a structural artifact of how AI Toolkit initialises weights, not a defect. The value is still displayed for reference
  - **Kohya mode:** all checks and balance thresholds apply as configured
  - Selection is remembered between sessions
- **Training Log analysis** (AI Toolkit **and** kohya / TrainerXL) — when `log.txt` is present alongside the LoRA file:
  - kohya logs are parsed too: `batch_size` and `gradient_accumulation_steps` are read from the log's "running training" block (kohya leaves them out of the `ss_arguments` metadata), so the effective batch and TOS/steps-per-image are computed correctly instead of being under-reported
  - Dataset image count and steps/image (flags undertrained < 80 or overfit risk > 400)
  - Loss trend: Q1 vs Q4 quarter averages — "still learning" or "plateaued"
  - Loss at this specific checkpoint (±50-step window average)
  - Late-stage noise as coefficient of variation %
  - Full checkpoint loss table with the lowest-loss checkpoint marked
- **Batch Compare** — point to a training output folder to rank all `.safetensors` candidates at once:
  - Runs all checks on every file in the background with a live progress bar
  - **Recommendation Profile** selector chooses the scoring strategy:
    - **Concept / Pose** (default): penalizes fail/warn checks, magnitude, dead layers, and balance — general-purpose; step count is not decisive; near-zero magnitude files are penalized as undercooked regardless of other scores
    - **Character / Identity**: disqualifies overbaked files, then ranks by step count (later = better) and magnitude (higher within safe range = stronger identity)
    - **Outfit / Costume**: disqualifies overbaked files, scores magnitude against a ~65% sweet spot to capture detail without bleeding into skin/hair
    - **Style / Art Direction**: disqualifies overbaked files, prefers lower magnitude (subtle influence over dominance) with a mild step preference
  - **Loss column** — reads `log.txt` once for the folder and shows each checkpoint's window-averaged loss; lowest is highlighted green
  - Winner banner shows steps/image and checkpoint loss alongside score and magnitude
  - Highlights the best candidate with a Copy Path button
  - **Open in Analyze ↗** on any row loads that file into the single-file tab for full module inspection
- **Headless batch CLI** (`health/batch_cli.py`) — runs the exact same analysis from the command line, no GUI:
  - `python health/batch_cli.py <output-folder> --profile identity|concept|outfit|style`
  - Imports the GUI's own `_analyse` / `_batch_label` / `_score_result` (no duplicated logic), so results match the app
  - Prints a ranked checkpoint table with the kohya/AI-Toolkit log's training context (TOS, loss curve, convergence) and marks the recommended checkpoint; `--json` dumps the raw rows
- Auto-detects SD 1.5 vs SDXL; manual override via dropdown
- Model Type, Trainer, and Profile selections remembered between sessions
- File metadata panel: filename, model type, size, rank, alpha, a/r ratio, layer count, base model
- Drag-and-drop file input
- Configurable thresholds — Strict / Standard / Relaxed presets per model type, with per-threshold manual overrides (amber fields, same pattern as Calculator TOS)

### Model Merge
Four-tab workspace for checkpoint and LoRA operations — all processing runs GPU-streamed (low VRAM: 2–3 tensors live at once regardless of model size).

#### Checkpoint Merge
- Merge two checkpoints (A + B) with optional third model (C) as a shared base
- **Six methods** — each with a built-in explanation panel:
  - **Weighted Sum** — linear blend `(1−α)·A + α·B`; simplest and most predictable
  - **Slerp** — spherical interpolation; smoother transitions for stylistically close models
  - **Add Difference** — delta injection `A + α·(B−C)`; graft a concept or style without retraining
  - **TIES** — Trim · Elect Sign · Disjoint Merge; reduces interference when both fine-tunes share a base
  - **DARE** — Drop And Rescale; random delta pruning to decorrelate conflicting parameters
  - **DARE+TIES** — DARE preprocessing followed by TIES sign election; strongest option for divergent fine-tunes
- **Selective Merge** — per-group alpha overrides for Text Encoders, Cross-Attention, Self-Attention, Feedforward, and Other layers
- Built-in and custom named presets (save/load)
- Optional VAE override — swap the VAE from any third file at merge time
- Output precision: fp16 / bf16 / fp32
- Health check runs automatically on every output

#### LoRA Merge
- Combine up to 4 LoRAs into a single file with independent weight and type controls per slot
- Per-slot LoRA Type selector (Character / Pose / Detail / Style / Concept) auto-fills recommended weights
- Combined-weight caution and warn thresholds displayed live
- Output precision selector

#### Bake LoRA
- Permanently bake up to 4 LoRAs into a checkpoint in a single pass
- Per-slot LoRA Type selector auto-fills recommended bake ratios (Character 0.85, Concept 0.70, etc.)
- All slots processed in one read/write cycle — no intermediate files, no VRAM spike
- Health check runs automatically on every output

#### Extract LoRA
- Extract a LoRA by SVD-decomposing the weight delta between a tuned model and its base
- Rank and optional conv rank controls
- Useful for packaging fine-tune diffs as reusable LoRAs

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
├── health/                  # LoRA Health analyzer (health_page.py + batch_cli.py headless CLI)
└── merge/                   # Model Merge (checkpoint merge, LoRA merge/bake/extract)
```

---

## License

This project is for personal and educational use.
Third-party model licenses apply — see the **Models Used** table above.
