# Lora Training Suite v2.0

A desktop application for managing SDXL LoRA training workflows — built with PySide6.

Designed by **Zero** | Built by **Jarvis**

---

## Features

### Tag Handler
- Browse and edit image datasets with a card-based gallery
- Auto-tag with **WD14 Ensemble** (ConvNextV2 + EVA02-Large)
- Generate natural language captions with **JoyCaption Alpha Two**
- Hybrid mode: WD14 tags + JoyCaption captions in one file
- Upscale images with **Real-ESRGAN** (photo and anime modes)
- Batch tag operations: add, remove, replace, shuffle, sort
- Tag frequency chart and profile system

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

---

## Installation

1. Clone or download this repository
2. Double-click **`INSTALL.bat`**
3. Follow the prompts — choose your GPU type when asked
4. Once complete, launch with **`run.bat`**

> Models are downloaded automatically on first use (WD14, Real-ESRGAN, RMBG-2.0, ToonOut).
> JoyCaption (~15 GB) is downloaded on first caption run.

---

## Models Used

| Model | Size | License | Use |
|-------|------|---------|-----|
| WD14 ConvNextV2 | ~600 MB | Apache 2.0 | Image tagging |
| WD EVA02-Large | ~600 MB | Apache 2.0 | Image tagging |
| JoyCaption Alpha Two | ~15 GB | Llama 3 | Captioning |
| BRIA RMBG-2.0 | ~885 MB | BRIA AI ToS | Background removal (realism) |
| ToonOut | ~885 MB | MIT | Background removal (anime) |
| Real-ESRGAN x4+ | ~64 MB | BSD-3 | Upscaling (photo) |
| Real-ESRGAN x4+ Anime | ~18 MB | BSD-3 | Upscaling (anime) |

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
└── enhancer/                # Enhancer / upscaling page
```

---

## License

This project is for personal and educational use.
Third-party model licenses apply — see the **Models Used** table above.
