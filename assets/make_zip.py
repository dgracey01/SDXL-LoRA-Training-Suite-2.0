"""
assets/make_zip.py  —  Build distribution zip for Lora Training Suite 2.0
Run once:  .venv\Scripts\python.exe assets\make_zip.py
"""
import os
import zipfile
from pathlib import Path

ROOT    = Path(__file__).parent.parent
OUTFILE = ROOT / "Lora Training Suite 2.0.zip"

# ── Files/dirs to include ─────────────────────────────────────────────────────
# Top-level files
TOP_FILES = [
    "main.py",
    "run.bat",
    "INSTALL.bat",
]

# Module directories — only .py files (no __pycache__, no user data, no models)
MODULE_DIRS = {
    "shared":     {"ext": {".py"}},
    "launcher":   {"ext": {".py"}},
    "calculator": {"ext": {".py"}, "extra_files": ["run_calculator.bat"]},
    "tags":       {"ext": {".py"}, "extra_files": ["run_tags.bat", "tags_data.json"]},
    "randomizer": {"ext": {".py"}, "extra_files": ["run_randomizer.bat"]},
}

# Assets — icons and source PNGs (not make_icons.py or make_zip.py)
ASSET_EXTS = {".ico", ".png"}

# ── Build ─────────────────────────────────────────────────────────────────────
if OUTFILE.exists():
    OUTFILE.unlink()

added = []

with zipfile.ZipFile(OUTFILE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:

    # Top-level files
    for name in TOP_FILES:
        src = ROOT / name
        if src.exists():
            zf.write(src, name)
            added.append(name)

    # Module directories
    for folder, cfg in MODULE_DIRS.items():
        folder_path = ROOT / folder
        if not folder_path.exists():
            continue
        # .py files (no __pycache__)
        for f in sorted(folder_path.iterdir()):
            if f.is_file() and f.suffix in cfg["ext"]:
                arc = f"{folder}/{f.name}"
                zf.write(f, arc)
                added.append(arc)
        # extra files (bat, json, etc.)
        for extra in cfg.get("extra_files", []):
            src = folder_path / extra
            if src.exists():
                arc = f"{folder}/{extra}"
                zf.write(src, arc)
                added.append(arc)

    # Assets
    assets_path = ROOT / "assets"
    for f in sorted(assets_path.iterdir()):
        if f.is_file() and f.suffix in ASSET_EXTS:
            arc = f"assets/{f.name}"
            zf.write(f, arc)
            added.append(arc)

print(f"\n{'=' * 60}")
print(f"  {OUTFILE.name}")
print(f"  {OUTFILE.stat().st_size / 1024 / 1024:.1f} MB   ({len(added)} files)")
print(f"{'=' * 60}")
for a in added:
    print(f"  + {a}")
print()
