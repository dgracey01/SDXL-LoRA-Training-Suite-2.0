"""
assets/make_icons.py  —  Convert PNG → multi-size ICO
Run once:  .venv\Scripts\python.exe assets\make_icons.py
"""
from pathlib import Path
from PIL import Image

SIZES   = [16, 32, 48, 64, 128, 256]
ASSETS  = Path(__file__).parent
ICONS   = [
    ("calculator.png", "calculator.ico"),
    ("random.png",     "random.ico"),
    ("launcher.png",   "launcher.ico"),
    ("tags.png",       "tags.ico"),
]

for src_name, dst_name in ICONS:
    src = ASSETS / src_name
    dst = ASSETS / dst_name
    img = Image.open(src).convert("RGBA")
    frames = [img.resize((s, s), Image.LANCZOS) for s in SIZES]
    frames[0].save(
        dst,
        format="ICO",
        sizes=[(s, s) for s in SIZES],
        append_images=frames[1:],
    )
    print(f"[OK] {dst_name}  ({', '.join(str(s) for s in SIZES)})")

print("\nDone.")
