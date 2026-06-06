"""
batch_cli.py — headless LoRA Health batch analysis.

Runs the exact same checks as the GUI Batch Compare tab, but from the command line
so it can be scripted / run on a training-output folder without opening the app.
It imports the analyzer's own functions (no logic is duplicated), so results match
the GUI byte-for-byte. Console output is plain ASCII (safe on the Windows cp1252
console) even though the GUI uses colour badges.

Usage:
    python batch_cli.py <folder> [--profile identity|concept|outfit|style]
                                 [--preset Strict|Standard|Relaxed]
                                 [--model-type sdxl|sd15] [--json out.json]

Every *.safetensors in <folder> is analysed and ranked; the best checkpoint for the
chosen profile is highlighted. If a kohya/sd-scripts log.txt sits next to the files,
its training context (TOS, loss curve, convergence) is summarised too.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # for `health_page`
sys.path.insert(0, os.path.dirname(_HERE))      # repo root, for `shared.theme`
import health_page as hp   # noqa: E402  (imports PySide6 but never creates a QApplication)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "health_config.json")

# GUI profile labels -> internal _score_result profile keys
_PROFILE_MAP = {
    "concept / pose": "concept", "concept": "concept", "pose": "concept",
    "character / identity": "identity", "identity": "identity", "character": "identity",
    "outfit": "outfit", "style": "style",
}

# kohya step filename: "{name}-step{N}.safetensors"; final save has no step suffix
_STEP_RE = re.compile(r"-step0*(\d+)\.safetensors$", re.IGNORECASE)


def _load_config() -> dict:
    cfg = dict(hp.DEFAULTS)
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def _make_get_threshold(cfg: dict):
    """Reproduce HealthPage.get_threshold: per-model override -> preset value."""
    def get_threshold(model_type: str, key: str) -> float:
        ov = str(cfg.get(f"{model_type}_overrides", {}).get(key, "")).strip()
        if ov:
            try:
                v = float(ov)
                if v >= 0:
                    return v
            except ValueError:
                pass
        preset_name = cfg.get(f"{model_type}_preset", "Standard")
        presets = hp.THRESHOLD_PRESETS.get(model_type, hp.THRESHOLD_PRESETS["sd15"])
        return presets.get(preset_name, presets["Standard"])[key]
    return get_threshold


def _step_of(path: str) -> int:
    m = _STEP_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else 1 << 30   # final (no-step) sorts last


def _read_metadata(path: str) -> dict:
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.metadata() or {}
    except Exception:
        return {}


def analyse_folder(folder: str, profile: str, model_type_override: str | None,
                   get_threshold) -> tuple[list[dict], dict]:
    """Return (rows, log_summary). rows are per-checkpoint dicts with label + score."""
    files = sorted(
        (os.path.join(folder, f) for f in os.listdir(folder)
         if f.lower().endswith(".safetensors")),
        key=_step_of,
    )
    if not files:
        return [], {}

    total_steps = max((_step_of(p) for p in files if _step_of(p) < (1 << 30)), default=None)

    # training context from the kohya log + final checkpoint metadata
    log_summary: dict = {}
    log_path = os.path.join(folder, "log.txt")
    if os.path.isfile(log_path):
        log_summary = hp._parse_kohya_log(log_path, _read_metadata(files[-1]))

    rows = []
    for path in files:
        step = _step_of(path)
        try:
            result = hp._analyse(path, get_threshold, model_type_override=model_type_override)
        except Exception as e:
            rows.append({"name": os.path.basename(path), "step": step,
                         "error": str(e), "score": float("inf")})
            continue
        mt = result.get("model_type", "sdxl")
        mag_warn = get_threshold(mt, "mag_warn")
        mag_fail = get_threshold(mt, "mag_fail")
        label, _color = hp._batch_label(result, mag_warn, mag_fail)
        score = hp._score_result(result, profile=profile, mag_warn=mag_warn,
                                 mag_fail=mag_fail,
                                 checkpoint_step=(None if step >= (1 << 30) else step),
                                 total_steps=total_steps)
        met = result.get("metrics", {})
        rows.append({
            "name": os.path.basename(path), "step": step, "label": label,
            "overall": result["overall"], "mean_mag": met.get("mean_mag", 0.0),
            "dead": met.get("total_dead", 0), "worst_bal": met.get("worst_balance"),
            "eff": met.get("mean_efficiency"), "rank": result.get("rank", "?"),
            "model_type": mt, "score": score,
            "mag_warn": mag_warn, "mag_fail": mag_fail,
        })
    return rows, log_summary


def _fmt_step(step: int) -> str:
    return "final" if step >= (1 << 30) else str(step)


def print_report(folder: str, profile: str, rows: list[dict], log: dict) -> None:
    print("=" * 92)
    print(f"LoRA Health - Batch Analysis   ({profile} profile)")
    print(f"Folder: {folder}")
    print("=" * 92)

    if log:
        eff = log.get("eff_batch")
        tos = log.get("actual_tos")
        print("Training context (from log.txt + metadata):")
        print(f"  images={log.get('image_count')}  total_steps={log.get('total_steps')}  "
              f"eff_batch={eff}  TOS={tos:.1f}" if tos else
              f"  images={log.get('image_count')}  total_steps={log.get('total_steps')}  eff_batch={eff}")
        print(f"  rank={log.get('network_rank')}  alpha={log.get('network_alpha')}  "
              f"loss {log.get('start_loss'):.4f} -> {log.get('end_loss'):.4f}  "
              f"converged={log.get('converged')}  late_cv={log.get('late_cv'):.3f}")
        print("-" * 92)

    hdr = f"{'checkpoint':<12}{'overall':<20}{'mean_mag':<12}{'dead':<6}{'worst_bal':<12}{'eff':<8}{'score':<10}"
    print(hdr)
    print("-" * 92)
    best = min((r for r in rows if r.get("score") not in (None,)), key=lambda r: r["score"], default=None)
    for r in rows:
        if "error" in r:
            print(f"{_fmt_step(r['step']):<12}ERROR: {r['error']}")
            continue
        warn, fail = r["mag_warn"], r["mag_fail"]
        mag = r["mean_mag"]
        flag = "!!" if mag >= fail else ("!" if mag >= warn else "")
        wb = r["worst_bal"]
        wb_s = "inf" if wb in (None, float("inf")) else f"{wb:.0f}x"
        eff = r["eff"]
        eff_s = "-" if eff is None else f"{eff*100:.0f}%"
        sc = r["score"]
        sc_s = "inf" if sc == float("inf") else f"{sc:.1f}"
        mark = "  <== BEST" if (best is not None and r is best) else ""
        print(f"{_fmt_step(r['step']):<12}{r['label']:<20}{mag:.4f}{flag:<4}{'':2}"
              f"{r['dead']:<6}{wb_s:<12}{eff_s:<8}{sc_s:<10}{mark}")
    print("-" * 92)
    if best is not None and "error" not in best:
        print(f"Recommended ({profile}): {best['name']}  "
              f"[{best['label']}, mean_mag={best['mean_mag']:.4f}, score={best['score']:.1f}]")
    print("=" * 92)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Headless LoRA Health batch analysis.")
    ap.add_argument("folder", help="training-output folder containing .safetensors checkpoints")
    ap.add_argument("--profile", default=None,
                    help="identity | concept | outfit | style (default: config batch_profile)")
    ap.add_argument("--preset", default=None, help="Strict | Standard | Relaxed (override config)")
    ap.add_argument("--model-type", default=None, help="force sdxl | sd15 (default: auto-detect)")
    ap.add_argument("--json", default=None, help="also write the raw rows to this JSON path")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.folder):
        print(f"Not a folder: {args.folder}", file=sys.stderr)
        return 2

    cfg = _load_config()
    if args.preset:
        cfg["sdxl_preset"] = cfg["sd15_preset"] = args.preset
    profile_label = (args.profile or cfg.get("batch_profile", "Concept / Pose")).lower()
    profile = _PROFILE_MAP.get(profile_label, "concept")

    get_threshold = _make_get_threshold(cfg)
    mt = args.model_type.lower() if args.model_type else None
    rows, log = analyse_folder(args.folder, profile, mt, get_threshold)
    if not rows:
        print(f"No .safetensors files found in {args.folder}", file=sys.stderr)
        return 1

    print_report(args.folder, profile, rows, log)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"folder": args.folder, "profile": profile, "rows": rows, "log": log},
                      f, indent=2, default=str)
        print(f"(rows written to {args.json})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
