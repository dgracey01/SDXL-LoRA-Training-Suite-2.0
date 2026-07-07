"""
shared/vram_guard.py — VRAM pre-flight for the Suite's heavy GPU modules (PoserXL, LoRA Health).

WHY THIS EXISTS
  The Suite loads heavy native libs IN-PROCESS: PoserXL pulls in MediaPipe + drives the SDXL engine,
  LoRA Health imports torch for SVD. When VRAM is already occupied — Jarvis routinely leaves LM Studio
  models resident (JoyCaption ~5.8 GB + a brain ~1 GB) and the render engine keeps its SDXL checkpoint
  (~6.7 GB) — opening one of those modules pushes the card over the top and the allocation OOMs. On
  ROCm an OOM is frequently a NATIVE crash (no Python traceback) that kills the whole Suite process.
  (joy_crash.log caught the exact signature: a model loading at "free VRAM: 0.0GB".)

  This module is the pre-flight: before a heavy module opens, free the reclaimable consumers (LM Studio
  models always; the engine checkpoint when the module doesn't need it), then report how much room there
  is so the caller can warn if it still won't fit. It CANNOT catch a native crash — it PREVENTS the OOM
  that causes one. (Crash containment + diagnostics for the ones it can't prevent live in crash_guard.py.)

  stdlib only — must not import torch (importing torch in the Suite process is itself part of the risk).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

# Hide the console window every subprocess call would otherwise pop under the pythonw (windowless)
# Suite — those stray conhost windows are what pile up in Task Manager.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# The render engine PoserXL shares (poserxl.engine_manager.BASE_URL).
DEFAULT_ENGINE_URL = "http://127.0.0.1:3681"
# The `lms` CLI (load/unload/ps). Same path Jarvis uses; falls back to PATH lookup if absent.
_LMS_CANDIDATES = ("C:/Users/Dago/.lmstudio/bin/lms.exe", "lms")


def _lms_exe() -> str:
    for c in _LMS_CANDIDATES:
        if c == "lms" or os.path.exists(c):
            return c
    return "lms"


def _get(url: str, timeout: float = 6.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def _post(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, data=b"", headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ── VRAM snapshot ────────────────────────────────────────────────────────────────────────────────
def snapshot(engine_url: str = DEFAULT_ENGINE_URL) -> dict | None:
    """Best-effort {used_gb,total_gb,free_gb,source} or None if VRAM can't be read.

    Prefers the render engine's /memory (authoritative, no torch import, usually up). Falls back to
    rocm-smi for AMD when the engine is down. None => proceed but treat 'fit' as unknown (warn softly)."""
    m = _get(engine_url + "/sdapi/v1/memory")
    try:
        cu = (m or {}).get("cuda", {}).get("system", {})
        used, total = cu.get("used"), cu.get("total")
        if used is not None and total:
            return {"used_gb": round(used / 1e9, 2), "total_gb": round(total / 1e9, 2),
                    "free_gb": round((total - used) / 1e9, 2), "source": "engine"}
    except Exception:
        pass
    # engine down → try rocm-smi (AMD). Silent if the tool isn't present.
    try:
        p = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                           capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW)
        d = json.loads(p.stdout or "{}")
        for card in d.values():
            tot = int(card.get("VRAM Total Memory (B)", 0))
            usd = int(card.get("VRAM Total Used Memory (B)", 0))
            if tot:
                return {"used_gb": round(usd / 1e9, 2), "total_gb": round(tot / 1e9, 2),
                        "free_gb": round((tot - usd) / 1e9, 2), "source": "rocm-smi"}
    except Exception:
        pass
    return None


# ── reclaim ──────────────────────────────────────────────────────────────────────────────────────
def _loaded_llms(lms_exe: str) -> list[str]:
    """Names of LM Studio models currently resident (via `lms ps`), best-effort."""
    try:
        p = subprocess.run([lms_exe, "ps"], capture_output=True, text=True, timeout=20,
                           creationflags=_NO_WINDOW)
        out = p.stdout or ""
        names = []
        for ln in out.splitlines():
            s = ln.strip()
            if not s or s.lower().startswith(("identifier", "loaded", "---")):
                continue
            names.append(s.split()[0])
        return names
    except Exception:
        return []


def free_llm(lms_exe: str | None = None) -> dict:
    """Evict ALL LM Studio models (`lms unload --all`). Cheap to reload; they are pure competition
    for a heavy Suite module and are the usual thing Jarvis leaves resident."""
    lms_exe = lms_exe or _lms_exe()
    before = _loaded_llms(lms_exe)
    if not before:
        return {"target": "llm", "unloaded": [], "ok": True, "note": "none loaded"}
    try:
        p = subprocess.run([lms_exe, "unload", "--all"], capture_output=True, text=True, timeout=60,
                           creationflags=_NO_WINDOW)
        return {"target": "llm", "unloaded": before, "ok": p.returncode == 0,
                "out": (p.stdout or p.stderr or "")[-200:]}
    except Exception as e:
        return {"target": "llm", "unloaded": [], "ok": False, "err": f"{type(e).__name__}: {e}"}


def free_engine(engine_url: str = DEFAULT_ENGINE_URL) -> dict:
    """Unload the render engine's SDXL checkpoint (/sdapi/v1/unload-checkpoint). Only for modules that
    do NOT need the engine (e.g. Health) — PoserXL needs the checkpoint, so it keeps it."""
    if not _get(engine_url + "/sdapi/v1/memory", timeout=3.0):
        return {"target": "engine", "unloaded": False, "ok": True, "note": "engine not up"}
    st, err = _post(engine_url + "/sdapi/v1/unload-checkpoint")
    return {"target": "engine", "unloaded": err is None, "ok": err is None, "err": err}


# ── the pre-flight the launcher calls ──────────────────────────────────────────────────────────────
def preflight(module: str, *, engine_url: str = DEFAULT_ENGINE_URL, lms_exe: str | None = None,
              need_gb: float = 3.0, free_engine_ckpt: bool = False, settle: float = 1.5) -> dict:
    """Free reclaimable VRAM before `module` opens, then report whether it should fit.

    module           label for the report/dialog (e.g. "PoserXL", "LoRA Health").
    need_gb          headroom we want free AFTER reclaiming — advisory, drives the 'fit' verdict/warning.
    free_engine_ckpt unload the engine's SDXL checkpoint too (Health=True; PoserXL=False, it needs it).

    Returns {module, before, after, freed:[...], fit: True|False|None, message}. Never raises."""
    lms_exe = lms_exe or _lms_exe()
    before = snapshot(engine_url)
    freed = [free_llm(lms_exe)]
    if free_engine_ckpt:
        freed.append(free_engine(engine_url))
    if any(f.get("unloaded") for f in freed):
        time.sleep(settle)                      # let the allocator actually release before we re-measure
    after = snapshot(engine_url)

    free_after = (after or {}).get("free_gb")
    fit = None if free_after is None else (free_after >= need_gb)
    reclaimed = [n for f in freed for n in (f.get("unloaded") if isinstance(f.get("unloaded"), list) else
                                            (["engine checkpoint"] if f.get("unloaded") else []))]
    if reclaimed:
        head = f"Freed VRAM for {module}: unloaded " + ", ".join(reclaimed) + "."
    else:
        head = f"No reclaimable models were resident before opening {module}."
    if fit is False:
        msg = (head + f"\n\nOnly ~{free_after:.1f} GB free (want ≥{need_gb:.0f} GB). "
               f"{module} may still run out of VRAM and crash. Open anyway?")
    elif fit is True:
        msg = head + f" ~{free_after:.1f} GB free."
    else:
        msg = head + " (VRAM level couldn't be read — proceeding.)"
    return {"module": module, "before": before, "after": after, "freed": freed,
            "fit": fit, "message": msg}


if __name__ == "__main__":
    import sys
    mod = sys.argv[1] if len(sys.argv) > 1 else "test"
    print(json.dumps(preflight(mod, free_engine_ckpt=(mod.lower().startswith("health"))), indent=2))
