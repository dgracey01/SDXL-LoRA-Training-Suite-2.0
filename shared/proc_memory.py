"""
shared/proc_memory.py — process-level memory reclaim after a heavy tab closes.

The Suite is single-process: torch's HIP context can't be freed mid-process (it lives until the process
exits), but two things CAN be returned after a heavy module is torn down:
  1. the CUDA/HIP allocator CACHE — empty_cache() hands reserved-but-unused blocks back to the driver
     (only effective AFTER the tensors' Python references are gone, hence we run this on a short delay so
     Qt's deleteLater has fired and gc has collected the page);
  2. Python's freed pages — Windows keeps them in the process working set; a one-shot trim returns them.

IMPORTANT: this module has stdlib-only IMPORTS. torch is touched ONLY if it is already loaded
(sys.modules), never imported here — importing torch into a light module is exactly the cost we avoid.
"""
from __future__ import annotations

import ctypes
import gc
import sys


def _trim_working_set() -> None:
    """Return freed pages to the OS on Windows via SetProcessWorkingSetSize(handle, -1, -1) — the documented
    'release my working set' call (same effect as EmptyWorkingSet). One-shot ONLY: forced trimming in a loop
    thrashes paging. No-op off Windows / on any failure. argtypes are set so the -1 sentinels aren't
    truncated to 32-bit on a 64-bit process."""
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        k32.SetProcessWorkingSetSize(k32.GetCurrentProcess(), -1, -1)
    except Exception:
        pass


def reclaim(trim_working_set: bool = True) -> None:
    """Best-effort reclaim after closing a heavy GPU/model tab:
        Python GC  →  torch CUDA/HIP allocator cache (only if torch is already loaded)  →  working-set trim.
    Never raises. Safe to call when torch was never imported (light tabs) — it just runs gc + the trim."""
    try:
        gc.collect()
    except Exception:
        pass
    torch = sys.modules.get("torch")     # touch torch ONLY if already imported; never import it here
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    if trim_working_set:
        _trim_working_set()
