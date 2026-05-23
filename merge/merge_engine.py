"""
Standalone merge engine for checkpoints and LoRAs.
Dependencies: torch, safetensors only (no SD.Next or Kohya imports).

GPU streaming mode: each tensor is loaded directly from disk to VRAM via
safe_open(device="cuda"), processed, then immediately moved to CPU.
Peak VRAM = 2–3 tensors at once regardless of total model size.
"""
from __future__ import annotations

import gc
import json
import os
import struct
import torch
from contextlib import contextmanager
from typing import Callable

from .merge_methods import METHODS, TWO_MODEL_METHODS, weighted_sum


# ── Device helpers ────────────────────────────────────────────────────────────

def get_available_vram_gb() -> float:
    """Return free VRAM in GB, or 0.0 if no CUDA device."""
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def get_total_vram_gb() -> float:
    """Return total VRAM in GB, or 0.0 if no CUDA device."""
    if not torch.cuda.is_available():
        return 0.0
    _, total = torch.cuda.mem_get_info()
    return total / (1024 ** 3)


def resolve_device(pref: str) -> str:
    """
    Resolve a device preference string to an actual torch device string.
    "auto"  → "cuda" if available, else "cpu"
    "cuda"  → "cuda" (raises later if unavailable)
    "cpu"   → "cpu"
    Works identically on ROCm (AMD) and CUDA (Nvidia) — both report as "cuda".
    """
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def _gc(device: str):
    """Release cached GPU memory after processing a batch of keys."""
    if "cuda" in device:
        torch.cuda.empty_cache()
    gc.collect()


# ── Model size estimation (reads only the safetensors header) ─────────────────

_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "BF16": 2, "F16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def estimate_model_gb(path: str) -> float:
    """
    Estimate loaded model size in GB by reading the safetensors header only.
    Header is at most a few hundred KB — no tensor data is read.
    """
    try:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header     = json.loads(f.read(header_len))
        total = 0
        for k, v in header.items():
            if k == "__metadata__":
                continue
            shape = v.get("shape", [])
            dtype = v.get("dtype", "F32")
            n = 1
            for s in shape:
                n *= s
            total += n * _DTYPE_BYTES.get(dtype, 4)
        return total / (1024 ** 3)
    except Exception:
        return os.path.getsize(path) / (1024 ** 3)


# ── I/O helpers ───────────────────────────────────────────────────────────────

@contextmanager
def _open(path: str, device: str):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device=device) as f:
        yield f


@contextmanager
def _maybe_open(path: str | None, device: str):
    """Open a safetensors file, or yield None if path is None."""
    if path is None:
        yield None
    else:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device=device) as f:
            yield f


def _save(tensors: dict, path: str, metadata: dict | None = None):
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()},
              path, metadata=metadata or {})


# ── Checkpoint merge ──────────────────────────────────────────────────────────

def _group_alpha(key: str, alpha: float, group_alphas: dict | None) -> float:
    """
    Return per-group alpha for a key when selective merge mode is active.
    group_alphas keys: "te", "ca", "sa", "ff", "other"
    Falls back to global alpha when group_alphas is None.
    """
    if group_alphas is None:
        return alpha
    k = key.lower()
    if key.startswith("conditioner.embedders."):
        return group_alphas.get("te", alpha)
    if ".attn2." in k:
        return group_alphas.get("ca", alpha)
    if ".attn1." in k:
        return group_alphas.get("sa", alpha)
    if "ff_net" in k or "geglu" in k:
        return group_alphas.get("ff", alpha)
    return group_alphas.get("other", alpha)


def check_checkpoint_health(output_path: str, reference_path: str | None = None,
                            merge_stats: dict | None = None) -> dict:
    """
    Run 5 quick diagnostic checks on a merged or baked checkpoint.
    Returns dict with keys: key_count, magnitude_drift, vae_integrity,
    alpha_consistency, merge_confirmed.
    Each value: {"status": "pass"|"warn"|"fail", "label": str, "detail": str}

    merge_stats — optional {"merged": int, "passthru"/"skipped": int} from the
    engine call.  Drives the merge_confirmed check; omitting it skips the check.
    """
    from safetensors import safe_open

    out_keys:       list[str]   = []
    out_mags:       list[float] = []
    vae_keys:       list[str]   = []
    vae_nan:        bool        = False
    alpha_tensors:  dict        = {}

    with safe_open(output_path, framework="pt", device="cpu") as f:
        out_keys = list(f.keys())
        for key in out_keys:
            t = f.get_tensor(key).float()
            if t.numel() > 1:
                out_mags.append(t.abs().mean().item())
            if key.startswith("first_stage_model."):
                vae_keys.append(key)
                if torch.isnan(t).any() or torch.isinf(t).any():
                    vae_nan = True
            if key in ("alphas_cumprod", "betas", "alphas_cumprod_prev",
                       "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"):
                alpha_tensors[key] = t

    out_mean  = sum(out_mags) / len(out_mags) if out_mags else 0.0
    out_count = len(out_keys)

    ref_keys: list[str]  = []
    ref_mean: float | None = None
    if reference_path and os.path.isfile(reference_path):
        ref_mags: list[float] = []
        with safe_open(reference_path, framework="pt", device="cpu") as f:
            ref_keys = list(f.keys())
            for key in ref_keys:
                t = f.get_tensor(key).float()
                if t.numel() > 1:
                    ref_mags.append(t.abs().mean().item())
        ref_mean = sum(ref_mags) / len(ref_mags) if ref_mags else None

    results: dict = {}

    # 1. Key count delta
    if ref_keys:
        ref_count = len(ref_keys)
        ratio     = out_count / ref_count if ref_count > 0 else 1.0
        delta     = out_count - ref_count
        status    = "pass" if ratio >= 0.95 else "warn" if ratio >= 0.80 else "fail"
        results["key_count"] = {
            "status": status, "label": "Key count delta",
            "detail": f"{out_count} keys  (ref {ref_count}, delta {delta:+d})",
        }
    else:
        status = "pass" if out_count > 100 else "warn" if out_count > 10 else "fail"
        results["key_count"] = {
            "status": status, "label": "Key count delta",
            "detail": f"{out_count} keys  (no reference)",
        }

    # 2. Weight magnitude drift
    if ref_mean is not None and ref_mean > 0:
        drift  = abs(out_mean - ref_mean) / ref_mean
        status = "pass" if drift < 0.10 else "warn" if drift < 0.30 else "fail"
        results["magnitude_drift"] = {
            "status": status, "label": "Weight magnitude drift",
            "detail": f"mean |w| {out_mean:.5f}  (drift {drift:.1%} from ref)",
        }
    else:
        status = "pass" if out_mean < 1.0 else "warn" if out_mean < 5.0 else "fail"
        results["magnitude_drift"] = {
            "status": status, "label": "Weight magnitude drift",
            "detail": f"mean |w| {out_mean:.5f}",
        }

    # 3. VAE integrity
    if not vae_keys:
        results["vae_integrity"] = {
            "status": "warn", "label": "VAE integrity",
            "detail": "No first_stage_model.* keys found",
        }
    elif vae_nan:
        results["vae_integrity"] = {
            "status": "fail", "label": "VAE integrity",
            "detail": "NaN / Inf detected in VAE tensors",
        }
    else:
        results["vae_integrity"] = {
            "status": "pass", "label": "VAE integrity",
            "detail": f"{len(vae_keys)} VAE keys — all finite",
        }

    # 4. Alpha key consistency (noise scheduler)
    # SD1.5 embeds scheduler tensors (alphas_cumprod etc.) directly in the safetensors.
    # SDXL single-file checkpoints do not — the scheduler is defined externally.
    # Detect SDXL by the presence of conditioner.* or model.diffusion_model.* with
    # SDXL-specific keys so we don't false-warn on every SDXL merge/bake.
    is_sdxl = any(
        k.startswith("conditioner.") or k.startswith("model.diffusion_model.output_blocks.8.")
        for k in out_keys
    )
    if not alpha_tensors:
        if is_sdxl:
            results["alpha_consistency"] = {
                "status": "info", "label": "Alpha key consistency",
                "detail": "SDXL format — scheduler defined externally, not embedded in checkpoint",
            }
        else:
            results["alpha_consistency"] = {
                "status": "warn", "label": "Alpha key consistency",
                "detail": "alphas_cumprod not found — may be partial checkpoint",
            }
    else:
        issues: list[str] = []
        for key, t in alpha_tensors.items():
            vals = t.flatten()
            if key in ("alphas_cumprod", "betas", "alphas_cumprod_prev"):
                if vals.min() < 0 or vals.max() > 1.0 + 1e-4:
                    issues.append(f"{key} out of [0, 1]")
                elif len(vals) > 1 and key != "betas":
                    # Tolerance of 1e-4 accounts for fp16 quantization noise
                    # at the schedule tail (timesteps ~900-999) where values
                    # approach 1e-5 and adjacent fp16 steps are ~1-6e-6 apart.
                    if not (vals[:-1] >= vals[1:] - 1e-4).all():
                        issues.append(f"{key} not monotone decreasing")
        if issues:
            status = "warn" if len(issues) == 1 else "fail"
            results["alpha_consistency"] = {
                "status": status, "label": "Alpha key consistency",
                "detail": "; ".join(issues),
            }
        else:
            results["alpha_consistency"] = {
                "status": "pass", "label": "Alpha key consistency",
                "detail": f"{', '.join(alpha_tensors.keys())} — valid",
            }

    # 5. Merge Confirmed
    if merge_stats is not None:
        merged   = merge_stats.get("merged", 0)
        extra    = merge_stats.get("passthru", merge_stats.get("skipped", 0))
        extra_lbl = "passthrough" if "passthru" in merge_stats else "skipped"
        if merged == 0:
            results["merge_confirmed"] = {
                "status": "fail", "label": "Merge Confirmed",
                "detail": f"0 layers merged ({extra} {extra_lbl}) — output is unchanged from input",
            }
        else:
            results["merge_confirmed"] = {
                "status": "pass", "label": "Merge Confirmed",
                "detail": f"{merged} layers merged ({extra} {extra_lbl})",
            }

    return results


def merge_checkpoints(
    model_a:     str,
    model_b:     str,
    model_c:     str | None,
    method:      str,
    alpha:       float,
    beta:        float,
    density:     float,
    output_path: str,
    precision:   str = "fp16",
    device:      str = "auto",
    vae_path:    str | None = None,
    group_alphas: dict | None = None,
    progress_fn: Callable[[int, int, str], None] | None = None,
) -> dict:
    """
    group_alphas — optional dict for selective merge mode:
      {"te": float, "ca": float, "sa": float, "ff": float, "other": float}
    Each value overrides alpha for that architectural group.
    None = uniform alpha for all keys (standard mode).
    """
    """
    Merge two (or three) checkpoint files key-by-key.

    GPU streaming: each tensor is loaded directly from disk to VRAM,
    processed, then immediately moved to CPU. Peak VRAM ≈ 2–3 tensors
    at once — no VRAM limit on model size.
    """
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    out_dtype = dtype_map.get(precision, torch.float16)

    dev = resolve_device(device)
    fn  = METHODS.get(method, weighted_sum)

    out:           dict = {}
    meta_a:        dict = {}
    merged_count   = 0
    passthru_count = 0

    # Pre-load replacement VAE keys if provided
    vae_replace: dict[str, torch.Tensor] = {}
    if vae_path and os.path.isfile(vae_path):
        with _open(vae_path, "cpu") as fv:
            for k in fv.keys():
                if k.startswith("first_stage_model."):
                    vae_replace[k] = fv.get_tensor(k).cpu().to(out_dtype)

    with _open(model_a, dev) as fa, \
         _open(model_b, dev) as fb, \
         _maybe_open(model_c, dev) as fc:

        meta_a  = fa.metadata() or {}
        keys_b  = set(fb.keys())
        keys_c  = set(fc.keys()) if fc is not None else set()
        all_keys = list(fa.keys())
        total    = len(all_keys)

        for i, key in enumerate(all_keys):
            if progress_fn and i % 20 == 0:
                progress_fn(i, total, key)

            a = fa.get_tensor(key).float()

            # Use replacement VAE tensor if provided
            if key.startswith("first_stage_model.") and vae_replace:
                out[key] = vae_replace.pop(key, a.cpu().to(out_dtype))
                del a
                continue

            if key not in keys_b:
                out[key] = a.cpu().to(out_dtype)
                passthru_count += 1
                del a
                continue

            b = fb.get_tensor(key).float()
            eff_alpha = _group_alpha(key, alpha, group_alphas)

            if fc is not None and key in keys_c and method not in TWO_MODEL_METHODS:
                c   = fc.get_tensor(key).float()
                res = fn(a, b, c, alpha=eff_alpha, beta=beta, density=density)
                del c
            elif method in TWO_MODEL_METHODS:
                res = fn(a, b, alpha=eff_alpha)
            else:
                res = weighted_sum(a, b, alpha=eff_alpha)

            out[key] = res.cpu().to(out_dtype)
            merged_count += 1
            del a, b, res

            # Free VRAM every 100 keys when on GPU
            if "cuda" in dev and i % 100 == 99:
                _gc(dev)

    # Add any VAE keys from replacement not present in model_a
    for k, t in vae_replace.items():
        if k not in out:
            out[k] = t

    if progress_fn:
        progress_fn(total, total,
                    f"Merged {merged_count} layers ({passthru_count} A-only passthrough) — Saving…")

    _gc(dev)
    _save(out, output_path, meta_a)
    return {"merged": merged_count, "passthru": passthru_count}


# ── LoRA delta helpers ────────────────────────────────────────────────────────

def _lora_delta(tensors: dict, dk: str) -> torch.Tensor | None:
    """Compute the full-rank delta for one LoRA layer from a pre-loaded dict."""
    uk = dk.replace("lora_down", "lora_up").replace("lora_A", "lora_B")
    if uk not in tensors:
        return None
    down = tensors[dk].float()
    up   = tensors[uk].float()

    alpha_key = dk.rsplit(".", 1)[0] + ".alpha"
    alpha = float(tensors[alpha_key].item()) if alpha_key in tensors else float(down.shape[0])
    dim   = down.shape[0]
    scale = alpha / dim

    if down.dim() == 4:
        d = (up.view(up.shape[0], -1) @ down.view(down.shape[0], -1)).view(
            up.shape[0], down.shape[1], down.shape[2], down.shape[3])
    else:
        d = up @ down

    return d * scale


def _lora_delta_from_handles(f, dk: str) -> torch.Tensor | None:
    """Compute delta for one layer using open safetensors file handles (streaming)."""
    uk = dk.replace("lora_down", "lora_up").replace("lora_A", "lora_B")
    all_k = set(f.keys())
    if uk not in all_k:
        return None

    down = f.get_tensor(dk).float()
    up   = f.get_tensor(uk).float()

    alpha_key = dk.rsplit(".", 1)[0] + ".alpha"
    alpha = float(f.get_tensor(alpha_key).item()) if alpha_key in all_k else float(down.shape[0])
    dim   = down.shape[0]
    scale = alpha / dim

    if down.dim() == 4:
        d = (up.view(up.shape[0], -1) @ down.view(down.shape[0], -1)).view(
            up.shape[0], down.shape[1], down.shape[2], down.shape[3])
    else:
        d = up @ down

    return d * scale


def _svd_decompose(delta: torch.Tensor, rank: int, ref_dtype: torch.dtype,
                   is_conv4d: bool, conv_shape: tuple | None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SVD-decompose a delta matrix into (down, up) pair.
    For conv layers: operates on the 2D view, reshapes down back to 4D.
    """
    d_2d = delta.view(delta.shape[0], -1) if is_conv4d else delta

    U, S, Vh = torch.linalg.svd(d_2d, full_matrices=False)
    rank_used = min(rank, len(S))
    U   = U[:, :rank_used]
    S   = S[:rank_used]
    Vh  = Vh[:rank_used, :]

    sqrt_s   = torch.sqrt(S.clamp(min=0))
    new_down = (sqrt_s.unsqueeze(-1) * Vh).to(ref_dtype)
    new_up   = (U * sqrt_s).to(ref_dtype)

    if is_conv4d and conv_shape is not None:
        new_down = new_down.view(rank_used, *conv_shape[1:])

    return new_down, new_up


# ── LoRA-to-LoRA merge ────────────────────────────────────────────────────────

def merge_loras(
    lora_paths:  list[str],
    weights:     list[float],
    output_path: str,
    output_rank: int = 0,
    device:      str = "auto",
    progress_fn: Callable[[int, int, str], None] | None = None,
) -> None:
    """
    Merge N LoRA files via weighted delta accumulation + SVD re-decomposition.

    GPU streaming loads each LoRA layer to VRAM for computation.
    SVD on GPU is 10–50× faster than CPU for larger rank matrices.
    output_rank=0 → use max rank found across input LoRAs.
    """
    dev = resolve_device(device)

    # Build key universe and collect rank/shape info (header reads only)
    all_down_keys: set[str] = set()
    file_handles = []

    # We'll process with streaming — open all handles at once
    # First pass: collect all lora_down keys from each file header
    for p in lora_paths:
        with _open(p, "cpu") as f:   # CPU for header scan — no tensor data
            for k in f.keys():
                if "lora_down" in k or "lora_A" in k:
                    all_down_keys.add(k)

    total        = len(all_down_keys)
    merged       = {}
    merged_count = 0
    skipped      = 0

    for i, dk in enumerate(sorted(all_down_keys)):
        if progress_fn:
            progress_fn(i, total, dk)

        delta:    torch.Tensor | None = None
        max_rank: int                 = 4
        ref_dtype  = torch.float16
        is_conv4d  = False
        conv_shape = None

        for p, w in zip(lora_paths, weights):
            with _open(p, dev) as f:
                fkeys = set(f.keys())
                if dk not in fkeys:
                    continue
                ref_dtype  = f.get_tensor(dk).dtype
                ref_down   = f.get_tensor(dk).float()
                is_conv4d  = ref_down.dim() == 4
                conv_shape = ref_down.shape if is_conv4d else None
                max_rank   = max(max_rank, ref_down.shape[0])
                del ref_down

                d = _lora_delta_from_handles(f, dk)
                if d is None:
                    continue
                delta = (d * w) if delta is None else (delta + d * w)
                del d

        if delta is None:
            skipped += 1
            continue

        rank     = output_rank if output_rank > 0 else max_rank
        new_down, new_up = _svd_decompose(
            delta, rank, ref_dtype, is_conv4d, conv_shape)

        uk = dk.replace("lora_down", "lora_up").replace("lora_A", "lora_B")
        merged[dk] = new_down.cpu()
        merged[uk] = new_up.cpu()
        merged[dk.rsplit(".", 1)[0] + ".alpha"] = torch.tensor(float(new_down.shape[0]))
        merged_count += 1

        del delta, new_down, new_up
        if "cuda" in dev and i % 50 == 49:
            _gc(dev)

    if merged_count == 0:
        raise RuntimeError(
            f"LoRA merge produced 0 output layers ({skipped} skipped — "
            "lora_up/lora_B keys missing or all deltas zero). "
            "Check that all input files are valid LoRAs."
        )

    if progress_fn:
        progress_fn(total, total,
                    f"Merged {merged_count} layers ({skipped} skipped) — Saving…")

    _gc(dev)
    _save(merged, output_path, {"merged_by": "Lora-Training-Suite"})


# ── Bake LoRA into checkpoint ─────────────────────────────────────────────────

# ── Diffusers → CompVis block index tables (SDXL) ────────────────────────────
# down_blocks.N.resnets/attentions.I → input_blocks index
_DIFF_DOWN_RES_TO_CV: dict[int, list[int]] = {0: [1, 2], 1: [4, 5], 2: [7, 8]}
_DIFF_DOWN_ATT_TO_CV: dict[int, list[int]] = {1: [4, 5], 2: [7, 8]}
_DIFF_DOWN_DS_TO_CV:  dict[int, int]       = {0: 3, 1: 6}
# up_blocks.N.resnets/attentions.I → output_blocks index
_DIFF_UP_RES_TO_CV: dict[int, list[int]] = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
_DIFF_UP_ATT_TO_CV: dict[int, list[int]] = {0: [0, 1, 2], 1: [3, 4, 5]}
_DIFF_UP_US_TO_CV:  dict[int, int]       = {0: 2, 1: 5}

# ResNet sub-layer: Diffusers name → CompVis name
_RESNET_LAYER: dict[str, str] = {
    "conv1":        "in_layers.2",
    "conv2":        "out_layers.3",
    "time_emb_proj":"emb_layers.1",
    "norm1":        "in_layers.0",
    "norm2":        "out_layers.0",
    "conv_shortcut":"skip_connection",
}


def _diffusers_stem_to_compvis(raw: str) -> str | None:
    """
    Convert a raw Diffusers LoRA UNet stem (underscores, NOT dot-converted)
    to its full CompVis checkpoint key (including .weight suffix).
    raw = stem after stripping 'lora_unet_' prefix.
    e.g. 'down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k'
    """
    import re

    def _attn_sublayer(rest: str) -> str | None:
        """Convert raw attention sub-layer to CompVis dot-path."""
        # transformer_blocks.N.attn1/2.to_out.0
        m = re.match(r"transformer_blocks_(\d+)_attn(\d+)_to_out_0$", rest)
        if m:
            return f"transformer_blocks.{m.group(1)}.attn{m.group(2)}.to_out.0"
        # transformer_blocks.N.attn1/2.to_k/q/v
        m = re.match(r"transformer_blocks_(\d+)_attn(\d+)_(to_[kqv])$", rest)
        if m:
            return f"transformer_blocks.{m.group(1)}.attn{m.group(2)}.{m.group(3)}"
        # transformer_blocks.N.ff.net.0.proj
        m = re.match(r"transformer_blocks_(\d+)_ff_net_0_proj$", rest)
        if m:
            return f"transformer_blocks.{m.group(1)}.ff.net.0.proj"
        # transformer_blocks.N.ff.net.2
        m = re.match(r"transformer_blocks_(\d+)_ff_net_2$", rest)
        if m:
            return f"transformer_blocks.{m.group(1)}.ff.net.2"
        # transformer_blocks.N.norm1/2/3
        m = re.match(r"transformer_blocks_(\d+)_(norm\d+)$", rest)
        if m:
            return f"transformer_blocks.{m.group(1)}.{m.group(2)}"
        # proj_in / proj_out / norm  (SpatialTransformer level)
        if rest in ("proj_in", "proj_out", "norm"):
            return rest
        return None

    # ── down_blocks ───────────────────────────────────────────────────────────

    m = re.match(r"down_blocks_(\d+)_downsamplers_0_conv$", raw)
    if m:
        cv = _DIFF_DOWN_DS_TO_CV.get(int(m.group(1)))
        if cv is not None:
            return f"model.diffusion_model.input_blocks.{cv}.0.op.weight"

    m = re.match(r"down_blocks_(\d+)_resnets_(\d+)_(.*)", raw)
    if m:
        bi, ri, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        blocks = _DIFF_DOWN_RES_TO_CV.get(bi, [])
        layer  = _RESNET_LAYER.get(rest)
        if ri < len(blocks) and layer:
            return f"model.diffusion_model.input_blocks.{blocks[ri]}.0.{layer}.weight"

    m = re.match(r"down_blocks_(\d+)_attentions_(\d+)_(.*)", raw)
    if m:
        bi, ai, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        blocks = _DIFF_DOWN_ATT_TO_CV.get(bi, [])
        layer  = _attn_sublayer(rest)
        if ai < len(blocks) and layer:
            return f"model.diffusion_model.input_blocks.{blocks[ai]}.1.{layer}.weight"

    # ── mid_block ─────────────────────────────────────────────────────────────

    m = re.match(r"mid_block_resnets_(\d+)_(.*)", raw)
    if m:
        ri, rest = int(m.group(1)), m.group(2)
        layer = _RESNET_LAYER.get(rest)
        if layer:
            return f"model.diffusion_model.middle_block.{ri*2}.{layer}.weight"

    m = re.match(r"mid_block_attentions_0_(.*)", raw)
    if m:
        layer = _attn_sublayer(m.group(1))
        if layer:
            return f"model.diffusion_model.middle_block.1.{layer}.weight"

    # ── up_blocks ─────────────────────────────────────────────────────────────

    m = re.match(r"up_blocks_(\d+)_upsamplers_0_conv$", raw)
    if m:
        cv = _DIFF_UP_US_TO_CV.get(int(m.group(1)))
        if cv is not None:
            return f"model.diffusion_model.output_blocks.{cv}.2.conv.weight"

    m = re.match(r"up_blocks_(\d+)_resnets_(\d+)_(.*)", raw)
    if m:
        bi, ri, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        blocks = _DIFF_UP_RES_TO_CV.get(bi, [])
        layer  = _RESNET_LAYER.get(rest)
        if ri < len(blocks) and layer:
            return f"model.diffusion_model.output_blocks.{blocks[ri]}.0.{layer}.weight"

    m = re.match(r"up_blocks_(\d+)_attentions_(\d+)_(.*)", raw)
    if m:
        bi, ai, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        blocks = _DIFF_UP_ATT_TO_CV.get(bi, [])
        layer  = _attn_sublayer(rest)
        if ai < len(blocks) and layer:
            return f"model.diffusion_model.output_blocks.{blocks[ai]}.1.{layer}.weight"

    return None


def _build_lora_map(ckpt_keys: set[str]) -> dict[str, str]:
    """
    Build a forward map: LoRA key stem → checkpoint key.

    Iterates checkpoint keys and derives expected LoRA stems — unambiguous
    because the checkpoint key is the ground truth (no underscore collision).
    Handles both CompVis-named and Diffusers-named LoRAs.
    """
    m: dict[str, str] = {}

    for ck in ckpt_keys:
        if not ck.endswith(".weight"):
            continue
        base = ck[:-7]  # strip .weight

        # UNet — CompVis format (model.diffusion_model.*)
        for pfx in ("model.diffusion_model.", "diffusion_model."):
            if base.startswith(pfx):
                stem = base[len(pfx):]
                m["lora_unet_" + stem.replace(".", "_")] = ck
                break

        # Text encoders — SDXL
        if "conditioner.embedders.0." in base:
            stem = base[base.index("conditioner.embedders.0.") + 24:]
            m["lora_te1_" + stem.replace(".", "_")] = ck
        if "conditioner.embedders.1." in base:
            stem = base[base.index("conditioner.embedders.1.") + 24:]
            m["lora_te2_" + stem.replace(".", "_")] = ck

        # Text encoder — SD1.5
        if base.startswith("cond_stage_model."):
            stem = base[17:]
            m["lora_te_" + stem.replace(".", "_")] = ck

    return m


def bake_loras(
    checkpoint_path: str,
    lora_entries:    list[tuple[str, float]],
    output_path:     str,
    precision:       str  = "fp16",
    device:          str  = "auto",
    progress_fn:     Callable[[int, int, str], None] | None = None,
) -> dict:
    """
    Bake one or more LoRAs into a checkpoint in a single pass.

    lora_entries: list of (lora_path, multiplier) tuples — up to 4.

    Strategy: checkpoint loads to CPU once (it's the mutable output).
    Each LoRA is opened one at a time; tensors load to GPU for delta
    computation, then move back to CPU to update the checkpoint dict.
    Net VRAM peak ≈ 2–3 LoRA tensors at once regardless of model size.
    """
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    out_dtype = dtype_map.get(precision, torch.float16)

    dev = resolve_device(device)

    # Load checkpoint to CPU — it stays in RAM as the accumulation target
    ckpt:      dict = {}
    ckpt_meta: dict = {}
    with _open(checkpoint_path, "cpu") as f:
        ckpt_meta = f.metadata() or {}
        for k in f.keys():
            ckpt[k] = f.get_tensor(k)

    # Forward map: lora_stem → ckpt_key  (unambiguous — derived from ckpt keys)
    lora_map = _build_lora_map(set(ckpt.keys()))

    def _resolve(dk: str) -> str | None:
        """Map a lora_down key to its checkpoint key using the forward map."""
        # Strip lora_down / lora_A suffix to get the stem
        stem = dk
        for sfx in (".lora_down.weight", ".lora_A.weight", ".lora_down", ".lora_A"):
            if stem.endswith(sfx):
                stem = stem[: -len(sfx)]
                break

        # Direct lookup (CompVis-named LoRA)
        if stem in lora_map:
            return lora_map[stem]

        # Diffusers-named LoRA: convert down_blocks/up_blocks → input_blocks
        if stem.startswith("lora_unet_"):
            inner = stem[len("lora_unet_"):]  # pass raw underscore stem, not dot-converted
            cv = _diffusers_stem_to_compvis(inner)
            if cv and cv in ckpt:
                return cv
            # Also try without .weight suffix that _diffusers_stem_to_compvis appends
            if cv:
                base = cv[:-7] if cv.endswith(".weight") else cv
                if base in ckpt:
                    return base
                if base + ".weight" in ckpt:
                    return base + ".weight"

        return None

    # Count total work across all LoRAs for progress reporting
    total_keys = 0
    for lora_path, _ in lora_entries:
        with _open(lora_path, "cpu") as fl:
            total_keys += sum(1 for k in fl.keys()
                              if "lora_down" in k or "lora_A" in k)

    done          = 0
    total_mapped  = 0
    total_skipped = 0
    for lora_idx, (lora_path, multiplier) in enumerate(lora_entries):
        lora_name = os.path.basename(lora_path)
        mapped = 0
        skipped = 0
        with _open(lora_path, dev) as fl:
            down_keys = [k for k in fl.keys() if "lora_down" in k or "lora_A" in k]

            for i, dk in enumerate(down_keys):
                if progress_fn:
                    progress_fn(done, total_keys,
                                f"[{lora_idx+1}/{len(lora_entries)}] {lora_name}  {dk[:50]}")

                delta = _lora_delta_from_handles(fl, dk)
                if delta is None:
                    skipped += 1
                    done += 1
                    continue

                ckpt_key = _resolve(dk)
                if ckpt_key is None:
                    del delta
                    skipped += 1
                    done += 1
                    continue

                base           = ckpt[ckpt_key].float().to(dev)
                ckpt[ckpt_key] = (base + delta * multiplier).cpu().to(out_dtype)
                del base, delta
                mapped += 1
                total_mapped += 1
                done += 1

                if "cuda" in dev and done % 100 == 0:
                    _gc(dev)

        total_skipped += skipped
        if progress_fn:
            progress_fn(done, total_keys,
                        f"{lora_name}: {mapped} layers baked, {skipped} skipped")

    if progress_fn:
        progress_fn(total_keys, total_keys, "Saving…")

    _gc(dev)
    _save(ckpt, output_path, ckpt_meta)
    return {"merged": total_mapped, "skipped": total_skipped}


# ── LoRA extraction from checkpoint delta ─────────────────────────────────────

def _ckpt_to_lora_key(key: str) -> str | None:
    """
    Map a checkpoint weight key to a LoRA key stem.
    Returns None for non-LoRA-able layers (biases, norms, VAE, etc.).
    """
    if not key.endswith(".weight"):
        return None
    base = key[:-7]  # strip ".weight"

    # UNet — SD 1.x and SDXL
    for pfx in ("model.diffusion_model.", "diffusion_model."):
        if base.startswith(pfx):
            return "lora_unet_" + base[len(pfx):].replace(".", "_")

    # Text encoders — SDXL dual encoders
    if "conditioner.embedders.0." in base:
        stem = base[base.index("conditioner.embedders.0.") + 24:]
        return "lora_te1_" + stem.replace(".", "_")
    if "conditioner.embedders.1." in base:
        stem = base[base.index("conditioner.embedders.1.") + 24:]
        return "lora_te2_" + stem.replace(".", "_")

    # Text encoder — SD 1.x
    if base.startswith("cond_stage_model."):
        return "lora_te_" + base[17:].replace(".", "_")

    return None


def extract_lora(
    model_tuned:  str,
    model_base:   str,
    output_path:  str,
    rank:         int  = 32,
    conv_layers:  bool = True,
    device:       str  = "auto",
    progress_fn:  Callable[[int, int, str], None] | None = None,
) -> None:
    """
    Extract a LoRA by SVD-decomposing the delta between a fine-tuned checkpoint
    and its original base.

    delta = fine_tuned[key] - base[key]
    SVD → (lora_down, lora_up) at the requested rank.

    Output is always fp16 — SVD runs in fp32 internally regardless of dtype.
    """
    dev = resolve_device(device)
    out: dict = {}

    with _open(model_tuned, dev) as ft, _open(model_base, dev) as fb:
        base_keys = set(fb.keys())
        all_keys  = [k for k in ft.keys() if k in base_keys]
        total     = len(all_keys)
        done      = 0

        for i, key in enumerate(all_keys):
            if progress_fn and i % 20 == 0:
                progress_fn(i, total, key)

            ta = ft.get_tensor(key).float()
            tb = fb.get_tensor(key).float()

            # Skip scalars, vectors, and non-weight tensors
            if ta.dim() < 2:
                del ta, tb
                continue

            is_conv = ta.dim() == 4
            if is_conv and not conv_layers:
                del ta, tb
                continue

            lora_key = _ckpt_to_lora_key(key)
            if lora_key is None:
                del ta, tb
                continue

            delta = ta - tb
            del ta, tb

            # Skip layers the fine-tune didn't touch
            if delta.abs().max().item() < 1e-6:
                del delta
                continue

            conv_shape = delta.shape if is_conv else None
            new_down, new_up = _svd_decompose(
                delta, rank, torch.float16, is_conv, conv_shape)
            del delta

            out[lora_key + ".lora_down.weight"] = new_down.cpu()
            out[lora_key + ".lora_up.weight"]   = new_up.cpu()
            out[lora_key + ".alpha"]             = torch.tensor(float(rank))
            done += 1

            if "cuda" in dev and done % 50 == 0:
                _gc(dev)

    if done == 0:
        raise RuntimeError(
            f"LoRA extraction produced 0 output layers (checked {total} keys — "
            "all tensors were 1-D, skipped by conv_layers filter, had no matching "
            "lora key, or delta was below threshold). "
            "Check that tuned and base are different checkpoints."
        )

    if progress_fn:
        progress_fn(total, total, f"Extracted {done} layers — Saving…")

    _gc(dev)
    _save(out, output_path, {
        "extracted_by": "Lora-Training-Suite",
        "rank": str(rank),
        "conv_layers": str(conv_layers),
    })
