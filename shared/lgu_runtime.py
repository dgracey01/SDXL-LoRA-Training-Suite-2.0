"""
shared/lgu_runtime.py — LoraGenesys Unit (.lgu) runtime.

Loads a .lgu (a zip: manifest.json + adapter.safetensors) and renders with it:
  frozen SDXL base  +  JoyCaption VISUAL brain (SigLIP+projector, from HF cache)  +  trained
  adapter (Perceiver Resampler pi + IP-Adapter cross-attn).

Division of labor proven 2026-07-16: IDENTITY comes from the base + the character NAME in the prompt
(canonical) or a LoRA (custom); the brain-adapter supplies COMPOSITION (pose/outfit/scene) from a
REFERENCE image. So render() takes both a prompt (identity) and a reference image (composition).

Self-contained (no dependency on X:\\LoraGenesys) so Jarvis and SD.UI can both call it. Mirrors the
proven infer_brain.py path; runs in the suite venv (diffusers 0.31, verified by the B-2 evals).
"""
from __future__ import annotations

import gc
import io
import json
import os
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Perceiver Resampler (pi) — must match phase2b_train_brain.BrainResampler exactly ────────────
class _CrossAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, q, kv):
        q, kv = self.norm_q(q), self.norm_kv(kv)
        B, Nq, _ = q.shape
        Q = self.to_q(q).view(B, Nq, self.h, self.dh).transpose(1, 2)
        K, V = self.to_kv(kv).chunk(2, dim=-1)
        K = K.view(B, kv.shape[1], self.h, self.dh).transpose(1, 2)
        V = V.view(B, kv.shape[1], self.h, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.to_out(out.transpose(1, 2).reshape(B, Nq, -1))


class _FF(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * mult),
                                 nn.GELU(), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class BrainResampler(nn.Module):
    def __init__(self, input_dim, cross_dim, num_tokens=16, depth=4, dim=768, heads=12):
        super().__init__()
        self.num_tokens, self.cross_dim = num_tokens, cross_dim
        self.latents = nn.Parameter(torch.randn(num_tokens, dim) * dim ** -0.5)
        self.in_proj = nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList([nn.ModuleList([_CrossAttn(dim, heads), _FF(dim)]) for _ in range(depth)])
        self.out_proj = nn.Linear(dim, cross_dim)
        self.norm_out = nn.LayerNorm(cross_dim)

    def forward(self, brain_latent):
        x = self.in_proj(brain_latent)
        lat = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        for attn, ff in self.layers:
            lat = attn(lat, x) + lat
            lat = ff(lat) + lat
        return self.norm_out(self.out_proj(lat))


def _wire_ip_attn(unet, num_tokens, dev):
    """Install IPAdapterAttnProcessor2_0 on the cross-attn (attn2), plain AttnProcessor2_0 on self-attn.
    Returns the ModuleList of IP procs (for loading the trained to_k_ip/to_v_ip). Matches the trainer."""
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0, AttnProcessor2_0
    boc = unet.config.block_out_channels
    procs = {}
    for name in unet.attn_processors.keys():
        cross = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden = boc[-1]
        elif name.startswith("up_blocks"):
            hidden = list(reversed(boc))[int(name[len("up_blocks."):].split(".")[0])]
        elif name.startswith("down_blocks"):
            hidden = boc[int(name[len("down_blocks."):].split(".")[0])]
        else:
            hidden = boc[0]
        if cross is None:
            procs[name] = AttnProcessor2_0()
        else:
            procs[name] = IPAdapterAttnProcessor2_0(hidden_size=hidden, cross_attention_dim=cross,
                                                    num_tokens=(num_tokens,), scale=1.0).to(dev, torch.float32)
    unet.set_attn_processor(procs)
    return nn.ModuleList([p for p in unet.attn_processors.values() if isinstance(p, nn.Module)])


# ── unit archive I/O ────────────────────────────────────────────────────────────────────────────
def _unpack(lgu_path: str, cache_root: str = ""):
    """Extract a .lgu next to itself under _unpacked/<name>/ (idempotent by mtime). Returns the dir."""
    lgu_path = os.path.abspath(lgu_path)
    name = os.path.splitext(os.path.basename(lgu_path))[0]
    root = cache_root or os.path.join(os.path.dirname(lgu_path), "_unpacked")
    dst = os.path.join(root, name)
    stamp = os.path.join(dst, ".stamp")
    if not (os.path.isfile(stamp) and open(stamp).read().strip() == str(os.path.getmtime(lgu_path))):
        os.makedirs(dst, exist_ok=True)
        with zipfile.ZipFile(lgu_path) as z:
            z.extractall(dst)
        with open(stamp, "w") as f:
            f.write(str(os.path.getmtime(lgu_path)))
    return dst


def _load_flat_adapter(path):
    """adapter.safetensors: flat keys 'image_proj.*' / 'ip_adapter.*' -> two nested state_dicts."""
    from safetensors.torch import load_file
    flat = load_file(path)
    image_proj, ip_adapter = {}, {}
    for k, v in flat.items():
        if k.startswith("image_proj."):
            image_proj[k[len("image_proj."):]] = v
        elif k.startswith("ip_adapter."):
            ip_adapter[k[len("ip_adapter."):]] = v
    return image_proj, ip_adapter


# ── the runtime ───────────────────────────────────────────────────────────────────────────────
class LoraGenesysUnit:
    def __init__(self, lgu_path: str, device: str | None = None, dtype: str = "fp16"):
        self.lgu_path = os.path.abspath(lgu_path)
        self.dir = _unpack(self.lgu_path)
        self.manifest = json.load(open(os.path.join(self.dir, "manifest.json"), encoding="utf-8"))
        self._dtype = torch.float16 if dtype == "fp16" else torch.float32
        self._device = device
        self._pipe = None
        self._vae = None
        self._resampler = None
        self._brain = None
        self._brain_proc = None

    def _dev(self):
        if self._device:
            return self._device
        return "cuda" if torch.cuda.is_available() else "cpu"

    # -- lazy heavy loads ----------------------------------------------------------------------
    def _ensure_pipe(self):
        if self._pipe is not None:
            return
        from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
        dev, fp16 = self._dev(), self._dtype
        base = self.manifest["components"]["base"]["source"]
        if not os.path.isfile(base):
            # allow a checkpoint_dir hint in the manifest's wiring for portability
            hint = self.manifest.get("wiring", {}).get("base_dir", "")
            cand = os.path.join(hint, os.path.basename(base)) if hint else ""
            if cand and os.path.isfile(cand):
                base = cand
            else:
                raise FileNotFoundError(f"[lgu] base checkpoint not found: {base}")
        if str(base).lower().endswith((".safetensors", ".ckpt")):
            pipe = StableDiffusionXLPipeline.from_single_file(base, torch_dtype=fp16)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=fp16)
        pipe.to(dev)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        self._vae = pipe.vae.to(torch.float32)      # fp32 decode (SDXL fp16 VAE NaN fix)
        self._sched = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        num_tokens = int(self.manifest["num_tokens"])
        brain_dim = int(self.manifest["brain_dim"])
        ip_layers = _wire_ip_attn(pipe.unet, num_tokens, dev)
        image_proj_sd, ip_sd = _load_flat_adapter(os.path.join(self.dir, "adapter.safetensors"))
        ip_layers.load_state_dict(ip_sd)
        ip_layers.to(dev, fp16)
        self._resampler = BrainResampler(brain_dim, pipe.unet.config.cross_attention_dim, num_tokens).to(dev, fp16).eval()
        self._resampler.load_state_dict(image_proj_sd)
        self._pipe = pipe

    def _ensure_brain(self):
        """JoyCaption VISUAL tap: vision_tower + multi_modal_projector only (no 8B Llama on GPU)."""
        if self._brain is not None:
            return
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        dev, fp16 = self._dev(), self._dtype
        b = self.manifest["components"]["brain"]
        repo, rev = b["source"], b.get("revision")
        self._brain_proc = AutoProcessor.from_pretrained(repo, revision=rev)
        m = LlavaForConditionalGeneration.from_pretrained(repo, revision=rev, torch_dtype=fp16).eval()
        m.vision_tower.to(dev)
        m.multi_modal_projector.to(dev)
        self._vfl = getattr(m.config, "vision_feature_layer", -2)
        self._vfs = getattr(m.config, "vision_feature_select_strategy", "full")
        self._brain = m

    def _brain_latent(self, reference):
        from PIL import Image
        dev, fp16 = self._dev(), self._dtype
        if isinstance(reference, str):
            reference = Image.open(reference).convert("RGB")
        elif not hasattr(reference, "convert"):
            reference = Image.open(io.BytesIO(reference)).convert("RGB")
        else:
            reference = reference.convert("RGB")
        px = self._brain_proc.image_processor(images=reference, return_tensors="pt")["pixel_values"].to(dev, fp16)
        with torch.no_grad():
            feats = self._brain.get_image_features(pixel_values=px, vision_feature_layer=self._vfl,
                                                   vision_feature_select_strategy=self._vfs)[0]
        return feats.to(dev, fp16).unsqueeze(0)      # [1, T, D]

    # -- the render ----------------------------------------------------------------------------
    @torch.no_grad()
    def render(self, prompt, reference, *, negative="worst quality, low quality, blurry, deformed",
               scale=0.7, steps=30, cfg=6.0, seed=-1, size=1024):
        """prompt = identity (include the character NAME for canonical chars). reference = composition."""
        self._ensure_pipe()
        self._ensure_brain()
        dev, fp16 = self._dev(), self._dtype
        pipe, unet, vae, sched = self._pipe, self._pipe.unet, self._vae, self._sched
        for p in unet.attn_processors.values():
            if hasattr(p, "scale"):
                p.scale = [float(scale)]

        brain = self._brain_latent(reference)
        image_tokens = self._resampler(brain)

        pe, npe, pooled, npooled = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, negative_prompt=negative, negative_prompt_2=negative,
            device=dev, num_images_per_prompt=1, do_classifier_free_guidance=True)
        ehs = torch.cat([torch.cat([npe, image_tokens], dim=1), torch.cat([pe, image_tokens], dim=1)], dim=0)
        S = int(size)
        time_ids = torch.tensor([[S, S, 0, 0, S, S]], device=dev, dtype=fp16).repeat(2, 1)
        added = {"text_embeds": torch.cat([npooled, pooled], dim=0), "time_ids": time_ids}

        gen = None if int(seed) < 0 else torch.Generator(dev).manual_seed(int(seed))
        sched.set_timesteps(int(steps), device=dev)
        latents = torch.randn((1, unet.config.in_channels, S // 8, S // 8), generator=gen, device=dev,
                              dtype=fp16) * sched.init_noise_sigma
        for t in sched.timesteps:
            lin = sched.scale_model_input(torch.cat([latents, latents], dim=0), t)
            pred = unet(lin, t, encoder_hidden_states=ehs, added_cond_kwargs=added).sample
            nu, nc = pred.chunk(2)
            latents = sched.step(nu + float(cfg) * (nc - nu), t, latents).prev_sample

        import numpy as np
        from PIL import Image
        img = vae.decode(latents.to(torch.float32) / vae.config.scaling_factor).sample
        arr = ((img[0].permute(1, 2, 0).float().cpu().numpy() / 2 + 0.5).clip(0, 1) * 255).round().astype("uint8")
        return Image.fromarray(arr)

    def unload(self):
        self._pipe = None; self._vae = None; self._resampler = None
        self._brain = None; self._brain_proc = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# module-level cache so a unit stays resident across calls (one at a time; VRAM arbiter friendly)
_UNIT_CACHE: dict[str, LoraGenesysUnit] = {}


def get_unit(lgu_path: str, device: str | None = None) -> LoraGenesysUnit:
    key = os.path.abspath(lgu_path)
    u = _UNIT_CACHE.get(key)
    if u is None:
        # keep only one resident
        for other in list(_UNIT_CACHE.values()):
            other.unload()
        _UNIT_CACHE.clear()
        u = LoraGenesysUnit(lgu_path, device=device)
        _UNIT_CACHE[key] = u
    return u


def render_unit(lgu_path, prompt, reference, **kw):
    """One-shot: load (cached) + render. Returns a PIL image."""
    return get_unit(lgu_path).render(prompt, reference, **kw)


def list_units(units_dir: str) -> list[str]:
    if not os.path.isdir(units_dir):
        return []
    return sorted(f for f in os.listdir(units_dir) if f.lower().endswith(".lgu"))
