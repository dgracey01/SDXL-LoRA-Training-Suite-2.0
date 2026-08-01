"""
shared/render_engine.py — Jarvis 2.0 native render engine (Track A / Phase 1).

A self-contained diffusers SDXL renderer: txt2img + a LoRA stack, with NO SD.Next server.
Deliberately standalone (no Jarvis/SD.UI imports) so BOTH apps can point at it — the
"shared render backend" the roadmap's parking-lot note wants (Jarvis now; a SD.UI
`sdnext <-> native` switch later).

It consumes the SAME `<lora:name:weight>`-tagged prompt string Jarvis already assembles from
the LoRA library: the engine parses the tags into diffusers `load_lora_weights` + `set_adapters`
calls and strips them from the text prompt. So the entire prompt-assembly layer is reused as-is.

Requires (install into the suite venv):  pip install diffusers transformers accelerate peft safetensors
Runs on the box's GPU (ROCm on the 7800 XT reports as cuda; CUDA on Goliath). Lazy-loads the model
on first render and keeps it resident; call .unload() to free VRAM (VRAM-arbiter hook).
"""
from __future__ import annotations

import base64
import gc
import io
import os
import re
import time

# ROCm gates its flash / mem-efficient attention kernels behind this flag on RDNA3. Unset, SDPA
# silently falls back to the `math` path, which on this box measured 485 ms and 3.15 GB per
# SDXL-1024 attention call vs 6 ms and 0.02 GB with flash (~80x slower, ~157x the VRAM). That
# per-call VRAM is what overflowed the 16 GB card into shared system RAM and made renders feel
# CPU-bound. Must be set before torch is imported; harmless on CUDA/CPU boxes.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# <lora:FileName:0.8>  (weight optional)
_LORA_TAG = re.compile(r"<lora:([^:>]+)(?::([0-9]*\.?[0-9]+))?>")


def _scheduler_for(name: str):
    """Map a webui-style sampler name -> (diffusers scheduler class, from_config kwargs).
    Returns None to leave the pipeline's default scheduler in place."""
    n = (name or "").lower()
    try:
        from diffusers import (
            DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler,
        )
    except Exception:
        return None
    karras = "karras" in n
    if "dpm++ 2m sde" in n or "sde" in n and "dpm" in n:
        return (DPMSolverMultistepScheduler,
                {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": karras or True})
    if "dpm++ 2m" in n or ("dpm" in n and "2m" in n):
        return (DPMSolverMultistepScheduler, {"use_karras_sigmas": karras})
    if "euler a" in n or "euler_a" in n or "ancestral" in n:
        return (EulerAncestralDiscreteScheduler, {})
    if "euler" in n:
        return (EulerDiscreteScheduler, {"use_karras_sigmas": karras})
    if "unipc" in n:
        return (UniPCMultistepScheduler, {})
    if "ddim" in n:
        return (DDIMScheduler, {})
    return None


class RenderEngine:
    """One resident SDXL pipeline + LoRA-stack management. Reusable across apps."""

    def __init__(self, checkpoint: str, lora_dir: str = "", device: str | None = None,
                 dtype: str = "fp16", lowvram: bool = True,
                 no_half_vae: bool = False, vae_override: str = "", vae_dir: str = ""):
        self.checkpoint = checkpoint
        self.lora_dir = lora_dir
        self._device = device
        self._dtype = dtype
        self._lowvram = lowvram   # medvram-equivalent: offload idle modules to CPU (fits 16GB cards)
        # VAE controls — default OFF so existing callers (Jarvis) are unchanged; SD.UI opts in via cfg.
        self._no_half_vae = bool(no_half_vae)   # run VAE in fp32 (upcast_vae) — the --no-half-vae fix
        self._vae_override = vae_override or ""  # a VAE file to replace the checkpoint's baked one ("" = keep)
        self._vae_dir = vae_dir or ""            # folder the override filename resolves against
        self._pipe = None
        self._loaded_ckpt = None
        # Resident LoRA cache: the (adapter, weight) stack currently injected on the shared modules.
        # Lets _apply_loras skip the disk read + peft re-injection when the requested stack is unchanged
        # (the common iterate-on-seed case, and every ADetailer sub-crop) and re-weight via set_adapters
        # alone when only the weights change. None = nothing loaded yet. Reset in unload().
        self._active_loras = None
        self._yolo_cache = {}     # ADetailer detector path -> resident ultralytics YOLO (avoid disk reload)
        # ── live preview (TAESD) ──────────────────────────────────────────────
        # A shared progress/preview state the render callbacks write and a host (SD.UI's
        # NativeClient.progress()) reads, so the preview box streams the in-progress image while a
        # native render runs. Harmless for other hosts (Jarvis): it just updates a dict.
        self.preview = True
        self._preview_vae = None          # lazy AutoencoderTiny (taesdxl); None + _preview_failed → rgb-approx
        self._preview_failed = False
        self._preview_interval = 0.4      # s — cap decode frequency independent of sampling speed
        self._last_preview_t = 0.0
        self.progress = {"active": False, "step": 0, "total": 0, "image_b64": None, "t0": 0.0}

    # ── lazy pipeline ────────────────────────────────────────────────────────
    def _pick_device(self):
        if self._device:
            return self._device
        try:
            import torch
            if torch.cuda.is_available():        # ROCm reports True here too
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _ensure_pipe(self):
        if self._pipe is not None and self._loaded_ckpt == self.checkpoint:
            return self._pipe
        # Rebuilding the base pipe (first load OR a checkpoint change): the derived pipes share the OLD
        # base.components and the LoRA cache refers to the OLD modules — both are now stale, so invalidate
        # them or a post-switch img2img/ControlNet render would silently use the previous checkpoint.
        self._img2img_pipe = None
        self._controlnet_pipe = None
        self._cn_key = None
        self._active_loras = None
        import torch
        from diffusers import StableDiffusionXLPipeline
        dtype = torch.float16 if self._dtype == "fp16" else torch.float32
        ckpt = self.checkpoint
        if str(ckpt).lower().endswith((".safetensors", ".ckpt")):
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(
                    f"[render_engine] checkpoint not found on this machine:\n    {ckpt}\n"
                    f"  Set a valid SDXL .safetensors path (engine.native_checkpoint).")
            pipe = StableDiffusionXLPipeline.from_single_file(ckpt, torch_dtype=dtype)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(ckpt, torch_dtype=dtype)
        # Memory efficiency — a BARE SDXL pipeline on a 16 GB card overflows into (≈10× slower) shared
        # system RAM and crawls. Tile/slice the VAE decode, and (lowvram) offload idle modules to CPU —
        # the diffusers equivalent of SD.Next's --medvram, which is why SD.Next fit where a raw pipe didn't.
        try:
            pipe.enable_vae_tiling()
            pipe.enable_vae_slicing()
        except Exception:
            pass
        # VAE override + fp32 upcast — BEFORE device placement/offload so the swapped VAE is moved/hooked
        # with the rest of the pipe. Fixes the SDXL fp16-VAE overflow that distorts some checkpoints.
        self._apply_vae(pipe, dtype)
        # NOTE: channels_last (NHWC) and fuse_qkv_projections — the "lossless" speedups from PyTorch's
        # SDXL guide — were both measured on this RX 7800 XT (gfx1101 / ROCm 7.2 / Windows) and NEITHER
        # helped: channels_last was bit-identical but neutral-to-slightly-slower (MIOpen's NHWC conv
        # kernels aren't tuned on this stack the way cuDNN's are), and fuse_qkv was slower AND not
        # bit-identical. They are CUDA wins that don't transfer here, so they are deliberately not applied.
        if self._lowvram:
            try:
                pipe.enable_model_cpu_offload()   # only the ACTIVE module sits on the GPU; do NOT also .to()
            except Exception:
                pipe = pipe.to(self._pick_device())
        else:
            pipe = pipe.to(self._pick_device())
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        self._pipe = pipe
        self._loaded_ckpt = ckpt
        return pipe

    def _apply_vae(self, pipe, dtype):
        """Optional VAE override + fp32 upcast. Two independent levers (SD.UI exposes both in Settings):
          • vae_override — replace the checkpoint's baked VAE with a chosen file. Some checkpoints ship a
            broken/missing VAE; a known-good one fixes distortion AND makes output consistent across a
            checkpoint rotation.
          • no_half_vae  — run the VAE in fp32 (diffusers upcast_vae == A1111/SD.Next --no-half-vae). SDXL's
            fp16 VAE overflows to NaN on some checkpoints → distorted/garbage output; fp32 decode fixes it.
        Both are best-effort and never abort a render: on failure we log and fall back to the baked/fp16 VAE."""
        ov = (self._vae_override or "").strip()
        if ov and ov.lower() not in ("automatic", "auto", "none", "baked", "default"):
            path = ov if os.path.isfile(ov) else os.path.join(self._vae_dir or "", ov)
            if os.path.isfile(path):
                try:
                    from diffusers import AutoencoderKL
                    pipe.vae = AutoencoderKL.from_single_file(path, torch_dtype=dtype)  # placed by caller
                except Exception as e:
                    print(f"[render_engine] VAE override failed ({os.path.basename(path)}): "
                          f"{type(e).__name__}: {e}; using the checkpoint's baked VAE")
            else:
                print(f"[render_engine] VAE override not found: {ov} (dir={self._vae_dir}); using baked VAE")
        if self._no_half_vae:
            try:
                pipe.upcast_vae()   # fp32 VAE — kills the SDXL fp16 overflow/NaN distortion
            except Exception as e:
                print(f"[render_engine] upcast_vae (no-half-vae) failed: {type(e).__name__}: {e}")

    # ── LoRA stack ───────────────────────────────────────────────────────────
    def _resolve_lora(self, name: str):
        """A LoRA display/file name -> (full path, stem). Handles space<->dash/underscore."""
        cands = [name, name.replace(" ", "-"), name.replace(" ", "_"), name.replace(" ", "")]
        for d in ([self.lora_dir] if self.lora_dir else []):
            if not os.path.isdir(d):
                continue
            for c in cands:
                for ext in (".safetensors", ".ckpt"):
                    p = os.path.join(d, c + ext)
                    if os.path.isfile(p):
                        return p, c
        return "", name

    def _apply_loras(self, pipe, loras):
        """loras: list of (name, weight). Load each + set per-adapter weights. Returns (applied, missing).
        Keeps the current stack resident and only touches disk when the SET of adapters changes — see
        self._active_loras. All three pipes (txt2img/img2img/controlnet) share the base modules, so the
        cache is engine-level and valid across a pipe switch."""
        want, missing = [], []      # want: (adapter, weight, path, orig_name) for each resolvable request
        for name, wt in loras:
            path, stem = self._resolve_lora(name)
            if not path:
                missing.append(name)
                continue
            want.append((re.sub(r"[^A-Za-z0-9_]", "_", stem), float(wt), path, name))
        want_names = [a for a, _, _, _ in want]
        prev_names = [a for a, _ in (self._active_loras or [])]

        # FAST PATH — the same adapters are already injected on the shared modules (previous render, or
        # the base pass before an ADetailer crop). Re-loading them from disk every render, and once per
        # detected ADetailer region, was the dominant avoidable per-render cost. set_adapters re-asserts
        # the active set + weights with NO disk read and NO peft re-injection; only a change in the set of
        # names falls through to a reload. (Order-sensitive by design: a reordered stack reloads, which is
        # rare and safe.)
        if want_names and want_names == prev_names:
            try:
                pipe.set_adapters(want_names, adapter_weights=[w for _, w, _, _ in want])
                self._active_loras = [(a, w) for a, w, _, _ in want]
                return want_names, missing
            except Exception:
                pass   # re-weight failed → fall through to a clean reload

        # SLOW PATH — the stack changed (or nothing was loaded). Drop the old adapters, load the new set.
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        names, weights = [], []
        for adapter, wt, path, orig_name in want:
            try:
                # low_cpu_mem_usage=False is a HARD REQUIREMENT here, not an optimization opt-out. The
                # default (True) makes peft inject LoRA adapters on the META device, whose init runs torch's
                # _prims/_decomp 'uniform' meta path -> a lazy import that trips PySide6's shiboken import
                # hook when this runs on a render WORKER thread -> Windows access violation that takes the
                # whole Suite down (logged twice, only on the ControlNet+LoRA path). Forcing real-device
                # init uses the native uniform kernel, no meta path, no crash. Older diffusers that don't
                # accept the kwarg fall back (they default to real-device init anyway).
                try:
                    pipe.load_lora_weights(os.path.dirname(path),
                                           weight_name=os.path.basename(path),
                                           adapter_name=adapter, low_cpu_mem_usage=False)
                except TypeError:
                    pipe.load_lora_weights(os.path.dirname(path),
                                           weight_name=os.path.basename(path),
                                           adapter_name=adapter)
                names.append(adapter)
                weights.append(float(wt))
            except Exception as e:
                missing.append(f"{orig_name} (load error: {e})")
        if names:
            try:
                pipe.set_adapters(names, adapter_weights=weights)
            except Exception:
                pass
        self._active_loras = list(zip(names, weights))
        return names, missing

    # ── live preview (TAESD) ─────────────────────────────────────────────────
    def _ensure_preview_vae(self):
        """Lazy tiny VAE (madebyollin/taesdxl) for cheap step previews. Returns the module, or None
        (→ caller uses the latent-RGB approximation). Never raises."""
        if self._preview_vae is not None or self._preview_failed:
            return self._preview_vae
        try:
            import torch
            from diffusers import AutoencoderTiny
            dtype = torch.float16 if self._dtype == "fp16" else torch.float32
            vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=dtype)
            dev = "cuda" if torch.cuda.is_available() else self._pick_device()
            self._preview_vae = vae.to(dev).eval()
        except Exception:
            self._preview_failed = True       # offline / no model → fall back to the rgb approximation
            self._preview_vae = None
        return self._preview_vae

    # SDXL latent → approximate RGB (ComfyUI factors) — the zero-dependency offline fallback preview.
    _LATENT_RGB = ((0.3651, 0.4232, 0.4341), (-0.2533, -0.0042, 0.1068),
                   (0.1076, 0.1111, -0.0362), (-0.3165, -0.2492, -0.2188))

    def _latents_to_b64(self, latents, max_side: int = 512):
        """Decode in-progress latents to a small base64 PNG for the live preview. Never raises."""
        if latents is None:
            return None
        try:
            import torch
            import numpy as np
            from PIL import Image
            lat = latents[:1].detach()
            vae = self._ensure_preview_vae()
            if vae is not None:
                with torch.no_grad():
                    dec = vae.decode(lat.to(vae.dtype).to(vae.device)).sample[0]
                arr = ((dec.float().cpu().permute(1, 2, 0).numpy() / 2 + 0.5).clip(0, 1) * 255).astype("uint8")
            else:
                m = torch.tensor(self._LATENT_RGB, dtype=lat.dtype, device=lat.device)   # [4,3]
                rgb = torch.einsum("chw,cr->hwr", lat[0], m)
                rgb = (rgb - rgb.amin()) / (rgb.amax() - rgb.amin() + 1e-5)
                arr = (rgb.float().cpu().numpy() * 255).astype("uint8")
            img = Image.fromarray(arr)
            w, h = img.size
            if max(w, h) > max_side:
                s = max_side / float(max(w, h))
                img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
            buf = io.BytesIO(); img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return None

    def _begin_progress(self, total_steps: int):
        """Reset the shared preview state and return a diffusers callback_on_step_end. The callback
        records step/total every step and decodes a TAESD preview at most every _preview_interval s."""
        self.progress = {"active": True, "step": 0, "total": max(1, int(total_steps)),
                         "image_b64": None, "t0": time.time()}
        self._last_preview_t = 0.0
        if self.preview:
            self._ensure_preview_vae()        # warm the tiny VAE once, before sampling (cached)

        def _cb(pipe, step, timestep, kw):
            self.progress["step"] = step + 1
            if self.preview:
                now = time.time()
                last = (step + 1) >= self.progress["total"]
                if last or (now - self._last_preview_t) >= self._preview_interval:
                    b64 = self._latents_to_b64(kw.get("latents"))
                    if b64:
                        self.progress["image_b64"] = b64
                        self._last_preview_t = now
            return kw
        return _cb

    def _end_progress(self):
        self.progress["active"] = False

    # ── the render ───────────────────────────────────────────────────────────
    def txt2img(self, prompt: str, negative: str = "", *, steps: int = 30, cfg: float = 6.0,
                width: int = 1024, height: int = 1024, seed: int = -1,
                sampler: str = "DPM++ 2M SDE", guidance_rescale: float = 0.0):
        """Render one image. `prompt` may contain <lora:...> tags (parsed + stripped here).
        guidance_rescale (0=off, ~0.5-0.7): the SDXL overexposure clamp (Lin et al.) — tames blown
        highlights from high CFG / over-driven LoRAs. Returns (PIL.Image, info_dict)."""
        import torch
        pipe = self._ensure_pipe()

        loras = [(m.group(1), float(m.group(2) or 1.0)) for m in _LORA_TAG.finditer(prompt or "")]
        clean = _LORA_TAG.sub("", prompt or "").strip()
        applied, missing = self._apply_loras(pipe, loras)

        sch = _scheduler_for(sampler)
        if sch is not None:
            cls, kw = sch
            try:
                pipe.scheduler = cls.from_config(pipe.scheduler.config, **kw)
            except Exception:
                try:
                    pipe.scheduler = cls.from_config(pipe.scheduler.config)
                except Exception:
                    pass

        # under model_cpu_offload pipe.device reports 'cpu'; the latents are made on the compute device
        _gd = "cuda" if torch.cuda.is_available() else "cpu"
        gen = None if int(seed) < 0 else torch.Generator(device=_gd).manual_seed(int(seed))
        cb = self._begin_progress(int(steps))
        try:
            image = pipe(
                prompt=clean, negative_prompt=negative or "",
                num_inference_steps=int(steps), guidance_scale=float(cfg),
                guidance_rescale=float(guidance_rescale),
                width=int(width), height=int(height), generator=gen,
                callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"],
            ).images[0]
        finally:
            self._end_progress()
        return image, {"loras": applied, "missing_loras": missing, "prompt": clean, "sampler": sampler}

    # ── scheduler helper (shared by img2img / hires) ─────────────────────────
    def _set_scheduler(self, pipe, sampler):
        sch = _scheduler_for(sampler)
        if sch is None:
            return
        cls, kw = sch
        try:
            pipe.scheduler = cls.from_config(pipe.scheduler.config, **kw)
        except Exception:
            try:
                pipe.scheduler = cls.from_config(pipe.scheduler.config)
            except Exception:
                pass

    # ── img2img (Phase 2) ────────────────────────────────────────────────────
    def _ensure_img2img_pipe(self):
        """Img2img pipeline that SHARES the txt2img components (no second model load)."""
        base = self._ensure_pipe()
        if getattr(self, "_img2img_pipe", None) is not None:
            return self._img2img_pipe
        from diffusers import StableDiffusionXLImg2ImgPipeline
        self._img2img_pipe = StableDiffusionXLImg2ImgPipeline(**base.components)
        try:
            self._img2img_pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        return self._img2img_pipe

    def img2img(self, image, prompt, negative="", *, strength=0.5, steps=30, cfg=6.0,
                seed=-1, sampler="DPM++ 2M SDE", preview=True, guidance_rescale=0.0):
        """Transform / refine a PIL image. `prompt` may carry <lora:> tags (parsed + stripped).
        preview=False for internal sub-passes (ADetailer crops) so they don't hijack the live box.
        guidance_rescale: SDXL overexposure clamp (see txt2img)."""
        import torch
        pipe = self._ensure_img2img_pipe()
        loras = [(m.group(1), float(m.group(2) or 1.0)) for m in _LORA_TAG.finditer(prompt or "")]
        clean = _LORA_TAG.sub("", prompt or "").strip()
        applied, missing = self._apply_loras(pipe, loras)
        self._set_scheduler(pipe, sampler)
        _gd = "cuda" if torch.cuda.is_available() else "cpu"
        gen = None if int(seed) < 0 else torch.Generator(device=_gd).manual_seed(int(seed))
        # img2img runs ~steps*strength actual steps → size the progress bar to that
        cb = self._begin_progress(max(1, round(int(steps) * float(strength)))) if preview else None
        try:
            out = pipe(prompt=clean, negative_prompt=negative or "", image=image,
                       strength=float(strength), num_inference_steps=int(steps),
                       guidance_scale=float(cfg), guidance_rescale=float(guidance_rescale),
                       generator=gen, callback_on_step_end=cb,
                       callback_on_step_end_tensor_inputs=["latents"]).images[0]
        finally:
            if preview:
                self._end_progress()
        return out, {"mode": "img2img", "strength": float(strength),
                     "loras": applied, "missing_loras": missing}

    # ── upscale (best-effort Real-ESRGAN; always returns something) ──────────
    def upscale(self, image, scale=2.0):
        """Upscale a PIL image. Reuses CaptionsAI's Real-ESRGAN if importable + its deps are present;
        otherwise falls back to high-quality Lanczos so it never fails."""
        from PIL import Image
        try:
            import numpy as np, cv2, sys, os
            _cap = r"F:/VC Projects/CaptionsAI"
            if os.path.isdir(_cap) and _cap not in sys.path:
                sys.path.insert(0, _cap)
            from upscaler import upscale as _cap_upscale     # CaptionsAI's tuned Real-ESRGAN
            bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            out = _cap_upscale(bgr)
            return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        except Exception:
            w, h = image.size
            return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # ── hires-fix (Phase 2): upscale, then img2img-refine at the higher res ──
    def hires_fix(self, image, prompt, negative="", *, hr_scale=1.5, hr_denoise=0.3,
                  steps=20, cfg=6.0, seed=-1, sampler="DPM++ 2M SDE", guidance_rescale=0.0):
        """HD pass: upscale the base render, then a low-denoise img2img refine so SDXL adds real detail."""
        from PIL import Image
        big = self.upscale(image, scale=float(hr_scale))
        w, h = ((big.size[0] // 8) * 8, (big.size[1] // 8) * 8)     # snap to /8 for SDXL
        if (w, h) != big.size:
            big = big.resize((w, h), Image.LANCZOS)
        return self.img2img(big, prompt, negative, strength=float(hr_denoise), steps=int(steps),
                            cfg=float(cfg), seed=seed, sampler=sampler,
                            guidance_rescale=float(guidance_rescale))

    # ── ControlNet (Phase 3) ─────────────────────────────────────────────────
    def _ensure_controlnet_pipe(self, controlnet_model):
        """ControlNet pipeline sharing the base components + a loaded ControlNetModel.
        controlnet_model: a local dir/.safetensors or an HF id for an SDXL ControlNet."""
        base = self._ensure_pipe()
        key = str(controlnet_model)
        if getattr(self, "_controlnet_pipe", None) is not None and getattr(self, "_cn_key", None) == key:
            return self._controlnet_pipe
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        dtype = torch.float16 if self._dtype == "fp16" else torch.float32
        if str(controlnet_model).lower().endswith(".safetensors"):
            cn = ControlNetModel.from_single_file(controlnet_model, torch_dtype=dtype)
        else:
            cn = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=dtype)
        pipe = StableDiffusionXLControlNetPipeline(controlnet=cn, **base.components)
        # ControlNet adds a ~2.5 GB conditioning model ON TOP of SDXL. It must stay fully on the GPU with
        # NO cpu-offload hooks: this pipe SHARES the base pipe's UNet, and model_cpu_offload hooks on that
        # shared UNet (while the base txt2img pipe keeps it resident) put the module in a conflicting device
        # state that NATIVE-CRASHED peft LoRA injection (Windows access violation in Linear.__init__ during
        # load_lora_weights — took the whole Suite down). Instead fit the SDXL+ControlNet pair via ATTENTION
        # SLICING + VAE tiling: large peak-memory savers that add no hooks, so per-render LoRA loading stays
        # safe and it still fits a 16 GB card without the shared-RAM thrash. (Slicing trades a little speed
        # for memory; ControlNet renders are deliberate work anyway.)
        pipe = pipe.to(self._pick_device())
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_tiling(); pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        self._controlnet_pipe = pipe
        self._cn_key = key
        return pipe

    def controlnet(self, control_image, prompt, negative="", *, controlnet_model="",
                   controlnet_scale=0.8, steps=30, cfg=6.0, width=1024, height=1024,
                   seed=-1, sampler="DPM++ 2M SDE", guidance_rescale=0.0):
        """Render conditioned on a control map (openpose/depth/canny...). controlnet_model = the SDXL
        ControlNet weights (local path / HF id). `prompt` may carry <lora:> tags."""
        import torch
        if not controlnet_model:
            raise ValueError("controlnet(): no ControlNet model set (pass controlnet_model=path).")
        pipe = self._ensure_controlnet_pipe(controlnet_model)
        loras = [(m.group(1), float(m.group(2) or 1.0)) for m in _LORA_TAG.finditer(prompt or "")]
        clean = _LORA_TAG.sub("", prompt or "").strip()
        applied, missing = self._apply_loras(pipe, loras)
        self._set_scheduler(pipe, sampler)
        _gd = "cuda" if torch.cuda.is_available() else "cpu"
        gen = None if int(seed) < 0 else torch.Generator(device=_gd).manual_seed(int(seed))
        cb = self._begin_progress(int(steps))
        kw = dict(prompt=clean, negative_prompt=negative or "", image=control_image,
                  controlnet_conditioning_scale=float(controlnet_scale),
                  num_inference_steps=int(steps), guidance_scale=float(cfg),
                  width=int(width), height=int(height), generator=gen,
                  callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
        if guidance_rescale and float(guidance_rescale) > 0:
            kw["guidance_rescale"] = float(guidance_rescale)   # SDXL-ControlNet pipeline lacks this on
        try:                                                   # diffusers 0.31 → TypeError-guarded below
            out = pipe(**kw).images[0]
        except TypeError:
            kw.pop("guidance_rescale", None)
            out = pipe(**kw).images[0]
        finally:
            self._end_progress()
        return out, {"mode": "controlnet", "controlnet_scale": float(controlnet_scale),
                     "loras": applied, "missing_loras": missing}

    # ── ADetailer (Phase 3): detect faces/hands, img2img-inpaint each for detail ──
    def adetailer(self, image, prompt, negative="", *, detectors=("face",), models=None,
                  strength=0.3, steps=20, cfg=6.0, conf=0.3, seed=-1, sampler="DPM++ 2M SDE",
                  guidance_rescale=0.0):
        """Detect regions (YOLO) and refine each via img2img for sharper faces/hands. `models` maps a
        detector name -> a YOLO .pt path. Returns the input unchanged if ultralytics / models are absent."""
        from PIL import Image, ImageDraw, ImageFilter
        info = {"mode": "adetailer", "detected": 0, "detectors": list(detectors)}
        models = models or {}
        try:
            import numpy as np
            from ultralytics import YOLO
        except Exception:
            info["note"] = "ultralytics not installed — ADetailer skipped"
            return image, info
        out = image.convert("RGB")
        for det in detectors:
            mp = models.get(det)
            if not mp or not os.path.isfile(mp):
                info.setdefault("skipped", []).append(f"{det} (no model)")
                continue
            try:
                yolo = self._yolo_cache.get(mp)     # keep detectors resident — re-instantiating YOLO from
                if yolo is None:                     # disk on every render (and every unit) is pure overhead
                    yolo = self._yolo_cache[mp] = YOLO(mp)
                res = yolo.predict(np.array(out), conf=float(conf), verbose=False)
            except Exception as e:
                info.setdefault("skipped", []).append(f"{det} ({e})")
                continue
            for r in res:
                boxes = r.boxes.xyxy.cpu().numpy() if getattr(r, "boxes", None) is not None else []
                for box in boxes:
                    x1, y1, x2, y2 = (int(v) for v in box[:4])
                    pad = int(0.15 * max(x2 - x1, y2 - y1))
                    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                    x2, y2 = min(out.width, x2 + pad), min(out.height, y2 + pad)
                    crop = out.crop((x1, y1, x2, y2))
                    cw, ch = crop.size
                    sc = 768.0 / max(cw, ch)                       # refine at ~768 for detail
                    ww, hh = (max(64, int(cw * sc)) // 8 * 8, max(64, int(ch * sc)) // 8 * 8)
                    fixed, _ = self.img2img(crop.resize((ww, hh), Image.LANCZOS), prompt, negative,
                                            strength=float(strength), steps=int(steps), cfg=float(cfg),
                                            seed=seed, sampler=sampler, preview=False,
                                            guidance_rescale=float(guidance_rescale))
                    fixed_full = fixed.resize((cw, ch), Image.LANCZOS)
                    # FEATHERED blend — NOT a hard rectangular paste. Pasting the crop straight back left a
                    # visible box seam at every inpaint region (the "hard blocks"). Build a mask that's white
                    # over the detected feature (inside the pad margin) and gaussian-fades to 0 before the
                    # crop edge, so the refined region blends smoothly into the surrounding pixels.
                    m = max(6, int(pad))
                    mask = Image.new("L", (cw, ch), 0)
                    ImageDraw.Draw(mask).rectangle([m, m, cw - m - 1, ch - m - 1], fill=255)
                    mask = mask.filter(ImageFilter.GaussianBlur(m * 0.6))
                    out.paste(fixed_full, (x1, y1), mask)
                    info["detected"] += 1
        return out, info

    # ── VRAM arbiter hook ────────────────────────────────────────────────────
    def unload(self):
        """Free the pipeline from VRAM (so the brain can take the card)."""
        self._pipe = None
        self._img2img_pipe = None       # shares components with _pipe
        self._controlnet_pipe = None    # Phase 3
        self._preview_vae = None        # tiny TAESD preview decoder
        self._loaded_ckpt = None
        self._active_loras = None       # drop the resident-LoRA record — the pipes it referred to are gone
        self._yolo_cache = {}           # free resident ADetailer detectors
        self.progress = {"active": False, "step": 0, "total": 0, "image_b64": None, "t0": 0.0}
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ── LoraGenesys Unit (.lgu) entry point ──────────────────────────────────────
# A .lgu is a self-contained "checkpoint that carries a brain": frozen SDXL + JoyCaption visual brain
# + trained composition adapter. Identity comes from the prompt (character NAME) / a LoRA; the brain
# supplies composition from a REFERENCE image. Both Jarvis and SD.UI render through this one call.
def unit(lgu_path, prompt, reference, *, negative="worst quality, low quality, blurry, deformed",
         scale=0.7, steps=30, cfg=6.0, seed=-1, size=1024, device=None):
    """Render with a LoraGenesys Unit. Returns a PIL.Image. `reference` = image path / PIL / png bytes."""
    from shared.lgu_runtime import get_unit
    return get_unit(lgu_path, device=device).render(
        prompt, reference, negative=negative, scale=scale, steps=steps, cfg=cfg, seed=seed, size=size)


def list_units(units_dir="E:/LoraGenesys"):
    """Available .lgu filenames in units_dir. Deliberately does NOT import lgu_runtime — that module
    imports torch at module level, and the UI lists units while building the checkpoint dropdown at
    startup. Dragging the whole ML stack in just to read a directory would stall the launcher (and, off
    the main thread, risks the native access-violation that killed the Suite). Reading a dir needs no torch."""
    if not os.path.isdir(units_dir):
        return []
    return sorted(f for f in os.listdir(units_dir) if f.lower().endswith(".lgu"))
