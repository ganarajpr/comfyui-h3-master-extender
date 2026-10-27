"""Opt-in PTD sampling for the Master Extender (A/B against the turbo-LoRA recipe).

accel_mode "PTD (adapter)" runs pass 1 with the per-hop PTD adapters from ComfyUI-MiniMaxH3-PTD (one model
evaluation per trained hop). pass2_engine "PTD last hop" replaces the pass-2 refine with the adapter's FINAL hop
only, run at pass-2 size on the re-noised, upscaled pass-1 latent.

All PTD numerics (loader, controller, hop clocks, heads, audio clock, x0 -> velocity) come from the installed
pack and ComfyUI-PTD; this module only adds the start-from-hop sampler and the glue.
"""

from __future__ import annotations

import importlib
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import torch

_LOG = logging.getLogger("MiniMaxH3Master")

PTD_ACCEL_MODE = "PTD (adapter)"
PASS2_ENGINE_LORA = "LoRA refine"
PASS2_ENGINE_PTD = "PTD last hop"
PASS2_ENGINES = [PASS2_ENGINE_LORA, PASS2_ENGINE_PTD]
PTD_FOLDER = "ptd"
PACK_URL = "https://github.com/dheeai/comfyui-minimaxh3-ptd"


def is_ptd_mode(accel_mode):
    return str(accel_mode) == PTD_ACCEL_MODE


def ptd_adapter_choices():
    """["none"] + the adapters in models/ptd. The folder is registered here too so load order does not matter."""
    try:
        import folder_paths

        if PTD_FOLDER not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(PTD_FOLDER, str(Path(folder_paths.models_dir) / PTD_FOLDER))
        files = [f for f in folder_paths.get_filename_list(PTD_FOLDER) if f.endswith(".safetensors")]
    except Exception:
        files = []
    return ["none"] + files


def default_ptd_adapter(choices=None):
    """The adapter to use when PTD mode is on but none is selected: the first RefAsymOPD file, else the first adapter,
    else None (nothing installed in models/ptd). The panel makes the same pick when PTD mode is switched on."""
    files = [f for f in (ptd_adapter_choices() if choices is None else choices) if f and f != "none"]
    return next((f for f in files if "refasymopd" in f.lower()), files[0] if files else None)


def ptd_pack():
    """The registered ComfyUI-MiniMaxH3-PTD modules (loader, patcher, sampler), looked up the way the pack
    itself finds ComfyUI-PTD: ComfyUI imports a custom-node folder under a path-derived module name."""
    import nodes

    loader_node = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3PTDLoader")
    if loader_node is None:
        raise RuntimeError(
            f"accel_mode '{PTD_ACCEL_MODE}' needs the ComfyUI-MiniMaxH3-PTD node pack (and ComfyUI-PTD) in "
            f"custom_nodes: {PACK_URL}. It is not loaded (no MiniMaxH3PTDLoader node)."
        )
    package = loader_node.__module__.rsplit(".", 1)[0]
    return SimpleNamespace(
        loader=importlib.import_module(package + ".loader"),
        patcher=importlib.import_module(package + ".patcher"),
        sampler=importlib.import_module(package + ".sampler"),
    )


def with_ptd_p1_settings(settings, adapter):
    """Add the adapter to the pass-1 fingerprint settings, only in PTD mode, so existing pass-1 caches keep hitting."""
    if adapter:
        settings = dict(settings)
        settings["ptd"] = str(adapter)
    return settings


def hop_tail_sigmas(grid, start_hop):
    """The checkpoint's hop knots from ``start_hop`` on: ``len(grid) - 1 - start_hop`` model evaluations."""
    hops = len(grid) - 1
    if isinstance(start_hop, bool) or not isinstance(start_hop, int) or not 0 <= start_hop < hops:
        raise ValueError(f"PTD start hop must be an integer in [0, {hops})")
    return grid[start_hop:]


def renoise_to_hop(noise, clean, video_numel, video_time, audio_time):
    """Flow-matching state at a hop: ``(1 - t) * clean + t * noise`` per stream, float64 then fp32 carry.

    ``noise`` and ``clean`` are ComfyUI's packed latents ``[B, 1, N]`` (video elements first, then audio). Both are
    already in the sampler's space (the PTD patch makes the audio carry physical, scale 1), so each stream is
    blended at its own clock: video at ``video_time``, audio at ``audio_time`` (PTD's audio_sigma of it)."""
    if not isinstance(noise, torch.Tensor) or not isinstance(clean, torch.Tensor):
        raise TypeError("PTD hop start needs packed tensor latents")
    if noise.shape != clean.shape:
        raise ValueError(f"noise {tuple(noise.shape)} and latent {tuple(clean.shape)} differ in shape")
    n = int(video_numel)
    if not 0 < n <= noise.shape[-1]:
        raise ValueError(f"video stream size {n} does not fit a packed latent of {noise.shape[-1]} elements")
    out = torch.empty(noise.shape, dtype=torch.float32, device=noise.device)
    for part, t in ((slice(0, n), video_time), (slice(n, None), audio_time)):
        t = float(t)
        out[..., part] = ((1.0 - t) * clean[..., part].to(torch.float64) + t * noise[..., part].to(torch.float64)).float()
    return out


def _video_numel(model):
    """Element count of the video stream in the packed latent, from the base model the guider set it on."""
    try:
        shapes = model.inner_model.inner_model.latent_shapes
        return math.prod(shapes[0][1:])
    except (AttributeError, TypeError, IndexError) as error:
        raise RuntimeError("PTD hop start could not read the latent stream layout from the sampler") from error


@torch.no_grad()
def sample_pdd_ptd_from_hop(model, x, sigmas, extra_args=None, callback=None, disable=None, *,
                            controller, start_hop, upstream=None, video_numel=None):
    """The pack's fixed-hop PTD recurrence, started at hop ``start_hop`` from a re-noised clean latent.

    Hops ``start_hop .. hops-1`` each run one evaluation under ``controller.activate(hop, time)`` (the pack's
    controller validates the hop's trained time), state carried in fp32, exactly as ``sample_pdd_ptd``. The start
    state is ``(1-t) * latent + t * noise`` at the hop's knot, with the audio stream on PTD's audio clock."""
    del disable
    if upstream is None:
        upstream = ptd_pack().patcher.ptd_modules()[1]
    grid = controller.sigmas
    tail = hop_tail_sigmas(grid, start_hop)
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.shape != tail.shape
            or not torch.equal(sigmas.detach().cpu(), tail.detach().cpu().to(dtype=sigmas.dtype))):
        raise ValueError(f"PTD hop {start_hop} start needs this checkpoint's hop grid from that hop on ({len(tail)} knots)")
    extra_args = {} if extra_args is None else dict(extra_args)
    if extra_args.get("denoise_mask") is not None:
        raise ValueError("PTD sampling does not support denoise masks")
    clean, noise = getattr(model, "latent_image", None), getattr(model, "noise", None)
    if clean is None or noise is None:
        raise RuntimeError("PTD hop start needs the sampler's latent and noise")
    start_time = float(grid[start_hop])
    state = renoise_to_hop(noise, clean, _video_numel(model) if video_numel is None else video_numel,
                           start_time, upstream.audio_sigma(start_time))
    if state.shape != x.shape:
        raise ValueError(f"hop start state {tuple(state.shape)} differs from the sampler latent {tuple(x.shape)}")
    batch = state.shape[0]
    knots = grid.to(device=state.device)
    for hop in range(start_hop, len(grid) - 1):
        time = float(knots[hop])
        with controller.activate(hop, time):
            denoised = model(state, knots[hop].expand(batch), **extra_args)
        velocity = upstream.denoised_to_velocity(state, denoised, time)
        if callback is not None:
            callback({"x": state, "i": hop - start_hop, "sigma": knots[hop],
                      "sigma_hat": knots[hop], "denoised": denoised})
        delta = (knots[hop + 1] - knots[hop]).float()
        state = state.float() + delta * velocity.float()
    return state


def make_hop_sampler(controller, start_hop):
    """A ComfyUI SAMPLER running hops ``start_hop..`` of one loaded checkpoint."""
    from comfy.samplers import KSAMPLER

    hop_tail_sigmas(controller.sigmas, start_hop)
    return KSAMPLER(sample_pdd_ptd_from_hop, extra_options={"controller": controller, "start_hop": int(start_hop)})
