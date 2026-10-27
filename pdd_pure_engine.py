"""
Pure 2-Stage 2-Pass MiniMax H3 Master Engine (HyperFlow 8-step / Turbo LoRA) + 3D Latent Upscaler
====================================================================================================

Stage 1: Pass 1 Low-Res Draft Ref2VA Generation (HyperFlow grid, or turbo LoRA + BasicScheduler)
Stage 2: 3D Latent Upscaling (MinimaxH3LatentUpscaler3D) + Concat AV
Stage 3: Pass 2 High-Res Refinement via the quality tail (denoise 0.25)
"""

import inspect
import math
import logging
import os
import sys
import torch
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.nested_tensor
import comfy.utils
from comfy.ldm.modules.attention import get_attention_function
import nodes
import node_helpers
from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ReferenceToVideo,
    MiniMaxH3SigmaShift,
    _empty_av_latent,
    _resize,
    CANVAS_MULTIPLE,
)
from .motion_context_ram import (
    MiniMaxH3MotionContextRAM,
    _streams_from_latent,
)
from . import x0_debug
from . import ptd_mode

_LOG = logging.getLogger("minimax_h3_master_extender.engine")


def parse_resolution(res_str: str, default_w: int, default_h: int):
    """Parse '1344x768 (16:9)' or '1344x768' into (w, h)."""
    if not res_str:
        return default_w, default_h
    try:
        core = res_str.split()[0]
        parts = core.split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return default_w, default_h


def duration_to_h3_frames(duration_sec: float) -> int:
    """Convert duration in seconds to nearest valid H3 frame count (17k + 5)."""
    frames = max(5, int(round(float(duration_sec) * 24.0)))
    while frames % 17 != 5:
        frames += 1
    return frames


def _safe_get_output(obj, idx: int, key: str = None):
    """Safely extracts an output by index or key from tuple, list, dict, or io.NodeOutput."""
    if obj is None:
        return None
    if hasattr(obj, "__getitem__"):
        try:
            return obj[idx]
        except (KeyError, TypeError, IndexError):
            pass
    if hasattr(obj, "result"):
        try:
            return obj.result[idx]
        except Exception:
            pass
    if isinstance(obj, dict):
        if key and key in obj:
            return obj[key]
        vals = list(obj.values())
        if idx < len(vals):
            return vals[idx]
    if hasattr(obj, "as_dict"):
        try:
            d = obj.as_dict()
            if key and key in d:
                return d[key]
            vals = list(d.values())
            if idx < len(vals):
                return vals[idx]
        except Exception:
            pass
    return obj


def apply_semantic_bridge(positive, adapter, alpha, magnitude_match, node_mappings, label="", log=None):
    """Run the BUNNY H3 Conditioning Bridge (a small residual MLP over the
    5120-d H3 text conditioning) on `positive`. `adapter` "none" / "" or
    alpha 0 = pass-through. The bridge pack (custom_nodes/BUNNY_H3_Conditioning_Bridge)
    is an optional dependency resolved from `node_mappings` at run time."""
    if not adapter or str(adapter).lower() == "none" or float(alpha) == 0.0:
        return positive
    bridge_cls = node_mappings.get("BunnyH3ConditioningBridge")
    if bridge_cls is None:
        raise RuntimeError(
            "semantic_bridge is set but the BUNNY H3 Conditioning Bridge node is not installed "
            "(clone github.com/aa335615543-ux/BUNNY_H3_Conditioning_Bridge into custom_nodes and "
            "restart ComfyUI), or set semantic_bridge to none."
        )
    res = bridge_cls().apply(positive, str(adapter), float(alpha), str(magnitude_match), True)
    bridged = _safe_get_output(res, 0, "conditioning")
    if log is not None:
        log.info("Semantic bridge%s: %s @ alpha %.2f (%s)",
                 f" ({label})" if label else "", adapter, float(alpha), magnitude_match)
    return bridged


_UPSCALER_PRECISION_CACHE = {}


def _upscaler_precision(model_name, default="fp16"):
    """Compute precision for the latent upscaler, matched to the checkpoint's own
    storage dtype (a bf16 requantization runs as bf16, an fp16 file as fp16; fp32
    files keep the fast fp16 default). Cached per file name."""
    if model_name in _UPSCALER_PRECISION_CACHE:
        return _UPSCALER_PRECISION_CACHE[model_name]
    precision = default
    try:
        import folder_paths
        from safetensors import safe_open
        path = folder_paths.get_full_path("latent_upscale_models", model_name)
        if path:
            with safe_open(path, framework="pt", device="cpu") as f:
                dtypes = {f.get_slice(k).get_dtype() for k in list(f.keys())[:64]}
            if "BF16" in dtypes:
                precision = "bf16"
            elif dtypes == {"F16"}:
                precision = "fp16"
    except Exception as exc:  # unknown format: keep the default
        _LOG.debug("upscaler precision probe failed for %s: %s", model_name, exc)
    _UPSCALER_PRECISION_CACHE[model_name] = precision
    _LOG.info("Latent upscaler %s -> compute precision %s", model_name, precision)
    return precision


def pass_sampler_name(turbo_mode, sampler_name):
    """HyperFlow was distilled for plain Euler; only Turbo LoRA mode uses the user's sampler."""
    return sampler_name if turbo_mode else "euler"


def hyperflow_tail_sigmas(grid, denoise):
    """Last round(steps x denoise) steps of HyperFlow's own 9-point grid (steps + 1 points)."""
    steps = len(grid) - 1
    tail = max(1, int(round(steps * float(denoise))))
    return grid[-(tail + 1):]


def capture_x0_callback(base, store):
    """Sampler callback that records the latest denoised x0 in `store["x0"]` and then defers to `base`.

    ComfyUI hands every callback (step, x0, x, total_steps), so the clean estimate needs no look inside the guider
    (CFGGuider drops its `inner_model` as soon as sample() returns)."""
    def callback(step, x0, x, total_steps):
        store["x0"] = x0
        if base is not None:
            return base(step, x0, x, total_steps)
    return callback


def clean_latent_from_x0(model_patcher, x0):
    """The last step's x0 (model latent space) mapped back to the latent the VAE and upscaler expect."""
    return model_patcher.model.process_latent_out(x0.to(torch.float32))


def dual_split_index(grid, split):
    """Grid index where pass 2 takes over in dual sampling: at least 1, and pass 2 keeps at least 1 step (split 7 = only
    HyperFlow's final step at pass-2 size)."""
    return max(1, min(int(split), len(grid) - 2))


def dual_pass1_sigmas(grid, split):
    """Pass 1 of dual sampling with early stop: only grid[:k+1]."""
    return grid[:dual_split_index(grid, split) + 1]


def dual_pass2_sigmas(grid, split):
    """Pass 2 of dual sampling: the grid tail from index k, i.e. len(grid) - 1 - k steps."""
    return grid[dual_split_index(grid, split):]


def with_dual_p1_settings(settings, active, split, early_stop):
    """Add the dual-sampling keys to the pass-1 fingerprint settings, only when they change what pass 1 produces
    (early stop). Otherwise the dict is returned untouched so existing pass-1 caches keep hitting."""
    if active and early_stop:
        settings = dict(settings)
        settings["dual"] = [int(split), True]
    return settings


def with_pass1_steps_p1_settings(settings, steps):
    """Add the pass-1 step cut to the pass-1 fingerprint settings, only when it is on (steps > 0). Otherwise the
    dict is returned untouched so existing pass-1 caches keep hitting."""
    if int(steps or 0) > 0:
        settings = dict(settings)
        settings["pass1_steps"] = int(steps)
    return settings


def pass1_steps_sigmas(grid, steps):
    """Pass 1 stopped after `steps` of the trained grid's steps: grid[:steps + 1]. Needs 1 <= steps < len(grid) - 1."""
    return grid[:int(steps) + 1]


PASS2_CHUNK_WHEN = ("above 720p", "always")
PASS2_CHUNK_MIN_SHORT_SIDE = 800


SPARSE_SINK_CHOICES = ("exact_kv_and_rows", "exact_kv", "off")


def resolve_sparse_settings(hyperflow, method, start_percent, dense_blocks, sink,
                            default_start=0.2):
    """Resolve the three 'auto' sparse-attention inputs.

    Auto means HyperFlow's validated sol-attn recipe (start 0.16, dense blocks
    0,1, sink off) when HyperFlow is the accelerator and the method is sol-attn,
    else the long-standing defaults (start ``default_start``, no dense blocks,
    exact_kv_and_rows). Any value the user set wins. Returns
    ``{"start", "dense_blocks", "sink", "sources"}`` where sources maps each
    field to "user", "hyperflow recipe" or "default".
    """
    recipe = bool(hyperflow) and method == "sol-attn"
    sources = {}
    try:
        start = float(start_percent)
    except (TypeError, ValueError):
        start = -1.0
    if start < 0:
        start, sources["start"] = (0.16, "hyperflow recipe") if recipe else (float(default_start), "default")
    else:
        sources["start"] = "user"
    blocks = "auto" if dense_blocks is None else str(dense_blocks).strip()
    if blocks.lower() == "auto":
        blocks, sources["dense_blocks"] = ("0,1", "hyperflow recipe") if recipe else ("", "default")
    else:
        sources["dense_blocks"] = "user"
    pick = str(sink or "auto").strip()
    if pick not in SPARSE_SINK_CHOICES:
        pick, sources["sink"] = ("off", "hyperflow recipe") if recipe else ("exact_kv_and_rows", "default")
    else:
        sources["sink"] = "user"
    return {"start": start, "dense_blocks": blocks, "sink": pick, "sources": sources}


class MasterEngine:
    def __init__(
        self,
        model,
        clip,
        vae,
        audio_vae,
        upscaler_model="minimax_h3_latent_upscaler_3d_fp16.safetensors",
        turbo_steps="8",
        shift_video=12.0,
        shift_audio=3.0,
        attention_backend="comfy kitchen attention",
        smart_offload=True,
        accel_mode="HyperFlow 8-step",
        turbo_lora="none",
        turbo_lora_strength=1.0,
        sampler_name="res_multistep",
        scheduler_name="simple",
        sla_enabled=False,
        sla_sparsity=0.9,
        pass2_chunk_frames=124,
        pass2_chunk_overlap=22,
        pass2_chunk_when="above 720p",
        background_busy=None,
        wait_background=None,
        sparse_method="sla",
        sparse_tau=1.3,
        hyperflow_file="custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors",
        hyperflow_curve_refit=True,
        hyperflow_strength=1.0,
        hyperflow_lora_mode="bypass",
        sparse_start_percent=-1.0,
        sparse_dense_blocks="auto",
        sparse_sink="auto",
        pass2_sla_sparsity=0.0,
        ptd_adapter="none",
    ):
        self.raw_model = model
        self.accel_mode = str(accel_mode)
        self.turbo_mode = self.accel_mode.lower().startswith("turbo")
        self.ptd_mode = ptd_mode.is_ptd_mode(self.accel_mode)
        self.hyperflow_mode = not (self.turbo_mode or self.ptd_mode)
        self.ptd_adapter = ptd_adapter if ptd_adapter and ptd_adapter != "none" else None
        self.ptd_controller = None
        self.ptd_base = None
        self.pass2_engine = ptd_mode.PASS2_ENGINE_LORA
        self.hyperflow_file = str(hyperflow_file)
        self.hyperflow_curve_refit = bool(hyperflow_curve_refit)
        self.hyperflow_strength = float(hyperflow_strength)
        self.hyperflow_lora_mode = "merge" if str(hyperflow_lora_mode).lower() == "merge" else "bypass"
        self.hyperflow_pruned_refit = False
        self.turbo_lora = turbo_lora if turbo_lora and turbo_lora != "none" else None
        self.turbo_lora_strength = float(turbo_lora_strength)
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.sla_enabled = bool(sla_enabled)
        self.sla_sparsity = float(sla_sparsity)
        self.pass2_sla_sparsity = float(pass2_sla_sparsity or 0.0)
        self.pass2_sla_model = None
        self.sparse_method = str(sparse_method or "sla")
        self.sparse_tau = float(sparse_tau)
        self.sparse_start_percent = sparse_start_percent
        self.sparse_dense_blocks = sparse_dense_blocks
        self.sparse_sink = sparse_sink
        self.pass2_chunk_frames = int(pass2_chunk_frames or 0)
        self.pass2_chunk_overlap = int(pass2_chunk_overlap or 0)
        self.pass2_chunk_when = pass2_chunk_when if pass2_chunk_when in PASS2_CHUNK_WHEN else "above 720p"
        # Hooks from the Master node: is a background decode running, and how
        # to wait for it. The reclaim must not evict models under a live decode.
        self.background_busy = background_busy or (lambda: False)
        self.wait_background = wait_background or (lambda: None)
        self.clip = clip
        self.vae = vae
        self.audio_vae = audio_vae
        self.upscaler_model = upscaler_model
        self.turbo_steps = str(turbo_steps)
        self.shift_video = float(shift_video)
        self.shift_audio = float(shift_audio)
        self.attention_backend = attention_backend
        self.smart_offload = smart_offload

        self.motion_ram = MiniMaxH3MotionContextRAM()
        self.prepared_model = None
        self.pass1_sigmas = None
        self.dual_active = False
        self.dual_split_step = 2
        self.pass1_early_stop = False

    def initialize_model(self):
        """Prepare base model with SigmaShift, the acceleration (Turbo LoRA or HyperFlow) and the attention backend."""
        if self.prepared_model is not None:
            return self.prepared_model, self.pass1_sigmas

        if self.ptd_mode:
            accel_desc = f"PTD adapter: {self.ptd_adapter or 'none'}"
        elif self.turbo_mode:
            accel_desc = f"Turbo LoRA: {self.turbo_lora or 'none'} ({self.turbo_steps} steps)"
        else:
            accel_desc = f"HyperFlow 8-step: {self.hyperflow_file}"
        _LOG.info(f"Initializing Master Engine: Shift ({self.shift_video}, {self.shift_audio}), {accel_desc}")

        # 1. Apply SigmaShift
        try:
            res_shift = MiniMaxH3SigmaShift.execute(
                self.raw_model,
                shift_video=self.shift_video,
                shift_audio=self.shift_audio,
            )
            shifted_model = res_shift[0] if hasattr(res_shift, "__getitem__") else getattr(res_shift, "result", [self.raw_model])[0]
        except Exception as e:
            _LOG.warning(f"Error applying MiniMaxH3SigmaShift: {e}")
            shifted_model = self.raw_model

        self.shifted_model = shifted_model

        # 2. Acceleration: a plain turbo / lightning LoRA, or HyperFlow
        if self.ptd_mode:
            accelerated, sigmas_p1 = self._prepare_ptd_model(shifted_model)
        elif self.turbo_mode:
            accelerated, sigmas_p1 = self._prepare_turbo_model(shifted_model)
        else:
            accelerated, sigmas_p1 = self._prepare_hyperflow_model(shifted_model)
        self.accel_model = accelerated
        accelerated = self._finish_model(accelerated)

        self.prepared_model = accelerated
        self.pass1_sigmas = sigmas_p1
        return self.prepared_model, self.pass1_sigmas

    def _finish_model(self, model, sparse=True, sparsity=None):
        """Dense attention backend, then optional block-sparse attention on top
        (SLA wraps the dense backend as its fall-through, so order matters)."""
        attention_name = {
            "comfy kitchen attention": "comfy_kitchen_int8",
            "sage attention 2.2": "sage",
            "pytorch attention": "pytorch",
        }.get(self.attention_backend)
        if attention_name is None:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")
        attention_function = get_attention_function(attention_name, None)
        if attention_function is None:
            raise RuntimeError(
                f"Attention backend '{self.attention_backend}' is unavailable. "
                "Check its installation or select another attention backend."
            )
        model = model.clone()
        model.set_model_optimized_attention(attention_function)
        _LOG.info("Sampling attention: %s", self.attention_backend)
        return self._apply_sla(model, sparsity) if sparse else model

    # ------------------------------------------------------------------
    # Pass-2 (refine) LoRA: a different LoRA for the high-res tail only
    # ------------------------------------------------------------------
    def configure_pass2_lora(self, lora="none", strength=1.0, mode="stack", steps=0):
        """`mode`: "stack" adds the LoRA on top of the engine's acceleration
        (turbo LoRA or HyperFlow); "replace" samples pass 2 with base model +
        this LoRA only (for step-distilled LoRAs such as a 3-step turbo).
        `steps` (0 = same as pass 1) is the full schedule length the tail is cut from."""
        self.pass2_lora = lora if lora and lora != "none" else None
        self.pass2_lora_strength = float(strength)
        self.pass2_lora_mode = "replace" if str(mode).lower().startswith("replace") else "stack"
        if self.hyperflow_mode and self.pass2_lora and self.pass2_lora_mode != "replace":
            _LOG.info("HyperFlow: pass-2 LoRA forces 'replace' mode (pass 2 runs on the SigmaShift model + this "
                      "LoRA, without HyperFlow, so HyperFlow never sees a non-trained sigma tail)")
            self.pass2_lora_mode = "replace"
        if self.ptd_mode and self.pass2_lora and self.pass2_lora_mode != "replace":
            _LOG.info("PTD: pass-2 LoRA forces 'replace' mode (a PTD model only samples under an active hop)")
            self.pass2_lora_mode = "replace"
        self.pass2_steps = int(steps or 0)
        self.pass2_model = None
        self.pass2_sla_model = None

    def configure_pass2_engine(self, engine=ptd_mode.PASS2_ENGINE_LORA):
        """"PTD last hop": pass 2 is the PTD adapter's final hop at pass-2 size (needs accel_mode PTD (adapter)).
        The default keeps the normal pass-2 refine. Call after configure_pass2_lora."""
        engine = str(engine or ptd_mode.PASS2_ENGINE_LORA)
        if engine not in ptd_mode.PASS2_ENGINES:
            raise ValueError(f"pass2_engine '{engine}' not in {ptd_mode.PASS2_ENGINES}")
        self.pass2_engine = engine
        if self.ptd_mode and self.ptd_adapter is None:
            # A workflow saved before the PTD inputs (or one left on "none") must still run: use the default adapter.
            pick = ptd_mode.default_ptd_adapter()
            if pick is None:
                raise ValueError(f"accel_mode '{ptd_mode.PTD_ACCEL_MODE}' needs a PTD adapter, and models/ptd has none")
            _LOG.warning("PTD: no ptd_adapter selected; using %s", pick)
            self.ptd_adapter = pick
        if self.ptd_mode and engine == ptd_mode.PASS2_ENGINE_LORA and not self._pass2_lora_active():
            # A PTD model only samples at its trained hops, so a LoRA refine without a LoRA has nothing to run on.
            # The adapter's own final hop is the PTD-native refine.
            _LOG.warning("PTD: pass2_engine 'LoRA refine' without a pass2_lora; running the adapter's last hop instead")
            self.pass2_engine = engine = ptd_mode.PASS2_ENGINE_PTD
        if engine == ptd_mode.PASS2_ENGINE_PTD:
            if not self.ptd_mode:
                # The panel hides pass2_engine outside PTD mode, so a value left over from a PTD run must not block the
                # job: without a PTD pass 1 there is no adapter to run, so pass 2 is the normal LoRA refine.
                _LOG.warning("pass2_engine '%s' needs accel_mode '%s'; running the normal pass-2 refine instead",
                             engine, ptd_mode.PTD_ACCEL_MODE)
                self.pass2_engine = ptd_mode.PASS2_ENGINE_LORA
                return
            if self._pass2_lora_active():
                _LOG.info("PTD last hop: pass-2 LoRA %s is not used", self.pass2_lora)

    def _match_hyperflow_build(self, pruned_base, installed=None):
        """A HyperFlow build that does not fit the loaded base (pruned vs full) is swapped for its twin when that is
        installed; the two builds differ only by '_pruned' in the name. Leaves the choice alone otherwise."""
        pruned_file = "_pruned" in self.hyperflow_file.lower()
        if pruned_base == pruned_file:
            return
        twin = (self.hyperflow_file.replace(".safetensors", "_pruned.safetensors") if pruned_base
                else self.hyperflow_file.replace("_pruned", ""))
        if installed is None:
            try:
                import folder_paths
                installed = folder_paths.get_filename_list("hyperflow")
            except Exception:
                installed = []
        if twin in installed:
            _LOG.warning("HyperFlow: %s is the %s-base build but the loaded base is %s; using %s", self.hyperflow_file,
                         "pruned" if pruned_file else "full", "pruned" if pruned_base else "full", twin)
            self.hyperflow_file = twin

    def _resolve_windowed_last_hop(self, frame_count, width, height, clip_index=0):
        """"PTD last hop" refines the whole clip in one evaluation; it has no windowed form. When the settings ask for a
        windowed pass 2 (1080p-class output, or pass2_chunk_when "always"), use the windowed LoRA refine if a pass-2
        LoRA is set; otherwise run the last hop on the whole clip anyway. Never raises."""
        if not (self.ptd_mode and self.pass2_engine == ptd_mode.PASS2_ENGINE_PTD
                and self._pass2_chunking_active(frame_count, width, height)):
            return
        if self._pass2_lora_active():
            _LOG.warning("Clip %d: 'PTD last hop' has no windowed form; this pass 2 is windowed, so it runs the "
                         "windowed LoRA refine (%s) instead", clip_index + 1, self.pass2_lora)
            self.pass2_engine = ptd_mode.PASS2_ENGINE_LORA
        else:
            _LOG.warning("Clip %d: 'PTD last hop' has no windowed form; refining the whole clip in one pass",
                         clip_index + 1)

    def _ptd_last_hop(self):
        return self.ptd_controller.hops - 1

    def configure_dual_sampling(self, enabled=False, split_step=2, early_stop=False):
        """HyperFlow dual sampling: pass 2 is the tail of the trained grid from index `split_step`, with no pass-2
        LoRA; `early_stop` also cuts pass 1 to grid[:split_step + 1]. Off (the default) changes nothing."""
        self.dual_active = False
        self.pass1_early_stop = False
        self.pass1_steps = 0
        self.dual_split_step = int(split_step)
        if not enabled:
            return
        if self.turbo_mode or getattr(self, "ptd_mode", False):
            _LOG.warning("Dual sampling needs HyperFlow 8-step; Turbo LoRA / PTD mode runs the normal two passes")
            return
        if self.pass2_lora:
            _LOG.warning("Dual sampling: pass 2 runs the HyperFlow grid tail without a refine LoRA; ignoring pass-2 LoRA %s",
                         self.pass2_lora)
            self.pass2_lora = None
            self.pass2_model = None
            self.pass2_sla_model = None
        self.dual_active = True
        self.pass1_early_stop = bool(early_stop)

    def configure_pass1_steps(self, steps=0):
        """Opt-in: stop HyperFlow's pass 1 after `steps` of its 8 steps and hand the model's clean x0 estimate to the
        normal upscale + pass-2 refine. 0 (the default) changes nothing. Call after configure_dual_sampling."""
        self.pass1_steps = 0
        steps = int(steps or 0)
        if steps <= 0:
            return
        if self.turbo_mode or getattr(self, "ptd_mode", False):
            _LOG.warning("pass1_steps=%d ignored: it needs HyperFlow 8-step, Turbo LoRA / PTD mode runs the full pass 1", steps)
            return
        if self.dual_active:
            _LOG.warning("pass1_steps=%d ignored: dual sampling sets the pass-1 length (dual_split_step)", steps)
            return
        self.pass1_steps = steps

    def configure_audio_refine(self, steps=0, denoise=0.5, cache=True):
        """Audio-only refine after pass 2 (ComfyUI-H3-AudioRefine). steps 0 = off."""
        self.audio_refine_steps = int(steps or 0)
        self.audio_refine_denoise = float(denoise)
        self.audio_refine_cache = bool(cache)
        if self.audio_refine_steps > 0 and nodes.NODE_CLASS_MAPPINGS.get("H3AudioRefineSampler") is None:
            raise RuntimeError("audio_refine_steps > 0 needs the ComfyUI-H3-AudioRefine node pack "
                               "(H3AudioRefineSampler) in custom_nodes: "
                               "git clone https://github.com/Adudeguyman/ComfyUI-H3-AudioRefine.git")

    def _refine_audio(self, latent, positive, seed, clip_index):
        """Audio-only refine of a finished clip: video frozen (bit-identical), audio re-noised to
        audio_refine_denoise and sampled for audio_refine_steps on the undistilled model.

        The model is SigmaShift only (self.shifted_model): no turbo LoRA or HyperFlow
        patch, so HyperFlow is never applied twice and the refine uses the full weights.
        Attention is the chosen dense backend; block-sparse attention is not applied."""
        steps = self.audio_refine_steps
        model = self._finish_model(self.shifted_model, sparse=False)
        tokens = self._packed_tokens(latent["samples"])
        model = self._with_ff_chunking(model, tokens)
        if self.audio_refine_cache:
            cache_cls = nodes.NODE_CLASS_MAPPINGS.get("H3FrozenVideoCache")
            if cache_cls is None:
                raise RuntimeError("audio_refine_cache needs the H3FrozenVideoCache node from ComfyUI-H3-AudioRefine")
            model = cache_cls().patch(
                model, enabled=True, cache_contents="hidden", backend="auto", precision="int4",
                refresh_interval=0, verbose=False, allow_disk=False, free_after_pass=True)[0]
        _LOG.info("Clip %d: audio-only refine, %d steps at denoise %.2f, %d tokens, frozen cache %s",
                  clip_index + 1, steps, self.audio_refine_denoise, tokens,
                  "on (hidden/int4)" if self.audio_refine_cache else "off")
        self._reclaim_vram(model, tokens=tokens)
        sampler = nodes.NODE_CLASS_MAPPINGS["H3AudioRefineSampler"]()
        refined = sampler.refine(model, positive, positive, latent, seed, steps, 1.0,
                                 "euler", "simple", self.audio_refine_denoise, 0.0)[0]
        out = dict(latent)
        out["samples"] = refined["samples"].to(comfy.model_management.intermediate_device())
        return out

    def configure_semantic_bridge(self, adapter="none", alpha=0.12, magnitude_match="per_token"):
        """BUNNY H3 Conditioning Bridge on the conditioning of both passes ("none" = off).
        Applied right after the Ref2VA encoder, before the motion context is attached."""
        self.semantic_bridge = adapter if adapter and str(adapter).lower() != "none" else None
        self.semantic_bridge_alpha = float(alpha)
        self.semantic_bridge_match = str(magnitude_match or "per_token")

    def _pass1_from_pass2(self, out_p2, ref_images, w1, h1, frame_count, clip_index):
        """Pass-1 conditioning + empty latent from the pass-2 encode: the text embedding is reused,
        only the image-reference latents are re-encoded at the pass-1 size, with the sizing
        MiniMaxH3ReferenceToVideo uses for ref_image_size="match". Video / audio reference blocks
        do not depend on the generation size in core, so they are carried over unchanged.
        Returns None (caller falls back to the two-encode path) when that cannot be reproduced."""
        cond = _safe_get_output(out_p2, 0, "positive")
        try:
            blocks = list(cond[0][1].get("minimax_refs") or [])
        except (TypeError, IndexError, AttributeError, KeyError):
            _LOG.info("Clip %d: single text encode unavailable (no conditioning structure), using two encodes", clip_index + 1)
            return None
        images = [img for img in ref_images.values() if img is not None]
        image_slots = [i for i, b in enumerate(blocks) if b.get("kind") == "image"]
        if len(image_slots) != len(images) or (images and self.vae is None):
            _LOG.info("Clip %d: single text encode unavailable (%d image blocks for %d reference images), using two encodes",
                      clip_index + 1, len(image_slots), len(images))
            return None
        for slot, img in zip(image_slots, images):
            h, w = img.shape[1], img.shape[2]
            scale = min(1.0, math.sqrt((w1 * h1) / (w * h)))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            blocks[slot] = {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                            "latent": self.vae.encode(resized)}
        cond_p1 = [[t.clone() if torch.is_tensor(t) else t, dict(d)] for t, d in cond]
        if blocks:
            cond_p1 = node_helpers.conditioning_set_values(cond_p1, {"minimax_refs": blocks})
        latent, _ = _empty_av_latent(w1, h1, frame_count)
        _LOG.info("Clip %d: single text encode: 1 encoder pass (pass-1 refs re-encoded at %dx%d)",
                  clip_index + 1, w1, h1)
        return cond_p1, latent

    def _bridge(self, positive, label=""):
        if getattr(self, "semantic_bridge", None) is None:
            return positive
        return apply_semantic_bridge(positive, self.semantic_bridge, self.semantic_bridge_alpha,
                                     self.semantic_bridge_match, nodes.NODE_CLASS_MAPPINGS, label, _LOG)

    def _pass2_lora_active(self):
        return getattr(self, "pass2_lora", None) is not None

    def initialize_pass2_model(self):
        """Model used for the refine pass: the pass-1 model unless a pass-2 LoRA is set."""
        prepared, _ = self.initialize_model()
        p2_sparsity = self._pass2_sla_override()
        if getattr(self, "ptd_mode", False) and self.pass2_engine == ptd_mode.PASS2_ENGINE_PTD:
            if p2_sparsity is None:
                return prepared
            if self.pass2_sla_model is None:
                self.pass2_sla_model = self._finish_model(self.ptd_base, sparsity=p2_sparsity)
            return self.pass2_sla_model
        if getattr(self, "ptd_mode", False) and not self._pass2_lora_active():
            raise ValueError("PTD (adapter) with pass2_engine 'LoRA refine' needs a pass2_lora (e.g. the DMAD LoRA): "
                             "the PTD model cannot sample outside its trained hops")
        if not self._pass2_lora_active():
            if p2_sparsity is None:
                return prepared
            if self.pass2_sla_model is None:
                self.pass2_sla_model = self._finish_model(self.accel_model, sparsity=p2_sparsity)
            return self.pass2_sla_model
        if getattr(self, "pass2_model", None) is not None:
            return self.pass2_model
        lora_cls = nodes.NODE_CLASS_MAPPINGS.get("LoraLoaderModelOnly")
        if lora_cls is None:
            _LOG.warning("LoraLoaderModelOnly unavailable; pass 2 uses the pass-1 model")
            self.pass2_model = prepared
            return prepared
        if self.pass2_lora_mode == "replace":
            base = self.shifted_model          # SigmaShift only: no turbo LoRA / HyperFlow
        else:
            base = self.accel_model            # turbo LoRA / HyperFlow already applied
        res = lora_cls().load_lora_model_only(base, self.pass2_lora, self.pass2_lora_strength)
        model = _safe_get_output(res, 0, "model")
        self.pass2_model = self._finish_model(model, sparsity=p2_sparsity)
        _LOG.info("Pass 2 LoRA: %s @ %.2f (%s engine LoRA), %s steps schedule",
                  self.pass2_lora, self.pass2_lora_strength,
                  "replacing" if self.pass2_lora_mode == "replace" else "stacked on",
                  self.pass2_steps or self.turbo_steps)
        return self.pass2_model

    # Sparse attention: dense for the first part of the schedule (structure is
    # decided there), sparse after. 0.2 is core's default and matches an
    # 8-step draft (first ~2 steps dense); pass 2's tail sits past it anyway.
    SPARSE_START_PERCENT = 0.2
    SPARSE_MIN_TOKENS = 12288
    SPARSE_EXTRA_TOKENS = 256

    def _sparse_settings(self):
        return resolve_sparse_settings(
            self.hyperflow_mode, str(getattr(self, "sparse_method", "sla") or "sla"),
            self.sparse_start_percent, self.sparse_dense_blocks, self.sparse_sink,
            default_start=self.SPARSE_START_PERCENT)

    # ------------------------------------------------------------------
    # Turbo LoRA mode + SLA
    # ------------------------------------------------------------------
    def _basic_sigmas(self, model, steps, denoise):
        """Sigmas from core's BasicScheduler for the engine's scheduler_name.
        `steps` is the number of steps that will actually run; with denoise < 1
        they are the tail of a longer schedule (core's hires-fix semantics)."""
        from comfy_extras.nodes_custom_sampler import BasicScheduler
        steps = max(1, int(steps))
        res = BasicScheduler.execute(model, self.scheduler_name, steps, float(denoise))
        sigmas = _safe_get_output(res, 0, "sigmas")
        if sigmas is None:
            raise RuntimeError("BasicScheduler returned no sigmas")
        return sigmas

    def _prepare_turbo_model(self, model):
        """Load the turbo LoRA (any step-distilled LoRA) and build the pass-1 schedule."""
        steps = int(self.turbo_steps)
        if self.turbo_lora is None:
            _LOG.warning("Turbo LoRA mode selected but no LoRA chosen; sampling the base model "
                         "with %d steps of %s/%s", steps, self.sampler_name, self.scheduler_name)
            return model, self._basic_sigmas(model, steps, 1.0)

        lora_cls = nodes.NODE_CLASS_MAPPINGS.get("LoraLoaderModelOnly")
        res = lora_cls().load_lora_model_only(model, self.turbo_lora, self.turbo_lora_strength)
        model = _safe_get_output(res, 0, "model")
        _LOG.info("Turbo LoRA: %s @ %.2f, %d steps, %s / %s", self.turbo_lora,
                  self.turbo_lora_strength, steps, self.sampler_name, self.scheduler_name)

        return model, self._basic_sigmas(model, steps, 1.0)

    def _prepare_ptd_model(self, model):
        """Patch the SigmaShift model with the PTD adapter bank + PDD heads (ComfyUI-MiniMaxH3-PTD) and return it with
        the checkpoint's own hop grid as the pass-1 schedule. The per-hop factors stay in host RAM; the pack uploads
        only the active layer's expert per forward, so no adapter weights sit on the GPU between calls."""
        if self.ptd_adapter is None:
            raise ValueError(f"accel_mode '{ptd_mode.PTD_ACCEL_MODE}' needs a PTD adapter (models/ptd): pick a PTD adapter in the panel's pass-1 section (ptd_adapter)")
        import folder_paths
        pack = ptd_mode.ptd_pack()
        path = folder_paths.get_full_path_or_raise(ptd_mode.PTD_FOLDER, self.ptd_adapter)
        checkpoint = pack.loader.load_ptd_checkpoint(path)
        patched = pack.patcher.patch_pdd_ptd_model(model, checkpoint, self.shift_video, self.shift_audio)
        self.ptd_controller = pack.patcher.get_controller(patched)
        self.ptd_pack = pack
        self.ptd_base = patched
        sigmas = self.ptd_controller.sigmas.float().cpu()
        _LOG.info("PTD adapter %s: %d hops (rank %d), grid %s, pass 1 = %d model evaluations; SigmaShift (%.1f, %.1f)",
                  self.ptd_adapter, checkpoint.num_hops, checkpoint.rank, [round(float(x), 4) for x in sigmas],
                  checkpoint.num_hops, self.shift_video, self.shift_audio)
        return patched, sigmas

    def _prepare_hyperflow_model(self, model):
        """HyperFlow 8-step (ComfyUI-HyperFlow-H3): backbone LoRA + two-time (t, r)
        conditioning. Its MODEL and its trained 9-point SIGMAS are used as-is; the
        schedule is never a BasicScheduler one, and turbo_steps does not apply."""
        apply_cls = nodes.NODE_CLASS_MAPPINGS.get("ApplyHyperFlowH3")
        if apply_cls is None:
            raise RuntimeError("HyperFlow 8-step needs the ComfyUI-HyperFlow-H3 node pack "
                               "(ApplyHyperFlowH3) in custom_nodes.")
        self.turbo_steps = "8"
        dm = model.get_model_object("diffusion_model")
        pruned_base = hasattr(dm, "adaln_t_table") or not hasattr(dm, "time_embedder")
        self._match_hyperflow_build(pruned_base)
        pruned_file = "_pruned" in self.hyperflow_file.lower()
        if pruned_base and not pruned_file:
            raise ValueError(f"{self.hyperflow_file} is the FULL-base build but the loaded base is a pruned/curve "
                             "MiniMax-H3 with no time_embedder. Select the pruned-base build "
                             "(custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors).")
        if not pruned_base and pruned_file:
            raise ValueError(f"{self.hyperflow_file} is the PRUNED-base build (backbone LoRA only) but the loaded "
                             "base is a full MiniMax-H3. Select custom_node_hyperflow_8step_v1.0_comfyui.safetensors "
                             "(the file without '_pruned') instead.")
        if pruned_base and not self.hyperflow_curve_refit:
            _LOG.warning("HyperFlow: pruned base without curve refit; running LoRA-only (single-time, off-recipe)")
        self.hyperflow_pruned_refit = pruned_base and self.hyperflow_curve_refit
        _LOG.info("HyperFlow: %s on a %s base, curve refit %s, strength %.2f, lora_mode %s",
                  self.hyperflow_file, "pruned" if pruned_base else "full",
                  "requested" if self.hyperflow_curve_refit else "off", self.hyperflow_strength,
                  self.hyperflow_lora_mode)
        res = apply_cls().apply(
            model,
            hyperflow_file=self.hyperflow_file,
            strength=self.hyperflow_strength,
            lora_mode=self.hyperflow_lora_mode,
            variant="auto",
            download_if_missing=False,
            verbose=False,
            experimental_curve_refit=self.hyperflow_curve_refit and pruned_base,
        )
        hf_model = _safe_get_output(res, 0, "model")
        sigmas = _safe_get_output(res, 1, "sigmas")
        if hf_model is None or sigmas is None:
            raise RuntimeError("ApplyHyperFlowH3 returned no model / sigmas")
        _LOG.info("HyperFlow: trained grid %s, sampler euler", [round(float(x), 4) for x in sigmas])
        if self.sla_enabled:
            cfg = self._sparse_settings()
            src = cfg["sources"]
            if self.sparse_method == "sol-attn":
                sel = "tau %.2f (HyperFlow recipe: 1.0)" % self.sparse_tau
            else:
                sel = "keep %.1f%%" % max(0.5, min(95.0, (1.0 - self.sla_sparsity) * 100.0))
            _LOG.info("HyperFlow: extender sparse attention is ON (%s, %s, start %.2f [%s], dense_blocks '%s' [%s], "
                      "sink %s [%s]); HyperFlow's validated sol-attn recipe is start 0.16, dense_blocks 0,1, tau 1.0, "
                      "sink off",
                      self.sparse_method, sel, cfg["start"], src["start"], cfg["dense_blocks"], src["dense_blocks"],
                      cfg["sink"], src["sink"])
        else:
            _LOG.info("HyperFlow: sparse attention is OFF (dense recipe)")
        return hf_model, sigmas

    def _pass2_sla_override(self):
        """Pass-2-only sparsity, or None when pass 2 keeps sla_sparsity (the default, and for sol-attn)."""
        if not self.sla_enabled or self.pass2_sla_sparsity <= 0.0:
            return None
        if self.sparse_method == "sol-attn":
            _LOG.info("pass2_sla_sparsity %.2f ignored: sparse_method sol-attn selects by tau, not keep percent",
                      self.pass2_sla_sparsity)
            return None
        return self.pass2_sla_sparsity

    def _apply_sla(self, model, sparsity=None):
        """Block-sparse attention for both passes. On long sequences it roughly
        halves the per-step time (measured 39 -> 22 s/step at 1344x768 / 362
        frames at 90% sparsity).

        Uses core's BlockSparseAttention (ComfyUI >= 0.35): it chains the
        dense backend selected above as its fall-through and re-installs itself
        every step, keeps the H3 text/audio/reference rows exact, and offers
        three selection methods (sla / sol-attn / vsa)."""
        if not self.sla_enabled:
            return model
        native = nodes.NODE_CLASS_MAPPINGS.get("BlockSparseAttention")
        if native is None:
            _LOG.warning("Sparse attention requested but core BlockSparseAttention is unavailable "
                         "(needs ComfyUI >= 0.35); continuing dense")
            return model
        method = str(getattr(self, "sparse_method", "sla") or "sla")
        if sparsity is None:
            sparsity = self.sla_sparsity
        keep_percent = max(0.5, min(95.0, (1.0 - sparsity) * 100.0))
        selection = {"selection": method}
        if method == "sol-attn":
            selection["tau"] = float(getattr(self, "sparse_tau", 1.3))
        else:
            selection["keep_percent"] = keep_percent
        cfg = self._sparse_settings()
        try:
            res = native.execute(
                model, selection=selection,
                start_percent=cfg["start"], end_percent=1.0,
                dense_blocks=cfg["dense_blocks"], min_tokens=self.SPARSE_MIN_TOKENS,
                extra_tokens=0 if method == "vsa" else self.SPARSE_EXTRA_TOKENS,
                sink_conditioning=cfg["sink"], verbose=False,
            )
            out = _safe_get_output(res, 0, "model")
            if out is not None:
                detail = (f"tau={selection['tau']}" if method == "sol-attn"
                          else f"keep={keep_percent:.1f}% (sparsity {sparsity:.2f})")
                _LOG.info("Sparse attention (core BlockSparseAttention, %s): %s, dense before %.0f%%, "
                          "dense_blocks '%s', sink %s, min_tokens %d, dense fall-through = %s", method, detail,
                          cfg["start"] * 100, cfg["dense_blocks"], cfg["sink"], self.SPARSE_MIN_TOKENS,
                          self.attention_backend)
                return out
        except Exception as exc:
            _LOG.warning("Core BlockSparseAttention could not be applied (%s); continuing dense", exc)
        return model

    def _pass2_sigmas(self, model, pass2_denoise):
        """Quality-tail schedule for pass 2: the tail of HyperFlow's own grid, or the last
        round(steps x denoise) steps of the turbo schedule. With a pass-2 LoRA
        the schedule length can differ from pass 1 (pass2_steps), and in
        "replace" mode the pass-1 grid no longer applies."""
        if self.dual_active:
            tail = dual_pass2_sigmas(self.pass1_sigmas, self.dual_split_step)
            _LOG.info("HyperFlow dual sampling: pass 2 runs grid[%d:] (%d steps): %s",
                      dual_split_index(self.pass1_sigmas, self.dual_split_step), len(tail) - 1,
                      [round(float(x), 4) for x in tail])
            return tail
        if self.hyperflow_mode and not self._pass2_lora_active():
            tail = hyperflow_tail_sigmas(self.pass1_sigmas, pass2_denoise)
            _LOG.info("HyperFlow: pass 2 uses the tail of HyperFlow's own grid (%d steps): %s",
                      len(tail) - 1, [round(float(x), 4) for x in tail])
            if self.hyperflow_pruned_refit:
                _LOG.info("HyperFlow: the curve fit is bound to the full 9-point grid, so this pass-2 tail runs "
                          "backbone-only (LoRA, single-time); the node logs 'curve refit disabled for an "
                          "unmatched sampling recipe' for it")
            return tail
        if self._pass2_lora_active() and (self.pass2_steps > 0 or self.pass2_lora_mode == "replace" or self.turbo_mode):
            steps = int(self.pass2_steps or self.turbo_steps)
            tail = max(1, int(round(steps * float(pass2_denoise))))
            return self._basic_sigmas(model, tail, float(pass2_denoise))
        if self.turbo_mode:
            steps = int(self.turbo_steps)
            tail = max(1, int(round(steps * float(pass2_denoise))))
            return self._basic_sigmas(model, tail, float(pass2_denoise))

    # Make room for activations before each sampling pass.
    # With dynamic VRAM enabled, core's free_memory() does not evict one dynamic
    # model for another ("0 models unloaded"), so the 32B text encoder that just
    # encoded the prompt stays resident while the UNet needs tens of GB of
    # activations for a long HD clip -> torch.OutOfMemoryError mid-pass
    # (reproduced on clip 3 of a 3 x 15 s @ 1344x768 project on a 32 GB card).
    # Estimate the activation demand from the packed token count, ask core to
    # evict everything except the sampling model, then return cached blocks to
    # the driver so the allocator can serve one large contiguous request.
    ACTIVATION_BYTES_PER_TOKEN = 64 * 1024      # fc1 output measured ~62 KB/token
    ACTIVATION_LIVE_TENSORS = 3
    # Feed-forward chunking: one chunk per this many packed tokens, so the fc1
    # peak stays roughly constant (~1.6 GB) whatever the clip length/resolution.
    FF_TOKENS_PER_CHUNK = 27000
    FF_CHUNK_THRESHOLD = 16384

    @staticmethod
    def _packed_tokens(samples):
        video = samples
        try:
            video, _ = _streams_from_latent({"samples": samples}, "samples")
        except Exception:
            pass
        shape = tuple(int(s) for s in getattr(video, "shape", ()))
        if len(shape) == 5:
            _, _, t, h, w = shape
        elif len(shape) == 4:
            _, t, h, w = shape
        else:
            t, h, w = 1, 64, 64
        return t * ((h + 1) // 2) * ((w + 1) // 2)

    def _with_ff_chunking(self, model, tokens):
        """Return `model` with KJNodes' MiniMaxChunkFeedForward applied, sized to
        the packed token count of this pass (KJNodes >= 1.5.1). On long/HD clips
        the OOM happens inside block.mlp: the fc1 output alone is
        tokens x 2*ffn_dim (~6.4 GB at 1344x768 / 362 frames, ~13 GB at 1080p).
        Splitting the token axis divides that peak by the chunk count at
        negligible cost; the object patches live on a clone so the prepared
        model is untouched. H3_MASTER_FF_CHUNKS forces a count (1 disables)."""
        chunk_cls = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxChunkFeedForward")
        if chunk_cls is None or tokens <= self.FF_CHUNK_THRESHOLD:
            return model
        forced = os.environ.get("H3_MASTER_FF_CHUNKS")
        if forced:
            chunks = int(forced)
        else:
            chunks = max(2, min(64, math.ceil(tokens / self.FF_TOKENS_PER_CHUNK)))
        if chunks <= 1:
            return model
        try:
            res = chunk_cls.execute(model, chunks=chunks, seq_threshold=self.FF_CHUNK_THRESHOLD)
            chunked = _safe_get_output(res, 0, "model")
            if chunked is not None:
                _LOG.info("Master engine: feed-forward chunking -- %d tokens -> %d chunks", tokens, chunks)
                return chunked
        except Exception as exc:
            _LOG.warning("Master engine: could not enable feed-forward chunking (%s)", exc)
        return model

    # Passes at or below this many packed tokens (draft passes) may run beside
    # a background decode instead of evicting it.
    BACKGROUND_TOLERANT_TOKENS = 30000

    @staticmethod
    def _release_node_scoped_vram():
        # ComfyUI frees these when a node finishes, but this engine runs a whole job
        # inside one node, so after every eviction they kept ~4 GB that later passes
        # and clips then started without (outside ComfyUI's model manager, so
        # free_memory cannot reach it):
        #  - bypass-LoRA hooks (HyperFlow's backbone LoRA) copy their adapter weights
        #    to the GPU on inject and never move them back on eject (~2 GB);
        #  - the compiler's CUDA graphs and prefetch state of the last sampled model (~2 GB).
        import gc
        try:
            from comfy.weight_adapter.bypass import BypassForwardHook
            cpu = torch.device("cpu")
            for obj in gc.get_objects():
                if isinstance(obj, BypassForwardHook) and obj.original_forward is None:
                    obj._move_adapter_weights_to_device(cpu)
        except ImportError:
            pass
        try:
            import comfy.model_prefetch
        except ImportError:
            return
        comfy.model_prefetch.cleanup_prefetch_queues()

    def _reclaim_vram(self, model_patcher, samples=None, tokens=None):
        import gc
        mm = comfy.model_management
        try:
            device = mm.get_torch_device()
            if tokens is None:
                tokens = self._packed_tokens(samples)
            required = int(tokens * self.ACTIVATION_BYTES_PER_TOKEN * self.ACTIVATION_LIVE_TENSORS)
            required = max(required, 6 * 1024 ** 3)
            gc.collect()
            if self.background_busy():
                if tokens <= self.BACKGROUND_TOLERANT_TOKENS:
                    mm.soft_empty_cache(force=True)
                    _LOG.info("Master engine: background decode in flight -- small pass (%d tokens), "
                              "sampling alongside it without evicting", tokens)
                    return
                _LOG.info("Master engine: waiting for the background decode before a %d-token pass", tokens)
                self.wait_background()
            before = mm.get_free_memory(device)
            keep = []
            if model_patcher is not None:
                keep = [lm for lm in mm.current_loaded_models if getattr(lm, "model", None) is model_patcher]
            # Evict EVERYTHING but the sampling model, not just enough for the
            # activation estimate. free_memory() stops as soon as the request is
            # met, and when the text encoder / VAEs were left resident the UNet
            # itself ended up partly paged out: identical drafts then ran at
            # 6-18 s/step instead of 2.5 s/step because every step re-streamed
            # weights. The VAEs reload in a second or two when the decode needs
            # them, which is far cheaper than a paging pass.
            mm.free_memory(1e30, device, keep_loaded=keep)
            self._release_node_scoped_vram()
            mm.soft_empty_cache(force=True)
            after = mm.get_free_memory(device)
            _LOG.info("Master engine: VRAM reclaim before sampling -- %d tokens, need ~%.1f GB, evicted all but the "
                      "sampling model: free %.1f -> %.1f GB",
                      tokens, required / 1024 ** 3, before / 1024 ** 3, after / 1024 ** 3)
        except Exception as exc:
            _LOG.warning("Master engine: VRAM reclaim skipped (%s)", exc)

    # ------------------------------------------------------------------
    # Chunked pass 2 (temporal windows via MMH3SplitUpscale)
    # ------------------------------------------------------------------
    def _pass2_chunking_active(self, frame_count, width=None, height=None):
        if self.pass2_chunk_frames <= 0:
            return False
        if int(frame_count) <= int(self.pass2_chunk_frames):
            return False
        if self.pass2_chunk_when == "above 720p" and width and height:
            short_side = min(int(width), int(height))
            if short_side <= PASS2_CHUNK_MIN_SHORT_SIDE:
                _LOG.info("Pass 2: whole clip (pass2_chunk_when=above 720p, short side %d px <= %d)",
                          short_side, PASS2_CHUNK_MIN_SHORT_SIDE)
                return False
        if nodes.NODE_CLASS_MAPPINGS.get("MMH3SplitUpscale") is None or \
                nodes.NODE_CLASS_MAPPINGS.get("MMH3TemporalSplitParamsV10") is None:
            return False
        return int(frame_count) > int(self.pass2_chunk_frames)

    def _sample_pass2_chunked(self, model, positive, latent, sigmas, seed):
        """Refine pass 2 in overlapping temporal windows instead of one shot.

        A 15 s clip at 1344x768 is ~108k packed tokens, at 1920x1088 ~218k;
        attention cost grows with the square of that and the activations no
        longer fit beside the UNet, so the step degenerates into weight
        paging (measured 239-367 s/step at 1080p vs ~21-36 s/step for 5 s
        windows). MMH3SplitUpscale (Comfyui_Minimax_h3_latent_Upscaler) does
        the H3-grid-aware windowing: it re-anchors the conditioning per
        window, carries motion/identity anchors across windows, matches
        colour, and blends the overlaps. At denoise 0.25 the windows only
        polish an already coherent draft, so seams stay invisible.
        """
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
        split_cls = nodes.NODE_CLASS_MAPPINGS["MMH3SplitUpscale"]
        param_cls = nodes.NODE_CLASS_MAPPINGS["MMH3TemporalSplitParamsV10"]

        # identity_anchor_frames must stay 0 here. The split node's identity
        # anchors are latent frames sampled from the *unrefined upscaled draft*
        # and injected as keyframe conditioning; H3 preserves keyframes, so the
        # refinement reproduces the latent upscaler's block pattern as visible
        # squares (worse with low-res drafts and sparse attention). Within one
        # clip every window already shares the same draft, so identity is
        # consistent without them. Motion anchors and the window-to-window
        # anchor come from refined output and are kept.
        tparam = _safe_get_output(param_cls.execute(
            chunk_frames=int(self.pass2_chunk_frames),
            temporal_overlap_frames=int(self.pass2_chunk_overlap),
            anchor_strength=0.999,
            motion_anchor_frames="22",
            identity_anchor_frames=0,
        ), 0, "temporal_split_param")

        video = latent["samples"].tensors[0]
        _, _, t_lat, h_lat, w_lat = (int(x) for x in video.shape)
        window_lat = min(t_lat, max(1, (int(self.pass2_chunk_frames) - 5) // 17 * 5 + 2))
        window_tokens = window_lat * ((h_lat + 1) // 2) * ((w_lat + 1) // 2)
        _LOG.info("Master engine: pass 2 in temporal windows of %d frames (overlap %d) -- %d latent frames "
                  "-> windows of %d (~%d tokens each)", self.pass2_chunk_frames, self.pass2_chunk_overlap,
                  t_lat, window_lat, window_tokens)

        model = self._with_ff_chunking(model, window_tokens)
        self._reclaim_vram(model, tokens=window_tokens)

        sampler_name = pass_sampler_name(getattr(self, "turbo_mode", False) or getattr(self, "ptd_mode", False), self.sampler_name)
        res = split_cls.execute(
            latent=latent,
            conditioning=positive,
            model=model,
            noise=Noise_RandomNoise(seed),
            sampler=comfy.samplers.sampler_object(sampler_name),
            sigmas=sigmas,
            negative=None,
            cfg=1.0,
            temporal_split_param=tparam,
            spatial_split_param=None,
            seam_polish="off",
            color_match=True,
        )
        out = _safe_get_output(res, 0, "latent")
        if not isinstance(out, dict) or "samples" not in out:
            raise RuntimeError("MMH3SplitUpscale returned no latent")
        result = dict(latent)
        result.pop("noise_mask", None)
        result["samples"] = out["samples"].to(comfy.model_management.intermediate_device())
        return result

    @staticmethod
    def _x0_dumper(model_patcher, sigmas):
        """Per-step x0 dump, only while `<ComfyUI base>/user/master_x0_debug` exists; None otherwise."""
        try:
            import folder_paths
            if not x0_debug.debug_enabled(folder_paths.base_path):
                return None
            out_dir = x0_debug.dump_dir(folder_paths.get_output_directory(), x0_debug._current_prompt_id())
            return x0_debug.X0Dump(
                out_dir, sigmas.detach().float().cpu().tolist(),
                lambda x: model_patcher.model.process_latent_out(x.to(torch.float32)))
        except Exception:
            _LOG.exception("x0 debug dump: setup failed, rendering without it")
            return None

    def _sample_euler(self, model, positive, latent_image, sigmas, seed, return_denoised=False, sampler=None):
        """Standard Euler sampling using ComfyUI core sampler internals directly.

        NOTE: Do NOT place any bare 'import comfy' statements inside this function.
        comfy.* is imported at module level; a local assignment to 'comfy' would cause
        Python to treat the name as unbound-local, raising UnboundLocalError.

        We directly use comfy.samplers.CFGGuider instead of going through the
        BasicGuider node, because ComfyUI v3 node calls return NodeOutput wrappers
        that don't expose .model_patcher, making them unusable for direct sampling.
        """
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
        try:
            import latent_preview as _lp
            _have_lp = True
        except ImportError:
            _have_lp = False

        # Noise_RandomNoise is a plain class: Noise_RandomNoise(seed)
        noise_obj = Noise_RandomNoise(seed)

        # Size the feed-forward chunking to this pass's packed token count.
        model = self._with_ff_chunking(model, self._packed_tokens(latent_image["samples"]))

        # Build guider directly — bypass ComfyUI v3 node wrapper (NodeOutput issue)
        # MiniMax H3 is a pure flow model: no negative conditioning needed.
        # CFGGuider.set_conds(positive, negative) — pass positive for both.
        # With set_cfg(1.0): output = neg + 1.0*(pos-neg) = pos, so negative is never used.
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, positive)
        guider.set_cfg(1.0)

        # Build sampler object: HyperFlow was distilled for plain Euler; turbo mode
        # uses the user's sampler (res_multistep by default).
        # PTD passes bring their own sampler; a PTD pass 1 followed by a LoRA refine uses the turbo sampler like Turbo mode.
        euler_sampler = sampler if sampler is not None else comfy.samplers.sampler_object(
            pass_sampler_name(self.turbo_mode or self.ptd_mode, self.sampler_name))

        # Fix empty latent channels if needed
        latent = dict(latent_image)
        latent["samples"] = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent["samples"],
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )

        noise_mask = latent.get("noise_mask")
        x0_output = {}

        if _have_lp:
            callback = _lp.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
        else:
            callback = None
        if return_denoised:
            callback = capture_x0_callback(callback, x0_output)

        dumper = self._x0_dumper(guider.model_patcher, sigmas)
        if dumper is not None:
            callback = dumper.wrap(callback)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        self._reclaim_vram(guider.model_patcher, latent["samples"])

        samples = guider.sample(
            noise_obj.generate_noise(latent),
            latent["samples"],
            euler_sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        if dumper is not None:
            dumper.save_final(samples)
            dumper.decode_frames(self.vae)
        if return_denoised:
            # The sigmas stop above 0, so `samples` still carries noise; the last step's x0 prediction is the clean estimate.
            samples = clean_latent_from_x0(guider.model_patcher, x0_output["x0"])
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        return out

    def _run_3d_upscaler(self, video_latent_4d_or_5d, target_w, target_h):
        """Upscale video latent using MinimaxH3LatentUpscaler3D."""
        upscaler_cls = nodes.NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
        if not upscaler_cls:
            raise RuntimeError("MinimaxH3LatentUpscaler3D node is not installed in ComfyUI.")

        # Ensure 5D tensor [B, C, T, H, W]
        if video_latent_4d_or_5d.ndim == 4:
            v_tensor = video_latent_4d_or_5d.unsqueeze(0)
        else:
            v_tensor = video_latent_4d_or_5d

        # Current Comfyui_Minimax_h3_latent_Upscaler builds expose execute() as
        # (latent, model_name, mode, align, enable_temporal_chunking, force_unload,
        # device, precision); the (keep_proportion, offload_after_upscale) kwargs
        # this engine was written against no longer exist and raise TypeError.
        # Resolve the mode key from the node's own enum when present and pass only
        # the kwargs the installed signature accepts, so either version works.
        mode_key = "target dimensions"
        try:
            _enum = getattr(sys.modules[upscaler_cls.__module__], "UpscaleMode", None)
            if _enum is not None:
                mode_key = getattr(_enum, "TARGET_DIMENSIONS", mode_key)
        except Exception:
            pass
        mode_config = {
            "mode": mode_key,
            "width": int(target_w),
            "height": int(target_h),
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        candidate_kwargs = {
            "latent": {"samples": v_tensor},
            "model_name": self.upscaler_model,
            "mode": mode_config,
            "align": 32,
            "keep_proportion": False,
            "device": device,
            "precision": _upscaler_precision(self.upscaler_model),
            "offload_after_upscale": self.smart_offload,
            "enable_temporal_chunking": True,
            "force_unload": bool(self.smart_offload),
        }
        try:
            accepted = set(inspect.signature(upscaler_cls.execute).parameters)
            call_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
        except (TypeError, ValueError):
            call_kwargs = candidate_kwargs
        res = upscaler_cls.execute(**call_kwargs)

        # Result is io.NodeOutput or tuple/dict containing {"samples": out}
        raw_out = _safe_get_output(res, 0, "latent")
        if isinstance(raw_out, dict) and "samples" in raw_out:
            upscaled = raw_out["samples"]
        elif hasattr(raw_out, "get") and raw_out.get("samples") is not None:
            upscaled = raw_out.get("samples")
        else:
            upscaled = raw_out

        if self.smart_offload and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return upscaled

    def render_clip(
        self,
        clip_index: int,
        prompt: str,
        duration_sec: float,
        seed: int,
        pass1_res: tuple[int, int],
        pass2_res: tuple[int, int],
        pass2_denoise: float = 0.25,
        ref_images: dict = None,
        previous_latent: dict = None,
        previous_draft: dict = None,
        context_length: str = "22",
        audio_context_length: int = 0,
        last_frame_guide: torch.Tensor = None,
        progress_cb=None,
        ref_videos: dict = None,
        ref_video_audios: dict = None,
        ref_audios: dict = None,
        pass2_enabled: bool = True,
        cached_pass1: dict = None,
        cached_trim: int = 0,
        p1_text_size: tuple = None,
    ):
        """Execute the complete 2-Stage 2-Pass Pipeline for a single clip.

        ``ref_videos`` (``ref_video_N`` -> 24 fps IMAGE batch), ``ref_video_audios``
        (``ref_video_audio_N`` -> AUDIO, the soundtrack of the same-numbered video)
        and ``ref_audios`` (``ref_audio_N`` -> AUDIO) go straight to the core
        reference node, which labels them <Video k> / <Audio j> after the pictures.

        ``p1_text_size`` is the encode size for the single text/reference pass when pass 2 is off.
        ``pass2_enabled=False`` stops after pass 1: the pass-1 latent is returned as the clip
        (``pass2_res`` must equal ``pass1_res``). ``cached_pass1`` is a LATENT-like object holding
        an earlier pass-1 result for this exact clip; pass-1 conditioning and sampling are skipped
        and it is used as if just sampled (``cached_trim`` is its motion-context trim, needed only
        when pass 2 is off).
        """
        model, sigmas_p1 = self.initialize_model()

        frame_count = duration_to_h3_frames(duration_sec)
        w1, h1 = pass1_res
        w2, h2 = pass2_res
        ref_images = dict(ref_images or {})

        # Explicit pictures keep their numbered slots; continuity uses a free slot.
        if last_frame_guide is not None:
            for index in range(9):
                key = f"ref_image_{index}"
                if key not in ref_images:
                    ref_images[key] = last_frame_guide
                    break
        _LOG.debug("Clip %d: %d reference picture(s)%s", clip_index + 1, len(ref_images),
                   " (the last is the previous clip's last frame)" if last_frame_guide is not None else "")

        # =====================================================================
        # STAGE 1: PASS 1 (Low-Res Ref2VA Generation)
        # =====================================================================
        cached = cached_pass1 is not None
        if progress_cb:
            progress_cb("pass1", f"Clip {clip_index + 1} - Pass 1 " + (f"loaded from cache ({w1}x{h1})" if cached else f"(Draft {w1}x{h1})") + "...", 0.1)

        if cached:
            _LOG.info(f"Clip {clip_index + 1}: Pass 1 ({w1}x{h1}, {frame_count} frames, seed={seed}) loaded from the pass-1 disk cache; pass-1 sampling skipped")
        else:
            _LOG.info(
                f"Clip {clip_index + 1}: Pass 1 Ref2VA at {w1}x{h1}, {frame_count} frames, seed={seed}"
            )

        # Generate Pass 1 conditioning + empty latent
        out_p1 = out_p2 = None
        pos_p1 = latent_p1 = None
        trim_p1 = 0
        if cached:
            pass
        elif getattr(self, "single_text_encode", False):
            # The text encoder sees the reference pictures at the encode size, so with pass 2 off the
            # single encode runs at ``p1_text_size`` (the size a standard final would use) to give the
            # same pass-1 conditioning as a full render that is finalised at that size.
            enc_w, enc_h = (w2, h2) if pass2_enabled or not p1_text_size else p1_text_size
            out_p2 = MiniMaxH3ReferenceToVideo.execute(
                clip=self.clip,
                vae=self.vae,
                audio_vae=self.audio_vae,
                prompt=prompt,
                width=enc_w,
                height=enc_h,
                length=frame_count,
                ref_image_size="match",
                ref_images=ref_images,
                ref_videos=ref_videos or None,
                ref_video_audios=ref_video_audios or None,
                ref_audios=ref_audios or None,
            )
            out_p1 = self._pass1_from_pass2(out_p2, ref_images, w1, h1, frame_count, clip_index)
        if out_p1 is None and not cached:
            out_p1 = MiniMaxH3ReferenceToVideo.execute(
                clip=self.clip,
                vae=self.vae,
                audio_vae=self.audio_vae,
                prompt=prompt,
                width=w1,
                height=h1,
                length=frame_count,
                ref_image_size="match",
                ref_images=ref_images,
                ref_videos=ref_videos or None,
                ref_video_audios=ref_video_audios or None,
                ref_audios=ref_audios or None,
            )
        if not cached:
            pos_p1 = self._bridge(_safe_get_output(out_p1, 0, "positive"), "pass 1")
            latent_p1 = _safe_get_output(out_p1, 1, "latent")

        # If extending (clip_index > 0) and previous latent available: attach motion context
        self.last_trim_frames = 0
        self.last_draft = None
        if not cached and clip_index > 0 and previous_draft is not None:
            _LOG.info(f"Clip {clip_index + 1}: Applying Pass 1 motion context ({context_length}f video)")
            pos_p1, trim_p1, _, _, _ = self.motion_ram.apply(
                pos_p1,
                latent_p1,
                previous_draft,
                str(context_length),
                int(audio_context_length),
            )

        # Pass 2 conditioning is encoded NOW, while the text encoder is still
        # resident, instead of between the passes. It only depends on the
        # prompt, the reference images (sized to the target resolution) and the
        # previous clip's final latent, all of which are known here -- so the
        # UNet can stay resident straight through pass 1 -> upscale -> pass 2
        # and the text-encoder/UNet round trip per clip is halved.
        if pass2_enabled and out_p2 is None:
            out_p2 = MiniMaxH3ReferenceToVideo.execute(
                clip=self.clip,
                vae=self.vae,
                audio_vae=self.audio_vae,
                prompt=prompt,
                width=w2,
                height=h2,
                length=frame_count,
                ref_image_size="match",
                ref_images=ref_images,
                ref_videos=ref_videos or None,
                ref_video_audios=ref_video_audios or None,
                ref_audios=ref_audios or None,
            )
        if pass2_enabled:
            pos_p2 = self._bridge(_safe_get_output(out_p2, 0, "positive"), "pass 2")
            latent_p2 = _safe_get_output(out_p2, 1, "latent")
            if clip_index > 0 and previous_latent is not None:
                pos_p2, trim_p2, _, _, _ = self.motion_ram.apply(
                    pos_p2,
                    latent_p2,
                    previous_latent,
                    str(context_length),
                    int(audio_context_length),
                )
                self.last_trim_frames = int(trim_p2)
        elif cached:
            self.last_trim_frames = int(cached_trim)
        else:
            self.last_trim_frames = int(trim_p1)
        del out_p1, out_p2

        if cached:
            sampled_p1 = {"samples": cached_pass1["samples"]}
        else:
            # Sample Pass 1 with the pass-1 sigmas
            early_stop = self.dual_active and self.pass1_early_stop and pass2_enabled
            if (not early_stop and self.pass1_steps > 0 and not self.dual_active and pass2_enabled
                    and self.pass1_steps < len(sigmas_p1) - 1):
                k = self.pass1_steps
                early_stop = True
                _LOG.info("Clip %d: Pass 1 early stop: %d of %d HyperFlow steps (sigma %.4f), x0 estimate -> upscale -> normal refine",
                          clip_index + 1, k, len(sigmas_p1) - 1, float(sigmas_p1[k]))
                sigmas_p1 = pass1_steps_sigmas(sigmas_p1, k)
            elif early_stop:
                sigmas_p1 = dual_pass1_sigmas(sigmas_p1, self.dual_split_step)
                _LOG.info("Clip %d: dual sampling early stop, pass 1 runs grid[:%d] (%d steps): %s",
                          clip_index + 1, len(sigmas_p1), len(sigmas_p1) - 1,
                          [round(float(x), 4) for x in sigmas_p1])
            p1_sampler = None
            if self.ptd_mode:
                if latent_p1.get("noise_mask") is not None:
                    raise ValueError("PTD pass 1 needs an empty generated latent without a noise mask")
                p1_sampler = self.ptd_pack.sampler.make_sampler(self.ptd_controller)
                _LOG.info("Clip %d: Pass 1 = PTD, %d hops (%d model evaluations)", clip_index + 1,
                          self.ptd_controller.hops, self.ptd_controller.hops)
            sampled_p1 = self._sample_euler(
                model=model,
                positive=pos_p1,
                latent_image=latent_p1,
                sigmas=sigmas_p1,
                seed=seed,
                return_denoised=early_stop,
                sampler=p1_sampler,
            )

        if not pass2_enabled:
            self.last_draft = None if cached else sampled_p1
            return sampled_p1, None

        # =====================================================================
        # LATENT 3D UPSCALE & ALIGNMENT
        # =====================================================================
        if progress_cb:
            progress_cb("upscale", f"Clip {clip_index + 1} - 3D Latent Upscaling to {w2}x{h2}...", 0.5)

        _LOG.info(f"Clip {clip_index + 1}: Separating AV latent and upscaling video to {w2}x{h2}...")

        self.last_draft = None if cached else sampled_p1
        # Separate AV latent
        video_p1, audio_p1 = _streams_from_latent(sampled_p1, "samples")

        # 3D Latent Upscaling on video stream
        video_upscaled = self._run_3d_upscaler(video_p1, w2, h2)

        actual_hw = tuple(int(x) for x in video_upscaled.shape[-2:])
        if actual_hw != (h2 // 16, w2 // 16):
            raise ValueError(
                f"Upscaler returned {actual_hw[1] * 16}x{actual_hw[0] * 16}; "
                f"expected {w2}x{h2}. Use dimensions aligned to 32."
            )

        # Rejoin with audio stream
        rejoined_latent = {
            "samples": comfy.nested_tensor.NestedTensor((video_upscaled, audio_p1))
        }

        # =====================================================================
        # STAGE 2: PASS 2 (High-Res Refinement via the quality tail)
        # =====================================================================
        if progress_cb:
            progress_cb("pass2", f"Clip {clip_index + 1} - Pass 2 (Refine {w2}x{h2} with quality tail)...", 0.7)

        _LOG.info(
            f"Clip {clip_index + 1}: Pass 2 refining at {w2}x{h2} (conditioning pre-encoded), quality tail denoise={pass2_denoise}"
        )

        self._resolve_windowed_last_hop(frame_count, w2, h2, clip_index)

        # Pass-2 model (a different LoRA for the refine tail when configured)
        model_p2 = self.initialize_pass2_model()

        if self.ptd_mode and self.pass2_engine == ptd_mode.PASS2_ENGINE_PTD:
            # The adapter's final hop only: re-noise the upscaled draft to that hop's knot and evaluate it once.
            hop = self._ptd_last_hop()
            sigmas_p2 = ptd_mode.hop_tail_sigmas(self.ptd_controller.sigmas, hop).float().cpu()
            _LOG.info("Clip %d: Pass 2 = PTD last hop (hop %d of %d, start sigma %.4f, 1 model evaluation); "
                      "pass2_denoise %.2f is not used", clip_index + 1, hop + 1, self.ptd_controller.hops,
                      float(sigmas_p2[0]), float(pass2_denoise))
            final_sampled = self._sample_euler(
                model=model_p2,
                positive=pos_p2,
                latent_image=rejoined_latent,
                sigmas=sigmas_p2,
                seed=seed,
                sampler=ptd_mode.make_hop_sampler(self.ptd_controller, hop),
            )

        else:
            # Quality-tail sigmas (HyperFlow tail, or the turbo schedule's tail)
            sigmas_p2 = self._pass2_sigmas(model_p2, pass2_denoise)

            # Sample Pass 2 (High-Res Refine): whole clip, or overlapping temporal
            # windows when the clip is longer than pass2_chunk_frames.
            if self._pass2_chunking_active(frame_count, w2, h2):
                final_sampled = self._sample_pass2_chunked(
                    model=model_p2,
                    positive=pos_p2,
                    latent=rejoined_latent,
                    sigmas=sigmas_p2,
                    seed=seed,
                )
            else:
                final_sampled = self._sample_euler(
                    model=model_p2,
                    positive=pos_p2,
                    latent_image=rejoined_latent,
                    sigmas=sigmas_p2,
                    seed=seed,
                )

        if getattr(self, "audio_refine_steps", 0) > 0:
            if progress_cb:
                progress_cb("audio_refine", f"Clip {clip_index + 1} - Audio-only refine ({self.audio_refine_steps} steps)...", 0.85)
            final_sampled = self._refine_audio(final_sampled, pos_p2, seed, clip_index)

        # The disk renderer extracts the guide from the corrected preview decode.
        return final_sampled, None
