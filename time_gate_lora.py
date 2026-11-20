"""Time-gated LoRA for MiniMax H3 pass 1.

A LoRA's effect is applied through forward hooks on the four linears of every DiT block, scaled per token by a gate built from the
packed layout: the video tokens of the latent frames inside the chosen time spans, nothing on motion-context tokens, a constant on
text / reference tokens. The LoRA is never added to the model patcher, so it costs nothing outside the hooks' lifetime.

Time mapping: video latent frame k covers the pixel frames FRAME_PER_TOKEN[k % 5] starting at the cumulative sum of the earlier
spans (1, 4, 4, 4, 4, 1, 4, ...), 24 fps. Its strength is the mean of the span strength over those pixel frames, where pixel frame p
is at clip time (p - trim_frames) / 24 (trim_frames = the continuation overlap carried from the previous clip, 0 for the first clip;
0 s = the clip's first new frame). Overlap frames get 0. Audio latent step j (40 Hz, channel-major stereo rows) is at clip time
(j + 0.5) / 40 - trim_frames / 24.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os

import torch
import torch.nn.functional as F

_LOG = logging.getLogger("minimax_h3_master_extender.time_gate")

FPS = 24.0
AUDIO_FPS = 40.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
LINEAR_NAMES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
SWIGLU_ROWS = 16384

_TENSOR_CACHE = {}


def parse_time_gate(text):
    """Parse the ``time_gate_json`` input. "" / whitespace -> None (feature off). Raises ValueError on a malformed spec."""
    if text is None or not str(text).strip():
        return None
    try:
        cfg = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"time_gate_json is not valid JSON: {exc}") from exc
    if not isinstance(cfg, dict) or not cfg.get("lora") or not isinstance(cfg.get("clips"), dict):
        raise ValueError('time_gate_json needs {"lora": "<loras file>", "clips": {"<clip index>": [[start_s, end_s, strength], ...]}}')
    clips = {}
    for key, spans in cfg["clips"].items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"time_gate_json: clip key {key!r} is not an integer") from None
        parsed = []
        for span in spans:
            if not isinstance(span, (list, tuple)) or len(span) != 3:
                raise ValueError(f"time_gate_json: clip {key} span {span!r} is not [start_s, end_s, strength]")
            start, end, strength = (float(v) for v in span)
            if end < start:
                raise ValueError(f"time_gate_json: clip {key} span {span!r} ends before it starts")
            parsed.append((start, end, strength))
        clips[index] = parsed
    return {"lora": str(cfg["lora"]), "other_tokens": float(cfg.get("other_tokens", 1.0)), "clips": clips}


def clip_spans(cfg, clip_index):
    """The spans of one clip, or None when the clip is not listed (then it renders with no LoRA at all)."""
    if not cfg:
        return None
    return cfg["clips"].get(int(clip_index))


def strength_at(spans, seconds):
    """Piecewise-constant strength at each time in ``seconds`` (float tensor): the first span with start <= t < end, else 0."""
    out = torch.zeros_like(seconds, dtype=torch.float32)
    taken = torch.zeros_like(seconds, dtype=torch.bool)
    for start, end, strength in spans:
        hit = (seconds >= start) & (seconds < end) & ~taken
        out[hit] = float(strength)
        taken |= hit
    return out


def video_frame_gates(latent_t, trim_frames, spans):
    """Gate per video latent frame: the mean span strength over the pixel frames it covers (overlap frames count 0)."""
    gates = torch.zeros(latent_t, dtype=torch.float32)
    pixel = 0
    for k in range(latent_t):
        n = FRAME_PER_TOKEN[k % 5]
        p = torch.arange(pixel, pixel + n, dtype=torch.float64)
        t = (p - float(trim_frames)) / FPS
        s = strength_at(spans, t)
        s[p < trim_frames] = 0.0
        gates[k] = s.mean()
        pixel += n
    return gates


def audio_step_gates(audio_t, trim_frames, spans):
    """Gate per audio latent step (40 Hz); steps inside the continuation overlap count 0."""
    t = (torch.arange(audio_t, dtype=torch.float64) + 0.5) / AUDIO_FPS - float(trim_frames) / FPS
    s = strength_at(spans, t)
    s[t < 0] = 0.0
    return s


def build_gate(segments, signature, spans, trim_frames, other_tokens=1.0):
    """Per-token gate [S] (float32, CPU) from a PackedLayout's segment table and signature.

    segments: [(start, stop, kind)], kinds text / cond / cond_audio / ref_img / ref_audio / audio / video.
    signature: (text_len, latent_t, lat_h, lat_w, audio_t).
    """
    total = segments[-1][1]
    gate = torch.zeros(total, dtype=torch.float32)
    _, latent_t, _, _, audio_t = signature
    vgates = video_frame_gates(latent_t, trim_frames, spans)
    agates = audio_step_gates(audio_t, trim_frames, spans) if audio_t else None
    for a, b, kind in segments:
        if kind in ("text", "ref_img", "ref_audio"):
            gate[a:b] = float(other_tokens)
        elif kind in ("cond", "cond_audio"):
            gate[a:b] = 0.0
        elif kind == "video":
            gate[a:b] = vgates.repeat_interleave((b - a) // latent_t)
        elif kind == "audio":
            gate[a:b] = agates.repeat(2)
    return gate


def _swiglu(x):
    import comfy.ops

    return comfy.ops.INPUT_ACT_EAGER["swiglu"](x)


def load_lora_tensors(name):
    """A/B pairs of the 200 block linears on the CPU, cached: {(block, "attn.qkv_proj"): (A [r, in], B [out, r])}."""
    import folder_paths
    import comfy.utils

    path = folder_paths.get_full_path("loras", name)
    if path is None:
        raise FileNotFoundError(f"time_gate_json: LoRA '{name}' not found in models/loras")
    key = (path, os.path.getmtime(path))
    if key not in _TENSOR_CACHE:
        _TENSOR_CACHE.clear()
        raw = comfy.utils.load_torch_file(path, safe_load=True)
        pairs = {}
        for tensor_name, tensor in raw.items():
            if not tensor_name.endswith(".lora_A.weight"):
                continue
            base = tensor_name[: -len(".lora_A.weight")]
            parts = base.split(".")
            if len(parts) < 4 or parts[0] != "diffusion_model" or parts[1] != "blocks":
                continue
            layer = ".".join(parts[3:])
            if layer not in LINEAR_NAMES:
                continue
            pairs[(int(parts[2]), layer)] = (tensor, raw[base + ".lora_B.weight"])
        if not pairs:
            raise ValueError(f"time_gate_json: '{name}' has no diffusion_model.blocks.N.* lora_A/lora_B pairs")
        _TENSOR_CACHE[key] = pairs
    return _TENSOR_CACHE[key]


class TimeGateLora:
    """Forward hooks that add ``gate * B(A(x))`` to the block linears while the context is open.

    fc2 never reaches a module hook (the quantized path calls the fused kernel directly), so its LoRA term is taken from the swiglu
    of fc1's output, kept as a rank-r intermediate, and added to the MLP output in a hook on the MLP module. fc1 / fc2 may be called
    on token chunks (KJNodes feed-forward chunking); a per-linear cursor tracks which rows a call covers.
    """

    def __init__(self, diffusion_model, pairs, spans, trim_frames, other_tokens=1.0, device=None, dtype=torch.bfloat16, swiglu=None):
        self.model = diffusion_model
        self.spans = list(spans)
        self.trim_frames = int(trim_frames)
        self.other_tokens = float(other_tokens)
        self.swiglu = swiglu or _swiglu
        self.device = device
        self.dtype = dtype
        self.pairs = {k: (a.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)) for k, (a, b) in pairs.items()}
        self._cast = {}
        self.gate = None
        self._layout = None
        self._gate_cpu = None
        self._cursor = {}
        self._pending = {}
        self._handles = []

    def _modules(self):
        for i, block in enumerate(self.model.blocks):
            yield i, block

    def _gate_for(self, layout, rows, device):
        if self._layout is not layout:
            self._gate_cpu = build_gate(layout.segments, layout.signature, self.spans, self.trim_frames, self.other_tokens)
            self._layout = layout
            self._gate_dev = {}
        if self._gate_cpu.shape[0] != rows:
            raise RuntimeError(f"time gate: layout has {self._gate_cpu.shape[0]} tokens but the block sees {rows}")
        if device not in self._gate_dev:
            self._gate_dev[device] = self._gate_cpu.to(device)
        return self._gate_dev[device]

    def _ab(self, key, x):
        slot = (key, x.dtype, x.device)
        if slot not in self._cast:
            a, b = self.pairs[key]
            self._cast[slot] = (a.to(device=x.device, dtype=x.dtype), b.to(device=x.device, dtype=x.dtype))
        return self._cast[slot]

    def _rows(self, i, name, n):
        total = self.gate.shape[0]
        if n == total:
            return self.gate
        c = self._cursor.get((i, name), 0)
        g = self.gate[c:c + n]
        self._cursor[(i, name)] = (c + n) % total
        return g

    @staticmethod
    def _add(out, mid, b):
        flat = out.reshape(-1, out.shape[-1])
        flat.addmm_(mid.to(flat.dtype), b.t().to(flat.dtype))
        return out if flat.data_ptr() == out.data_ptr() else flat.view_as(out)

    def _block_pre(self, i):
        def hook(module, args, kwargs):
            self._cursor = {k: v for k, v in self._cursor.items() if k[0] != i}
            self._pending.pop(i, None)
            if i == 0:
                options = kwargs.get("transformer_options") or {}
                layout = options.get("minimax_h3_layout")
                if layout is None:
                    self.gate = None
                else:
                    x = args[0] if args else kwargs["x"]
                    self.gate = self._gate_for(layout, x.shape[-2], x.device)
        return hook

    def _linear_hook(self, i, name):
        key = (i, name)

        def hook(module, args, out):
            if self.gate is None:
                return None
            x = args[0]
            n = x.shape[-2]
            g = self._rows(i, name, n)
            a, b = self._ab(key, x)
            mid = F.linear(x, a)
            mid = (mid.reshape(-1, mid.shape[-1]) * g.to(mid.dtype)[:, None])
            out = self._add(out, mid, b)
            if name == "mlp.fc1":
                fc2 = self._ab((i, "mlp.fc2"), x)
                flat = out.reshape(-1, out.shape[-1])
                mids = []
                for r in range(0, flat.shape[0], SWIGLU_ROWS):
                    act = self.swiglu(flat[r:r + SWIGLU_ROWS])
                    mids.append(F.linear(act, fc2[0]))
                mid2 = torch.cat(mids, dim=0) * g.to(flat.dtype)[:, None]
                self._pending.setdefault(i, []).append(mid2)
            return out
        return hook

    def _mlp_hook(self, i):
        def hook(module, args, out):
            parts = self._pending.pop(i, None)
            if not parts or self.gate is None:
                return None
            mid = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            if mid.shape[0] != out.reshape(-1, out.shape[-1]).shape[0]:
                _LOG.warning("time gate: block %d fc2 rows %d != mlp output rows %d; skipped", i, mid.shape[0], out.shape[-2])
                return None
            _, b = self._ab((i, "mlp.fc2"), out)
            return self._add(out, mid.to(out.dtype), b)
        return hook

    def install(self):
        for i, block in self._modules():
            self._handles.append(block.register_forward_pre_hook(self._block_pre(i), with_kwargs=True))
            self._handles.append(block.attn.qkv_proj.register_forward_hook(self._linear_hook(i, "attn.qkv_proj")))
            self._handles.append(block.attn.out_proj.register_forward_hook(self._linear_hook(i, "attn.out_proj")))
            self._handles.append(block.mlp.fc1.register_forward_hook(self._linear_hook(i, "mlp.fc1")))
            self._handles.append(block.mlp.register_forward_hook(self._mlp_hook(i)))
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.pairs = {}
        self._cast = {}
        self._pending = {}
        self._cursor = {}
        self.gate = self._gate_cpu = self._layout = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False


def scope(model_patcher, cfg, clip_index, trim_frames):
    """Context manager for one clip's pass 1: the hooks when the clip is listed in the config, else a no-op."""
    spans = clip_spans(cfg, clip_index)
    if spans is None:
        return contextlib.nullcontext()
    import comfy.model_management

    diffusion_model = model_patcher.get_model_object("diffusion_model")
    pairs = load_lora_tensors(cfg["lora"])
    device = comfy.model_management.get_torch_device()
    _LOG.info("Clip %d: time-gated LoRA %s on pass 1, spans %s, other tokens %.2f, motion-context overlap %d frames",
              clip_index + 1, cfg["lora"], spans, cfg["other_tokens"], trim_frames)
    return TimeGateLora(diffusion_model, pairs, spans, trim_frames, cfg["other_tokens"], device=device)
