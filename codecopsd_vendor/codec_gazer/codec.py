"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Codec pruning is a contribution of this paper; please cite it if you use, modify, or port this code.

Codec gazer: the condition tokens worth giving to the model, read from an H.264 re-encode of the condition.

The condition is re-encoded with x264 and decoded with a patched FFmpeg that returns the motion vectors, the luma
residual and the skipped macroblocks (the package `codecopsd-codec`). A token (32 x 32 pixels in one token frame)
is kept when the largest motion in it reaches the motion threshold or its mean absolute residual exceeds the residual
threshold; a token whose content the codec predicts from earlier frames for free is dropped. The settings are the ones
the CodecOPSD students were trained with. See README.md next to this file for the settings, the sources and the speed rules.

On top of that threshold decision a video can be cut to a share of its tokens (priority drop; with `select: max` the
threshold decision itself when it already drops more) or by its own density (elastic drop), and the student's text encoder gets a mask of its own (`Gaze.vlm_keep`).
"""

import os
import subprocess
import tempfile
from typing import NamedTuple

import imageio_ffmpeg
import numpy as np
import torch

from ..configs import load

_CONFIG = load("codec_gazer")  # src/codecopsd/configs/codec_gazer.yaml
# kind -> settings of the two forms (motion threshold in pixels at 480 x 640, mean |residual| threshold, x264 preset /
# crf / gop); CONDITIONS: per-condition replacements of some of them (`settings`)
SETTINGS = _CONFIG["kinds"]
CONDITIONS = _CONFIG["conditions"]
# Token frames of the VAE: (pixel frames per token frame after the first frame, frames after which that pattern
# restarts). MiniMax-H3 encodes 17-frame clips, each one frame then 4 x 4 frames (17n + 5 frames -> 5n + 2 token frames).
MINIMAX_H3 = (4, 17)
# Cosmos 3 (its video VAE): one frame, then 4 frames per token frame, with no clip restart (4n + 1 frames -> n + 1).
COSMOS3 = (4, None)
FPS = _CONFIG["encode"]["fps"]
DECODE_THREADS = min(8, os.cpu_count() or 1)
# PORT NOTE: the host plugin sets this to its own ffmpeg so the x264 build (and therefore the
# masks) does not depend on whichever imageio-ffmpeg version the user's environment happens
# to ship. None falls back to imageio's binary, as upstream does.
FFMPEG_EXE = None


def _ffmpeg_exe():
    return FFMPEG_EXE or imageio_ffmpeg.get_ffmpeg_exe()
PRIORITY_DROP = _CONFIG["priority_drop"]["video_default"]
PRIORITY_SELECT = _CONFIG["priority_drop"].get("select", "exact")  # "max": drop = max(threshold drop, share)
VLM_PICTURE_SCALE = _CONFIG["vlm_drop"]["picture_threshold_scale"]
VLM_VIDEO = _CONFIG["vlm_drop"].get("video", "threshold")  # "dit": a video's text encoder mask = the transformer's


class Gaze(NamedTuple):
    """The masks of one condition, bool [T', H / 32, W / 32] (T' = 1 for a picture).

    keep: the tokens the transformer gets. vlm_keep: the tokens the student's text encoder reads (a superset of `keep`;
    None unless asked for).
    """

    keep: torch.Tensor
    vlm_keep: torch.Tensor | None = None


def settings(kind):
    """Gazer settings of a form ("video" / "image") or of a named condition (`conditions` of the config: its form's
    settings with the condition's own entries replacing theirs; every named condition is a video)."""
    if kind in SETTINGS:
        return SETTINGS[kind]
    if kind in CONDITIONS:
        return {**SETTINGS["video"], **CONDITIONS[kind]}
    raise ValueError(f"unknown gazer kind {kind!r}: one of {sorted(SETTINGS)} or a condition of {sorted(CONDITIONS)}")


def frame_groups(num_frames, grid=MINIMAX_H3):
    """First pixel frame of every token frame of a `num_frames` clip on the token frames `grid`."""
    step, clip = grid
    clip = clip or num_frames
    starts = []
    for start in range(0, num_frames, clip):
        starts += [start, *range(start + 1, min(start + clip, num_frames), step)]
    return starts


def to_rgb24(condition):
    """Video [T, 3, H, W] or image [3, H, W], float in [-1, 1] or uint8 -> uint8 RGB frames [T, H', W', 3] on the
    condition's device, rounded as the students were trained (elementwise float32 maths: the same bytes on the GPU and
    on the CPU). A height or width that is not a multiple of 32 is reflect-padded at the bottom / right to the next
    one (H', W'), as the Cosmos 3 students were trained: the transformer pads such a latent grid to whole tokens, and
    the partial last token row / column is gazed from the reflected pixels."""
    frames = condition[None] if condition.ndim == 3 else condition
    if frames.dtype != torch.uint8:
        frames = frames.float().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
    for dim in (2, 3):
        size = frames.shape[dim]
        if size % 32:
            index = torch.arange(size + -size % 32, device=frames.device)
            index = torch.where(index < size, index, 2 * (size - 1) - index)  # reflect, the edge not repeated
            frames = frames.index_select(dim, index)
    return frames.permute(0, 2, 3, 1).contiguous()


def keep_mask(condition, kind="video", grid=MINIMAX_H3, drop=None):
    """Token keep-mask of a condition (the transformer's: `gaze(...).keep`).

    Args:
        condition: video [T, 3, H, W] or image [3, H, W], float in [-1, 1] or uint8, on any device; H and W
            multiples of 32, or reflect-padded to them (`to_rgb24`).
        kind: "video", or a condition named in the config's `conditions` (its own settings); an image is always
            "image".
        grid: token frames of the VAE (`MINIMAX_H3`, `COSMOS3`).
        drop: videos only. None: the threshold decision; "elastic": elastic drop on top of it; a float in (0, 1):
            priority drop of exactly that share of the tokens, or with `priority_drop.select: max` the threshold
            decision when it already drops more (`PRIORITY_DROP` = 0.70 is the default of the pipeline).

    Returns:
        bool [T', ceil(H / 32), ceil(W / 32)] (T' = 1 for an image); at least one token is kept.
    """
    return gaze(condition, kind, grid, drop).keep


def drop_mask(condition, kind="video", grid=MINIMAX_H3, drop=None):
    """The tokens the codec gazer drops: `~keep_mask(...)`, same arguments. bool [T', ceil(H / 32), ceil(W / 32)],
    True = dropped (never every token: at least one is kept)."""
    return ~keep_mask(condition, kind, grid, drop)


def gaze(condition, kind="video", grid=MINIMAX_H3, drop=None, vlm=False, overrides=None):
    """The `Gaze` of a condition: `keep_mask`'s mask and, with `vlm`, the student text encoder's mask (`vlm_drop` of the
    config; a number = the picture threshold scale instead of the config's; "dit" = the transformer's mask itself).
    `overrides`: {form: {setting: value}}, e.g. {"video": {"motion_px": 6.0, "residual": 1.2}}, replaces the config's
    settings of that form ("video" / "image") for this call (the demo's threshold fields); {"video": {"select":
    "exact"}} makes a priority drop drop exactly its share even when the threshold decision drops more ("max", the
    config's `priority_drop.select`). Other arguments as in `keep_mask`."""
    return _gaze(to_rgb24(condition).cpu(), "image" if condition.ndim == 3 else kind, grid, drop, vlm,
                 overrides=overrides)


def submit_gaze(executor, condition, kind="video", grid=MINIMAX_H3, drop=None, vlm=False, overrides=None):
    """`gaze` in a thread of `executor`; returns a future of the `Gaze`.

    The condition is quantized on its own device and a GPU condition reaches the host by an asynchronous copy into
    pinned memory, so this returns at once and the caller goes on (text and condition encoding on the GPU) while the
    CPU gazes. Threads, not processes: the reader and x264 run outside the Python interpreter lock.
    """
    rgb, copied = to_rgb24(condition), None
    if rgb.is_cuda:
        host = torch.empty(rgb.shape, dtype=torch.uint8, pin_memory=True)
        host.copy_(rgb, non_blocking=True)
        copied = torch.cuda.Event()
        copied.record()
        rgb = host
    return executor.submit(_gaze, rgb, "image" if condition.ndim == 3 else kind, grid, drop, vlm, copied, overrides)


def _signals(rgb, cfg, grid, token_px=32):
    """Host uint8 frames [T, H, W, 3] -> per token (`token_px` x `token_px` pixels, 32 or 16): largest motion (pixels)
    and mean |luma residual|, float32 [T', H / token_px, W / token_px] each, from one x264 encode with the settings
    `cfg`."""
    try:
        from codecopsd_codec import read  # the H.264 reader, a dependency of this package (pyproject.toml)
    except ImportError as e:
        raise ImportError("the codec gazer's H.264 reader (package codecopsd-codec, Linux x86_64) is missing: "
                          "reinstall with `pip install -e .`") from e

    t, h, w, _ = rgb.shape
    encode, gop = _CONFIG["encode"], cfg["gop"]
    ffmpeg_exe = _ffmpeg_exe()
    # PORT NOTE (Windows): a NamedTemporaryFile stays open with an exclusive lock, so ffmpeg cannot
    # write it and exits with -13. mkstemp + close, removed in finally, behaves the same everywhere.
    fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        # x264's thread count changes its decisions: pin it to x264's own choice on a machine with many CPUs (one
        # thread per two macroblock rows), which the students were trained with.
        subprocess.run(
            [ffmpeg_exe, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
             f"{w}x{h}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", cfg["preset"], "-crf",
             str(cfg["crf"]), "-g", str(gop), "-threads", str(max(1, (h + 15) // 16 // 2)), "-pix_fmt", encode["pix_fmt"],
             "-x264-params", f"keyint={gop}:min-keyint={gop}:{encode['x264_params']}", tmp_name],
            input=memoryview(rgb.numpy()).cast("B"), check=True,
        )
        signals = read(tmp_name, t, h, w, DECODE_THREADS)
    finally:
        try:
            os.remove(tmp_name)
        except OSError:
            pass

    # per 16 x 16 macroblock -> per token: n x n macroblocks (n = token_px / 16), the most of the frames of a token frame
    motion, res, skip = (np.frombuffer(b, d).reshape(t, h // 16, w // 16)
                         for b, d in zip(signals, (np.int32, np.int32, np.uint8)))
    coded = skip == 0  # a skipped macroblock counts as no motion and no residual
    groups = frame_groups(t, grid)
    n = token_px // 16
    motion = np.maximum.reduceat((motion * coded).reshape(t, h // token_px, n, w // token_px, n).max((2, 4)), groups)
    res = np.maximum.reduceat((res * coded).reshape(t, h // token_px, n, w // token_px, n).sum((2, 4)), groups)
    return np.sqrt(motion.astype(np.float32)), res.astype(np.float32) / np.float32(token_px * token_px)


def _threshold(motion, res, cfg, h, w, scale=1.0):
    """The threshold decision (both thresholds times `scale`). The motion threshold, given at 480 x 640, grows with
    sqrt(area); the residual threshold of a form with `residual_short_side` s0 (pictures: 1024) is given at short side
    s0 and is multiplied by sqrt(s0 / max(s0, short side))."""
    motion_px, residual = cfg["motion_px"] * scale, cfg["residual"] * scale
    s0 = cfg["residual_short_side"]
    if s0 is not None:
        residual *= (s0 / max(s0, min(h, w))) ** 0.5
    return (motion >= np.float32(motion_px * (h * w / (480 * 640)) ** 0.5)) | (res > np.float32(residual))


def _gaze(rgb, kind, grid, drop=None, vlm=False, copied=None, overrides=None):
    """`Gaze` of host uint8 frames [T, H, W, 3]; `copied` = event of their device -> host copy; `overrides` as in
    `gaze`."""
    if copied is not None:
        copied.synchronize()
    t, h, w, _ = rgb.shape
    cfg = {**settings(kind), **((overrides or {}).get(kind) or {})}  # kind: "video" / "image" or a named condition
    motion, res = _signals(rgb, cfg, grid)
    floor = _threshold(motion, res, cfg, h, w)
    keep = torch.from_numpy(floor)
    if t > 1 and drop == "elastic":
        keep = _elastic_drop(keep, torch.from_numpy(motion), torch.from_numpy(res))
    elif t > 1 and drop is not None:
        if not 0.0 < float(drop) < 1.0:
            raise ValueError(f"drop is None, 'elastic' or a share in (0, 1), got {drop!r}")
        select = cfg.get("select", PRIORITY_SELECT)  # "max" (config default) or "exact" (priority drop alone)
        if not (select == "max" and 1.0 - float(floor.mean()) >= float(drop)):  # max: a threshold decision
            keep = _priority_drop(torch.from_numpy(motion), torch.from_numpy(res), float(drop))  # that drops more stays
    if not keep.any():
        keep.view(-1)[0] = True
    vlm_keep = None
    if vlm == "dit":  # the student's text encoder reads exactly the transformer's tokens (the extreme mode)
        vlm_keep = keep.clone()
    elif vlm and t > 1 and VLM_VIDEO == "dit":  # a video: the student's text encoder reuses the transformer's mask
        vlm_keep = keep.clone()
    elif vlm:  # the student's text encoder: a picture at scaled thresholds, a video at the thresholds; always OR `keep`
        scale = VLM_PICTURE_SCALE if vlm is True else float(vlm)
        vlm_floor = _threshold(motion, res, cfg, h, w, scale) if t == 1 else floor
        vlm_keep = torch.from_numpy(vlm_floor) | keep
    return Gaze(keep, vlm_keep)


def _rank(x, mask):
    """Percentile rank in [0, 1] of the masked entries among themselves, ties at their mean rank; 0 elsewhere (and
    everywhere when the masked values do not vary)."""
    out = torch.zeros(x.numel(), dtype=torch.float32)
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    vals = x.reshape(-1)[idx].float()
    if vals.numel() <= 1 or bool(vals.max() == vals.min()):
        return out.reshape(x.shape)
    sorted_vals, order = torch.sort(vals)
    _, counts = torch.unique_consecutive(sorted_vals, return_counts=True)
    mean_rank = torch.repeat_interleave(torch.cumsum(counts, 0) - counts + (counts - 1) / 2.0, counts)
    ranks = torch.empty_like(mean_rank)
    ranks[order] = mean_rank
    out[idx] = ranks / (vals.numel() - 1)
    return out.reshape(x.shape)


def _top(score, k):
    """bool mask of the `k` highest scores; equal scores: the earlier token (token frame, row, column) first."""
    keep = torch.zeros(score.numel(), dtype=torch.bool)
    keep[torch.argsort(score.reshape(-1), descending=True, stable=True)[:k]] = True
    return keep.reshape(score.shape)


def _priority_drop(motion, res, drop):
    """Priority drop: of all the video's tokens, drop exactly `drop` (keep max(1, round((1 - drop) * n))), the
    lowest priority r * (1 + lam * m) first, r / m = percentile ranks of residual / motion among all the tokens. No
    threshold decision is involved: any token, token frame 0's included, may go."""
    everything = torch.ones(motion.shape, dtype=torch.bool)
    score = _rank(res, everything) * (1.0 + _CONFIG["priority_drop"]["lam"] * _rank(motion, everything))
    return _top(score, max(1, round((1.0 - drop) * motion.numel())))


def _elastic_drop(floor, motion, res):
    """Elastic drop: a video whose threshold decision (`floor`, bool [T', h, w]) keeps many tokens gives fewer, by its
    own drop d: d >= max_drop -> the floor as is; d >= static_drop (static) -> token frame 0's floor tokens and
    keep_static of the later frames' floor tokens; else (moving) keep_moving of them. The total drop is then clamped into
    [min_drop, max_drop] and never below d. The later tokens are taken by priority r * (1 + lam * m), r / m = percentile
    ranks of residual / motion among the floor tokens. The result is always a subset of the floor."""
    c = _CONFIG["elastic_drop"]
    n, first, rest = floor.numel(), int(floor[0].sum()), int(floor[1:].sum())
    floor_drop = 1.0 - (first + rest) / n
    if floor_drop >= c["max_drop"] or first + rest == 0:
        return floor
    share = c["keep_static"] if floor_drop >= c["static_drop"] else c["keep_moving"]
    drop = max(min(max(1.0 - (first + round(share * rest)) / n, c["min_drop"]), c["max_drop"]), floor_drop)
    k = max(1, round((1.0 - drop) * n))
    score = _rank(res, floor) * (1.0 + c["lam"] * _rank(motion, floor))
    key = torch.where(floor, score, torch.full_like(score, -10.0))
    key[0] = torch.where(floor[0], score[0] + 10.0, key[0])  # token frame 0's floor tokens first
    return _top(key, k)
