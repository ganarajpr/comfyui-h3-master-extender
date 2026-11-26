"""
CodecOPSD reference-token pruning for the Master Extender.

Method: CodecOPSD (arXiv:2611.31184, github.com/Kiteretsu77/CodecOPSD). Every reference is
re-encoded with x264 and the tokens the codec predicts for free are dropped from the packed
sequence; kept tokens keep their original rotary position. The decision code is vendored in
codecopsd_vendor/ (see THIRD_PARTY_NOTICE.md).

Nothing here runs unless pruning is requested: the reader, the gazer and the runtime patch are
all loaded lazily. A reference block gains a bool "keep" mask (one entry per dense reference
token row, flatten order of the (token frame, h/32, w/32) mask); the patch reads it.

Runtime patch (installed on the first pruned render, wrapping whatever is current):
  * PackedLayout.__init__ is wrapped in place: the original builds the dense layout, then the
    dropped reference rows are cut out of position_ids / img_pos / audio_pos / the update masks /
    segments / seq_len. signature is left alone.
  * MiniMaxH3Model._cond_video_rows is wrapped: the original builds all condition rows, then the
    rows of pruned references are dropped, so the visual condition rows stay aligned with
    ~img_update.
"""
import hashlib
import logging
import math
import os
import sys
import types

import torch

_LOG = logging.getLogger("minimax_h3_master_extender.codec_prune")

PATCH_MARKER = "_h3_codec_prune_patch"
_HERE = os.path.dirname(os.path.abspath(__file__))
_READER_EXE = os.path.join(_HERE, "bin", "codec_reader.exe")

REF_SEGMENT_KINDS = ("ref_img", "ref_audio")
VIDEO_DROP = 0.70

_reader_ready = False
_patched = False
_orig_layout_init = None
_orig_cond_rows = None
_keep_cache = {}
_KEEP_CACHE_MAX = 64


# --- reader ---------------------------------------------------------------------------------

def _exe_read(path, t, h, w, threads):
    """Same contract as the codecopsd_codec wheel: (motion_bytes, residual_bytes, skip_bytes)."""
    import subprocess
    import tempfile

    mbw = (w + 15) // 16
    mbh = (h + 15) // 16
    n = mbw * mbh * t
    fd, out = tempfile.mkstemp(suffix=".signals.bin")
    os.close(fd)
    try:
        p = subprocess.run([_READER_EXE, str(path), str(t), str(w), str(h), str(max(1, int(threads))), out],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError("codec_reader failed (%d): %s" % (p.returncode, p.stderr.strip()[:500]))
        with open(out, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(out)
        except OSError:
            pass
    if len(data) != n * 9:
        raise RuntimeError("codec_reader produced %d bytes, expected %d" % (len(data), n * 9))
    return data[:4 * n], data[4 * n:8 * n], data[8 * n:9 * n]


def reader_available():
    if "codecopsd_codec" in sys.modules:
        return True
    try:
        import codecopsd_codec  # noqa: F401
        return True
    except ImportError:
        return os.path.isfile(_READER_EXE)


def ensure_reader():
    """Make `import codecopsd_codec` resolve. Raises when no reader exists; never falls back silently."""
    global _reader_ready
    if _reader_ready:
        return
    try:
        import codecopsd_codec  # noqa: F401  (upstream wheel)
        _LOG.info("[codecprune] reader: upstream codecopsd_codec")
    except ImportError:
        if not os.path.isfile(_READER_EXE):
            raise RuntimeError(
                "ref_codec_prune is on but no H.264 reader is installed: neither the codecopsd_codec "
                "package nor %s exists. Place the built codec_reader.exe there, or turn ref_codec_prune off."
                % _READER_EXE)
        shim = types.ModuleType("codecopsd_codec")
        shim.read = _exe_read
        sys.modules["codecopsd_codec"] = shim
        _LOG.info("[codecprune] reader: %s", _READER_EXE)
    _reader_ready = True


def _gazer():
    ensure_reader()
    from .codecopsd_vendor import codec_gazer
    return codec_gazer


# --- keep masks -----------------------------------------------------------------------------

def _crop_to_grid(keep, t, h, w):
    """Codec masks are sized ceil(px/32); the packed layout uses floor(latent/2)."""
    if keep.shape[0] < t or keep.shape[1] < h or keep.shape[2] < w:
        raise ValueError("codec mask %s is smaller than the layout grid (%d,%d,%d); the reference size is "
                         "not 32-aligned" % (tuple(keep.shape), t, h, w))
    return keep[:t, :h, :w].contiguous()


def _digest(x):
    return hashlib.sha1(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def picture_keep(resized):
    """resized: comfy IMAGE [B,H,W,3] in 0..1 (H, W multiples of 32) -> bool [1, H//32, W//32]."""
    h, w = int(resized.shape[1]), int(resized.shape[2])
    key = ("image", h, w, _digest(resized[:1]))
    hit = _keep_cache.get(key)
    if hit is not None:
        return hit.clone()
    g = _gazer()
    x = resized[:1].permute(0, 3, 1, 2).contiguous().float() * 2.0 - 1.0
    keep = g.keep_mask(x[0], kind="image").cpu()
    keep = _crop_to_grid(keep, 1, h // 32, w // 32)
    if len(_keep_cache) >= _KEEP_CACHE_MAX:
        _keep_cache.pop(next(iter(_keep_cache)))
    _keep_cache[key] = keep
    return keep.clone()


def video_keep(frames, latent_t, drop=VIDEO_DROP):
    """frames: comfy IMAGE [T,H,W,3] in 0..1 -> bool [latent_t, H//32, W//32]."""
    h, w = int(frames.shape[1]), int(frames.shape[2])
    key = ("video", h, w, int(frames.shape[0]), float(drop), _digest(frames))
    hit = _keep_cache.get(key)
    if hit is not None:
        return hit.clone()
    g = _gazer()
    x = frames.permute(0, 3, 1, 2).contiguous().float() * 2.0 - 1.0
    keep = g.keep_mask(x, kind="video", grid=g.MINIMAX_H3, drop=drop).cpu()
    keep = _crop_to_grid(keep, latent_t, h // 32, w // 32)
    if len(_keep_cache) >= _KEEP_CACHE_MAX:
        _keep_cache.pop(next(iter(_keep_cache)))
    _keep_cache[key] = keep
    return keep.clone()


def picture_size(img, width, height, ref_image_size):
    """Resized (tw, th) of a reference picture, exactly as MiniMaxH3ReferenceToVideo computes it."""
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE
    h, w = img.shape[1], img.shape[2]
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((width * height) / (w * h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return tw, th


def resized_picture(img, width, height, ref_image_size):
    from comfy_extras.nodes_minimax_h3 import _resize
    tw, th = picture_size(img, width, height, ref_image_size)
    return _resize(img[:1], tw, th, "disabled"), tw, th


def _video_frames(video_frames, frame_count):
    """The reference video as the core node feeds it to the VAE (canvas, frame trim)."""
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, _resize, adapt_canvas
    vh, vw = video_frames.shape[1], video_frames.shape[2]
    cw, ch = adapt_canvas(vw, vh)
    if vw * vh < cw * ch:
        cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    frames = _resize(video_frames, cw, ch, "disabled")
    if frames.shape[0] > frame_count:
        frames = frames[:frame_count]
    n = frames.shape[0]
    if n < 5:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
    while n % 17 != 5:
        n -= 1
    return frames[:n], cw, ch


def attach_keep(cond, ref_images, width, height, ref_image_size="match", ref_videos=None, length=None,
                video_drop=VIDEO_DROP):
    """Return `cond` with "keep" masks on its image (and video) reference blocks.

    cond comes from the core MiniMaxH3ReferenceToVideo called with the same ref_images / width /
    height / ref_image_size (and ref_videos / length when videos are pruned). The conditioning and
    its blocks are copied, never mutated: a cached conditioning stays as it was.
    """
    try:
        blocks = list(cond[0][1].get("minimax_refs") or [])
    except (TypeError, IndexError, AttributeError, KeyError):
        return cond
    images = [img for img in (ref_images or {}).values() if img is not None]
    image_idx = [i for i, b in enumerate(blocks) if b.get("kind") == "image"]
    if len(image_idx) != len(images):
        raise RuntimeError("codec prune: %d image blocks for %d reference pictures" % (len(image_idx), len(images)))
    for slot, img in zip(image_idx, images):
        resized, tw, th = resized_picture(img, width, height, ref_image_size)
        keep = picture_keep(resized)
        blocks[slot] = {**blocks[slot], "keep": keep}
        _LOG.info("[codecprune] picture %dx%d: kept %d/%d tokens (%.1f%%)", tw, th,
                  int(keep.sum()), keep.numel(), 100.0 * float(keep.float().mean()))

    videos = [v for v in (ref_videos or {}).values() if v is not None]
    video_idx = [i for i, b in enumerate(blocks) if b.get("kind") in ("video", "video_audio")]
    if videos:
        if len(video_idx) != len(videos) or length is None:
            raise RuntimeError("codec prune: %d video blocks for %d reference videos" % (len(video_idx), len(videos)))
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent
        _, frame_count = _empty_av_latent(width, height, length)
        for slot, vid in zip(video_idx, videos):
            frames, cw, ch = _video_frames(vid, frame_count)
            blk = blocks[slot]
            keep = video_keep(frames, int(blk["latent_t"]), video_drop)
            blocks[slot] = {**blk, "keep": keep}
            _LOG.info("[codecprune] video %dx%d/%df: kept %d/%d tokens (%.1f%%)", cw, ch, frames.shape[0],
                      int(keep.sum()), keep.numel(), 100.0 * float(keep.float().mean()))

    return [[t, {**d, "minimax_refs": blocks}] for t, d in cond]


def keep_flat(keep, n_full, label="reference"):
    """Validate a keep mask against the dense row count. None = keep everything."""
    if keep is None:
        return None
    k = (keep if torch.is_tensor(keep) else torch.as_tensor(keep)).reshape(-1).to(torch.bool).cpu()
    if k.numel() != n_full:
        raise ValueError("%s: keep mask has %d entries, the dense layout has %d rows" % (label, k.numel(), n_full))
    if bool(k.all()):
        return None
    if not bool(k.any()):
        k = k.clone()
        k[0] = True
    return k


# --- layout patch ---------------------------------------------------------------------------

def _expected_ref_segments(blk):
    kind = blk.get("kind")
    if kind == "image":
        return ("ref_img",)
    if kind == "audio":
        return ("ref_audio",) if int(blk.get("ref_audio_t", 0)) > 0 else ()
    if kind in ("video", "video_audio"):
        return ("ref_audio", "ref_img") if int(blk.get("ref_audio_t", 0)) > 0 else ("ref_img",)
    raise RuntimeError("codec prune: unknown reference kind %r" % (kind,))


def filter_layout(layout, refs):
    """Cut the dropped reference rows out of a dense, already-built PackedLayout in place."""
    ref_segs = [(a, b, k) for a, b, k in layout.segments if k in REF_SEGMENT_KINDS]
    wanted = [(i, k) for i, blk in enumerate(refs or []) for k in _expected_ref_segments(blk)]
    if len(wanted) != len(ref_segs):
        raise RuntimeError("codec prune: reference/layout segment count mismatch (%d expected, %d present)"
                           % (len(wanted), len(ref_segs)))
    mask = torch.ones(layout.seq_len, dtype=torch.bool)
    for (idx, want), (a, b, got) in zip(wanted, ref_segs):
        if got != want:
            raise RuntimeError("codec prune: reference segment order changed (%r expected, %r found)" % (want, got))
        if want != "ref_img":
            continue
        k = keep_flat(refs[idx].get("keep"), b - a, "reference %d (%s)" % (idx, refs[idx].get("kind")))
        if k is not None:
            mask[a:b] = k
    if bool(mask.all()):
        return layout

    new_index = torch.cumsum(mask.to(torch.long), 0) - 1
    layout.position_ids = layout.position_ids[mask.to(layout.position_ids.device)]
    for pos_name, upd_name in (("img_pos", "img_update"), ("audio_pos", "audio_update")):
        pos = getattr(layout, pos_name)
        sel = mask[pos]
        setattr(layout, pos_name, new_index[pos[sel]])
        setattr(layout, upd_name, getattr(layout, upd_name)[sel])
    segs, off = [], 0
    for a, b, kind in layout.segments:
        n = int(mask[a:b].sum())
        segs.append((off, off + n, kind))
        off += n
    layout.segments = segs
    layout.seq_len = int(mask.sum())
    return layout


def _any_keep(refs):
    return any(isinstance(b, dict) and b.get("keep") is not None for b in (refs or ()))


def _patched_layout_init(self, *args, **kwargs):
    _orig_layout_init(self, *args, **kwargs)
    refs = kwargs["refs"] if "refs" in kwargs else (args[6] if len(args) > 6 else None)
    if _any_keep(refs):
        filter_layout(self, refs)


setattr(_patched_layout_init, PATCH_MARKER, True)


# --- condition-row patch --------------------------------------------------------------------

def _latent_rows(mm, model, z):
    b, _, t, h, w = z.shape
    pt, ph, pw = model.patch_size
    return b * (t // pt) * (h // ph) * (w // pw)


def _patched_cond_video_rows(self, payload, device):
    rows = _orig_cond_rows(self, payload, device)
    refs = payload.get("refs")
    if rows is None or not _any_keep(refs):
        return rows
    import comfy.ldm.minimax.model as mm
    items = [(kf["latent"], None) for kf in (payload.get("keyframes") or ()) if kf.get("latent") is not None]
    items += [(r["latent"], r.get("keep")) for r in (refs or ()) if "latent" in r]
    counts = [_latent_rows(mm, self, z) for z, _ in items]
    if sum(counts) != rows.shape[0]:
        raise RuntimeError("codec prune: %d condition rows expected from the latents, the model built %d"
                           % (sum(counts), rows.shape[0]))
    mask = torch.ones(rows.shape[0], dtype=torch.bool)
    off = 0
    for (z, keep), n in zip(items, counts):
        k = keep_flat(keep, n, "condition rows")
        if k is not None:
            mask[off:off + n] = k
        off += n
    return rows[mask.to(rows.device)]


setattr(_patched_cond_video_rows, PATCH_MARKER, True)


def install():
    """Wrap PackedLayout.__init__ and MiniMaxH3Model._cond_video_rows. Idempotent."""
    global _patched, _orig_layout_init, _orig_cond_rows
    import comfy.ldm.minimax.model as mm
    cls, model_cls = mm.PackedLayout, mm.MiniMaxH3Model
    if getattr(cls.__init__, PATCH_MARKER, False) and getattr(model_cls._cond_video_rows, PATCH_MARKER, False):
        _patched = True
        return True
    if getattr(cls.__init__, PATCH_MARKER, False) != getattr(model_cls._cond_video_rows, PATCH_MARKER, False):
        raise RuntimeError("codec prune: only one of the two patches is installed; restart ComfyUI")
    _orig_layout_init = cls.__init__
    _orig_cond_rows = model_cls._cond_video_rows
    cls.__init__ = _patched_layout_init
    model_cls._cond_video_rows = _patched_cond_video_rows
    _patched = True
    _LOG.info("[codecprune] patched PackedLayout.__init__ and MiniMaxH3Model._cond_video_rows")
    return True


def is_installed():
    return _patched
