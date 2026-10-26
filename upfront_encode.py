"""Up-front text encode: every clip's text+reference encode runs once before clip 1 and is held on the CPU.

The helpers are duck-typed (no torch import) so they run, and are tested, without ComfyUI.
"""

from __future__ import annotations

import time


def _map(obj, fn):
    if isinstance(obj, dict):
        return {k: _map(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_map(v, fn) for v in obj)
    if hasattr(obj, "to") and not isinstance(obj, (str, bytes)):
        return fn(obj)
    return obj


def tree_to(obj, device):
    """Copy of a conditioning / latent structure with every tensor (or nested tensor) moved to ``device``."""
    return _map(obj, lambda t: t.to(device))


def tree_bytes(obj):
    """Bytes held by the tensors in a nested structure."""
    if isinstance(obj, dict):
        return sum(tree_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tree_bytes(v) for v in obj)
    tensors = getattr(obj, "tensors", None)
    if tensors is not None:
        return tree_bytes(list(tensors))
    if hasattr(obj, "numel") and hasattr(obj, "element_size"):
        return int(obj.numel()) * int(obj.element_size())
    return 0


def output_parts(out):
    """The positive conditioning / latent pair of an encode result as plain objects (NodeOutput, tuple or dict)."""
    for attr in ("result", "args"):
        parts = getattr(out, attr, None)
        if parts is not None:
            return tuple(parts)
    if isinstance(out, dict):
        return (out.get("positive"), out.get("latent"))
    return tuple(out)


def first_render_index(clips, existing, stored, fingerprint_fn):
    """Index of the first clip the job will render rather than reuse from the disk cache.

    Mirrors the loop's reuse test: a clip is reused when it is on disk, is validated or does not re-roll its seed,
    and its stored fingerprint equals the one its inputs and every earlier clip give (a validated clip with no stored
    fingerprint passes). The chain is cumulative, so every clip after the first one rendered renders too.
    ``fingerprint_fn(previous_fp, clip_cfg)`` raises TypeError / ValueError for an unfingerprintable clip.
    """
    previous = None
    for i, cfg in enumerate(clips):
        validated = str(cfg.get("validated", False)).lower() == "true"
        rerolls = str(cfg.get("seed_mode", "fixed")).lower() in ("randomize", "increment", "decrement")
        if i >= existing or not (validated or not rerolls):
            return i
        try:
            expected = fingerprint_fn(previous, cfg)
        except (TypeError, ValueError):
            return i
        have = stored[i] if i < len(stored) else None
        if not (expected and (have == expected or (validated and not have))):
            return i
        previous = expected
    return len(clips)


def upfront_indices(clips, first, run_mode, identity_continuity):
    """Clip indices to encode before the loop: the clips from ``first`` on, only the first in clip_by_clip mode,
    and with identity_continuity only clip 1 (a later clip's guide frame is the previous clip's output)."""
    if first >= len(clips):
        return []
    if str(run_mode) == "clip_by_clip":
        indices = [first]
    else:
        indices = list(range(first, len(clips)))
    if identity_continuity:
        indices = [i for i in indices if i == 0]
    return indices


class UpfrontEncodeCache:
    """Encode results by clip index, held on the CPU until the clip is rendered."""

    def __init__(self):
        self._items = {}
        self.seconds = 0.0
        self.bytes = 0
        self.count = 0

    def __len__(self):
        return len(self._items)

    def __contains__(self, index):
        return index in self._items

    def fill(self, indices, encode_fn, device="cpu"):
        """Run ``encode_fn(index)`` for each clip index and store its result on ``device``."""
        start = time.perf_counter()
        for index in indices:
            self._items[index] = tree_to(output_parts(encode_fn(index)), device)
        self.seconds = time.perf_counter() - start
        self.bytes = sum(tree_bytes(v) for v in self._items.values())
        self.count = len(self._items)

    def take(self, index, device):
        """Remove and return the stored result for ``index`` on ``device``, or None."""
        item = self._items.pop(index, None)
        return None if item is None else tree_to(item, device)

    def clear(self):
        self._items.clear()

    def summary(self):
        return "Up-front text encode: %d clips in %.1f s, %.2f GB on CPU" % (self.count, self.seconds, self.bytes / 1e9)
