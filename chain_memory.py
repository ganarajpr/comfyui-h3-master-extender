"""Chain memory: pick the previous clip's last frame that shows the people, and label it in the next clip's prompt."""

from __future__ import annotations

import logging
import math
import time
import os
import re
from pathlib import Path

_LOG = logging.getLogger("minimax_h3_master_extender.chain_memory")

MODES = ("off", "picture", "context", "picture + context")

MIN_SHOT_FRAMES = 6
CUT_FLOOR = 0.06
CUT_MEDIAN_FACTOR = 4.0
FACE_SCORE = 0.5
# The memory picture carries scene state (positions, contact, clothes), not detail: it is passed at most this wide. At 720p a full-size
# extra reference made the text encode and the DiT see a ~1280x720 picture (+11 s per join measured, 2026-11-20).
MEMORY_MAX_WIDTH = 640

MEMORY_DEFINITION = "<Subject {s}> is the state of this scene at the moment just before, in <Picture {p}>: a weak reference for continuity only."
MEMORY_RETENTION = (
    "<Subject {s}> (<Picture {p}>, the moment just before): partially_preserved - carry over from it where each person is relative to the others "
    "and to the room, which side they are on and the distance between them; their poses (sitting, standing, kneeling or lying, which way each "
    "faces, where their arms and hands are); any physical contact between them (holding hands, a hand on a shoulder, an arm around someone, "
    "leaning); what each holds and in which hand; the state of their clothes (worn or taken off, open or closed, sleeves, wet, torn or dirty); "
    "any cuts, bandages or marks and which side they are on; glasses, jewellery, hats and the state of their hair; and any changes to the room "
    "(moved props, doors, lamps). Do not take its framing, camera angle or shot size, its facial expressions, or anyone's face or identity; "
    "faces and identity come from the character pictures."
)

SECTION_NAMES = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music")

_HASH_HEADER = re.compile(r"^##\s+(\S.*?)\s*$")
_BARE_HEADER = re.compile(r"^(%s)\s*:\s*$" % "|".join(SECTION_NAMES))
_SUBJECT = re.compile(r"<Subject\s+(\d+)>")


def parse_mode(value):
    """The chain_memory input as one of MODES. A missing value, a non-listed value and False are "off"; a literal True is "picture"."""
    if value is True:
        return "picture"
    if isinstance(value, str) and value in MODES:
        return value
    return "off"


def uses_picture(mode):
    return mode in ("picture", "picture + context")


def uses_context(mode):
    return mode in ("context", "picture + context")


def _header_at(line):
    """(style, name) when ``line`` opens a section: style '#' for '## name', ':' for a bare 'name:' line."""
    m = _HASH_HEADER.match(line)
    if m:
        return "#", m.group(1).strip().lower()
    m = _BARE_HEADER.match(line.strip())
    if m:
        return ":", m.group(1)
    return None


def _section_end(lines, start, style):
    """Index just past the last non-blank line of the section whose header is at ``start``."""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        h = _header_at(lines[j])
        if h and h[0] == style:
            end = j
            break
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return end


def _find_section(lines, name):
    for i, line in enumerate(lines):
        h = _header_at(line)
        if h and h[1] == name:
            return i, h[0]
    return None


def next_subject_number(prompt):
    numbers = [int(n) for n in _SUBJECT.findall(prompt)]
    return max(numbers) + 1 if numbers else 1


def append_memory_text(prompt, picture_number):
    prompt = str(prompt or "")
    s = next_subject_number(prompt)
    definition = MEMORY_DEFINITION.format(s=s, p=picture_number)
    retention = MEMORY_RETENTION.format(s=s, p=picture_number)
    lines = prompt.split("\n")
    inserts = []
    leftovers = []
    for name, text in (("subject_definitions", definition), ("retention_analysis", retention)):
        found = _find_section(lines, name)
        if found is None:
            leftovers.append(text)
        else:
            inserts.append((_section_end(lines, found[0], found[1]), text))
    for at, text in sorted(inserts, key=lambda x: -x[0]):
        lines.insert(at, text)
    out = "\n".join(lines)
    if leftovers:
        out = out.rstrip("\n") + "\n\n" + "\n".join(leftovers)
    return out


_detector = None
_detector_failed = False


def _get_detector():
    global _detector, _detector_failed
    if _detector is not None or _detector_failed:
        return _detector
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", root=os.path.join(str(Path.home()), ".insightface"),
                           allowed_modules=["detection"], providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _detector = app
    except Exception as exc:  # noqa: BLE001 -- any import or model-load failure means "no detector"
        _detector_failed = True
        _LOG.warning("Chain memory: face detector unavailable (%s); using the clip's last frame", exc)
    return _detector


def _count_faces(detector, frame):
    """Faces with det_score >= FACE_SCORE in one (H, W, 3) float [0,1] frame."""
    import numpy as np

    rgb = (frame.clamp(0, 1) * 255).round().to("cpu").numpy().astype(np.uint8)
    faces = detector.get(np.ascontiguousarray(rgb[:, :, ::-1]))
    return sum(1 for f in faces if float(getattr(f, "det_score", 1.0)) >= FACE_SCORE)


def _small_gray(video):
    import torch
    import torch.nn.functional as F

    out = []
    for a in range(0, video.shape[0], 32):
        chunk = video[a:a + 32].to(torch.float32)
        gray = (chunk[..., 0] * 0.299 + chunk[..., 1] * 0.587 + chunk[..., 2] * 0.114).unsqueeze(1)
        out.append(F.adaptive_avg_pool2d(gray, (36, 64)))
    return torch.cat(out, 0)


def _shot_ranges(video):
    """[(start, end_exclusive)] of the clip's shots."""
    import torch

    total = int(video.shape[0])
    if total < 2:
        return [(0, total)]
    small = _small_gray(video)
    diffs = (small[1:] - small[:-1]).abs().mean(dim=(1, 2, 3))
    threshold = max(CUT_FLOOR, CUT_MEDIAN_FACTOR * float(torch.median(diffs)))
    starts = [0] + [i + 1 for i, d in enumerate(diffs.tolist()) if d > threshold]
    shots = [[s, (starts[k + 1] if k + 1 < len(starts) else total)] for k, s in enumerate(starts)]
    while len(shots) > 1:
        short = next((k for k, (a, b) in enumerate(shots) if b - a < MIN_SHOT_FRAMES), None)
        if short is None:
            break
        if short > 0:
            shots[short - 1][1] = shots[short][1]
        else:
            shots[1][0] = shots[0][0]
        del shots[short]
    return [(a, b) for a, b in shots]


def _sample_indices(start, end):
    last = end - 1
    span = end - start - 1
    picks = {start + int(0.5 * span), start + int(0.75 * span), end - MIN_SHOT_FRAMES}
    return sorted(min(max(p, start), last) for p in picks)


def select_memory_frame(video):
    """The clip's latest frame that shows the most people: (frame (1, H, W, 3) on CPU, info dict).

    ``video`` is the decoded clip, a float (T, H, W, 3) tensor in [0, 1]. Falls back to the last frame
    when no face detector is available or no shot has a face.
    """
    started = time.time()
    total = int(video.shape[0])
    last = total - 1
    info = {"index": last, "shots": [], "faces": [], "fallback": True, "shot_end": total, "tail_after_shot": 0}
    detector = _get_detector()
    if detector is not None:
        shots = _shot_ranges(video)
        info["shots"] = [list(s) for s in shots]
        best = None
        for a, b in shots:
            counts = {idx: _count_faces(detector, video[idx]) for idx in _sample_indices(a, b)}
            info["faces"].append({str(k): v for k, v in counts.items()})
            top = max(counts.values())
            if top > 0 and (best is None or top >= best[0]):
                best = (top, max(i for i, c in counts.items() if c == top), b)
        if best is not None:
            info["index"] = best[1]
            info["fallback"] = False
            info["shot_end"] = best[2]
            info["tail_after_shot"] = total - best[2]
    _LOG.info("Chain memory: frame %d of %d (shots %s, faces per sampled frame %s%s; picked in %.2f s)", info["index"], total, info["shots"],
              info["faces"], ", fallback to the last frame" if info["fallback"] else "", time.time() - started)
    frame = video[info["index"]:info["index"] + 1].to(device="cpu")
    info["seconds"] = round(time.time() - started, 2)
    return _shrink(frame), info


def _shrink(frame):
    """(1, H, W, 3) -> at most MEMORY_MAX_WIDTH wide, aspect kept, even size."""
    import torch.nn.functional as F

    h, w = int(frame.shape[1]), int(frame.shape[2])
    if w <= MEMORY_MAX_WIDTH:
        return frame.clone()
    nw = MEMORY_MAX_WIDTH
    nh = max(2, int(round(h * nw / w / 2.0)) * 2)
    out = F.interpolate(frame.permute(0, 3, 1, 2).float(), size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)
    return out.clamp(0, 1).permute(0, 2, 3, 1).contiguous()


# Same constants as motion_context_ram (kept here so this module imports without ComfyUI).
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0
_STEPS_PER_CYCLE = 5
_FRAMES_PER_CYCLE = 17


def _pixel_frames(latent_t):
    return sum(_FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def truncate_context_latent_info(latent, drop_frames, context_length=22):
    """(latent, info): ``latent`` without its trailing ``ceil(drop_frames / 17)`` cycles of 5 video latent steps.

    Whole cycles keep the clip on the 17k+5 grid and the tail phase the motion-context tail needs. The audio keeps
    LAST round(5/3 x kept frames) steps of the clip's audio: the sound still comes from the clip's real end (so it continues
    seamlessly across the join) while the picture comes from the earlier shot. The original latent is returned unchanged when there is nothing to drop or the
    result would be shorter than ``context_length`` + 17 frames; ``info["applied"]`` says which.
    """
    video, audio = _streams_from_latent(latent)
    total_t = int(video.shape[2])
    total_frames = _pixel_frames(total_t)
    info = {"applied": False, "requested": int(drop_frames), "dropped_frames": 0, "dropped_steps": 0,
            "context_end_frame": total_frames, "reason": ""}
    if int(drop_frames) <= 0:
        info["reason"] = "nothing to drop"
        return latent, info
    n = math.ceil(int(drop_frames) / _FRAMES_PER_CYCLE)
    keep_t = total_t - n * _STEPS_PER_CYCLE
    kept_frames = _pixel_frames(keep_t) if keep_t > 0 else 0
    if keep_t < 2 or kept_frames < int(context_length) + _FRAMES_PER_CYCLE:
        info["reason"] = f"only {kept_frames} frames would remain (need {int(context_length) + _FRAMES_PER_CYCLE})"
        return latent, info
    keep_audio = int(round(_FRAME_RESCALE * kept_frames))
    new_video = video[:, :, :keep_t]
    # Audio from the TRUE tail (founder 2026-11-20): only the video jumps back to the shot with the people.
    new_audio = audio[..., int(audio.shape[-1]) - keep_audio:]
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        import comfy.nested_tensor

        parts = list(samples.unbind())
        parts[0], parts[1] = new_video, new_audio
        new_samples = comfy.nested_tensor.NestedTensor(tuple(parts))
    else:
        parts = list(samples)
        parts[0], parts[1] = new_video, new_audio
        new_samples = type(samples)(parts)
    out = {k: latent[k] for k in latent.keys() if k != "samples"}
    out["samples"] = new_samples
    info.update(applied=True, dropped_frames=total_frames - kept_frames, dropped_steps=n * _STEPS_PER_CYCLE,
                context_end_frame=kept_frames)
    return out, info


def truncate_context_latent(latent, drop_frames, context_length=22):
    return truncate_context_latent_info(latent, drop_frames, context_length)[0]


def _streams_from_latent(latent):
    samples = latent["samples"]
    parts = list(samples.unbind()) if getattr(samples, "is_nested", False) else list(samples)
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    return video, audio


def map_early_pick(info, shift, final_total):
    """(index, tail_after_shot) in the FINAL decoded clip for a pick made on the pass-1 draft.

    The draft pick runs on the draft's frames from the join's trim onward, so index 0 is the clip's first
    own frame. The final clip starts ``shift`` frames later in the same numbering (the seam shift chosen
    by the pair decode), so a draft index ``j`` is final index ``j - shift``. A draft pick that fell back
    to the last frame stays the final clip's last frame with no tail.
    """
    last = int(final_total) - 1
    if info.get("fallback"):
        return last, 0
    index = min(max(int(info["index"]) - int(shift), 0), last)
    shot_end = min(max(int(info["shot_end"]) - int(shift), index + 1), int(final_total))
    return index, max(0, int(final_total) - shot_end)


class EarlyPick:
    """The memory-frame pick for one clip, run on a CPU thread over the pass-1 draft while pass 2 samples."""

    def __init__(self, decoder="", decode_seconds=0.0):
        self.decoder = decoder
        self.decode_seconds = decode_seconds
        self.info = None
        self.error = None
        self._thread = None

    def start(self, frames):
        import threading

        def run():
            try:
                _frame, self.info = select_memory_frame(frames)
            except Exception as exc:  # noqa: BLE001 -- an early pick that fails just means "use the slow path"
                self.error = str(exc)
                _LOG.warning("Chain memory: early pick failed (%s); the final decode will pick the frame", exc)

        self._thread = threading.Thread(target=run, name="chain-memory-early-pick", daemon=True)
        self._thread.start()
        return self

    def result(self, timeout=120.0):
        """The pick's info dict, or None when it failed or is not done within ``timeout`` seconds."""
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                _LOG.warning("Chain memory: early pick not finished after %.0f s; the final decode will pick the frame", timeout)
                return None
        return self.info
