"""Chain memory: pick the previous clip's last frame that shows the people, and label it in the next clip's prompt."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

_LOG = logging.getLogger("minimax_h3_master_extender.chain_memory")

MIN_SHOT_FRAMES = 6
CUT_FLOOR = 0.06
CUT_MEDIAN_FACTOR = 4.0
FACE_SCORE = 0.5

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
    total = int(video.shape[0])
    last = total - 1
    info = {"index": last, "shots": [], "faces": [], "fallback": True}
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
                best = (top, max(i for i, c in counts.items() if c == top))
        if best is not None:
            info["index"] = best[1]
            info["fallback"] = False
    _LOG.info("Chain memory: frame %d of %d (shots %s, faces per sampled frame %s%s)", info["index"], total, info["shots"], info["faces"],
              ", fallback to the last frame" if info["fallback"] else "")
    return video[info["index"]:info["index"] + 1].to(device="cpu").clone(), info
