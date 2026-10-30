"""Built-in prompt rewriter for the Master Extender.

Turns each clip's panel prompt into a MiniMax-H3 description before the run
renders anything, using the engines of the MiniMax-H3-Prompt-Rewriter pack
(an optional dependency, resolved lazily so this node imports without it):

- one ``llama-server`` per run, holding the writer GGUF (+ its mmproj) once;
- a writer that can see gets the reference pictures themselves in the planner call
  and in every clip's writer call (downscaled once, first in the user message, in a
  fixed order, so the server reads them once); there is no caption step. Reference
  videos are still captioned on that server, and cached by frame hash;
- every pending clip is written on the same server, several at a time, with
  llama.cpp's thinking budget applied to the writer only (captions never think);
- the rewritten text replaces ``clip["prompt"]``, the original is kept in
  ``clip["prompt_raw"]`` and ``clip["prompt_rewritten"]`` marks it done, so the
  next run leaves it alone until the panel restores the raw text.

Nothing here touches the render path: when ``rewrite_mode`` is off the extender
behaves exactly as before.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

try:
    from . import e4_engine
    from . import shared_server
    from . import story_planner
    from . import strata_backend
except ImportError:  # imported as a top-level module (tests)
    import e4_engine
    import shared_server
    import story_planner
    import strata_backend

_LOG = logging.getLogger("minimax_h3_master_extender.rewriter")

_PUNCT = str.maketrans({"\u2013": "-", "\u2014": "-", "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                        "\u2026": "...", "\u00a0": " ", "\u2022": "*", "\u00d7": "x", "\u2192": "->"})


def _console_encoding() -> str:
    return getattr(sys.stdout, "encoding", None) or "utf-8"


def _console_safe(text) -> str:
    """Text the console can print: common typographic punctuation to ASCII, the rest as \\uXXXX escapes.

    ComfyUI's logger writes to a cp1252 console on Windows, where a model's en dash or any Devanagari line raises
    UnicodeEncodeError inside the logging call. Only what goes to the logger is changed; the text sent to the
    panel and stored in the clips is left as it is.
    """
    out = str(text).translate(_PUNCT)
    encoding = _console_encoding()
    try:
        return out.encode(encoding, "backslashreplace").decode(encoding)
    except (LookupError, UnicodeError):
        return out.encode("ascii", "backslashreplace").decode("ascii")


class ConsoleSafeFilter(logging.Filter):
    """Makes every record of the logger it is attached to printable on the console (model and user text included)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        record.msg = _console_safe(message)
        record.args = None
        return True


_LOG.addFilter(ConsoleSafeFilter())

PACK_MODULE = "MiniMax-H3-Prompt-Rewriter-ComfyUI"
PACK_SUB = "minimax_h3_rewriter"
MISSING = "(install MiniMax-H3-Prompt-Rewriter-ComfyUI for the built-in rewriter)"

MODES = ["off", "pending clips", "all clips"]
TASKS = ["auto", "Ref2VA", "T2VA"]
CAPTION_LENGTHS = ["brief", "standard", "detailed"]
RAW_ASKS = "raw asks"
FINAL_PROMPTS = "final prompts"
PLANNER_REFS = ["images", "captions", "off"]
STORY_ENGINES = e4_engine.ENGINES
E4_SCORES = e4_engine.SCORES
E4_DECISION_BUDGETS = e4_engine.DECISION_BUDGETS

REASONING_BUDGETS = ["1024", "2048", "4096"]
SHARED_SLOTS = 8   # slots of a server the Shot Writer leaves for this rewriter (ninfer, 65536 per slot: 8 clients gave ~1,110 tok/s against ~770 at 4)


def reasoning_budget(value) -> str:
    """The nearest allowed thinking budget for whatever an old workflow or an API graph sent (-1/0/junk -> 4096)."""
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return REASONING_BUDGETS[-1]
    if number <= 0:
        return REASONING_BUDGETS[-1]
    return str(min((int(b) for b in REASONING_BUDGETS), key=lambda b: (abs(b - number), -b)))
CONTINUITY = ["off", RAW_ASKS, FINAL_PROMPTS]

CONTINUITY_RULE = (
    "\n\nContinuity: the task message may carry a 'previous_clips' block. Those are the earlier "
    "clips of the same continuous video, in order, already rendered. The target video begins "
    "exactly where the last of them ends: keep the subjects, wardrobe, setting, lighting and "
    "visual style continuous with them, do not re-describe or repeat their events, and do not "
    "count them as shots of this clip."
)

CONTINUITY_RULE_FINAL = (
    "\n\nContinuity: the task message may carry a 'previous_clips' block with two parts. "
    "'Story so far' is the short raw asks of the earlier clips, as story memory only. "
    "'Previous clip, final prompt' is the finished H3 prompt of the clip immediately before this one, "
    "already rendered. The target video begins exactly where the LAST shot of that final prompt ends: "
    "take its end state (positions, wardrobe, props, light, set, who is where) as this clip's opening "
    "state, and keep the subjects and visual style continuous with it. Do not repeat its events, do not "
    "re-describe what it already showed, and do not count its shots as shots of this clip."
)

STORY_RULE = (
    "\n\nStory: the task message may carry a 'story' block. It is the whole film's story, from start to end. "
    "Use it only to decide what THIS clip covers: the next beat after where the previous clips end "
    "(clip 1 = the opening beat). Do not stage events that belong later in the story, do not compress the "
    "remaining story into this clip, and do not copy the story's wording or dialogue into the prompt unless "
    "the clip's own ask calls for it. The clip's own ask (original_prompt) wins over the story when they disagree. The story is also where wardrobe "
    "comes from: if it says what a character wears, or changes into, by this point, that is their outfit in this clip."
)

NO_CAPTIONS_RULE = (
    "\n\nNo reference picture is described in this task: each reference line only says a picture is attached. Do not "
    "guess what a picture shows. Define each person as the one whose face, build, skin tone and hair come from "
    "<Picture N>, and take the wardrobe, props and accessories from sources (1) to (3) above; when none of them says, "
    "give a plain outfit that suits the story. retention_analysis uses partially_preserved for that subject and says the "
    "identity features are retained from the picture and the clothing is replaced by the outfit defined in subject_definitions."
)

REFERENCE_RULE = (
    "\n\nReferences, identity versus wardrobe: a reference picture supplies IDENTITY only: face, build, skin tone, hair, "
    "apparent age, distinguishing marks. It does not decide what the character wears. In subject_definitions, define each "
    "person from the picture's identity features PLUS "
    "the wardrobe, props and accessories this clip needs, written out in full (garment, colour, fabric, trim, accessories), "
    "e.g. '<Subject 1> is the woman whose face, build, skin tone and hair come from <Picture 1>, in a deep red silk wedding "
    "saree with a gold border and a gold necklace.' Decide the wardrobe in this order: (1) the clip's own ask, including any "
    "'OPENS EXACTLY WHERE' or 'AT THE END OF THIS CLIP' block; (2) the state the previous clip ended in (previous_clips); "
    "(3) the story block; (4) only when none of these says what the character wears, the outfit shown on the picture. Use the "
    "same outfit wording in every shot, so it stays the same across the clip. When the chosen outfit is not the one shown on "
    "the picture, never describe the picture's own outfit anywhere in the prompt, and never write what the character is not "
    "wearing. For that subject, retention_analysis uses partially_preserved and says which identity features are retained "
    "from the picture and that the clothing is replaced by the outfit defined in subject_definitions. When the outfit IS the "
    "one shown on the picture, use fully_preserved."
)

ATTACHED_RULE = (
    "\n\nReference pictures: the task message opens with the reference pictures themselves, in order, each labelled "
    "'Picture N:' right before its image, and reference_assets repeats those labels. Look at each one. For a person take "
    "the identity only; for a prop or a location take its material, shape, layout and look. Cite each as <Picture N>."
)

STATE_RULE = (
    "\n\nContinuity state in the ask: the original_prompt may carry a 'CONTINUITY \u2014 THIS CLIP OPENS EXACTLY WHERE CLIP N "
    "ENDED' block and an 'AT THE END OF THIS CLIP' block. They are binding. The first shot opens in exactly the opening "
    "state (same place in the set, same positions and facing, same wardrobe and props, same time and light, continuing from "
    "the stated last action), without re-staging how it got there. The last shot ends in the state of the closing block. "
    "With no opening block (the first clip), wardrobe and props are those of the closing block unless a shot visibly changes them. "
    "Where the opening and closing wardrobe differ, the change happens inside this clip: shots before it show the opening "
    "outfit, shots after it the closing one, and the shot that makes the change says how. 'as on the reference' means the "
    "outfit shown on the picture."
)

CAPTION_TOKENS = 512

PICTURES_HEADER = "Reference pictures, in order (each labelled before its image):"

_LOCK = threading.Lock()


# --------------------------------------------------------------------------- pack


def _pack_root() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), PACK_MODULE)


def available() -> bool:
    return os.path.isdir(os.path.join(_pack_root(), PACK_SUB))


def _mod(name: str):
    """One module of the rewriter pack, imported the way ComfyUI did (or would).

    ComfyUI registers the pack under its folder name, so its modules are
    ``<folder>.minimax_h3_rewriter.<name>`` in ``sys.modules``. If the pack has
    not been loaded yet (custom nodes load alphabetically and this one sorts
    first), a bare package pointing at the folder is enough to import the
    sub-package without running the pack's own node registration.
    """
    full = f"{PACK_MODULE}.{PACK_SUB}.{name}"
    found = sys.modules.get(full)
    if found is not None:
        return found
    with _LOCK:
        if PACK_MODULE not in sys.modules:
            root = _pack_root()
            if not os.path.isdir(root):
                raise RuntimeError(MISSING)
            shell = types.ModuleType(PACK_MODULE)
            shell.__path__ = [root]
            sys.modules[PACK_MODULE] = shell
        return importlib.import_module(full)


def writer_choices() -> list[str]:
    try:
        # The pack's list is built from GGUF files only (catalog entries without 'repo'/'file' are skipped),
        # so an endpoint backend is appended here, last: the pack's first entry stays the dropdown default.
        return list(_mod("nodes").writer_choices()) + [strata_backend.LABEL]
    except Exception:
        return [MISSING]


# --------------------------------------------------------------------------- helpers


def _same_file(a: str, b: str) -> bool:
    return bool(a and b) and os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def shared_plan(inputs, model_path: str, nodes, paths):
    """What a server the Shot Writer starts must carry for the Master Extender (its ``inputs`` in the API prompt) that reads the writer's clips, or None.

    None unless that Extender will rewrite (``rewrite_mode`` not off) with the SAME model file as the writer, from plain widget values (a wired widget cannot be read here).
    Otherwise ``{"thinking", "budget", "message"}``: the rewriter's thinking switch, its budget and its budget message, which ninfer fixes at launch."""
    if not isinstance(inputs, dict) or not model_path:
        return None

    def plain(name, default):
        value = inputs.get(name, default)
        return None if isinstance(value, list) else value

    mode, label = plain("rewrite_mode", "off"), plain("rewrite_writer_model", "")
    if not isinstance(mode, str) or mode == "off" or not isinstance(label, str) or not label or label.startswith("(") or label == strata_backend.LABEL:
        return None
    try:
        files = _vision_files(nodes, paths, label)
    except Exception:
        return None
    if not _same_file(files[0], model_path):
        return None
    thinking = plain("rewrite_thinking", False)
    return {"thinking": bool(thinking), "budget": int(reasoning_budget(plain("rewrite_reasoning_budget", "4096"))), "message": str(plain("rewrite_reasoning_budget_message", "") or "").strip()}


def _writer_file(nodes, paths, label: str) -> str:
    choice = nodes._resolve_writer_choice(label)
    path = choice.reference if choice.local else (paths.catalog_file(choice.reference, choice.file) or "")
    return path if path and os.path.isfile(path) else ""


def _same_choice(writer, captioner) -> bool:
    """A writer entry and a captioner entry are one model when they name the same GGUF."""
    if writer.local or captioner.local:
        return bool(writer.local and captioner.local) and _same_file(writer.reference, captioner.reference)
    return writer.reference == captioner.reference and writer.file == captioner.file


def _vision_files(nodes, paths, writer_label: str) -> tuple[str, str]:
    """The writer's own model and mmproj when it can see images, else ("", "").

    The pack's captioner list is the catalog of models that ship an mmproj (or run on ninfer-serve with a
    built-in vision tower); a writer sees exactly when it is one of those entries and its files are on disk.
    """
    writer = nodes._resolve_writer_choice(writer_label)
    for label in nodes.captioner_choices():
        if label.startswith("("):
            continue
        try:
            captioner = nodes._resolve_captioner_choice(label)
        except Exception:
            continue
        if _same_choice(writer, captioner):
            return _captioner_files(nodes, paths, label)
    return "", ""


def _captioner_files(nodes, paths, label: str) -> tuple[str, str]:
    choice = nodes._resolve_captioner_choice(label)
    if choice.local:
        return choice.reference, choice.mmproj
    model = paths.catalog_file(choice.reference, choice.file) or ""
    mmproj = paths.catalog_file(choice.reference, choice.mmproj) or ""
    if model.lower().endswith(".ninfer") and os.path.isfile(model):
        return model, ""  # NInfer artifacts carry their own vision tower
    if model and mmproj and os.path.isfile(model) and os.path.isfile(mmproj):
        return model, mmproj
    return "", ""


def _reference_pictures(ordered: list) -> list[dict]:
    """The reference pictures as they are sent: downscaled to ``story_planner.REF_IMAGE_MAX_SIDE`` on the long side,
    PNG, as data URIs. Built once per run and reused byte for byte by the planner and every clip writer, so a server
    that caches the prompt prefix reads each picture once. ``tokens`` is the estimate for the sent size."""
    import base64
    import io

    import numpy
    from PIL import Image

    pictures = []
    for slot, tensor in ordered:
        frame = tensor[0] if tensor.dim() == 4 else tensor
        array = numpy.clip(frame.detach().cpu().float().numpy() * 255.0 + 0.5, 0, 255).astype(numpy.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        elif array.ndim == 3 and array.shape[-1] > 3:
            array = array[..., :3]
        picture = Image.fromarray(array)
        size = story_planner.fitted_size(*picture.size)
        if size != picture.size:
            picture = picture.resize(size, Image.LANCZOS)
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        pictures.append({"label": f"Picture {slot + 1}", "slot": slot, "size": size,
                         "tokens": story_planner.size_tokens(*size),
                         "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")})
    return pictures


def system_rules(*, continuity: str, chained: bool, later_clips: bool, story: bool, task: str, sees: bool,
                 state: bool) -> str:
    """What is appended to the writer's system prompt. Decided per run, never per clip, so the system prompt is
    identical in every clip's call and a server that caches the prompt prefix keeps the pictures after it."""
    out = ""
    if continuity != "off" and later_clips:
        out += CONTINUITY_RULE_FINAL if chained else CONTINUITY_RULE
    if story:
        out += STORY_RULE
    if task == "Ref2VA":
        out += REFERENCE_RULE + (ATTACHED_RULE if sees else NO_CAPTIONS_RULE)
    if state:
        out += STATE_RULE
    return out


VIDEO_FRAMES = 8
VIDEO_FPS = 24.0


def _video_key(frames, model_path: str, length: str) -> str:
    """A stable id for one reference clip: the frames the captioner will sample."""
    import torch

    count = int(frames.shape[0]) if frames.dim() == 4 else 1
    if count <= VIDEO_FRAMES:
        picks = list(range(count))
    else:
        step = (count - 1) / (VIDEO_FRAMES - 1)
        picks = sorted({int(round(i * step)) for i in range(VIDEO_FRAMES)})
    digest = hashlib.sha1()
    for index in picks:
        frame = frames[index] if frames.dim() == 4 else frames
        digest.update((frame.detach().float().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy().tobytes())
    return hashlib.sha1(f"video|{count}|{digest.hexdigest()}|{os.path.basename(model_path)}|{length}".encode()).hexdigest()


def _ordered(items: dict, prefix: str) -> list[tuple[int, object]]:
    out = []
    for key, value in (items or {}).items():
        if value is None or not str(key).startswith(prefix):
            continue
        try:
            out.append((int(str(key).rsplit("_", 1)[1]), value))
        except ValueError:
            continue
    out.sort(key=lambda item: item[0])
    return out


def _cache_path() -> str:
    import folder_paths

    folder = os.path.join(folder_paths.get_user_directory(), "minimax_h3_master")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "ref_captions.json")


def _load_cache() -> dict:
    try:
        with open(_cache_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        path = _cache_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        _LOG.debug("caption cache not saved", exc_info=True)


def _ordered_refs(refs: dict) -> list[tuple[int, object]]:
    """``(slot, tensor)`` pairs in slot order; slot 0 is ``<Picture 1>``."""
    out = []
    for key, value in (refs or {}).items():
        if value is None or not str(key).startswith("ref_image_"):
            continue
        try:
            out.append((int(str(key).rsplit("_", 1)[1]), value))
        except ValueError:
            continue
    out.sort(key=lambda item: item[0])
    return out


def source_of(clip: dict) -> str:
    """The text a rewrite starts from: the kept original, or the prompt as typed."""
    raw = clip.get("prompt_raw")
    if clip.get("prompt_rewritten") and isinstance(raw, str):
        return raw
    prompt = clip.get("prompt") if isinstance(clip.get("prompt"), str) else ""
    # Flag cleared but the text still is the previous rewrite: start from the original,
    # not from the model's own output.
    if isinstance(raw, str) and raw.strip() and prompt and prompt == clip.get("rewrite_text"):
        return raw
    return prompt


def fingerprint(source: str, duration: float, resolution: str, task: str, previous: str = "") -> str:
    """What a clip's rewrite depends on. A clip whose stored fingerprint differs is out of date.

    Its raw ask, duration, aspect and task, and the raw asks of the clips before
    it (``previous``, independent of the continuity setting): an earlier ask that
    changes re-renders this clip anyway. The writer, captioner, system prompt,
    references, continuity setting and the film story (``rewrite_story``) are
    left out on purpose: changing them applies to clips rewritten from now on
    and keeps the rendered ones.
    """
    payload = json.dumps([source.strip(), f"{float(duration):.2f}", resolution, task, previous.strip()],
                         ensure_ascii=False)
    return "v2:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def final_text_of(clip) -> str:
    """A clip's final prompt: the rewrite once rewritten, otherwise the prompt as typed."""
    if not isinstance(clip, dict):
        return ""
    return clip.get("prompt").strip() if isinstance(clip.get("prompt"), str) else ""


def final_hash(clip) -> str:
    return hashlib.sha1(final_text_of(clip).encode("utf-8")).hexdigest()


def _raw_ask_lines(clips: list, count: int) -> list[str]:
    lines = []
    for i, clip in enumerate(clips[:count]):
        if not isinstance(clip, dict):
            continue
        ask = " ".join(source_of(clip).split())
        if ask:
            lines.append(f"Clip {i + 1} ({float(clip.get('duration', 15) or 15):g}s): {ask}")
    return lines


_SHOT_AT = re.compile(r"^[ \t]*\[Shot \d+\]", re.MULTILINE)
_SECTION_AFTER = re.compile(r"^[ \t]*[*_#>\-]*[ \t]*(overall_soundscape|non_diegetic_music)", re.IGNORECASE | re.MULTILINE)


def last_shot_of(final: str) -> str:
    """The last [Shot N] paragraph of a final prompt's detailed_description, or '' when it has none."""
    starts = [m.start() for m in _SHOT_AT.finditer(final or "")]
    if not starts:
        return ""
    tail = final[starts[-1]:]
    end = _SECTION_AFTER.search(tail)
    return (tail[:end.start()] if end else tail).strip()


def continuity_block(clips: list, index: int, continuity: str) -> str:
    """What the writer is told about clips 1..index-1.

    ``raw asks``: their raw asks, in order. Short, in the user's words, and
    available before any rewrite runs, so pending clips can be written in parallel.

    ``final prompts``: the full final prompt of clip index-1 (what it actually
    became), led by the raw asks of clips 1..index-2 as short story memory.
    Needs clip index-1 already written, so the clips are written in order.
    """
    if continuity == "off" or index <= 0:
        return ""
    if continuity == FINAL_PROMPTS:
        parts = []
        memory = _raw_ask_lines(clips, index - 1)
        if memory:
            parts.append("Story so far (raw asks of the earlier clips, memory only):\n" + "\n".join(memory))
        last = clips[index - 1]
        final = final_text_of(last)
        if final:
            seconds = float(last.get("duration", 15) or 15)
            parts.append(f"Previous clip, final prompt (clip {index}, {seconds:g}s, already rendered; "
                         f"this clip starts where its last shot ends):\n{final}")
            ending = last_shot_of(final)
            if ending:
                parts.append(f"Where clip {index} ends (its last shot; this clip opens from exactly this moment: same places, "
                             f"positions, wardrobe, props and light):\n{ending}")
        return "\n\n".join(parts)
    return "\n".join(_raw_ask_lines(clips, index))


def colon_labels(text: str, names) -> str:
    """Give bare section-label lines the official ``label:`` form.

    The Studio builder prompt lays its six sections out as a label alone on its own line
    (``subject_definitions``), while the official format and the pack's field parser want
    ``subject_definitions:``. Only a line that is exactly one of ``names`` (plus markdown
    decoration) is touched; prose that merely mentions a label is left alone.
    """
    if not text or not names:
        return text
    pattern = re.compile(r"^([ \t]*[*_#>\-]*[ \t]*)(" + "|".join(re.escape(n) for n in names) + r")([*_ \t]*)$",
                         re.IGNORECASE | re.MULTILINE)
    return pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3).rstrip()}:", text)


def with_previous(user_prompt: str, previous: str, story: str = "") -> str:
    """Insert 'story:' then 'previous_clips:' blocks ahead of the original prompt in the task message."""
    story = (story or "").strip()
    if not previous and not story:
        return user_prompt
    block = (f"story:\n{story}\n" if story else "") + (f"previous_clips:\n{previous}\n" if previous else "")
    marker = "original_prompt:"
    at = user_prompt.rfind(marker)
    if at < 0:
        return f"{user_prompt.rstrip()}\n{block}"
    return user_prompt[:at] + block + user_prompt[at:]


def choose_system(given: str, task: str) -> tuple[str, str]:
    """The writer's system prompt: ``(text, name)``; empty text means MiniMax's official guide.

    A socket or widget text wins; ``@official`` selects the official guide; empty means the Studio builder
    (``prompts/builder.md``) for Ref2VA. The builder is written for Ref2VA (``<Subject N>`` / ``<Picture N>``
    labels and its six sections), so T2VA keeps the official guide.
    """
    given = (given or "").strip()
    if given.lower() == story_planner.OFFICIAL_SENTINEL:
        return "", "MiniMax's official writing guide (chosen with @official)"
    if given:
        return given, "the custom system prompt"
    if task == "Ref2VA":
        try:
            return story_planner.load_prompt("builder"), "the Studio builder prompt (default for Ref2VA)"
        except OSError:
            return "", "MiniMax's official writing guide (prompts/builder.md not found)"
    return "", "MiniMax's official writing guide (the builder is written for Ref2VA; this run is " + task + ")"


def pending_indices(clips: list, mode: str, current: dict | None = None, continuity: str = RAW_ASKS) -> list[int]:
    """Which clips this run rewrites.

    ``all clips``: every clip with text. ``pending clips``: clips never
    rewritten, marked pending by the panel, or whose inputs changed since (the
    stored fingerprint in ``rewrite_meta`` differs from ``current[index]``); a
    rewrite from before fingerprints existed counts as out of date once.

    With ``final prompts`` continuity a rewritten clip is also out of date when
    the final prompt of the clip before it differs from the one it was written
    from (``rewrite_meta["prev_final"]``), or when that clip is rewritten in this
    same run. A clip with no ``prev_final`` (written under another continuity
    setting) is never stale for that reason: switching the setting keeps it.
    """
    wanted = []
    chosen = set()
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            continue
        source = source_of(clip)
        if not source.strip():
            continue
        if mode == "all clips" or not clip.get("prompt_rewritten"):
            wanted.append(index)
            chosen.add(index)
            continue
        meta = clip.get("rewrite_meta") or {}
        stale = current is not None and meta.get("fingerprint") != current.get(index)
        if not stale and continuity == FINAL_PROMPTS and index > 0 and meta.get("prev_final"):
            stale = (index - 1) in chosen or meta["prev_final"] != final_hash(clips[index - 1])
        if stale:
            wanted.append(index)
            chosen.add(index)
    return wanted


# --------------------------------------------------------------------------- main


def rewrite_clips(clips: list, refs: dict, settings: dict, *, aspect_text: str, progress_cb=None, stream_cb=None,
                  videos: dict | None = None, video_audios: dict | None = None, audios: dict | None = None,
                  plan_cb=None) -> list[str]:
    """Rewrite the pending clips in place. Returns human-readable notes.

    ``settings`` is the extender's kwargs (the ``rewrite_*`` widgets and the
    ``rewrite_system_prompt_in`` socket). ``refs`` is the extender's reference
    dict (``ref_image_N`` -> IMAGE). ``aspect_text`` is the target frame size,
    e.g. ``"1280x736"``; the pack turns it into the guide's aspect label.
    ``stream_cb(index, clip_id, phase, text, source)`` is called while a clip is being
    written -- phase ``thinking`` / ``writing`` with the text so far, then
    ``done`` with the final answer -- so the panel can show the rewrite live.
    ``videos`` (``ref_video_N`` -> 24 fps frame batch) are described from a few
    sampled frames as <Video k>; ``video_audios`` / ``audios`` cannot be heard by a
    vision-only captioner, so they enter the reference block as labelled but
    undescribed <Audio j> lines, numbered the way the core node numbers them.
    ``plan_cb(clips, positions)`` is called once when ``auto_clips`` planned the story into the clip list,
    so the panel can show the planned asks.
    """
    mode = str(settings.get("rewrite_mode", "off"))
    if mode == "off":
        return []
    if not available():
        raise RuntimeError(
            "rewrite_mode is on, but the MiniMax-H3-Prompt-Rewriter-ComfyUI pack is not installed "
            "beside this node. Install it (or set rewrite_mode to off)."
        )

    def say(stage, message, pct=0.0):
        _LOG.info("Rewriter: %s", message)
        if progress_cb is not None:
            try:
                progress_cb(stage, message, pct)
            except Exception:
                pass

    nodes = _mod("nodes")
    paths = _mod("paths")
    guides = _mod("guides")
    guide_prompt = _mod("guide_prompt")
    fields = _mod("fields")
    checks = _mod("checks")
    mtmd = _mod("mtmd_engine")
    aspect = _mod("aspect")
    constants = _mod("constants")

    writer_label = str(settings.get("rewrite_writer_model", ""))
    length = str(settings.get("rewrite_caption_length", "standard"))
    greedy = bool(settings.get("rewrite_greedy", True))
    temperature = float(settings.get("rewrite_temperature", 0.7))
    max_new_tokens = int(settings.get("rewrite_max_new_tokens", 4096))
    seed = int(settings.get("rewrite_seed", 42))
    thinking = bool(settings.get("rewrite_thinking", False))
    budget = int(reasoning_budget(settings.get("rewrite_reasoning_budget", "4096")))
    budget_message = str(settings.get("rewrite_reasoning_budget_message", "") or "")
    slots = max(1, int(settings.get("rewrite_parallel", 3)))
    continuity = str(settings.get("rewrite_previous_clips", "raw asks"))
    if continuity not in CONTINUITY:
        continuity = RAW_ASKS
    chained = continuity == FINAL_PROMPTS
    story = str(settings.get("rewrite_story") or "").strip()
    system_given = str(settings.get("rewrite_system_prompt_in") or "").strip() or str(settings.get("rewrite_system_prompt") or "").strip()
    auto_clips = max(0, int(settings.get("auto_clips", 0) or 0))
    clip_seconds = story_planner.clip_seconds(settings.get("auto_clip_seconds", story_planner.CLIP_SECONDS))
    planner_refs = str(settings.get("planner_refs", "images"))
    if planner_refs not in PLANNER_REFS:
        planner_refs = "images"
    story_engine = e4_engine.normalize_engine(settings.get("story_engine"))

    if writer_label.startswith("(") or not writer_label:
        raise RuntimeError("rewrite_writer_model: pick a GGUF from the list (the rewriter pack's model list).")

    strata = writer_label == strata_backend.LABEL
    writer_file = "" if strata else _writer_file(nodes, paths, writer_label)
    ordered = _ordered_refs(refs)
    ordered_videos = _ordered(videos, "ref_video_")
    ordered_audios = _ordered(audios, "ref_audio_")
    soundtracks = {slot for slot, _value in _ordered(video_audios, "ref_video_audio_")}
    any_refs = bool(ordered or ordered_videos or ordered_audios)
    task = str(settings.get("rewrite_task", "auto"))
    if task == "auto":
        task = "Ref2VA" if any_refs else "T2VA"
    if task == "Ref2VA" and not any_refs:
        say("rewrite", "no references connected, writing T2VA instead of Ref2VA")
        task = "T2VA"

    system_given, system_name = choose_system(system_given, task)
    _LOG.info("Rewriter: system prompt = %s", system_name)

    # One model writes, and looks at the pictures itself: they ride in the planner call and in every clip's writer call.
    # It sees when the pack lists it as a captioner (it ships an mmproj, or is a NInfer artifact with its own tower),
    # or, for Strata, when its config has the vision section and the --vision flag. Any other writer does not: the
    # references are then labelled but not described.
    model_path, mmproj_path = ("", "") if strata else _vision_files(nodes, paths, writer_label)
    sees = strata_backend.vision_available() if strata else bool(model_path)
    if not sees:
        model_path = writer_file
    can_caption = sees and not strata
    if task != "T2VA" and not sees and (ordered or ordered_videos):
        say("rewrite", "writer has no vision: reference images not interpreted (no captions, the rewrite is text-only)")

    writer_on_server = sees and not strata
    if writer_file.lower().endswith(".ninfer") and not writer_on_server:
        raise RuntimeError(
            "rewrite_writer_model: a NInfer model only runs on ninfer-serve, and this entry has no vision "
            "entry in the rewriter pack's model list to run it through. Pick another writer."
        )
    on_server = writer_on_server or strata
    if thinking and not on_server:
        say("rewrite", "thinking needs a writer with vision (one server session) or Strata; writing without it")
    reasoning = {"enabled": thinking and on_server, "budget": budget, "message": budget_message}
    writer_budget = max_new_tokens + (max(budget, 0) if reasoning["enabled"] else 0)
    resolution = aspect.resolve(aspect_text, "16:9")

    pictures = _reference_pictures(ordered) if task == "Ref2VA" and sees else []
    picture_tokens = sum(item["tokens"] for item in pictures)
    if pictures:
        _LOG.info("Rewriter: %d reference picture(s) go to the writer as images (no caption step), sent at %s, ~%d image tokens",
                  len(pictures), ", ".join(f"{item['label']} {item['size'][0]}x{item['size'][1]}" for item in pictures),
                  picture_tokens)

    # ---- what is out of date --------------------------------------------------
    described = task != "T2VA" and can_caption
    video_keys = [(slot, _video_key(frames, model_path, length)) for slot, frames in ordered_videos] if described else []
    plan_slots = story_planner.plan_slots(clips, auto_clips) if (auto_clips and story) else []
    # plan_ctx is None for a first plan with nothing typed (the whole story into N). Otherwise the planner sees the
    # existing clips: typed ones around the empty slots of a first plan, or every clip when auto_clips is raised past a
    # film that was planned once (only the new clips are planned, after the existing ones).
    plan_ctx = story_planner.plan_around(clips, auto_clips, plan_slots) if plan_slots else None
    if not plan_slots and auto_clips and story:
        # An e4 film has the number of clips the engine chose, not auto_clips (which only switches planning on): a list
        # it wrote is complete, so a reload of its saved workflow must not plan the difference as 'more' clips.
        written_by_e4 = e4_engine.is_e4(story_engine) and any(
            isinstance(c, dict) and isinstance(c.get("rewrite_meta"), dict) and c["rewrite_meta"].get("engine") == "e4" for c in clips)
        if not written_by_e4:
            plan_ctx = story_planner.plan_more(clips, auto_clips)
    planning = bool(plan_slots) or bool(plan_ctx)
    plan_n = len(plan_ctx["numbers"]) if plan_ctx else auto_clips
    # story_engine=e4.10 / e4.11 plans and writes a whole film in one run: only a first plan of a clip list that holds no text qualifies.
    e4_plan, e4_why = False, ""
    if e4_engine.is_e4(story_engine) and planning:
        if plan_ctx or any(story_planner._ask_text(c).strip() for c in clips if isinstance(c, dict)):
            e4_why = "clips are already typed or planned ('Plan more' and typed clips are not supported by the e4 engine in this version)"
        elif task != "Ref2VA" or not ordered:
            e4_why = "no reference picture is connected (e4 writes Ref2VA prompts)"
        elif not on_server:
            raise RuntimeError(
                f"story_engine={story_engine} needs a rewriter model that runs on a server (a vision model from the pack's list, or Strata); "
                "this writer runs in-process. Pick another writer, or set story_engine to builder.")
        else:
            e4_plan = True
        if not e4_plan:
            say("rewrite", f"story_engine={story_engine} is not used for this run: {e4_why}; planning with the builder planner", 0.05)
    if e4_plan:
        reasoning = dict(reasoning, enabled=True, budget=e4_engine.PLAN_BUDGET, message=reasoning["message"] or e4_engine.BUDGET_MESSAGE)
        writer_budget = max_new_tokens + e4_engine.PLAN_BUDGET
        if clip_seconds != story_planner.CLIP_SECONDS:
            say("rewrite", f"e4 plans {story_planner.CLIP_SECONDS} s clips (auto_clip_seconds {clip_seconds} is not used by this engine)", 0.05)
        if auto_clips:
            say("rewrite", f"e4 decides the number of clips from the story (auto_clips {auto_clips} only switches planning on)", 0.05)
    sizing_clips = clips
    if planning:
        sizing_clips = copy.deepcopy(clips)
        if plan_ctx:
            placeholders, _none, sizing_slots = story_planner.place(["x" * 1500] * plan_n, [], plan_ctx["numbers"])
            story_planner.apply_plan(sizing_clips, placeholders, sizing_slots, seconds=clip_seconds)
        else:
            story_planner.apply_plan(sizing_clips, ["x" * 1500] * auto_clips, plan_slots, seconds=clip_seconds)

    def assess(cl: list) -> tuple[dict, list[int]]:
        current = {}
        for index, clip in enumerate(cl):
            if not isinstance(clip, dict):
                continue
            duration = float(clip.get("duration", 15) or 15)
            current[index] = fingerprint(source_of(clip), duration, resolution, task,
                                         previous=continuity_block(cl, index, "raw asks"))
            meta = clip.get("rewrite_meta") or {}
            stored = str(meta.get("fingerprint") or "")
            # A rewrite stored before v2 fingerprints is kept while its raw ask is unchanged.
            if clip.get("prompt_rewritten") and stored and not stored.startswith("v2:") \
                    and (clip.get("prompt_raw") or "").strip() == source_of(clip).strip():
                clip["rewrite_meta"] = dict(meta, fingerprint=current[index])
        wanted = pending_indices(cl, mode, current, continuity)
        if chained:
            # Clips written under another setting get their baseline now, so a later
            # change to the previous final prompt is noticed; switching never invalidates.
            for index, clip in enumerate(cl):
                meta = clip.get("rewrite_meta") if isinstance(clip, dict) else None
                if index > 0 and index not in wanted and clip.get("prompt_rewritten") and isinstance(meta, dict) \
                        and not meta.get("prev_final"):
                    clip["rewrite_meta"] = dict(meta, prev_final=final_hash(cl[index - 1]))
        return current, wanted

    def log_reasons(cl: list, wanted: list[int]) -> None:
        reasons = []
        for index in wanted:
            clip = cl[index]
            if not clip.get("prompt_rewritten"):
                reasons.append(f"clip {index + 1}: pending")
            elif mode == "all clips":
                reasons.append(f"clip {index + 1}: all clips")
            else:
                reasons.append(f"clip {index + 1}: inputs changed")
        _LOG.info("Rewriter: %s", "; ".join(reasons))

    current, todo = assess(sizing_clips)
    if not todo and not planning:
        _LOG.info("Rewriter: nothing to do (every clip prompt is rewritten, up to date, or empty)")
        return ["rewriter: nothing pending"]
    if not planning:
        log_reasons(clips, todo)

    guide = "" if system_given else guides.text(guide_prompt.GUIDE_FOR_MODE[task], True, None)

    # ---- captions -------------------------------------------------------------
    cache = _load_cache()
    video_captions: dict[int, str] = {}
    videos_to_describe: list[tuple[int, object, str]] = []
    if described:
        for (slot, frames), (_slot, key) in zip(ordered_videos, video_keys):
            hit = cache.get(key)
            if isinstance(hit, dict) and hit.get("caption"):
                video_captions[slot] = hit["caption"]
            else:
                videos_to_describe.append((slot, frames, key))

    # The KV pool must hold the parallel writers (guide + block + answer each).
    stand_in = "" if task == "T2VA" else "x" * (1600 * max(len(ordered) + len(ordered_videos) + len(ordered_audios), 1))
    longest = max((source_of(sizing_clips[i]) for i in todo), key=len)
    rough = guide_prompt.build_messages(guide, task, longest, resolution, 15.0, stand_in, system=system_given)
    if chained and len(clips) > 1:
        # The previous clip's final prompt is a full six-section answer: size for one
        # as long as the writer may produce, not for the raw ask it is today.
        tail = continuity_block(clips, len(clips) - 1, FINAL_PROMPTS)
        pad = max(0, max_new_tokens * 4 - len(final_text_of(clips[-2])))
        rough[1]["content"] += "\n" + tail + "x" * pad + CONTINUITY_RULE_FINAL
    elif continuity != "off" and len(clips) > 1:
        rough[1]["content"] += "\n" + continuity_block(clips, len(clips) - 1, RAW_ASKS)
    rough[0]["content"] += system_rules(continuity=continuity, chained=chained, later_clips=len(clips) > 1, story=bool(story),
                                        task=task, sees=sees, state=True)
    if story:
        rough[1]["content"] += "\nstory:\n" + story
    writer_ctx = guide_prompt.context_needed(rough, writer_budget + picture_tokens)
    writers_at_once = min(len(todo), slots) if writer_on_server and not chained else 1
    pool_ctx = writer_ctx * writers_at_once if writer_on_server else 0
    plan_tokens = max(max_new_tokens, 900 * plan_n + 1500) if planning else 0
    plan_budget = plan_tokens + (max(budget, 0) if reasoning["enabled"] else 0)
    # What the planner is shown of the references: the pictures themselves (the same data URIs the clip writers get),
    # and the captions of reference videos. planner_refs=captions is kept for old workflows and now means images.
    plan_refs_mode = "off"
    if planning and task != "T2VA" and planner_refs != "off":
        plan_refs_mode = "images" if pictures else ("captions" if (ordered_videos and described) else "off")
    plan_image_tokens = picture_tokens if plan_refs_mode == "images" else 0
    plan_ctx_tokens = 0
    if planning and writer_on_server:
        stand = None
        if plan_refs_mode != "off":
            stand = {"pictures": [{"label": f"Picture {slot + 1}", "caption": "", "image": None} for slot, _t in ordered],
                     "videos": [{"label": f"Video {k}", "caption": "x" * 1600} for k, _v in enumerate(ordered_videos, start=1)]}
        plan_ctx_tokens = guide_prompt.context_needed(
            [{"role": "user", "content": story_planner.build_user_content(story, plan_n, stand, plan_ctx, clip_seconds)}],
            plan_budget + plan_image_tokens)
    caption_jobs = len(videos_to_describe)
    caption_slots = max(1, min(caption_jobs, slots)) if caption_jobs else 1
    server_slots = max(caption_slots, writers_at_once)
    pool_ctx = max(pool_ctx, plan_ctx_tokens * server_slots)
    e4_workers = 1
    if e4_plan:
        e4_workers = max(1, min(e4_engine.MAX_WORKERS, slots))
        server_slots = max(server_slots, e4_workers)
        pool_ctx = max(pool_ctx, e4_engine.slot_ctx() * server_slots)

    started = time.time()
    notes: list[str] = []
    open_session = caption_jobs > 0 or writer_on_server
    if open_session:
        need_slot_ctx = max(writer_ctx, plan_ctx_tokens, e4_engine.slot_ctx() if e4_plan else 0)

        def launch():
            stack = contextlib.ExitStack()
            try:
                opened = stack.enter_context(mtmd.session(
                    model_path, mmproj_path,
                    assets=caption_jobs, attachments=max(VIDEO_FRAMES if videos_to_describe else 1, len(pictures)),
                    gpu_layers=-1, n_ctx=0, device="auto", backend="auto", auto_download=True,
                    progress=None, slots=server_slots, force=writer_on_server,
                    reasoning=reasoning if writer_on_server else None, pool_ctx=pool_ctx,
                ))
            except BaseException:
                stack.close()
                raise
            return opened, stack.close, {}

        def fits(meta):
            """A server the Shot Writer left running is used as launched: ninfer fixes the context per slot, the thinking budget and its message at start."""
            if meta.get("slot_ctx", 0) < need_slot_ctx:
                return False
            if writer_on_server and reasoning["enabled"] and not e4_plan:
                had = meta.get("reasoning") or {}
                if not had.get("enabled") or int(had.get("budget", -1)) != budget or (budget_message and str(had.get("message") or "") != budget_message.strip()):
                    return False
            return True

        session = shared_server.use(shared_server.key_of(model_path, mmproj_path), launch, fits=fits, say=lambda message: say("rewrite", message, 0.1))
    else:
        session = contextlib.nullcontext(None)

    with contextlib.ExitStack() as stack:
        server = stack.enter_context(session)
        have = getattr(server, "slots", None)
        if isinstance(have, int) and have > 0:   # an attached server may have fewer slots than this run would use
            writers_at_once, caption_slots, e4_workers = (max(1, min(n, have)) for n in (writers_at_once, caption_slots, e4_workers))
        if videos_to_describe:
            video_question = nodes.caption_question("Video", length)
            say("rewrite", f"describing {len(videos_to_describe)} reference video(s) from {VIDEO_FRAMES} sampled frames each", 0.15)

            def describe_video(item):
                slot, frames, key = item
                count = int(frames.shape[0]) if frames.dim() == 4 else 1
                instruction = f"{mtmd.clip_note(min(count, VIDEO_FRAMES), count / VIDEO_FPS)}\n\n{video_question}"
                text = mtmd.describe(
                    model_path=model_path, mmproj_path=mmproj_path, instruction=instruction,
                    image=frames, max_frames=VIDEO_FRAMES, gpu_layers=-1, n_ctx=0, seed=seed, greedy=True,
                    max_new_tokens=CAPTION_TOKENS, temperature=0.7, top_p=0.8, top_k=20,
                    device="auto", backend="auto", auto_download=True, progress=None, server=server,
                )
                return slot, key, " ".join((text or "").split())

            if server is not None and caption_slots > 1 and len(videos_to_describe) > 1:
                with ThreadPoolExecutor(max_workers=caption_slots) as pool:
                    video_results = list(pool.map(describe_video, videos_to_describe))
            else:
                video_results = [describe_video(item) for item in videos_to_describe]
            for slot, key, caption in video_results:
                video_captions[slot] = caption
                if caption:
                    cache[key] = {"caption": caption, "model": os.path.basename(model_path), "length": length, "kind": "video", "at": time.time()}
                else:
                    notes.append(f"Video {slot + 1}: empty caption")
            _save_cache(cache)

        if strata:
            say("rewrite", "starting Strata" + (f" (thinking budget {budget})" if reasoning["enabled"] else ""), 0.18)
            server = stack.enter_context(strata_backend.open_strata(
                budget, adopt=getattr(_mod("runner"), "_adopt", None),
                on_wait=lambda seconds: say("rewrite", f"waiting for Strata to load ({seconds:.0f} s)", 0.18)))

        # The reference block in the core node's label order: pictures, then each
        # video (its soundtrack's <Audio j> right before it), then standalone audio.
        lines = []
        if task != "T2VA":
            for slot, _tensor in ordered:
                lines.append(f"Picture {slot + 1}: the attached image labelled 'Picture {slot + 1}:' at the top of this message" if pictures
                             else f"Picture {slot + 1}: an attached reference picture (not described here)")
            audio_no = 0
            for k, (slot, _frames) in enumerate(ordered_videos, start=1):
                if slot in soundtracks:
                    audio_no += 1
                    lines.append(f"Audio {audio_no}: the soundtrack of Video {k} (voice and sound to reuse; not described here)")
                lines.append(f"Video {k}: {video_captions[slot]}" if video_captions.get(slot)
                             else f"Video {k}: an attached reference video (not described here)")
            for _slot, _audio in ordered_audios:
                audio_no += 1
                lines.append(f"Audio {audio_no}: an attached audio reference (voice or sound to reuse; not described here)")
        block = "\n".join(lines)

        pack_settings = dict(nodes.DEFAULT_OPTIONS)
        pack_settings.update(max_new_tokens=max_new_tokens, temperature=temperature)
        names = guide_prompt.FIELDS_FOR_MODE[task]

        # ---- planning with E4: the whole film, planned AND written, in one subprocess ---------------------
        if e4_plan:
            say("rewrite", f"story_engine={story_engine}: planning and writing the film with {e4_engine.engine_name(story_engine)} (planner, bible and writer think {e4_engine.PLAN_BUDGET} tokens at most, the per-clip decisions {e4_engine.decision_label(e4_engine.decision_budget(settings.get('e4_decision_budget', e4_engine.DECISION_BUDGET_DEFAULT)))})", 0.2)
            endpoint = e4_engine.endpoint_of(server, model_path=model_path, strata=strata, is_ninfer=_mod("server_engine").is_ninfer)
            labels = [slot + 1 for slot, _t in ordered]
            by_label = {int(item["slot"]) + 1: item["image"] for item in pictures}
            interrupted = getattr(_mod("runner"), "interrupted", None)

            def e4_say(message, fraction):
                say("rewrite", message, 0.2 + 0.7 * max(0.0, min(1.0, fraction)))

            t_e4 = time.time()
            e4_failed = None
            try:
                result = e4_engine.plan_film(
                    story=story, language=str(settings.get("e4_language") or ""), score=str(settings.get("e4_score", "off")), labels=labels, pictures=by_label,
                    notes=e4_engine.parse_notes(settings.get("e4_picture_notes")), sees=sees, endpoint=endpoint,
                    decision=e4_engine.decision_budget(settings.get("e4_decision_budget", e4_engine.DECISION_BUDGET_DEFAULT)), workers=e4_workers,
                    say=e4_say, interrupted=interrupted, engine=story_engine)
            except (e4_engine.E4Interrupted, e4_engine.E4SetupError):
                raise
            except e4_engine.E4Error as error:
                e4_failed = error
            if e4_failed is not None:
                cause = e4_failed.cause or str(e4_failed).splitlines()[0]
                kept = e4_failed.run or "the e4 run folder"
                warning = f"story_engine={story_engine} FAILED and is NOT used for this run: {cause} (run kept in {kept}); planning with the builder planner instead"
                _LOG.warning("Rewriter: %s\n%s", warning, e4_failed)
                say("rewrite", "WARNING: " + warning, 0.2)
                notes.append("WARNING: " + warning)
                e4_plan = False
                plan_budget = plan_tokens + e4_engine.PLAN_BUDGET
            else:
                positions = e4_engine.apply_clips(clips, result, model=os.path.basename(writer_file or writer_label))
                current, _pending = assess(clips)
                for position in positions:
                    meta = clips[position].get("rewrite_meta")
                    if isinstance(meta, dict) and "fingerprint" in meta:
                        meta["fingerprint"] = current.get(position)
                for position in positions:
                    _LOG.info("Rewriter plan clip %d (e4, %s s):\n%s", position + 1, clips[position].get("duration"), clips[position].get("prompt_raw") or clips[position].get("prompt"))
                mapping = result.get("mapping") or {}
                bound = [f"Picture {p['picture']} = {p['entity']}" for p in mapping.get("pictures", []) if p.get("entity")]
                _LOG.info("Rewriter: e4 bound the pictures (%s): %s; unused: %s", mapping.get("mode"), ", ".join(bound) or "none", mapping.get("unused") or "none")
                if mapping.get("issues"):
                    _LOG.warning("Rewriter: e4 picture binding: %s", " | ".join(mapping["issues"]))
                written_clips = [c for c in result["clips"] if c.get("prompt")]
                for position in positions:
                    clip = clips[position]
                    if clip.get("prompt_rewritten") and stream_cb is not None:
                        try:
                            stream_cb(position, clip.get("id"), "done", clip["prompt"], clip.get("prompt_raw") or "")
                        except Exception:
                            _LOG.debug("stream callback failed", exc_info=True)
                notes.append(f"e4: planned and wrote {len(written_clips)} of {len(result['clips'])} clip(s) from the story in {time.time() - t_e4:.0f} s"
                             + (f", {len(result['clips']) - len(written_clips)} left for the builder" if len(written_clips) < len(result["clips"]) else "")
                             + f" (run kept in {result['run_dir']})")
                if len(clips) > len(positions):
                    notes.append(f"e4: {len(clips) - len(positions)} empty clip(s) are left after the film's {len(positions)}; delete them")
                if plan_cb is not None:
                    try:
                        plan_cb(clips, positions)
                    except Exception:
                        _LOG.debug("plan callback failed", exc_info=True)
                planning = False
                current, todo = assess(clips)
                if not todo:
                    return notes
                log_reasons(clips, todo)

        # ---- planning: the film story into the clip list, once ----------------
        if planning:
            say("rewrite", (f"planning {plan_n} clip(s) ({story_planner._spans(plan_ctx['numbers'])}) around the {len([1 for _n, t in plan_ctx['existing'] if t])} existing clip(s)" if plan_ctx else f"planning the story into {auto_clips} clip(s) of {clip_seconds} s")
                + (f" with thinking (budget {budget})" if reasoning["enabled"] else ""), 0.2)

            def plan_chat(messages):
                if server is not None and on_server:
                    text = server.chat(
                        messages, seed=seed, greedy=greedy, max_new_tokens=plan_budget,
                        temperature=temperature, top_p=0.8, top_k=20, repeat_penalty=1.05,
                        enable_thinking=reasoning["enabled"],
                        on_text=lambda whole: bool(checks.looping(whole)),
                    )
                else:
                    plan_settings = dict(pack_settings, max_new_tokens=plan_tokens)
                    progress = _mod("progress").NodeProgress(None)
                    text = nodes.run_messages(writer_label, messages, greedy, seed, False, plan_settings, progress, label=task)
                return constants.answer_only((text or "").replace("\r\n", "\n")).strip()

            plan_refs = None
            if plan_refs_mode != "off":
                plan_refs = {
                    "pictures": pictures if plan_refs_mode == "images" else [],
                    "videos": [{"label": f"Video {k}", "caption": video_captions.get(slot)
                                or f"an attached reference video (not described here)"}
                               for k, (slot, _f) in enumerate(ordered_videos, start=1)],
                }
                _LOG.info("Rewriter: planner sees the references as %s (%d picture(s), %d video(s), ~%d image tokens)",
                          plan_refs_mode, len(plan_refs["pictures"]), len(ordered_videos), plan_image_tokens)
            t_plan = time.time()
            plan_states: list = []
            asks = story_planner.plan_story(plan_chat, story, plan_n, refs=plan_refs, more=plan_ctx, states_out=plan_states, seconds=clip_seconds,
                                            log=lambda m: _LOG.info("Rewriter planner: %s", m))
            if plan_ctx:
                full_asks, full_states, apply_slots = story_planner.place(asks, plan_states, plan_ctx["numbers"])
                written_slots = story_planner.apply_plan(clips, full_asks, apply_slots, states=full_states, seconds=clip_seconds)
                plan_numbers = plan_ctx["numbers"][:len(asks)]
            else:
                written_slots = story_planner.apply_plan(clips, asks, plan_slots, states=plan_states, seconds=clip_seconds)
                plan_numbers = list(range(1, len(asks) + 1))
            _LOG.info("Rewriter: planned %d clip(s) in %.1f s; wrote %d into the clip list (typed clips untouched)",
                      len(asks), time.time() - t_plan, len(written_slots))
            for n_, ask in zip(plan_numbers, asks):
                _LOG.info("Rewriter plan clip %d:\n%s", n_, ask)
            notes.append(f"planned {len(asks)} clip(s) from the story, wrote {len(written_slots)} new ask(s)")
            if plan_cb is not None:
                try:
                    plan_cb(clips, written_slots)
                except Exception:
                    _LOG.debug("plan callback failed", exc_info=True)
            current, todo = assess(clips)
            if not todo:
                return notes
            log_reasons(clips, todo)

        # ---- writing ----------------------------------------------------------
        say("rewrite", f"writing {len(todo)} clip prompt(s) as {task}"
            + (f" with thinking (budget {budget})" if reasoning["enabled"] else "")
            + (f", {writers_at_once} at once" if writers_at_once > 1 else "")
            + (", one after another (continuity 'final prompts': each clip continues from the previous clip's final prompt)"
               if chained else ""), 0.3)

        marks = (story_planner.OPENS_MARK[:20], story_planner.ENDS_MARK[:20])
        system_tail = system_rules(
            continuity=continuity, chained=chained, later_clips=len(clips) > 1, story=bool(story), task=task, sees=sees,
            state=any(mark in source_of(clip) for clip in clips if isinstance(clip, dict) for mark in marks))

        def write_one(index: int) -> tuple[int, str, str, float]:
            clip = clips[index]
            clip_id = clip.get("id")
            source = source_of(clip)
            duration = float(clip.get("duration", 15) or 15)
            messages = guide_prompt.build_messages(guide, task, source, resolution, duration, block, system=system_given)
            previous = continuity_block(clips, index, continuity)
            messages[0]["content"] = messages[0]["content"].rstrip() + system_tail
            if previous or story:
                messages[1]["content"] = with_previous(messages[1]["content"], previous, story)
            if pictures:
                messages[1] = dict(messages[1], content=[{"type": "text", "text": PICTURES_HEADER}]
                                   + story_planner.picture_parts(pictures)
                                   + [{"type": "text", "text": messages[1]["content"]}])

            last = {"at": 0.0}

            def stream(phase, text, force=False):
                if stream_cb is None:
                    return
                now = time.time()
                if not force and now - last["at"] < 0.3:
                    return
                last["at"] = now
                try:
                    stream_cb(index, clip_id, phase, text, source)
                except Exception:
                    _LOG.debug("stream callback failed", exc_info=True)

            def on_text(whole):
                stream("writing", whole)
                return bool(checks.looping(whole))

            def on_reasoning(whole):
                stream("thinking", whole)

            stream("thinking" if reasoning["enabled"] else "writing", "", force=True)
            t0 = time.time()
            if server is not None and on_server:
                text = server.chat(
                    messages, seed=seed, greedy=greedy, max_new_tokens=writer_budget,
                    temperature=temperature, top_p=0.8, top_k=20, repeat_penalty=1.05,
                    enable_thinking=reasoning["enabled"],
                    on_text=on_text,
                    on_reasoning=on_reasoning if reasoning["enabled"] else None,
                )
            else:
                progress = _mod("progress").NodeProgress(None)
                text = nodes.run_messages(writer_label, messages, greedy, seed, False, pack_settings, progress, label=task)
            text = constants.answer_only((text or "").replace("\r\n", "\n")).strip()
            stream("done", text, force=True)
            return index, source, text, time.time() - t0

        def apply(result) -> None:
            index, source, text, seconds = result
            clip = clips[index]
            if not text:
                notes.append(f"Clip {index + 1}: rewriter returned nothing, prompt left as typed")
                return
            text = colon_labels(text, names)
            sections = fields.split_fields(text, names)
            missing = fields.missing(sections, names)
            if missing:
                notes.append(f"Clip {index + 1}: missing {', '.join(missing)}")
            clip["prompt_raw"] = source
            clip["prompt"] = text
            clip["rewrite_text"] = text
            clip["prompt_rewritten"] = True
            clip["validated"] = False
            meta = {
                "model": os.path.basename(writer_file or writer_label),
                "task": task,
                "thinking": bool(reasoning["enabled"]),
                "budget": budget if reasoning["enabled"] else 0,
                "seconds": round(seconds, 1),
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fingerprint": current.get(index),
            }
            if chained and index > 0:
                meta["prev_final"] = final_hash(clips[index - 1])
            clip["rewrite_meta"] = meta
            _LOG.info("Rewriter: clip %d rewritten in %.1f s (%d chars)", index + 1, seconds, len(text))

        if chained:
            written = []
            for index in todo:
                result = write_one(index)
                apply(result)
                written.append(result)
        else:
            if server is not None and writer_on_server and writers_at_once > 1:
                with ThreadPoolExecutor(max_workers=writers_at_once) as pool:
                    written = list(pool.map(write_one, todo))
            else:
                written = [write_one(index) for index in todo]
            for result in written:
                apply(result)
        usage = getattr(server, "last_usage", None)
        if pictures and isinstance(usage, dict) and usage.get("prompt_tokens"):
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            _LOG.info("Rewriter: the server reported %s prompt tokens on the last call (%s of them cached), with %d picture(s) "
                      "~%d image tokens", usage["prompt_tokens"], cached, len(pictures), picture_tokens)

    total = time.time() - started
    say("rewrite", f"rewrote {len(written)} clip(s) in {total:.0f} s", 1.0)
    notes.insert(0, f"rewriter: {len(written)} clip(s), {len(videos_to_describe)} new video caption(s), {total:.0f} s")
    return notes
