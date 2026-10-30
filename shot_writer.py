"""H3 Shot Writer: a free-form brief in, a timed shot list for every clip out, ready to feed the Master Extender's ``clips_in``.

A separate node from the Story Writer (it changes nothing there) and from the Master Extender's planner. The brief can be anything the founder types:
numbered clips ("Clip 1 - ..."), ranges ("3-6 - Battle."), plain prose, "and so on", dialogue in any language. There is no parsing of it in code: the
prompts carry the interpretation. Calls go to the same local model server the Story Writer opens (``open_model``), closed again when the node finishes.

1. PLAN (one call; parts of ``PLAN_PART`` clips when a film is longer than one answer): title, summary, cast, locations and, per clip, ``goal``,
   ``end_state`` and the founder's own dialogue lines that belong in it. A wrong clip count is asked again once with the complaint; excess clips are cut.
2. SHOTS (one call per clip, ``concurrency`` at a time): each sees the cast, locations, its goal, the previous clip's goal and end state and the next
   clip's goal, and answers 2 to 6 timed shots (start second, framing, one short action, optional dialogue).
3. CODE CHECKS on every clip's shots: timing (first shot at 0, starts strictly increasing and below the clip length, every shot at least 2 s), dialogue
   in the target script, words that fit the shot, and the founder's own lines present verbatim. A clip that fails is asked again once with the specific
   complaint; whatever is best after that is kept and a warning logged.

Every reply is requested with a JSON schema (``response_format``) and also described in the prompt; it is parsed tolerantly all the same.
The pure engine (``ShotEngine``) takes any ``chat(messages, max_tokens=, budget=, kind=, temperature=, response_format=) -> Reply``. ComfyUI is imported
lazily, so the engine runs and is tested without it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from . import e4_engine
    from . import prompt_rewriter
    from . import story_planner
    from . import strata_backend
    from . import story_writer as sw
    from .story_writer import (BUDGET_MESSAGE, DEFAULT_CONCURRENCY, DEFAULT_LANGUAGE, DEFAULT_SECONDS, MAX_CLIPS, MAX_CONCURRENCY, THINKING_BUDGET,
                               MinorContentError, Reply, ServerChat, StoryWriterError, is_wide, output_room, parse_json_object, quoted_lines, script_note)
except ImportError:  # imported as a top-level module (tests)
    import e4_engine
    import prompt_rewriter
    import story_planner
    import strata_backend
    import story_writer as sw
    from story_writer import (BUDGET_MESSAGE, DEFAULT_CONCURRENCY, DEFAULT_LANGUAGE, DEFAULT_SECONDS, MAX_CLIPS, MAX_CONCURRENCY, THINKING_BUDGET,
                              MinorContentError, Reply, ServerChat, StoryWriterError, is_wide, output_room, parse_json_object, quoted_lines, script_note)

_LOG = logging.getLogger("minimax_h3_master_extender.shot_writer")

VERSION = "1"
MIN_CLIPS, MIN_SECONDS, MAX_SECONDS = 0, 4, 30
REF_SLOTS = 9             # the Extender's ref_image_1 .. ref_image_9
PLAN_PART = 20            # clips planned per call; a longer film is planned in parts
MIN_SHOT_SECONDS = 2
MIN_SHOTS, MAX_SHOTS = 2, 6
WIDE_WORDS_FACTOR = 0.75  # a word of a non-Latin language is longer than an English one
OUTPUT_SUBDIR = "h3_shot_writer"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# Unicode ranges of the script each language is written in. A language that is not Latin and not listed here is only checked for Latin letters.
SCRIPT_RANGES = {
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "bengali": [(0x0980, 0x09FF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "odia": [(0x0B00, 0x0B7F)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "cyrillic": [(0x0400, 0x04FF)],
    "greek": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "hebrew": [(0x0590, 0x05FF)],
    "thai": [(0x0E00, 0x0E7F)],
    "han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "kana": [(0x3040, 0x30FF), (0x31F0, 0x31FF)],
    "hangul": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
}
LANGUAGE_SCRIPTS = {
    "hindi": ["devanagari"], "marathi": ["devanagari"], "sanskrit": ["devanagari"], "nepali": ["devanagari"], "konkani": ["devanagari"],
    "bengali": ["bengali"], "bangla": ["bengali"], "assamese": ["bengali"], "punjabi": ["gurmukhi"], "gujarati": ["gujarati"], "odia": ["odia"],
    "tamil": ["tamil"], "telugu": ["telugu"], "kannada": ["kannada"], "malayalam": ["malayalam"],
    "arabic": ["arabic"], "urdu": ["arabic"], "persian": ["arabic"], "farsi": ["arabic"],
    "russian": ["cyrillic"], "ukrainian": ["cyrillic"], "bulgarian": ["cyrillic"], "serbian": ["cyrillic"],
    "greek": ["greek"], "hebrew": ["hebrew"], "thai": ["thai"],
    "chinese": ["han"], "mandarin": ["han"], "cantonese": ["han"], "japanese": ["han", "kana"], "korean": ["hangul", "han"],
}
# The founder's own lines are only recognised by script, so for a Latin-script film only the default language is enforced (an English line in a French
# film is a line to translate, and nothing in code can tell the two apart).
VERBATIM_LATIN_LANGUAGES = {"english"}


# --------------------------------------------------------------------------- text helpers


def normalize(text) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text or "")).split())


def script_ranges(language: str) -> list | None:
    names = LANGUAGE_SCRIPTS.get(str(language or DEFAULT_LANGUAGE).strip().lower())
    return [r for name in names for r in SCRIPT_RANGES[name]] if names else None


def foreign_letters(line: str, language: str) -> list:
    """Letters of ``line`` outside the script of ``language`` (nothing for a Latin-script language). Digits, punctuation and combining marks are never counted."""
    if not is_wide(language):
        return []
    ranges = script_ranges(language)
    bad = []
    for ch in str(line or ""):
        if not ch.isalpha():
            continue
        code = ord(ch)
        if ranges is None:
            if code < 0x250:   # an unlisted language: only Latin letters are certainly wrong
                bad.append(ch)
        elif not any(lo <= code <= hi for lo, hi in ranges):
            bad.append(ch)
    return bad


def max_line_words(shot_seconds: float, language: str) -> int:
    return max(2, int(shot_seconds * sw.SPEECH_WORDS_PER_SECOND * (WIDE_WORDS_FACTOR if is_wide(language) else 1.0)))


def founder_lines(brief: str, language: str) -> list:
    """The quoted strings of the brief that are already in the film's language, in order, without repeats: lines the writer must keep verbatim."""
    wide = is_wide(language)
    if not wide and str(language or DEFAULT_LANGUAGE).strip().lower() not in VERBATIM_LATIN_LANGUAGES:
        return []
    found, seen = [], set()
    for text in quoted_lines(brief):
        line = normalize(text)
        if not line or line in seen or not any(ch.isalpha() for ch in line):
            continue
        if wide and (foreign_letters(line, language) or not any(ch.isalpha() for ch in line)):
            continue   # written in another language: it is translated, not kept
        seen.add(line)
        found.append(line)
    return found


def refs_from_inputs(inputs: dict) -> dict:
    """The node's ``ref_image_N`` sockets (1-based, like the Extender's) as the Extender's reference dict: socket N is key ``ref_image_{N-1}``
    (master_node.extend), so ``prompt_rewriter._ordered_refs`` gives the very ``Picture N`` the rewriter and the text encoder use. Empty sockets are
    skipped and the rest are NOT renumbered: pictures on sockets 1, 2 and 4 are Picture 1, 2 and 4."""
    return {f"ref_image_{i - 1}": inputs[f"ref_image_{i}"] for i in range(1, REF_SLOTS + 1) if inputs.get(f"ref_image_{i}") is not None}


def pictures_of(inputs: dict) -> list:
    """The connected pictures as the planner and the clip writers send them (``prompt_rewriter._reference_pictures``: 896 px long side, PNG data URIs)."""
    return prompt_rewriter._reference_pictures(prompt_rewriter._ordered_refs(refs_from_inputs(inputs)))


def message(text: str, pictures: list) -> dict:
    """A user turn: the labelled pictures first (``story_planner.picture_parts``, the same layout the planner uses), then the text."""
    parts = story_planner.picture_parts(pictures)
    return {"role": "user", "content": parts + [{"type": "text", "text": text}] if parts else text}


def text_of(content) -> str:
    return content if isinstance(content, str) else "\n".join(p.get("text", "") for p in content if p.get("type") == "text")


def with_complaint(msg: dict, complaint: str) -> dict:
    if isinstance(msg["content"], str):
        return dict(msg, content=msg["content"] + complaint)
    parts = [dict(p) for p in msg["content"]]
    last = max(i for i, p in enumerate(parts) if p.get("type") == "text")
    parts[last]["text"] += complaint
    return dict(msg, content=parts)


def pictures_note(pictures: list, sees: bool) -> str:
    if not pictures:
        return ""
    labels = ", ".join(p["label"] for p in pictures)
    if not sees:
        return (f"REFERENCE PICTURES connected: {labels}. You cannot see them. The brief refers to them by number (\"picture 4\" is <Picture 4>): keep the references "
                f"as <Picture N> where the brief makes them, and do not invent what a picture shows.")
    return (f"REFERENCE PICTURES ({labels}) are attached above, each labelled right before its image. The brief refers to them by that number (\"picture 4\" is "
            f"<Picture 4>). Look at each: say who or what it shows (a character, a prop, a costume, a location) and use it for looks.")


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"shot_writer_{name}.md").read_text(encoding="utf-8").strip()


def fill(template: str, **values) -> str:
    return sw.fill(template, **values)


# --------------------------------------------------------------------------- JSON schemas (response_format) and parsing

def _named(*extra):
    keys = ["name", "look", *extra]
    return {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False, "required": keys, "properties": {k: {"type": "string"} for k in keys}}}


def _strings():
    return {"type": "array", "items": {"type": "string"}}


def _list_of(**fields):
    return {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": list(fields), "properties": fields}}


STATE_FIELDS = ["wardrobe", "props", "position", "state"]
PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["title", "summary", "minor_in_sexual_content", "start_place", "pictures", "cast", "locations", "clips"],
    "properties": {
        "title": {"type": "string"}, "summary": {"type": "string"}, "minor_in_sexual_content": {"type": "boolean"}, "start_place": {"type": "string"},
        "pictures": _list_of(label={"type": "string"}, shows={"type": "string"}),
        "cast": _named("wardrobe", "props"), "locations": _named(),
        "clips": {"type": "array", "minItems": 1, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["clip", "seconds", "goal", "founder_lines", "place", "present", "end_action", "changes", "place_changes", "new_recurring_visuals"],
            "properties": {
                "clip": {"type": "integer"}, "seconds": {"type": "integer"}, "goal": {"type": "string"}, "founder_lines": _strings(), "place": {"type": "string"},
                "present": _strings(), "end_action": {"type": "string"},
                "changes": _list_of(character={"type": "string"}, field={"type": "string", "enum": STATE_FIELDS}, to={"type": "string"}),
                "place_changes": _list_of(location={"type": "string"}, to={"type": "string"}),
                "new_recurring_visuals": _strings()}}},
    },
}
SHOTS_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["clip", "seconds", "shots"],
    "properties": {
        "clip": {"type": "integer"}, "seconds": {"type": "integer"},
        "shots": {"type": "array", "minItems": 1, "items": {
            "type": "object", "additionalProperties": False, "required": ["start", "framing", "action", "dialogue"],
            "properties": {"start": {"type": "integer"}, "framing": {"type": "string"}, "action": {"type": "string"},
                           "dialogue": {"anyOf": [{"type": "null"}, {"type": "object", "additionalProperties": False, "required": ["speaker", "line"],
                                                                      "properties": {"speaker": {"type": "string"}, "line": {"type": "string"}}}]}}}},
    },
}


def response_format(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def _num(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def _text(value) -> str:
    return " ".join(str(value or "").split())


def _pictures_list(value) -> list:
    return [{"label": _text(p.get("label")), "shows": _text(p.get("shows"))} for p in value or [] if isinstance(p, dict) and _text(p.get("label")) and _text(p.get("shows"))]


def _cast(value) -> list:
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and _text(item.get("name")):
            out.append({"name": _text(item["name"]), "look": _text(item.get("look") or item.get("description")), "wardrobe": _text(item.get("wardrobe")),
                        "props": _text(item.get("props")) or "none"})
    return out


def coerce_plan(obj) -> dict | None:
    """A plan from a parsed reply: tolerant of alternate keys and of clips given as plain strings. None when no clip with a goal can be found."""
    if not isinstance(obj, dict):
        return None
    raw = obj.get("clips") if isinstance(obj.get("clips"), list) else obj.get("plan")
    clips = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str):
            item = {"goal": item}
        if not isinstance(item, dict):
            continue
        goal = _text(item.get("goal") or item.get("beat") or item.get("description"))
        if not goal:
            continue
        lines = item.get("founder_lines")
        changes = [{"character": _text(c.get("character")), "field": _text(c.get("field")).lower(), "to": _text(c.get("to"))}
                   for c in item.get("changes") or [] if isinstance(c, dict) and _text(c.get("to"))]
        places = [{"location": _text(c.get("location")), "to": _text(c.get("to"))} for c in item.get("place_changes") or [] if isinstance(c, dict) and _text(c.get("to"))]
        clips.append({"clip": int(_num(item.get("clip"), len(clips) + 1)), "seconds": _num(item.get("seconds")), "goal": goal,
                      "founder_lines": [normalize(x) for x in lines if isinstance(x, str) and normalize(x)] if isinstance(lines, list) else [],
                      "place": _text(item.get("place")), "present": [_text(x) for x in item.get("present") or [] if isinstance(x, str) and _text(x)],
                      "end_action": _text(item.get("end_action")), "changes": changes, "place_changes": places,
                      "new_recurring_visuals": [_text(x) for x in item.get("new_recurring_visuals") or [] if isinstance(x, str) and _text(x)]})
    if not clips:
        return None
    return {"title": _text(obj.get("title")), "summary": _text(obj.get("summary")), "minor": bool(obj.get("minor_in_sexual_content")),
            "start_place": _text(obj.get("start_place")), "pictures": _pictures_list(obj.get("pictures")), "cast": _cast(obj.get("cast")), "locations": sw._named_list(obj.get("locations")), "clips": clips}


# --------------------------------------------------------------------------- state ledger (folded in code)

SKIN_MARKS = re.compile(r"\b(wound\w*|blood\w*|bleed\w*|scar\w*|bruis\w*|soot\w*|gash\w*|bloody|burn marks?|mole\w*|handprints?)\b", re.I)


def _member(plan: dict, name: str):
    wanted = name.casefold()
    return next((c["name"] for c in plan["cast"] if c["name"].casefold() == wanted), None)


def _place(plan: dict, name: str) -> str:
    wanted = name.casefold()
    return next((c["name"] for c in plan["locations"] if c["name"].casefold() == wanted), name)


def fold_states(plan: dict, upto: int | None = None) -> None:
    """Give every clip ``start_state`` and ``end_state`` (story_planner's end-state shape: location, time_light, end_action, characters of
    position / wardrobe / props / state). A clip's start is the previous clip's end; its end is that state with the clip's own ``changes`` applied, so
    a wardrobe, a prop or a place keeps its value until a clip changes it. Characters the model names that are not in the cast are ignored here and
    reported by ``check_plan``. ``upto`` folds only the first clips (the part of a long film planned so far)."""
    chars = {c["name"]: {"position": "", "wardrobe": c["wardrobe"], "props": c["props"], "state": ""} for c in plan["cast"]}
    places = {}
    place = plan.get("start_place") or (plan["locations"][0]["name"] if plan["locations"] else "")
    action = ""
    present: list = []
    for clip in plan["clips"][:upto]:
        def snapshot(names, where, doing):
            return {"location": _location_text(where, places), "time_light": "", "end_action": doing,
                    "characters": [dict(name=n, **chars[n]) for n in names if n in chars]}
        names = list(dict.fromkeys(present + [m for m in (_member(plan, x) for x in clip["present"]) if m]))
        clip["start_state"] = snapshot(names, place, action)
        for change in clip["changes"]:
            who = _member(plan, change["character"])
            if who and change["field"] in STATE_FIELDS:
                chars[who][change["field"]] = change["to"]
        for change in clip["place_changes"]:
            places[_place(plan, change["location"])] = change["to"]
        place = _place(plan, clip["place"]) if clip["place"] else place
        action = clip["end_action"] or action
        present = [m for m in (_member(plan, x) for x in clip["present"]) if m]
        clip["end_state"] = snapshot(present, place, action)


def _location_text(place: str, places: dict) -> str:
    changed = "; ".join(f"{name}: {state}" for name, state in places.items())
    return f"{place} ({changed})" if place and changed else place


def state_lines(state: dict | None) -> str:
    """A state as lines for the shot writer: ``Kael wears: ...; holds: ...; position: ...; condition: ...``."""
    if not state:
        return "nothing is established yet"
    lines = [f"Place: {state['location']}"] if state.get("location") else []
    if state.get("end_action"):
        lines.append(f"Last action: {state['end_action']}")
    for c in state["characters"]:
        parts = [f"{label}: {c[key]}" for key, label in (("wardrobe", "wears"), ("props", "holds / wears"), ("position", "position"), ("state", "condition")) if c.get(key)]
        lines.append(f"{c['name']} " + "; ".join(parts))
    return "\n".join(lines) or "nothing is established yet"


def check_plan(plan: dict, first: int = 1) -> list:
    """Problems of the state part of a plan: names outside the cast, a field that is not one of the four, a missing outfit, and any mark on skin."""
    problems = []
    for c in plan["cast"]:
        if not c["wardrobe"]:
            problems.append(f"cast member {c['name']} has no wardrobe; write the full outfit they wear when the film opens")
    for index, clip in enumerate(plan["clips"], start=first):
        for name in clip["present"] + [c["character"] for c in clip["changes"]]:
            if not _member(plan, name):
                problems.append(f"clip {index} names \"{name}\", who is not in the cast list; use the cast names exactly (or add them to the cast)")
        for change in clip["changes"]:
            if change["field"] not in STATE_FIELDS:
                problems.append(f"clip {index} has a change with field \"{change['field']}\"; the field is one of {', '.join(STATE_FIELDS)}")
        text = " ".join([clip["end_action"]] + [c["to"] for c in clip["changes"]] + [c["to"] for c in clip["place_changes"]])
        if SKIN_MARKS.search(text):
            problems.append(f"clip {index} describes a mark on the skin (\"{SKIN_MARKS.search(text).group(0)}\"); keep skin clean and show the effect on clothes, gear and posture instead")
    return list(dict.fromkeys(problems))


def coerce_shots(obj) -> dict | None:
    """The shots of one clip from a parsed reply, in the order given. None when there is no shot."""
    if not isinstance(obj, dict) or not isinstance(obj.get("shots"), list):
        return None
    shots = []
    for item in obj["shots"]:
        if not isinstance(item, dict):
            continue
        action = _text(item.get("action"))
        start = _num(item.get("start"))
        if not action or start is None:
            continue
        dialogue = item.get("dialogue")
        if isinstance(dialogue, dict) and _text(dialogue.get("line")):
            dialogue = {"speaker": _text(dialogue.get("speaker")) or "Voice", "line": normalize(dialogue.get("line"))}
        else:
            dialogue = None
        shots.append({"start": int(round(start)), "framing": _text(item.get("framing")), "action": action, "dialogue": dialogue})
    return {"shots": shots} if shots else None


# --------------------------------------------------------------------------- checks (code, no model)


def check_shots(shots: list, seconds: int, language: str, required_lines=()) -> list:
    """What is wrong with one clip's shot list, as sentences the writer can act on. Empty when it is fine."""
    problems = []
    if not shots:
        return ["there are no shots"]
    if not MIN_SHOTS <= len(shots) <= MAX_SHOTS:
        problems.append(f"the clip has {len(shots)} shots; write {MIN_SHOTS} to {MAX_SHOTS}")
    starts = [s["start"] for s in shots]
    if starts[0] != 0:
        problems.append(f"shot 1 starts at {starts[0]}s; it must start at 0")
    if any(b <= a for a, b in zip(starts, starts[1:])):
        problems.append(f"the shot starts {starts} do not strictly increase")
    late = [i + 1 for i, s in enumerate(starts) if s >= seconds]
    if late:
        problems.append(f"shot(s) {', '.join(map(str, late))} start at or after the end of the {seconds}-second clip; every start must be below {seconds}")
    ends = starts[1:] + [seconds]
    for i, (start, end) in enumerate(zip(starts, ends), start=1):
        if end - start < MIN_SHOT_SECONDS and i not in late:
            problems.append(f"shot {i} lasts {end - start}s (from {start}s to {end}s); every shot must last at least {MIN_SHOT_SECONDS}s")
    for i, (shot, start, end) in enumerate(zip(shots, starts, ends), start=1):
        dialogue = shot.get("dialogue")
        if not dialogue:
            continue
        line = dialogue["line"]
        bad = foreign_letters(line, language)
        if bad:
            problems.append(f"shot {i}: the line \"{line}\" has letters that are not in {language}'s own script ({''.join(sorted(set(bad))[:8])}); "
                            f"write every spoken line in {language} in its own script")
        words = len(line.split())
        allowed = max_line_words(max(end - start, 1), language)
        if words > allowed:
            problems.append(f"shot {i}: the line has {words} words but a {max(end - start, 1)}-second shot fits at most {allowed}; shorten the line or give the shot more time")
    spoken = [normalize(s["dialogue"]["line"]) for s in shots if s.get("dialogue")]
    for line in required_lines:
        if not any(line in said for said in spoken):
            problems.append(f"the founder's line \"{line}\" belongs in this clip and must appear word for word in one shot's dialogue")
    return problems


def check_costume(shots: list, start_state: dict | None) -> list:
    """The first shot that names a character must say what they wear, in a word of their outfit from the clip's start state (the raw prompt then carries it)."""
    problems = []
    for c in (start_state or {}).get("characters", []):
        words = {w.casefold() for w in re.findall(r"[^\W\d_]{4,}", c.get("wardrobe", ""))}
        if not words or c["wardrobe"].casefold().startswith("as on the reference"):
            continue
        first = next((s for s in shots if c["name"].casefold() in (s["action"] + " " + s.get("framing", "")).casefold()), None)
        if first and not any(w in first["action"].casefold() for w in words):
            problems.append(f"{c['name']} first appears in the shot at {first['start']}s without saying what they wear; add a short phrase with their outfit from the start state ({c['wardrobe']})")
    return problems


def repair_timing(shots: list, seconds: int) -> list:
    """Best effort once the retry also failed: sorted, first shot at 0, no repeated or out-of-range start."""
    out, last = [], None
    for shot in sorted(shots, key=lambda s: s["start"]):
        if shot["start"] >= seconds or shot["start"] == last:
            continue
        out.append(dict(shot))
        last = shot["start"]
    if out:
        out[0]["start"] = 0
    return out


# --------------------------------------------------------------------------- output formatting


def raw_text(shots: list) -> str:
    """``Shot N – <start>s – <framing>: <action>.`` and, with dialogue, `` <Speaker> says "<line>"``, one shot a line."""
    lines = []
    for i, shot in enumerate(shots, start=1):
        action = shot["action"].rstrip().rstrip(".")
        head = f"Shot {i} – {shot['start']}s – {shot['framing']}: " if shot.get("framing") else f"Shot {i} – {shot['start']}s: "
        text = head + action + "."
        if shot.get("dialogue"):
            text += f" {shot['dialogue']['speaker']} says \"{shot['dialogue']['line'].replace(chr(34), chr(0x201D))}\""
        lines.append(text)
    return "\n".join(lines)


def story_text(plan: dict) -> str:
    shown = "\n".join(f"- {p['label']}: {p['shows']}" for p in plan.get("pictures", []))
    parts = [plan["title"], plan["summary"], "Cast:\n" + sw.format_cast(plan["cast"]), "Locations:\n" + sw.format_locations(plan["locations"]),
             ("Reference pictures:\n" + shown) if shown else ""]
    return "\n\n".join(p for p in parts if p.strip()) + "\n"


def import_clips(text) -> list:
    """The Master Extender's clip entries for the JSON the Shot Writer's ``clips`` output holds: the shot list is the raw prompt, not yet rewritten.

    The seeds are fixed and derived from the text, so the same writer output always renders the same clips.
    """
    try:
        items = json.loads(text) if isinstance(text, str) else text
    except ValueError:
        items = None
    if not isinstance(items, list):
        return []
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    clips = []
    for item in items:
        if not isinstance(item, dict) or not str(item.get("raw") or "").strip():
            continue
        n = len(clips)
        raw = str(item["raw"]).strip()
        start, end = story_planner._coerce_end_state(item.get("start_state")), story_planner._coerce_end_state(item.get("end_state"))
        # The same blocks a planned story clip's ask carries: the state it opens in (from the clip before) and the state it ends in. Their marks switch
        # on the rewriter's STATE_RULE, so wardrobe and props come from the state and not from the reference picture.
        if start and n > 0:
            raw = f"{story_planner.OPENS_MARK.format(n=n)}\n{story_planner.format_end_state(start)}\n\n{raw}"
        if end:
            raw = f"{raw}\n\n{story_planner.ENDS_MARK}\n{story_planner.format_end_state(end)}"
        duration = int(round(_num(item.get("seconds"), DEFAULT_SECONDS)))
        clip = {"id": n, "title": f"Clip {n + 1}", "prompt": raw, "prompt_raw": raw, "prompt_rewritten": False, "duration": duration,
                "seed": int(digest[:8], 16) % 10**8 + n, "seed_mode": "fixed", "validated": False, "loras": [], "from_writer": True}
        if end:
            clip["plan_end_state"] = end
        if duration > 15:
            clip["beyond"] = True
        clips.append(clip)
    return clips


def has_prompt(clips) -> bool:
    """True when the clip list (JSON text or a list) holds a clip with a non-empty prompt."""
    try:
        items = json.loads(clips) if isinstance(clips, str) else clips
    except ValueError:
        return False
    return any(isinstance(c, dict) and str(c.get("prompt") or c.get("prompt_raw") or "").strip() for c in items or []) if isinstance(items, list) else False


# --------------------------------------------------------------------------- the engine


class ShotEngine:
    """Brief to clips with shot lists, with an injected ``chat``. See the module docstring."""

    def __init__(self, chat, work_dir, *, brief: str, clips: int, seconds: int = DEFAULT_SECONDS, language: str = DEFAULT_LANGUAGE, slot_ctx: int = 65536,
                 budget: int = THINKING_BUDGET, concurrency: int = DEFAULT_CONCURRENCY, plan_part: int = PLAN_PART, pictures=None, sees: bool = True,
                 progress=None, say=None):
        self.chat = chat
        self.work = Path(work_dir)
        self.brief = brief.strip()
        self.clips = int(clips)
        self.seconds = int(seconds)
        self.language = (language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
        self.slot_ctx = int(slot_ctx)
        self.budget = int(budget)
        self.concurrency = max(1, min(MAX_CONCURRENCY, int(concurrency)))
        self.plan_part = max(1, int(plan_part))
        self.progress = progress or (lambda done, total: None)
        self.say = say or (lambda message: None)
        self.sees = bool(sees)
        self.pictures = list(pictures or []) if self.sees else []
        self.picture_tokens = sum(p.get("tokens", 0) for p in self.pictures)
        self.founder = founder_lines(self.brief, self.language)
        self.labels = [p["label"] for p in pictures or []]
        if pictures and not self.sees:
            self.say("the writer model has no vision: the reference pictures are only labelled, not looked at")
            _LOG.warning("shot writer: the writer model has no vision; %d reference picture(s) are labelled but not interpreted", len(pictures))
        self.calls: list = []
        self.warnings: list = []
        self.retries = {"plan": 0, "shots": 0}
        self._lock = threading.Lock()
        self._limit = threading.BoundedSemaphore(self.concurrency)
        self._abort = threading.Event()
        (self.work / "clips").mkdir(parents=True, exist_ok=True)
        (self.work / "calls").mkdir(parents=True, exist_ok=True)
        self.calls_seen = len(list((self.work / "calls").glob("*.json")))

    # ---- calls

    def _warn(self, message: str) -> None:
        _LOG.warning("shot writer: %s", message)
        with self._lock:
            self.warnings.append(message)

    def _map(self, fn, items) -> list:
        items = list(items)
        if len(items) <= 1 or self.concurrency <= 1:
            return [fn(x) for x in items]
        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=min(len(items), self.concurrency)) as pool:
            futures = {pool.submit(fn, x): i for i, x in enumerate(items)}
            try:
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            except BaseException:
                self._abort.set()
                for future in futures:
                    future.cancel()
                raise
        return results

    def _ask(self, kind: str, messages, *, answer: int, schema_name: str, schema: dict) -> Reply:
        room = max(sw.MIN_OUTPUT, output_room([{"role": m["role"], "content": text_of(m["content"])} for m in messages], self.slot_ctx) - self.picture_tokens)
        cap = int(min(room, self.budget + answer + 512))
        with self._limit:
            if self._abort.is_set():
                raise StoryWriterError("stopped: another request failed")
            reply = self.chat(messages, max_tokens=cap, budget=self.budget, kind=kind, temperature=0.5, response_format=response_format(schema_name, schema))
        with self._lock:
            self.calls_seen += 1
            number = self.calls_seen
            meta = {"n": number, "kind": kind, "max_tokens": cap, "budget": self.budget, "finish": reply.finish, "prompt_tokens": reply.prompt_tokens,
                    "completion_tokens": reply.completion_tokens, "thinking_tokens": reply.thinking_tokens, "secs": round(reply.secs, 1)}
            self.calls.append(meta)
            record = dict(meta, at=time.strftime("%Y-%m-%d %H:%M:%S"), pictures=self.labels if self.pictures else [],
                          messages=[{"role": m["role"], "content": text_of(m["content"])} for m in messages], reply=reply.text)
            (self.work / "calls" / f"{number:03d}_{kind}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            with open(self.work / "calls.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(meta) + "\n")
        _LOG.info("shot writer: %s -> %s, %d tokens (thinking %d), cap %d, %.0f s", kind, reply.finish, reply.completion_tokens, reply.thinking_tokens, cap, reply.secs)
        return reply

    def _checked(self, kind: str, messages, *, answer: int, schema_name: str, schema: dict, parse, check):
        """Ask, parse, check; one retry with the complaint appended. Returns ``(value, problems)``: the better of the two answers, None when neither parsed."""
        best = None
        for attempt in range(2):
            reply = self._ask(kind, messages, answer=answer, schema_name=schema_name, schema=schema)
            value = parse(reply.text)
            problems = check(value) if value is not None else ["the answer was not one JSON object in the shape described"]
            if value is not None and (best is None or len(problems) < len(best[1])):
                best = (value, problems)
            if not problems:
                return value, []
            if attempt == 0:
                self._bump(kind)
                self.say(f"{kind}: {problems[0]}; asking again")
                complaint = "\n\nYOUR PREVIOUS ANSWER HAD THESE PROBLEMS. Fix every one and answer again with the complete JSON object:\n" + "\n".join(f"- {p}" for p in problems)
                messages = [with_complaint(messages[0], complaint)] + list(messages[1:])
        return best if best else (None, ["no usable answer"])

    def _bump(self, kind: str) -> None:
        with self._lock:
            key = "plan" if kind.startswith("plan") else "shots"
            self.retries[key] += 1

    def _lang(self) -> dict:
        return {"language": self.language, "script_note": script_note(self.language), "seconds": self.seconds}

    # ---- stage 1: the plan

    def plan(self) -> dict:
        stored = self.work / "plan.json"
        if stored.is_file():
            return json.loads(stored.read_text(encoding="utf-8"))
        self.say("plan: cast, locations, a goal and a state for every clip")
        total = self.clips
        parts = [(1, total)] if total <= self.plan_part else [(a, min(a + self.plan_part - 1, total)) for a in range(1, total + 1, self.plan_part)]
        merged = None
        for index, (first, last) in enumerate(parts):
            out = self._plan_part(first, last, total, merged, index + 1, len(parts))
            if out["minor"]:
                raise MinorContentError("the brief puts a minor in a sexual context; nothing was written", work_dir=str(self.work))
            if merged is None:
                merged = out
            else:
                merged["clips"] += out["clips"]
                for key in ("cast", "locations"):
                    have = {c["name"].casefold() for c in merged[key]}
                    merged[key] += [c for c in out[key] if c["name"].casefold() not in have]
        plan = merged
        seen = set()
        for number, clip in enumerate(plan["clips"], start=1):
            clip["clip"] = number
            seconds = clip.get("seconds")
            clip["seconds"] = int(round(seconds)) if seconds and MIN_SECONDS <= seconds <= MAX_SECONDS else self.seconds
            fresh = [v for v in clip["new_recurring_visuals"] if v.casefold() not in seen]
            seen.update(v.casefold() for v in fresh)
            clip["new_recurring_visuals"] = fresh
        fold_states(plan)
        listed = [f"clip {c['clip']}: " + ", ".join(c["new_recurring_visuals"]) for c in plan["clips"] if c["new_recurring_visuals"]]
        if listed:
            _LOG.warning("shot writer: new recurring visuals that need a reference image: %s", "; ".join(listed))
        stored.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
        return plan

    def _plan_part(self, first: int, last: int, total: int, so_far, part: int, parts: int) -> dict:
        n = last - first + 1
        extra = ""
        if so_far is not None:
            fold_states(so_far)
            done = "\n".join(f"{c['clip']}. {c['goal']}" for c in so_far["clips"])
            extra = (f"\nThis is part {part} of {parts}: the whole film is {total} clips and clips 1 to {first - 1} are already planned:\n{done}\n"
                     f"The state those clips left behind (what every character wears, holds and where they are; it carries into clip {first}, and your "
                     f"\"changes\" are changes to it):\n{state_lines(so_far['clips'][-1]['end_state'])}\n"
                     f"Plan ONLY clips {first} to {last}, continuing from where clip {first - 1} ended; the ending of the film belongs to the last part. "
                     f"Repeat the cast (with the same opening wardrobe and props) and locations of the film so far in your answer, adding any new member.\n")
        elif parts > 1:
            extra = (f"\nThis is part 1 of {parts}: the whole film is {total} clips; plan ONLY clips {first} to {last} now (the next parts continue the story, so do not end it yet).\n")
        if self.clips > 0:
            rule = (f"The film has EXACTLY {total} clips. If the brief covers fewer, continue the story with new clips to a real ending; if it covers more, compress by "
                    f"merging the least important ones, never by dropping its ending.")
            line = f"exactly {n} objects, numbered {first} to {last}"
        else:
            rule = ("You decide the number of clips from the brief: as many as it describes (a range counts as its distinct beats), at most "
                    f"{MAX_CLIPS}. If the brief ends with something like 'and so on', continue the story to a real ending.")
            line = "as many objects as the film needs, numbered from 1"
        values = dict(self._lang(), brief=self.brief, max_cast=sw.max_cast(total or MAX_CLIPS), count_rule=rule, clips_line=line, first=first, extra=extra)
        values["pictures_note"] = pictures_note(self.pictures or ([{"label": x} for x in self.labels]), self.sees)
        messages = [message(fill(load_prompt("plan"), **values), self.pictures)]
        answer = 900 + n * (420 if is_wide(self.language) else 330) + 500

        def check(plan):
            problems = []
            if self.clips > 0 and len(plan["clips"]) != n:
                problems.append(f"the plan has {len(plan['clips'])} clips; it must have exactly {n}" + (f" (clips {first} to {last})" if parts > 1 else ""))
            return problems + check_plan(plan, first)

        plan, problems = self._checked("plan", messages, answer=answer, schema_name="shot_plan", schema=PLAN_SCHEMA, parse=lambda t: coerce_plan(parse_json_object(t)), check=check)
        if plan is None:
            raise StoryWriterError("the plan could not be read from the model's answer; nothing was written", work_dir=str(self.work))
        for problem in problems:
            self._warn(f"plan: {problem}")
        if self.clips > 0 and len(plan["clips"]) > n:
            plan["clips"] = plan["clips"][:n]
        return plan

    # ---- stage 2: the shots of every clip

    def _clip_file(self, number: int) -> Path:
        return self.work / "clips" / f"{number:03d}.json"

    def shots_for(self, plan: dict, index: int) -> dict:
        number = index + 1
        stored = self._clip_file(number)
        if stored.is_file():
            return json.loads(stored.read_text(encoding="utf-8"))
        clips = plan["clips"]
        clip = clips[index]
        seconds = clip["seconds"]
        previous = (f"The clip before it ({index}) is: {clips[index - 1]['goal']} It ended: {clips[index - 1]['end_state']['end_action'] or 'where it left off'}" if index else "Nothing comes before it: this clip opens the film.")
        following = f"The clip after it ({number + 1}) is: {clips[index + 1]['goal']}" if index + 1 < len(clips) else "Nothing comes after it: this clip ends the film."
        required = [line for line in clip.get("founder_lines", []) if line in self.founder]
        lines_rule = ("- The founder wrote these lines for this clip, already in the film's language. Use each in one shot's dialogue, copied character for character: "
                      + "; ".join(f"\"{line}\"" for line in required)) if required else ""
        values = dict(self._lang(), cast=sw.format_cast(plan["cast"]), locations=sw.format_locations(plan["locations"]), brief=self.brief, clip=number, total=len(clips),
                      goal=clip["goal"], previous=previous, next=following, shot_range=f"{MIN_SHOTS} to {min(MAX_SHOTS, max(MIN_SHOTS, seconds // MIN_SHOT_SECONDS))}",
                      min_shots=MIN_SHOTS, lines_rule=lines_rule, extra="", seconds=seconds,
                      start_state=state_lines(clip["start_state"]), end_state=state_lines(clip["end_state"]))
        shown = "; ".join(f"{p['label']}: {p['shows']}" for p in plan.get("pictures", []))
        values["pictures_note"] = (pictures_note(self.pictures or ([{"label": x} for x in self.labels]), self.sees) + (f" What they show (from the plan): {shown}." if shown else "")) if self.labels else ""
        messages = [message(fill(load_prompt("shots"), **values), self.pictures)]
        answer = 250 + MAX_SHOTS * (260 if is_wide(self.language) else 150)

        def check(parsed):
            return check_shots(parsed["shots"], seconds, self.language, required) + check_costume(parsed["shots"], clip["start_state"])

        parsed, problems = self._checked(f"shots_{number:03d}", messages, answer=answer, schema_name="clip_shots", schema=SHOTS_SCHEMA,
                                         parse=lambda t: coerce_shots(parse_json_object(t)), check=check)
        if parsed is None:
            raise StoryWriterError(f"clip {number}: the shots could not be read from the model's answer", work_dir=str(self.work))
        shots = parsed["shots"]
        if problems:
            self._warn(f"clip {number}: " + "; ".join(problems))
            shots = repair_timing(shots, seconds) or shots
        result = {"clip": number, "seconds": seconds, "goal": clip["goal"], "raw": raw_text(shots), "shots": shots,
                  "start_state": clip["start_state"], "end_state": clip["end_state"], "new_recurring_visuals": clip["new_recurring_visuals"]}
        stored.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        self.progress(number + 1, len(clips) + 1)
        return result

    def check_film(self, out: list) -> None:
        said = [normalize(s["dialogue"]["line"]) for c in out for s in c["shots"] if s.get("dialogue")]
        for line in self.founder:
            if not any(line in x for x in said):
                self._warn(f"the founder's line \"{line}\" is not in any shot")

    def run(self) -> dict:
        plan = self.plan()
        self.progress(1, len(plan["clips"]) + 1)
        self.say(f"shots: {len(plan['clips'])} clips")
        out = self._map(lambda i: self.shots_for(plan, i), range(len(plan["clips"])))
        self.check_film(out)
        return {"clips": out, "plan": plan, "story": story_text(plan)}


# --------------------------------------------------------------------------- files


def job_hash(brief: str, clips: int, seconds: int, language: str, seed: int, model: str, budget: int, pictures=()) -> str:
    shown = [[p["label"], hashlib.sha1(str(p.get("image", "")).encode("utf-8")).hexdigest()] for p in pictures or []]
    payload = json.dumps([brief.strip(), int(clips), int(seconds), language, int(seed), model, int(budget), shown, VERSION], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def output_root(base=None) -> Path:
    if base is not None:
        return Path(base)
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory()) / OUTPUT_SUBDIR
    except Exception:
        return Path.cwd() / "output" / OUTPUT_SUBDIR


def write_outputs(root: Path, engine: ShotEngine, result: dict, *, model: str, seed: int, wall: float, style: str = "") -> tuple:
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}_{len(result['clips'])}clips"
    clips_path = root / f"{stem}.clips.json"
    clips_path.write_text(json.dumps(result["clips"], ensure_ascii=False, indent=1), encoding="utf-8")
    story_path = root / f"{stem}.story.md"
    story_path.write_text(result["story"], encoding="utf-8")
    calls = engine.calls
    side = {
        "brief_sha1": hashlib.sha1(engine.brief.encode("utf-8")).hexdigest(), "brief_chars": len(engine.brief), "clips_requested": engine.clips,
        "clips_written": len(result["clips"]), "seconds": engine.seconds, "language": engine.language, "seed": seed, "model": model, "api_style": style,
        "version": VERSION, "title": result["plan"]["title"], "cast": [c["name"] for c in result["plan"]["cast"]], "founder_lines": engine.founder,
        "new_recurring_visuals": {str(c["clip"]): c["new_recurring_visuals"] for c in result["clips"] if c["new_recurring_visuals"]},
        "shots_per_clip": [len(c["shots"]) for c in result["clips"]], "calls": len(calls),
        "tokens": {"prompt": sum(c["prompt_tokens"] for c in calls), "completion": sum(c["completion_tokens"] for c in calls), "thinking": sum(c["thinking_tokens"] for c in calls)},
        "warnings": engine.warnings, "retries": engine.retries, "concurrency": engine.concurrency, "pictures": engine.labels, "wall_seconds": round(wall, 1), "work_dir": str(engine.work),
        "call_log": calls, "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = root / f"{stem}.json"
    json_path.write_text(json.dumps(side, indent=1, ensure_ascii=False), encoding="utf-8")
    return clips_path, story_path, json_path, side


def write_shots(chat, *, brief: str, clips: int, seconds: int = DEFAULT_SECONDS, language: str = DEFAULT_LANGUAGE, seed: int = 0, model: str = "model", style: str = "",
                budget: int = THINKING_BUDGET, slot_ctx: int = 65536, out_dir=None, progress=None, say=None, concurrency: int = DEFAULT_CONCURRENCY,
                plan_part: int = PLAN_PART, pictures=None, sees: bool = True) -> dict:
    """The whole job with an injected ``chat``: validate, resume or start, write, save. Returns ``{clips, story, clips_path, story_path, json_path, sidecar}``."""
    brief = str(brief or "").strip()
    if not brief:
        raise StoryWriterError("brief is empty")
    clips = int(clips)
    if not MIN_CLIPS <= clips <= MAX_CLIPS:
        raise StoryWriterError(f"clips must be {MIN_CLIPS} (the writer decides) to {MAX_CLIPS}, got {clips}")
    root = output_root(out_dir)
    work = root / "_work" / job_hash(brief, clips, seconds, language, seed, model, budget, pictures)
    started = time.time()
    engine = ShotEngine(chat, work, brief=brief, clips=clips, seconds=seconds, language=language, slot_ctx=slot_ctx, budget=budget, concurrency=concurrency,
                        plan_part=plan_part, pictures=pictures, sees=sees, progress=progress, say=say)
    result = engine.run()
    clips_path, story_path, json_path, side = write_outputs(root, engine, result, model=model, seed=seed, wall=time.time() - started, style=style)
    (work / ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return {"clips": json.dumps(result["clips"], ensure_ascii=False, indent=2), "story": result["story"], "clips_path": str(clips_path), "story_path": str(story_path),
            "json_path": str(json_path), "sidecar": side}


# --------------------------------------------------------------------------- the ComfyUI node


def open_model(*args, **kwargs):
    return sw.open_model(*args, **kwargs)


class H3ShotWriter:
    """Free-form brief to a timed shot list per clip. Feed its ``clips`` to the Master Extender's ``clips_in`` and its ``story`` to ``rewrite_story``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "brief": ("STRING", {"default": "", "multiline": True, "tooltip": "Your clips in any form: 'Clip 1 - ...' lines, ranges like '3-6 - Battle.', plain prose, 'and so on'. Spoken lines you write are kept (translated into the film's language if they are in another). Everything else the writer may improve."}),
            "clips": ("INT", {"default": 0, "min": MIN_CLIPS, "max": MAX_CLIPS, "tooltip": "How many clips. 0 = the writer decides from the brief. More than the brief covers: the story is continued to a real ending."}),
            "seconds": ("INT", {"default": DEFAULT_SECONDS, "min": MIN_SECONDS, "max": MAX_SECONDS, "tooltip": "Seconds of screen time per clip. Each clip is cut into 2 to 6 timed shots of at least 2 seconds."}),
            "language": ("STRING", {"default": DEFAULT_LANGUAGE, "tooltip": "The language of every spoken line, written in its own script (Hindi in Devanagari, Kannada in Kannada script). A per-character language instruction in the brief overrides it."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "control_after_generate": True, "tooltip": "Part of the key under which a half-finished run is resumed; change it to write a different version of the same brief. Leave the control on fixed: ComfyUI then reuses the result until an input changes."}),
            "writer_model": (prompt_rewriter.writer_choices(), {"tooltip": "The rewriter pack's model list, the same as the Story Writer's writer_model. The server is started for this node and closed after it, so the render that follows has the GPU."}),
            "thinking_budget": (prompt_rewriter.REASONING_BUDGETS, {"default": "4096", "tooltip": "Thinking tokens at most per call, on top of the answer."}),
        }, "optional": {
            "concurrency": ("INT", {"default": DEFAULT_CONCURRENCY, "min": 1, "max": MAX_CONCURRENCY, "tooltip": "Requests sent to the model server at the same time: the clips' shot lists are written in parallel. The model server this node starts gets that many slots, at most 4; a higher value is lowered to the slots the server has."}),
            **{f"ref_image_{i}": ("IMAGE", {"tooltip": f"Reference picture {i}, wired from the same Load Image as the Master Extender's ref_image_{i}. The writer looks at it and the brief may call it 'picture {i}'; it stays <Picture {i}> in the raw prompts."}) for i in range(1, REF_SLOTS + 1)},
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("clips", "story")
    FUNCTION = "write"
    CATEGORY = "MiniMax H3 Master"
    OUTPUT_NODE = False

    @classmethod
    def VALIDATE_INPUTS(cls, thinking_budget=None):
        return True

    def write(self, brief, clips, seconds, language, seed, writer_model, thinking_budget="4096", concurrency=DEFAULT_CONCURRENCY, **refs):
        pictures = pictures_of(refs)
        strata = str(writer_model) == strata_backend.LABEL
        sees = strata_backend.vision_available() if strata else True
        budget = int(prompt_rewriter.reasoning_budget(thinking_budget))
        slot_ctx = e4_engine.slot_ctx()
        bar = None
        try:
            import comfy.utils

            bar = comfy.utils.ProgressBar(max(int(clips), 1) + 1)
        except Exception:
            bar = None
        last = {"n": 0}

        def progress(done, total):
            if bar is not None and done > last["n"]:
                bar.update(done - last["n"])
                last["n"] = done

        def say(message):
            _LOG.info("H3 Shot Writer: %s", message)

        with open_model(str(writer_model), budget, slot_ctx=slot_ctx, concurrency=concurrency, say=say) as (endpoint, runner):
            concurrency = min(int(concurrency), int(endpoint.get("slots") or concurrency))
            chat = ServerChat(endpoint, interrupted=getattr(runner, "interrupted", None))
            result = write_shots(chat, brief=brief, clips=clips, seconds=seconds, language=language, seed=seed, model=endpoint["model"], style=endpoint["style"],
                                 budget=budget, slot_ctx=slot_ctx, progress=progress, say=say, concurrency=concurrency, pictures=pictures, sees=sees)
        side = result["sidecar"]
        print(f"[H3 Shot Writer] {side['clips_written']} clips, {sum(side['shots_per_clip'])} shots, {side['wall_seconds']:.0f} s -> {result['clips_path']}", flush=True)
        for warning in side["warnings"]:
            print(f"[H3 Shot Writer] warning: {warning}", flush=True)
        for clip, visuals in side["new_recurring_visuals"].items():
            print(f"[H3 Shot Writer] warning: clip {clip} introduces recurring visuals that need a reference image: {', '.join(visuals)}", flush=True)
        return (result["clips"], result["story"])


NODE_CLASS_MAPPINGS = {"H3ShotWriter": H3ShotWriter}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ShotWriter": "H3 Shot Writer (brief to shot lists)"}
