"""H3 Story Writer: a short brief in, a full story out, sized so that E4 plans about N clips from it.

A separate node from the Master Extender (it changes nothing there). Wired ``[H3 Story Writer] --STRING--> [Master Extender story input]`` with
story_engine=e4.10: E4 treats its story input as a finished story (one clip per distinct dramatic beat, every spoken line copied verbatim), so a
three-line brief gives about four clips with invented dialogue. This node writes the story first: exactly N beats of about ``seconds`` seconds,
one paragraph each, every spoken line in quotes with its speaker.

Two stages, both on the model server the rewriter pack opens for the Extender's E4 (same ``mtmd.session`` / ``open_strata`` path, same wires, same
thinking budgets), closed again when the node finishes so the render that follows has the card:

1. OUTLINE (one call, or acts then one call per act in parallel from ``TWO_LEVEL_MIN`` beats): cast, locations and exactly N one-line beats, each with an
   ``end_state`` (place, who is present, visible state of clothes, body and props), as JSON. The reply is parsed tolerantly (alternate keys, beats as
   objects, broken quoting salvaged by key). A wrong count is repaired without a second full outline: too many are merged in code (a few) or by one call
   that names only the merges, too few are split in parallel, one call per longest beat.
2. WRITE (chunks of 4 to 8 beats sized so a story goes out in one wave, ``concurrency`` requests at a time): no chunk reads another's prose. Each gets the brief, the cast sheet,
   the locations, the whole outline with every beat's end state, the beats around its own, and the opening state of its first beat. Beats are delimited
   by ``=== N ===`` lines the code strips, so a cut reply still yields every complete beat. A call that is cut or returns the wrong count is retried once
   with a larger cap and a terser instruction, then the chunk is split in half. Every accepted beat is written to disk at once; a run that dies resumes
   from them when the same inputs run again. Afterwards cheap code checks look for seams (an opening in a place or with a person the previous beat's end
   state rules out, a name that is not in the cast sheet); they are reported as warnings and never cause a retry.

The pure engine (``StoryEngine``) takes any ``chat(messages, max_tokens=, budget=, kind=) -> Reply``; ``ServerChat`` is the HTTP one. ComfyUI is imported
lazily, so the engine runs and is tested without it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

try:
    from . import e4_engine
    from . import prompt_rewriter
    from . import strata_backend
    from . import vram_release
except ImportError:  # imported as a top-level module (tests)
    import e4_engine
    import prompt_rewriter
    import strata_backend
    import vram_release

_LOG = logging.getLogger("minimax_h3_master_extender.story_writer")

VERSION = "2"
MIN_CLIPS, MAX_CLIPS = 1, 60
DEFAULT_SECONDS = 15
DEFAULT_LANGUAGE = "English"
MIN_CHUNK_BEATS, CHUNK_BEATS = 4, 8   # chunk size is auto: about one chunk per concurrent request, between these
DEFAULT_CONCURRENCY, MAX_CONCURRENCY = 6, 8
TWO_LEVEL_MIN = 31      # from this many beats the outline is acts first, then the beats of every act in parallel
ACT_BEATS = 10
THINKING_BUDGET = 4096
BUDGET_MESSAGE = e4_engine.BUDGET_MESSAGE
MIN_OUTPUT = 4096
MARGIN = 1024
LATIN_CHARS_PER_TOKEN = 2.5
MIN_BEAT_CHARS = 40
SPEECH_WORDS_PER_SECOND = 2.0
E4_LATIN_CHARS_PER_CLIP, E4_WIDE_CHARS_PER_CLIP = 700, 380   # e4/bridge/scale.mjs weightOf: how E4 sizes a story it must plan in parts
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SENTINEL_MINOR = "MINOR_IN_SEXUAL_CONTENT"
OUTPUT_SUBDIR = "h3_story_writer"

LATIN_LANGUAGES = {
    "english", "spanish", "french", "german", "portuguese", "italian", "dutch", "polish", "turkish", "indonesian", "malay", "swedish", "danish", "norwegian",
    "finnish", "romanian", "czech", "slovak", "hungarian", "vietnamese", "hinglish", "swahili", "filipino", "tagalog", "latin", "catalan", "croatian",
}


class StoryWriterError(RuntimeError):
    """The story could not be written. ``work_dir`` holds what was kept; the same inputs resume from it."""

    def __init__(self, message, *, work_dir=None):
        super().__init__(message)
        self.work_dir = work_dir


class LLMUnavailable(StoryWriterError):
    """The model server did not answer."""


class MinorContentError(StoryWriterError):
    """The brief puts a minor in a sexual context. Nothing is written."""


@dataclass
class Reply:
    text: str
    finish: str = "stop"           # "stop" or "length"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    secs: float = 0.0


# --------------------------------------------------------------------------- language and sizing


def is_wide(language: str) -> bool:
    """True for a language written outside Latin script (E4 sizes those by 380 characters a clip instead of 700)."""
    return str(language or DEFAULT_LANGUAGE).strip().lower() not in LATIN_LANGUAGES


def script_note(language: str) -> str:
    if str(language).strip().lower() == "hinglish":
        return " (Hindi in Latin script, the way it is typed)"
    return ", in its own native script (never romanised)" if is_wide(language) else ""


def beat_words(seconds: int, language: str) -> int:
    base = 115 * seconds / 15
    return int(round(max(55, min(170, base * (0.6 if is_wide(language) else 1.0))) / 5.0) * 5)


def max_speech_words(seconds: int) -> int:
    return max(8, int(seconds * SPEECH_WORDS_PER_SECOND))


def max_cast(clips: int) -> int:
    return max(3, min(8, 2 + clips // 5))


def e4_weight(text: str) -> float:
    latin = wide = 0
    for ch in str(text or ""):
        if ord(ch) < 0x250:
            latin += 1
        else:
            wide += 1
    return latin / E4_LATIN_CHARS_PER_CLIP + wide / E4_WIDE_CHARS_PER_CLIP


def estimate_prompt_tokens(messages) -> int:
    """e4/planpath410/transport.mjs estimatePromptTokens: Latin text at 2.5 characters a token, every other character one token."""
    latin = wide = 0
    for message in messages or []:
        for ch in str(message.get("content") or ""):
            if ord(ch) < 0x250:
                latin += 1
            else:
                wide += 1
        latin += 10
    return int(math.ceil(latin / LATIN_CHARS_PER_TOKEN + wide))


def output_room(messages, slot_ctx: int) -> int:
    """The most a reply to these messages may use: the slot minus the prompt minus a margin (transport.mjs outputRoom)."""
    return max(MIN_OUTPUT, slot_ctx - int(math.ceil(estimate_prompt_tokens(messages) * 1.05)) - MARGIN)


def answer_tokens(beats: int, seconds: int, language: str, *, wide_per_word=6.5, latin_per_word=1.8) -> int:
    per_word = wide_per_word if is_wide(language) else latin_per_word
    return int(beats * (beat_words(seconds, language) * per_word + 40))


# --------------------------------------------------------------------------- prompts


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"story_writer_{name}.md").read_text(encoding="utf-8").strip()


def fill(template: str, **values) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def format_cast(cast: list) -> str:
    return "\n".join(f"- {c['name']}: {c['look']}" if c.get("look") else f"- {c['name']}" for c in cast) or "- (none named)"


def format_locations(locations: list) -> str:
    return "\n".join(f"- {c['name']}: {c['look']}" if c.get("look") else f"- {c['name']}" for c in locations) or "- (none named)"


def format_outline(beats: list, states: list, first: int, last: int) -> str:
    lines = []
    for i, beat in enumerate(beats, start=1):
        mark = "  <== WRITE THIS ONE" if first <= i <= last else ""
        state = state_text(states[i - 1]) if i - 1 < len(states) else ""
        lines.append(f"{i}. {beat}" + (f"\n   ends: {state}" if state else "") + mark)
    return "\n".join(lines)


# --------------------------------------------------------------------------- parsing


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return text.split("</think>")[-1].strip()


def _balanced_at(text: str, start: int):
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _loads(candidate: str):
    for text in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        try:
            return json.loads(text, strict=False)
        except ValueError:
            continue
    return None


def _clean_body(text: str) -> str:
    body = strip_think(text)
    body = re.sub(r"^```[a-zA-Z]*\s*", "", body.strip())
    return re.sub(r"```\s*$", "", body).strip()


def repair_brackets(text: str) -> str:
    """JSON whose brackets are wrong: a closer that does not match the open bracket becomes the right one, what follows the root is dropped, missing closers are added."""
    out, stack, in_str, esc = [], [], False, False
    for ch in text:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
            out.append(ch)
        elif ch in "}]":
            if not stack:
                break
            out.append(stack.pop())
            if not stack:
                break
        else:
            out.append(ch)
    out.extend(reversed(stack))
    return "".join(out)


def parse_json_object(text: str, *, repair: bool = True):
    """The first JSON object in a model reply (fences and prose around it ignored; trailing commas, raw line breaks inside strings and wrong or missing
    closing brackets forgiven), or None.

    Each ``{`` is tried in order, outermost first: an object whose own closing brace is missing is repaired as a whole, never mistaken for the first
    complete object nested inside it (a cast entry, say), which is what an unrepaired scan would return.
    """
    body = _clean_body(text)
    for match in list(re.finditer(r"\{", body))[:40]:
        chunk = _balanced_at(body, match.start())
        value = _loads(chunk) if chunk is not None else None
        if not isinstance(value, dict) and repair:
            value = _loads(repair_brackets(body[match.start():]))
        if isinstance(value, dict):
            return value
    return None


def _named_list(value) -> list:
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and str(item.get("name") or "").strip():
            out.append({"name": str(item["name"]).strip(), "look": " ".join(str(item.get("look") or item.get("description") or "").split())})
        elif isinstance(item, str) and item.strip():
            out.append({"name": item.strip(), "look": ""})
    return out


BEAT_KEYS = ("beats", "outline", "scenes", "shots", "story_beats", "moments", "beat_list", "clips", "sequence")
TEXT_KEYS = ("beat", "text", "description", "summary", "moment", "content", "scene", "action", "line")
STATE_KEYS = ("end_state", "endstate", "state_at_end", "ends")
EMPTY_STATE = {"place": "", "present": [], "state": ""}


def norm_state(value) -> dict:
    """An end state as ``{place, present, state}`` from whatever shape the model used (a bare string becomes the visible state)."""
    if isinstance(value, str):
        return {"place": "", "present": [], "state": " ".join(value.split())}
    if not isinstance(value, dict):
        return dict(EMPTY_STATE, present=[])
    present = value.get("present") or value.get("who") or value.get("characters") or []
    if isinstance(present, str):
        present = re.split(r"\s*[,;&]\s*|\s+and\s+", present)
    present = [" ".join(str(p).split()) for p in present if isinstance(p, (str, int)) and str(p).strip()] if isinstance(present, list) else []
    state = value.get("state") or value.get("visible_state") or value.get("status") or ""
    if isinstance(state, dict):
        state = "; ".join(f"{k}: {v}" for k, v in state.items())
    elif isinstance(state, list):
        state = "; ".join(str(v) for v in state)
    return {"place": " ".join(str(value.get("place") or value.get("location") or "").split()), "present": present, "state": " ".join(str(state).split())}


def state_text(state: dict) -> str:
    parts = []
    if state.get("place"):
        parts.append(f"place: {state['place']}")
    if state.get("present"):
        parts.append("present: " + ", ".join(state["present"]))
    if state.get("state"):
        parts.append(f"visible state: {state['state']}")
    return "; ".join(parts)


def _beat_entry(item):
    if isinstance(item, str):
        text, state = item, None
    elif isinstance(item, dict):
        text = next((item[k] for k in TEXT_KEYS if isinstance(item.get(k), str) and item[k].strip()), "")
        state = next((item[k] for k in STATE_KEYS if k in item), None)
    else:
        return None
    text = " ".join(str(text).split())
    return (text, norm_state(state)) if text else None


def _beat_items(raw) -> list:
    if isinstance(raw, str):
        raw = [re.sub(r"^\s*(?:\d+\s*[.):\-]|[-*•])\s*", "", line) for line in raw.splitlines() if line.strip()]
    if isinstance(raw, dict):
        keyed = sorted(raw.items(), key=lambda kv: int(kv[0]) if str(kv[0]).strip().isdigit() else 10 ** 9)
        raw = [v for _, v in keyed]
    return [e for e in (_beat_entry(i) for i in raw if i is not None) if e] if isinstance(raw, list) else []


def find_beats(obj: dict) -> list:
    """The beats of an outline object under any of the usual keys, or (failing that) the biggest list of beat-like items."""
    lowered = {str(k).lower(): v for k, v in obj.items()}
    for key in BEAT_KEYS:
        items = _beat_items(lowered.get(key))
        if items:
            return items
    best = []
    for key, value in lowered.items():
        if key in ("cast", "characters", "locations", "settings", "places", "acts"):
            continue
        items = _beat_items(value) if isinstance(value, list) else []
        if len(items) > len(best):
            best = items
    return best if len(best) >= 2 else []


def coerce_outline(obj) -> dict | None:
    """A reply's JSON as ``{title, minor, cast, locations, beats, end_states}``; None when it has no usable beats and is not the minor refusal."""
    if isinstance(obj, list):
        obj = {"beats": obj}
    if not isinstance(obj, dict):
        return None
    minor = obj.get("minor_in_sexual_content") is True
    items = find_beats(obj)
    if not items and not minor:
        return None
    lowered = {str(k).lower(): v for k, v in obj.items()}
    return {"title": str(obj.get("title") or "").strip(), "minor": minor, "cast": _named_list(lowered.get("cast") or lowered.get("characters")),
            "locations": _named_list(lowered.get("locations") or lowered.get("settings") or lowered.get("places")),
            "beats": [t for t, _ in items], "end_states": [s for _, s in items]}


def _array_after(body: str, key: str):
    m = re.search(r'["\']?' + key + r'["\']?\s*[:=]\s*\[', body, re.IGNORECASE)
    if not m:
        return None
    start = m.end() - 1
    depth = 0
    for i in range(start, len(body)):
        if body[i] == "[":
            depth += 1
        elif body[i] == "]":
            depth -= 1
            if depth == 0:
                return body[start:i + 1]
    return body[start:]


SALVAGE_ITEM = re.compile(r'"beat"\s*:\s*"(.*?)"\s*,\s*"end_state"\s*:\s*(\{[^{}]*\})', re.DOTALL)


def salvage_outline(text: str) -> dict | None:
    """Last resort for a reply whose JSON does not parse (typically a straight double quote inside a spoken line): the beats are cut out by their keys."""
    body = _clean_body(text)
    region = None
    for key in BEAT_KEYS:
        region = _array_after(body, key)
        if region:
            break
    items = []
    if region:
        for m in SALVAGE_ITEM.finditer(region):
            state = _loads(m.group(2))
            items.append((" ".join(m.group(1).replace('\\"', '"').split()), norm_state(state)))
        if not items:
            inner = region.strip()[1:]
            inner = inner[:inner.rfind("]")] if "]" in inner else inner
            pieces = re.split(r'"\s*,\s*\n\s*"', inner.strip().strip(",").strip())
            items = [(" ".join(p.strip().strip('"').replace('\\"', '"').split()), dict(EMPTY_STATE, present=[])) for p in pieces if len(p.strip().strip('"')) >= MIN_BEAT_CHARS]
    if not items:
        numbered = [m.group(1) for m in re.finditer(r"^\s*\d+\s*[.)]\s*(.{%d,})$" % MIN_BEAT_CHARS, body, re.MULTILINE)]
        items = [(" ".join(t.split()), dict(EMPTY_STATE, present=[])) for t in numbered]
    if len(items) < 2:
        return None
    cast = _array_after(body, "cast")
    places = _array_after(body, "locations")
    title = re.search(r'"title"\s*:\s*"(.*?)"\s*,', body)
    return {"title": title.group(1).strip() if title else "", "minor": False, "cast": _named_list(_loads(cast) if cast else None),
            "locations": _named_list(_loads(places) if places else None), "beats": [t for t, _ in items], "end_states": [s for _, s in items]}


def parse_outline_reply(text: str, finish: str = "stop"):
    """``(outline, why)``: the outline in a model reply, or None and the reason it was unusable.

    A reply cut by the token cap is taken only when its JSON object is complete. A reply that stopped is parsed tolerantly (alternate keys, beats as
    objects, trailing commas, raw line breaks) and, if the JSON itself is broken, salvaged by key.
    """
    body = _clean_body(text)
    if not body:
        return None, "empty reply (thinking used the whole budget?)"
    obj = parse_json_object(body, repair=finish != "length")
    if obj is None and body.startswith("["):
        value = _loads(body)
        obj = {"beats": value} if isinstance(value, list) else None
    if obj is not None:
        parsed = coerce_outline(obj)
        if parsed:
            return parsed, ""
    if finish == "length":
        return None, "cut by the token cap"
    parsed = salvage_outline(body)
    if parsed:
        return parsed, ""
    return None, "JSON without a beats list" if obj is not None else "no parseable JSON"


def salvage_acts(text: str, count: int):
    """The acts of a plan whose JSON is too broken to parse, cut out by their keys: ``(cast, locations, title, acts)`` or None."""
    body = _clean_body(text)
    summaries = [" ".join(m.group(1).split()) for m in re.finditer(r'"summary"\s*:\s*"(.*?)"\s*,\s*"(?:start_state|end_state)"', body, re.DOTALL)]
    if len(summaries) != count:
        return None
    starts = [norm_state(_loads(m.group(1))) for m in re.finditer(r'"start_state"\s*:\s*(\{[^{}]*\})', body)]
    ends = [norm_state(_loads(m.group(1))) for m in re.finditer(r'"end_state"\s*:\s*(\{[^{}]*\})', body)]
    empty = dict(EMPTY_STATE, present=[])
    acts = [{"summary": t, "start": starts[k] if len(starts) == count else empty, "end": ends[k] if len(ends) == count else empty} for k, t in enumerate(summaries)]
    cast, places = _array_after(body, "cast"), _array_after(body, "locations")
    title = re.search(r'"title"\s*:\s*"(.*?)"\s*,', body)
    return (_named_list(_loads(repair_brackets(cast)) if cast else None), _named_list(_loads(repair_brackets(places)) if places else None),
            title.group(1).strip() if title else "", acts)


def coerce_acts(obj, count: int):
    """The acts of an act-plan reply as a list of ``{summary, start, end}`` of exactly ``count`` acts, or None."""
    acts = None
    if isinstance(obj, dict):
        lowered = {str(k).lower(): v for k, v in obj.items()}
        acts = lowered.get("acts") or lowered.get("outline") or lowered.get("parts")
    if not isinstance(acts, list):
        return None
    out = []
    for item in acts:
        if isinstance(item, str) and item.strip():
            out.append({"summary": " ".join(item.split()), "start": dict(EMPTY_STATE, present=[]), "end": dict(EMPTY_STATE, present=[])})
        elif isinstance(item, dict):
            text = next((item[k] for k in TEXT_KEYS if isinstance(item.get(k), str) and item[k].strip()), "")
            if text:
                out.append({"summary": " ".join(text.split()), "start": norm_state(item.get("start_state") or item.get("start")),
                            "end": norm_state(next((item[k] for k in STATE_KEYS if k in item), None))})
    return out if len(out) == count else None


MARKER = re.compile(r"^[ \t]*=+[ \t]*(END|\d+)[ \t]*=+[ \t]*$", re.IGNORECASE | re.MULTILINE)
LABEL = re.compile(r"^\s*(?:\**\s*)?(?:beat|clip|scene|shot)\s*\d+\s*[:.\-–—)]*\s*\**\s*", re.IGNORECASE)


def clean_beat(text: str) -> str:
    """One beat as one plain paragraph: markdown, labels and line breaks removed."""
    text = re.sub(r"^\s*#+\s*.*$", "", text, flags=re.MULTILINE) if re.match(r"\s*#", text) else text
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = " ".join(text.split())
    return LABEL.sub("", text).strip()


def parse_beats(text: str, count: int, *, truncated: bool):
    """The beats of a writer reply, in order: ``(beats, saw_end)``.

    Blocks are cut at the ``=== N ===`` lines. A reply that was cut (``truncated``) or never reached ``=== END ===`` loses its last block, which may be
    half a paragraph. A block shorter than MIN_BEAT_CHARS ends the accepted run. At most ``count`` beats are returned. An ``=== END ===`` line followed by
    another beat (some models close every beat with one) is not the end.
    """
    text = strip_think(text)
    marks = list(MARKER.finditer(text))
    blocks = []
    for i, mark in enumerate(marks):
        if mark.group(1).upper() == "END":
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        blocks.append(text[mark.end():end])
    saw_end = bool(marks) and marks[-1].group(1).upper() == "END"
    if not marks:
        return [], False
    if truncated and not saw_end and blocks:
        blocks = blocks[:-1]
    out = []
    for block in blocks[:count]:
        beat = clean_beat(block)
        if len(beat) < MIN_BEAT_CHARS:
            break
        out.append(beat)
    return out, saw_end


QUOTED = re.compile(r"\"([^\"]+)\"|“([^”]+)”|«([^»]+)»")


def quoted_words(text: str) -> int:
    return sum(len((a or b or c).split()) for a, b, c in QUOTED.findall(text))


def quoted_lines(text: str) -> list:
    return [(a or b or c).strip() for a, b, c in QUOTED.findall(text)]


# --------------------------------------------------------------------------- the model server


def _anthropic_parts(content):
    """A message's content for /v1/messages: text stays, OpenAI ``image_url`` data-URI parts become ``image`` blocks."""
    if not isinstance(content, list):
        return content
    out = []
    for part in content:
        url = (part.get("image_url") or {}).get("url", "") if part.get("type") == "image_url" else ""
        if url.startswith("data:") and ";base64," in url:
            head, data = url.split(";base64,", 1)
            out.append({"type": "image", "source": {"type": "base64", "media_type": head[5:], "data": data}})
        else:
            out.append(part)
    return out


class ServerChat:
    """A chat call on the model server the rewriter opened (``e4_engine.endpoint_of`` describes it): ninfer-serve's /v1/messages, or chat completions.

    ninfer: ``thinking.budget_tokens`` is a real per-request cap and thinking counts toward ``max_tokens``. llama.cpp and Strata: the top-level
    ``reasoning_budget_tokens`` and ``reasoning_budget_message``. Transient errors are retried; a server that stays down raises LLMUnavailable.
    """

    def __init__(self, endpoint: dict, *, budget_message: str = BUDGET_MESSAGE, interrupted=None, retries: int = 3, timeout: float = 1800.0, sleep=time.sleep):
        self.url = str(endpoint["url"]).rstrip("/")
        self.model = endpoint.get("model") or "local"
        self.style = endpoint["style"]
        self.budget_message = budget_message
        self.interrupted = interrupted
        self.retries = retries
        self.timeout = timeout
        self.sleep = sleep

    def request(self, messages, max_tokens: int, budget: int, temperature: float, response_format=None):
        if self.style == "ninfer-messages":
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "thinking": {"type": "enabled", "budget_tokens": max(1024, budget)},
                    "messages": [dict(m, content=_anthropic_parts(m["content"])) for m in messages if m["role"] != "system"]}
            if system:
                body["system"] = system
            return f"{self.url}/v1/messages", {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}, body
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "reasoning_budget_tokens": max(1024, budget), "reasoning_budget_message": self.budget_message}
        if response_format:   # ninfer and Strata enforce it on chat completions; /v1/messages gets no schema (the prompt describes the JSON)
            body["response_format"] = response_format
        return f"{self.url}/v1/chat/completions", {"Content-Type": "application/json"}, body

    def parse(self, payload: dict, secs: float) -> Reply:
        if self.style == "ninfer-messages":
            blocks = payload.get("content") or []
            text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
            usage = payload.get("usage") or {}
            think = (usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0
            return Reply(strip_think(text), "length" if payload.get("stop_reason") == "max_tokens" else "stop", int(usage.get("input_tokens") or 0),
                         int(usage.get("output_tokens") or 0), int(think), secs)
        choice = (payload.get("choices") or [{}])[0]
        usage = payload.get("usage") or {}
        think = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        return Reply(strip_think((choice.get("message") or {}).get("content") or ""), "length" if choice.get("finish_reason") == "length" else "stop",
                     int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), int(think), secs)

    def __call__(self, messages, *, max_tokens: int, budget: int, kind: str = "", temperature: float = 0.6, response_format=None) -> Reply:
        url, headers, body = self.request(messages, max_tokens, budget, temperature, response_format)
        last = ""
        for attempt in range(1, self.retries + 1):
            if self.interrupted is not None and self.interrupted():
                raise_interrupt()
            started = time.time()
            try:
                request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as answer:
                    payload = json.loads(answer.read().decode("utf-8"))
                if isinstance(payload, dict) and not payload.get("error"):
                    return self.parse(payload, time.time() - started)
                last = str(payload.get("error"))[:300]
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:300]
                last = f"HTTP {error.code} {detail}"
                if error.code in (400, 422) and "response_format" in body:   # a server that rejects the schema still answers without it
                    _LOG.warning("story writer: the server rejected response_format (%s); asking without it", last)
                    body.pop("response_format")
                    continue
                if error.code in (400, 401, 403, 404, 422):
                    raise StoryWriterError(f"the model server refused the {kind or 'request'}: {last}") from error
            except (urllib.error.URLError, OSError, ValueError) as error:
                last = str(error)
            _LOG.warning("story writer: %s attempt %d/%d failed: %s", kind or "call", attempt, self.retries, last)
            if attempt < self.retries:
                self.sleep(4 * attempt)
        raise LLMUnavailable(f"the model server at {self.url} did not answer ({last}); nothing was written")


def raise_interrupt():
    try:
        import comfy.model_management as mm
    except ImportError:
        raise KeyboardInterrupt("interrupted") from None
    raise mm.InterruptProcessingException()


# --------------------------------------------------------------------------- seam checks (code, no model)


def _aliases(name: str) -> list:
    tokens = [t for t in re.split(r"[\s,.]+", name) if len(t) >= 3]
    return [name] + [t for t in tokens if t != name]


def mentions(text: str, name: str, *, whole: bool = False) -> bool:
    haystack = text.casefold()
    names = [name] if whole else _aliases(name)
    return any(n.casefold() in haystack for n in names if n.strip())


OPENING_CHARS = 320
COMMON_CAPS = {"the", "and", "but", "then", "she", "he", "they", "his", "her", "its", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
               "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}


def seam_issues(index: int, beat: str, outline_line: str, previous_state: dict, own_state: dict, cast: list, locations: list) -> list:
    """Obvious contradictions between the opening of beat ``index`` (0-based) and the state beat ``index - 1`` ended in. Cheap string checks only:

    - the opening names a location that is neither where the previous beat ended, nor named by this beat's own outline line, nor where this beat ends;
    - the opening names a cast member who was not present at the end of the previous beat and whom this beat's outline line (or end state) does not bring in.
    """
    if index <= 0 or not previous_state:
        return []
    opening = beat[:OPENING_CHARS]
    issues = []
    place = previous_state.get("place") or ""
    if place:
        expected = [loc["name"] for loc in locations if mentions(outline_line, loc["name"], whole=True)] + [place]
        if own_state and own_state.get("place"):
            expected.append(own_state["place"])
        seen = [loc["name"] for loc in locations if mentions(opening, loc["name"], whole=True)]
        if seen and not any(mentions(s, e, whole=True) or mentions(e, s, whole=True) for s in seen for e in expected):
            issues.append(f"beat {index + 1} opens in '{seen[0]}' but beat {index} ended in '{place}' and its outline line does not move there")
    present = previous_state.get("present") or []
    if present:
        for person in cast:
            name = person["name"]
            brought = list(present) + list((own_state or {}).get("present") or [])
            if mentions(opening, name) and not any(mentions(p, name) or mentions(name, p) for p in brought) and not mentions(outline_line, name):
                issues.append(f"beat {index + 1} opens with {name}, who was not present at the end of beat {index} and is not brought in by its outline line")
    return issues


def unknown_names(beats: list, cast: list, locations: list, language: str) -> list:
    """Capitalised words in the middle of a sentence, used at least twice, that are in no cast or location name: a drifted or invented name. Latin script only."""
    if is_wide(language):
        return []
    known = {t.casefold() for c in list(cast) + list(locations) for t in re.split(r"[\s,.'’-]+", c["name"]) if t}
    counts: dict = {}
    for beat in beats:
        for m in re.finditer(r"(?<=[a-z,;] )([A-Z][a-z]{2,})\b", beat):
            word = m.group(1)
            if word.casefold() not in known and word.casefold() not in COMMON_CAPS:
                counts[word] = counts.get(word, 0) + 1
    return sorted(w for w, c in counts.items() if c >= 2)


def split_chunks(n: int, have, chunk_beats: int) -> list:
    """The runs of beats still to write as ``(start, end)`` ranges, each run cut into the fewest balanced chunks of at most ``chunk_beats``."""
    out, i = [], 0
    while i < n:
        if have(i):
            i += 1
            continue
        j = i
        while j < n and not have(j):
            j += 1
        length = j - i
        parts = -(-length // chunk_beats)
        size, extra = divmod(length, parts)
        start = i
        for p in range(parts):
            end = start + size + (1 if p < extra else 0)
            out.append((start, end))
            start = end
        i = j
    return out


def auto_chunk_beats(n: int, concurrency: int) -> int:
    """About one chunk per concurrent request, so a story goes out in a single wave: 40 beats on 8 slots is 5 a chunk, on 6 slots 7; never under 4 (more calls, more repeated outline) or over 8."""
    return max(MIN_CHUNK_BEATS, min(CHUNK_BEATS, -(-n // max(1, concurrency))))


def act_sizes(n: int, per_act: int = ACT_BEATS) -> list:
    acts = max(2, -(-n // per_act))
    size, extra = divmod(n, acts)
    return [size + (1 if k < extra else 0) for k in range(acts)]


# --------------------------------------------------------------------------- the engine


class StoryEngine:
    """Brief to story with an injected ``chat``. See the module docstring."""

    def __init__(self, chat, work_dir, *, brief: str, clips: int, seconds: int = DEFAULT_SECONDS, language: str = DEFAULT_LANGUAGE, slot_ctx: int = 65536,
                 budget: int = THINKING_BUDGET, chunk_beats: int | None = None, concurrency: int = DEFAULT_CONCURRENCY, two_level_min: int = TWO_LEVEL_MIN,
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
        self.chunk_beats = max(1, int(chunk_beats)) if chunk_beats else auto_chunk_beats(self.clips, self.concurrency)
        self.two_level_min = int(two_level_min)
        self.progress = progress or (lambda done, total: None)
        self.say = say or (lambda message: None)
        self.calls: list = []
        self.warnings: list = []
        self.seam_issues: list = []
        self.chunks: list = []
        self.retries = {"outline": 0, "outline_fix": 0, "write": 0}
        self.outline_mode = ""
        self.total_steps = self.clips + 1
        self._lock = threading.Lock()
        self._limit = threading.BoundedSemaphore(self.concurrency)
        self._abort = threading.Event()
        (self.work / "beats").mkdir(parents=True, exist_ok=True)
        (self.work / "calls").mkdir(parents=True, exist_ok=True)
        self.calls_seen = len(list((self.work / "calls").glob("*.json")))

    # ---- calls

    def _bump(self, key: str) -> None:
        with self._lock:
            self.retries[key] = self.retries.get(key, 0) + 1

    def _map(self, fn, items) -> list:
        """``fn`` over ``items`` on up to ``concurrency`` threads, results in item order; the first failure stops the queued work and is raised."""
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

    def _ask(self, kind: str, messages, *, answer: int, budget=None, scale: float = 1.0, temperature: float = 0.6) -> Reply:
        budget = self.budget if budget is None else budget
        room = output_room(messages, self.slot_ctx)
        cap = int(min(room, max(budget + 512, int((budget + answer) * scale) + 512)))
        with self._limit:
            if self._abort.is_set():
                raise StoryWriterError("stopped: another request failed")
            reply = self.chat(messages, max_tokens=cap, budget=budget, kind=kind, temperature=temperature)
        with self._lock:
            self.calls_seen += 1
            number = self.calls_seen
            meta = {"n": number, "kind": kind, "max_tokens": cap, "budget": budget, "finish": reply.finish, "prompt_tokens": reply.prompt_tokens,
                    "completion_tokens": reply.completion_tokens, "thinking_tokens": reply.thinking_tokens, "secs": round(reply.secs, 1)}
            self.calls.append(meta)
            record = dict(meta, at=time.strftime("%Y-%m-%d %H:%M:%S"), messages=messages, reply=reply.text)
            (self.work / "calls" / f"{number:03d}_{kind}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            with open(self.work / "calls.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(meta) + "\n")
        _LOG.info("story writer: %s -> %s, %d tokens (thinking %d), cap %d, %.0f s", kind, reply.finish, reply.completion_tokens, reply.thinking_tokens, cap, reply.secs)
        return reply

    def _lang(self) -> dict:
        return {"language": self.language, "script_note": script_note(self.language), "seconds": self.seconds, "clips": self.clips}

    def _answer_room(self, n: int) -> int:
        return 500 + n * (320 if is_wide(self.language) else 150)

    # ---- stage 1: the outline

    def outline(self) -> dict:
        stored = self.work / "outline.json"
        if stored.is_file():
            out = json.loads(stored.read_text(encoding="utf-8"))
            out.setdefault("end_states", [dict(EMPTY_STATE, present=[]) for _ in out["beats"]])
            return out
        self.say(f"outline: cast, locations and {self.clips} beats")
        out = None
        if self.clips >= self.two_level_min:
            out = self._outline_two_level()
        if out is None:
            out = self._outline_whole()
            if out is not None and not out["minor"] and not plausible_count(len(out["beats"]), self.clips):
                self.say(f"outline has {len(out['beats'])} beats, {self.clips} wanted: too far off to repair, planning it in parts")
                out = None
            out = out or self._outline_in_parts()
        if out is None:
            raise StoryWriterError("the model did not return a usable outline in three attempts (replies are kept in " + str(self.work / "calls") + ")", work_dir=str(self.work))
        if out["minor"]:
            raise MinorContentError("the brief puts a minor in a sexual context; no story was written")
        out = self._fix_count(out, self.clips)
        stored.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        self.progress(1, self.total_steps)
        return out

    def _outline_messages(self, n: int, extra: str = "", total: int | None = None) -> list:
        values = dict(self._lang(), brief=self.brief, max_cast=max_cast(total or n), max_words=max_speech_words(self.seconds), extra=extra)
        values["clips"] = n
        return [{"role": "user", "content": fill(load_prompt("outline"), **values)}]

    def _parsed_call(self, kind: str, messages, answer: int, *, temperature: float = 0.5):
        """Up to two asks of one outline-shaped question; the second is larger only when the first was cut."""
        cut = False
        for attempt in range(2):
            reply = self._ask(kind, messages, answer=answer, scale=2.0 if cut else 1.0, temperature=temperature)
            parsed, why = parse_outline_reply(reply.text, reply.finish)
            cut = reply.finish == "length"
            if parsed:
                return parsed
            self.say(f"{kind} reply unusable ({why})" + (", asking again" if attempt == 0 else ""))
            if attempt == 0:
                self._bump("outline")
        return None

    def _outline_call(self, n: int, extra: str = "", total: int | None = None):
        return self._parsed_call("outline", self._outline_messages(n, extra, total), self._answer_room(n))

    def _outline_whole(self):
        self.outline_mode = "whole"
        return self._outline_call(self.clips)

    def _outline_in_parts(self):
        """Fallback when the whole outline cannot be had in one reply: the first half, then the second half given the first."""
        if self.clips < 4:
            return None
        self.outline_mode = "two parts"
        first = self.clips // 2
        second = self.clips - first
        self.say(f"outline in two parts ({first} + {second} beats)")
        head = self._outline_call(first, f"This is the FIRST PART of a story of {self.clips} beats: write only its first {first} beats, ending on a moment that leads on. "
                                  "Do not end the story. The cast may include characters who only appear in the second part.", total=self.clips)
        if head is None:
            return None
        head = self._fix_count(head, first)
        prior = "\n".join(f"{i}. {b}" for i, b in enumerate(head["beats"], start=1))
        tail = self._outline_call(second, f"This is the SECOND PART of a story of {self.clips} beats. The first {first} beats are already written:\n{prior}\n"
                                  f"Cast so far:\n{format_cast(head['cast'])}\nWrite ONLY the last {second} beats, which continue from beat {first} and end the story. "
                                  "Reuse the cast above under the same names; add a character only if the story needs one.", total=self.clips)
        if tail is None:
            return None
        tail = self._fix_count(tail, second)
        names = {c["name"] for c in head["cast"]}
        places = {c["name"] for c in head["locations"]}
        return {"title": head["title"] or tail["title"], "minor": head["minor"] or tail["minor"], "beats": head["beats"] + tail["beats"],
                "end_states": head["end_states"] + tail["end_states"],
                "cast": head["cast"] + [c for c in tail["cast"] if c["name"] not in names],
                "locations": head["locations"] + [c for c in tail["locations"] if c["name"] not in places]}

    def _outline_two_level(self):
        """Acts first (one call: cast, locations, a summary and the start and end state of each act), then the beats of every act in parallel."""
        sizes = act_sizes(self.clips)
        acts_n = len(sizes)
        self.say(f"outline in {acts_n} acts of about {sizes[0]} beats, written in parallel")
        values = dict(self._lang(), brief=self.brief, act_count=acts_n, max_cast=max_cast(self.clips), max_words=max_speech_words(self.seconds))
        messages = [{"role": "user", "content": fill(load_prompt("acts"), **values)}]
        plan = None
        for attempt in range(2):
            reply = self._ask("outline_acts", messages, answer=700 + acts_n * (500 if is_wide(self.language) else 260), scale=1.0 if attempt == 0 else 2.0, temperature=0.5)
            obj = parse_json_object(reply.text) if reply.finish != "length" else None
            acts = coerce_acts(obj, acts_n)
            if obj is not None and acts:
                plan = (obj, acts)
                break
            salvaged = salvage_acts(reply.text, acts_n) if reply.finish != "length" else None
            if salvaged:
                plan = ({"cast": salvaged[0], "locations": salvaged[1], "title": salvaged[2]}, salvaged[3])
                break
            self.say("act plan unusable" + (", asking again" if attempt == 0 else ""))
            if attempt == 0:
                self._bump("outline")
        if plan is None:
            return None
        obj, acts = plan
        if obj.get("minor_in_sexual_content") is True:
            return {"title": "", "minor": True, "cast": [], "locations": [], "beats": [], "end_states": []}
        cast, locations = _named_list(obj.get("cast")), _named_list(obj.get("locations"))
        for k, act in enumerate(acts):
            if k and not act["start"].get("place"):
                act["start"] = acts[k - 1]["end"]
        overview = "\n".join(f"{k + 1}. ({sizes[k]} beats) {a['summary']} Ends: {state_text(a['end'])}" for k, a in enumerate(acts))

        def one(k):
            act = acts[k]
            values = dict(self._lang(), brief=self.brief, cast=format_cast(cast), locations=format_locations(locations), acts=overview, act=k + 1, act_count=acts_n,
                          count=sizes[k], start_state=state_text(act["start"]) or "as the previous act ends" if k else "the story opens here; nothing came before",
                          end_state=state_text(act["end"]) or "as the act's summary ends", max_words=max_speech_words(self.seconds))
            part = self._parsed_call("outline_act", [{"role": "user", "content": fill(load_prompt("act_beats"), **values)}], self._answer_room(sizes[k]))
            if part is None:
                raise StoryWriterError(f"act {k + 1} of the outline could not be had")
            if part["minor"]:
                raise MinorContentError("the brief puts a minor in a sexual context; no story was written")
            part["cast"], part["locations"] = cast, locations
            return self._fix_count(part, sizes[k])

        try:
            parts = self._map(one, range(acts_n))
        except (MinorContentError, LLMUnavailable):
            raise
        except StoryWriterError as error:
            self.say(f"{error}; falling back to one outline call")
            self._abort.clear()
            return None
        self.outline_mode = f"{acts_n} acts"
        return {"title": str(obj.get("title") or "").strip(), "minor": False, "cast": cast, "locations": locations,
                "beats": [b for p in parts for b in p["beats"]], "end_states": [s for p in parts for s in p["end_states"]]}

    # ---- count repair: in code where it can be, one small call where it cannot

    def _fix_count(self, out: dict, n: int) -> dict:
        """Exactly ``n`` beats. Too many: a few by merging the shortest neighbours in code, more by one call that names only the merges. Too few: the longest beats are split, in parallel."""
        beats, states = list(out["beats"]), list(out.get("end_states") or [])
        states += [dict(EMPTY_STATE, present=[]) for _ in range(len(beats) - len(states))]
        got = len(beats)
        if got == n:
            return dict(out, end_states=states)
        self.say(f"outline has {got} beats, {n} wanted: repairing")
        self._bump("outline_fix")
        if got > n:
            if got - n > small_excess(n):
                beats, states = self._merge_call(beats, states, n)
            while len(beats) > n:
                i = min(range(len(beats) - 1), key=lambda k: len(beats[k]) + len(beats[k + 1]))
                beats[i:i + 2] = [beats[i].rstrip() + " " + beats[i + 1].lstrip()]
                states[i:i + 2] = [states[i + 1]]
        else:
            while len(beats) < n:
                need = n - len(beats)
                order = sorted(range(len(beats)), key=lambda k: -len(beats[k]))[:need]
                halves = self._map(lambda k: self._split_beat(beats[k], states[k]), order)
                for k, pair in sorted(zip(order, halves), key=lambda x: -x[0]):
                    beats[k:k + 1] = [pair[0][0], pair[1][0]]
                    states[k:k + 1] = [pair[0][1], pair[1][1]]
        if len(beats) != n:
            raise StoryWriterError(f"could not bring the outline to {n} beats (have {len(beats)})", work_dir=str(self.work))
        self.warnings.append(f"outline repaired from {got} to {n} beats")
        return dict(out, beats=beats, end_states=states)

    def _merge_call(self, beats: list, states: list, n: int):
        got = len(beats)
        excess = got - n
        values = dict(self._lang(), got=got, excess=excess, numbered="\n".join(f"{i}. {b}" for i, b in enumerate(beats, start=1)))
        values["clips"] = n
        reply = self._ask("outline_merge", [{"role": "user", "content": fill(load_prompt("merge"), **values)}], answer=300 + excess * (260 if is_wide(self.language) else 130),
                          temperature=0.4)
        obj = parse_json_object(reply.text) if reply.finish != "length" else None
        groups, used, removed = [], set(), 0
        for g in (obj or {}).get("merges", []) if isinstance(obj, dict) and isinstance(obj.get("merges"), list) else []:
            nums = g.get("from") if isinstance(g, dict) else None
            if not (isinstance(nums, list) and len(nums) >= 2 and all(isinstance(x, int) for x in nums)):
                continue
            nums = sorted(nums)
            text = " ".join(str(g.get("beat") or "").split())
            if (nums != list(range(nums[0], nums[0] + len(nums))) or nums[0] < 1 or nums[-1] > got or used & set(nums) or not text
                    or removed + len(nums) - 1 > excess):
                continue
            used |= set(nums)
            removed += len(nums) - 1
            groups.append((nums[0] - 1, nums[-1] - 1, text, norm_state(g.get("end_state")) if g.get("end_state") else states[nums[-1] - 1]))
        if not groups:
            return beats, states
        for first, last, text, state in sorted(groups, reverse=True):
            beats[first:last + 1] = [text]
            states[first:last + 1] = [state]
        return beats, states

    def _split_beat(self, beat: str, state: dict) -> list:
        """One beat as two ``(text, end_state)`` pairs: by the model, else by code."""
        values = dict(self._lang(), beat=beat, end_state=state_text(state) or "(not given)")
        messages = [{"role": "user", "content": fill(load_prompt("split_beat"), **values)}]
        for _ in range(2):
            reply = self._ask("split_beat", messages, answer=700, temperature=0.4)
            parsed, _why = parse_outline_reply(reply.text, reply.finish)
            if parsed and len(parsed["beats"]) == 2:
                ends = parsed["end_states"]
                ends[1] = ends[1] if state_text(ends[1]) else state
                return list(zip(parsed["beats"], ends))
        first, second = code_split(beat)
        return [(first, dict(EMPTY_STATE, present=[])), (second, state)]

    # ---- stage 2: the beats

    def _beat_file(self, i: int) -> Path:
        return self.work / "beats" / f"{i + 1:03d}.txt"

    def _have(self, i: int) -> str | None:
        path = self._beat_file(i)
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def _store(self, i: int, text: str) -> None:
        self._beat_file(i).write_text(text, encoding="utf-8")

    def write_all(self, out: dict) -> list:
        n = self.clips
        self.chunks = split_chunks(n, lambda k: self._beat_file(k).is_file(), self.chunk_beats)
        self.say(f"writing {len(self.chunks)} chunk(s) on {min(self.concurrency, max(1, len(self.chunks)))} at a time")
        self._map(lambda c: self._write_range(out, c[0], c[1]), self.chunks)
        beats = [self._have(k) for k in range(n)]
        if any(b is None for b in beats):
            raise StoryWriterError("some beats are still missing after writing", work_dir=str(self.work))
        return beats

    def _write_messages(self, out: dict, start: int, end: int, *, extra: str = "") -> list:
        count = end - start
        states = out.get("end_states") or []
        beats = out["beats"]
        if start == 0:
            opening = "OPENING STATE: beat 1 opens the story; nothing came before it."
        else:
            prev = state_text(states[start - 1]) if start - 1 < len(states) else ""
            opening = (f"OPENING STATE OF BEAT {start + 1} (the world as beat {start} left it; your first beat begins in it unless its outline line names a cut): "
                       + (prev or f"not given, take it from the outline line of beat {start}") + ".")
        neighbours = []
        if start > 0:
            neighbours.append(f"THE BEAT BEFORE YOURS (beat {start}, written by someone else at the same time; do not repeat it): {beats[start - 1]}")
        if end < len(beats):
            neighbours.append(f"THE BEAT AFTER YOURS (beat {end + 1}, written by someone else; do not run into it, end beat {end} in the state its \"ends:\" line gives): {beats[end]}")
        values = dict(self._lang(), brief=self.brief, cast=format_cast(out["cast"]), locations=format_locations(out["locations"]),
                      outline=format_outline(beats, states, start + 1, end), opening=opening, neighbours="\n".join(neighbours), first=start + 1, last=end, count=count,
                      plural="" if count == 1 else "s", beat_words=beat_words(self.seconds, self.language), max_words=max_speech_words(self.seconds), extra=extra)
        return [{"role": "user", "content": fill(load_prompt("write"), **values)}]

    def _write_range(self, out: dict, start: int, end: int) -> None:
        """Beats [start, end) written and stored. Two tries (the second larger and terser), then the range is halved; one beat that fails thrice raises."""
        for attempt in range(2):
            count = end - start
            terse = attempt > 0
            extra = ""
            if terse:
                self._bump("write")
                words = max(40, int(beat_words(self.seconds, self.language) * 0.7))
                extra = (f"- Your earlier answer was cut off or had the wrong number of beats. Write exactly {count} beat{'' if count == 1 else 's'}, "
                         f"each at most {words} words, and finish with the === END === line.")
            messages = self._write_messages(out, start, end, extra=extra)
            reply = self._ask("write", messages, answer=answer_tokens(count, self.seconds, self.language), scale=1.0 if not terse else 1.8)
            if SENTINEL_MINOR in reply.text:
                raise MinorContentError("the story would put a minor in a sexual context; no story was written")
            beats, saw_end = parse_beats(reply.text, count, truncated=reply.finish == "length")
            for k, beat in enumerate(beats):
                self._store(start + k, beat)
                self._check_beat(start + k, beat)
            if beats:
                with self._lock:
                    self.progress(1 + sum(1 for k in range(self.clips) if self._beat_file(k).is_file()), self.total_steps)
            start += len(beats)
            if start >= end:
                return
            self.say(f"beats {start + 1}-{end}: {'cut' if reply.finish == 'length' else 'wrong count'}, retrying")
        count = end - start
        if count > 1:
            self._bump("write")
            mid = start + count // 2
            self._write_range(out, start, mid)
            self._write_range(out, mid, end)
            return
        self._bump("write")
        messages = self._write_messages(out, start, end, extra="- Write ONE beat of at most 70 words and finish with the === END === line.")
        reply = self._ask("write_last", messages, answer=answer_tokens(1, self.seconds, self.language), budget=1024, scale=2.5)
        beats, _ = parse_beats(reply.text, 1, truncated=reply.finish == "length")
        if not beats:
            raise StoryWriterError(f"beat {start + 1} could not be written in three attempts (replies are kept in {self.work / 'calls'}); run again to resume", work_dir=str(self.work))
        self._store(start, beats[0])
        self._check_beat(start, beats[0])
        with self._lock:
            self.progress(1 + sum(1 for k in range(self.clips) if self._beat_file(k).is_file()), self.total_steps)

    def _check_beat(self, i: int, beat: str) -> None:
        words = quoted_words(beat)
        cap = max_speech_words(self.seconds) * 1.25
        if words > cap:
            self.warnings.append(f"beat {i + 1}: {words} spoken words, over the {int(cap)} that fit {self.seconds} s")

    def check_seams(self, out: dict, beats: list) -> list:
        """Run the cheap seam checks over the finished story; findings are kept in ``seam_issues`` and reported as warnings, never retried."""
        states = out.get("end_states") or []
        found = []
        for i in range(1, len(beats)):
            prev = states[i - 1] if i - 1 < len(states) else {}
            own = states[i] if i < len(states) else {}
            found += seam_issues(i, beats[i], out["beats"][i], prev, own, out["cast"], out["locations"])
        for name in unknown_names(beats, out["cast"], out["locations"], self.language):
            found.append(f"the name '{name}' appears in more than one beat but is in neither the cast sheet nor the locations")
        self.seam_issues = found
        self.warnings.extend("seam: " + f for f in found)
        return found

    # ---- the whole

    def run(self) -> dict:
        outline = self.outline()
        beats = self.write_all(outline)
        self.check_seams(outline, beats)
        story = "\n\n".join(beats)
        return {"story": story, "beats": beats, "outline": outline}


def plausible_count(got: int, n: int) -> bool:
    return n / 2 <= got <= n * 2


def small_excess(n: int) -> int:
    return max(2, n // 10)


def code_split(beat: str) -> list:
    """Last resort: one beat into two by code, at the sentence, comma or 'and' nearest the middle."""
    middle = len(beat) / 2
    cuts = [m.end() for m in re.finditer(r"[.!?।]\s+", beat)] or [m.end() for m in re.finditer(r",\s+|;\s+", beat)] or [m.end() for m in re.finditer(r"\s+and\s+", beat)]
    if cuts:
        cut = min(cuts, key=lambda c: abs(c - middle))
        first, second = beat[:cut].strip(), beat[cut:].strip()
        if first and second:
            return [first, second]
    return [beat, "Moments later, the moment plays out: " + beat]


# --------------------------------------------------------------------------- files


def job_hash(brief: str, clips: int, seconds: int, language: str, seed: int, model: str, budget: int) -> str:
    payload = json.dumps([brief.strip(), int(clips), int(seconds), language, int(seed), model, int(budget), VERSION], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def output_root(base=None) -> Path:
    if base is not None:
        return Path(base)
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory()) / OUTPUT_SUBDIR
    except Exception:
        return Path.cwd() / "output" / OUTPUT_SUBDIR


def write_outputs(root: Path, engine: StoryEngine, result: dict, *, model: str, seed: int, wall: float, style: str = "") -> tuple:
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}_{engine.clips}clips"
    story_path = root / f"{stem}.md"
    story_path.write_text(result["story"] + "\n", encoding="utf-8")
    calls = engine.calls
    stops: dict = {}
    for c in calls:
        stops[c["finish"]] = stops.get(c["finish"], 0) + 1
    beats = result["beats"]
    side = {
        "brief_sha1": hashlib.sha1(engine.brief.encode("utf-8")).hexdigest(), "brief_chars": len(engine.brief), "clips_requested": engine.clips, "beats_written": len(beats),
        "seconds": engine.seconds, "language": engine.language, "seed": seed, "model": model, "api_style": style, "version": VERSION,
        "title": result["outline"].get("title"), "cast": [c["name"] for c in result["outline"]["cast"]],
        "story_chars": len(result["story"]), "story_words": len(result["story"].split()), "e4_estimated_clips": int(math.ceil(e4_weight(result["story"]))),
        "spoken_words_per_beat": [quoted_words(b) for b in beats], "calls": len(calls),
        "tokens": {"prompt": sum(c["prompt_tokens"] for c in calls), "completion": sum(c["completion_tokens"] for c in calls), "thinking": sum(c["thinking_tokens"] for c in calls)},
        "stop_reasons": stops, "warnings": engine.warnings, "seam_issues": engine.seam_issues, "retries": engine.retries, "outline_mode": engine.outline_mode,
        "concurrency": engine.concurrency, "chunk_beats": engine.chunk_beats, "chunks": engine.chunks, "wall_seconds": round(wall, 1), "work_dir": str(engine.work),
        "call_log": calls, "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = root / f"{stem}.json"
    json_path.write_text(json.dumps(side, indent=1, ensure_ascii=False), encoding="utf-8")
    return story_path, json_path, side


def write_story(chat, *, brief: str, clips: int, seconds: int = DEFAULT_SECONDS, language: str = DEFAULT_LANGUAGE, seed: int = 0, model: str = "model", style: str = "",
                budget: int = THINKING_BUDGET, slot_ctx: int = 65536, out_dir=None, progress=None, say=None, chunk_beats: int | None = None, concurrency: int = DEFAULT_CONCURRENCY, two_level_min: int = TWO_LEVEL_MIN) -> dict:
    """The whole job with an injected ``chat``: validate, resume or start, write, save. Returns ``{story, story_path, json_path, sidecar}``."""
    brief = str(brief or "").strip()
    if not brief:
        raise StoryWriterError("brief is empty")
    clips = int(clips)
    if not MIN_CLIPS <= clips <= MAX_CLIPS:
        raise StoryWriterError(f"clips must be {MIN_CLIPS} to {MAX_CLIPS}, got {clips}")
    root = output_root(out_dir)
    work = root / "_work" / job_hash(brief, clips, seconds, language, seed, model, budget)
    started = time.time()
    engine = StoryEngine(chat, work, brief=brief, clips=clips, seconds=seconds, language=language, slot_ctx=slot_ctx, budget=budget, chunk_beats=chunk_beats,
                         concurrency=concurrency, two_level_min=two_level_min, progress=progress, say=say)
    result = engine.run()
    story_path, json_path, side = write_outputs(root, engine, result, model=model, seed=seed, wall=time.time() - started, style=style)
    (work / ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return {"story": result["story"], "story_path": str(story_path), "json_path": str(json_path), "sidecar": side}


# --------------------------------------------------------------------------- the ComfyUI node


def _pack():
    if not prompt_rewriter.available():
        raise StoryWriterError("the MiniMax-H3-Prompt-Rewriter-ComfyUI pack is not installed beside this node; the story writer runs on the model server it opens")
    return (prompt_rewriter._mod("nodes"), prompt_rewriter._mod("paths"), prompt_rewriter._mod("mtmd_engine"), prompt_rewriter._mod("server_engine"),
            prompt_rewriter._mod("runner"))


def server_slots(concurrency: int) -> int:
    """Slots of the model server this node starts: as many as the writer sends requests at once, at most ``e4_engine.MAX_WORKERS`` (4 slots of ``slot_ctx`` fit the card with the model)."""
    return max(1, min(e4_engine.MAX_WORKERS, int(concurrency)))


@contextlib.contextmanager
def open_model(writer_label: str, budget: int, *, slot_ctx: int, concurrency: int = 1, say=None):
    """The model server for the story writer: the rewriter pack's own session (a vision model from its list, on ninfer-serve or llama-server) or Strata, closed on exit.

    The local server is started with ``server_slots(concurrency)`` slots and a pool of ``slot_ctx`` per slot (ninfer-serve: ``--max-concurrency`` = slots, ``--max-context`` = the pool
    divided by the slots), so the writer's parallel requests run at the same time instead of queueing behind one slot. The endpoint says how many slots it has (``slots``)."""
    say = say or (lambda message: None)
    nodes, paths, mtmd, server_engine, runner = _pack()
    if not writer_label or writer_label.startswith("("):
        raise StoryWriterError("writer_model: pick a model from the list (the rewriter pack's model list)")
    strata = writer_label == strata_backend.LABEL
    model_path, mmproj_path = ("", "") if strata else prompt_rewriter._vision_files(nodes, paths, writer_label)
    if not strata and not model_path:
        raise StoryWriterError("the story writer needs a model that runs on a server (a vision model from the pack's list, or Strata); "
                               f"'{writer_label}' does not. Pick another writer_model.")
    reasoning = {"enabled": True, "budget": budget, "message": BUDGET_MESSAGE}
    slots = server_slots(concurrency)
    try:
        with contextlib.ExitStack() as stack:
            if strata:
                say("starting Strata")
                server = stack.enter_context(strata_backend.open_strata(budget, adopt=getattr(runner, "_adopt", None), on_wait=lambda s: say(f"waiting for Strata ({s:.0f} s)")))
            else:
                say(f"loading {os.path.basename(model_path)}")
                server = stack.enter_context(mtmd.session(model_path, mmproj_path, assets=0, attachments=1, gpu_layers=-1, n_ctx=0, device="auto", backend="auto",
                                                         auto_download=True, progress=None, slots=slots, force=True, reasoning=reasoning, pool_ctx=slot_ctx * slots))
            if server is None:
                raise StoryWriterError(f"the model server for '{writer_label}' did not start (see the ComfyUI log: minimax_h3_rewriter.server_engine)")
            endpoint = e4_engine.endpoint_of(server, model_path=model_path, strata=strata, is_ninfer=server_engine.is_ninfer)
            endpoint["label"] = os.path.basename(model_path) or writer_label
            if not strata:
                endpoint["slots"] = slots
            yield endpoint, runner
    finally:
        vram_release.release("h3 story writer finished")


class H3StoryWriter:
    """Brief to full story. Feed its ``story`` to the Master Extender's story input (story_engine=e4.10)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "brief": ("STRING", {"default": "", "multiline": True, "tooltip": "The idea, however short: who, where, what happens, the genre and tone. Spoken lines you already want are kept word for word."}),
            "clips": ("INT", {"default": 8, "min": MIN_CLIPS, "max": MAX_CLIPS, "tooltip": "How many clips the story should become. The story has exactly this many beats, one distinct dramatic moment each, so E4 (auto mode) plans about this many clips."}),
            "seconds": ("INT", {"default": DEFAULT_SECONDS, "min": 4, "max": 30, "tooltip": "Seconds of screen time per clip. Sizes each beat and the most speech it carries (about 2 words per second at most, leaving room for action)."}),
            "language": ("STRING", {"default": DEFAULT_LANGUAGE, "tooltip": "The language of every spoken line and of the narration (same meaning as the Extender's e4_language). A language not written in Latin script is written in its own script."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "control_after_generate": True, "tooltip": "Part of the key under which a half-finished run is resumed; change it to write a different version of the same brief. Leave the control on fixed: ComfyUI then reuses the story until an input changes."}),
            "writer_model": (prompt_rewriter.writer_choices(), {"tooltip": "The rewriter pack's model list, the same as the Extender's rewrite_writer_model: a vision model on ninfer-serve or llama-server, or Strata. The server is started for this node and closed after it, so the render that follows has the GPU."}),
            "thinking_budget": (prompt_rewriter.REASONING_BUDGETS, {"default": "4096", "tooltip": "Thinking tokens at most per call, on top of the answer."}),
        }, "optional": {
            "concurrency": ("INT", {"default": DEFAULT_CONCURRENCY, "min": 1, "max": MAX_CONCURRENCY, "tooltip": "Requests sent to the model server at the same time: the chunks of beats are written in parallel (each from the outline and its neighbours' end states, not from each other's prose), and so are the acts of a long outline. The model server this node starts gets that many slots, at most 4 (4 slots of 65536 tokens fit the card with the model); a higher value is lowered to the slots the server has. 1 writes one chunk after another."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("story",)
    FUNCTION = "write"
    CATEGORY = "MiniMax H3 Master"
    OUTPUT_NODE = False

    @classmethod
    def VALIDATE_INPUTS(cls, thinking_budget=None):
        return True

    def write(self, brief, clips, seconds, language, seed, writer_model, thinking_budget="4096", concurrency=DEFAULT_CONCURRENCY):
        budget = int(prompt_rewriter.reasoning_budget(thinking_budget))
        slot_ctx = e4_engine.slot_ctx()
        bar = None
        try:
            import comfy.utils

            bar = comfy.utils.ProgressBar(int(clips) + 1)
        except Exception:
            bar = None
        last = {"n": 0}

        def progress(done, total):
            if bar is not None and done > last["n"]:
                bar.update(done - last["n"])
                last["n"] = done

        def say(message):
            _LOG.info("H3 Story Writer: %s", message)

        with open_model(str(writer_model), budget, slot_ctx=slot_ctx, concurrency=concurrency, say=say) as (endpoint, runner):
            concurrency = min(int(concurrency), int(endpoint.get("slots") or concurrency))
            chat = ServerChat(endpoint, interrupted=getattr(runner, "interrupted", None))
            result = write_story(chat, brief=brief, clips=clips, seconds=seconds, language=language, seed=seed, model=endpoint["model"], style=endpoint["style"],
                                 budget=budget, slot_ctx=slot_ctx, progress=progress, say=say, concurrency=concurrency)
        side = result["sidecar"]
        print(f"[H3 Story Writer] {side['beats_written']} beats, {side['story_words']} words, ~{side['e4_estimated_clips']} clips by E4's size estimate, "
              f"{side['wall_seconds']:.0f} s -> {result['story_path']}", flush=True)
        for warning in side["warnings"]:
            print(f"[H3 Story Writer] warning: {warning}", flush=True)
        return (result["story"],)


NODE_CLASS_MAPPINGS = {"H3StoryWriter": H3StoryWriter}
NODE_DISPLAY_NAME_MAPPINGS = {"H3StoryWriter": "H3 Story Writer (brief to story)"}
