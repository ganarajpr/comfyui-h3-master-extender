"""The H3 Story Writer node: parsing, count checks and repairs, chunking, truncation handling, resume, the HTTP wire, the node.

No model, no ComfyUI, no torch: a scripted ``chat`` stands in for the model, and a local HTTP stub for the server.
"""

import contextlib
import json
import re
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import story_writer as sw  # noqa: E402


def end_state(i=0):
    return {"place": "Hall", "present": ["Ana"], "state": f"Ana wears her grey coat, door {i} open"}


def outline_reply(n, *, cast=None, minor=False, extra_beats=0, plain=False):
    lines = [f"Beat line {i}: Ana crosses the hall and says goodbye to the door number {i}." for i in range(1, n + 1 + extra_beats)]
    beats = lines if plain else [{"beat": line, "end_state": end_state(i)} for i, line in enumerate(lines, start=1)]
    return json.dumps({"title": "T", "minor_in_sexual_content": minor, "cast": cast or [{"name": "Ana", "look": "a tall woman in a grey coat"}],
                       "locations": [{"name": "Hall", "look": "a long dim hall"}], "beats": [] if minor else beats})


def beat_text(i, words=30):
    return f"Ana walks through the hall number {i} and says, \"Door {i} is open.\" " + " ".join(["She waits."] * (words // 2))


def writer_reply(first, last, *, drop_last=0, end=True, number_offset=0):
    out = []
    for i in range(first, last + 1 - drop_last):
        out.append(f"=== {i + number_offset} ===\n{beat_text(i)}\n")
    if end:
        out.append("=== END ===")
    return "\n".join(out)


class Script:
    """A scripted chat: ``handler(kind, messages, call_no) -> str | Reply``; every call is recorded."""

    def __init__(self, handler, delay=0.0):
        self.handler = handler
        self.calls = []
        self.delay = delay
        self.lock = threading.Lock()
        self.live = 0
        self.peak = 0

    def __call__(self, messages, *, max_tokens, budget, kind="", temperature=0.6):
        with self.lock:
            self.calls.append({"kind": kind, "messages": messages, "max_tokens": max_tokens, "budget": budget})
            number = len(self.calls)
            self.live += 1
            self.peak = max(self.peak, self.live)
        try:
            if self.delay:
                time.sleep(self.delay)
            result = self.handler(kind, messages, number)
        finally:
            with self.lock:
                self.live -= 1
        return result if isinstance(result, sw.Reply) else sw.Reply(result, "stop", 100, 200, 50, 0.1)

    def kinds(self):
        return [c["kind"] for c in self.calls]


def span(messages):
    m = re.search(r"WHAT TO WRITE: beats (\d+) to (\d+)", messages[0]["content"])
    return int(m.group(1)), int(m.group(2))


def acts_reply(prompt):
    count = int(re.search(r"Exactly (\d+) acts", prompt).group(1))
    return json.dumps({"title": "T", "minor_in_sexual_content": False, "cast": [{"name": "Ana", "look": "tall"}], "locations": [{"name": "Hall", "look": "dim"}],
                       "acts": [{"summary": f"Act {k} happens.", "start_state": end_state(), "end_state": end_state()} for k in range(1, count + 1)]})


def good_handler(n):
    def handler(kind, messages, call_no):
        if kind == "outline":
            return outline_reply(n)
        if kind == "outline_acts":
            return acts_reply(messages[0]["content"])
        if kind == "outline_act":
            return outline_reply(int(re.search(r"EXACTLY (\d+) beats", messages[0]["content"]).group(1)))
        first, last = span(messages)
        return writer_reply(first, last)
    return handler


def make_engine(chat, tmp, *, clips=8, **kw):
    kw.setdefault("two_level_min", 999)
    return sw.StoryEngine(chat, Path(tmp) / "work", brief="A woman leaves a house.", clips=clips, **kw)


class ParsingTests(unittest.TestCase):
    def test_json_object_variants(self):
        self.assertEqual(sw.parse_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(sw.parse_json_object('```json\n{"a": [1, 2,]}\n```'), {"a": [1, 2]})
        self.assertEqual(sw.parse_json_object('Sure! Here it is: {"a": {"b": "}"}} hope that helps'), {"a": {"b": "}"}})
        self.assertEqual(sw.parse_json_object('<think>{"x":0}</think>{"a": 2}'), {"a": 2})
        self.assertIsNone(sw.parse_json_object("no json at all"))
        self.assertIsNone(sw.parse_json_object('{"a": '))
        self.assertEqual(sw.parse_json_object('{"a": "line one\nline two"}'), {"a": "line one\nline two"})

    def test_outline_coercion(self):
        out = sw.coerce_outline({"title": "x", "cast": [{"name": "Ana", "look": "tall"}, "Bo", {"look": "nameless"}], "beats": ["one  line", "", 3, "two"]})
        self.assertEqual(out["beats"], ["one line", "two"])
        self.assertEqual([c["name"] for c in out["cast"]], ["Ana", "Bo"])
        self.assertIsNone(sw.coerce_outline({"beats": []}))
        self.assertIsNone(sw.coerce_outline({"cast": []}))
        self.assertTrue(sw.coerce_outline({"minor_in_sexual_content": True, "beats": []})["minor"])

    def test_outline_under_alternate_keys_and_shapes(self):
        long = "Ana crosses the hall and leaves the house behind her"
        for obj in ({"outline": [long, long + " 2"]}, {"scenes": [{"text": long}, {"description": long + " 2"}]}, {"story_beats": {"2": long + " 2", "1": long}},
                    {"x": 1, "moments": [{"beat": long, "end_state": "wet"}, {"summary": long}]}, [long, long + " 2"], {"things": [long, long + " 2"]},
                    {"beats": "1. " + long + "\n2. " + long + " 2"}):
            out = sw.coerce_outline(obj)
            self.assertEqual(len(out["beats"]), 2, obj)
            self.assertEqual(len(out["end_states"]), 2)
        self.assertEqual(sw.coerce_outline({"beats": {"2": "second line here", "1": "first line here"}})["beats"], ["first line here", "second line here"])
        out = sw.coerce_outline({"beats": [{"beat": long, "end_state": {"place": "Hall", "present": "Ana, Bo", "state": ["wet", "cold"]}}, long]})
        self.assertEqual(out["end_states"][0], {"place": "Hall", "present": ["Ana", "Bo"], "state": "wet; cold"})
        self.assertEqual(out["end_states"][1], {"place": "", "present": [], "state": ""})

    def test_straight_quotes_inside_beats_are_salvaged(self):
        broken = ('Here: {"title": "T", "cast": [{"name": "Ana", "look": "tall"}], "locations": [{"name": "Hall", "look": "dim"}], "beats": ['
                  '{"beat": "Ana opens the door and says, "Hello there, Bo." Bo nods slowly.", "end_state": {"place": "Hall", "present": ["Ana", "Bo"], "state": "coat on"}}, '
                  '{"beat": "Bo answers, "Come in", and Ana steps into the hall, shutting the door.", "end_state": {"place": "Hall", "present": ["Ana"], "state": "door shut"}}]}')
        self.assertNotIn("beats", sw.parse_json_object(broken) or {})
        out, why = sw.parse_outline_reply(broken)
        self.assertEqual(why, "")
        self.assertEqual(len(out["beats"]), 2)
        self.assertIn("\"Hello there, Bo.\"", out["beats"][0])
        self.assertEqual(out["end_states"][1]["state"], "door shut")
        self.assertEqual(out["cast"][0]["name"], "Ana")

    def test_unusable_reasons(self):
        self.assertIn("empty", sw.parse_outline_reply("<think>x</think>")[1])
        self.assertIn("cut", sw.parse_outline_reply('{"beats": ["a", "b"', "length")[1])
        self.assertIn("no parseable JSON", sw.parse_outline_reply("I refuse")[1])
        self.assertIn("without a beats list", sw.parse_outline_reply('{"title": "x"}')[1])
        self.assertEqual(sw.parse_outline_reply(outline_reply(3), "length")[1], "")

    def test_beats_are_cut_at_markers_and_cleaned(self):
        text = "=== 1 ===\n**Beat 1:** Ana opens the door.\nShe says, \"Hello there.\"\n\n=== 2 ===\n## Clip 2\n- Bo waves from the stairs with a very long sentence here.\n=== END ==="
        beats, saw_end = sw.parse_beats(text, 2, truncated=False)
        self.assertTrue(saw_end)
        self.assertEqual(beats[0], "Ana opens the door. She says, \"Hello there.\"")
        self.assertTrue(beats[1].startswith("Bo waves"))
        self.assertNotIn("\n", beats[0])
        self.assertNotIn("Clip", beats[1])

    def test_end_marker_after_every_beat(self):
        text = "".join(f"=== {i} ===\n{beat_text(i)}\n\n=== END ===\n\n" for i in range(1, 4))
        beats, saw_end = sw.parse_beats(text, 3, truncated=False)
        self.assertEqual(beats, [beat_text(i) for i in range(1, 4)])
        self.assertTrue(saw_end)
        self.assertEqual(len(sw.parse_beats(text.rsplit("=== END ===", 1)[0], 3, truncated=True)[0]), 2)

    def test_truncated_reply_loses_only_the_last_block(self):
        text = writer_reply(1, 4, end=False)
        beats, saw_end = sw.parse_beats(text, 4, truncated=True)
        self.assertFalse(saw_end)
        self.assertEqual(len(beats), 3)
        beats, _ = sw.parse_beats(text, 4, truncated=False)
        self.assertEqual(len(beats), 4)
        self.assertEqual(len(sw.parse_beats(writer_reply(1, 4), 4, truncated=True)[0]), 4)

    def test_extra_beats_ignored_and_short_block_stops_the_run(self):
        self.assertEqual(len(sw.parse_beats(writer_reply(1, 6), 3, truncated=False)[0]), 3)
        text = "=== 1 ===\n" + beat_text(1) + "\n=== 2 ===\nok\n=== 3 ===\n" + beat_text(3) + "\n=== END ==="
        self.assertEqual(len(sw.parse_beats(text, 3, truncated=False)[0]), 1)
        self.assertEqual(sw.parse_beats("just prose, no markers", 2, truncated=False), ([], False))

    def test_spoken_words(self):
        text = 'Ana says, "Come here now." Bo answers, “No way at all.” He doesn\'t move.'
        self.assertEqual(sw.quoted_words(text), 7)
        self.assertEqual(sw.quoted_lines(text), ["Come here now.", "No way at all."])

    def test_code_split(self):
        a, b = sw.code_split("Ana enters the hall. She finds the letter on the table. She reads it twice.")
        self.assertTrue(a.endswith(".") and b)
        self.assertEqual(len(sw.code_split("short")), 2)

    def test_sizes(self):
        self.assertTrue(sw.is_wide("Hindi"))
        self.assertFalse(sw.is_wide("english"))
        self.assertLess(sw.beat_words(15, "Hindi"), sw.beat_words(15, "English"))
        self.assertEqual(sw.max_speech_words(15), 30)
        self.assertEqual(sw.max_cast(6), 3)
        self.assertEqual(sw.max_cast(60), 8)
        self.assertAlmostEqual(sw.e4_weight("x" * 700), 1.0)
        self.assertAlmostEqual(sw.e4_weight("अ" * 380), 1.0)

    def test_output_room_follows_the_prompt(self):
        small = [{"role": "user", "content": "hi"}]
        big = [{"role": "user", "content": "word " * 20000}]
        self.assertGreater(sw.output_room(small, 65536), sw.output_room(big, 65536))
        self.assertGreaterEqual(sw.output_room(big, 16384), sw.MIN_OUTPUT)
        wide = [{"role": "user", "content": "अ" * 10000}]
        self.assertLess(sw.output_room(wide, 65536), sw.output_room([{"role": "user", "content": "a" * 10000}], 65536))


class OutlineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def engine(self, handler, clips=6):
        chat = Script(handler)
        return chat, make_engine(chat, self.tmp.name, clips=clips)

    def test_exact_count_is_one_call(self):
        chat, eng = self.engine(lambda k, m, n: outline_reply(6))
        out = eng.outline()
        self.assertEqual(len(out["beats"]), 6)
        self.assertEqual(chat.kinds(), ["outline"])
        self.assertEqual(out["cast"][0]["name"], "Ana")

    def test_small_excess_is_merged_in_code_without_a_call(self):
        chat, eng = self.engine(lambda k, m, n: outline_reply(6, extra_beats=1))
        out = eng.outline()
        self.assertEqual((len(out["beats"]), len(out["end_states"])), (6, 6))
        self.assertEqual(chat.kinds(), ["outline"])
        self.assertTrue(any("repaired from 7 to 6" in w for w in eng.warnings))
        self.assertEqual(out["end_states"][-1], end_state(7))

    def test_large_excess_is_one_merge_call_naming_only_the_merges(self):
        def handler(kind, messages, n):
            if kind == "outline_merge":
                return json.dumps({"merges": [{"from": [2, 3, 4], "beat": "Ana crosses the hall, stops, and answers the door.", "end_state": end_state(99)},
                                              {"from": [10, 11], "beat": "Ana leaves for good.", "end_state": end_state(98)}]})
            return outline_reply(20, extra_beats=3)
        chat, eng = self.engine(handler, clips=20)
        out = eng.outline()
        self.assertEqual(chat.kinds(), ["outline", "outline_merge"])
        self.assertEqual((len(out["beats"]), len(out["end_states"])), (20, 20))
        self.assertEqual(out["beats"][1], "Ana crosses the hall, stops, and answers the door.")
        self.assertEqual(out["end_states"][1], end_state(99))
        self.assertEqual(out["beats"][7], "Ana leaves for good.")
        self.assertIn("3 too many", chat.calls[1]["messages"][0]["content"])
        self.assertEqual(eng.retries["outline"], 0)

    def test_bad_merge_answer_falls_back_to_code(self):
        def handler(kind, messages, n):
            if kind == "outline_merge":
                return json.dumps({"merges": [{"from": [1, 3], "beat": "x"}, {"from": [2, 3, 4, 5, 6, 7, 8, 9], "beat": "too many removed"}]})
            return outline_reply(20, extra_beats=7)
        chat, eng = self.engine(handler, clips=20)
        out = eng.outline()
        self.assertEqual(len(out["beats"]), 20)
        self.assertEqual(chat.kinds(), ["outline", "outline_merge"])

    def test_too_few_splits_the_longest_beats_in_parallel(self):
        def handler(kind, messages, n):
            if kind == "split_beat":
                return json.dumps({"beats": [{"beat": "first half moment", "end_state": end_state(1)}, {"beat": "second half moment", "end_state": end_state(2)}]})
            return outline_reply(4)
        chat, eng = self.engine(handler)
        out = eng.outline()
        self.assertEqual((len(out["beats"]), len(out["end_states"])), (6, 6))
        self.assertEqual(sorted(chat.kinds()), ["outline", "split_beat", "split_beat"])
        self.assertEqual(out["beats"].count("first half moment"), 2)

    def test_every_beat_split_when_half_are_missing(self):
        def handler(kind, messages, n):
            if kind == "split_beat":
                return json.dumps({"beats": ["first half moment of it", "second half moment of it"]})
            return outline_reply(3)
        chat, eng = self.engine(handler, clips=6)
        self.assertEqual(len(eng.outline()["beats"]), 6)
        self.assertEqual(chat.kinds().count("split_beat"), 3)

    def test_split_that_fails_falls_back_to_code(self):
        def handler(kind, messages, n):
            return "not json" if kind == "split_beat" else outline_reply(5)
        chat, eng = self.engine(handler)
        out = eng.outline()
        self.assertEqual((len(out["beats"]), len(out["end_states"])), (6, 6))
        self.assertEqual(chat.kinds().count("split_beat"), 2)

    def test_wildly_wrong_count_is_planned_in_parts(self):
        def handler(kind, messages, n):
            text = messages[0]["content"]
            if "FIRST PART" in text or "SECOND PART" in text:
                return outline_reply(3)
            return outline_reply(20)
        chat, eng = self.engine(handler)
        self.assertEqual(len(eng.outline()["beats"]), 6)
        self.assertEqual(chat.kinds(), ["outline"] * 3)

    def test_straight_quotes_in_beats_cost_no_retry(self):
        broken = ('{"title": "T", "cast": [{"name": "Ana", "look": "tall"}], "locations": [{"name": "Hall", "look": "dim"}], "beats": [' + ", ".join(
            '{"beat": "Ana says, "Number %d, come here" and walks the hall slowly.", "end_state": {"place": "Hall", "present": ["Ana"], "state": "coat"}}' % i
            for i in range(1, 7)) + "]}")
        chat, eng = self.engine(lambda k, m, n: broken)
        out = eng.outline()
        self.assertEqual(len(out["beats"]), 6)
        self.assertEqual(chat.kinds(), ["outline"])

    def test_plain_string_beats_still_work(self):
        chat, eng = self.engine(lambda k, m, n: outline_reply(6, plain=True))
        out = eng.outline()
        self.assertEqual(len(out["end_states"]), 6)
        self.assertEqual(out["end_states"][0]["place"], "")

    def test_two_level_outline_for_long_stories(self):
        chat = Script(good_handler(40), delay=0.03)
        eng = make_engine(chat, self.tmp.name, clips=40, two_level_min=31, concurrency=4)
        out = eng.outline()
        self.assertEqual(sorted(chat.kinds()), ["outline_act"] * 4 + ["outline_acts"])
        self.assertEqual(chat.kinds()[0], "outline_acts")
        self.assertEqual((len(out["beats"]), len(out["end_states"])), (40, 40))
        self.assertEqual(eng.outline_mode, "4 acts")
        self.assertGreater(chat.peak, 1)
        act = [c for c in chat.calls if "YOUR ACT: act 3." in c["messages"][0]["content"]][0]["messages"][0]["content"]
        self.assertIn("EXACTLY 10 beats", act)
        self.assertIn("act 3 of 4", act)
        self.assertIn("It begins in this state: place: Hall", act)

    def test_two_level_act_with_wrong_count_is_repaired_and_failure_falls_back(self):
        def handler(kind, messages, n):
            if kind == "outline_act":
                return outline_reply(11)
            return good_handler(40)(kind, messages, n)
        chat = Script(handler)
        eng = make_engine(chat, self.tmp.name, clips=40, two_level_min=31)
        self.assertEqual(len(eng.outline()["beats"]), 40)

        def failing(kind, messages, n):
            if kind == "outline_act":
                return "garbage"
            return good_handler(40)(kind, messages, n)
        chat = Script(failing)
        eng = sw.StoryEngine(chat, Path(self.tmp.name) / "w2", brief="b", clips=40, two_level_min=31)
        self.assertEqual(len(eng.outline()["beats"]), 40)
        self.assertEqual(eng.outline_mode, "whole")

    def test_minor_in_an_act_plan_raises(self):
        def handler(kind, messages, n):
            return json.dumps({"minor_in_sexual_content": True, "acts": [{"summary": "x"}] * 4}) if kind == "outline_acts" else "x"
        with self.assertRaises(sw.MinorContentError):
            make_engine(Script(handler), self.tmp.name, clips=40, two_level_min=31).outline()

    def test_unparseable_twice_then_two_parts(self):
        def handler(kind, messages, n):
            text = messages[0]["content"]
            if "FIRST PART" in text:
                return outline_reply(3)
            if "SECOND PART" in text:
                return outline_reply(3, cast=[{"name": "Bo", "look": "short"}])
            return "I cannot produce JSON"
        chat, eng = self.engine(handler)
        out = eng.outline()
        self.assertEqual(len(out["beats"]), 6)
        self.assertEqual({c["name"] for c in out["cast"]}, {"Ana", "Bo"})
        self.assertEqual(chat.kinds(), ["outline"] * 4)

    def test_cut_outline_is_retried_larger(self):
        def handler(kind, messages, n):
            return sw.Reply('{"beats": ["a", "b"', "length") if n == 1 else outline_reply(6)
        chat, eng = self.engine(handler)
        eng.outline()
        self.assertGreater(chat.calls[1]["max_tokens"], chat.calls[0]["max_tokens"])

    def test_nothing_usable_raises_and_keeps_replies(self):
        chat, eng = self.engine(lambda k, m, n: "nope", clips=2)
        with self.assertRaises(sw.StoryWriterError):
            eng.outline()
        self.assertTrue(list((Path(self.tmp.name) / "work" / "calls").glob("*.json")))

    def test_minor_flag_raises(self):
        chat, eng = self.engine(lambda k, m, n: outline_reply(6, minor=True))
        with self.assertRaises(sw.MinorContentError):
            eng.outline()

    def test_outline_is_stored_and_reused(self):
        chat, eng = self.engine(lambda k, m, n: outline_reply(6))
        eng.outline()
        again = make_engine(Script(lambda k, m, n: self.fail("no call expected")), self.tmp.name, clips=6)
        self.assertEqual(len(again.outline()["beats"]), 6)

    def test_prompt_carries_the_rules(self):
        chat, eng = self.engine(lambda k, m, n: outline_reply(6))
        eng.outline()
        text = chat.calls[0]["messages"][0]["content"]
        for needle in ("EXACTLY 6 beats", "about 15 seconds", "A lighthouse" if False else "A woman leaves a house.", "at most 3 named", "minor_in_sexual_content"):
            self.assertIn(needle, text)
        self.assertNotIn("{{", text)


class WriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_chunks_are_balanced_parallel_and_independent_of_each_others_prose(self):
        chat = Script(good_handler(20))
        eng = make_engine(chat, self.tmp.name, clips=20, chunk_beats=8, concurrency=6)
        result = eng.run()
        self.assertEqual(chat.kinds()[0], "outline")
        spans = sorted(span(c["messages"]) for c in chat.calls[1:])
        self.assertEqual(spans, [(1, 7), (8, 14), (15, 20)])
        paragraphs = result["story"].split("\n\n")
        self.assertEqual(len(paragraphs), 20)
        self.assertEqual(paragraphs[7], beat_text(8))
        self.assertNotIn("===", result["story"])
        self.assertNotRegex(result["story"], r"(?i)\b(beat|clip) \d")
        second = [c for c in chat.calls[1:] if span(c["messages"]) == (8, 14)][0]["messages"][0]["content"]
        self.assertNotIn("is open.", second)
        self.assertNotIn("THE LAST PARAGRAPHS", second)
        self.assertIn("OPENING STATE OF BEAT 8", second)
        self.assertIn("door 7 open", second)
        self.assertIn("THE BEAT BEFORE YOURS (beat 7", second)
        self.assertIn("THE BEAT AFTER YOURS (beat 15", second)
        self.assertIn("ends: place: Hall; present: Ana; visible state: Ana wears her grey coat, door 14 open", second)
        self.assertIn("<== WRITE THIS ONE", second)
        first = [c for c in chat.calls[1:] if span(c["messages"]) == (1, 7)][0]["messages"][0]["content"]
        self.assertIn("beat 1 opens the story", first)
        self.assertNotIn("THE BEAT BEFORE", first)
        last = [c for c in chat.calls[1:] if span(c["messages"]) == (15, 20)][0]["messages"][0]["content"]
        self.assertNotIn("THE BEAT AFTER", last)
        self.assertEqual(sorted(eng.chunks), [(0, 7), (7, 14), (14, 20)])

    def test_split_chunks_plan(self):
        self.assertEqual(sw.split_chunks(20, lambda k: False, 5), [(0, 5), (5, 10), (10, 15), (15, 20)])
        self.assertEqual(sw.split_chunks(11, lambda k: False, 5), [(0, 4), (4, 8), (8, 11)])
        self.assertEqual(sw.split_chunks(10, lambda k: k in (0, 1, 5), 4), [(2, 5), (6, 10)])
        self.assertEqual(sw.split_chunks(3, lambda k: True, 5), [])
        self.assertEqual(sw.act_sizes(40), [10, 10, 10, 10])
        self.assertEqual(sw.act_sizes(45), [9, 9, 9, 9, 9])
        self.assertEqual(sum(sw.act_sizes(33)), 33)

    def test_chunk_size_follows_concurrency(self):
        self.assertEqual([sw.auto_chunk_beats(40, c) for c in (1, 4, 6, 8)], [8, 8, 7, 5])
        self.assertEqual((sw.auto_chunk_beats(20, 8), sw.auto_chunk_beats(20, 6), sw.auto_chunk_beats(3, 6), sw.auto_chunk_beats(60, 6)), (4, 4, 4, 8))
        self.assertEqual(sw.StoryEngine(Script(good_handler(2)), Path(self.tmp.name) / "a", brief="b", clips=40, concurrency=6).chunk_beats, 7)
        self.assertEqual(sw.StoryEngine(Script(good_handler(2)), Path(self.tmp.name) / "b", brief="b", clips=40, chunk_beats=3).chunk_beats, 3)

    def test_limiter_caps_requests_in_flight(self):
        for setting, expect_peak in ((1, 1), (3, 3), (50, sw.MAX_CONCURRENCY)):
            with tempfile.TemporaryDirectory() as tmp:
                chat = Script(good_handler(40), delay=0.03)
                eng = make_engine(chat, tmp, clips=40, chunk_beats=2, concurrency=setting)
                eng.run()
                self.assertLessEqual(chat.peak, expect_peak)
                if setting > 1:
                    self.assertGreaterEqual(chat.peak, 2)
        self.assertEqual(make_engine(Script(good_handler(2)), self.tmp.name, clips=2, concurrency=0).concurrency, 1)

    def test_story_is_in_beat_order_whatever_finishes_first(self):
        def handler(kind, messages, n):
            if kind == "outline":
                return outline_reply(20)
            first, last = span(messages)
            time.sleep(0.2 - first * 0.01)
            return writer_reply(first, last)
        result = make_engine(Script(handler), self.tmp.name, clips=20, chunk_beats=4, concurrency=8).run()
        self.assertEqual(result["beats"], [beat_text(i) for i in range(1, 21)])

    def test_a_failure_stops_queued_chunks(self):
        def handler(kind, messages, n):
            if kind == "outline":
                return outline_reply(20)
            first, last = span(messages)
            if first == 1:
                return sw.SENTINEL_MINOR
            time.sleep(0.05)
            return writer_reply(first, last)
        chat = Script(handler)
        with self.assertRaises(sw.MinorContentError):
            make_engine(chat, self.tmp.name, clips=20, chunk_beats=2, concurrency=2).run()
        self.assertLess(len(chat.calls), 11)

    def test_seam_problems_are_reported_not_retried(self):
        def handler(kind, messages, n):
            if kind == "outline":
                return outline_reply(4)
            first, last = span(messages)
            return writer_reply(first, last).replace(beat_text(3), "In the Kitchen, Bob walks in and says, \"Door 3 is open.\" " + "She waits. " * 15)
        chat = Script(handler)
        eng = make_engine(chat, self.tmp.name, clips=4, chunk_beats=2)
        out = eng.outline()
        out["locations"].append({"name": "Kitchen", "look": "small"})
        out["cast"].append({"name": "Bob", "look": "short"})
        beats = eng.write_all(out)
        found = eng.check_seams(out, beats)
        self.assertEqual(chat.kinds().count("write"), 2)
        self.assertTrue(any("opens in 'Kitchen' but beat 2 ended in 'Hall'" in f for f in found))
        self.assertTrue(any("Bob, who was not present" in f for f in found))
        self.assertTrue(all(w in eng.warnings for w in ["seam: " + f for f in found]))

    def test_seam_issues_unit(self):
        cast = [{"name": "Ana Ruiz", "look": ""}, {"name": "Bo", "look": ""}]
        places = [{"name": "Hall", "look": ""}, {"name": "Kitchen", "look": ""}]
        prev = {"place": "Hall", "present": ["Ana Ruiz"], "state": "wet"}
        none = {}
        ok = "Ana stands in the Hall, still wet, and looks at the door."
        self.assertEqual(sw.seam_issues(3, ok, "Ana waits.", prev, none, cast, places), [])
        self.assertEqual(sw.seam_issues(0, "In the Kitchen, Bo laughs.", "x", prev, none, cast, places), [])
        self.assertEqual(sw.seam_issues(3, "In the Kitchen Ana dries her hair.", "Ana moves to the Kitchen.", prev, none, cast, places), [])
        self.assertEqual(sw.seam_issues(3, "In the Kitchen Ana dries her hair.", "Ana dries her hair.", prev, {"place": "Kitchen"}, cast, places), [])
        self.assertEqual(len(sw.seam_issues(3, "In the Kitchen Ana dries her hair.", "Ana dries her hair.", prev, none, cast, places)), 1)
        self.assertEqual(len(sw.seam_issues(3, "Bo steps in and nods to Ana.", "Ana waits.", prev, none, cast, places)), 1)
        self.assertEqual(sw.seam_issues(3, "Bo steps in and nods to Ana.", "Bo arrives and Ana waits.", prev, none, cast, places), [])
        self.assertEqual(sw.seam_issues(3, "Bo steps in and nods to Ana.", "Ana waits.", prev, {"present": ["Ana Ruiz", "Bo"]}, cast, places), [])
        self.assertEqual(sw.seam_issues(3, "Bo steps in.", "x", {}, none, cast, places), [])

    def test_mismatched_brackets_are_repaired(self):
        self.assertEqual(sw.parse_json_object('{"a": {"b": [1, 2}, "d": 2}'), {"a": {"b": [1, 2], "d": 2}})
        self.assertIsNone(sw.parse_json_object('{"a": [1, 2', repair=False))
        self.assertEqual(sw.parse_json_object('{"a": [1, 2'), {"a": [1, 2]})
        self.assertEqual(sw.parse_json_object('{"a": "x]"} trailing } junk'), {"a": "x]"})

    def test_missing_final_brace_does_not_return_a_nested_object(self):
        cut = '{"title": "T", "cast": [{"name": "Ana", "look": "tall"}], "beats": ["one line is here and long", "two line is here and long"]'
        self.assertEqual(list(sw.parse_json_object(cut)), ["title", "cast", "beats"])
        out, why = sw.parse_outline_reply(cut)
        self.assertEqual((len(out["beats"]), why), (2, ""))
        self.assertEqual(sw.parse_json_object('Sure {not json}: {"a": 1}'), {"a": 1})
        self.assertEqual(sw.parse_json_object(cut, repair=False), {"name": "Ana", "look": "tall"})

    def test_act_plan_with_broken_structure_is_salvaged(self):
        broken = ('{"title": "T", "cast": [{"name": "Ana", "look": "tall"}], "locations": [{"name": "Hall", "look": "dim"}], "acts": ['
                  '{"summary": "Act one \u2014 she arrives.", "start_state": {"place": "Hall", "present": ["Ana"], "state": "coat"}, "end_state": {"place": "Hall", "present": ["Ana"], "state": "wet"}}, '
                  '{"summary": "Act two, she leaves.", "start_state": {"place": "Hall", "present": ["Ana"], "state": "wet"}, {"end_state": {"place": "Hall", "present": [], "state": "gone"}}]}')
        self.assertIsNone(sw.coerce_acts(sw.parse_json_object(broken), 2))
        cast, places, title, acts = sw.salvage_acts(broken, 2)
        self.assertEqual((cast[0]["name"], places[0]["name"], title, len(acts)), ("Ana", "Hall", "T", 2))
        self.assertEqual(acts[0]["end"]["state"], "wet")
        self.assertIsNone(sw.salvage_acts(broken, 3))

    def test_broken_act_plan_costs_no_retry(self):
        def handler(kind, messages, n):
            if kind == "outline_acts":
                return ('{"title": "T", "cast": [{"name": "Ana", "look": "tall"}], "locations": [{"name": "Hall", "look": "dim"}], "acts": [' + ", ".join(
                    '{"summary": "Act %d.", "start_state": {"place": "Hall", "present": ["Ana"], "state": "s"}], "end_state": {"place": "Hall", "present": ["Ana"], "state": "e"}}' % k
                    for k in range(1, 5)) + "]}")
            return good_handler(40)(kind, messages, n)
        chat = Script(handler)
        eng = make_engine(chat, self.tmp.name, clips=40, two_level_min=31)
        self.assertEqual(len(eng.outline()["beats"]), 40)
        self.assertEqual(chat.kinds().count("outline_acts"), 1)
        self.assertEqual(eng.outline_mode, "4 acts")

    def test_unknown_names_unit(self):
        cast = [{"name": "Ana Ruiz", "look": ""}]
        beats = ["Ana opens the door and Rohan nods. Later, Ana sees Rohan again.", "She tells Maya nothing."]
        self.assertEqual(sw.unknown_names(beats, cast, [], "English"), ["Rohan"])
        self.assertEqual(sw.unknown_names(beats, cast, [], "Hindi"), [])

    def test_hard_failures_retry_only_the_chunk(self):
        state = {"n": 0}

        def handler(kind, messages, n):
            if kind == "outline":
                return outline_reply(10)
            first, last = span(messages)
            if first == 6 and state["n"] == 0:
                state["n"] += 1
                return writer_reply(first, first)
            return writer_reply(first, last)
        chat = Script(handler)
        eng = make_engine(chat, self.tmp.name, clips=10, chunk_beats=5, concurrency=2)
        eng.run()
        self.assertEqual(sorted(span(c["messages"]) for c in chat.calls[1:]), [(1, 5), (6, 10), (7, 10)])
        self.assertEqual(eng.retries["write"], 1)

    def test_every_call_carries_a_budget_and_room(self):
        chat = Script(good_handler(10))
        make_engine(chat, self.tmp.name, clips=10, budget=2048).run()
        for call in chat.calls:
            self.assertEqual(call["budget"], 2048)
            self.assertGreater(call["max_tokens"], call["budget"])
            self.assertLessEqual(call["max_tokens"], 65536)

    def test_truncated_reply_keeps_complete_beats_and_retries_the_rest(self):
        state = {"cut": False}

        def handler(kind, messages, n):
            if kind == "outline":
                return outline_reply(8)
            first, last = span(messages)
            if not state["cut"]:
                state["cut"] = True
                return sw.Reply(writer_reply(first, last, drop_last=4, end=False), "length")
            return writer_reply(first, last)
        chat = Script(handler)
        eng = make_engine(chat, self.tmp.name, clips=8, chunk_beats=8)
        result = eng.run()
        self.assertEqual(len(result["beats"]), 8)
        self.assertEqual([span(c["messages"]) for c in chat.calls[1:]], [(1, 8), (4, 8)])
        self.assertGreater(chat.calls[2]["max_tokens"], chat.calls[1]["max_tokens"] - 1)
        self.assertIn("at most", chat.calls[2]["messages"][0]["content"])

    def test_wrong_count_halves_the_chunk(self):
        def handler(kind, messages, n):
            if kind == "outline":
                return outline_reply(8)
            first, last = span(messages)
            if last - first + 1 > 4:
                return writer_reply(first, first)
            return writer_reply(first, last)
        chat = Script(handler)
        result = make_engine(chat, self.tmp.name, clips=8, chunk_beats=8).run()
        self.assertEqual(len(result["beats"]), 8)
        self.assertIn("write", chat.kinds())

    def test_a_beat_that_never_comes_raises_and_resumes(self):
        def failing(kind, messages, n):
            if kind == "outline":
                return outline_reply(4)
            first, last = span(messages)
            if first >= 3:
                return "I could not do that"
            return writer_reply(first, min(last, 2))
        chat = Script(failing)
        eng = make_engine(chat, self.tmp.name, clips=4, chunk_beats=4)
        with self.assertRaises(sw.StoryWriterError) as ctx:
            eng.run()
        self.assertIn("run again to resume", str(ctx.exception))
        kept = sorted(p.name for p in (Path(self.tmp.name) / "work" / "beats").glob("*.txt"))
        self.assertEqual(kept, ["001.txt", "002.txt"])
        resumed = Script(lambda k, m, n: self.fail("outline must be reused") if k == "outline" else writer_reply(*span(m)))
        result = make_engine(resumed, self.tmp.name, clips=4, chunk_beats=4).run()
        self.assertEqual(len(result["beats"]), 4)
        self.assertEqual([span(c["messages"]) for c in resumed.calls], [(3, 4)])

    def test_minor_sentinel_in_a_chunk_raises(self):
        def handler(kind, messages, n):
            return outline_reply(3) if kind == "outline" else sw.SENTINEL_MINOR
        with self.assertRaises(sw.MinorContentError):
            make_engine(Script(handler), self.tmp.name, clips=3).run()

    def test_speech_over_the_limit_is_reported(self):
        def handler(kind, messages, n):
            if kind == "outline":
                return outline_reply(2)
            first, last = span(messages)
            talk = "=== 1 ===\nAna says, \"" + " ".join(["word"] * 60) + "\" and leaves the hall slowly.\n=== 2 ===\n" + beat_text(2) + "\n=== END ==="
            return talk
        eng = make_engine(Script(handler), self.tmp.name, clips=2)
        eng.run()
        self.assertTrue(any("beat 1" in w and "spoken words" in w for w in eng.warnings))

    def test_hindi_prompt_asks_for_native_script(self):
        chat = Script(good_handler(2))
        sw.StoryEngine(chat, Path(self.tmp.name) / "hi", brief="b", clips=2, language="Hindi").run()
        self.assertIn("own native script", chat.calls[0]["messages"][0]["content"])
        self.assertIn("Hindi", chat.calls[1]["messages"][0]["content"])


class WriteStoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_files_and_sidecar(self):
        result = sw.write_story(Script(good_handler(6)), brief="A woman leaves a house.", clips=6, out_dir=self.tmp.name, model="m", style="ninfer-messages", seed=7)
        story = Path(result["story_path"])
        self.assertTrue(story.name.endswith("_6clips.md"))
        self.assertEqual(story.read_text(encoding="utf-8").strip(), result["story"])
        side = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
        for key in ("brief_sha1", "clips_requested", "beats_written", "model", "tokens", "stop_reasons", "e4_estimated_clips", "warnings", "call_log", "wall_seconds"):
            self.assertIn(key, side)
        self.assertEqual((side["clips_requested"], side["beats_written"], side["model"], side["seed"]), (6, 6, "m", 7))
        self.assertEqual(side["tokens"]["thinking"], 50 * side["calls"])
        self.assertEqual(side["stop_reasons"], {"stop": side["calls"]})

    def test_inputs_are_checked(self):
        for bad in (dict(brief="  ", clips=3), dict(brief="x", clips=0), dict(brief="x", clips=61)):
            with self.assertRaises(sw.StoryWriterError):
                sw.write_story(Script(good_handler(3)), out_dir=self.tmp.name, **bad)

    def test_sixty_clips(self):
        chat = Script(good_handler(60))
        result = sw.write_story(chat, brief="b", clips=60, out_dir=self.tmp.name)
        self.assertEqual(len(result["story"].split("\n\n")), 60)
        self.assertEqual(chat.kinds()[:1], ["outline_acts"])
        self.assertEqual(chat.kinds().count("outline_act"), 6)
        self.assertEqual(chat.kinds().count("write"), 8)

    def test_a_new_seed_is_a_new_run(self):
        a = sw.job_hash("b", 6, 15, "English", 1, "m", 4096)
        self.assertNotEqual(a, sw.job_hash("b", 6, 15, "English", 2, "m", 4096))
        self.assertNotEqual(a, sw.job_hash("b", 7, 15, "English", 1, "m", 4096))
        self.assertEqual(a, sw.job_hash(" b ", 6, 15, "English", 1, "m", 4096))


class Stub(BaseHTTPRequestHandler):
    seen = []
    mode = "ok"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append((self.path, body))
        if Stub.mode == "slow":   # a long generation: the answer comes after the client has been cancelled
            time.sleep(3)
        if Stub.mode == "400":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "context"}')
            return
        text = Stub.reply(body)
        if self.path == "/v1/messages":
            payload = {"content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": text}], "stop_reason": "end_turn",
                       "usage": {"input_tokens": 11, "output_tokens": 22, "output_tokens_details": {"thinking_tokens": 5}}}
        else:
            payload = {"choices": [{"message": {"content": "<think>x</think>" + text}, "finish_reason": "length"}],
                       "usage": {"prompt_tokens": 11, "completion_tokens": 22, "completion_tokens_details": {"reasoning_tokens": 5}}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    @staticmethod
    def reply(body):
        content = body["messages"][0]["content"]
        if "shooting outline" in content:
            return outline_reply(3)
        return writer_reply(*span(body["messages"]))


@contextlib.contextmanager
def stub_server(mode="ok"):
    Stub.seen, Stub.mode = [], mode
    server = HTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class WireTests(unittest.TestCase):
    def test_ninfer_messages_request_and_reply(self):
        with stub_server() as url:
            chat = sw.ServerChat({"url": url, "model": "m", "style": "ninfer-messages"})
            reply = chat([{"role": "system", "content": "S"}, {"role": "user", "content": "shooting outline"}], max_tokens=9000, budget=4096, kind="outline")
        path, body = Stub.seen[0]
        self.assertEqual(path, "/v1/messages")
        self.assertEqual(body["thinking"], {"type": "enabled", "budget_tokens": 4096})
        self.assertEqual((body["max_tokens"], body["system"], body["model"]), (9000, "S", "m"))
        self.assertEqual([m["role"] for m in body["messages"]], ["user"])
        self.assertEqual((reply.finish, reply.prompt_tokens, reply.completion_tokens, reply.thinking_tokens), ("stop", 11, 22, 5))
        self.assertIn("beats", reply.text)

    def test_llama_chat_request_and_reply(self):
        with stub_server() as url:
            chat = sw.ServerChat({"url": url, "model": "m", "style": "llama-chat"})
            reply = chat([{"role": "user", "content": "shooting outline"}], max_tokens=9000, budget=2048, kind="outline")
        path, body = Stub.seen[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["reasoning_budget_tokens"], 2048)
        self.assertEqual(body["reasoning_budget_message"], sw.BUDGET_MESSAGE)
        self.assertNotIn("thinking", body)
        self.assertEqual(reply.finish, "length")
        self.assertFalse(reply.text.startswith("<think>"))

    def test_unreachable_server_raises_after_retries(self):
        slept = []
        chat = sw.ServerChat({"url": "http://127.0.0.1:9", "model": "m", "style": "ninfer-messages"}, sleep=slept.append, timeout=2)
        with self.assertRaises(sw.LLMUnavailable) as ctx:
            chat([{"role": "user", "content": "x"}], max_tokens=5000, budget=1024, kind="outline")
        self.assertIn("nothing was written", str(ctx.exception))
        self.assertEqual(slept, [4, 8])

    def test_refused_request_is_not_retried(self):
        with stub_server("400") as url:
            chat = sw.ServerChat({"url": url, "model": "m", "style": "ninfer-messages"}, sleep=lambda s: self.fail("no retry"))
            with self.assertRaises(sw.StoryWriterError) as ctx:
                chat([{"role": "user", "content": "x"}], max_tokens=5000, budget=1024, kind="write")
        self.assertNotIsInstance(ctx.exception, sw.LLMUnavailable)
        self.assertEqual(len(Stub.seen), 1)

    def test_interrupt_is_honoured(self):
        with stub_server() as url:
            chat = sw.ServerChat({"url": url, "model": "m", "style": "ninfer-messages"}, interrupted=lambda: True)
            with self.assertRaises(KeyboardInterrupt):
                chat([{"role": "user", "content": "x"}], max_tokens=5000, budget=1024)
        self.assertEqual(Stub.seen, [])

    def test_a_cancel_stops_a_request_in_flight(self):
        """ComfyUI's cancel lands within about half a second, not when the model's answer (a minute or more) arrives."""
        with stub_server("slow") as url:
            cancel_at = time.time() + 0.3
            chat = sw.ServerChat({"url": url, "model": "m", "style": "ninfer-messages"}, interrupted=lambda: time.time() > cancel_at)
            started = time.time()
            with self.assertRaises(KeyboardInterrupt):
                chat([{"role": "user", "content": "x"}], max_tokens=5000, budget=1024)
            waited = time.time() - started
        self.assertLess(waited, 1.5)
        self.assertEqual(len(Stub.seen), 1)

    def test_engine_over_http(self):
        with stub_server() as url, tempfile.TemporaryDirectory() as tmp:
            chat = sw.ServerChat({"url": url, "model": "m", "style": "ninfer-messages"})
            result = sw.write_story(chat, brief="b", clips=3, out_dir=tmp)
        self.assertEqual(len(result["story"].split("\n\n")), 3)


class NodeTests(unittest.TestCase):
    def test_registered_beside_the_extender_node(self):
        self.assertIn("H3StoryWriter", sw.NODE_CLASS_MAPPINGS)
        self.assertIn("H3StoryWriter", sw.NODE_DISPLAY_NAME_MAPPINGS)
        spec = sw.H3StoryWriter.INPUT_TYPES()["required"]
        self.assertEqual(list(spec), ["brief", "clips", "seconds", "language", "seed", "writer_model", "thinking_budget"])
        conc = sw.H3StoryWriter.INPUT_TYPES()["optional"]["concurrency"][1]
        self.assertEqual((conc["default"], conc["min"], conc["max"]), (6, 1, 8))
        self.assertTrue(conc["tooltip"])
        self.assertEqual((spec["clips"][1]["default"], spec["clips"][1]["min"], spec["clips"][1]["max"]), (8, 1, 60))
        self.assertEqual(sw.H3StoryWriter.RETURN_TYPES, ("STRING",))
        self.assertEqual(sw.H3StoryWriter.RETURN_NAMES, ("story",))
        init = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("STORY_WRITER_NODE_CLASS_MAPPINGS", init)

    def test_write_runs_the_engine_and_returns_the_story(self):
        with stub_server() as url, tempfile.TemporaryDirectory() as tmp:
            @contextlib.contextmanager
            def fake_open(label, budget, slot_ctx, concurrency, say):
                yield {"url": url, "model": "m", "style": "ninfer-messages", "slots": 4}, mock.Mock(interrupted=lambda: False)

            with mock.patch.object(sw, "open_model", fake_open), mock.patch.object(sw, "output_root", lambda base=None: Path(tmp) / "out"):
                (story,) = sw.H3StoryWriter().write("a brief", 3, 15, "English", 1, "w", 4096)
            self.assertEqual(len(story.split("\n\n")), 3)
            self.assertEqual(len(list((Path(tmp) / "out").glob("*_3clips.md"))), 1)
            self.assertEqual(len(list((Path(tmp) / "out").glob("*_3clips.json"))), 1)

    def test_server_is_closed_and_vram_released(self):
        closed = []

        @contextlib.contextmanager
        def session(*args, **kwargs):
            self.assertTrue(kwargs["force"])
            self.assertEqual(kwargs["slots"], 1)
            self.assertEqual(kwargs["reasoning"], {"enabled": True, "budget": 4096, "message": sw.BUDGET_MESSAGE})
            try:
                yield mock.Mock(base="http://127.0.0.1:1", model_id="mid")
            finally:
                closed.append(True)

        pack = (mock.Mock(), mock.Mock(), mock.Mock(session=session), mock.Mock(is_ninfer=lambda p: True), mock.Mock())
        with mock.patch.object(sw, "_pack", lambda: pack), mock.patch.object(sw.prompt_rewriter, "_vision_files", lambda n, p, l: ("w.ninfer", "")), \
                mock.patch.object(sw.vram_release, "release") as release:
            with self.assertRaises(RuntimeError):
                with sw.open_model("writer", 4096, slot_ctx=65536) as (endpoint, runner):
                    self.assertEqual((endpoint["style"], endpoint["model"]), ("ninfer-messages", "mid"))
                    raise RuntimeError("boom")
        self.assertEqual(closed, [True])
        release.assert_called_once()

    def _open_with(self, concurrency, *, strata=False):
        seen = {}

        @contextlib.contextmanager
        def session(*args, **kwargs):
            seen.update(kwargs)
            yield mock.Mock(base="http://127.0.0.1:1", model_id="mid")

        pack = (mock.Mock(), mock.Mock(), mock.Mock(session=session), mock.Mock(is_ninfer=lambda p: True), mock.Mock())
        with mock.patch.object(sw, "_pack", lambda: pack), mock.patch.object(sw.prompt_rewriter, "_vision_files", lambda n, p, l: ("w.ninfer", "")), mock.patch.object(sw.vram_release, "release"):
            with sw.open_model("writer", 4096, slot_ctx=65536, concurrency=concurrency) as (endpoint, runner):
                seen["endpoint"] = endpoint
        return seen

    def test_the_server_gets_the_writers_concurrency_as_slots_and_a_pool_per_slot(self):
        """Before: slots=1, pool_ctx=65536 whatever the concurrency, so the parallel chunks queued behind one slot. Now ninfer-serve starts with --max-concurrency = slots
        and --max-context = pool / slots = 65536."""
        for asked, slots in ((1, 1), (2, 2), (4, 4), (6, 4), (8, 4)):
            seen = self._open_with(asked)
            self.assertEqual(slots, seen["slots"], asked)
            self.assertEqual(65536 * slots, seen["pool_ctx"], asked)
            self.assertEqual(slots, seen["endpoint"]["slots"], asked)
        self.assertEqual(4, sw.server_slots(99))
        self.assertEqual(1, sw.server_slots(0))

    def test_the_writer_never_sends_more_requests_than_the_server_has_slots(self):
        seen = {}

        @contextlib.contextmanager
        def fake_open(label, budget, slot_ctx, concurrency, say):
            seen["opened_for"] = concurrency
            yield {"url": "http://127.0.0.1:1", "model": "m", "style": "ninfer-messages", "slots": 4}, mock.Mock(interrupted=lambda: False)

        def fake_write_story(chat, **kwargs):
            seen["concurrency"] = kwargs["concurrency"]
            return {"story": "s", "story_path": "p", "json_path": "j", "sidecar": {"beats_written": 1, "story_words": 1, "e4_estimated_clips": 1, "wall_seconds": 1.0, "warnings": []}}

        with mock.patch.object(sw, "open_model", fake_open), mock.patch.object(sw, "write_story", fake_write_story):
            sw.H3StoryWriter().write("a brief", 3, 15, "English", 1, "w", 4096, concurrency=8)
        self.assertEqual((8, 4), (seen["opened_for"], seen["concurrency"]))

    def test_writer_without_a_server_is_refused(self):
        pack = (mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock())
        with mock.patch.object(sw, "_pack", lambda: pack), mock.patch.object(sw.prompt_rewriter, "_vision_files", lambda n, p, l: ("", "")):
            with self.assertRaises(sw.StoryWriterError) as ctx:
                with sw.open_model("plain gguf", 4096, slot_ctx=65536):
                    pass
        self.assertIn("runs on a server", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
