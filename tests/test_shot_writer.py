"""The H3 Shot Writer node and the Master Extender's clips_in / prompts_only: plan and shots, code checks and their one retry, the clips output, the lazy input.

No model, no ComfyUI, no torch: a scripted ``chat`` stands in for the model; the Master node's methods are cut out of its source and run with stubs.
"""

import ast
import contextlib
import json
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(ROOT / "tests"))
import shot_writer as sh  # noqa: E402
import story_writer as sw  # noqa: E402

HINDI_BRIEF = 'Clip 1 - Kael faces the monster. Kael says "आज नहीं।"\nClip 2 - She falls back. "I will not run."'


def plan_json(n, *, lines=None, minor=False, changes=None, pictures=None, visuals=None):
    """A plan: Kael opens in a grey cloak; in clip 2 she puts on a ring and her armour forms (a change the code must carry to every later clip)."""
    changes = changes if changes is not None else {2: [{"character": "Kael", "field": "wardrobe", "to": "full silver plate armour"},
                                                       {"character": "Kael", "field": "props", "to": "a tiny silver ring on her right hand; sword lying by the well"}]}
    visuals = visuals if visuals is not None else {2: ["Kael's silver plate armour", "the tiny silver ring"]}
    clips = [{"clip": i, "seconds": 15, "goal": f"Kael does thing {i} near the boulder.", "founder_lines": (lines or {}).get(i, []), "place": "Village", "present": ["Kael"],
              "end_action": f"Kael stands at spot {i}.", "changes": changes.get(i, []), "place_changes": [], "new_recurring_visuals": visuals.get(i, [])}
             for i in range(1, n + 1)]
    return json.dumps({"title": "The Monster", "summary": "Kael fights a monster.", "minor_in_sexual_content": minor, "start_place": "Village", "pictures": pictures or [],
                       "cast": [{"name": "Kael", "look": "a tall woman, 30", "wardrobe": "a grey cloak over a blue dress", "props": "none"}],
                       "locations": [{"name": "Village", "look": "a burnt village"}], "clips": clips}, ensure_ascii=False)


def shot(start, action="Kael, in her grey cloak and silver plate armour, steps forward", dialogue=None, framing="wide shot"):
    return {"start": start, "framing": framing, "action": action, "dialogue": dialogue}


def shots_json(number, shots, seconds=15):
    return json.dumps({"clip": number, "seconds": seconds, "shots": shots}, ensure_ascii=False)


GOOD = [shot(0, "Kael, in her grey cloak and silver plate armour, steps forward"), shot(5, "The monster turns", framing="close up, slow motion"), shot(10, "Kael raises her sword")]


class Script:
    """A scripted chat: ``handler(kind, messages, call_no) -> str``; every call is recorded."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, messages, *, max_tokens, budget, kind="", temperature=0.6, response_format=None):
        with self.lock:
            self.calls.append({"kind": kind, "messages": messages, "max_tokens": max_tokens, "budget": budget, "response_format": response_format})
            number = len(self.calls)
        return sw.Reply(self.handler(kind, messages, number), "stop", 100, 200, 50, 0.1)

    def kinds(self):
        return [c["kind"] for c in self.calls]


def engine(chat, tmp, **kw):
    values = dict(brief="Clip 1 - a fight.", clips=2, seconds=15, language="English", concurrency=1)
    values.update(kw)
    return sh.ShotEngine(chat, Path(tmp) / "work", **values)


def user(call):
    return sh.text_of(call["messages"][0]["content"])


class HappyPathTests(unittest.TestCase):
    def test_plan_then_one_shot_call_per_clip_with_schemas_and_budget(self):
        def handler(kind, messages, n):
            return plan_json(3) if kind == "plan" else shots_json(int(kind.split("_")[1]), GOOD)

        chat = Script(handler)
        with tempfile.TemporaryDirectory() as tmp:
            out = engine(chat, tmp, clips=3).run()
        self.assertEqual(sorted(chat.kinds()), ["plan", "shots_001", "shots_002", "shots_003"])
        self.assertEqual(len(out["clips"]), 3)
        for call in chat.calls:
            self.assertEqual(call["budget"], sw.THINKING_BUDGET)
            self.assertGreaterEqual(call["max_tokens"], call["budget"] + 512)
            rf = call["response_format"]
            self.assertEqual((rf["type"], rf["json_schema"]["strict"]), ("json_schema", True))
        plan_call = chat.calls[0]
        self.assertEqual(plan_call["response_format"]["json_schema"]["schema"]["properties"]["clips"]["minItems"], 1)
        shots_schema = chat.calls[1]["response_format"]["json_schema"]["schema"]["properties"]["shots"]
        self.assertEqual(shots_schema["minItems"], 1)
        self.assertIn('"shots": [', user(chat.calls[1]))   # the JSON is described in the prompt as well as enforced

    def test_shot_call_sees_cast_neighbours_and_clip_goal(self):
        chat = Script(lambda kind, m, n: plan_json(3) if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp:
            engine(chat, tmp, clips=3).run()
        second = next(c for c in chat.calls if c["kind"] == "shots_002")
        text = user(second)
        self.assertIn("a tall woman, 30", text)
        self.assertIn("Kael does thing 2 near the boulder.", text)
        self.assertIn("Kael does thing 1 near the boulder. It ended: Kael stands at spot 1.", text)
        self.assertIn("Kael does thing 3 near the boulder.", text)
        self.assertNotIn("{{", text)

    def test_clips_output_shape_and_raw_format(self):
        dialogue = {"speaker": "Kael", "line": "आज नहीं।"}
        shots = [shot(0, "Kael stands at the boulder.", framing="medium wide"), shot(3, "The monster turns toward her", dialogue=dialogue, framing="close up on the monster's face"),
                 shot(8, "She lunges", framing="wide side shot, slow motion")]
        chat = Script(lambda kind, m, n: plan_json(1) if kind == "plan" else shots_json(1, shots))
        with tempfile.TemporaryDirectory() as tmp:
            result = sh.write_shots(chat, brief="Clip 1 - fight", clips=1, language="Hindi", out_dir=tmp, concurrency=1)
            self.assertTrue(Path(result["clips_path"]).is_file())
            self.assertTrue(Path(result["story_path"]).is_file())
            self.assertEqual(json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))["clips_written"], 1)
        (clip,) = json.loads(result["clips"])
        self.assertEqual(set(clip), {"clip", "seconds", "goal", "raw", "shots", "start_state", "end_state", "new_recurring_visuals"})
        self.assertEqual((clip["clip"], clip["seconds"]), (1, 15))
        self.assertEqual(clip["raw"].split("\n"), [
            "Shot 1 – 0s – medium wide: Kael stands at the boulder.",
            "Shot 2 – 3s – close up on the monster's face: The monster turns toward her. Kael says \"आज नहीं।\"",
            "Shot 3 – 8s – wide side shot, slow motion: She lunges."])
        self.assertTrue(result["story"].startswith("The Monster"))
        self.assertIn("- Kael: a tall woman", result["story"])
        self.assertIn("- Village: a burnt village", result["story"])

    def test_clips_zero_lets_the_writer_decide(self):
        chat = Script(lambda kind, m, n: plan_json(4) if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp:
            out = engine(chat, tmp, clips=0).run()
        self.assertEqual(len(out["clips"]), 4)
        self.assertIn("You decide the number of clips", user(chat.calls[0]))
        self.assertEqual(chat.kinds().count("plan"), 1)

    def test_a_long_film_is_planned_in_parts_and_cast_is_merged(self):
        def handler(kind, messages, n):
            if kind != "plan":
                return shots_json(1, GOOD)
            first = int(re.search(r"numbered (\d+) to (\d+)", user({"messages": messages})).group(1))
            last = int(re.search(r"numbered (\d+) to (\d+)", user({"messages": messages})).group(2))
            data = json.loads(plan_json(last - first + 1))
            for i, c in enumerate(data["clips"]):
                c["clip"] = first + i
            if first > 1:
                data["cast"].append({"name": "Ogre", "look": "a huge ogre", "wardrobe": "a torn leather apron", "props": "none"})
            return json.dumps(data)

        chat = Script(handler)
        with tempfile.TemporaryDirectory() as tmp:
            out = engine(chat, tmp, clips=5, plan_part=2).run()
        self.assertEqual([c["clip"] for c in out["clips"]], [1, 2, 3, 4, 5])
        self.assertEqual(chat.kinds().count("plan"), 3)
        self.assertEqual([c["name"] for c in out["plan"]["cast"]], ["Kael", "Ogre"])
        self.assertIn("clips 1 to 2 are already planned", user(chat.calls[1]))

    def test_resume_from_disk_skips_finished_work(self):
        chat = Script(lambda kind, m, n: plan_json(2) if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp:
            engine(chat, tmp).run()
            calls = len(chat.calls)
            engine(chat, tmp).run()
        self.assertEqual(len(chat.calls), calls)

    def test_minor_in_sexual_content_raises(self):
        chat = Script(lambda kind, m, n: plan_json(1, minor=True))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(sw.MinorContentError):
            engine(chat, tmp, clips=1).run()
        self.assertEqual(chat.kinds(), ["plan"])


class StateTests(unittest.TestCase):
    """The ledger: a change stays until another clip changes it, folded in code; the shot call and the imported clip carry it."""

    def run_film(self, n=4, **plan_kw):
        chat = Script(lambda kind, m, k: plan_json(n, **plan_kw) if kind == "plan" else shots_json(1, GOOD))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = engine(chat, tmp.name, clips=n).run()
        return chat, out

    def kael(self, state):
        return next(c for c in state["characters"] if c["name"] == "Kael")

    def test_armour_ring_and_sword_carry_to_every_later_clip(self):
        chat, out = self.run_film()
        clips = out["clips"]
        self.assertIn("grey cloak", self.kael(clips[0]["start_state"])["wardrobe"])
        self.assertIn("grey cloak", self.kael(clips[0]["end_state"])["wardrobe"])
        for clip in clips[1:]:   # the change is in clip 2's end state, so clip 3 and on START in it
            self.assertEqual(self.kael(clip["end_state"])["wardrobe"], "full silver plate armour")
        for clip in clips[2:]:
            kael = self.kael(clip["start_state"])
            self.assertEqual(kael["wardrobe"], "full silver plate armour")
            self.assertIn("tiny silver ring", kael["props"])
            self.assertIn("sword lying by the well", kael["props"])
        for a, b in zip(clips, clips[1:]):   # a clip starts where the one before ended
            self.assertEqual(b["start_state"]["characters"], a["end_state"]["characters"])
            self.assertEqual(b["start_state"]["end_action"], a["end_state"]["end_action"])

    def test_the_shot_call_states_the_start_state_explicitly(self):
        chat, out = self.run_film()
        third = user(next(c for c in chat.calls if c["kind"] == "shots_003"))
        start = third.split("STATE AT THE START OF THIS CLIP", 1)[1].split("STATE AT THE END", 1)[0]
        self.assertIn("Kael wears: full silver plate armour", start)
        self.assertIn("sword lying by the well", start)
        self.assertNotIn("grey cloak", start)
        second = user(next(c for c in chat.calls if c["kind"] == "shots_002"))
        self.assertIn("wears: a grey cloak over a blue dress", second.split("STATE AT THE START OF THIS CLIP", 1)[1].split("STATE AT THE END", 1)[0])
        self.assertIn("wears: full silver plate armour", second.split("STATE AT THE END OF THIS CLIP", 1)[1])
        self.assertIn("never in the outfit of her reference picture", third)

    def test_first_appearance_must_say_what_the_character_wears(self):
        bare = shots_json(3, [shot(0, "Kael steps forward"), shot(5), shot(10)])
        replies = iter([bare, shots_json(3, GOOD)])
        chat = Script(lambda kind, m, k: plan_json(3) if kind == "plan" else (next(replies) if kind == "shots_003" else shots_json(1, GOOD)))
        with tempfile.TemporaryDirectory() as tmp:
            engine(chat, tmp, clips=3).run()
        retry = [c for c in chat.calls if c["kind"] == "shots_003"][1]
        self.assertIn("Kael first appears in the shot at 0s without saying what they wear", user(retry))
        self.assertIn("full silver plate armour", user(retry).split("PREVIOUS ANSWER", 1)[1])

    def test_skin_marks_in_the_plan_are_a_complaint(self):
        bad = {2: [{"character": "Kael", "field": "state", "to": "a bleeding gash on her arm"}]}
        replies = iter([plan_json(2, changes=bad), plan_json(2)])
        chat = Script(lambda kind, m, k: next(replies) if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp:
            engine(chat, tmp, clips=2).run()
        plans = [c for c in chat.calls if c["kind"] == "plan"]
        self.assertEqual(len(plans), 2)
        self.assertIn("describes a mark on the skin", user(plans[1]))

    def test_unknown_character_is_a_complaint_and_ignored_by_the_fold(self):
        bad = {1: [{"character": "Ghost", "field": "wardrobe", "to": "a sheet"}]}
        problems = sh.check_plan(sh.coerce_plan(json.loads(plan_json(1, changes=bad))))
        self.assertEqual(len(problems), 1)
        self.assertIn("\"Ghost\", who is not in the cast list", problems[0])
        plan = sh.coerce_plan(json.loads(plan_json(2, changes=bad)))
        sh.fold_states(plan)
        self.assertIn("grey cloak", self.kael(plan["clips"][1]["end_state"])["wardrobe"])

    def test_new_recurring_visuals_are_output_deduplicated_and_logged(self):
        visuals = {2: ["Kael's silver plate armour", "the tiny silver ring"], 3: ["the tiny silver ring", "the sword"]}
        with self.assertLogs("minimax_h3_master_extender.shot_writer", "WARNING") as logs:
            chat, out = self.run_film(visuals=visuals)
        self.assertEqual([c["new_recurring_visuals"] for c in out["clips"]], [[], ["Kael's silver plate armour", "the tiny silver ring"], ["the sword"], []])
        self.assertEqual(len([l for l in logs.output if "recurring visuals" in l]), 1)
        self.assertIn("clip 2: Kael's silver plate armour, the tiny silver ring; clip 3: the sword", logs.output[-1])

    def test_imported_clips_carry_the_state_blocks_the_rewriter_keys_on(self):
        chat, out = self.run_film()
        clips = sh.import_clips(json.dumps(out["clips"], ensure_ascii=False))
        first, third = clips[0], clips[2]
        self.assertNotIn(sh.story_planner.OPENS_MARK[:20], first["prompt"])
        self.assertIn(sh.story_planner.ENDS_MARK, first["prompt"])
        self.assertIn(sh.story_planner.OPENS_MARK.format(n=2), third["prompt"])
        opens = third["prompt"].split(sh.story_planner.ENDS_MARK)[0]
        self.assertIn("wardrobe: full silver plate armour", opens)
        self.assertEqual(third["prompt"], third["prompt_raw"])
        self.assertEqual(third["plan_end_state"]["characters"][0]["wardrobe"], "full silver plate armour")
        self.assertIn("Shot 1 – 0s", third["prompt"])

    @unittest.skipUnless(torch, "needs torch (the box has it)")
    def test_the_rewriter_task_message_after_the_armour_change_says_armour(self):
        import stub_pack

        chat, out = self.run_film()
        clips = sh.import_clips(json.dumps(out["clips"], ensure_ascii=False))
        settings = {"rewrite_mode": "pending clips", "rewrite_writer_model": "w", "rewrite_task": "Ref2VA", "rewrite_parallel": 1, "rewrite_previous_clips": "raw asks",
                    "rewrite_max_new_tokens": 100, "auto_clips": 0}
        rec, notes, done = stub_pack.run_rewrite(ROOT, settings, clips, refs=stub_pack.make_pictures((0,)), reply=lambda m: "FINAL")
        writer = [c["messages"] for c in rec.chats]
        self.assertEqual(len(writer), 4)
        task = {i: " ".join(sw_text(m["content"]) for m in call if m["role"] == "user") for i, call in enumerate(writer)}
        self.assertEqual(sum("Break the chapter" in t for t in task.values()), 0)   # the planner is skipped: the clips exist
        third = task[2]
        self.assertIn("THIS CLIP OPENS EXACTLY WHERE CLIP 2 ENDED:", third)
        self.assertIn("wardrobe: full silver plate armour", third)
        self.assertIn("a tiny silver ring on her right hand", third)
        self.assertNotIn("a grey cloak over a blue dress", third.split("original_prompt:", 1)[1])   # this clip's own ask never carries the old outfit
        self.assertIn("wardrobe: a grey cloak over a blue dress", task[0])


def sw_text(content):
    return content if isinstance(content, str) else " ".join(p.get("text", "") for p in content if p.get("type") == "text")


PICTURES = [{"label": "Picture 1", "image": "data:image/png;base64,AAAA", "tokens": 10}, {"label": "Picture 4", "image": "data:image/png;base64,BBBB", "tokens": 10}]


class PictureTests(unittest.TestCase):
    @unittest.skipUnless(torch, "needs torch (the box has it)")
    def test_numbering_matches_the_extender_with_a_gap(self):
        t = {i: torch.full((1, 40, 40, 3), i / 10.0) for i in (1, 2, 4)}
        inputs = {f"ref_image_{i}": t.get(i) for i in range(1, 10)}
        refs = sh.refs_from_inputs(inputs)
        self.assertEqual(sorted(refs), ["ref_image_0", "ref_image_1", "ref_image_3"])   # socket N is key ref_image_{N-1}, never compacted
        master = {}
        for idx in range(1, 10):   # the loop of master_node.extend, verbatim in effect
            if f"ref_image_{idx - 1}" not in master and f"ref_image_{idx}" in inputs and inputs[f"ref_image_{idx}"] is not None:
                master[f"ref_image_{idx - 1}"] = inputs[f"ref_image_{idx}"]
        self.assertEqual(refs.keys(), master.keys())
        pictures = sh.pictures_of(inputs)
        self.assertEqual([p["label"] for p in pictures], ["Picture 1", "Picture 2", "Picture 4"])
        self.assertEqual([p["slot"] for p in pictures], [0, 1, 3])
        self.assertTrue(all(p["image"].startswith("data:image/png;base64,") for p in pictures))
        text = (ROOT / "master_node.py").read_text(encoding="utf-8")
        self.assertIn('refs[f"ref_image_{idx - 1}"] = kwargs[ref_k]', text)

    def test_the_plan_and_every_shot_call_carry_the_pictures_in_planner_layout(self):
        plan = plan_json(2, pictures=[{"label": "Picture 4", "shows": "silver plate armour"}])
        chat = Script(lambda kind, m, k: plan if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp:
            out = engine(chat, tmp, brief="9 - armour as shown in picture 4", pictures=PICTURES).run()
        self.assertEqual(len(chat.calls), 3)
        for call in chat.calls:
            content = call["messages"][0]["content"]
            self.assertEqual([p["type"] for p in content], ["text", "image_url", "text", "image_url", "text"])
            self.assertEqual([content[0]["text"], content[2]["text"]], ["Picture 1:", "Picture 4:"])
            self.assertEqual([content[1]["image_url"]["url"], content[3]["image_url"]["url"]], [p["image"] for p in PICTURES])
            self.assertIn("<Picture 4>", content[-1]["text"])
        self.assertIn("silver plate armour", user(chat.calls[1]))   # the plan's reading of the picture reaches the shot call
        self.assertIn("Reference pictures:\n- Picture 4: silver plate armour", out["story"])
        self.assertGreater(chat.calls[0]["max_tokens"], 0)

    def test_a_complaint_is_appended_to_the_text_part(self):
        replies = iter([shots_json(1, [shot(2), shot(8)]), shots_json(1, GOOD)])
        chat = Script(lambda kind, m, k: plan_json(1) if kind == "plan" else next(replies))
        with tempfile.TemporaryDirectory() as tmp:
            engine(chat, tmp, clips=1, pictures=PICTURES).run()
        retry = [c for c in chat.calls if c["kind"] == "shots_001"][1]["messages"][0]["content"]
        self.assertEqual(len([p for p in retry if p["type"] == "image_url"]), 2)
        self.assertIn("it must start at 0", retry[-1]["text"])

    def test_without_vision_the_pictures_are_labelled_not_sent_and_it_is_logged(self):
        chat = Script(lambda kind, m, k: plan_json(1) if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp, self.assertLogs("minimax_h3_master_extender.shot_writer", "WARNING") as logs:
            engine(chat, tmp, clips=1, pictures=PICTURES, sees=False).run()
        for call in chat.calls:
            self.assertIsInstance(call["messages"][0]["content"], str)
            self.assertIn("You cannot see them", user(call))
            self.assertIn("Picture 1, Picture 4", user(call))
        self.assertTrue(any("has no vision" in l for l in logs.output))

    def test_no_pictures_leaves_the_message_a_plain_string(self):
        chat = Script(lambda kind, m, k: plan_json(1) if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp:
            engine(chat, tmp, clips=1).run()
        self.assertTrue(all(isinstance(c["messages"][0]["content"], str) for c in chat.calls))

    def test_prompt_rules(self):
        plan = " ".join(sh.load_prompt("plan").split())
        for phrase in ("stays as <Picture 4>", "the silver plate armour of <Picture 4>", "contradicts their picture", '"pictures"'):
            self.assertIn(phrase, plan)

    def test_job_hash_depends_on_the_pictures(self):
        a = sh.job_hash("b", 1, 15, "English", 1, "m", 4096, PICTURES)
        self.assertNotEqual(a, sh.job_hash("b", 1, 15, "English", 1, "m", 4096, PICTURES[:1]))
        self.assertEqual(a, sh.job_hash("b", 1, 15, "English", 1, "m", 4096, list(PICTURES)))

    def test_anthropic_wire_turns_data_uris_into_image_blocks(self):
        chat = sw.ServerChat({"url": "http://h", "model": "m", "style": "ninfer-messages"})
        content = sh.message("q", PICTURES)["content"]
        body = chat.request([{"role": "user", "content": content}], 6000, 4096, 0.5)[2]
        blocks = body["messages"][0]["content"]
        self.assertEqual(blocks[1], {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}})


class RangeAndCountTests(unittest.TestCase):
    def test_range_brief_gives_that_many_distinct_clips_and_prompt_says_so(self):
        brief = "Clip 1 - She sees the monster.\n3-6 - Battle.\nClip 2 - She hides."
        chat = Script(lambda kind, m, n: plan_json(6) if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp:
            out = engine(chat, tmp, brief=brief, clips=6).run()
        self.assertEqual(len(out["clips"]), 6)
        text = user(chat.calls[0])
        self.assertIn("becomes that many SEPARATE clips", text)
        self.assertIn("3-6 - Battle.", text)
        self.assertIn("exactly 6 objects, numbered 1 to 6", text)

    def test_wrong_count_is_asked_again_once_with_the_complaint(self):
        replies = iter([plan_json(3), plan_json(6)])
        chat = Script(lambda kind, m, n: next(replies) if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp:
            out = engine(chat, tmp, clips=6).run()
        self.assertEqual(len(out["clips"]), 6)
        plans = [c for c in chat.calls if c["kind"] == "plan"]
        self.assertEqual(len(plans), 2)
        self.assertIn("the plan has 3 clips; it must have exactly 6", user(plans[1]))
        self.assertNotIn("PROBLEMS", user(plans[0]))

    def test_count_still_wrong_after_the_retry_warns_and_cuts_the_excess(self):
        chat = Script(lambda kind, m, n: plan_json(9) if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp:
            eng = engine(chat, tmp, clips=6)
            out = eng.run()
        self.assertEqual(len(out["clips"]), 6)
        self.assertEqual(chat.kinds().count("plan"), 2)
        self.assertTrue(any("must have exactly 6" in w for w in eng.warnings))


class TimingTests(unittest.TestCase):
    def run_one(self, replies, **kw):
        it = iter(replies)
        chat = Script(lambda kind, m, n: plan_json(1) if kind == "plan" else next(it))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        eng = engine(chat, tmp.name, clips=1, **kw)
        return eng, chat, eng.run()

    def test_check_shots_names_each_problem(self):
        self.assertEqual(sh.check_shots(GOOD, 15, "English"), [])
        self.assertIn("must start at 0", sh.check_shots([shot(2), shot(8)], 15, "English")[0])
        self.assertTrue(any("do not strictly increase" in p for p in sh.check_shots([shot(0), shot(8), shot(8)], 15, "English")))
        self.assertTrue(any("at or after the end" in p for p in sh.check_shots([shot(0), shot(8), shot(15)], 15, "English")))
        self.assertTrue(any("shot 1 lasts 1s" in p for p in sh.check_shots([shot(0), shot(1), shot(9)], 15, "English")))
        self.assertTrue(any("shot 2 lasts 1s" in p for p in sh.check_shots([shot(0), shot(14)], 15, "English")))
        self.assertTrue(any("no shots" in p for p in sh.check_shots([], 15, "English")))
        self.assertTrue(any("7 shots" in p for p in sh.check_shots([shot(i * 2) for i in range(7)], 15, "English")))

    def test_timing_failure_retries_once_with_the_complaint(self):
        bad = shots_json(1, [shot(1), shot(2), shot(9)])
        eng, chat, out = self.run_one([bad, shots_json(1, GOOD)])
        calls = [c for c in chat.calls if c["kind"] == "shots_001"]
        self.assertEqual(len(calls), 2)
        self.assertIn("shot 1 starts at 1s; it must start at 0", user(calls[1]))
        self.assertIn("shot 1 lasts 1s", user(calls[1]))
        self.assertEqual(out["clips"][0]["shots"][0]["start"], 0)
        self.assertEqual(eng.warnings, [])
        self.assertEqual(eng.retries["shots"], 1)

    def test_still_bad_after_the_retry_keeps_best_effort_and_warns(self):
        bad = shots_json(1, [shot(2), shot(2), shot(20), shot(9)])
        eng, chat, out = self.run_one([bad, bad, bad])
        self.assertEqual(len([c for c in chat.calls if c["kind"] == "shots_001"]), 2)   # one retry, never more
        starts = [s["start"] for s in out["clips"][0]["shots"]]
        self.assertEqual(starts, [0, 9])
        self.assertTrue(any(w.startswith("clip 1:") for w in eng.warnings))

    def test_a_retry_that_is_worse_does_not_replace_a_better_first_answer(self):
        first = shots_json(1, [shot(0), shot(5), shot(14)])    # one problem: last shot 1 s
        second = shots_json(1, [shot(2), shot(2), shot(9)])    # more problems
        eng, chat, out = self.run_one([first, second])
        self.assertEqual([s["start"] for s in out["clips"][0]["shots"]], [0, 5, 14])

    def test_unreadable_answer_is_retried(self):
        eng, chat, out = self.run_one(["not json at all", shots_json(1, GOOD)])
        self.assertEqual(len(out["clips"][0]["shots"]), 3)

    def test_words_that_do_not_fit_the_shot_are_a_complaint(self):
        long_line = {"speaker": "Kael", "line": " ".join(["word"] * 12)}
        problems = sh.check_shots([shot(0, dialogue=long_line), shot(5), shot(10)], 15, "English")
        self.assertEqual(len(problems), 1)
        self.assertIn("12 words but a 5-second shot fits at most 10", problems[0])
        self.assertLess(sh.max_line_words(5, "Hindi"), sh.max_line_words(5, "English"))


class ScriptTests(unittest.TestCase):
    def test_english_line_in_a_hindi_clip_is_a_complaint_and_retried(self):
        bad = shots_json(1, [shot(0), shot(5, dialogue={"speaker": "Kael", "line": "Not today."}), shot(10)])
        good = shots_json(1, [shot(0), shot(5, dialogue={"speaker": "Kael", "line": "आज नहीं।"}), shot(10)])
        replies = iter([bad, good])
        chat = Script(lambda kind, m, n: plan_json(1) if kind == "plan" else next(replies))
        with tempfile.TemporaryDirectory() as tmp:
            eng = engine(chat, tmp, clips=1, language="Hindi")
            out = eng.run()
        retry = [c for c in chat.calls if c["kind"] == "shots_001"][1]
        self.assertIn("Not today.", user(retry))
        self.assertIn("not in Hindi's own script", user(retry))
        self.assertEqual(out["clips"][0]["shots"][1]["dialogue"]["line"], "आज नहीं।")
        self.assertEqual(eng.warnings, [])

    def test_script_ranges(self):
        self.assertEqual(sh.foreign_letters("आज नहीं, 12! ok", "Hindi"), ["o", "k"])
        self.assertEqual(sh.foreign_letters("ನಾನು ಬರುತ್ತೇನೆ", "Kannada"), [])
        self.assertEqual(sh.foreign_letters("आज", "Kannada"), ["आ", "ज"])
        self.assertEqual(sh.foreign_letters("Hello wörld", "English"), [])       # a Latin language is never checked
        self.assertEqual(sh.foreign_letters("hello", "Klingon"), ["h", "e", "l", "l", "o"])   # unlisted wide language: Latin letters only
        self.assertEqual(sh.foreign_letters("привет", "Russian"), [])
        self.assertEqual(sh.foreign_letters("こんにちは 漢字", "Japanese"), [])


class FounderLineTests(unittest.TestCase):
    def test_lines_already_in_the_film_language_are_found(self):
        self.assertEqual(sh.founder_lines(HINDI_BRIEF, "Hindi"), ["आज नहीं।"])   # the English one is translated, not kept
        self.assertEqual(sh.founder_lines(HINDI_BRIEF, "English"), ["आज नहीं।", "I will not run."])
        self.assertEqual(sh.founder_lines(HINDI_BRIEF, "French"), [])
        self.assertEqual(sh.founder_lines('"  आज   नहीं।  "', "Hindi"), ["आज नहीं।"])

    def test_a_missing_founder_line_is_a_complaint_naming_it(self):
        plan = plan_json(2, lines={1: ["आज नहीं।"]})
        no_line = shots_json(1, [shot(0), shot(5, dialogue={"speaker": "Kael", "line": "कभी नहीं।"}), shot(10)])
        with_line = shots_json(1, [shot(0), shot(5, dialogue={"speaker": "Kael", "line": "आज  नहीं।"}), shot(10)])   # whitespace collapses
        replies = iter([no_line, with_line])
        chat = Script(lambda kind, m, n: plan if kind == "plan" else (next(replies) if kind == "shots_001" else shots_json(2, GOOD)))
        with tempfile.TemporaryDirectory() as tmp:
            eng = engine(chat, tmp, brief=HINDI_BRIEF, language="Hindi")
            out = eng.run()
        calls = [c for c in chat.calls if c["kind"] == "shots_001"]
        self.assertIn("आज नहीं।", user(calls[0]))                           # the first ask already names the line the clip must keep
        self.assertIn("must appear word for word", user(calls[1]))
        self.assertEqual(out["clips"][0]["shots"][1]["dialogue"]["line"], "आज नहीं।")   # stored NFC-normalised, one space
        self.assertEqual(eng.warnings, [])

    def test_only_lines_the_brief_really_has_are_required(self):
        plan = plan_json(1, lines={1: ["a line the model invented"]})
        chat = Script(lambda kind, m, n: plan if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp:
            eng = engine(chat, tmp, clips=1, brief=HINDI_BRIEF, language="Hindi")
            eng.run()
        self.assertEqual(chat.kinds().count("shots_001"), 1)

    def test_a_founder_line_nowhere_in_the_film_is_a_warning(self):
        chat = Script(lambda kind, m, n: plan_json(2) if kind == "plan" else shots_json(1, GOOD))
        with tempfile.TemporaryDirectory() as tmp:
            eng = engine(chat, tmp, brief=HINDI_BRIEF, language="Hindi")
            eng.run()
        self.assertTrue(any("आज नहीं।" in w and "not in any shot" in w for w in eng.warnings))


class PromptTests(unittest.TestCase):
    def test_prompts_carry_the_brief_rules(self):
        plan = " ".join(sh.load_prompt("plan").split())
        for phrase in ("freely improve", "keeps its number", "SEPARATE clips", "translate it into natural spoken", "never force a line into every clip",
                       "Devanagari", "per-character language instruction", "and so on", "founder_lines"):
            self.assertIn(phrase, plan)
        shots = " ".join(sh.load_prompt("shots").split())
        for phrase in ("at least 2 seconds", "never forced into every shot", "2 words per second", "null", "starts exactly where the previous clip ended"):
            self.assertIn(phrase, shots)


class ChatWireTests(unittest.TestCase):
    def test_response_format_goes_into_chat_completions_only(self):
        rf = sh.response_format("x", {"type": "object"})
        chat = sw.ServerChat({"url": "http://h", "model": "m", "style": "chat"})
        url, _, body = chat.request([{"role": "user", "content": "q"}], 6000, 4096, 0.5, rf)
        self.assertEqual(body["response_format"], rf)
        self.assertEqual((body["reasoning_budget_tokens"], body["reasoning_budget_message"]), (4096, sw.BUDGET_MESSAGE))
        self.assertNotIn("response_format", chat.request([{"role": "user", "content": "q"}], 6000, 4096, 0.5)[2])
        ninfer = sw.ServerChat({"url": "http://h", "model": "m", "style": "ninfer-messages"})
        self.assertNotIn("response_format", ninfer.request([{"role": "user", "content": "q"}], 6000, 4096, 0.5, rf)[2])


class NodeTests(unittest.TestCase):
    def test_registered_with_the_inputs_and_outputs_asked_for(self):
        self.assertIn("H3ShotWriter", sh.NODE_CLASS_MAPPINGS)
        self.assertIn("H3ShotWriter", sh.NODE_DISPLAY_NAME_MAPPINGS)
        spec = sh.H3ShotWriter.INPUT_TYPES()
        self.assertEqual(list(spec["required"]), ["brief", "clips", "seconds", "language", "seed", "writer_model", "thinking_budget"])
        self.assertEqual(list(spec["optional"]), ["concurrency"] + [f"ref_image_{i}" for i in range(1, 10)])   # the Extender's names, appended after the rest
        self.assertTrue(all(spec["optional"][f"ref_image_{i}"][0] == "IMAGE" for i in range(1, 10)))
        clips, seconds = spec["required"]["clips"][1], spec["required"]["seconds"][1]
        self.assertEqual((clips["default"], clips["min"], clips["max"]), (0, 0, 60))
        self.assertEqual((seconds["default"], seconds["min"], seconds["max"]), (15, 4, 30))
        self.assertTrue(spec["required"]["brief"][1]["multiline"])
        self.assertEqual(sh.H3ShotWriter.RETURN_TYPES, ("STRING", "STRING"))
        self.assertEqual(sh.H3ShotWriter.RETURN_NAMES, ("clips", "story"))
        init = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("SHOT_WRITER_NODE_CLASS_MAPPINGS", init)

    def test_write_runs_the_engine_and_returns_clips_and_story(self):
        chat = Script(lambda kind, m, n: plan_json(2) if kind == "plan" else shots_json(1, GOOD))

        @contextlib.contextmanager
        def fake_open(label, budget, slot_ctx, concurrency, say):
            yield {"url": "http://x", "model": "m", "style": "chat", "slots": 4}, mock.Mock(interrupted=lambda: False)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sh, "open_model", fake_open), mock.patch.object(sh, "ServerChat", lambda *a, **k: chat), \
                mock.patch.object(sh, "output_root", lambda base=None: Path(tmp) / "out"):
            clips_json, story = sh.H3ShotWriter().write("Clip 1 - a fight", 2, 15, "English", 1, "w", "4096")
            self.assertEqual(len(list((Path(tmp) / "out").glob("*_2clips.clips.json"))), 1)
        self.assertEqual(len(json.loads(clips_json)), 2)
        self.assertIn("The Monster", story)


# --------------------------------------------------------------------------- the Master Extender side


def master_source():
    return (ROOT / "master_node.py").read_text(encoding="utf-8")


def master_method(name, **namespace):
    """One method of MiniMaxH3MasterExtender (or module function) compiled on its own; master_node itself needs ComfyUI and torch."""
    tree = ast.parse(master_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = dict(namespace)
            exec(compile(ast.Module(body=[node], type_ignores=[]), "master_node.py", "exec"), ns)
            return ns[name]
    raise AssertionError(name)


class LazyStatusTests(unittest.TestCase):
    def check(self, clips_json, *, clips_in=None, linked=True):
        fn = master_method("check_lazy_status", shot_writer=sh)
        prompt = {"7": {"inputs": {"clips_in": ["3", 0]} if linked else {}}}
        return fn(object(), clips_json=clips_json, clips_in=clips_in, unique_id="7", prompt=prompt)

    def test_empty_clip_list_asks_for_the_writer(self):
        self.assertEqual(self.check("[]"), ["clips_in"])
        self.assertEqual(self.check(json.dumps([{"id": 0, "prompt": "", "duration": 15}])), ["clips_in"])
        self.assertEqual(self.check(json.dumps([{"id": 0, "prompt": "  "}])), ["clips_in"])

    def test_a_clip_with_a_prompt_never_runs_the_writer(self):
        self.assertEqual(self.check(json.dumps([{"id": 0, "prompt": "Kael walks."}])), [])
        self.assertEqual(self.check(json.dumps([{"id": 0, "prompt": "", "prompt_raw": "Kael walks."}])), [])

    def test_already_evaluated_or_unwired_is_never_asked(self):
        self.assertEqual(self.check("[]", clips_in="[]"), [])
        self.assertEqual(self.check("[]", linked=False), [])

    def test_the_input_is_the_last_lazy_forced_socket(self):
        text = master_source()
        body = text[text.index("def INPUT_TYPES"):text.index("def VALIDATE_INPUTS")]
        names = re.findall(r'^\s{16}"(\w+)": \(', body, re.M)
        self.assertEqual(names[-1], "clips_in")
        self.assertRegex(body, r'"clips_in": \("STRING", \{"forceInput": True, "lazy": True')


class ImportTests(unittest.TestCase):
    WRITER = json.dumps([
        {"clip": 1, "seconds": 15, "goal": "g1", "raw": "Shot 1 – 0s – wide: Kael waits.\nShot 2 – 6s – close up: Kael says \"आज नहीं।\"", "shots": []},
        {"clip": 2, "seconds": 12, "goal": "g2", "raw": "Shot 1 – 0s – wide: The monster roars.", "shots": []}], ensure_ascii=False)

    def test_clip_entries(self):
        clips = sh.import_clips(self.WRITER)
        self.assertEqual([c["id"] for c in clips], [0, 1])
        c = clips[0]
        self.assertEqual((c["title"], c["duration"], c["seed_mode"], c["validated"], c["loras"], c["from_writer"], c["prompt_rewritten"]), ("Clip 1", 15, "fixed", False, [], True, False))
        self.assertEqual(c["prompt"], c["prompt_raw"])
        self.assertIn("आज नहीं।", c["prompt"])
        self.assertEqual(clips[1]["duration"], 12)
        self.assertEqual(len({x["seed"] for x in clips}), 2)
        self.assertEqual([x["seed"] for x in clips], [x["seed"] for x in sh.import_clips(self.WRITER)])   # re-imports reproduce
        self.assertNotIn("planned", c)

    def test_bad_input_gives_no_clips(self):
        self.assertEqual(sh.import_clips("nope"), [])
        self.assertEqual(sh.import_clips("{}"), [])
        self.assertEqual(sh.import_clips(json.dumps([{"raw": ""}])), [])

    def test_has_prompt(self):
        self.assertFalse(sh.has_prompt("[]"))
        self.assertFalse(sh.has_prompt("garbage"))
        self.assertFalse(sh.has_prompt(None))
        self.assertTrue(sh.has_prompt(json.dumps([{"prompt": "x"}])))

    def test_extend_imports_only_into_an_empty_list_and_skips_the_planner(self):
        text = master_source()
        start = text.index("clips = json.loads(clips_json)")
        block = text[start:text.index("# Parse resolution settings")]
        self.assertIn("not shot_writer.has_prompt(clips)", block)
        self.assertIn('kwargs["auto_clips"] = 0', block)
        self.assertIn("_send_plan(owner, clips, range(len(clips)))", block)
        self.assertIn("clips_metadata.embed_clips(unique_id, prompt, extra_pnginfo, clips)", block)
        self.assertLess(block.index("shot_writer.import_clips"), block.index("raise ValueError"))   # an empty list is only an error when nothing was imported
        self.assertLess(text.index("clips_in=None,"), text.index("**kwargs,\n    ):\n        accel_mode"))


class PromptsOnlyTests(unittest.TestCase):
    def test_returns_blocked_outputs_and_a_status_without_rendering(self):
        sent = []
        blocked = object()
        fn = master_method("_prompts_only", _send_clips=lambda *a, **k: sent.append((a, k)), _LOG=mock.Mock(), _blocked=lambda: blocked)
        clips = [{"id": 0, "prompt": "a"}, {"id": 1, "prompt": "b"}]
        cache, count, validated, status, frame = fn(object(), "7", clips, ["planned 2 clip(s)"], True)
        self.assertIs(cache, blocked)
        self.assertIs(frame, blocked)
        self.assertEqual((count, validated), (2, 0))
        self.assertTrue(status.startswith("prompts only: 2 clips written, nothing rendered"))
        self.assertIn("planned 2 clip(s)", status)
        self.assertEqual(sent[0][0][:2], ("7", clips))
        self.assertIn("rewrite_text", sent[0][1]["fields"])
        self.assertIn("rewrite_mode is off", fn(object(), "7", clips, [], False)[3])

    def test_blocker_is_comfys_execution_blocker_with_no_message(self):
        fn = master_method("_blocked")
        graph_utils = mock.Mock()
        graph_utils.ExecutionBlocker = lambda message: ("blocker", message)
        with mock.patch.dict(sys.modules, {"comfy_execution": mock.Mock(), "comfy_execution.graph_utils": graph_utils}):
            self.assertEqual(fn(), ("blocker", None))
        with mock.patch.dict(sys.modules, {"comfy_execution": None}):
            self.assertIsNone(fn())

    def test_the_run_stops_after_the_rewriter_and_before_any_h3_work(self):
        text = master_source()
        stop = text.index('if str(run_mode) == "prompts_only":')
        self.assertGreater(stop, text.index("prompt_rewriter.rewrite_clips("))
        self.assertGreater(stop, text.index("clips_metadata.embed_clips(unique_id, prompt, extra_pnginfo, clips)"))
        for later in ("MasterEngine(", "_manifest_for_first(", "load_final_vae(", "engine.render_clip("):
            self.assertLess(stop, text.index(later, text.index("def extend(")), later)
        self.assertIn('"clip_by_clip", "full_batch", "prompts_only"', text)


class JsPanelTests(unittest.TestCase):
    def test_panel_wiring(self):
        js = (ROOT / "web" / "master_extender.js").read_text(encoding="utf-8")
        self.assertIn("prompts only: write raw + H3 prompts, render nothing; edit them, then switch to full batch or clip by clip.", js)
        self.assertIn("Re-import from writer", js)
        self.assertIn('i.name === "clips_in" && i.link != null', js)
        self.assertIn("incoming.from_writer === true", js)


if __name__ == "__main__":
    unittest.main()
