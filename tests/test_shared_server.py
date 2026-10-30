"""One model server per job: the Shot Writer leaves it running, the Extender's rewriter attaches (no second launch); and the writer's output saved in the video.

No model, no ComfyUI, no torch: the rewriter pack's session is a fake that counts launches and closes.
"""

import ast
import contextlib
import copy
import hashlib
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import clips_metadata as cm  # noqa: E402
import prompt_rewriter as pr  # noqa: E402
import shared_server as ss  # noqa: E402
import shot_writer as sh  # noqa: E402
import story_writer as sw  # noqa: E402
import stub_pack  # noqa: E402
import test_shot_writer as tsw  # noqa: E402


def _master_projects():
    """The clip-identity functions of master_projects, cut out of its source (the module itself needs ComfyUI)."""
    tree = ast.parse((ROOT / "master_projects.py").read_text(encoding="utf-8"))
    wanted = {"_clip_identity", "_clip_fingerprint", "_chain_key"}
    ns = {"json": json, "hashlib": hashlib}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "master_projects.py", "exec"), ns)
    return types.SimpleNamespace(**{name: ns[name] for name in wanted})


mp = _master_projects()


class FakeServer:
    def __init__(self, slots=8):
        self.base, self.model_id, self.slots, self.n_ctx = "http://127.0.0.1:1", "mid", slots, 32768
        self.chats = []

    def chat(self, messages, **kw):
        self.chats.append(kw)
        return "FINAL"


class Launcher:
    """A ``launch`` for shared_server: counts starts and closes."""

    def __init__(self, meta=None, server=True):
        self.starts, self.closes, self.meta, self.server = 0, 0, meta or {}, server

    def __call__(self):
        self.starts += 1
        return (FakeServer() if self.server else None), self.close, dict(self.meta)

    def close(self):
        self.closes += 1


class Clean(unittest.TestCase):
    def setUp(self):
        ss.close()
        self.addCleanup(ss.close)


class SharedServerTests(Clean):
    KEY = ss.key_of("/m/model.ninfer", "")

    def test_a_parked_server_is_attached_not_launched_again(self):
        launch = Launcher()
        with ss.use(self.KEY, launch, keep=lambda: True) as first:
            pass
        self.assertEqual((launch.starts, launch.closes), (1, 0))
        with ss.use(self.KEY, launch) as second:
            self.assertIs(first, second)
        self.assertEqual((launch.starts, launch.closes), (1, 1))   # attached, and closed by its last user
        self.assertIsNone(ss.current())

    def test_another_model_or_a_server_that_does_not_fit_closes_the_parked_one_and_launches(self):
        launch = Launcher({"slot_ctx": 1000})
        with ss.use(self.KEY, launch, keep=lambda: True):
            pass
        other = Launcher()
        with ss.use(ss.key_of("/m/other.ninfer"), other):
            pass
        self.assertEqual((launch.closes, other.starts, other.closes), (1, 1, 1))
        with ss.use(self.KEY, launch, keep=lambda: True):
            pass
        refit = Launcher({"slot_ctx": 70000})
        with ss.use(self.KEY, refit, fits=lambda meta: meta["slot_ctx"] >= 65536):
            pass
        self.assertEqual((launch.closes, refit.starts), (2, 1))

    def test_an_error_or_a_cancel_closes_even_when_the_server_was_to_be_kept(self):
        launch = Launcher()
        with self.assertRaises(KeyboardInterrupt):
            with ss.use(self.KEY, launch, keep=lambda: True):
                raise KeyboardInterrupt()
        self.assertEqual(launch.closes, 1)
        self.assertIsNone(ss.current())

    def test_no_server_is_not_kept_and_leaves_nothing_behind(self):
        launch = Launcher(server=False)
        with ss.use(self.KEY, launch, keep=lambda: True) as server:
            self.assertIsNone(server)
        self.assertEqual(launch.closes, 1)
        self.assertIsNone(ss.current())

    def test_close_parked_only_closes_a_parked_server(self):
        launch = Launcher()
        with ss.use(self.KEY, launch):
            ss.close_parked()
            self.assertEqual(launch.closes, 0)
        launch = Launcher()
        with ss.use(self.KEY, launch, keep=lambda: True):
            pass
        ss.close_parked()
        self.assertEqual(launch.closes, 1)

    def test_comfys_cancel_closes_a_parked_server(self):
        mm = types.SimpleNamespace(interrupt_current_processing=lambda value=True: None)
        self.assertTrue(ss.install(mm))
        self.assertFalse(ss.install(mm))
        launch = Launcher()
        with ss.use(self.KEY, launch, keep=lambda: True):
            pass
        mm.interrupt_current_processing(False)
        self.assertEqual(launch.closes, 0)
        mm.interrupt_current_processing()
        self.assertEqual(launch.closes, 1)

    def test_comfys_unload_and_a_timer_close_a_parked_server(self):
        launch = Launcher()
        with ss.use(self.KEY, launch, keep=lambda: True):
            pass
        ss.vram_release.release()
        self.assertEqual(launch.closes, 1)
        with mock.patch.object(ss, "PARK_SECONDS", 0.05):
            launch = Launcher()
            with ss.use(self.KEY, launch, keep=lambda: True):
                pass
            time.sleep(0.4)
        self.assertEqual(launch.closes, 1)


class OpenModelTests(Clean):
    """``story_writer.open_model`` with the pack's session faked."""

    PATH = "/m/swift.ninfer"
    EXT = {"rewrite_mode": "pending clips", "rewrite_writer_model": "w", "rewrite_thinking": True, "rewrite_reasoning_budget": "2048", "rewrite_reasoning_budget_message": "Wrap up."}

    def open(self, share, *, concurrency=3, max_slots=8, fail=False, same_model=True, label="writer"):
        seen = {"sessions": [], "closed": 0}

        @contextlib.contextmanager
        def session(*args, **kwargs):
            seen["sessions"].append(kwargs)
            try:
                yield FakeServer(kwargs["slots"])
            finally:
                seen["closed"] += 1

        pack = (mock.Mock(), mock.Mock(), mock.Mock(session=session), mock.Mock(is_ninfer=lambda p: True), mock.Mock())
        files = lambda n, p, l: (self.PATH if (l == "writer" or same_model) else "/m/other.ninfer", "")
        with mock.patch.object(sw, "_pack", lambda: pack), mock.patch.object(sw.prompt_rewriter, "_vision_files", files), mock.patch.object(sw.vram_release, "release") as release:
            try:
                with sw.open_model(label, 4096, slot_ctx=65536, concurrency=concurrency, share=share, max_slots=max_slots) as (endpoint, runner):
                    seen["endpoint"] = endpoint
                    if fail:
                        raise RuntimeError("boom")
            except RuntimeError:
                pass
        seen["released"] = release.call_count
        return seen

    def test_linked_extender_with_the_same_model_keeps_the_server_with_flags_both_can_use(self):
        seen = self.open(self.EXT, concurrency=3)
        self.assertEqual(len(seen["sessions"]), 1)
        kw = seen["sessions"][0]
        self.assertEqual((kw["slots"], kw["pool_ctx"]), (8, 65536 * 8))   # --max-concurrency 8, --max-context 65536
        self.assertEqual(kw["reasoning"], {"enabled": True, "budget": 2048, "message": "Wrap up."})   # the Extender's budget and message, fixed at launch
        self.assertEqual((seen["closed"], seen["released"]), (0, 0))      # no close, no "story writer finished" release
        self.assertTrue(ss.current().parked)
        self.assertEqual(seen["endpoint"]["slots"], 8)

    def test_an_empty_budget_message_falls_back_to_the_pack_default_and_no_thinking_keeps_the_writers_budget(self):
        kw = self.open(dict(self.EXT, rewrite_reasoning_budget_message="", rewrite_thinking=False))["sessions"][0]
        self.assertEqual(kw["reasoning"], {"enabled": True, "budget": 4096, "message": sw.BUDGET_MESSAGE})

    def test_closed_when_nothing_will_attach(self):
        for name, share, kw in (("no extender", None, {}), ("rewrite off", dict(self.EXT, rewrite_mode="off"), {}), ("another model", self.EXT, {"same_model": False}),
                                ("wired widget", dict(self.EXT, rewrite_mode=["5", 0]), {}), ("strata", dict(self.EXT, rewrite_writer_model=sw.strata_backend.LABEL), {})):
            ss.close()
            seen = self.open(share, **kw)
            self.assertEqual((seen["closed"], seen["released"]), (1, 1), name)
            self.assertIsNone(ss.current(), name)

    def test_without_a_shared_extender_the_slots_follow_the_concurrency_up_to_the_limit(self):
        self.assertEqual(self.open(None, concurrency=8, max_slots=8)["sessions"][0]["slots"], 8)
        self.assertEqual(self.open(None, concurrency=8)["sessions"][0]["slots"], 8)
        self.assertEqual(self.open(None, concurrency=8, max_slots=4)["sessions"][0]["slots"], 4)

    def test_an_error_in_the_writer_closes_the_kept_server(self):
        seen = self.open(self.EXT, fail=True)
        self.assertEqual((seen["closed"], seen["released"]), (1, 1))
        self.assertIsNone(ss.current())


class RewriterAttachTests(Clean):
    """The Extender's rewriter on a server the writer parked: no launch, attached, closed when it is done."""

    SETTINGS = {"rewrite_mode": "pending clips", "rewrite_writer_model": "w", "rewrite_task": "Ref2VA", "rewrite_parallel": 3, "rewrite_max_new_tokens": 100,
                "rewrite_thinking": False, "rewrite_greedy": True, "rewrite_seed": 42}

    def rewrite(self, parked_meta, settings=None):
        folder = tempfile.mkdtemp()
        key = ss.key_of(str(Path(folder) / "writer-stub.gguf"), "mm.gguf")
        server = FakeServer(8)
        closer = Launcher()
        ss.acquire(key, lambda: (server, closer.close, parked_meta))
        ss.park(ss.current())
        clips = [{"id": 0, "title": "c", "prompt": "Ana walks.", "duration": 15, "seed": 1, "seed_mode": "fixed", "validated": False, "loras": []}]
        with mock.patch("tempfile.mkdtemp", return_value=folder):
            rec, notes, done = stub_pack.run_rewrite(ROOT, dict(self.SETTINGS, **(settings or {})), clips, refs=stub_pack.make_pictures((0,)))
        return rec, done, server, closer

    FITS = {"slots": 8, "slot_ctx": 65536, "reasoning": {"enabled": True, "budget": 4096, "message": "Time"}}

    def test_attaches_without_a_second_launch_and_closes_at_the_end(self):
        rec, clips, server, closer = self.rewrite(self.FITS)
        self.assertEqual(rec.sessions, [])             # the pack's session (vram_release + ninfer start) never ran
        self.assertEqual(rec.chats, [])                # the rewriter talked to the server it attached to
        self.assertEqual(len(server.chats), 1)
        self.assertEqual(clips[0]["prompt"], "FINAL")
        self.assertEqual(closer.closes, 1)
        self.assertIsNone(ss.current())

    def test_thinking_rewriter_needs_the_budget_and_message_the_server_was_started_with(self):
        thinking = {"rewrite_thinking": True, "rewrite_reasoning_budget": "2048", "rewrite_reasoning_budget_message": "Wrap up."}
        rec, clips, server, closer = self.rewrite(self.FITS, thinking)
        self.assertEqual(len(rec.sessions), 1)         # budget 4096 != 2048: closed, launched as before
        self.assertEqual(closer.closes, 1)
        rec, clips, server, closer = self.rewrite(dict(self.FITS, reasoning={"enabled": True, "budget": 2048, "message": "Wrap up."}), thinking)
        self.assertEqual(rec.sessions, [])

    def test_a_server_with_too_small_a_context_is_not_used(self):
        rec, clips, server, closer = self.rewrite(dict(self.FITS, slot_ctx=1024))
        self.assertEqual(len(rec.sessions), 1)


class ExtenderSourceTests(unittest.TestCase):
    def test_every_path_closes_the_kept_server_before_h3_loads(self):
        text = tsw.master_source()
        body = text[text.index("def extend("):]
        call = body.index("prompt_rewriter.rewrite_clips(")
        close = body.index("shared_server.close()\n\n        if str(run_mode) == \"prompts_only\":")
        self.assertGreater(close, call)                                          # after the rewriter, also when it is off or had nothing to do
        self.assertLess(close, body.index("_manifest_for_first("))              # before anything of H3
        self.assertLess(close, body.index("return self._prompts_only("))
        self.assertIn("except BaseException:\n                shared_server.close()\n                raise", body)   # a failing or cancelled rewrite

    @unittest.skipUnless((ROOT / "__init__.py").exists(), "no __init__.py in a scratch copy")
    def test_init_installs_the_cancel_hook(self):
        self.assertIn("shared_server.install()", (ROOT / "__init__.py").read_text(encoding="utf-8"))


class WriterMetadataTests(unittest.TestCase):
    OUT = [{"clip": 1, "seconds": 15, "goal": "Kael fights.", "raw": "0-5 s wide: Kael steps forward.", "shots": [{"start": 0}], "start_state": {}, "end_state": {"end_action": "Kael stands."},
            "new_recurring_visuals": ["the ring"]}]

    def graphs(self):
        prompt = {"3": {"class_type": "H3ShotWriter", "inputs": {"brief": "b", "last_output": "{}", "concurrency": 8}},
                  "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 1]}},
                  "5": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
                  "7": {"class_type": "MiniMaxH3MasterExtender", "inputs": {"clips_in": ["3", 0], "clips_json": "[]"}}}
        workflow = {"nodes": [{"id": 3, "widgets_values": ["b", 0, 15, "English", 42, "fixed", "w", "4096", 8, "{}"]}, {"id": 4, "widgets_values": [""]},
                              {"id": 5, "widgets_values": [""]}, {"id": 7, "widgets_values": ["x", "[]"]}]}
        return prompt, {"workflow": workflow}

    def test_the_writers_output_goes_into_the_api_graph_and_the_workflow(self):
        prompt, extra = self.graphs()
        self.assertEqual(cm.embed_writer_output(3, prompt, extra, self.OUT, "STORY", model="swift"), (True, True, 1))
        saved = json.loads(prompt["3"]["inputs"]["last_output"])
        self.assertEqual((saved["clips"], saved["story"], saved["model"]), (self.OUT, "STORY", "swift"))
        self.assertRegex(saved["written_at"], r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$")
        values = extra["workflow"]["nodes"][0]["widgets_values"]
        self.assertEqual(json.loads(values[-1]), saved)
        self.assertEqual(values[:-1], ["b", 0, 15, "English", 42, "fixed", "w", "4096", 8])   # nothing else moved
        self.assertEqual(extra["workflow"]["nodes"][3]["widgets_values"], ["x", "[]"])

    def test_a_preview_fed_by_the_story_output_shows_the_story_and_one_fed_by_the_clips_does_not(self):
        prompt, extra = self.graphs()
        cm.embed_writer_output(3, prompt, extra, self.OUT, "STORY")
        nodes = {n["id"]: n for n in extra["workflow"]["nodes"]}
        self.assertEqual(nodes[4]["widgets_values"], ["STORY"])
        self.assertEqual(nodes[5]["widgets_values"], [""])

    def test_a_workflow_converted_from_the_api_prompt_is_matched_by_node_type(self):
        """A script's UI workflow converted from its API prompt renumbers the nodes (API 990 -> UI 9): the one node of that type is used."""
        prompt, extra = self.graphs()
        prompt["990"] = prompt.pop("3")
        for n in extra["workflow"]["nodes"]:
            n["type"] = {3: "H3ShotWriter", 4: "PreviewAny", 5: "PreviewAny", 7: "MiniMaxH3MasterExtender"}[n["id"]]
        self.assertEqual(cm.embed_writer_output(990, prompt, extra, self.OUT, "STORY")[:2], (True, True))
        self.assertEqual(json.loads(extra["workflow"]["nodes"][0]["widgets_values"][-1])["story"], "STORY")
        prompt["980"] = prompt.pop("7")
        self.assertEqual(cm.embed_clips(980, prompt, extra, [{"id": 0, "prompt": "p"}]), (True, True))
        self.assertEqual(json.loads(extra["workflow"]["nodes"][3]["widgets_values"][1])[0]["prompt"], "p")

    def test_an_empty_queue_time_value_still_finds_the_last_slot(self):
        prompt, extra = self.graphs()
        prompt["3"]["inputs"]["last_output"] = ""
        extra["workflow"]["nodes"][0]["widgets_values"][-1] = ""
        self.assertEqual(cm.embed_writer_output(3, prompt, extra, self.OUT, "STORY")[:2], (True, True))
        self.assertEqual(json.loads(extra["workflow"]["nodes"][0]["widgets_values"][-1])["story"], "STORY")

    def test_a_graph_that_does_not_look_right_is_left_alone(self):
        self.assertEqual(cm.embed_writer_output(None, {}, {}, [], ""), (False, False, 0))
        prompt, extra = self.graphs()
        prompt["3"]["inputs"]["last_output"] = ["9", 0]
        self.assertEqual(cm.embed_writer_output(3, prompt, extra, self.OUT, "S")[:2], (False, False))
        self.assertEqual(cm.embed_writer_output(3, None, None, self.OUT, "S"), (False, False, 0))

    def test_write_embeds_what_it_wrote(self):
        prompt, extra = self.graphs()
        chat = tsw.Script(lambda kind, m, n: tsw.plan_json(1) if kind == "plan" else tsw.shots_json(1, tsw.GOOD))

        @contextlib.contextmanager
        def fake_open(label, budget, slot_ctx, concurrency, say, **kw):
            yield {"url": "http://x", "model": "m", "label": "swift.ninfer", "style": "chat", "slots": 8}, mock.Mock(interrupted=lambda: False)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sh, "open_model", fake_open), mock.patch.object(sh, "ServerChat", lambda *a, **k: chat), \
                mock.patch.object(sh, "output_root", lambda base=None: Path(tmp) / "out"):
            clips_json, story = sh.H3ShotWriter().write("Clip 1 - a fight", 1, 15, "English", 1, "w", "4096", unique_id="3", prompt=prompt, extra_pnginfo=extra)
        saved = json.loads(prompt["3"]["inputs"]["last_output"])
        self.assertEqual(saved["clips"], json.loads(clips_json))
        self.assertEqual((saved["story"], saved["model"]), (story, "swift.ninfer"))
        self.assertEqual(json.loads(extra["workflow"]["nodes"][0]["widgets_values"][-1])["story"], story)
        self.assertEqual(extra["workflow"]["nodes"][1]["widgets_values"], [story])

    def test_the_widget_is_the_last_optional_input_and_the_node_asks_for_the_workflow(self):
        spec = sh.H3ShotWriter.INPUT_TYPES()
        self.assertEqual(list(spec["optional"])[-1], "last_output")
        self.assertEqual(spec["optional"]["last_output"][1]["default"], "{}")
        self.assertEqual(spec["hidden"]["extra_pnginfo"], "EXTRA_PNGINFO")
        js = (ROOT / "web" / "master_extender.js").read_text(encoding="utf-8")
        self.assertIn('name !== "H3ShotWriter"', js)
        self.assertIn('w.name === "last_output"', js)

    def test_the_extender_leaves_the_writers_saved_output_alone(self):
        """The writer is not re-run (its output is a widget value of the loaded workflow): the Extender's own embedding touches only its clips_json."""
        prompt, extra = self.graphs()
        saved = json.dumps({"clips": self.OUT, "story": "OLD", "written_at": "2026-10-30 10:00:00", "model": "m"}, indent=2)
        prompt["3"]["inputs"]["last_output"] = saved
        extra["workflow"]["nodes"][0]["widgets_values"][-1] = saved
        before = copy.deepcopy((prompt["3"], extra["workflow"]["nodes"][0], extra["workflow"]["nodes"][1]))
        self.assertEqual(cm.embed_clips(7, prompt, extra, [{"id": 0, "prompt": "p"}]), (True, True))
        self.assertEqual(before, (prompt["3"], extra["workflow"]["nodes"][0], extra["workflow"]["nodes"][1]))
        self.assertEqual(json.loads(prompt["7"]["inputs"]["clips_json"]), [{"id": 0, "prompt": "p"}])


class ImportedClipsKeepTheWritersFieldsTests(unittest.TestCase):
    def test_each_clip_carries_the_writers_fields_and_they_change_no_fingerprint(self):
        clips = sh.import_clips(json.dumps(WriterMetadataTests.OUT))
        self.assertEqual(clips[0]["writer"], {"goal": "Kael fights.", "shots": [{"start": 0}], "start_state": {}, "end_state": {"end_action": "Kael stands."}, "new_recurring_visuals": ["the ring"]})
        self.assertTrue(clips[0]["from_writer"])
        bare = {k: v for k, v in clips[0].items() if k != "writer"}
        self.assertEqual(mp._clip_identity(clips[0]), mp._clip_identity(bare))
        self.assertEqual(mp._clip_fingerprint("p", clips[0], 1), mp._clip_fingerprint("p", bare, 1))
        self.assertEqual(mp._chain_key(clips), mp._chain_key([bare]))
        self.assertEqual(pr.source_of(clips[0]), pr.source_of(bare))
        self.assertEqual(pr.fingerprint(pr.source_of(clips[0]), 15, "16:9", "T2VA"), pr.fingerprint(pr.source_of(bare), 15, "16:9", "T2VA"))
        self.assertEqual(clips_metadata_roundtrip(clips), clips)


def clips_metadata_roundtrip(clips):
    return json.loads(cm.serialize(clips))


if __name__ == "__main__":
    unittest.main()
