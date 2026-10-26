import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clips_metadata as cm  # noqa: E402

QUEUED = json.dumps([{"id": 0, "prompt": "", "duration": 15}], indent=2)
FINAL = [{"id": 0, "prompt": "shot_1: ä", "planned": True, "prompt_rewritten": True, "rewrite_text": "x",
          "rewrite_meta": {"engine": "e4", "fingerprint": "v2:abc"}, "seed": 7, "validated": False},
         {"id": 1, "prompt": "second", "planned": True, "seed": 8}]


def graphs(values, uid="12"):
    prompt = {uid: {"class_type": "MiniMaxH3MasterExtender", "inputs": {"clips_json": QUEUED, "auto_clips": 40}},
              "13": {"class_type": "SaveVideo", "inputs": {}}}
    workflow = {"nodes": [{"id": 3, "type": "Other", "widgets_values": [QUEUED]},
                          {"id": int(uid), "type": "MiniMaxH3MasterExtender", "widgets_values": values}]}
    return prompt, {"workflow": workflow}


class EmbedClips(unittest.TestCase):
    def test_list_widgets_values_and_api_graph(self):
        prompt, extra = graphs(["a", 1, QUEUED, True, "b"])
        self.assertEqual((True, True), cm.embed_clips("12", prompt, extra, FINAL))
        self.assertEqual(FINAL, json.loads(prompt["12"]["inputs"]["clips_json"]))
        values = extra["workflow"]["nodes"][1]["widgets_values"]
        self.assertEqual(FINAL, json.loads(values[2]))
        self.assertEqual(["a", 1, True, "b"], values[:2] + values[3:])
        self.assertEqual(QUEUED, extra["workflow"]["nodes"][0]["widgets_values"][0], "another node is untouched")
        self.assertIn("ä", values[2], "not ASCII-escaped, like the panel's JSON.stringify")

    def test_dict_widgets_values(self):
        prompt, extra = graphs({"clips_json": QUEUED, "auto_clips": 40})
        self.assertEqual((True, True), cm.embed_clips(12, prompt, extra, FINAL))
        values = extra["workflow"]["nodes"][1]["widgets_values"]
        self.assertEqual(FINAL, json.loads(values["clips_json"]))
        self.assertEqual(40, values["auto_clips"])

    def test_string_node_ids(self):
        prompt, extra = graphs(["a", QUEUED])
        extra["workflow"]["nodes"][1]["id"] = "12"
        self.assertEqual((True, True), cm.embed_clips("12", prompt, extra, FINAL))

    def test_missing_extra_pnginfo_still_updates_api_graph(self):
        prompt, _ = graphs([QUEUED])
        self.assertEqual((True, False), cm.embed_clips("12", prompt, None, FINAL))
        self.assertEqual(FINAL, json.loads(prompt["12"]["inputs"]["clips_json"]))

    def test_missing_prompt_updates_nothing_in_the_workflow(self):
        _, extra = graphs([QUEUED])
        before = copy.deepcopy(extra)
        self.assertEqual((False, False), cm.embed_clips("12", None, extra, FINAL))
        self.assertEqual(before, extra, "without the queue-time value the slot cannot be proven, so the graph is left alone")

    def test_missing_node_is_a_no_op(self):
        prompt, extra = graphs([QUEUED])
        before = copy.deepcopy((prompt, extra))
        self.assertEqual((False, False), cm.embed_clips("99", prompt, extra, FINAL))
        self.assertEqual(before, (prompt, extra))

    def test_node_missing_from_workflow_only(self):
        prompt, extra = graphs([QUEUED])
        extra["workflow"]["nodes"].pop()
        self.assertEqual((True, False), cm.embed_clips("12", prompt, extra, FINAL))

    def test_ambiguous_list_slot_is_skipped(self):
        prompt, extra = graphs([QUEUED, QUEUED])
        self.assertEqual((True, False), cm.embed_clips("12", prompt, extra, FINAL))
        self.assertEqual([QUEUED, QUEUED], extra["workflow"]["nodes"][1]["widgets_values"])

    def test_compact_format_without_the_widget_is_skipped(self):
        prompt, extra = graphs(["a", 1, True])
        self.assertEqual((True, False), cm.embed_clips("12", prompt, extra, FINAL))
        self.assertEqual(["a", 1, True], extra["workflow"]["nodes"][1]["widgets_values"])

    def test_linked_clips_json_is_not_overwritten(self):
        prompt, extra = graphs([QUEUED])
        prompt["12"]["inputs"]["clips_json"] = ["5", 0]
        self.assertEqual((False, False), cm.embed_clips("12", prompt, extra, FINAL))
        self.assertEqual(["5", 0], prompt["12"]["inputs"]["clips_json"])

    def test_junk_never_raises(self):
        for args in ((None, None, None, None), ("12", {"12": 5}, {"workflow": 3}, FINAL), ("12", [], "x", FINAL),
                     ("12", {"12": {"inputs": None}}, {"workflow": {"nodes": [None, 4]}}, FINAL), ("12", {}, {}, "notalist")):
            self.assertEqual((False, False), cm.embed_clips(*args))

    def test_second_write_replaces_the_first(self):
        prompt, extra = graphs([QUEUED])
        cm.embed_clips("12", prompt, extra, FINAL[:1])
        self.assertEqual((True, True), cm.embed_clips("12", prompt, extra, FINAL))
        self.assertEqual(FINAL, json.loads(extra["workflow"]["nodes"][1]["widgets_values"][0]))
        self.assertEqual(FINAL, json.loads(prompt["12"]["inputs"]["clips_json"]))

    def test_round_trip_is_the_clip_list(self):
        prompt, extra = graphs([QUEUED])
        cm.embed_clips("12", prompt, extra, FINAL)
        self.assertEqual(FINAL, json.loads(extra["workflow"]["nodes"][1]["widgets_values"][0]))
        self.assertEqual(prompt["12"]["inputs"]["clips_json"], extra["workflow"]["nodes"][1]["widgets_values"][0])


if __name__ == "__main__":
    unittest.main()
