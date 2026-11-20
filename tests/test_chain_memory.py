import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("chain_memory_t", ROOT / "chain_memory.py")
cm = importlib.util.module_from_spec(_spec)
sys.modules["chain_memory_t"] = cm
_spec.loader.exec_module(cm)

SIX = """## subject_definitions
<Subject 1> is a man.
<Subject 2> is a woman.
<Subject 3> is a room.

## summary
They talk.

## retention_analysis
<Subject 1> (<Picture 1>): fully_preserved
<Subject 2> (<Picture 2>): fully_preserved
<Subject 3> (<Picture 3>): fully_preserved

## detailed_description
[Shot 1] A talk.

## overall_soundscape
Room tone.

## non_diegetic_music
N/A"""


class AppendText(unittest.TestCase):
    def test_hash_sections(self):
        out = cm.append_memory_text(SIX, 4).split("\n")
        d = cm.MEMORY_DEFINITION.format(s=4, p=4)
        r = cm.MEMORY_RETENTION.format(s=4, p=4)
        self.assertEqual(out[out.index("<Subject 3> is a room.") + 1], d)
        self.assertEqual(out[out.index("<Subject 3> (<Picture 3>): fully_preserved") + 1], r)
        self.assertEqual(out[out.index(d) + 1], "")
        self.assertEqual(out[out.index(r) + 1], "")
        self.assertEqual([x for x in out if x not in (d, r)], SIX.split("\n"))

    def test_bare_sections(self):
        prompt = "subject_definitions:\n<Subject 1> is a man.\nsummary:\nTalk.\nretention_analysis:\n<Subject 1> (<Picture 1>): fully_preserved\ndetailed_description:\nShot."
        out = cm.append_memory_text(prompt, 2).split("\n")
        self.assertEqual(out[1:3], ["<Subject 1> is a man.", cm.MEMORY_DEFINITION.format(s=2, p=2)])
        self.assertEqual(out[out.index("retention_analysis:") + 2], cm.MEMORY_RETENTION.format(s=2, p=2))
        self.assertEqual(out[out.index("detailed_description:") - 1], cm.MEMORY_RETENTION.format(s=2, p=2))

    def test_plain_prompt(self):
        out = cm.append_memory_text("A man walks.", 1)
        self.assertEqual(out, "A man walks.\n\n" + cm.MEMORY_DEFINITION.format(s=1, p=1) + "\n" + cm.MEMORY_RETENTION.format(s=1, p=1))

    def test_one_section_missing(self):
        prompt = "## subject_definitions\n<Subject 1> is a man.\n\n## summary\nx"
        out = cm.append_memory_text(prompt, 2)
        self.assertIn("<Subject 1> is a man.\n" + cm.MEMORY_DEFINITION.format(s=2, p=2) + "\n\n## summary", out)
        self.assertTrue(out.endswith("\n\n" + cm.MEMORY_RETENTION.format(s=2, p=2)))

    def test_highest_subject_number_anywhere(self):
        self.assertEqual(cm.next_subject_number("<Subject 2> <Subject 7> <Subject 3>"), 8)
        self.assertEqual(cm.next_subject_number("nothing"), 1)
        out = cm.append_memory_text("see <Subject 6> here", 5)
        self.assertIn("<Subject 7> is the state", out)


class FakeFace:
    def __init__(self, score):
        self.det_score = score


class FakeDetector:
    """Counts faces from the frame's top-left pixel: value v in [0,1] -> round(v * 10) faces."""

    def get(self, bgr):
        return [FakeFace(0.9)] * int(round(bgr[0, 0, 0] / 255 * 10)) + [FakeFace(0.1)]


def make_video(shots):
    """shots: [(length, brightness, faces)]; each shot a flat colour (distinct from its neighbours) with faces encoded in pixel (0,0)."""
    frames = []
    for length, level, faces in shots:
        f = torch.full((length, 36, 64, 3), level)
        f[:, 0, 0, :] = faces / 10
        frames.append(f)
    return torch.cat(frames, 0)


@unittest.skipIf(torch is None, "torch not installed")
class Select(unittest.TestCase):
    def setUp(self):
        self._det, self._failed = cm._detector, cm._detector_failed
        cm._detector, cm._detector_failed = FakeDetector(), False

    def tearDown(self):
        cm._detector, cm._detector_failed = self._det, self._failed

    def test_cuts(self):
        video = make_video([(30, 0.2, 0), (30, 0.8, 0), (20, 0.4, 0)])
        self.assertEqual(cm._shot_ranges(video), [(0, 30), (30, 60), (60, 80)])

    def test_short_shot_merges(self):
        video = make_video([(30, 0.2, 0), (3, 0.9, 0), (30, 0.5, 0)])
        self.assertEqual(cm._shot_ranges(video), [(0, 33), (33, 63)])

    def test_latest_shot_with_max_faces(self):
        video = make_video([(30, 0.2, 2), (30, 0.8, 2), (20, 0.4, 0)])
        frame, info = cm.select_memory_frame(video)
        self.assertEqual(frame.shape, (1, 36, 64, 3))
        self.assertFalse(info["fallback"])
        self.assertTrue(30 <= info["index"] < 60)
        self.assertEqual(info["index"], 54)

    def test_more_faces_earlier_wins(self):
        video = make_video([(30, 0.2, 3), (30, 0.8, 1), (20, 0.4, 0)])
        _, info = cm.select_memory_frame(video)
        self.assertTrue(info["index"] < 30)

    def test_no_faces_falls_back_to_last(self):
        video = make_video([(30, 0.2, 0), (30, 0.8, 0)])
        frame, info = cm.select_memory_frame(video)
        self.assertTrue(info["fallback"])
        self.assertEqual(info["index"], 59)
        self.assertTrue(torch.equal(frame, video[59:60]))

    def test_detector_unavailable(self):
        cm._detector, cm._detector_failed = None, True
        video = make_video([(30, 0.2, 2), (30, 0.8, 2)])
        frame, info = cm.select_memory_frame(video)
        self.assertTrue(info["fallback"])
        self.assertTrue(torch.equal(frame, video[-1:]))


class Modes(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(cm.parse_mode(None), "off")
        self.assertEqual(cm.parse_mode(False), "off")
        self.assertEqual(cm.parse_mode(True), "picture")
        self.assertEqual(cm.parse_mode(""), "off")
        self.assertEqual(cm.parse_mode("junk"), "off")
        self.assertEqual(cm.parse_mode(1), "off")
        for m in cm.MODES:
            self.assertEqual(cm.parse_mode(m), m)
        self.assertTrue(cm.uses_context("context") and cm.uses_context("picture + context") and not cm.uses_context("picture"))
        self.assertTrue(cm.uses_picture("picture") and cm.uses_picture("picture + context") and not cm.uses_picture("context"))


def make_latent(t, nested=False):
    video = torch.arange(t, dtype=torch.float32).reshape(1, 1, t, 1, 1).expand(1, 2, t, 3, 3).clone()
    audio_t = int(round(cm._FRAME_RESCALE * cm._pixel_frames(t)))
    audio = torch.arange(audio_t, dtype=torch.float32).reshape(1, 1, 1, audio_t).expand(1, 2, 2, audio_t).clone()
    return {"samples": [video, audio], "other": "keep"}


@unittest.skipIf(torch is None, "torch not installed")
class Truncate(unittest.TestCase):
    T = 27  # 2 mod 5, 90 frames

    def check(self, drop, steps):
        lat = make_latent(self.T)
        out, info = cm.truncate_context_latent_info(lat, drop)
        v, a = out["samples"]
        self.assertEqual(int(v.shape[2]), self.T - steps)
        self.assertEqual(int(v.shape[2]) % 5, 2)
        self.assertEqual(int(a.shape[-1]), int(round(5 / 3 * cm._pixel_frames(self.T - steps))))
        self.assertEqual(out["other"], "keep")
        self.assertEqual(int(v[0, 0, -1, 0, 0]), self.T - steps - 1)
        # audio comes from the clip's TRUE end: its last step is the original last step
        full_audio = int(round(5 / 3 * cm._pixel_frames(self.T)))
        self.assertEqual(int(a[0, 0, 0, -1]), full_audio - 1)
        self.assertEqual(int(a[0, 0, 0, 0]), full_audio - a.shape[-1])
        self.assertEqual(info["dropped_steps"], steps)
        self.assertEqual(info["dropped_frames"], 17 * (steps // 5))
        self.assertEqual(len(lat["samples"]), 2)
        self.assertEqual(lat["samples"][0].shape[2], self.T)

    def test_drop_0_unchanged(self):
        lat = make_latent(self.T)
        out, info = cm.truncate_context_latent_info(lat, 0)
        self.assertIs(out, lat)
        self.assertFalse(info["applied"])

    def test_drop_10(self):
        self.check(10, 5)

    def test_drop_17(self):
        self.check(17, 5)

    def test_drop_30(self):
        self.check(30, 10)

    def test_not_enough_frames(self):
        lat = make_latent(self.T)
        # T=27 is 90 frames; dropping 4 cycles (68 frames) leaves 22 < 22 + 17
        out, info = cm.truncate_context_latent_info(lat, 68)
        self.assertIs(out, lat)
        self.assertFalse(info["applied"])
        self.assertIn("would remain", info["reason"])
        out, _ = cm.truncate_context_latent_info(lat, 34, context_length=56)
        self.assertIs(out, lat)


if __name__ == "__main__":
    unittest.main()
