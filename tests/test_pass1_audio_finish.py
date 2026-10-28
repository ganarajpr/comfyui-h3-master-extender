import ast
import hashlib
import json
import logging
import re
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pdd_pure_engine.py"
NODE = ROOT / "master_node.py"
PROJECTS = ROOT / "master_projects.py"
METHODS = ("configure_pass1_audio_finish", "_av_noise_mask", "_finish_pass1_audio", "_packed_tokens")
FUNCS = ("pass1_audio_finish_active", "with_pass1_audio_finish_p1_settings")


class FakeTensor:
    def __init__(self, shape, fill=None):
        self.shape = shape
        self.fill = fill


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def full(shape, value, dtype=None):
        return FakeTensor(tuple(shape), value)


class FakeNested:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)


class FakeNodes:
    NODE_CLASS_MAPPINGS = {}


class FakeCache:
    calls = []

    def patch(self, model, **kw):
        FakeCache.calls.append(kw)
        return (f"cached({model})",)


def _load():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MasterEngine")
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    stub = ast.ClassDef(name="Engine", bases=[], keywords=[], body=body, decorator_list=[])
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
    comfy = types.SimpleNamespace(nested_tensor=types.SimpleNamespace(NestedTensor=FakeNested))
    ns = {"_LOG": logging.getLogger("test_pass1_audio_finish"), "torch": FakeTorch, "comfy": comfy, "nodes": FakeNodes,
          "time": __import__("time"),
          "_streams_from_latent": lambda latent, name: latent[name].tensors}
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs + [stub], type_ignores=[])), str(ENGINE), "exec"), ns)
    return ns


NS = _load()


def make(steps=6):
    e = NS["Engine"]()
    e.pass1_steps = steps
    e.sampled = []
    e._sample_euler = lambda **kw: (e.sampled.append(kw), {"samples": "out", "noise_mask": "m"})[1]
    e._packed_tokens = lambda samples: 123
    return e


def latent():
    return {"samples": FakeNested([FakeTensor((1, 128, 4, 22, 40)), FakeTensor((1, 32, 2, 50))])}


class Active(unittest.TestCase):
    def test_only_after_an_early_stop(self):
        f = NS["pass1_audio_finish_active"]
        self.assertTrue(f(True, 6))
        self.assertTrue(f(True, 7))
        self.assertFalse(f(True, 0))
        self.assertFalse(f(True, 8))
        self.assertFalse(f(False, 6))

    def test_fingerprint_only_when_on(self):
        f = NS["with_pass1_audio_finish_p1_settings"]
        base = {"a": 1}
        self.assertIs(f(base, False), base)
        self.assertEqual(f(base, True), {"a": 1, "pass1_audio_finish": True})
        self.assertEqual(base, {"a": 1})


class Configure(unittest.TestCase):
    def setUp(self):
        FakeNodes.NODE_CLASS_MAPPINGS = {"H3FrozenVideoCache": FakeCache}

    def test_off_by_default(self):
        e = make()
        e.configure_pass1_audio_finish()
        self.assertFalse(e.pass1_audio_finish)

    def test_on(self):
        e = make()
        e.configure_pass1_audio_finish(True)
        self.assertTrue(e.pass1_audio_finish)

    def test_ignored_without_an_early_stop(self):
        e = make(steps=0)
        with self.assertLogs("test_pass1_audio_finish", "WARNING"):
            e.configure_pass1_audio_finish(True)
        self.assertFalse(e.pass1_audio_finish)

    def test_missing_pack_raises_a_clear_error(self):
        FakeNodes.NODE_CLASS_MAPPINGS = {}
        with self.assertRaisesRegex(RuntimeError, "ComfyUI-H3-AudioRefine"):
            make().configure_pass1_audio_finish(True)


class Finish(unittest.TestCase):
    def setUp(self):
        FakeCache.calls = []
        FakeNodes.NODE_CLASS_MAPPINGS = {"H3FrozenVideoCache": FakeCache}

    def test_mask_freezes_video_and_frees_audio(self):
        v, a = NS["Engine"]._av_noise_mask(latent()["samples"], False, True).tensors
        self.assertEqual((v.shape, v.fill), ((1, 1, 4, 22, 40), 0.0))
        self.assertEqual((a.shape, a.fill), ((1, 1, 2, 50), 1.0))
        v, a = NS["Engine"]._av_noise_mask(latent()["samples"], True, False).tensors
        self.assertEqual((v.fill, a.fill), (1.0, 0.0))

    def test_samples_the_remaining_grid_with_the_mask_and_cache(self):
        e = make()
        grid = [1.0, 0.9, 0.8, 0.75, 0.72, 0.7, 0.6969, 0.3, 0.0]
        out = e._finish_pass1_audio("model", "pos", latent(), grid[6:], 42, 0)
        self.assertEqual(len(e.sampled), 1)
        kw = e.sampled[0]
        self.assertEqual(kw["sigmas"], [0.6969, 0.3, 0.0])
        self.assertEqual((kw["model"], kw["positive"], kw["seed"]), ("cached(model)", "pos", 42))
        self.assertEqual([t.fill for t in kw["latent_image"]["noise_mask"].tensors], [0.0, 1.0])
        self.assertEqual(FakeCache.calls[0]["cache_contents"], "hidden")
        self.assertEqual(FakeCache.calls[0]["precision"], "int4")
        self.assertTrue(FakeCache.calls[0]["free_after_pass"])
        self.assertNotIn("noise_mask", out)


class NodeAndFingerprint(unittest.TestCase):
    def test_input_is_boolean_and_appended_last(self):
        text = NODE.read_text(encoding="utf-8")
        self.assertRegex(text, r'"pass1_audio_finish": \("BOOLEAN", \{"default": False')
        names = re.findall(r'^\s{16}"(\w+)": \(', text[text.index("def INPUT_TYPES"):], re.M)
        self.assertEqual(names[-1], "pass1_audio_finish")

    def test_clip_fingerprint_changes_only_when_set(self):
        tree = ast.parse(PROJECTS.read_text(encoding="utf-8"))
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_clip_identity", "_clip_fingerprint")]
        ns = {"json": json, "hashlib": hashlib}
        exec(compile(ast.Module(body=fns, type_ignores=[]), str(PROJECTS), "exec"), ns)
        fp, clip = ns["_clip_fingerprint"], {"prompt": "x"}
        base = fp("", clip, 1, None, True)
        self.assertEqual(base, fp("", clip, 1, None, True, False, None, False, False))
        frozen = fp("", clip, 1, None, True, False, None, True, False)
        finish = fp("", clip, 1, None, True, False, None, False, True)
        self.assertEqual(len({base, frozen, finish, fp("", clip, 1, None, True, True)}), 4)


if __name__ == "__main__":
    unittest.main()
