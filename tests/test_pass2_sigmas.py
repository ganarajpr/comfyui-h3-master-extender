import ast
import logging
import math
import re
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pdd_pure_engine.py"
NODE = ROOT / "master_node.py"
PROJECTS = ROOT / "master_projects.py"
METHODS = ("configure_pass2_extras", "_basic_sigmas", "_pass2_sigmas", "_pass2_lora_active")

CALLS = []


class BasicScheduler:
    @classmethod
    def execute(cls, model, scheduler, steps, denoise):
        CALLS.append((scheduler, steps, denoise))
        return ([float(steps)],)


class FakeTensor(list):
    pass


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def tensor(values, dtype=None):
        t = FakeTensor(values)
        t.dtype = dtype
        return t


def _load():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MasterEngine")
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    stub = ast.ClassDef(name="Engine", bases=[], keywords=[], body=body, decorator_list=[])
    parse = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_pass2_sigmas")
    ns = {"_LOG": logging.getLogger("test_pass2_sigmas"), "_safe_get_output": lambda res, i, name: res[i],
          "math": math, "torch": FakeTorch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[parse, stub], type_ignores=[])), str(ENGINE), "exec"), ns)
    return ns["Engine"], ns["parse_pass2_sigmas"]


Engine, parse = _load()


def make(**kw):
    e = Engine()
    e.dual_active = False
    e.hyperflow_mode = False
    e.turbo_mode = True
    e.turbo_steps = 8
    e.scheduler_name = "sgm_uniform"
    e.pass2_steps = 0
    e.pass2_lora_mode = "replace"
    e.pass2_lora = None
    e.pass1_sigmas = [1.0, 0.9, 0.5, 0.2, 0.0]
    e.dual_split_step = 2
    for k, v in kw.items():
        setattr(e, k, v)
    return e


class Parse(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse("0.909375, 0.857143, 0.666667, 0"), [0.909375, 0.857143, 0.666667, 0.0])
        self.assertEqual(parse(" 1 ,0 "), [1.0, 0.0])

    def test_empty_is_none(self):
        self.assertIsNone(parse(""))
        self.assertIsNone(parse("   "))
        self.assertIsNone(parse(None))

    def test_invalid(self):
        for bad in ("0.9", "0.9, abc, 0", "0.5, 0.7, 0", "0.9, 0.9, 0", "0.9, 0.5", "1.2, 0.5, 0", "0.9, -0.1, 0", "0.9, nan, 0"):
            with self.assertRaises(ValueError, msg=bad):
                parse(bad)


class Override(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        mod = types.ModuleType("comfy_extras.nodes_custom_sampler")
        mod.BasicScheduler = BasicScheduler
        self._saved = {k: sys.modules.get(k) for k in ("comfy_extras", "comfy_extras.nodes_custom_sampler")}
        sys.modules["comfy_extras"] = types.ModuleType("comfy_extras")
        sys.modules["comfy_extras.nodes_custom_sampler"] = mod

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_default_unchanged(self):
        e = make()
        e.configure_pass2_extras()
        self.assertFalse(e.pass2_audio_from_p1)
        self.assertIsNone(e.pass2_sigmas_override)
        e._pass2_sigmas(None, 0.4)
        self.assertEqual(CALLS, [("sgm_uniform", 3, 0.4)])

    def test_override_replaces_turbo(self):
        e = make()
        e.configure_pass2_extras("refine", "0.909375, 0.857143, 0.666667, 0")
        out = e._pass2_sigmas(None, 0.4)
        self.assertEqual(list(out), [0.909375, 0.857143, 0.666667, 0.0])
        self.assertEqual(CALLS, [])

    def test_override_replaces_lora_tail_and_hyperflow_tail(self):
        e = make(pass2_lora="dmad", pass2_steps=4)
        e.configure_pass2_extras("refine", "0.9, 0.5, 0")
        self.assertEqual(list(e._pass2_sigmas(None, 0.25)), [0.9, 0.5, 0.0])
        e = make(turbo_mode=False, hyperflow_mode=True)
        e.configure_pass2_extras("refine", "0.9, 0.5, 0")
        self.assertEqual(list(e._pass2_sigmas(None, 0.25)), [0.9, 0.5, 0.0])

    def test_dual_sampling_ignores_override(self):
        e = make(turbo_mode=False, hyperflow_mode=True, dual_active=True)
        with self.assertLogs("test_pass2_sigmas", "WARNING") as cm:
            e.configure_pass2_extras("refine", "0.9, 0.5, 0")
        self.assertIn("pass2_sigmas ignored", cm.output[0])
        self.assertIsNone(e.pass2_sigmas_override)

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            make().configure_pass2_extras("refine", "0.5, 0.9, 0")

    def test_audio_option(self):
        e = make()
        e.configure_pass2_extras("pass 1")
        self.assertTrue(e.pass2_audio_from_p1)
        e.configure_pass2_extras("refine")
        self.assertFalse(e.pass2_audio_from_p1)
        self.assertFalse(e.pass2_audio_frozen)

    def test_frozen_option_keeps_the_pass1_audio(self):
        e = make()
        e.configure_pass2_extras("frozen")
        self.assertTrue(e.pass2_audio_frozen)
        self.assertTrue(e.pass2_audio_from_p1)


class NodeAndFingerprint(unittest.TestCase):
    def test_inputs_appended_last(self):
        text = NODE.read_text(encoding="utf-8")
        self.assertRegex(text, r'"pass2_audio": \(\["refine", "pass 1", "frozen"\], \{"default": "refine"')
        self.assertRegex(text, r'"pass2_sigmas": \("STRING", \{"default": ""')
        names = re.findall(r'^\s{16}"(\w+)": \(', text[text.index("def INPUT_TYPES"):], re.M)
        self.assertEqual(names[-3:], ["pass2_audio", "pass2_sigmas", "pass1_audio_finish"])

    def test_fingerprint_changes_only_when_set(self):
        tree = ast.parse(PROJECTS.read_text(encoding="utf-8"))
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_clip_identity", "_clip_fingerprint")]
        import hashlib, json
        ns = {"json": json, "hashlib": hashlib}
        exec(compile(ast.Module(body=fns, type_ignores=[]), str(PROJECTS), "exec"), ns)
        fp, clip = ns["_clip_fingerprint"], {"prompt": "x"}
        base = fp("", clip, 1, None, True)
        self.assertEqual(base, fp("", clip, 1, None, True, False, None))
        self.assertNotEqual(base, fp("", clip, 1, None, True, True))
        self.assertNotEqual(base, fp("", clip, 1, None, True, False, [0.9, 0.5, 0.0]))
        self.assertNotEqual(fp("", clip, 1, None, True, False, [0.9, 0.5, 0.0]), fp("", clip, 1, None, True, False, [0.8, 0.5, 0.0]))


if __name__ == "__main__":
    unittest.main()
