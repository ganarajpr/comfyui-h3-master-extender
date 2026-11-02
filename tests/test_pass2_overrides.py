import ast
import logging
import re
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pdd_pure_engine.py"
NODE = ROOT / "master_node.py"
METHODS = ("configure_pass2_extras", "_basic_sigmas", "_pass2_basic_sigmas", "_pass2_sigmas", "_pass2_lora_active")

CALLS = []


class BasicScheduler:
    @classmethod
    def execute(cls, model, scheduler, steps, denoise):
        CALLS.append((scheduler, steps, denoise))
        return ([float(steps)],)


def _engine_class():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MasterEngine")
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    stub = ast.ClassDef(name="Engine", bases=[], keywords=[], body=body, decorator_list=[])
    ns = {"_LOG": logging.getLogger("test_pass2_overrides"), "_safe_get_output": lambda res, i, name: res[i]}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), str(ENGINE), "exec"), ns)
    return ns["Engine"]


Engine = _engine_class()


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
    for k, v in kw.items():
        setattr(e, k, v)
    return e


class Pass2Overrides(unittest.TestCase):
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

    def test_default_uses_pass1_scheduler(self):
        e = make()
        e._pass2_sigmas(None, 0.4)
        self.assertEqual(CALLS, [("sgm_uniform", 3, 0.4)])
        e.configure_pass2_extras()
        CALLS.clear()
        e._pass2_sigmas(None, 0.4)
        self.assertEqual(CALLS, [("sgm_uniform", 3, 0.4)])

    def test_override_in_turbo_path(self):
        e = make()
        e.configure_pass2_extras("beta", "pass 2 (refined)")
        e._pass2_sigmas(None, 0.4)
        self.assertEqual(CALLS, [("beta", 3, 0.4)])
        self.assertFalse(e.pass2_audio_from_p1)

    def test_override_in_pass2_lora_path(self):
        e = make(pass2_lora="dmad", pass2_steps=4)
        e.configure_pass2_extras("beta", "pass 1")
        e._pass2_sigmas(None, 0.25)
        self.assertEqual(CALLS, [("beta", 1, 0.25)])
        self.assertTrue(e.pass2_audio_from_p1)

    def test_pass1_schedule_is_not_overridden(self):
        e = make()
        e.configure_pass2_extras("beta")
        e._basic_sigmas(None, 8, 1.0)
        self.assertEqual(CALLS, [("sgm_uniform", 8, 1.0)])


class NodeInputs(unittest.TestCase):
    def test_new_inputs_and_resolutions(self):
        text = NODE.read_text(encoding="utf-8")
        self.assertIn('"736x416 (16:9)", "416x736 (9:16)"', text)
        self.assertRegex(text, r'"pass2_scheduler": \(\["same as pass 1"\] \+ list\(comfy\.samplers\.KSampler\.SCHEDULERS\), \{"default": "same as pass 1"')
        self.assertRegex(text, r'"pass2_audio": \(\["pass 2 \(refined\)", "pass 1"\], \{"default": "pass 2 \(refined\)"')
        names = re.findall(r'^\s{16}"(\w+)": \(', text[text.index("def INPUT_TYPES"):], re.M)
        self.assertEqual(names[-3:-1], ["pass2_scheduler", "pass2_audio"])

    def test_parse_resolution(self):
        tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_resolution")
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ENGINE), "exec"), ns)
        self.assertEqual(ns["parse_resolution"]("736x416 (16:9)", 1, 1), (736, 416))
        self.assertEqual(ns["parse_resolution"]("416x736 (9:16)", 1, 1), (416, 736))


if __name__ == "__main__":
    unittest.main()
