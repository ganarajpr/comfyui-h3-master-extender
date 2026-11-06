import ast
import logging
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pdd_pure_engine.py"
NODE = ROOT / "master_node.py"
GRID = [1.0, 0.9939, 0.9843, 0.9661, 0.9231, 0.8349, 0.6969, 0.4688, 0.0]
MODE = "HyperFlow Flash (GGUF)"
REF = "minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf"
FL = "minimax_h3_fl2va_hyperflow_flash_14b-Q8_0.gguf"


def _engine_ns():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body
              if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("FLASH_ACCEL_MODE", "FLASH_STEPS", "FLASH_DEFAULT_FILE")
                                                    for t in n.targets))
              or (isinstance(n, ast.FunctionDef) and n.name in ("resolve_flash_file", "check_flash_steps", "hyperflow_tail_sigmas",
                                                               "dual_split_index", "dual_pass1_sigmas", "dual_pass2_sigmas"))]
    ns = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns


NS = _engine_ns()


def _engine_cls(*names):
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MasterEngine")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    stub = ast.ClassDef(name="Engine", bases=[], keywords=[], body=methods, decorator_list=[])
    ns = dict(NS, _LOG=logging.getLogger("test_hyperflow_flash"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), str(ENGINE), "exec"), ns)
    return ns["Engine"]


Engine = _engine_cls("configure_dual_sampling", "configure_pass1_steps", "configure_pass2_lora", "_pass2_sigmas", "_pass2_lora_active")


def flash_engine(lora=None):
    e = Engine()
    e.turbo_mode, e.flash_mode, e.hyperflow_mode = False, True, False
    e.hyperflow_pruned_refit = False
    e.pass2_lora = lora
    e.pass2_model = None
    e.pass2_steps = 4
    e.turbo_steps = "8"
    e.pass1_sigmas = list(GRID)
    e.dual_active = False
    e.dual_split_step = 2
    e.pass1_early_stop = False
    e.pass1_steps = 0
    return e


class FlashFile(unittest.TestCase):
    def test_ref2va_file_is_used_as_is(self):
        self.assertEqual(NS["resolve_flash_file"](REF, [REF, FL]), (REF, None))

    def test_fl2va_pick_is_swapped_for_the_ref2va_sibling(self):
        name, note = NS["resolve_flash_file"](FL, [REF, FL])
        self.assertEqual(name, REF)
        self.assertIn("fl2va", note)

    def test_fl2va_without_a_sibling_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "ref2va"):
            NS["resolve_flash_file"](FL, [FL])

    def test_unknown_file_is_an_error_and_empty_means_default(self):
        with self.assertRaises(ValueError):
            NS["resolve_flash_file"]("other_flash.gguf", [REF])
        self.assertEqual(NS["resolve_flash_file"]("", [REF])[0], REF)

    def test_steps(self):
        check = NS["check_flash_steps"]
        self.assertEqual([check(s) for s in (4, 6, 8, "6")], [4, 6, 8, 6])
        self.assertEqual(check(None), 6)
        for bad in (5, 7, 12):
            with self.assertRaises(ValueError):
                check(bad)


class FlashEngine(unittest.TestCase):
    def test_dual_sampling_and_pass1_steps_are_ignored(self):
        e = flash_engine()
        e.configure_dual_sampling(True, 2, True)
        self.assertFalse(e.dual_active)
        self.assertFalse(e.pass1_early_stop)
        e.configure_pass1_steps(4)
        self.assertEqual(e.pass1_steps, 0)

    def test_pass2_lora_is_forced_to_replace(self):
        e = flash_engine()
        e.configure_pass2_lora("dmad.safetensors", 1.0, "stack on engine LoRA", 4)
        self.assertEqual(e.pass2_lora_mode, "replace")

    def test_pass2_with_dmad_uses_the_scheduler_tail(self):
        e = flash_engine(lora="dmad.safetensors")
        e.pass2_lora_mode = "replace"
        calls = []
        e._pass2_basic_sigmas = lambda model, tail, denoise: calls.append((tail, denoise)) or "sig"
        self.assertEqual(e._pass2_sigmas(None, 0.25), "sig")
        self.assertEqual(calls, [(1, 0.25)])

    def test_pass2_without_a_lora_cuts_the_flash_grid_tail(self):
        e = flash_engine()
        self.assertEqual(e._pass2_sigmas(None, 0.25), GRID[-3:])


class NodeWiring(unittest.TestCase):
    text = NODE.read_text(encoding="utf-8")

    def test_mode_is_appended_last_and_default_is_unchanged(self):
        tree = ast.parse(self.text)
        modes = next(ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ACCEL_MODES" for t in n.targets))
        self.assertEqual(modes[-1], MODE)
        self.assertEqual(modes[:3], ["HyperFlow 8-step", "Turbo LoRA", "HyperFlow 8-step · dual sampling"])
        self.assertRegex(self.text, r'DUAL_SAMPLING_ACCEL_MODE = ACCEL_MODES\[2\]')

    def test_new_inputs_are_last_before_the_clips_in_socket(self):
        import re
        names = re.findall(r'^\s{16}"(\w+)": \(', self.text[self.text.index("def INPUT_TYPES"):], re.M)
        self.assertEqual(names[-3:], ["hyperflow_flash_file", "hyperflow_flash_steps", "clips_in"])
        self.assertRegex(self.text, r'"hyperflow_flash_file": \(_flash_files\(\), \{"default": FLASH_DEFAULT_FILE')
        self.assertRegex(self.text, r'"hyperflow_flash_steps": \("INT", \{"default": 6, "min": 4, "max": 8, "step": 2')

    def test_flash_fingerprint_key_only_in_flash_mode(self):
        i = self.text.index("if accel_mode == FLASH_ACCEL_MODE:\n            # Only in Flash mode")
        self.assertIn('p1_settings["flash"] = [flash_file, flash_steps]', self.text[i:i + 300])
        # the literal that other modes' caches are keyed on is unchanged
        tree = ast.parse(self.text)
        literal = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                       and any(getattr(t, "id", "") == "p1_settings" for t in n.targets))
        self.assertNotIn("flash", [k.value for k in literal.value.keys])

    def test_dual_sampling_is_off_in_flash_mode(self):
        self.assertIn("accel_mode != FLASH_ACCEL_MODE", self.text)


if __name__ == "__main__":
    unittest.main()
