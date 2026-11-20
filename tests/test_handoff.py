import ast
import logging
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pdd_pure_engine.py"
NODE = ROOT / "master_node.py"
MODE = "Pika → Flash hand-off"
GRID = [1.0, 0.9939, 0.9843, 0.9661, 0.9231, 0.8349, 0.6969, 0.4688, 0.0]
PIKA = [1.0, 0.98, 0.9, 0.7, 0.4, 0.15, 0.0]


def _ns():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body
              if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("HANDOFF_ACCEL_MODE", "HANDOFF_STEP_DEFAULT", "HANDOFF_STEP_OPTIONS", "HANDOFF_STEP_DEFAULT_LABEL") for t in n.targets))
              or (isinstance(n, ast.FunctionDef) and n.name in ("clamp_handoff_step", "split_handoff_sigmas", "parse_handoff_step"))]
    ns = {"re": re, "_LOG": logging.getLogger("test_handoff")}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns


NS = _ns()
NODE_TEXT = NODE.read_text(encoding="utf-8")
ENGINE_TEXT = ENGINE.read_text(encoding="utf-8")


class Split(unittest.TestCase):
    def test_split_at_step_3(self):
        pika, flash, sigma = NS["split_handoff_sigmas"](PIKA, 3, GRID)
        self.assertEqual(pika, [1.0, 0.98, 0.9, 0.7])
        self.assertEqual(sigma, 0.7)
        self.assertEqual(flash, [0.7, 0.6969, 0.4688, 0.0])

    def test_flash_grid_points_at_or_above_sigma_are_dropped(self):
        _, flash, _ = NS["split_handoff_sigmas"](PIKA, 1, GRID)
        self.assertEqual(flash[0], 0.98)
        self.assertTrue(all(g < 0.98 for g in flash[1:]))
        self.assertEqual(flash[-1], 0.0)

    def test_a_grid_point_equal_to_sigma_is_not_repeated(self):
        _, flash, _ = NS["split_handoff_sigmas"]([1.0, 0.9231, 0.0], 1, GRID)
        self.assertEqual(flash, [0.9231, 0.8349, 0.6969, 0.4688, 0.0])

    def test_clamp(self):
        clamp = NS["clamp_handoff_step"]
        self.assertEqual(clamp(3, 6), 3)
        self.assertEqual(clamp(7, 6), 5)
        self.assertEqual(clamp(9, 4), 3)
        self.assertEqual(clamp(0, 6), 1)
        with self.assertLogs("test_handoff", "INFO") as cm:
            clamp(7, 6)
        self.assertIn("clamped to 5", cm.output[0])


class Inputs(unittest.TestCase):
    def test_option_is_appended_last(self):
        tree = ast.parse(NODE_TEXT)
        modes = next(ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ACCEL_MODES" for t in n.targets))
        self.assertEqual(modes[:2], ["Turbo LoRA", "HyperFlow Flash (GGUF)"])
        self.assertEqual(modes[2], MODE)
        self.assertEqual(NS["HANDOFF_ACCEL_MODE"], MODE)

    def test_handoff_step_is_appended_after_the_flash_widgets(self):
        names = re.findall(r'^\s{16}"(\w+)": \(', NODE_TEXT[NODE_TEXT.index("def INPUT_TYPES"):], re.M)
        i = names.index("handoff_step")
        self.assertEqual(names[i - 1:i + 1], ["hyperflow_flash_lora_strength", "handoff_step"])
        self.assertEqual(names[-1], "clips_in")
        self.assertRegex(NODE_TEXT, r'"handoff_step": \(HANDOFF_STEP_OPTIONS, \{"default": HANDOFF_STEP_DEFAULT_LABEL')
        self.assertEqual(NS["HANDOFF_STEP_DEFAULT"], 3)
        self.assertEqual(NS["HANDOFF_STEP_OPTIONS"], ["1 \u00b7 Pika to 0.98", "2 \u00b7 Pika to 0.96", "3 \u00b7 Pika to 0.92", "4 \u00b7 Pika to 0.86", "5 \u00b7 Pika to 0.71"])
        self.assertEqual(NS["HANDOFF_STEP_DEFAULT_LABEL"], "3 \u00b7 Pika to 0.92")


class Parse(unittest.TestCase):
    def test_label_to_int(self):
        parse = NS["parse_handoff_step"]
        self.assertEqual([parse(o) for o in NS["HANDOFF_STEP_OPTIONS"]], [1, 2, 3, 4, 5])
        self.assertEqual(parse("garbage"), 3)


class Fingerprint(unittest.TestCase):
    def test_handoff_key_only_in_handoff_mode(self):
        tree = ast.parse(NODE_TEXT)
        literal = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                       and any(getattr(t, "id", "") == "p1_settings" for t in n.targets))
        self.assertNotIn("handoff", [k.value for k in literal.value.keys])
        i = NODE_TEXT.index("if accel_mode == HANDOFF_ACCEL_MODE:\n            # Only in hand-off mode")
        block = NODE_TEXT[i:i + 500]
        self.assertIn('p1_settings["handoff"] = [flash_file, flash_steps, flash_lora, flash_lora_strength, handoff_step]', block)

    def test_flash_fingerprint_is_still_flash_mode_only(self):
        self.assertIn("if accel_mode == FLASH_ACCEL_MODE:\n            # Only in Flash mode", NODE_TEXT)

    def test_fingerprint_differs_by_handoff_step(self):
        # the node builds the key from the clamped step, so two steps give two lists and therefore two cache keys
        import json
        a = json.dumps(["f", 6, "none", 1.0, 3])
        b = json.dumps(["f", 6, "none", 1.0, 4])
        self.assertNotEqual(a, b)
        self.assertIn('handoff_step = clamp_handoff_step(parse_handoff_step(kwargs.get("handoff_step")), int(pdd_nfe))', NODE_TEXT)
        self.assertIn("clamp_handoff_step(parse_handoff_step(kwargs.get(\"handoff_step\")), int(pdd_nfe))", NODE_TEXT)
        parse = NS["parse_handoff_step"]
        self.assertEqual(NS["clamp_handoff_step"](parse("5 \u00b7 Pika to 0.71"), 4), 3)


class Validation(unittest.TestCase):
    def test_validate_requires_turbo_lora_and_flash_file(self):
        i = NODE_TEXT.index("def VALIDATE_INPUTS")
        block = NODE_TEXT[i:NODE_TEXT.index("def IS_CHANGED")]
        self.assertIn("turbo_lora=None", block)
        self.assertIn('== HANDOFF_ACCEL_MODE and turbo_lora in ("", "none")', block)
        self.assertIn("in (FLASH_ACCEL_MODE, HANDOFF_ACCEL_MODE):\n            if hyperflow_flash_file", block)

    def test_extend_rejects_none_lora(self):
        self.assertIn('raise ValueError("Pika \\u2192 Flash hand-off needs a turbo_lora', NODE_TEXT)


class EngineWiring(unittest.TestCase):
    def test_pass1_branches_to_handoff_and_flash_is_unloaded_after(self):
        self.assertIn("sampled_p1 = self._sample_handoff(model, sigmas_p1, pos_p1, latent_p1, seed)", ENGINE_TEXT)
        self.assertIn("if (self.flash_mode or self.handoff_mode) and pass2_enabled:\n                self._unload_flash()", ENGINE_TEXT)

    def test_flash_segment_continues_without_noise_on_euler(self):
        self.assertIn('ff_chunk=False, sampler_name="euler", add_noise=False', ENGINE_TEXT)
        self.assertIn("Noise_RandomNoise(seed) if add_noise else Noise_EmptyNoise()", ENGINE_TEXT)

    def test_handoff_keeps_the_pika_pass1_sigmas(self):
        self.assertIn("if not self.handoff_mode:\n            self.pass1_sigmas = sigmas", ENGINE_TEXT)

    def test_log_line(self):
        self.assertIn("Pika -> Flash hand-off: Pika steps 1..%d (sigma 1.0->%.4f), Flash %d steps (%.4f->0) [%s]", ENGINE_TEXT)

    def test_no_arrow_in_log_messages(self):
        for text in (ENGINE_TEXT, NODE_TEXT):
            for m in re.finditer(r'_LOG\.\w+\(\s*"([^"]*)"', text):
                self.assertNotIn("\\u2192", m.group(1))
                self.assertNotIn("\u2192", m.group(1))


if __name__ == "__main__":
    unittest.main()
