import ast
import json
import logging
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pdd_pure_engine.py"
NODE = ROOT / "master_node.py"
ENGINE_TEXT = ENGINE.read_text(encoding="utf-8")
NODE_TEXT = NODE.read_text(encoding="utf-8")
CFG_MODE = "Base + CFG (undistilled)"
HANDOFF_MODE = "CFG → Pika hand-off"
PIKA = [1.0, 0.98, 0.9, 0.7, 0.4, 0.15, 0.0]
CFG24 = [1.0 - i / 24 for i in range(25)]


def _ns():
    tree = ast.parse(ENGINE_TEXT)
    consts = ("CFG_ACCEL_MODE", "CFG_HANDOFF_ACCEL_MODE", "CFG_STEPS_DEFAULT", "PASS1_CFG_DEFAULT", "CFG_HANDOFF_STEPS_DEFAULT")
    funcs = ("is_cfg_accel_mode", "clamp_cfg_handoff_steps", "cfg_pass1_settings", "split_handoff_sigmas")
    wanted = [n for n in tree.body
              if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in consts for t in n.targets))
              or (isinstance(n, ast.FunctionDef) and n.name in funcs)]
    ns = {"_LOG": logging.getLogger("test_cfg_pass1"), "re": re}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns


NS = _ns()


def _input_types():
    i = NODE_TEXT.index("def INPUT_TYPES")
    return re.findall(r'^\s{16}"(\w+)": \((.*)$', NODE_TEXT[i:], re.M)


class Modes(unittest.TestCase):
    def test_values_and_order(self):
        tree = ast.parse(NODE_TEXT)
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", "") == "ACCEL_MODES" for t in n.targets))
        self.assertEqual(ast.literal_eval(node.value),
                         ["Turbo LoRA", "HyperFlow Flash (GGUF)", "Pika \u2192 Flash hand-off", CFG_MODE, HANDOFF_MODE])
        self.assertEqual(NS["CFG_ACCEL_MODE"], CFG_MODE)
        self.assertEqual(NS["CFG_HANDOFF_ACCEL_MODE"], HANDOFF_MODE)

    def test_is_cfg_mode(self):
        f = NS["is_cfg_accel_mode"]
        self.assertTrue(f(CFG_MODE) and f(HANDOFF_MODE))
        self.assertFalse(f("Turbo LoRA") or f("Pika → Flash hand-off") or f("HyperFlow Flash (GGUF)"))


class Inputs(unittest.TestCase):
    def test_appended_after_handoff_step_before_clips_in(self):
        names = [n for n, _ in _input_types()]
        self.assertEqual(names[-6:], ["handoff_step", "pass1_cfg", "cfg_steps", "negative_prompt", "cfg_handoff_steps", "clips_in"])

    def test_defaults(self):
        self.assertEqual(NS["PASS1_CFG_DEFAULT"], 4.5)
        self.assertEqual(NS["CFG_STEPS_DEFAULT"], 24)
        self.assertEqual(NS["CFG_HANDOFF_STEPS_DEFAULT"], 8)
        decl = dict(_input_types())
        self.assertIn('"FLOAT", {"default": PASS1_CFG_DEFAULT, "min": 1.0', decl["pass1_cfg"])
        self.assertIn('"INT", {"default": CFG_STEPS_DEFAULT', decl["cfg_steps"])
        self.assertIn('"STRING", {"multiline": True, "default": ""', decl["negative_prompt"])
        self.assertIn('"INT", {"default": CFG_HANDOFF_STEPS_DEFAULT', decl["cfg_handoff_steps"])

    def test_default_accel_mode_is_unchanged(self):
        self.assertIn('"accel_mode": (ACCEL_MODES, {"default": "Turbo LoRA"', NODE_TEXT)

    def test_hand_off_validation_needs_a_turbo_lora(self):
        block = NODE_TEXT[NODE_TEXT.index("def VALIDATE_INPUTS"):NODE_TEXT.index("def IS_CHANGED")]
        self.assertIn('== CFG_HANDOFF_ACCEL_MODE and turbo_lora in ("", "none")', block)


class Split(unittest.TestCase):
    def test_clamp(self):
        c = NS["clamp_cfg_handoff_steps"]
        self.assertEqual(c(8, 24), 8)
        self.assertEqual(c(0, 24), 1)
        self.assertEqual(c(99, 24), 23)
        self.assertEqual(c(5, 2), 1)

    def test_cfg_segment_then_pika_grid_below_the_hand_off_sigma(self):
        cfg_seg, pika_seg, sigma = NS["split_handoff_sigmas"](CFG24, 8, PIKA)
        self.assertEqual(len(cfg_seg), 9)
        self.assertEqual(cfg_seg[0], 1.0)
        self.assertAlmostEqual(sigma, 1.0 - 8 / 24)
        self.assertEqual(cfg_seg[-1], sigma)
        self.assertEqual(pika_seg, [sigma, 0.4, 0.15, 0.0])

    def test_early_hand_off_leaves_most_of_pika(self):
        _, pika_seg, sigma = NS["split_handoff_sigmas"](CFG24, 1, PIKA)
        self.assertAlmostEqual(sigma, 1.0 - 1 / 24)
        self.assertEqual(pika_seg, [sigma] + PIKA[2:])

    def test_late_hand_off_leaves_one_pika_step(self):
        _, pika_seg, sigma = NS["split_handoff_sigmas"](CFG24, 23, PIKA)
        self.assertEqual(pika_seg, [sigma, 0.0])


class Fingerprint(unittest.TestCase):
    base = (CFG_MODE, 4.5, 24, "blurry", 8)

    def test_none_outside_cfg_modes(self):
        f = NS["cfg_pass1_settings"]
        self.assertIsNone(f("Turbo LoRA", 4.5, 24, "blurry", 8))
        self.assertIsNone(f("Pika → Flash hand-off", 4.5, 24, "blurry", 8))

    def test_each_input_changes_it(self):
        f = NS["cfg_pass1_settings"]
        key = lambda *a: json.dumps(f(*a))
        ref = key(*self.base)
        self.assertNotEqual(ref, key(CFG_MODE, 1.0, 24, "blurry", 8))
        self.assertNotEqual(ref, key(CFG_MODE, 4.5, 30, "blurry", 8))
        self.assertNotEqual(ref, key(CFG_MODE, 4.5, 24, "", 8))
        self.assertNotEqual(ref, key(CFG_MODE, 4.5, 24, "blurry", 8) + "x")
        self.assertEqual(ref, key(CFG_MODE, 4.5, 24, "blurry", 3))

    def test_hand_off_step_only_matters_in_hand_off_mode(self):
        f = NS["cfg_pass1_settings"]
        self.assertNotEqual(f(HANDOFF_MODE, 4.5, 24, "", 8), f(HANDOFF_MODE, 4.5, 24, "", 9))
        self.assertEqual(f(CFG_MODE, 4.5, 24, "", 8), f(CFG_MODE, 4.5, 24, "", 9))

    def test_cfg_one_never_shares_a_key_with_turbo(self):
        # accel_mode is a base key of p1_settings, so a CFG-1 render of the new mode differs from Turbo LoRA's
        self.assertIn('"accel_mode": str(accel_mode)', NODE_TEXT)
        self.assertIsNotNone(NS["cfg_pass1_settings"](CFG_MODE, 1.0, 24, "", 8))

    def test_node_adds_the_cfg_key_from_the_helper(self):
        self.assertIn('p1_settings["cfg"] = cfg_entry', NODE_TEXT)
        self.assertIn("cfg_pass1_settings(accel_mode, pass1_cfg, cfg_steps, negative_prompt, cfg_handoff_steps)", NODE_TEXT)


class EngineWiring(unittest.TestCase):
    def test_cfg_one_is_the_old_path(self):
        self.assertIn("self.cfg_active = (self.cfg_mode or self.cfg_handoff_mode) and self.pass1_cfg > 1.0 + 1e-6", ENGINE_TEXT)
        self.assertIn("if not cached and self.cfg_active:", ENGINE_TEXT)
        self.assertIn("guider.set_conds(positive, positive)\n            guider.set_cfg(1.0)", ENGINE_TEXT)
        self.assertIn("use_cfg = negative is not None and float(cfg) > 1.0 + 1e-6", ENGINE_TEXT)

    def test_negative_carries_the_references_and_motion_context(self):
        i = ENGINE_TEXT.index("prompt=self.negative_prompt,")
        block = ENGINE_TEXT[i:i + 400]
        for arg in ("ref_images=ref_images", "ref_videos=ref_videos or None", "ref_audios=ref_audios or None", "width=w1"):
            self.assertIn(arg, block)
        self.assertIn("neg_p1 = self.motion_ram.apply(", ENGINE_TEXT)

    def test_hand_off_reuses_the_split_and_continues_without_noise(self):
        i = ENGINE_TEXT.index("def _sample_cfg_handoff")
        block = ENGINE_TEXT[i:ENGINE_TEXT.index("def _sample_euler")]
        self.assertIn("split_handoff_sigmas(cfg_sigmas, k, pika_grid)", block)
        self.assertIn("add_noise=False", block)
        self.assertIn("cfg=self.pass1_cfg, negative=negative", block)
        self.assertIn("self._sample_cfg_handoff(self.initialize_cfg_model(), model, sigmas_p1, pos_p1, neg_p1,", ENGINE_TEXT)

    def test_pure_cfg_mode_has_no_turbo_lora_and_keeps_the_pass2_schedule(self):
        self.assertIn("if self.cfg_mode:\n            accelerated, sigmas_p1 = self._prepare_cfg_base(shifted_model)", ENGINE_TEXT)
        self.assertIn("or self.cfg_mode or self.cfg_handoff_mode)", ENGINE_TEXT)

    def test_no_arrow_in_log_messages(self):
        for m in re.finditer(r'_LOG\.\w+\(\s*"([^"]*)"', ENGINE_TEXT + NODE_TEXT):
            self.assertNotIn("→", m.group(1))


if __name__ == "__main__":
    unittest.main()
