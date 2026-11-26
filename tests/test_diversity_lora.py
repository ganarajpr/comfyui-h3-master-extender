import ast
import logging
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pdd_pure_engine.py"
NODE = ROOT / "master_node.py"
PRESETS = ROOT / "web" / "presets.js"
LORA = "h3\\rediverse_pika_h3_flex_ft_comfyui.safetensors"
PIKA_GRID = [1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0]


def _ns():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body
              if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "TURBO_ACCEL_MODE" for t in n.targets))
              or (isinstance(n, ast.FunctionDef) and n.name in ("clamp_diversity_steps", "split_diversity_sigmas", "diversity_p1_settings"))]
    ns = {"_LOG": logging.getLogger("test_diversity")}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns


NS = _ns()
NODE_TEXT = NODE.read_text(encoding="utf-8")
ENGINE_TEXT = ENGINE.read_text(encoding="utf-8")


class Split(unittest.TestCase):
    def test_one_step_matches_the_authors_recipe(self):
        div, turbo = NS["split_diversity_sigmas"](PIKA_GRID, 1)
        self.assertEqual(div, [1.0, 0.972973])
        self.assertEqual(turbo, [0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0])

    def test_segments_share_the_boundary_and_cover_the_schedule(self):
        for k in range(1, 6):
            div, turbo = NS["split_diversity_sigmas"](PIKA_GRID, k)
            self.assertEqual(div[-1], turbo[0])
            self.assertEqual(div + turbo[1:], PIKA_GRID)

    def test_clamp_keeps_one_plain_turbo_step(self):
        clamp = NS["clamp_diversity_steps"]
        self.assertEqual([clamp(0, 6), clamp(1, 6), clamp(5, 6), clamp(9, 6)], [1, 1, 5, 5])


class CacheKey(unittest.TestCase):
    def test_off_or_other_modes_add_no_entry(self):
        f = NS["diversity_p1_settings"]
        self.assertIsNone(f("Turbo LoRA", "none", 1.0, 1))
        self.assertIsNone(f("Turbo LoRA", "", 1.0, 1))
        self.assertIsNone(f("HyperFlow Flash (GGUF)", LORA, 1.0, 1))

    def test_entry_changes_with_lora_strength_and_steps(self):
        f = NS["diversity_p1_settings"]
        base = f("Turbo LoRA", LORA, 1.0, 1)
        self.assertEqual(base, [LORA, 1.0, 1])
        self.assertNotEqual(base, f("Turbo LoRA", LORA, 0.8, 1))
        self.assertNotEqual(base, f("Turbo LoRA", LORA, 1.0, 2))
        self.assertNotEqual(base, f("Turbo LoRA", "other.safetensors", 1.0, 1))

    def test_node_adds_the_key_only_when_active(self):
        i = NODE_TEXT.index("div_entry = diversity_p1_settings(")
        block = NODE_TEXT[i:i + 700]
        self.assertIn("if engine._diversity_active() else None", block)
        self.assertIn("if div_entry is not None:", block)
        self.assertIn('p1_settings["diversity"] = div_entry', block)


class Wiring(unittest.TestCase):
    def test_inputs_are_appended_before_the_time_gate_socket_and_clips_in(self):
        names = re.findall(r'^\s{16}"(\w+)": \(', NODE_TEXT[NODE_TEXT.index("def INPUT_TYPES"):], re.M)
        i = names.index("diversity_lora")
        self.assertEqual(names[i - 1], "pih3_refine_tail")
        self.assertEqual(names[i:i + 8], ["diversity_lora", "diversity_lora_strength", "diversity_steps", "cache_clips", "ref_image_size", "ref_codec_prune", "time_gate_json", "clips_in"])

    def test_defaults_are_off_at_the_node_level(self):
        self.assertRegex(NODE_TEXT, r'"diversity_lora": \(\["none"\] \+ folder_paths\.get_filename_list\("loras"\), \{"default": "none"')
        self.assertRegex(NODE_TEXT, r'"diversity_lora_strength": \("FLOAT", \{"default": 1\.0')
        self.assertRegex(NODE_TEXT, r'"diversity_steps": \("INT", \{"default": 1, "min": 1')

    def test_split_happens_in_pass_1_for_turbo_mode_only(self):
        self.assertIn("elif self._diversity_active() and not early_stop:", ENGINE_TEXT)
        i = ENGINE_TEXT.index("def configure_diversity")
        block = ENGINE_TEXT[i:ENGINE_TEXT.index("def _diversity_active")]
        self.assertIn("self.accel_mode != TURBO_ACCEL_MODE or self.turbo_lora is None", block)
        self.assertIn("ignored", block)

    def test_second_segment_adds_no_noise_with_the_same_seed(self):
        i = ENGINE_TEXT.index("def _sample_diversity")
        block = ENGINE_TEXT[i:ENGINE_TEXT.index("def configure_pass1_steps")]
        self.assertIn("add_noise=False", block)
        self.assertEqual(len(re.findall(r"_sample_euler\(", block)), 2)

    def test_pika_presets_set_it(self):
        text = PRESETS.read_text(encoding="utf-8")
        self.assertIn('const REDIVERSE = "h3\\\\rediverse_pika_h3_flex_ft_comfyui.safetensors"', text)
        self.assertEqual(text.count("diversity_lora: REDIVERSE"), 2)


if __name__ == "__main__":
    unittest.main()
