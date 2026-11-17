import ast
import logging
import re
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pdd_pure_engine.py"
NODE = ROOT / "master_node.py"

PIKA = "h3\\pika_h3_flex_ref2va_comfy_bf16.safetensors"
DMAD = "h3\\minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors"
REMOVED_INPUTS = ["hyperflow_file", "hyperflow_curve_refit", "hyperflow_strength", "hyperflow_lora_mode",
                  "dual_sampling", "dual_split_step", "pass1_early_stop", "pass1_steps"]

# Every input name up to sparse_sink, in order, with the removed HyperFlow 8-step inputs gone.
KEPT_INPUT_NAMES = [
    'model', 'clip', 'vae', 'audio_vae', 'pdd_file', 'upscaler_model', 'run_mode', 'pass1_resolution', 'pass2_resolution',
    'pass2_denoise', 'pdd_nfe', 'context_length', 'audio_context_length', 'identity_continuity', 'smart_offload', 'clips_json',
    'refs_json', 'ref_image_1', 'ref_image_2', 'ref_image_3', 'ref_image_4', 'ref_image_5', 'ref_image_6', 'ref_image_7',
    'ref_image_8', 'ref_image_9', 'attention_backend', 'sla_enabled', 'sla_sparsity', 'pass2_chunk_frames', 'pass2_chunk_overlap',
    'async_decode', 'sparse_method', 'sparse_tau', 'accel_mode', 'turbo_lora', 'turbo_lora_strength', 'turbo_sampler',
    'turbo_scheduler', 'pass2_lora', 'pass2_lora_strength', 'pass2_lora_mode', 'pass2_steps', 'semantic_bridge',
    'semantic_bridge_alpha', 'semantic_bridge_match', 'clip_prompt_1', 'clip_prompt_2', 'clip_prompt_3', 'clip_prompt_4',
    'rewrite_mode', 'rewrite_task', 'rewrite_writer_model', 'rewrite_caption_length', 'rewrite_greedy', 'rewrite_temperature',
    'rewrite_max_new_tokens', 'rewrite_seed', 'rewrite_thinking', 'rewrite_reasoning_budget', 'rewrite_reasoning_budget_message',
    'rewrite_parallel', 'rewrite_system_prompt', 'rewrite_system_prompt_in', 'ref_video_1', 'ref_video_2', 'ref_video_3',
    'ref_video_audio_1', 'ref_video_audio_2', 'ref_video_audio_3', 'ref_audio_1', 'ref_audio_2', 'ref_audio_3',
    'rewrite_previous_clips',
    'audio_refine_steps', 'audio_refine_denoise', 'audio_refine_cache', 'final_decode', 'rewrite_story', 'auto_clips',
    'planner_refs', 'auto_clip_seconds', 'story_engine', 'e4_language', 'e4_score', 'e4_decision_budget', 'e4_picture_notes',
    'sparse_start_percent', 'sparse_dense_blocks', 'sparse_sink',
]

# p1_settings exactly as master_node builds it for a default (Pika) job.
P1_TODAY = {
    "p1_res": [1056, 608], "pdd_file": "", "pdd_nfe": "6", "accel_mode": "Turbo LoRA", "turbo_lora": PIKA,
    "turbo_lora_strength": 1.0, "sampler": "euler", "scheduler": "simple", "attention": "comfy kitchen attention",
    "sla": [True, 0.9], "sparse": ["sla", 1.3, -1.0, "auto", "auto"],
    "hyperflow": ["custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors", True, 1.0, "bypass"],
    "context": ["22", 0, True], "bridge": ["none", 0.12, "per_token"], "single_te": True, "refs": "digest",
}


def _functions(*names, **extra):
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(extra)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns


def _engine_methods(*names):
    """MasterEngine's own methods, lifted out of the class so they run without ComfyUI or torch."""
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MasterEngine")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    stub = ast.ClassDef(name="Engine", bases=[], keywords=[], body=methods, decorator_list=[])
    ns = dict(_FNS, _LOG=logging.getLogger("test_hyperflow_removed"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), str(ENGINE), "exec"), ns)
    return ns["Engine"]


def _node_namespace():
    """master_node's accel-mode constants and helpers, lifted out so they run without ComfyUI."""
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    names = ("ACCEL_MODES", "PIKA_LORA", "PASS2_LORA_DEFAULT")
    wanted = [n for n in tree.body
              if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets))
              or (isinstance(n, ast.FunctionDef) and n.name in ("is_legacy_accel_mode", "normalize_accel_mode", "migrate_legacy_accel"))]
    ns = {"_LOG": logging.getLogger("test_hyperflow_removed")}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(NODE), "exec"), ns)
    return ns


_NODE = _node_namespace()
_FNS = _functions("hyperflow_tail_sigmas")

HF_SAVED = {"pdd_nfe": "8", "turbo_lora": "none", "turbo_lora_strength": 1.0, "turbo_sampler": "res_multistep",
            "turbo_scheduler": "simple", "pass2_lora": "none", "pass2_lora_strength": 1.0,
            "pass2_lora_mode": "stack on engine LoRA", "pass2_steps": 0}
RECIPE = {"pdd_nfe": "6", "turbo_lora": PIKA, "turbo_lora_strength": 1.0, "turbo_sampler": "euler", "turbo_scheduler": "simple",
          "pass2_lora": DMAD, "pass2_lora_strength": 1.0, "pass2_lora_mode": "replace engine LoRA", "pass2_steps": 4}


class HyperFlowEightStepMigration(unittest.TestCase):
    MODES = ("HyperFlow 8-step", "HyperFlow 8-step \u00b7 dual sampling", "PDD 8-step")

    def test_modes_are_turbo_then_flash_only(self):
        self.assertEqual(_NODE["ACCEL_MODES"], ["Turbo LoRA", "HyperFlow Flash (GGUF)", "Pika \u2192 Flash hand-off"])

    def test_old_modes_load_and_run_as_pika_dmad(self):
        for mode in self.MODES:
            with self.assertLogs("test_hyperflow_removed", level="WARNING"):
                accel, out = _NODE["migrate_legacy_accel"](mode, dict(HF_SAVED))
            self.assertEqual(accel, "Turbo LoRA", mode)
            self.assertEqual(out, RECIPE, mode)

    def test_the_input_dict_is_not_mutated(self):
        saved = dict(HF_SAVED)
        with self.assertLogs("test_hyperflow_removed", level="WARNING"):
            _NODE["migrate_legacy_accel"]("HyperFlow 8-step", saved)
        self.assertEqual(saved, HF_SAVED)

    def test_only_unset_fields_are_filled(self):
        mine = dict(HF_SAVED, pass2_lora="h3\\mine.safetensors", pass2_lora_mode="stack on engine LoRA", pass2_steps=3)
        with self.assertLogs("test_hyperflow_removed", level="WARNING"):
            _, out = _NODE["migrate_legacy_accel"]("HyperFlow 8-step", mine)
        self.assertEqual({k: out[k] for k in mine if k.startswith("pass2")}, {k: mine[k] for k in mine if k.startswith("pass2")})
        self.assertEqual(out["turbo_lora"], PIKA)
        own_turbo = dict(HF_SAVED, turbo_lora="h3\\turbo.safetensors", pdd_nfe="4", turbo_sampler="dpmpp_2m", turbo_scheduler="beta")
        with self.assertLogs("test_hyperflow_removed", level="WARNING"):
            _, out = _NODE["migrate_legacy_accel"]("HyperFlow 8-step", own_turbo)
        self.assertEqual((out["turbo_lora"], out["pdd_nfe"], out["turbo_sampler"], out["turbo_scheduler"]),
                         ("h3\\turbo.safetensors", "4", "dpmpp_2m", "beta"))
        self.assertEqual(out["pass2_lora"], DMAD)

    def test_a_pass2_lora_with_no_steps_gets_the_standard_steps_only_when_it_was_unset(self):
        with self.assertLogs("test_hyperflow_removed", level="WARNING"):
            _, out = _NODE["migrate_legacy_accel"]("HyperFlow 8-step", dict(HF_SAVED, pass2_lora="h3\\mine.safetensors"))
        self.assertEqual(out["pass2_steps"], 0)

    def test_other_modes_pass_through_untouched(self):
        for mode in ("Turbo LoRA", "HyperFlow Flash (GGUF)"):
            settings = dict(HF_SAVED)
            accel, out = _NODE["migrate_legacy_accel"](mode, settings)
            self.assertEqual(accel, mode)
            self.assertIs(out, settings)
            self.assertEqual(_NODE["normalize_accel_mode"](mode), mode)

    def test_the_server_accepts_the_old_names_and_still_rejects_unknown_ones(self):
        text = NODE.read_text(encoding="utf-8")
        self.assertIn("not is_legacy_accel_mode(accel_mode)", text)
        self.assertTrue(_NODE["is_legacy_accel_mode"]("HyperFlow 8-step \u00b7 dual sampling"))
        self.assertFalse(_NODE["is_legacy_accel_mode"]("HyperFlow Flash (GGUF)"))
        self.assertFalse(_NODE["is_legacy_accel_mode"]("Turbo LoRA"))

    def test_extend_migrates_before_the_engine_is_built(self):
        text = NODE.read_text(encoding="utf-8")
        run = text[text.index("legacy_mode = is_legacy_accel_mode(accel_mode)"):]
        self.assertLess(run.index("migrate_legacy_accel("), run.index("engine = MasterEngine("))
        self.assertLess(run.index("kwargs.update("), run.index("engine.configure_pass2_lora("))


class StaleApiKeys(unittest.TestCase):
    """API-format jobs (scripts) may still send the removed keys: extend takes **kwargs and nothing reads them."""

    def test_no_code_path_reads_a_removed_input(self):
        text = NODE.read_text(encoding="utf-8")
        engine = ENGINE.read_text(encoding="utf-8")
        for name in REMOVED_INPUTS:
            self.assertNotIn(f'kwargs.get("{name}"', text, name)
            self.assertNotIn(name, engine, name)
        self.assertIn("**kwargs", text[text.index("def extend("):text.index("def extend(") + 2500])

    def test_validation_does_not_name_a_removed_input(self):
        text = NODE.read_text(encoding="utf-8")
        sig = text[text.index("def VALIDATE_INPUTS"):text.index("def VALIDATE_INPUTS") + 400]
        self.assertNotIn("hyperflow_file", sig)
        self.assertNotIn("_hyperflow_files", text)

    def test_removed_inputs_are_not_declared(self):
        text = NODE.read_text(encoding="utf-8")
        declared = re.findall(r'^\s{16}"(\w+)": \(', text[text.index("def INPUT_TYPES"):], re.M)
        for name in REMOVED_INPUTS:
            self.assertNotIn(name, declared, name)


class Defaults(unittest.TestCase):
    text = NODE.read_text(encoding="utf-8")

    def test_the_defaults_are_pika_dmad(self):
        self.assertRegex(self.text, r'"accel_mode": \(ACCEL_MODES, \{"default": "Turbo LoRA"')
        self.assertRegex(self.text, r'"pdd_nfe": \(\[[^\]]*\], \{"default": "6"')
        self.assertRegex(self.text, r'"turbo_lora": \(\["none"\] \+ folder_paths.get_filename_list\("loras"\), \{"default": PIKA_LORA if PIKA_LORA in')
        self.assertRegex(self.text, r'"turbo_sampler": \(comfy.samplers.KSampler.SAMPLERS, \{"default": "euler"')
        self.assertRegex(self.text, r'"turbo_scheduler": \(comfy.samplers.KSampler.SCHEDULERS, \{"default": "simple"')
        self.assertRegex(self.text, r'"pass2_lora": \(\["none"\] \+ folder_paths.get_filename_list\("loras"\), \{"default": PASS2_LORA_DEFAULT if')
        self.assertRegex(self.text, r'"pass2_lora_mode": \(\["stack on engine LoRA", "replace engine LoRA"\], \{"default": "replace engine LoRA"')
        self.assertRegex(self.text, r'"pass2_steps": \("INT", \{"default": 4,')
        self.assertEqual(_NODE["PIKA_LORA"], PIKA)
        self.assertEqual(_NODE["PASS2_LORA_DEFAULT"], DMAD)

    def test_the_extend_signature_defaults_are_pika_dmad(self):
        sig = self.text[self.text.index("    def extend("):self.text.index("        clips_in=None,\n        **kwargs,")]
        for want in ('pdd_nfe="6"', 'accel_mode="Turbo LoRA"', "turbo_lora=PIKA_LORA", 'turbo_sampler="euler"', 'turbo_scheduler="simple"'):
            self.assertIn(want, sig)

    def test_the_input_order_is_the_old_order_minus_the_removed_inputs(self):
        names = re.findall(r'^\s{16}"(\w+)": \(', self.text[self.text.index("def INPUT_TYPES"):], re.M)
        self.assertEqual(names[:len(KEPT_INPUT_NAMES)], KEPT_INPUT_NAMES)
        self.assertEqual(names[len(KEPT_INPUT_NAMES):], ["pass2_chunk_when", "pass2_sla_sparsity", "pass2_scheduler", "pass2_audio", "hyperflow_flash_file", "hyperflow_flash_steps", "hyperflow_flash_lora", "hyperflow_flash_lora_strength", "handoff_step", "clips_in"])

    def test_the_pass1_fingerprint_keeps_its_keys(self):
        tree = ast.parse(self.text)
        literal = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                       and any(getattr(t, "id", "") == "p1_settings" for t in n.targets))
        self.assertEqual([k.value for k in literal.value.keys], list(P1_TODAY))


class FlashIsUnaffected(unittest.TestCase):
    def test_flash_keeps_its_mode_inputs_and_engine_support(self):
        text = NODE.read_text(encoding="utf-8")
        engine = ENGINE.read_text(encoding="utf-8")
        for name in ("hyperflow_flash_file", "hyperflow_flash_steps"):
            self.assertIn(f'"{name}": (', text)
        self.assertIn("def _prepare_flash_base", engine)
        self.assertIn("def hyperflow_tail_sigmas", engine)

    def test_flash_pass2_tail_is_still_cut_from_its_own_grid(self):
        grid = [1.0, 0.9939, 0.9843, 0.9661, 0.9231, 0.8349, 0.6969, 0.4688, 0.0]
        self.assertEqual(_FNS["hyperflow_tail_sigmas"](grid, 0.25), grid[-3:])
        self.assertEqual(_FNS["hyperflow_tail_sigmas"](grid, 1.0), grid)

    def test_a_flash_job_is_not_migrated(self):
        settings = dict(HF_SAVED)
        accel, out = _NODE["migrate_legacy_accel"]("HyperFlow Flash (GGUF)", settings)
        self.assertEqual((accel, out), ("HyperFlow Flash (GGUF)", settings))


class Pass2Chunking(unittest.TestCase):
    def _chunk_engine(self, frames, when, class_registry=True):
        Engine = _engine_methods("_pass2_chunking_active")
        e = Engine()
        e.pass2_chunk_frames = frames
        e.pass2_chunk_when = when
        mapping = {"MMH3SplitUpscale": object(), "MMH3TemporalSplitParamsV10": object()}
        Engine._pass2_chunking_active.__globals__["nodes"] = types.SimpleNamespace(NODE_CLASS_MAPPINGS=mapping)
        Engine._pass2_chunking_active.__globals__["PASS2_CHUNK_MIN_SHORT_SIDE"] = 800
        return e

    def test_above_720p_does_not_window_720p_class_output(self):
        e = self._chunk_engine(124, "above 720p")
        for w, h in ((1280, 736), (1280, 720), (1344, 768), (720, 1280)):
            with self.assertLogs("test_hyperflow_removed", level="INFO") as cm:
                self.assertFalse(e._pass2_chunking_active(361, w, h))
            self.assertIn("Pass 2: whole clip (pass2_chunk_when=above 720p, short side", cm.output[0])

    def test_above_720p_windows_1080p_class_output(self):
        e = self._chunk_engine(124, "above 720p")
        self.assertTrue(e._pass2_chunking_active(361, 1920, 1088))
        self.assertTrue(e._pass2_chunking_active(361, 1080, 1920))

    def test_always_windows_720p(self):
        self.assertTrue(self._chunk_engine(124, "always")._pass2_chunking_active(361, 1280, 736))

    def test_zero_chunk_frames_never_windows(self):
        for when in ("above 720p", "always"):
            self.assertFalse(self._chunk_engine(0, when)._pass2_chunking_active(361, 1920, 1088))


if __name__ == "__main__":
    unittest.main()
