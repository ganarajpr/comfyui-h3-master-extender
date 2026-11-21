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
LEGACY_INPUTS = ["hyperflow_file", "hyperflow_curve_refit", "hyperflow_strength", "hyperflow_lora_mode",
                 "dual_sampling", "dual_split_step", "pass1_early_stop", "pass1_steps"]
HYPERFLOW_FILE = "custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors"

# Every input name up to sparse_sink, in order. The legacy HyperFlow 8-step inputs are appended later (after tst_tau),
# so every workflow saved since 2026-11-17 keeps its widget order.
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
    ns = dict(_FNS, _LOG=logging.getLogger("test_hyperflow_legacy"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), str(ENGINE), "exec"), ns)
    return ns["Engine"]


def _node_namespace():
    """master_node's accel-mode constants and helpers, lifted out so they run without ComfyUI."""
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    names = ("ACCEL_MODES", "PIKA_LORA", "PASS2_LORA_DEFAULT", "HYPERFLOW_DEFAULT_FILE", "DUAL_SAMPLING_ACCEL_MODE")
    wanted = [n for n in tree.body
              if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets))
              or (isinstance(n, ast.FunctionDef) and n.name in ("is_legacy_accel_mode", "normalize_accel_mode", "is_hyperflow_accel_mode",
                                                                 "resolve_accel_mode", "_legacy_int"))]
    ns = {"_LOG": logging.getLogger("test_hyperflow_legacy"), "HYPERFLOW_ACCEL_MODE": "HyperFlow 8-step"}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(NODE), "exec"), ns)
    return ns


_NODE = _node_namespace()
_FNS = _functions("hyperflow_tail_sigmas", "dual_split_index", "dual_pass1_sigmas", "dual_pass2_sigmas", "with_dual_p1_settings",
                  "with_pass1_steps_p1_settings", "pass1_steps_sigmas", "resolve_sparse_settings", "pass_sampler_name",
                  SPARSE_SINK_CHOICES=("exact_kv_and_rows", "exact_kv", "off"))
GRID = [1.0, 0.9939, 0.9843, 0.9661, 0.9231, 0.8349, 0.6969, 0.4688, 0.0]


class LegacyModes(unittest.TestCase):
    def test_the_two_hyperflow_modes_are_appended_last_and_the_default_is_turbo(self):
        self.assertEqual(_NODE["ACCEL_MODES"], ["Turbo LoRA", "HyperFlow Flash (GGUF)", "Pika \u2192 Flash hand-off", "Base + CFG (undistilled)",
                                                "CFG \u2192 Pika hand-off", "HyperFlow 8-step", "HyperFlow 8-step \u00b7 dual sampling", "pi-H3 (FL2VA adapter)"])
        self.assertEqual(_NODE["DUAL_SAMPLING_ACCEL_MODE"], "HyperFlow 8-step \u00b7 dual sampling")
        self.assertRegex(NODE.read_text(encoding="utf-8"), r'"accel_mode": \(ACCEL_MODES, \{"default": "Turbo LoRA"')

    def test_pdd_runs_hyperflow_8_step_and_nothing_else_is_rewritten(self):
        with self.assertLogs("test_hyperflow_legacy", level="WARNING"):
            self.assertEqual(_NODE["normalize_accel_mode"]("PDD 8-step"), "HyperFlow 8-step")
        for mode in ("HyperFlow 8-step", "HyperFlow 8-step \u00b7 dual sampling", "Turbo LoRA", "HyperFlow Flash (GGUF)"):
            self.assertEqual(_NODE["normalize_accel_mode"](mode), mode)
        self.assertTrue(_NODE["is_legacy_accel_mode"]("PDD 8-step"))
        for mode in ("HyperFlow 8-step", "HyperFlow 8-step \u00b7 dual sampling", "Turbo LoRA", "HyperFlow Flash (GGUF)"):
            self.assertFalse(_NODE["is_legacy_accel_mode"](mode), mode)

    def test_the_migration_to_pika_is_gone(self):
        self.assertNotIn("migrate_legacy_accel", NODE.read_text(encoding="utf-8"))

    def test_resolve_accel_mode(self):
        resolve = _NODE["resolve_accel_mode"]
        self.assertEqual(resolve("HyperFlow 8-step \u00b7 dual sampling", 124), ("HyperFlow 8-step", True, 0))
        self.assertEqual(resolve("HyperFlow 8-step", 124), ("HyperFlow 8-step", False, 124))
        self.assertEqual(resolve("Turbo LoRA", 96), ("Turbo LoRA", False, 96))
        self.assertEqual(resolve("HyperFlow Flash (GGUF)", 124), ("HyperFlow Flash (GGUF)", False, 124))
        with self.assertLogs("test_hyperflow_legacy", level="WARNING"):
            self.assertEqual(resolve("PDD 8-step", 124), ("HyperFlow 8-step", False, 124))

    def test_is_hyperflow_accel_mode(self):
        is_hf = _NODE["is_hyperflow_accel_mode"]
        self.assertTrue(is_hf("HyperFlow 8-step") and is_hf("HyperFlow 8-step \u00b7 dual sampling") and is_hf("PDD 8-step"))
        self.assertFalse(is_hf("Turbo LoRA") or is_hf("HyperFlow Flash (GGUF)") or is_hf("Pika \u2192 Flash hand-off") or is_hf(None))

    def test_extend_resolves_the_mode_before_the_engine_is_built(self):
        text = NODE.read_text(encoding="utf-8")
        run = text[text.index("    def extend("):]
        self.assertLess(run.index("resolve_accel_mode(accel_mode, pass2_chunk_frames)"), run.index("engine = MasterEngine("))
        self.assertLess(run.index("engine.configure_pass2_lora("), run.index("engine.configure_dual_sampling("))
        self.assertLess(run.index("engine.configure_dual_sampling("), run.index("engine.configure_pass1_steps("))

    def test_one_info_line_says_the_legacy_engine_renders_as_saved(self):
        text = NODE.read_text(encoding="utf-8")
        i = text.index("if accel_mode == HYPERFLOW_ACCEL_MODE:\n            _LOG.info(")
        self.assertIn("Legacy HyperFlow 8-step engine (retired 2026-11-17) is rendering this workflow as saved", text[i:i + 300])


class LegacyInputs(unittest.TestCase):
    text = NODE.read_text(encoding="utf-8")

    def test_the_eight_inputs_are_appended_after_tst_tau_before_clips_in(self):
        names = re.findall(r'^\s{16}"(\w+)": \(', self.text[self.text.index("def INPUT_TYPES"):], re.M)
        self.assertEqual(names[-11:], ["tst_enabled", "tst_tau"] + LEGACY_INPUTS + ["clips_in"][:1])
        for name in LEGACY_INPUTS:
            self.assertEqual(names.count(name), 1, name)

    def test_definitions_are_the_old_ones(self):
        t = self.text
        self.assertRegex(t, r'"hyperflow_file": \(_hyperflow_files\(\), \{"default": HYPERFLOW_DEFAULT_FILE')
        self.assertRegex(t, r'"hyperflow_curve_refit": \("BOOLEAN", \{"default": True')
        self.assertRegex(t, r'"hyperflow_strength": \("FLOAT", \{"default": 1\.0, "min": 0\.0, "max": 2\.0, "step": 0\.05')
        self.assertRegex(t, r'"hyperflow_lora_mode": \(\["bypass", "merge"\], \{"default": "bypass"')
        self.assertRegex(t, r'"dual_sampling": \("BOOLEAN", \{"default": False')
        self.assertRegex(t, r'"dual_split_step": \("INT", \{"default": 2, "min": 1, "max": 7, "step": 1')
        self.assertRegex(t, r'"pass1_early_stop": \("BOOLEAN", \{"default": False')
        self.assertRegex(t, r'"pass1_steps": \("INT", \{"default": 0, "min": 0, "max": 8, "step": 1')
        self.assertEqual(_NODE["HYPERFLOW_DEFAULT_FILE"], HYPERFLOW_FILE)

    def test_tooltips_say_they_are_legacy(self):
        for name in LEGACY_INPUTS:
            line = next(l for l in self.text.splitlines() if l.lstrip().startswith(f'"{name}": ('))
            self.assertIn("workflows saved before 2026-11-17", line, name)

    def test_validation_checks_the_file_only_for_the_hyperflow_modes(self):
        sig = self.text[self.text.index("def VALIDATE_INPUTS"):self.text.index("def VALIDATE_INPUTS") + 200]
        self.assertIn("hyperflow_file=None", sig)
        body = self.text[self.text.index("def VALIDATE_INPUTS"):self.text.index("def IS_CHANGED")]
        i = body.index("if not is_hyperflow_accel_mode(accel_mode) or not hyperflow_file:")
        self.assertIn("return True", body[i:i + 120])
        self.assertIn("is not installed in models/hyperflow", body[i:])
        self.assertIn("legacy", body[i:])
        self.assertIn("switch accel_mode to Turbo LoRA", body[i:])

    def test_pdd_is_still_accepted_by_validation(self):
        self.assertIn("not is_legacy_accel_mode(accel_mode)", self.text)


class Fingerprint(unittest.TestCase):
    """Pass-1 caches written before 2026-11-17 (HyperFlow and Turbo) and since keep their fingerprints."""
    text = NODE.read_text(encoding="utf-8")

    def _literal(self):
        tree = ast.parse(self.text)
        return next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                    and any(getattr(t, "id", "") == "p1_settings" for t in n.targets)).value

    def _hyperflow_entry(self, kwargs):
        lit = self._literal()
        expr = next(v for k, v in zip(lit.keys, lit.values) if k.value == "hyperflow")
        src = ast.get_source_segment(self.text, expr)
        return eval(src, {"kwargs": kwargs, "HYPERFLOW_DEFAULT_FILE": HYPERFLOW_FILE})

    def test_keys_are_unchanged(self):
        self.assertEqual([k.value for k in self._literal().keys], list(P1_TODAY))

    def test_defaults_give_the_constant_every_turbo_and_flash_cache_since_1117_was_keyed_on(self):
        self.assertEqual(self._hyperflow_entry({}), P1_TODAY["hyperflow"])
        self.assertEqual(self._hyperflow_entry({"hyperflow_file": ""}), P1_TODAY["hyperflow"])

    def test_saved_values_are_in_it_like_before_1117(self):
        entry = self._hyperflow_entry({"hyperflow_file": "x.safetensors", "hyperflow_curve_refit": False,
                                       "hyperflow_strength": 0.8, "hyperflow_lora_mode": "merge"})
        self.assertEqual(entry, ["x.safetensors", False, 0.8, "merge"])

    def test_dual_and_pass1_steps_keys_only_when_they_change_pass_1(self):
        base = dict(P1_TODAY)
        dual, steps = _FNS["with_dual_p1_settings"], _FNS["with_pass1_steps_p1_settings"]
        self.assertIs(dual(base, False, 2, True), base)
        self.assertIs(dual(base, True, 2, False), base)
        self.assertEqual(dual(base, True, 4, True)["dual"], [4, True])
        self.assertNotIn("dual", base)
        self.assertIs(steps(base, 0), base)
        self.assertEqual(steps(base, 6)["pass1_steps"], 6)

    def test_they_are_applied_to_the_fingerprint_after_the_literal(self):
        i = self.text.index("p1_settings = with_dual_p1_settings(")
        self.assertIn("dual_active and pass2_enabled", self.text[i:i + 200])
        self.assertIn("p1_settings = with_pass1_steps_p1_settings(p1_settings, engine.pass1_steps)", self.text[i:i + 400])

    def test_dual_is_active_only_in_hyperflow_8_step(self):
        self.assertIn("dual_active = dual_requested and accel_mode == HYPERFLOW_ACCEL_MODE", self.text)


class LegacyEngine(unittest.TestCase):
    engine_text = ENGINE.read_text(encoding="utf-8")

    def make(self, mode):
        Engine = _engine_methods("configure_dual_sampling", "configure_pass1_steps", "configure_pass2_lora", "_pass2_sigmas", "_pass2_lora_active")
        e = Engine()
        e.accel_mode = mode
        e.hyperflow_mode = mode == "HyperFlow 8-step"
        e.flash_mode = mode == "HyperFlow Flash (GGUF)"
        e.turbo_mode = not e.hyperflow_mode and not e.flash_mode
        e.hyperflow_pruned_refit = False
        e.pass2_lora, e.pass2_model, e.pass2_sla_model = None, None, None
        e.pass2_lora_strength, e.pass2_lora_mode, e.pass2_steps = 1.0, "stack", 0
        e.turbo_steps = "8"
        e.pass1_sigmas = list(GRID)
        e.dual_active, e.dual_split_step, e.pass1_early_stop, e.pass1_steps = False, 2, False, 0
        return e

    def test_hyperflow_mode_is_exactly_the_8_step_mode(self):
        self.assertIn("self.hyperflow_mode = self.accel_mode == HYPERFLOW_ACCEL_MODE", self.engine_text)
        self.assertIn('HYPERFLOW_ACCEL_MODE = "HyperFlow 8-step"', self.engine_text)

    def test_dual_sampling_and_pass1_steps_work_only_in_hyperflow_8_step(self):
        e = self.make("HyperFlow 8-step")
        e.configure_dual_sampling(True, 3, True)
        self.assertTrue(e.dual_active)
        self.assertTrue(e.pass1_early_stop)
        self.assertEqual(e.dual_split_step, 3)
        self.assertEqual(list(e._pass2_sigmas(None, 0.25)), GRID[3:])
        e = self.make("HyperFlow 8-step")
        e.configure_dual_sampling(False, 2, False)
        e.configure_pass1_steps(5)
        self.assertEqual(e.pass1_steps, 5)
        for mode in ("Turbo LoRA", "HyperFlow Flash (GGUF)", "Pika \u2192 Flash hand-off", "Base + CFG (undistilled)", "CFG \u2192 Pika hand-off"):
            e = self.make(mode)
            with self.assertLogs("test_hyperflow_legacy", level="WARNING"):
                e.configure_dual_sampling(True, 2, True)
            self.assertFalse(e.dual_active, mode)
            self.assertFalse(e.pass1_early_stop, mode)
            with self.assertLogs("test_hyperflow_legacy", level="WARNING"):
                e.configure_pass1_steps(4)
            self.assertEqual(e.pass1_steps, 0, mode)

    def test_dual_sampling_drops_the_pass2_lora_and_pass1_steps_yield_to_it(self):
        e = self.make("HyperFlow 8-step")
        e.configure_pass2_lora("dmad.safetensors", 1.0, "replace engine LoRA", 4)
        with self.assertLogs("test_hyperflow_legacy", level="WARNING"):
            e.configure_dual_sampling(True, 2, False)
        self.assertIsNone(e.pass2_lora)
        with self.assertLogs("test_hyperflow_legacy", level="WARNING"):
            e.configure_pass1_steps(4)
        self.assertEqual(e.pass1_steps, 0)

    def test_a_pass2_lora_forces_replace_mode_under_hyperflow_only(self):
        e = self.make("HyperFlow 8-step")
        with self.assertLogs("test_hyperflow_legacy", level="INFO"):
            e.configure_pass2_lora("dmad.safetensors", 1.0, "stack on engine LoRA", 4)
        self.assertEqual(e.pass2_lora_mode, "replace")
        e = self.make("Turbo LoRA")
        e.configure_pass2_lora("dmad.safetensors", 1.0, "stack on engine LoRA", 4)
        self.assertEqual(e.pass2_lora_mode, "stack")

    def test_pass2_without_a_refine_lora_is_cut_from_the_trained_grid(self):
        e = self.make("HyperFlow 8-step")
        with self.assertLogs("test_hyperflow_legacy", level="INFO"):
            self.assertEqual(list(e._pass2_sigmas(None, 0.25)), GRID[-3:])

    def test_the_engine_still_calls_apply_hyperflow_and_fails_loudly_without_the_pack(self):
        t = self.engine_text
        self.assertIn('nodes.NODE_CLASS_MAPPINGS.get("ApplyHyperFlowH3")', t)
        i = t.index('apply_cls = nodes.NODE_CLASS_MAPPINGS.get("ApplyHyperFlowH3")')
        err = t[i:i + 700]
        for needle in ("ComfyUI-HyperFlow-H3", "ApplyHyperFlowH3", "retired 2026-11-17", "switch accel_mode to Turbo LoRA", "Pika + DMAD"):
            self.assertIn(needle, err)
        self.assertIn("accelerated, sigmas_p1 = self._prepare_hyperflow_model(shifted_model)", t)

    def test_pass1_early_stop_and_x0_helpers_are_back(self):
        t = self.engine_text
        for needle in ("def capture_x0_callback", "def clean_latent_from_x0", "def dual_split_index", "def pass1_steps_sigmas",
                       "return_denoised=early_stop", "def configure_dual_sampling", "def configure_pass1_steps"):
            self.assertIn(needle, t)

    def test_the_sol_attn_recipe_is_back_for_hyperflow_only(self):
        resolve = _FNS["resolve_sparse_settings"]
        r = resolve(True, "sol-attn", -1.0, "auto", "auto")
        self.assertEqual((r["start"], r["dense_blocks"], r["sink"]), (0.16, "0,1", "off"))
        r = resolve(False, "sol-attn", -1.0, "auto", "auto")
        self.assertEqual((r["start"], r["dense_blocks"], r["sink"]), (0.2, "", "exact_kv_and_rows"))
        self.assertIn("self.hyperflow_mode, str(getattr(self, \"sparse_method\", \"sla\") or \"sla\")", self.engine_text)

    def test_hyperflow_samples_with_euler_and_the_tst_hook_is_shared_with_the_other_engines(self):
        self.assertEqual(_FNS["pass_sampler_name"](False, "res_multistep"), "euler")
        self.assertIn("accelerated = self._finish_model(accelerated, tst=True)", self.engine_text)


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

    def test_the_input_order_is_unchanged_up_to_the_legacy_block(self):
        names = re.findall(r'^\s{16}"(\w+)": \(', self.text[self.text.index("def INPUT_TYPES"):], re.M)
        self.assertEqual(names[:len(KEPT_INPUT_NAMES)], KEPT_INPUT_NAMES)
        self.assertEqual(names[len(KEPT_INPUT_NAMES):], ["pass2_chunk_when", "pass2_sla_sparsity", "pass2_scheduler", "pass2_audio", "hyperflow_flash_file", "hyperflow_flash_steps", "hyperflow_flash_lora", "hyperflow_flash_lora_strength", "handoff_step", "pass1_cfg", "cfg_steps", "negative_prompt", "cfg_handoff_steps", "tst_enabled", "tst_tau", *LEGACY_INPUTS, "clips_in"])


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

    def test_a_flash_job_is_not_a_hyperflow_8_step_job(self):
        self.assertEqual(_NODE["resolve_accel_mode"]("HyperFlow Flash (GGUF)", 124), ("HyperFlow Flash (GGUF)", False, 124))


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
            with self.assertLogs("test_hyperflow_legacy", level="INFO") as cm:
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
