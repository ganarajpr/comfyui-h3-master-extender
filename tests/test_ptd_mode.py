"""PTD mode: hop selection, start-from-hop math, the hop sampler and the engine's mode wiring (CPU, no ComfyUI)."""

import ast
import importlib.util
import logging
import os
import re
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pdd_pure_engine.py"
NODE = ROOT / "master_node.py"

PACK_SCHEDULE = Path(os.environ.get(
    "PTD_PACK_DIR", str(Path.home() / "Projects/comfyui-minimaxh3-ptd/comfyui_node/ComfyUI-MiniMaxH3-PTD"))) / "schedule.py"


def _load_ptd_mode():
    spec = importlib.util.spec_from_file_location("ptd_mode_under_test", ROOT / "ptd_mode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ptd = _load_ptd_mode()

# The 6-hop file's grid, as the pack's schedule.hop_sigmas(6, 6, 12.0) produces it (checked against the pack below).
GRID6 = torch.tensor([0.999, 0.982623, 0.95904, 0.922154, 0.856286, 0.705176, 0.0], dtype=torch.float64)


class FakeUpstream:
    """ptd_comfy.sampling's audio clock and x0 -> velocity, copied only as far as the tests need them."""

    T_MAX = 0.999

    @classmethod
    def audio_sigma(cls, t):
        if t == cls.T_MAX:
            return cls.T_MAX
        return 0.25 * t * cls.T_MAX / (cls.T_MAX + (0.25 - 1.0) * t)

    @staticmethod
    def denoised_to_velocity(state, denoised, time):
        return (state.float() - denoised.float()) / time


class FakeController:
    def __init__(self, sigmas=GRID6):
        self.sigmas = sigmas.clone()
        self.hops = len(sigmas) - 1
        self.activated = []

    @contextmanager
    def activate(self, hop, sigma):
        if abs(float(sigma) - float(self.sigmas[hop])) > 2e-6:
            raise ValueError("wrong time for hop")
        self.activated.append((hop, float(sigma)))
        yield self


class FakeModel:
    """A KSamplerX0Inpaint stand-in whose 'network' always predicts the clean latent ``x0``."""

    def __init__(self, noise, latent, x0, shapes):
        self.noise, self.latent_image, self.x0 = noise, latent, x0
        self.inner_model = types.SimpleNamespace(inner_model=types.SimpleNamespace(latent_shapes=shapes))
        self.calls = []

    def __call__(self, x, sigma, **extra):
        self.calls.append((x.clone(), float(sigma[0]), dict(extra)))
        return self.x0.clone()


class GridTests(unittest.TestCase):
    def test_tail_is_the_grid_from_the_start_hop(self):
        self.assertEqual(ptd.hop_tail_sigmas(GRID6, 0).tolist(), GRID6.tolist())
        self.assertEqual(ptd.hop_tail_sigmas(GRID6, 5).tolist(), [float(GRID6[5]), 0.0])

    def test_the_last_hop_is_a_single_evaluation(self):
        self.assertEqual(len(ptd.hop_tail_sigmas(GRID6, len(GRID6) - 2)) - 1, 1)

    def test_start_hop_is_bounded(self):
        for bad in (-1, 6, 7, True, 1.0):
            with self.assertRaises(ValueError):
                ptd.hop_tail_sigmas(GRID6, bad)

    @unittest.skipUnless(PACK_SCHEDULE.exists(), "PTD pack checkout not found (set PTD_PACK_DIR)")
    def test_grid_matches_the_packs_schedule_for_the_six_hop_file(self):
        spec = importlib.util.spec_from_file_location("ptd_pack_schedule", PACK_SCHEDULE)
        schedule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schedule)
        grid = schedule.hop_sigmas(6, 6, 12.0)
        self.assertEqual(len(grid), 7)
        self.assertTrue(torch.allclose(grid, GRID6, atol=1e-6))
        self.assertAlmostEqual(float(grid[-2]), 0.7052, places=4)


class RenoiseTests(unittest.TestCase):
    def test_each_stream_is_blended_at_its_own_clock(self):
        noise = torch.arange(1.0, 9.0).view(1, 1, 8)
        clean = torch.full((1, 1, 8), 10.0)
        out = ptd.renoise_to_hop(noise, clean, 5, 0.7, 0.4)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.allclose(out[..., :5], 0.3 * 10.0 + 0.7 * noise[..., :5]))
        self.assertTrue(torch.allclose(out[..., 5:], 0.6 * 10.0 + 0.4 * noise[..., 5:]))

    def test_zero_time_is_the_clean_latent_and_full_time_is_noise(self):
        noise, clean = torch.randn(2, 1, 6), torch.randn(2, 1, 6)
        self.assertTrue(torch.allclose(ptd.renoise_to_hop(noise, clean, 4, 0.0, 0.0), clean))
        self.assertTrue(torch.allclose(ptd.renoise_to_hop(noise, clean, 4, 1.0, 1.0), noise))

    def test_shape_and_split_are_checked(self):
        with self.assertRaises(ValueError):
            ptd.renoise_to_hop(torch.zeros(1, 1, 4), torch.zeros(1, 1, 5), 2, 0.5, 0.5)
        with self.assertRaises(ValueError):
            ptd.renoise_to_hop(torch.zeros(1, 1, 4), torch.zeros(1, 1, 4), 9, 0.5, 0.5)
        with self.assertRaises(TypeError):
            ptd.renoise_to_hop([1.0], [1.0], 1, 0.5, 0.5)


class HopSamplerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.noise = torch.randn(1, 1, 10)
        self.latent = torch.randn(1, 1, 10)
        self.x0 = torch.randn(1, 1, 10)
        self.model = FakeModel(self.noise, self.latent, self.x0, [(1, 2, 3), (1, 4)])
        self.controller = FakeController()

    def run_from(self, hop, **kw):
        tail = ptd.hop_tail_sigmas(self.controller.sigmas, hop).float()
        return ptd.sample_pdd_ptd_from_hop(
            self.model, torch.zeros(1, 1, 10), tail, {"denoise_mask": None, "seed": 3},
            controller=self.controller, start_hop=hop, upstream=FakeUpstream, **kw)

    def test_last_hop_runs_one_evaluation_at_its_trained_time_and_returns_x0(self):
        out = self.run_from(5)
        self.assertEqual([h for h, _ in self.controller.activated], [5])
        self.assertAlmostEqual(self.controller.activated[0][1], 0.705176, places=6)
        self.assertEqual(len(self.model.calls), 1)
        self.assertTrue(torch.allclose(out, self.x0, atol=1e-5))

    def test_start_state_is_the_renoised_latent_with_audio_on_the_audio_clock(self):
        self.run_from(5)
        state, sigma, extra = self.model.calls[0]
        t = float(GRID6[5])
        ta = FakeUpstream.audio_sigma(t)
        n = 6
        self.assertTrue(torch.allclose(state[..., :n], (1 - t) * self.latent[..., :n] + t * self.noise[..., :n], atol=1e-6))
        self.assertTrue(torch.allclose(state[..., n:], (1 - ta) * self.latent[..., n:] + ta * self.noise[..., n:], atol=1e-6))
        self.assertAlmostEqual(sigma, t, places=6)
        self.assertEqual(extra, {"seed": 3, "denoise_mask": None})

    def test_start_at_hop_zero_walks_every_hop_in_order(self):
        self.run_from(0)
        self.assertEqual([h for h, _ in self.controller.activated], list(range(6)))
        self.assertEqual(len(self.model.calls), 6)

    def test_callback_steps_are_relative_to_the_start_hop(self):
        seen = []
        self.run_from(4, callback=None)
        ptd.sample_pdd_ptd_from_hop(
            self.model, torch.zeros(1, 1, 10), ptd.hop_tail_sigmas(self.controller.sigmas, 4).float(), {},
            callback=lambda d: seen.append(d["i"]), controller=self.controller, start_hop=4, upstream=FakeUpstream)
        self.assertEqual(seen, [0, 1])

    def test_wrong_grid_mask_or_shape_are_refused(self):
        with self.assertRaises(ValueError):
            ptd.sample_pdd_ptd_from_hop(self.model, torch.zeros(1, 1, 10), GRID6.float(), {},
                                        controller=self.controller, start_hop=5, upstream=FakeUpstream)
        with self.assertRaises(ValueError):
            ptd.sample_pdd_ptd_from_hop(self.model, torch.zeros(1, 1, 10), GRID6[5:].float(),
                                        {"denoise_mask": torch.ones(1)}, controller=self.controller,
                                        start_hop=5, upstream=FakeUpstream)
        with self.assertRaises(ValueError):
            ptd.sample_pdd_ptd_from_hop(self.model, torch.zeros(1, 1, 11), GRID6[5:].float(), {},
                                        controller=self.controller, start_hop=5, upstream=FakeUpstream)


class Model:
    def __init__(self, tag="base"):
        self.tag = tag

    def clone(self):
        return Model(self.tag)

    def set_model_optimized_attention(self, fn):
        pass


class Lora:
    def load_lora_model_only(self, model, name, strength):
        return (Model(model.tag + "+lora"),)


def _engine_class():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MasterEngine")
    names = ("configure_pass2_lora", "configure_pass2_engine", "_ptd_last_hop", "initialize_pass2_model",
             "_pass2_sla_override", "_finish_model", "_pass2_lora_active", "_apply_sla")
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    stub = ast.ClassDef(name="Engine", bases=[], keywords=[], body=body, decorator_list=[])
    ns = {
        "_LOG": logging.getLogger("test_ptd_mode"),
        "ptd_mode": ptd,
        "nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS={"LoraLoaderModelOnly": Lora}),
        "_safe_get_output": lambda res, i, name: res[i],
        "get_attention_function": lambda name, _: object(),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), str(ENGINE), "exec"), ns)
    return ns["Engine"]


Engine = _engine_class()


def make(ptd_on=True, adapter="a.safetensors"):
    e = Engine()
    e.ptd_mode = ptd_on
    e.hyperflow_mode = not ptd_on
    e.turbo_mode = False
    e.ptd_adapter = adapter if ptd_on else None
    e.sla_enabled = False
    e.attention_backend = "comfy kitchen attention"
    e.prepared = Model("ptd+attn")
    e.initialize_model = lambda: (e.prepared, None)
    e.ptd_base = Model("ptd")
    e.accel_model = e.ptd_base
    e.shifted_model = Model("shifted")
    e.pass2_sla_sparsity = 0.0
    e.pass2_model = None
    e.pass2_sla_model = None
    e.ptd_controller = FakeController()
    return e


class EngineWiring(unittest.TestCase):
    def test_arm_c_pass2_uses_the_ptd_model_and_needs_no_lora(self):
        e = make()
        e.configure_pass2_lora("none", 1.0, "replace engine LoRA", 4)
        e.configure_pass2_engine(ptd.PASS2_ENGINE_PTD)
        self.assertIs(e.initialize_pass2_model(), e.prepared)
        self.assertEqual(e._ptd_last_hop(), 5)

    def test_arm_b_pass2_is_the_dmad_lora_on_the_plain_shifted_model(self):
        e = make()
        e.configure_pass2_lora("dmad.safetensors", 1.0, "stack on engine LoRA", 4)
        e.configure_pass2_engine(ptd.PASS2_ENGINE_LORA)
        self.assertEqual(e.pass2_lora_mode, "replace")
        self.assertEqual(e.initialize_pass2_model().tag, "shifted+lora")

    def test_arm_b_without_a_pass2_lora_is_refused(self):
        e = make()
        e.configure_pass2_lora("none", 1.0, "replace engine LoRA", 4)
        e.configure_pass2_engine(ptd.PASS2_ENGINE_LORA)
        with self.assertRaisesRegex(ValueError, "needs a pass2_lora"):
            e.initialize_pass2_model()

    def test_ptd_last_hop_without_ptd_mode_falls_back_to_lora_refine(self):
        # The panel hides pass2_engine outside PTD mode, so a leftover "PTD last hop" must not fail a HyperFlow run.
        e = make(ptd_on=False)
        e.configure_pass2_lora("dmad.safetensors", 1.0, "replace engine LoRA", 4)
        e.configure_pass2_engine(ptd.PASS2_ENGINE_PTD)
        self.assertEqual(e.pass2_engine, ptd.PASS2_ENGINE_LORA)

    def test_ptd_mode_needs_an_adapter(self):
        e = make(adapter=None)
        e.configure_pass2_lora("dmad.safetensors", 1.0, "replace engine LoRA", 4)
        with self.assertRaisesRegex(ValueError, "pick a PTD adapter"):
            e.configure_pass2_engine(ptd.PASS2_ENGINE_LORA)

    def test_unknown_engine_is_refused(self):
        with self.assertRaises(ValueError):
            make().configure_pass2_engine("nope")

    def test_non_ptd_default_is_untouched(self):
        e = make(ptd_on=False)
        e.configure_pass2_lora("dmad.safetensors", 1.0, "stack on engine LoRA", 4)
        e.configure_pass2_engine(ptd.PASS2_ENGINE_LORA)
        self.assertEqual(e.pass2_lora_mode, "replace")  # HyperFlow already forced replace
        e2 = make(ptd_on=False)
        e2.hyperflow_mode = False
        e2.configure_pass2_lora("dmad.safetensors", 1.0, "stack on engine LoRA", 4)
        self.assertEqual(e2.pass2_lora_mode, "stack")


class ModeWiring(unittest.TestCase):
    text = NODE.read_text(encoding="utf-8")
    engine_text = ENGINE.read_text(encoding="utf-8")

    def test_accel_mode_value_matches_the_module_constant_and_is_last(self):
        modes = ast.literal_eval(re.search(r"^ACCEL_MODES = (\[.*\])$", self.text, re.M).group(1))
        self.assertEqual(modes[-1], ptd.PTD_ACCEL_MODE)
        self.assertEqual(modes[:3], ["HyperFlow 8-step", "Turbo LoRA", "HyperFlow 8-step · dual sampling"])

    def test_new_inputs_are_optional_appended_last_and_default_to_the_old_behaviour(self):
        optional = self.text[self.text.index('"optional": {'):self.text.index('"hidden": {')]
        names = re.findall(r'^\s{16}"(\w+)": \(', optional, re.M)
        self.assertEqual(names[-2:], ["ptd_adapter", "pass2_engine"])
        self.assertIn('"default": "none"', re.search(r'"ptd_adapter".*', optional).group(0))
        self.assertIn("ptd_mode.PASS2_ENGINE_LORA", re.search(r'"pass2_engine".*', optional).group(0))

    def test_engine_gets_the_adapter_and_pass2_engine(self):
        self.assertIn('ptd_adapter=kwargs.get("ptd_adapter", "none")', self.text)
        self.assertIn('engine.configure_pass2_engine(kwargs.get("pass2_engine"', self.text)

    def test_ptd_is_not_dual_sampling(self):
        self.assertIn("not ptd_mode.is_ptd_mode(accel_mode)", self.text)

    def test_pass1_fingerprint_changes_only_in_ptd_mode(self):
        base = {"accel_mode": "Turbo LoRA"}
        self.assertIs(ptd.with_ptd_p1_settings(base, None), base)
        a = ptd.with_ptd_p1_settings({"accel_mode": ptd.PTD_ACCEL_MODE}, "a.safetensors")
        b = ptd.with_ptd_p1_settings({"accel_mode": ptd.PTD_ACCEL_MODE}, "b.safetensors")
        self.assertNotEqual(a, b)
        self.assertEqual(a["ptd"], "a.safetensors")

    def test_pass1_ptd_uses_the_packs_sampler_and_pass2_the_hop_sampler(self):
        self.assertIn("self.ptd_pack.sampler.make_sampler(self.ptd_controller)", self.engine_text)
        self.assertIn("ptd_mode.make_hop_sampler(self.ptd_controller, hop)", self.engine_text)

    def test_missing_pack_is_a_clear_error(self):
        fake_nodes = types.ModuleType("nodes")
        fake_nodes.NODE_CLASS_MAPPINGS = {}
        old = sys.modules.get("nodes")
        sys.modules["nodes"] = fake_nodes
        try:
            with self.assertRaisesRegex(RuntimeError, "ComfyUI-MiniMaxH3-PTD"):
                ptd.ptd_pack()
        finally:
            if old is None:
                del sys.modules["nodes"]
            else:
                sys.modules["nodes"] = old


if __name__ == "__main__":
    unittest.main()
