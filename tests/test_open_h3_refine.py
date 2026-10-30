"""Open-H3 refine (pass2_refiner): region start noise and the per-window face-map wrapper. Needs torch and ComfyUI's
comfy package (run on the box with ComfyUI on sys.path); skipped otherwise."""
import ast
import sys
import types
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "pdd_pure_engine.py"

try:
    import torch
    import comfy.nested_tensor
    import comfy.sample
except Exception:      # no torch / ComfyUI here
    torch = None


def _load():
    """The module-level _RegionNoise class and MasterEngine._with_region_strength, without importing the engine."""
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_RegionNoise"]
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MasterEngine")
    fn = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "_with_region_strength")
    fn.decorator_list = []
    ns = {"torch": torch, "comfy": comfy}
    exec(compile(ast.Module(body=keep + [fn], type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns["_RegionNoise"], ns["_with_region_strength"]


class Patcher:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        p = Patcher()
        p.model_options = dict(self.model_options)
        return p

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn


@unittest.skipIf(torch is None, "needs torch + ComfyUI")
class OpenH3RefineTests(unittest.TestCase):
    def setUp(self):
        self.RegionNoise, self.with_region = _load()
        self.up = torch.randn(1, 4, 10, 3, 5)
        self.mask = torch.ones(10, 3, 5)
        self.mask[:, 1, 2] = 0.55

    def test_start_state_is_m_times_sigma(self):
        latent = {"samples": comfy.nested_tensor.NestedTensor((self.up, torch.zeros(1, 2, 3, 7)))}
        plain = comfy.sample.prepare_noise(latent["samples"], 7, None).unbind()[0]
        video, audio = self.RegionNoise(7, self.mask, self.up).generate_noise(latent).unbind()
        s = 0.909375
        start = s * video + (1 - s) * self.up        # ComfyUI's flow noise scaling
        want = (self.mask * s)[None, None] * plain + (1 - (self.mask * s)[None, None]) * self.up
        self.assertTrue(torch.allclose(start, want, atol=1e-5))
        self.assertEqual(tuple(audio.shape), (1, 2, 3, 7))

    def test_all_ones_map_is_the_plain_noise(self):
        latent = {"samples": comfy.nested_tensor.NestedTensor((self.up, torch.zeros(1, 2, 3, 7)))}
        plain = comfy.sample.prepare_noise(latent["samples"], 7, None).unbind()[0]
        video, _ = self.RegionNoise(7, torch.ones(10, 3, 5), self.up).generate_noise(latent).unbind()
        self.assertTrue(torch.equal(video, plain))

    def test_split_node_requests_get_region_noise_for_video_only(self):
        rn = self.RegionNoise(3, self.mask, self.up)
        v = rn.generate_noise({"samples": torch.zeros_like(self.up)})
        a = rn.generate_noise({"samples": torch.zeros(1, 2, 3, 7)})
        self.assertFalse(torch.equal(v, comfy.sample.prepare_noise(torch.zeros_like(self.up), 3, None)))
        self.assertTrue(torch.equal(a, comfy.sample.prepare_noise(torch.zeros(1, 2, 3, 7), 3, None)))

    def test_windows_get_their_slice_in_order(self):
        sigmas = torch.FloatTensor([0.909375, 0.857143, 0.666667, 0.0])
        bounds = [(0, 6), (4, 10)]
        seen = []

        def apply_model(x, t, **c):
            seen.append(c["denoise_mask"].clone())
            return x

        model = self.with_region(Patcher(), self.mask * torch.arange(1, 11).view(10, 1, 1) / 10, sigmas, bounds)
        wrap = model.model_options["model_function_wrapper"]
        for k0, k1 in bounds:
            for s in sigmas[:-1]:
                wrap(apply_model, {"input": torch.zeros(1), "timestep": torch.tensor([float(s)]),
                                   "c": {"latent_shapes": [[1, 4, k1 - k0, 3, 5]], "audio_denoise_mask": 0}})
        self.assertEqual(len(seen), 6)
        self.assertEqual(tuple(seen[0].shape), (1, 1, 6, 3, 5))
        self.assertAlmostEqual(float(seen[3][0, 0, 0, 0, 0]), 0.5)      # window 2 starts at latent frame 4
        self.assertAlmostEqual(float(seen[0][0, 0, 0, 0, 0]), 0.1)

    def test_a_wrong_window_length_raises(self):
        model = self.with_region(Patcher(), self.mask, torch.FloatTensor([0.9, 0.0]), [(0, 10)])
        wrap = model.model_options["model_function_wrapper"]
        with self.assertRaises(RuntimeError):
            wrap(lambda x, t, **c: x, {"input": torch.zeros(1), "timestep": torch.tensor([0.9]),
                                       "c": {"latent_shapes": [[1, 4, 9, 3, 5]]}})


if __name__ == "__main__":
    unittest.main()
