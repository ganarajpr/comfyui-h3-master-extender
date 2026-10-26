import ast
import importlib.util
import logging
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("upfront_encode", ROOT / "upfront_encode.py")
ue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ue)


class T:
    """Stand-in tensor that remembers its device and counts moves."""

    def __init__(self, n, device="cuda"):
        self.n, self.device = n, device

    def to(self, device):
        return T(self.n, device)

    def numel(self):
        return self.n

    def element_size(self):
        return 4


class Nested:
    def __init__(self, *tensors):
        self.tensors = tensors

    def to(self, device):
        return Nested(*(t.to(device) for t in self.tensors))


def fake_encode(index):
    cond = [[T(10), {"minimax_refs": [{"kind": "image", "latent": T(5)}], "tag": index}]]
    return (cond, {"samples": Nested(T(100), T(20))})


class CacheTests(unittest.TestCase):
    def test_fill_moves_to_cpu_and_counts_bytes(self):
        cache = ue.UpfrontEncodeCache()
        cache.fill([0, 2], fake_encode)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.bytes, 2 * (10 + 5 + 100 + 20) * 4)
        cond, latent = cache._items[0]
        self.assertEqual(cond[0][0].device, "cpu")
        self.assertEqual(cond[0][1]["minimax_refs"][0]["latent"].device, "cpu")
        self.assertEqual(latent["samples"].tensors[0].device, "cpu")
        self.assertEqual(cond[0][1]["tag"], 0)

    def test_take_returns_on_device_once(self):
        cache = ue.UpfrontEncodeCache()
        cache.fill([1], fake_encode)
        cond, latent = cache.take(1, "cuda")
        self.assertEqual(cond[0][0].device, "cuda")
        self.assertEqual(latent["samples"].tensors[1].device, "cuda")
        self.assertIsNone(cache.take(1, "cuda"))
        self.assertIsNone(cache.take(7, "cuda"))

    def test_encode_runs_once_per_index_and_summary(self):
        calls = []
        cache = ue.UpfrontEncodeCache()
        cache.fill([0, 1, 2], lambda i: calls.append(i) or fake_encode(i))
        self.assertEqual(calls, [0, 1, 2])
        cache.take(0, "cpu")
        self.assertRegex(cache.summary(), r"^Up-front text encode: 3 clips in \d+\.\d s, \d+\.\d\d GB on CPU$")
        cache.clear()
        self.assertEqual(len(cache), 0)

    def test_output_parts_accepts_node_output_shapes(self):
        pair = ("pos", "lat")
        self.assertEqual(ue.output_parts(pair), pair)
        self.assertEqual(ue.output_parts(types.SimpleNamespace(result=pair)), pair)
        self.assertEqual(ue.output_parts(types.SimpleNamespace(args=list(pair))), pair)
        self.assertEqual(ue.output_parts({"positive": "pos", "latent": "lat"}), pair)


class PlanTests(unittest.TestCase):
    clips = [{"prompt": f"p{i}"} for i in range(4)]

    def fp(self, prev, cfg):
        return f"{prev}>{cfg['prompt']}"

    def stored(self, n):
        out, prev = [], None
        for c in self.clips[:n]:
            prev = self.fp(prev, c)
            out.append(prev)
        return out

    def test_first_render_index(self):
        self.assertEqual(ue.first_render_index(self.clips, 0, [], self.fp), 0)
        self.assertEqual(ue.first_render_index(self.clips, 2, self.stored(2), self.fp), 2)
        self.assertEqual(ue.first_render_index(self.clips, 4, self.stored(4), self.fp), 4)
        stale = self.stored(3)
        stale[1] = "changed"
        self.assertEqual(ue.first_render_index(self.clips, 3, stale, self.fp), 1)

    def test_rerolling_clip_renders_unless_validated(self):
        clips = [dict(c, seed_mode="randomize") for c in self.clips]
        self.assertEqual(ue.first_render_index(clips, 4, self.stored(4), self.fp), 0)
        clips[0]["validated"] = True
        self.assertEqual(ue.first_render_index(clips, 4, self.stored(4), self.fp), 1)
        self.assertEqual(ue.first_render_index(clips[:1], 1, [], self.fp), 1)

    def test_unfingerprintable_clip_renders(self):
        def bad(prev, cfg):
            raise ValueError
        self.assertEqual(ue.first_render_index(self.clips, 4, ["x"] * 4, bad), 0)

    def test_upfront_indices(self):
        self.assertEqual(ue.upfront_indices(self.clips, 1, "full_batch", False), [1, 2, 3])
        self.assertEqual(ue.upfront_indices(self.clips, 1, "clip_by_clip", False), [1])
        self.assertEqual(ue.upfront_indices(self.clips, 4, "full_batch", False), [])
        self.assertEqual(ue.upfront_indices(self.clips, 0, "full_batch", True), [0])
        self.assertEqual(ue.upfront_indices(self.clips, 1, "full_batch", True), [])


class Core:
    calls = []

    @classmethod
    def execute(cls, **kw):
        cls.calls.append(kw)
        return ("pos", "lat")


def _engine():
    tree = ast.parse((ROOT / "pdd_pure_engine.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MasterEngine")
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "encode_clip_text"]
    stub = ast.ClassDef(name="Engine", bases=[], keywords=[], body=body, decorator_list=[])
    ns = {"MiniMaxH3ReferenceToVideo": Core, "duration_to_h3_frames": lambda d: int(d * 24) + 1, "_LOG": logging.getLogger("t")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), "engine", "exec"), ns)
    return ns["Engine"]


class EncodeCallTests(unittest.TestCase):
    def make(self):
        e = _engine()()
        e.clip, e.vae, e.audio_vae = "clip", "vae", "avae"
        Core.calls.clear()
        return e

    def test_encodes_at_pass2_size(self):
        e = self.make()
        refs = {"ref_image_0": "img"}
        e.encode_clip_text("hello", 15, (1280, 736), (608, 352), refs, True, (1280, 736))
        kw = Core.calls[0]
        self.assertEqual((kw["width"], kw["height"], kw["length"]), (1280, 736, 361))
        self.assertEqual((kw["prompt"], kw["ref_image_size"], kw["ref_images"]), ("hello", "match", refs))
        self.assertIsNot(kw["ref_images"], refs)
        self.assertIsNone(kw["ref_videos"])

    def test_pass2_off_uses_nominal_final_size(self):
        e = self.make()
        e.encode_clip_text("x", 15, (608, 352), (608, 352), {}, False, (1280, 736))
        self.assertEqual((Core.calls[0]["width"], Core.calls[0]["height"]), (1280, 736))
        e.encode_clip_text("x", 15, (608, 352), (608, 352), {}, False, None)
        self.assertEqual((Core.calls[1]["width"], Core.calls[1]["height"]), (608, 352))


class WiringTests(unittest.TestCase):
    def test_input_is_last_optional_boolean_default_false(self):
        tree = ast.parse((ROOT / "master_node.py").read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "INPUT_TYPES")
        ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
        optional = next(v for k, v in zip(ret.value.keys, ret.value.values) if getattr(k, "value", None) == "optional")
        last_key, last_val = optional.keys[-1], optional.values[-1]
        self.assertEqual(last_key.value, "upfront_text_encode")
        self.assertEqual(last_val.elts[0].value, "BOOLEAN")
        default = next(v for k, v in zip(last_val.elts[1].keys, last_val.elts[1].values) if k.value == "default")
        self.assertIs(default.value, False)


if __name__ == "__main__":
    unittest.main()
