import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tg = _load("time_gate_lora_t", ROOT / "time_gate_lora.py")

SPANS = [(0.0, 0.5, 0.0), (0.5, 1.0, 0.5), (1.0, 2.0, 1.0), (2.0, 3.0, 0.0)]


def fake_swiglu(x):
    a, b = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(b) * a


class FakeLayout:
    # text 4 | cond 3 | ref_img 5 | audio 2*12 | video 7 latent frames x 2 rows
    def __init__(self):
        self.signature = (4, 7, 4, 4, 12)
        self.segments = [(0, 4, "text"), (4, 7, "cond"), (7, 12, "ref_img"), (12, 36, "audio"), (36, 50, "video")]


class GateTests(unittest.TestCase):
    def test_parse(self):
        self.assertIsNone(tg.parse_time_gate(""))
        self.assertIsNone(tg.parse_time_gate("  \n"))
        cfg = tg.parse_time_gate(json.dumps({"lora": "h3\\x.safetensors", "clips": {"0": [[0, 1, 1]], "2": [[0, 3, 0.5]]}}))
        self.assertEqual(cfg["other_tokens"], 1.0)
        self.assertEqual(tg.clip_spans(cfg, 0), [(0.0, 1.0, 1.0)])
        self.assertIsNone(tg.clip_spans(cfg, 1))
        with self.assertRaises(ValueError):
            tg.parse_time_gate("{nope")
        with self.assertRaises(ValueError):
            tg.parse_time_gate(json.dumps({"lora": "x", "clips": {"0": [[0, 1]]}}))

    def test_video_frame_time_mapping(self):
        # latent frames cover pixel frames [0], [1-4], [5-8], [9-12], [13-16], [17], [18-21]; time = pixel / 24
        g = tg.video_frame_gates(7, 0, [(0.0, 0.25, 1.0)])  # pixel frames 0..5 (t < 0.25 s)
        self.assertAlmostEqual(float(g[0]), 1.0)
        self.assertAlmostEqual(float(g[1]), 1.0)
        self.assertAlmostEqual(float(g[2]), 1.0 / 4.0)  # pixel 5 (0.208 s) in, 6..8 (>= 0.25 s) out
        self.assertAlmostEqual(float(g[3]), 0.0)

    def test_context_overlap_is_zero_and_time_is_relative_to_first_new_frame(self):
        # 5 overlap pixel frames: first new frame is pixel 5, clip time 0
        g = tg.video_frame_gates(7, 5, [(0.0, 100.0, 1.0)])
        self.assertAlmostEqual(float(g[0]), 0.0)   # pixel 0
        self.assertAlmostEqual(float(g[1]), 0.0)   # pixels 1-4
        self.assertAlmostEqual(float(g[2]), 1.0)   # pixels 5-8
        a = tg.audio_step_gates(40, 24, [(0.0, 100.0, 1.0)])  # 24 overlap frames = 1.0 s = 40 steps
        self.assertAlmostEqual(float(a.sum()), 0.0)

    def test_audio_time_mapping(self):
        a = tg.audio_step_gates(80, 0, [(0.5, 1.0, 1.0)])  # 40 Hz: steps 20..39
        self.assertEqual(a.nonzero().flatten().tolist(), list(range(20, 40)))

    def test_build_gate_layout(self):
        lay = FakeLayout()
        spans = [(0.0, 100.0, 0.75)]
        gate = tg.build_gate(lay.segments, lay.signature, spans, trim_frames=0, other_tokens=1.0)
        self.assertEqual(gate.shape[0], 50)
        self.assertTrue(torch.all(gate[0:4] == 1.0))      # text
        self.assertTrue(torch.all(gate[4:7] == 0.0))      # motion-context cond
        self.assertTrue(torch.all(gate[7:12] == 1.0))     # reference image
        self.assertTrue(torch.allclose(gate[12:36], torch.full((24,), 0.75)))  # audio, both channels
        self.assertTrue(torch.allclose(gate[36:50], torch.full((14,), 0.75)))  # video
        off = tg.build_gate(lay.segments, lay.signature, [(0.0, 0.1, 1.0)], trim_frames=0, other_tokens=0.5)
        self.assertTrue(torch.all(off[0:4] == 0.5))
        self.assertTrue(torch.all(off[7:12] == 0.5))
        # video frame 0 (pixel 0, t 0) in span, frame 1 (pixels 1-4) partly
        self.assertAlmostEqual(float(off[36]), 1.0)
        self.assertAlmostEqual(float(off[38]), 0.5)  # pixels 1,2 at t < 0.1 s (0.042, 0.083); 3, 4 at 0.125, 0.167 are out
        self.assertAlmostEqual(float(off[40]), 0.0)

    def test_two_rows_per_frame_repeat_in_frame_major_order(self):
        lay = FakeLayout()
        gate = tg.build_gate(lay.segments, lay.signature, SPANS, trim_frames=0)
        video = gate[36:50].view(7, 2)
        self.assertTrue(torch.all(video[:, 0] == video[:, 1]))


class TinyBlock(nn.Module):
    def __init__(self, d, ffn, chunks=1):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(d, 3 * d, bias=False)
        self.attn.out_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(d, 2 * ffn, bias=False)
        self.mlp.fc2 = nn.Linear(ffn, d, bias=False)
        self.chunks = chunks
        d_ = d
        self.d = d_

        def mlp_forward(x):
            out = torch.empty_like(x)
            off = 0
            for c in torch.chunk(x, self.chunks, dim=0):
                # like the fused path: fc2's forward is not used, so a module hook on fc2 would never fire
                a = fake_swiglu(self.mlp.fc1(c))
                out[off:off + c.shape[0]] = torch.nn.functional.linear(a, self.mlp.fc2.weight)
                off += c.shape[0]
            return out

        self.mlp.forward = mlp_forward

    def forward(self, x, t_emb, mod_segments, rope, transformer_options={}, attention=None):
        q = self.attn.qkv_proj(x)[:, :self.d]
        x = x + self.attn.out_proj(q)
        return x + self.mlp(x)


class TinyModel(nn.Module):
    def __init__(self, d=8, ffn=12, n=2, chunks=1):
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock(d, ffn, chunks) for _ in range(n))

    def run(self, x, layout):
        opts = {"minimax_h3_layout": layout}
        for blk in self.blocks:
            x = blk(x, None, None, None, transformer_options=opts)
        return x


def reference(model, x, gate, pairs):
    g = gate[:, None]

    def lin(layer, i, name, inp):
        a, b = pairs[(i, name)]
        return layer(inp) + g * (inp @ a.t() @ b.t())

    for i, blk in enumerate(model.blocks):
        q = lin(blk.attn.qkv_proj, i, "attn.qkv_proj", x)[:, :blk.d]
        x = x + lin(blk.attn.out_proj, i, "attn.out_proj", q)
        h1 = lin(blk.mlp.fc1, i, "mlp.fc1", x)
        act = fake_swiglu(h1)
        a, b = pairs[(i, "mlp.fc2")]
        x = x + torch.nn.functional.linear(act, blk.mlp.fc2.weight) + g * (act @ a.t() @ b.t())
    return x


class HookTests(unittest.TestCase):
    def _check(self, chunks):
        torch.manual_seed(0)
        d, ffn = 8, 12
        model = TinyModel(d, ffn, 2, chunks)
        shapes = {"attn.qkv_proj": (d, 3 * d), "attn.out_proj": (d, d), "mlp.fc1": (d, 2 * ffn), "mlp.fc2": (ffn, d)}
        pairs = {(i, n): (torch.randn(4, fin) * 0.3, torch.randn(fout, 4) * 0.3)
                 for i in range(2) for n, (fin, fout) in shapes.items()}
        lay = FakeLayout()
        x = torch.randn(50, d)
        base = model.run(x, lay)
        spans = [(0.0, 0.2, 1.0), (0.2, 0.4, 0.5)]
        gate = tg.build_gate(lay.segments, lay.signature, spans, 0, 1.0)
        want = reference(model, x, gate, pairs)
        lora = tg.TimeGateLora(model, pairs, spans, 0, 1.0, device="cpu", dtype=torch.float32, swiglu=fake_swiglu)
        with lora:
            got = model.run(x, lay)
        self.assertTrue(torch.allclose(got, want, atol=1e-4), float((got - want).abs().max().detach()))
        self.assertFalse(torch.allclose(got, base, atol=1e-3))
        self.assertTrue(torch.allclose(model.run(x, lay), base))   # hooks are gone
        self.assertEqual(sum(len(b._forward_hooks) + len(b._forward_pre_hooks) for b in model.blocks), 0)

    def test_hooks_match_reference_unchunked(self):
        self._check(1)

    def test_hooks_match_reference_with_token_chunks(self):
        self._check(3)

    def test_hooks_removed_on_exception(self):
        model = TinyModel()
        pairs = {(i, n): (torch.zeros(2, 8), torch.zeros(8, 2)) for i in range(2) for n in tg.LINEAR_NAMES}
        try:
            with tg.TimeGateLora(model, pairs, [], 0, device="cpu", dtype=torch.float32, swiglu=fake_swiglu):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertEqual(sum(len(m._forward_hooks) for m in model.modules()), 0)


if __name__ == "__main__":
    unittest.main()
