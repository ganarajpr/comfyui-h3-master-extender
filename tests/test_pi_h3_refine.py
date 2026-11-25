"""The DWD frame path of the final video: the same frame count, crop, seam shift and colour correction as the VAE decode."""
import ast
import collections
import hashlib
import logging
import math
import threading
import time
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DISK = ROOT / "motion_context_disk.py"
RAM = ROOT / "motion_context_ram.py"
DISK_FUNCS = {"dwd_latent_key", "_video_t", "_pair_previous_tail_frames", "register_dwd_frames", "clear_dwd_frames", "dwd_guide_frame",
              "_take_dwd_full", "_dwd_tail", "_dwd_pair_decoded", "_render_one_final_video_segment", "_decode_pair_video",
              "_build_pair_video", "_correct_current_segment", "decode_guide_frame"}
DISK_NAMES = {"_DWD_FRAMES", "_DWD_LOCK", "_DWD_MAX_ENTRIES", "PAIR_PREVIOUS_TAIL_T"}
RAM_FUNCS = {"_frames_from_video_t", "_pixel_frames", "_steps_for_frames", "_luma_map", "_luma_stats", "_auto_early_seam_shift",
             "_temporal_history_descriptor", "_temporal_candidate_score", "_temporal_gray", "_temporal_motion_delta",
             "_temporal_frame_delta"}
RAM_NAMES = {"FRAME_PER_TOKEN", "FPS"}


def _pick(path, funcs, names, ns):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in funcs)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets))]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)


NS = {"torch": torch, "math": math, "time": time, "hashlib": hashlib, "collections": collections, "threading": threading,
      "log": logging.getLogger("test_pi_h3_refine"), "np": __import__("numpy")}
_pick(RAM, RAM_FUNCS, RAM_NAMES, NS)
_pick(DISK, DISK_FUNCS, DISK_NAMES, NS)

C, H, W = 4, 3, 5
FRAMES_T = 12  # 39 frames


def fake_decode(latent):
    """A stand-in for the VAE: every latent step k (phase k % 5 inside the buffer) paints its frames with one value per step,
    so a frame depends only on its own latent step. Returns (F, h, w, 3)."""
    steps = latent.shape[2]
    out = []
    for k in range(steps):
        value = float(latent[0, 0, k, 0, 0])
        out += [torch.full((4, 6, 3), value)] * NS["FRAME_PER_TOKEN"][k % 5]
    return torch.stack(out)


def make_clip(seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.rand(1, C, FRAMES_T, H, W, generator=g)
    v[0, 0, :, :, :] = torch.rand(FRAMES_T, 1, 1, generator=g)
    return v


class Chain(unittest.TestCase):
    def setUp(self):
        NS["_DWD_FRAMES"].clear()
        self.v0, self.v1 = make_clip(1), make_clip(2)
        self.segments = [
            {"index": 0, "frames": 39, "trim_frames": 0, "latent": self.v0},
            {"index": 1, "frames": 39, "trim_frames": 22, "latent": self.v1},
        ]
        self.vae_calls = []

        def decode_video_latent(vae, latent):
            self.vae_calls.append(tuple(latent.shape))
            return fake_decode(latent)

        NS["_load_segment_video"] = lambda data_path, desc: desc["latent"]
        NS["decode_video_latent"] = decode_video_latent

    def render(self, i, **kw):
        return NS["_render_one_final_video_segment"]("data", self.segments, i, object(), **kw)

    def register(self, which=(0, 1)):
        for i in which:
            v = self.segments[i]["latent"]
            assert NS["register_dwd_frames"](v, fake_decode(v).half().float().half()) is True

    def reference(self):
        out = [self.render(0), self.render(1)]
        calls = list(self.vae_calls)
        self.vae_calls.clear()
        return out, calls

    def test_the_two_clip_chain_matches_the_vae_path(self):
        (ref0, ref1), calls = self.reference()
        self.assertEqual(len(calls), 2)
        self.register()
        with self.assertLogs("test_pi_h3_refine", "INFO") as cm:
            dwd0, dwd1 = self.render(0), self.render(1)
        self.assertEqual(self.vae_calls, [], "no VAE decode when both clips have DWD frames")
        self.assertEqual(sum("video output (DWD)" in line for line in cm.output), 2)
        self.assertEqual(dwd0[1], ref0[1])
        self.assertEqual(dwd1[1], ref1[1])
        for dwd, ref in ((dwd0[0], ref0[0]), (dwd1[0], ref1[0])):
            self.assertEqual(tuple(dwd.shape), tuple(ref.shape))
            self.assertEqual(dwd.dtype, torch.float32)
            self.assertTrue(torch.allclose(dwd, ref, atol=2e-3))
        self.assertEqual(dwd0[0].shape[0], 39)
        self.assertEqual(dwd1[0].shape[0], 39 - 22)

    def test_a_forced_seam_shift_crops_the_same_frames(self):
        ref1 = self.render(1, forced_shift=-1)
        self.render(0)
        self.vae_calls.clear()
        self.register()
        dwd1 = self.render(1, forced_shift=-1)
        self.assertEqual(tuple(dwd1[0].shape), tuple(ref1[0].shape))
        self.assertEqual(dwd1[1], -1)
        self.assertTrue(torch.allclose(dwd1[0], ref1[0], atol=2e-3))

    def test_a_clip_without_dwd_frames_uses_the_vae(self):
        self.register((0,))
        self.render(0)
        self.assertEqual(self.vae_calls, [])
        self.render(1)
        self.assertEqual(len(self.vae_calls), 1)

    def test_full_pixels_are_released_after_use_and_only_tails_remain(self):
        self.register()
        self.render(0)
        self.render(1)
        entries = list(NS["_DWD_FRAMES"].values())
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(e["full"] is None for e in entries))
        self.assertTrue(all(e["tail"] is not None and e["tail"].shape[0] <= 39 for e in entries))
        NS["clear_dwd_frames"]()
        self.assertEqual(len(NS["_DWD_FRAMES"]), 0)

    def test_registry_is_bounded(self):
        for seed in range(NS["_DWD_MAX_ENTRIES"] + 3):
            v = make_clip(100 + seed)
            NS["register_dwd_frames"](v, fake_decode(v).half())
        self.assertEqual(len(NS["_DWD_FRAMES"]), NS["_DWD_MAX_ENTRIES"])

    def test_a_wrong_frame_count_is_refused(self):
        v = make_clip(5)
        with self.assertLogs("test_pi_h3_refine", "WARNING"):
            self.assertFalse(NS["register_dwd_frames"](v, fake_decode(v)[:-1].half()))
        self.assertEqual(len(NS["_DWD_FRAMES"]), 0)

    def test_the_latent_is_not_modified_and_keys_are_stable(self):
        before = self.v0.clone()
        self.register((0,))
        self.render(0)
        self.assertTrue(torch.equal(before, self.v0))
        self.assertEqual(NS["dwd_latent_key"](self.v0), NS["dwd_latent_key"](self.v0.squeeze(0).clone()))

    def test_guide_frame_comes_from_the_dwd_pixels(self):
        self.register((0,))
        frame = NS["decode_guide_frame"](None, self.v0)
        self.assertEqual(frame.shape[0], 1)
        self.assertTrue(torch.allclose(frame, fake_decode(self.v0)[-1:], atol=2e-3))
        self.assertEqual(self.vae_calls, [])


if __name__ == "__main__":
    unittest.main()
