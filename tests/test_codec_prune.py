"""CodecOPSD reference pruning: layout post-filter, condition rows, picture masks.

Needs ComfyUI's comfy.ldm.minimax.model (run with the ComfyUI python and PYTHONPATH on ComfyUI). The
oracle for the layout tests is the third-party port's PrunedPackedLayout (CODECPRUNE_ORACLE_DIR, the
folder holding codecprune_patch.py); its layout class is correct for refs without keyframes.
"""
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

try:
    import comfy.ldm.minimax.model as mm
except Exception:  # noqa: BLE001
    mm = None

PKG = "mcd_codec_prune_test"


def _load():
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(ROOT)]
    sys.modules[PKG] = pkg
    spec = importlib.util.spec_from_file_location(PKG + ".codec_prune", ROOT / "codec_prune.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG + ".codec_prune"] = mod
    spec.loader.exec_module(mod)
    return mod


cp = _load()


def _oracle_class():
    d = os.environ.get("CODECPRUNE_ORACLE_DIR")
    if not d or mm is None:
        return None
    spec = importlib.util.spec_from_file_location("oracle_codecprune_patch", Path(d) / "codecprune_patch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_pruned_layout_class(mm)


def _rand_keep(n, gen, p=0.5):
    return torch.rand(n, generator=gen) < p


def _image_ref(lh, lw, keep=None):
    blk = {"kind": "image", "latent_h": lh, "latent_w": lw, "latent": torch.randn(1, 24, 1, lh, lw)}
    if keep is not None:
        blk["keep"] = keep
    return blk


def _audio_ref(rt):
    return {"kind": "audio", "ref_audio_t": rt, "audio_latent": torch.randn(1, 32, 2, rt)}


def _strip(refs):
    return [{k: v for k, v in b.items() if k != "keep"} for b in refs]


def _payload(refs, keyframes=None, strip=False):
    refs = _strip(refs) if strip else refs
    kfs = keyframes or []
    return {"refs": refs, "keyframes": kfs, "visual_cond_noise_aug": 1.0, "seed": 1,
            "cond_video_latents": [k["latent"] for k in kfs if k.get("latent") is not None]
            + [r["latent"] for r in refs if "latent" in r]}


LAYOUT_FIELDS = ("position_ids", "img_pos", "img_update", "audio_pos", "audio_update")


@unittest.skipIf(mm is None, "comfy.ldm.minimax.model not importable")
class LayoutFilter(unittest.TestCase):
    def build(self, cls, refs, audio_t, keyframes=None):
        return cls(5, 3, 8, 10, audio_t, keyframes=keyframes, refs=refs)

    def test_matches_the_port_oracle(self):
        oracle = _oracle_class()
        if oracle is None:
            self.skipTest("CODECPRUNE_ORACLE_DIR not set")
        gen = torch.Generator().manual_seed(7)
        for audio_t in (0, 9):
            for order in ("iia", "iai", "aii"):
                for trial in range(6):
                    refs = []
                    for ch in order:
                        if ch == "i":
                            n = (8 // 2) * (12 // 2)
                            p = [0.5, 0.0, 1.0, 0.2, 0.8, 0.5][trial]
                            refs.append(_image_ref(8, 12, _rand_keep(n, gen, p)))
                        else:
                            refs.append(_audio_ref(3))
                    want = self.build(oracle, refs, audio_t)
                    got = self.build(mm.PackedLayout, _strip(refs), audio_t)
                    cp.filter_layout(got, refs)
                    for f in LAYOUT_FIELDS:
                        self.assertTrue(torch.equal(getattr(want, f), getattr(got, f)), f"{f} {order} {audio_t} {trial}")
                    self.assertEqual(want.segments, got.segments)
                    self.assertEqual(want.seq_len, got.seq_len)
                    self.assertEqual(want.signature, got.signature)

    def test_img_update_invariant_and_cond_rows(self):
        gen = torch.Generator().manual_seed(3)
        refs = [_image_ref(8, 12, _rand_keep(24, gen)), _audio_ref(3), _image_ref(6, 6, _rand_keep(9, gen))]
        layout = self.build(mm.PackedLayout, _strip(refs), 9)
        dense_false = int((~layout.img_update).sum())
        cp.filter_layout(layout, refs)
        fake = types.SimpleNamespace(patch_size=(1, 2, 2))
        payload = _payload(refs)
        dense_payload = _payload(refs, strip=True)
        dense = mm.MiniMaxH3Model._cond_video_rows(fake, dense_payload, "cpu")
        self.assertEqual(dense.shape[0], dense_false)

        orig = mm.MiniMaxH3Model._cond_video_rows
        cp._orig_cond_rows = orig
        rows = cp._patched_cond_video_rows(fake, payload, "cpu")
        self.assertEqual(rows.shape[0], int((~layout.img_update).sum()))
        mask = torch.cat([refs[0]["keep"].reshape(-1), refs[2]["keep"].reshape(-1)])
        self.assertTrue(torch.equal(rows, dense[mask]))

    def test_keyframes_only_ref_rows_are_removed(self):
        gen = torch.Generator().manual_seed(11)
        kf_latent = torch.randn(1, 24, 2, 8, 10)
        kfs = [{"resolved_frame_index": 0, "latent": kf_latent}]
        refs = [_image_ref(8, 12, _rand_keep(24, gen)), _audio_ref(2)]
        dense = self.build(mm.PackedLayout, _strip(refs), 9, keyframes=kfs)
        pruned = self.build(mm.PackedLayout, _strip(refs), 9, keyframes=kfs)
        cp.filter_layout(pruned, refs)
        removed = 24 - int(refs[0]["keep"].sum())
        self.assertEqual(dense.seq_len - pruned.seq_len, removed)
        seg_d = {k: (b - a) for a, b, k in dense.segments}
        seg_p = {k: (b - a) for a, b, k in pruned.segments}
        for k in ("text", "cond", "ref_audio", "audio", "video"):
            self.assertEqual(seg_d[k], seg_p[k], k)
        self.assertEqual(seg_p["ref_img"], 24 - removed)
        cond_d = next((a, b) for a, b, k in dense.segments if k == "cond")
        cond_p = next((a, b) for a, b, k in pruned.segments if k == "cond")
        self.assertTrue(torch.equal(dense.position_ids[slice(*cond_d)], pruned.position_ids[slice(*cond_p)]))
        vd = next((a, b) for a, b, k in dense.segments if k == "video")
        vp = next((a, b) for a, b, k in pruned.segments if k == "video")
        self.assertTrue(torch.equal(dense.position_ids[slice(*vd)], pruned.position_ids[slice(*vp)]))

        fake = types.SimpleNamespace(patch_size=(1, 2, 2))
        payload = _payload(refs, kfs)
        cp._orig_cond_rows = mm.MiniMaxH3Model._cond_video_rows
        rows = cp._patched_cond_video_rows(fake, payload, "cpu")
        self.assertEqual(rows.shape[0], int((~pruned.img_update).sum()))

    def test_wrapper_installs_once_and_filters(self):
        orig_init, orig_rows = mm.PackedLayout.__init__, mm.MiniMaxH3Model._cond_video_rows
        try:
            cp.install()
            cp.install()
            self.assertTrue(getattr(mm.PackedLayout.__init__, cp.PATCH_MARKER))
            self.assertIs(cp._orig_layout_init, orig_init)
            gen = torch.Generator().manual_seed(5)
            refs = [_image_ref(8, 12, _rand_keep(24, gen))]
            lay = mm.PackedLayout(5, 3, 8, 10, 0, keyframes=None, refs=refs)
            dense = _DenseLayout(orig_init, refs)
            self.assertEqual(lay.seq_len, dense.seq_len - (24 - int(refs[0]["keep"].sum())))
            plain = mm.PackedLayout(5, 3, 8, 10, 0, keyframes=None, refs=_strip(refs))
            self.assertEqual(plain.seq_len, dense.seq_len)
        finally:
            mm.PackedLayout.__init__, mm.MiniMaxH3Model._cond_video_rows = orig_init, orig_rows

    def test_size_mismatch_raises(self):
        refs = [_image_ref(8, 12, torch.ones(5, dtype=torch.bool))]
        layout = self.build(mm.PackedLayout, _strip(refs), 0)
        with self.assertRaises(ValueError):
            cp.filter_layout(layout, refs)


def _DenseLayout(orig_init, refs):
    obj = mm.PackedLayout.__new__(mm.PackedLayout)
    orig_init(obj, 5, 3, 8, 10, 0, keyframes=None, refs=_strip(refs))
    return obj


@unittest.skipUnless(cp.reader_available(), "no H.264 reader (codecopsd_codec or bin/codec_reader.exe)")
class PictureKeep(unittest.TestCase):
    def test_textured_square_kept_flat_background_mostly_dropped(self):
        gen = torch.Generator().manual_seed(1)
        img = torch.full((1, 256, 256, 3), 0.5)
        img[:, 64:128, 64:128, :] = torch.rand(1, 64, 64, 3, generator=gen)
        keep = cp.picture_keep(img)
        self.assertEqual(tuple(keep.shape), (1, 8, 8))
        self.assertTrue(bool(keep[0, 2:4, 2:4].all()))
        outside = keep.clone()
        outside[0, 2:4, 2:4] = False
        self.assertLess(int(outside.sum()), 0.5 * (64 - 4))


if __name__ == "__main__":
    unittest.main()
