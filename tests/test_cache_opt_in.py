"""Cache clips is opt-in: OFF chains through a per-node scratch folder wiped at run start, ON is the persistent chain as before."""
import ast
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "master_node.py"


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def load_disk():
    for n in ("comfy", "comfy.nested_tensor", "comfy.utils"):
        _stub(n)
    pkg = _stub("mcd_cache_test")
    pkg.__path__ = [str(ROOT)]
    _stub("mcd_cache_test.vram_release", register=lambda *a, **k: None)
    ns = {"torch": torch, "math": math}
    tree = ast.parse((ROOT / "motion_context_ram.py").read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("FPS", "FRAME_PER_TOKEN", "AUDIO_HZ") for t in n.targets))]
    exec(compile(ast.Module(body=body, type_ignores=[]), "ram", "exec"), ns)
    _stub("mcd_cache_test.motion_context_ram", **{k: v for k, v in ns.items() if not k.startswith("__")})
    spec = importlib.util.spec_from_file_location("mcd_cache_test.motion_context_disk", ROOT / "motion_context_disk.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mcd_cache_test.motion_context_disk"] = mod
    spec.loader.exec_module(mod)
    return mod


DISK = load_disk()


def clip(seed, t=7):
    g = torch.Generator().manual_seed(seed)
    video = torch.rand(1, 4, t, 2, 3, generator=g)
    audio = torch.rand(1, 2, 2, int(round((5 + 17 * ((t - 2) // 5)) / 24 * 40)), generator=g)
    return {"samples": (video, audio)}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.saved = (DISK._CACHE_ROOT, DISK._SCRATCH_ROOT, DISK._LEGACY_MIGRATED)
        DISK._CACHE_ROOT, DISK._SCRATCH_ROOT, DISK._LEGACY_MIGRATED = root / "cache", root / "scratch", True
        self.addCleanup(lambda: (setattr(DISK, "_CACHE_ROOT", self.saved[0]), setattr(DISK, "_SCRATCH_ROOT", self.saved[1]),
                                 setattr(DISK, "_LEGACY_MIGRATED", self.saved[2])))
        self.join = DISK.MiniMaxH3MotionContextDiskJoin()

    def render_two_clips(self, owner):
        """The extender's per-run sequence: first manifest, clip 1, clip 2 chained through the returned handle."""
        DISK._manifest_for_first(owner, 24.0)
        r0 = self.join.join(samples=clip(1), trim_frames=0, run_mode="full_batch", fps=24.0, previous_cache=None, unique_id=owner)
        r1 = self.join.join(samples=clip(2), trim_frames=0, run_mode="full_batch", fps=24.0, previous_cache=r0[0], unique_id=owner)
        return r1[0]


class ScratchPaths(Base):
    def test_scratch_owner_resolves_to_its_node_folder_and_others_stay_persistent(self):
        owner = DISK.scratch_owner("m_ab12")
        data, manifest = DISK._chain_paths(owner)
        draft_data, _ = DISK._chain_paths(owner + "_draft")
        self.assertEqual(data.parent, DISK._SCRATCH_ROOT / "m_ab12")
        self.assertEqual(draft_data.parent, data.parent)
        persistent, _ = DISK._chain_paths("master_v2_c_123")
        self.assertEqual(persistent.parent, DISK._CACHE_ROOT)

    def test_wipe_keeps_only_the_other_nodes_of_the_running_prompt(self):
        for name in ("a", "b", "old_job"):
            (DISK._SCRATCH_ROOT / name).mkdir(parents=True)
            (DISK._SCRATCH_ROOT / name / "chain.h3cache").write_bytes(b"x")
        removed = DISK.wipe_scratch(["b", "other"])
        self.assertEqual(sorted(removed), ["a", "old_job"])
        self.assertEqual([p.name for p in DISK._SCRATCH_ROOT.iterdir()], ["b"])
        DISK.wipe_scratch([])
        self.assertEqual(list(DISK._SCRATCH_ROOT.iterdir()), [])

    def test_wipe_without_a_scratch_root_is_a_no_op(self):
        self.assertEqual(DISK.wipe_scratch(["x"]), [])


class OffAndOn(Base):
    def test_off_never_reads_an_existing_persistent_chain_and_leaves_it_alone(self):
        persistent = self.render_two_clips("master_v2_c_same")
        data = Path(persistent["data_path"])
        before = (data.stat().st_size, data.stat().st_mtime_ns)
        self.assertEqual(len(DISK._load_manifest(persistent)[2]["segments"]), 2)

        scratch = DISK.scratch_owner("m_job1")
        DISK.wipe_scratch([])
        _, _, manifest = DISK._manifest_for_first(scratch, 24.0)
        self.assertEqual(manifest["segments"], [], "nothing is reused from the persistent chain")
        self.assertEqual((data.stat().st_size, data.stat().st_mtime_ns), before)

    def test_on_reuses_the_persistent_chain_as_before(self):
        self.render_two_clips("master_v2_c_same")
        _, _, manifest = DISK._manifest_for_first("master_v2_c_same", 24.0)
        self.assertEqual(len(manifest["segments"]), 2)

    def test_off_run_wipes_the_previous_run_and_final_decode_reads_the_chain_of_a_two_clip_run(self):
        owner = DISK.scratch_owner("m_job1")
        first = self.render_two_clips(owner)
        self.assertEqual(Path(first["data_path"]).parent.resolve(), (DISK._SCRATCH_ROOT / "m_job1").resolve())
        # what MiniMaxH3MasterFinalDecode does first with the `cache` socket
        data_path, manifest_path, manifest = DISK._load_manifest(first)
        self.assertEqual([s["index"] for s in manifest["segments"]], [0, 1])
        self.assertEqual(manifest["final_frame_count"], 22 + 22)

        DISK.wipe_scratch([])
        self.assertFalse(Path(first["data_path"]).exists())
        _, _, fresh = DISK._manifest_for_first(owner, 24.0)
        self.assertEqual(fresh["segments"], [], "the next run starts from clip 1")
        second = self.render_two_clips(owner)
        self.assertEqual(len(DISK._load_manifest(second)[2]["segments"]), 2)
        self.assertEqual(len(list(DISK._SCRATCH_ROOT.iterdir())), 1, "one scratch folder per node, not one per run")


class NodeWiring(unittest.TestCase):
    text = NODE.read_text(encoding="utf-8")

    def test_input_is_boolean_default_off_and_appended_before_the_sockets(self):
        self.assertRegex(self.text, r'"cache_clips": \("BOOLEAN", \{"default": False')
        i = self.text.index('"cache_clips": (')
        self.assertLess(self.text.index('"diversity_steps": ('), i)
        self.assertLess(i, self.text.index('"time_gate_json": ('))
        self.assertLess(i, self.text.index('"clips_in": ('))

    def test_missing_or_non_boolean_value_is_off(self):
        self.assertIn('cache_clips = kwargs.get("cache_clips") is True', self.text)
        self.assertNotIn('kwargs.get("cache_clips", True)', self.text)

    def test_off_uses_the_scratch_owner_and_on_the_persistent_one(self):
        i = self.text.index("cache_clips = kwargs.get")
        block = self.text[i:i + 2200]
        self.assertIn("if cache_clips:\n            cache_owner = f\"master_v2_{chain_key}\"", block)
        self.assertIn("cache_owner = motion_context_disk.scratch_owner(unique_id)", block)
        self.assertIn("draft_owner = f\"{cache_owner}_draft\"", block)

    def test_scratch_is_wiped_before_any_chain_is_opened_and_spares_other_nodes_of_the_prompt(self):
        wipe = self.text.index("motion_context_disk.wipe_scratch(extender_ids - {str(unique_id)})")
        self.assertLess(wipe, self.text.index("data_path, manifest_path, manifest = _manifest_for_first(cache_owner, FPS)"))
        self.assertIn('v.get("class_type") == "MiniMaxH3MasterExtender"', self.text)


if __name__ == "__main__":
    unittest.main()
