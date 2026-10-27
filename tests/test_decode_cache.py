import ast
import copy
import logging
import os
import re
from pathlib import Path
import tempfile
import time
import unittest
import uuid

import torch


SOURCE = Path(__file__).resolve().parents[1] / "motion_context_disk.py"


class DecodeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "chain.h3cache"
        self.manifest_path = self.root / "chain.json"
        self.state = {"segments": [{"index": 0, "validated": False}], "final_frame_count": 4}
        self.video_calls = self.audio_calls = 0
        self.rgb = torch.rand(4, 8, 12, 3)
        self.profile = {"codec": "H.264", "crf": 17, "preset": "fast"}
        self.ns = {
            "Path": Path, "os": os, "torch": torch, "uuid": uuid, "re": re, "time": time,
            "_LOG": logging.getLogger("test"),
            "FULL_BATCH_H264_CACHE_CRF": 17,
            "FULL_BATCH_H264_CACHE_PRESET": "fast",
            "FULL_BATCH_H264_CACHE_PROFILE": "preview",
            "normalize_full_batch_export_profile": lambda p: p,
            "_normalize_color_adjustment": lambda p: p,
            "_load_manifest_from_paths": lambda *a: copy.deepcopy(self.state),
            "_write_json_atomic": self.write_manifest, "_replace_with_retry": os.replace,
            "_ensure_cache_root": lambda: self.root,
            "_find_ffmpeg": lambda: "ffmpeg",
            "_render_one_final_video_segment": self.decode_video,
            "_render_one_final_audio_segment": self.decode_audio,
            "_load_cached_decoded_audio": lambda data, desc: desc.get("decoded_audio"),
            "_encode_corrected_segment_video_mp4": self.encode_video,
            "_cache_candidate_render": self.cache_candidate,
            "_ensure_ref2va_final_segment_cache": self.ensure_final,
            "_ref2va_final_segment_cache_path": lambda *a: self.root / "final.mp4",
            "_final_segment_cache_meta_matches": lambda desc, profile, *a: desc.get("profile") == profile,
            "_safe_name": str,
            "_validated_prefix_count": lambda segments: sum(s["validated"] for s in segments),
            "_upgrade_cached_audio_gain_chain": lambda d, p, m, *a: m,
            "_sync_committed_preview": lambda d, p, m, *a: (m, self.root / "committed.mp4", self.root / "committed-video.mp4"),
            "_copy_blob_to_file": lambda d, s, p: Path(p).write_bytes(b"video"),
            "_reserve_preview_temp_path": lambda *a: self.root / "preview.mp4",
            "_assemble_progressive_preview": lambda *a: None,
        }
        names = {"_guide_frame_path", "_load_guide_frame", "_cache_guide_frame",
                 "cache_full_batch_ref2va_segment", "_export_live_candidate_preview", "_truncate_chain"}
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        # Run the actual cache orchestration without importing the ComfyUI server.
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), self.ns)

    def write_manifest(self, path, manifest):
        self.state = copy.deepcopy(manifest)

    def decode_video(self, *args, **kwargs):
        self.video_calls += 1
        return self.rgb.clone(), 0

    def decode_audio(self, *args, **kwargs):
        self.audio_calls += 1
        return {"sample_rate": 48000, "waveform": torch.zeros(1, 2, 8)}

    def encode_video(self, ffmpeg, video, fps, path, token, **kwargs):
        self.assertTrue(torch.equal(video, self.rgb))
        Path(path).write_bytes(b"video")

    def cache_candidate(self, data, path, manifest, index, rendered_mp4=None,
                        rendered_audio=None, seam_shift=None, video_profile=None):
        manifest = copy.deepcopy(manifest)
        desc = manifest["segments"][index]
        if rendered_mp4 is not None:
            desc["decoded_mp4_blob"] = {"offset": 0}
        if rendered_audio is not None:
            desc["decoded_audio"] = rendered_audio
        if seam_shift is not None:
            desc["decoded_seam_shift"] = seam_shift
        self.write_manifest(path, manifest)
        return manifest, desc

    def ensure_final(self, data, path, manifest, index, vae, fps, ffmpeg, profile,
                     decoded_video=None, **kwargs):
        manifest = copy.deepcopy(manifest)
        desc = manifest["segments"][index]
        decoded_now = False
        if desc.get("profile") != profile:
            if decoded_video is None:
                decoded_video, _ = self.decode_video()
                decoded_now = True
            self.assertTrue(torch.equal(decoded_video, self.rgb))
            desc["profile"] = profile
        self.write_manifest(path, manifest)
        return manifest, self.root / "final.mp4", decoded_now

    def prepare(self, index=0):
        return self.ns["cache_full_batch_ref2va_segment"](
            self.data, self.manifest_path, index, object(), object(), 24,
            export_profile=self.profile,
        )

    def preview(self):
        return self.ns["_export_live_candidate_preview"](
            self.data, self.manifest_path, self.state, self.state["segments"],
            object(), object(), 24, "ffmpeg", "test", export_profile=self.profile,
        )

    def test_generation_preview_and_resume_reuse(self):
        self.prepare()
        self.preview()
        self.preview()
        self.assertEqual((self.video_calls, self.audio_calls), (1, 1))
        guide = self.ns["_load_guide_frame"](self.data, self.state["segments"][0])
        self.assertTrue(torch.equal(guide, self.rgb[-1:]))
        self.assertEqual(guide.untyped_storage().nbytes(), guide.numel() * guide.element_size())
        self.state["segments"][0]["validated"] = True
        self.state["segments"].append({"index": 1, "validated": False})
        self.prepare(1)
        self.preview()
        self.assertEqual((self.video_calls, self.audio_calls), (2, 2))

    def test_full_batch_reuses_prepared_segments(self):
        self.prepare()
        self.state["segments"].append({"index": 1, "validated": False})
        self.prepare(1)
        self.prepare(0)
        self.prepare(1)
        self.assertEqual((self.video_calls, self.audio_calls), (2, 2))

    def test_missing_cache_and_changed_profile_repair(self):
        self.prepare()
        del self.state["segments"][0]["decoded_audio"]
        self.preview()
        self.assertEqual((self.video_calls, self.audio_calls), (1, 2))
        self.profile = {**self.profile, "crf": 20}
        self.preview()
        self.assertEqual((self.video_calls, self.audio_calls), (2, 2))
        del self.state["segments"][0]["decoded_mp4_blob"]
        self.preview()
        self.assertEqual((self.video_calls, self.audio_calls), (3, 2))

    def test_stale_guide_is_not_used_for_replacement(self):
        self.prepare()
        self.assertIsNone(self.ns["_load_guide_frame"](self.data, {"index": 0}))
        self.ns["_guide_frame_path"](self.data, 0).unlink()
        self.assertIsNone(self.ns["_load_guide_frame"](self.data, self.state["segments"][0]))

    def test_truncation_removes_only_replaced_guide_suffix(self):
        self.prepare()
        self.state["segments"].append({"index": 1, "validated": False})
        self.prepare(1)
        first = self.ns["_guide_frame_path"](self.data, 0)
        second = self.ns["_guide_frame_path"](self.data, 1)
        self.data.write_bytes(b"0123456789")
        self.ns.update({
            "_DATA_START": 0,
            "_segment_end": lambda desc: 5,
            "_final_frame_count": lambda segments: len(segments) * 4,
            "_decoded_preview_cache_path": lambda data: self.root / "preview.mp4",
            "_decoded_preview_video_cache_path": lambda data: self.root / "preview-video.mp4",
            "_ensure_data_file": lambda data: None,
            "_truncate_decoded_audio_cache": lambda *a: None,
        })
        self.ns["_truncate_chain"](self.data, self.manifest_path, self.state, 1)
        self.assertTrue(first.exists())
        self.assertFalse(second.exists())
        self.assertEqual(len(self.state["segments"]), 1)
        self.assertEqual(self.data.stat().st_size, 5)


if __name__ == "__main__":
    unittest.main()
