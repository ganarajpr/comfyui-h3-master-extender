"""Read the prompt back out of a rendered H3 video (or image).

ComfyUI's SaveVideo, VHS and this pack's Final Decode all write the API prompt
graph into the MP4 container's ``prompt`` tag (and sometimes the UI graph into
``workflow``). Generic metadata readers look for CLIPTextEncode nodes and find
nothing in a Master Extender graph, where the prompts live inside the
``clips_json`` input. This node understands that layout and hands the clip
prompts back as STRING outputs that plug straight into ``clip_prompt_N``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from typing import Any

import folder_paths

log = logging.getLogger("MiniMaxH3Master.metadata")

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".m4v")
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg")
MAX_CLIP_OUTPUTS = 4


def _ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def _ffmpeg_path() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def read_container_tags(path: str) -> dict[str, str]:
    """Format-level tags of a media file, via ffprobe or ffmpeg's ffmetadata dump."""
    probe = _ffprobe_path()
    if probe:
        try:
            cp = subprocess.run(
                [probe, "-v", "quiet", "-print_format", "json", "-show_format", path],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=60, check=False, creationflags=_creationflags(),
            )
            if cp.returncode == 0 and cp.stdout.strip():
                return dict(json.loads(cp.stdout).get("format", {}).get("tags", {}) or {})
        except Exception as exc:  # fall through to ffmpeg
            log.warning("ffprobe failed on %s: %s", path, exc)
    ffmpeg = _ffmpeg_path()
    if ffmpeg:
        try:
            cp = subprocess.run(
                [ffmpeg, "-v", "quiet", "-i", path, "-f", "ffmetadata", "-"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=60, check=False, creationflags=_creationflags(),
            )
            tags: dict[str, str] = {}
            for line in cp.stdout.splitlines():
                if "=" in line and not line.startswith(";"):
                    k, v = line.split("=", 1)
                    # ffmetadata escapes = ; # \ and newlines with a backslash
                    v = v.replace("\\\n", "\n").replace("\\=", "=").replace("\\;", ";").replace("\\#", "#").replace("\\\\", "\\")
                    tags[k.strip()] = v
            return tags
        except Exception as exc:
            log.warning("ffmpeg metadata dump failed on %s: %s", path, exc)
    return {}


def read_image_tags(path: str) -> dict[str, str]:
    try:
        from PIL import Image
        with Image.open(path) as img:
            info = dict(img.info or {})
        return {k: v for k, v in info.items() if isinstance(v, str)}
    except Exception as exc:
        log.warning("PIL could not read %s: %s", path, exc)
        return {}


def _load_json(text: str | None) -> Any:
    if not text or not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith("Prompt:"):
        t = t[7:].strip()
    try:
        return json.loads(t)
    except Exception:
        return None


def _clips_from_api_graph(graph: dict) -> tuple[list[dict], dict | None, str | None]:
    """Return (clips, master_inputs, node_id) from an API-format prompt graph."""
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")).startswith("MiniMaxH3MasterExtender"):
            inputs = node.get("inputs", {}) or {}
            clips = _load_json(inputs.get("clips_json")) if isinstance(inputs.get("clips_json"), str) else inputs.get("clips_json")
            if isinstance(clips, list):
                return [c for c in clips if isinstance(c, dict)], inputs, str(node_id)
    return [], None, None


def _clips_from_ui_graph(graph: dict) -> list[dict]:
    """UI-format workflow: the clips_json string sits in widgets_values."""
    for node in graph.get("nodes", []) or []:
        if str(node.get("type", "")).startswith("MiniMaxH3MasterExtender"):
            named = node.get("widgets_values_named") or {}
            candidates = [named.get("clips_json")] if isinstance(named, dict) else []
            candidates += [v for v in (node.get("widgets_values") or []) if isinstance(v, str) and v.lstrip().startswith("[")]
            for cand in candidates:
                clips = _load_json(cand)
                if isinstance(clips, list) and clips and isinstance(clips[0], dict) and "prompt" in clips[0]:
                    return [c for c in clips if isinstance(c, dict)]
    return []


def _generic_prompts(graph: dict) -> list[str]:
    """Fallback for non-Master graphs: text/prompt inputs of encode-style nodes."""
    found: list[str] = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        ct = str(node.get("class_type", ""))
        inputs = node.get("inputs", {}) or {}
        for key in ("text", "prompt", "positive", "positive_prompt", "string", "value"):
            v = inputs.get(key)
            if isinstance(v, str) and len(v.strip()) > 20 and ("TextEncode" in ct or "Prompt" in ct or "String" in ct or "Primitive" in ct or key == "prompt"):
                if v not in found:
                    found.append(v)
    return found


def _resolve_video_path(video: str, video_path: str) -> str:
    if video_path and video_path.strip():
        p = os.path.expandvars(os.path.expanduser(video_path.strip().strip('"')))
        if not os.path.isabs(p):
            for base in (folder_paths.get_output_directory(), folder_paths.get_input_directory()):
                cand = os.path.join(base, p)
                if os.path.isfile(cand):
                    return cand
        return p
    return folder_paths.get_annotated_filepath(video)


def _settings_summary(inputs: dict | None, clips: list[dict]) -> str:
    lines: list[str] = []
    if inputs:
        accel = inputs.get("accel_mode", "HyperFlow 8-step")
        steps = inputs.get("pdd_nfe", "?")
        if str(accel).lower().startswith("turbo"):
            lines.append(f"Engine: Turbo LoRA {inputs.get('turbo_lora', '')} x{inputs.get('turbo_lora_strength', 1.0)} · {steps} steps · {inputs.get('turbo_sampler', '')} / {inputs.get('turbo_scheduler', '')}")
        else:
            refit = "curve refit on" if inputs.get("hyperflow_curve_refit", True) else "curve refit off"
            label = "HyperFlow · dual sampling" if "dual sampling" in str(accel) else "HyperFlow 8-step"
            lines.append(f"Engine: {label} {inputs.get('hyperflow_file', '')} x{inputs.get('hyperflow_strength', 1.0)} · {refit} · {inputs.get('hyperflow_lora_mode', 'bypass')} · euler{' · pass 1 stops at step ' + str(inputs.get('pass1_steps')) + ' of 8' if int(inputs.get('pass1_steps', 0) or 0) > 0 and 'dual sampling' not in str(accel) else ''}")
        lines.append(f"Quality: draft {inputs.get('pass1_resolution', '?')} → refine {inputs.get('pass2_resolution', '?')} · denoise {inputs.get('pass2_denoise', '?')}{' · sigmas ' + str(inputs.get('pass2_sigmas')).strip() if str(inputs.get('pass2_sigmas') or '').strip() else ''}{' · pass-1 audio' if inputs.get('pass2_audio') == 'pass 1' else ''} · upscaler {inputs.get('upscaler_model', '?')}")
        lines.append(f"Continuity: motion {inputs.get('context_length', '?')} f · audio {inputs.get('audio_context_length', '?')} f · identity {inputs.get('identity_continuity', '?')}")
        sla = inputs.get("sla_enabled", False)
        lines.append(f"Performance: {inputs.get('attention_backend', '?')} · SLA {'on ' + str(inputs.get('sla_sparsity', '')) + (' (pass 2 ' + str(inputs.get('pass2_sla_sparsity')) + ')' if inputs.get('pass2_sla_sparsity') else '') + ' ' + str(inputs.get('sparse_method', '')) + ' start ' + str(inputs.get('sparse_start_percent', 'auto')) + ' dense ' + str(inputs.get('sparse_dense_blocks', 'auto')) + ' sink ' + str(inputs.get('sparse_sink', 'auto')) if sla else 'off'} · chunk {inputs.get('pass2_chunk_frames', '?')}/{inputs.get('pass2_chunk_overlap', '?')} ({inputs.get('pass2_chunk_when', 'above 720p')}) · run {inputs.get('run_mode', '?')}")
        refs = _load_json(inputs.get("refs_json")) if isinstance(inputs.get("refs_json"), str) else inputs.get("refs_json")
        if isinstance(refs, dict):
            imgs = [r for r in refs.get("images", []) if r]
            if imgs:
                lines.append("References: " + ", ".join(str(r) for r in imgs))
    for i, c in enumerate(clips, 1):
        lines.append(f"Clip {i}: {c.get('duration', '?')} s · seed {c.get('seed', '?')} ({c.get('seed_mode', '?')})" + (" · validated" if c.get("validated") else ""))
    return "\n".join(lines) if lines else "No Master Extender settings found in this file."


class MiniMaxH3PromptFromVideo:
    """Extract clip prompts and settings from a video or image rendered by ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files: list[str] = []
        try:
            for f in sorted(os.listdir(input_dir)):
                if f.lower().endswith(VIDEO_EXTS + IMAGE_EXTS) and os.path.isfile(os.path.join(input_dir, f)):
                    files.append(f)
        except Exception:
            pass
        return {
            "required": {
                "video": (files or [""], {"video_upload": True, "tooltip": "A video or image from the input folder. Ignored when video_path is set."}),
                "clip_index": ("INT", {"default": 1, "min": 1, "max": 99, "step": 1, "tooltip": "Which clip's prompt goes to the `prompt` output."}),
            },
            "optional": {
                "video_path": ("STRING", {"default": "", "multiline": False,
                                          "tooltip": "Absolute path, or a path relative to the output/input folder, e.g. video\\MiniMax_H3_Master_Turbo_00043_.mp4. Overrides `video`."}),
            },
        }

    RETURN_TYPES = ("STRING",) * (2 + MAX_CLIP_OUTPUTS) + ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompt",) + tuple(f"clip_{i}_prompt" for i in range(1, MAX_CLIP_OUTPUTS + 1)) + ("clips_json", "settings", "raw_prompt_json", "seed", "clip_count")
    FUNCTION = "extract"
    CATEGORY = "MiniMax H3 Master"
    DESCRIPTION = ("Reads the prompt graph a ComfyUI render embedded in its MP4/PNG and returns the Master Extender clip "
                   "prompts (wire them into clip_prompt_N), the whole clips_json, a readable settings summary, and the raw graph.")

    @classmethod
    def IS_CHANGED(cls, video, clip_index, video_path=""):
        try:
            path = _resolve_video_path(video, video_path)
            st = os.stat(path)
            return hashlib.sha256(f"{path}|{st.st_size}|{st.st_mtime_ns}|{clip_index}".encode()).hexdigest()
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, video, clip_index, video_path=""):
        if video_path and video_path.strip():
            return True
        if not video or not folder_paths.exists_annotated_filepath(video):
            return f"File not found: {video}"
        return True

    def extract(self, video, clip_index, video_path=""):
        path = _resolve_video_path(video, video_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"MiniMax H3 Prompt From Video: file not found: {path}")

        tags = read_image_tags(path) if path.lower().endswith(IMAGE_EXTS) else read_container_tags(path)
        raw_prompt = tags.get("prompt") or tags.get("comment") or ""
        api_graph = _load_json(raw_prompt)
        ui_graph = _load_json(tags.get("workflow"))

        clips: list[dict] = []
        master_inputs = None
        if isinstance(api_graph, dict):
            clips, master_inputs, _ = _clips_from_api_graph(api_graph)
        if not clips and isinstance(ui_graph, dict):
            clips = _clips_from_ui_graph(ui_graph)
        if not clips and isinstance(api_graph, dict):
            clips = [{"id": i, "title": f"Prompt {i + 1}", "prompt": p} for i, p in enumerate(_generic_prompts(api_graph))]

        if not clips:
            available = ", ".join(sorted(tags)) or "none"
            raise ValueError(
                f"No ComfyUI prompt metadata with clip prompts found in {os.path.basename(path)}. "
                f"Container tags present: {available}. Files re-encoded by other tools lose these tags."
            )

        prompts = [str(c.get("prompt", "") or "") for c in clips]
        idx = max(1, min(int(clip_index), len(prompts))) - 1
        clip_outputs = tuple(prompts[i] if i < len(prompts) else "" for i in range(MAX_CLIP_OUTPUTS))
        seed = 0
        try:
            seed = int(clips[idx].get("seed", 0) or 0)
        except Exception:
            pass
        return (
            prompts[idx],
            *clip_outputs,
            json.dumps(clips, indent=2, ensure_ascii=False),
            _settings_summary(master_inputs, clips),
            json.dumps(api_graph, indent=2, ensure_ascii=False) if isinstance(api_graph, dict) else raw_prompt,
            seed,
            len(clips),
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptFromVideo": MiniMaxH3PromptFromVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptFromVideo": "MiniMax H3 Prompt From Video",
}
