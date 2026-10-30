"""Write the Master Extender's final clip list (and the Shot Writer's output) into the metadata ComfyUI embeds in the saved video.

Story mode (rewrite_story + auto_clips) plans and rewrites the clips while the node runs, so the queue-time graph holds an
empty clip list. ComfyUI hands this node the same ``prompt`` and ``extra_pnginfo`` objects it later gives the save nodes
(``DynamicPrompt.get_original_prompt()`` and ``extra_data['extra_pnginfo']``, no copies), so updating them in place makes the
saved MP4 carry the clips that were really rendered. Nothing here raises: a graph that does not look as expected is left alone.
"""

import json
import logging
import time

_LOG = logging.getLogger("MiniMaxH3Master")

WIDGET = "clips_json"
WRITER_WIDGET = "last_output"
EXTENDER_CLASS = "MiniMaxH3MasterExtender"
WRITER_CLASS = "H3ShotWriter"
PREVIEW_CLASS = "PreviewAny"
PREVIEW_INPUT = "source"


def serialize(clips):
    """Same text the panel stores in the clips_json widget (JSON.stringify(clips, null, 2))."""
    return json.dumps(clips, indent=2, ensure_ascii=False)


def _is_list_json(text):
    try:
        return isinstance(json.loads(text), list)
    except Exception:
        return False


def _is_object_json(text):
    try:
        return isinstance(json.loads(text), dict)
    except Exception:
        return False


def _update_api_graph(prompt, uid, text, widget=WIDGET):
    node = prompt.get(uid)
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if not isinstance(inputs, dict):
        return None
    old = inputs.get(widget)
    if not isinstance(old, str):  # missing, or a link to another node
        return None
    inputs[widget] = text
    return old


def _update_ui_graph(extra_pnginfo, uid, text, old, widget=WIDGET, valid=_is_list_json, class_type=None, last=False):
    """``class_type``: when no UI node has this id (a graph converted from an API prompt renumbers its nodes), the one node of that type.
    ``last``: the widget is the node's last one (a hidden widget appended last), so its slot is the last value even when the queue-time
    value is empty and cannot pick a slot by itself."""
    workflow = extra_pnginfo.get("workflow")
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    if not isinstance(nodes, list):
        return False
    matches = [n for n in nodes if isinstance(n, dict) and str(n.get("id")) == uid]
    if not matches and class_type:
        matches = [n for n in nodes if isinstance(n, dict) and n.get("type") == class_type]
    if len(matches) != 1:
        return False
    values = matches[0].get("widgets_values")
    if isinstance(values, dict):
        if isinstance(values.get(widget), str):
            values[widget] = text
            return True
        return False
    if isinstance(values, list):
        if last and values and isinstance(values[-1], str) and (values[-1] == old or (not values[-1].strip() and not (old or "").strip())):
            values[-1] = text
            return True
        # The slot is found by the queue-time value (the API graph and the UI graph are saved from the same widget),
        # never by position: frontends list widgets differently.
        slots = [i for i, v in enumerate(values) if isinstance(v, str) and v == old] if old is not None else []
        if len(slots) == 1 and valid(old):
            values[slots[0]] = text
            return True
    return False


def embed_clips(unique_id, prompt, extra_pnginfo, clips):
    """Replace this node's clips_json in the API graph and in the UI workflow. Returns what was updated: (api, ui)."""
    try:
        if unique_id is None or not isinstance(clips, list):
            return (False, False)
        uid = str(unique_id)
        text = serialize(clips)
        old = _update_api_graph(prompt, uid, text) if isinstance(prompt, dict) else None
        api = old is not None
        ui = _update_ui_graph(extra_pnginfo, uid, text, old, class_type=EXTENDER_CLASS) if isinstance(extra_pnginfo, dict) else False
        if not (api and ui):
            _LOG.info("Clip metadata: node %s clips_json updated in the API graph: %s, in the workflow: %s", uid, api, ui)
        return (api, ui)
    except Exception as exc:  # metadata is a convenience; a render must never fail on it
        _LOG.warning("Clip metadata not updated: %s", exc)
        return (False, False)


def embed_writer_output(unique_id, prompt, extra_pnginfo, clips, story, model=""):
    """Save the Shot Writer's output in the video: ``last_output`` (its hidden widget) gets ``{clips, story, written_at, model}`` in the API graph and in the UI workflow.

    A ``PreviewAny`` node fed by this node's story output (output 1) gets the story as its saved text in the UI workflow, so reopening the video shows it.
    Returns (api, ui, previews)."""
    try:
        if unique_id is None:
            return (False, False, 0)
        uid = str(unique_id)
        payload = {"clips": clips, "story": story, "written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model}
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        old = _update_api_graph(prompt, uid, text, WRITER_WIDGET) if isinstance(prompt, dict) else None
        api = old is not None
        ui = _update_ui_graph(extra_pnginfo, uid, text, old, WRITER_WIDGET, _is_object_json, class_type=WRITER_CLASS, last=True) if isinstance(extra_pnginfo, dict) else False
        previews = _update_previews(prompt, extra_pnginfo, uid, story) if isinstance(prompt, dict) and isinstance(extra_pnginfo, dict) else 0
        if not (api and ui):
            _LOG.info("Writer metadata: node %s last_output updated in the API graph: %s, in the workflow: %s", uid, api, ui)
        return (api, ui, previews)
    except Exception as exc:  # metadata is a convenience; a render must never fail on it
        _LOG.warning("Writer metadata not updated: %s", exc)
        return (False, False, 0)


def _update_previews(prompt, extra_pnginfo, uid, story) -> int:
    """Set the saved text of every Preview-as-Text node wired to this node's story output (output 1). The API graph holds only the link, so only the workflow changes."""
    ids = {str(node_id) for node_id, node in prompt.items() if isinstance(node, dict) and node.get("class_type") == PREVIEW_CLASS
           and isinstance((node.get("inputs") or {}).get(PREVIEW_INPUT), list) and [str(v) for v in node["inputs"][PREVIEW_INPUT]] == [uid, "1"]}
    workflow = extra_pnginfo.get("workflow")
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    count = 0
    for node in nodes or []:
        if isinstance(node, dict) and str(node.get("id")) in ids:
            node["widgets_values"] = [story]
            count += 1
    return count
