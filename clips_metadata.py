"""Write the Master Extender's final clip list into the metadata ComfyUI embeds in the saved video.

Story mode (rewrite_story + auto_clips) plans and rewrites the clips while the node runs, so the queue-time graph holds an
empty clip list. ComfyUI hands this node the same ``prompt`` and ``extra_pnginfo`` objects it later gives the save nodes
(``DynamicPrompt.get_original_prompt()`` and ``extra_data['extra_pnginfo']``, no copies), so updating them in place makes the
saved MP4 carry the clips that were really rendered. Nothing here raises: a graph that does not look as expected is left alone.
"""

import json
import logging

_LOG = logging.getLogger("MiniMaxH3Master")

WIDGET = "clips_json"


def serialize(clips):
    """Same text the panel stores in the clips_json widget (JSON.stringify(clips, null, 2))."""
    return json.dumps(clips, indent=2, ensure_ascii=False)


def _is_list_json(text):
    try:
        return isinstance(json.loads(text), list)
    except Exception:
        return False


def _update_api_graph(prompt, uid, text):
    node = prompt.get(uid)
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if not isinstance(inputs, dict):
        return None
    old = inputs.get(WIDGET)
    if not isinstance(old, str):  # missing, or a link to another node
        return None
    inputs[WIDGET] = text
    return old


def _update_ui_graph(extra_pnginfo, uid, text, old):
    workflow = extra_pnginfo.get("workflow")
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    if not isinstance(nodes, list):
        return False
    matches = [n for n in nodes if isinstance(n, dict) and str(n.get("id")) == uid]
    if len(matches) != 1:
        return False
    values = matches[0].get("widgets_values")
    if isinstance(values, dict):
        if isinstance(values.get(WIDGET), str):
            values[WIDGET] = text
            return True
        return False
    if isinstance(values, list):
        # The slot is found by the queue-time value (the API graph and the UI graph are saved from the same widget),
        # never by position: frontends list widgets differently.
        slots = [i for i, v in enumerate(values) if isinstance(v, str) and v == old] if old is not None else []
        if len(slots) == 1 and _is_list_json(old):
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
        ui = _update_ui_graph(extra_pnginfo, uid, text, old) if isinstance(extra_pnginfo, dict) else False
        if not (api and ui):
            _LOG.info("Clip metadata: node %s clips_json updated in the API graph: %s, in the workflow: %s", uid, api, ui)
        return (api, ui)
    except Exception as exc:  # metadata is a convenience; a render must never fail on it
        _LOG.warning("Clip metadata not updated: %s", exc)
        return (False, False)
