"""The JSON answer of a finished job: GET /wrapper/jobs/{job_id}/text.

Some workflows answer with data rather than (or as well as) a file:
image-describe's caption and objects, image-segment's detections. Their graph
tags the node that shows that data -- a PreviewAny, whose text lands in the
job's ComfyUI history like any other output -- with ``_meta.wrapper_result``:
``{"workflow": <name>, ...whatever that workflow's parser needs}``. This module
finds the tagged node in a finished job and hands its text, the tag and the
job's saved images to the workflow's ``text_result`` parser.

Pure: no comfy imports and no I/O, so the lookup is tested on plain dicts. The
result lives exactly as long as the job's history entry, like the job's other
outputs (DELETE /wrapper/jobs/{job_id} removes both).
"""

RESULT_META_KEY = "wrapper_result"


def tag_result(node, workflow, **details):
    """Mark ``node`` as the one whose text is ``workflow``'s JSON answer; the
    details are handed back to the workflow's parser. Returns the node."""
    node["_meta"] = {RESULT_META_KEY: {"workflow": workflow, **details}}
    return node


def view_url(item):
    """The /view URL of one saved output ({filename, subfolder, type})."""
    return "/view?filename={}&subfolder={}&type={}".format(
        item["filename"], item.get("subfolder", ""), item.get("type", "output"))


def saved_images(outputs):
    """/view URLs of every image a job saved to the output folder, in history
    order (a batch keeps its order, so a mask batch reads union first)."""
    return [view_url(item)
            for node_outputs in (outputs or {}).values() if isinstance(node_outputs, dict)
            for item in node_outputs.get("images") or () if isinstance(item, dict) and item.get("filename")
            and item.get("type", "output") == "output"]


def _result_node(graph):
    """(node id, tag) of the graph's tagged result node, or None."""
    for node_id, node in (graph or {}).items():
        meta = node.get("_meta") if isinstance(node, dict) else None
        tag = meta.get(RESULT_META_KEY) if isinstance(meta, dict) else None
        if isinstance(tag, dict):
            return node_id, tag
    return None


def text_result(graph, outputs, workflows):
    """The JSON answer of a finished job, or None when its graph tags no result
    node (every workflow that saves only files). ``graph`` is the job's prompt,
    ``outputs`` its history outputs, ``workflows`` the WORKFLOWS registry."""
    found = _result_node(graph)
    if found is None:
        return None
    node_id, tag = found
    parse = (workflows.get(tag.get("workflow")) or {}).get("text_result")
    if parse is None:
        return None
    texts = ((outputs or {}).get(node_id) or {}).get("text") or ()
    text = texts[0] if texts and isinstance(texts[0], str) else ""
    return parse(text, tag, saved_images(outputs))
