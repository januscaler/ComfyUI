"""Requests and answers of the image-segment workflow (SAM 3).

SAM3_Detect (comfy_extras/nodes_sam3.py) selects what a phrase, boxes or points
name in a still. These pure functions -- free of comfy imports, like
workflows.py -- turn the form fields into its inputs and its detections into
the JSON GET /wrapper/jobs/{job_id}/text answers with. Every coordinate is in
pixels of the upright image, i.e. after its EXIF orientation is applied, which
is what LoadImage hands the node and what the masks are sized to.
"""

import json
import math
import os

SEGMENT_MODES = ("prompt", "boxes", "points")
SEGMENT_DEFAULT_THRESHOLD = 0.5
SEGMENT_DEFAULT_MAX_OBJECTS = 8
# Detections kept per phrase. Each is refined by its own decoder passes and,
# with individual_masks, saved as its own PNG.
SEGMENT_MAX_OBJECTS = 32
# Comma-separated phrases each run their own detector pass.
SEGMENT_MAX_PHRASES = 8
SEGMENT_MAX_POINTS = 64
# Every mask is held in host RAM as float32 and again as an RGB float32 image
# for the PNG encoder, ~16 bytes per pixel per mask on top of the decoded
# still. The default (5e8 pixel-masks, ~8 GB) is e.g. a 12 MP photo with 33
# masks or a 48 MP one with 9; set COMFY_IMAGE_SEGMENT_MAX_PIXEL_MASKS to change it.
SEGMENT_MAX_PIXEL_MASKS = max(1, int(float(
    os.environ.get("COMFY_IMAGE_SEGMENT_MAX_PIXEL_MASKS", "500000000"))))

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def parse_flag(raw, default):
    """A form boolean ('true'/'false', '1'/'0', 'yes'/'no', 'on'/'off')."""
    value = str(raw).strip().lower() if raw is not None else ""
    if not value:
        return default
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"'{raw}' is not a boolean; send true or false.")


def _bounded(raw, default, low, high, field, kind):
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = kind(str(raw).strip())
    except ValueError:
        value = None
    if value is None or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{field} must be a {'whole ' if kind is int else ''}number from {low} to {high}.")
    return value


def sam3_phrases(prompt):
    """The comma-separated phrases of ``prompt``, cleaned of what SAM 3's
    tokenizer reads as syntax: parentheses (it drops them) and ':' (a trailing
    ':N' sets a phrase's detection cap, which max_objects owns here)."""
    cleaned = prompt.replace("(", " ").replace(")", " ").replace(":", " ")
    return [" ".join(part.split()) for part in cleaned.split(",") if part.split()]


def sam3_text_prompt(phrases, max_objects):
    """SAM 3's text input for ``phrases``, e.g. 'sky:8, clouds:8'. Its tokenizer
    keeps a single detection per phrase unless the phrase ends in ':N'
    (comfy/text_encoders/sam3_clip.py)."""
    return ", ".join(f"{phrase}:{max_objects}" for phrase in phrases)


def _json_list(raw, field, example):
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        value = None
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a JSON list, e.g. {example}.")
    return value


def _coords(values, field):
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError(f"{field} must hold numbers.")
    return [float(v) for v in values]


def parse_boxes(raw, width, height):
    """``[[x, y, w, h], ...]`` (or a single ``[x, y, w, h]``) in pixels ->
    SAM3_Detect's bboxes ({x, y, width, height} dicts). A box reaching past
    the image is clipped to it; one with no positive area inside it is refused."""
    example = "[[120, 40, 300, 260]]"
    value = _json_list(raw, "boxes", example)
    boxes = value if isinstance(value[0], list) else [value]
    if len(boxes) > SEGMENT_MAX_OBJECTS:
        raise ValueError(f"boxes takes at most {SEGMENT_MAX_OBJECTS} boxes.")
    parsed = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"every box is [x, y, w, h] in pixels, e.g. {example}.")
        x, y, w, h = _coords(box, "boxes")
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(width), x + w), min(float(height), y + h)
        if w <= 0 or h <= 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"box {box} has no area inside the {width}x{height} image.")
        parsed.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})
    return parsed


def parse_points(raw, width, height, field):
    """``[[x, y], ...]`` or ``[{"x": .., "y": ..}, ...]`` (or a single point)
    in pixels -> SAM3_Detect's {x, y} points. A point outside the image is
    refused: it is almost always a coordinate in the wrong space."""
    example = "[[512, 300]]"
    value = _json_list(raw, field, example)
    points = value if isinstance(value[0], (list, dict)) else [value]
    if len(points) > SEGMENT_MAX_POINTS:
        raise ValueError(f"{field} takes at most {SEGMENT_MAX_POINTS} points.")
    parsed = []
    for point in points:
        if isinstance(point, dict):
            point = [point.get("x"), point.get("y")]
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{field} holds [x, y] pixel points, e.g. {example}.")
        x, y = _coords(point, field)
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"{field} point ({x:g}, {y:g}) lies outside the {width}x{height} image.")
        parsed.append({"x": min(x, width - 1.0), "y": min(y, height - 1.0)})
    return parsed


def segment_request(fields, width, height):
    """Check an image-segment request against the upright image size and
    return the SAM3_Detect settings build_image_segment takes.

    Exactly one of ``prompt`` (a phrase, or several separated by commas),
    ``boxes`` or ``positive_coords`` (with optional ``negative_coords``)
    selects what to segment. A ``raw_prompt`` wins over ``prompt``, as it
    does in every workflow (a serverless job sends its prompt that way).
    Points always make one mask, so ``individual_masks`` only applies to
    phrases and boxes."""
    raw = {name: (fields.get(name) or "").strip()
           for name in ("prompt", "boxes", "positive_coords", "negative_coords")}
    raw["prompt"] = (fields.get("raw_prompt") or "").strip() or raw["prompt"]
    if raw["negative_coords"] and not raw["positive_coords"]:
        raise ValueError("negative_coords only refine positive_coords; send at least one positive point.")
    chosen = [mode for mode, field in zip(SEGMENT_MODES, ("prompt", "boxes", "positive_coords")) if raw[field]]
    if len(chosen) != 1:
        raise ValueError("Send exactly one of 'prompt' (a phrase such as 'the sky'), 'boxes' "
                         "([[x, y, w, h], ...] in pixels) or 'positive_coords' ([[x, y], ...] in pixels, "
                         "optionally with negative_coords).")
    mode = chosen[0]
    max_objects = _bounded(fields.get("max_objects"), SEGMENT_DEFAULT_MAX_OBJECTS, 1, SEGMENT_MAX_OBJECTS,
                           "max_objects", int)
    request = {
        "mode": mode,
        "threshold": _bounded(fields.get("threshold"), SEGMENT_DEFAULT_THRESHOLD, 0.0, 1.0, "threshold", float),
        "individual_masks": parse_flag(fields.get("individual_masks"), False) and mode != "points",
    }
    if mode == "prompt":
        phrases = sam3_phrases(raw["prompt"])
        if not phrases:
            raise ValueError("prompt names nothing to select; send a phrase such as 'the sky'.")
        if len(phrases) > SEGMENT_MAX_PHRASES:
            raise ValueError(f"prompt takes at most {SEGMENT_MAX_PHRASES} comma-separated phrases.")
        request.update(text=sam3_text_prompt(phrases, max_objects), max_detections=len(phrases) * max_objects)
    elif mode == "boxes":
        request["boxes"] = parse_boxes(raw["boxes"], width, height)
    else:
        request["positive"] = parse_points(raw["positive_coords"], width, height, "positive_coords")
        request["negative"] = (parse_points(raw["negative_coords"], width, height, "negative_coords")
                               if raw["negative_coords"] else [])
    return request


def segment_summary(request):
    """One line on what a request selects and which masks it saves, for the
    X-Wrapper-Note header."""
    mode = request["mode"]
    if mode == "prompt":
        what = f"'{request['text']}'"
    elif mode == "boxes":
        what = f"{len(request['boxes'])} box(es)"
    else:
        what = f"{len(request['positive'])} positive and {len(request['negative'])} negative point(s)"
    masks = "the union mask, then one per object" if request["individual_masks"] else "the union mask"
    return f"{what}; saves {masks}"


def segment_mask_count(request):
    """PNG masks a request can save: the union, plus one per object with
    individual_masks (every possible detection, or one per box)."""
    if not request["individual_masks"]:
        return 1
    return 1 + (request["max_detections"] if request["mode"] == "prompt" else len(request["boxes"]))


def segment_check_budget(width, height, masks):
    """An error detail when the masks would not fit this host's memory budget, else None."""
    used = width * height * masks
    if used <= SEGMENT_MAX_PIXEL_MASKS:
        return None
    return (f"{masks} mask(s) of {width}x{height} is {used / 1e6:.0f} megapixel-masks, above this host's "
            f"{SEGMENT_MAX_PIXEL_MASKS / 1e6:.0f} budget. Send a smaller picture, fewer max_objects, or "
            "individual_masks=false.")


def box_xywh(box):
    """A {x, y, width, height} dict -> [x, y, w, h], to a tenth of a pixel."""
    return [round(float(box[key]), 1) for key in ("x", "y", "width", "height")]


def _detections(text):
    """SAM3_Detect's first-frame detections from PreviewAny's JSON print."""
    try:
        frames = json.loads(text) if text else []
    except ValueError:
        return []
    first = frames[0] if isinstance(frames, list) and frames else []
    return [d for d in first if isinstance(d, dict)] if isinstance(first, list) else []


def segment_result(text, tag, images):
    """The image-segment JSON answer.

    ``text`` is SAM3_Detect's detections as PreviewAny printed them, ``tag``
    the result node's tag (mode, upright size, individual_masks and, for box
    requests, the boxes), ``images`` the job's saved masks in order: the union,
    then one per object when individual_masks was on.

    ``objects`` are a phrase request's detections, best first, with their
    confidence; for a box request they are the (clipped) boxes, score null,
    one mask each; a point request has none. With individual_masks every
    object carries the URL of its own mask, in the same order as its PNG."""
    if tag.get("mode") == "prompt":
        objects = [{"score": round(float(d.get("score", 0.0)), 4), "bbox": box_xywh(d)}
                   for d in _detections(text)]
    elif tag.get("mode") == "boxes":
        objects = [{"score": None, "bbox": list(box)} for box in tag.get("boxes") or ()]
    else:
        objects = []
    if tag.get("individual_masks"):
        for obj, url in zip(objects, images[1:]):
            obj["mask"] = url
    return {
        "mode": tag.get("mode"),
        "width": tag.get("width"),
        "height": tag.get("height"),
        "mask": images[0] if images else None,
        "objects": objects,
    }
