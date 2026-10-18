"""Requests and answers of the image-describe workflow (Qwen3-VL).

TextGenerate (comfy_extras/nodes_textgen.py) shows the still to the Qwen3-VL 8B
text encoder Qwen Image 2.1 already uses and asks it, with a fixed prompt, for
a caption and the things in the picture as JSON. These pure functions --
free of comfy imports, like workflows.py -- hold that prompt and turn the
model's answer into ``{"caption", "objects": [{"label", "bbox"}]}``, with every
bbox as [x, y, w, h] fractions (0-1) of the image.

The answer is parsed tolerantly: prose or code fences around the JSON, a
0-1000 (Qwen3-VL's grounding scale) or 0-1 coordinate scale, and an answer cut
off by max_length (the complete objects before the cut are kept) all work. An
answer with no JSON at all is taken as the caption itself.
"""

import json
import re

DESCRIBE_TASKS = ("caption", "objects", "both")
DESCRIBE_DEFAULT_TASK = "both"
DESCRIBE_DEFAULT_MAX_LENGTH = 1024
DESCRIBE_MAX_LENGTH = 4096
# Objects the prompt asks for; the parser keeps at most a few more, in case
# the model overshoots.
DESCRIBE_PROMPT_OBJECTS = 20
DESCRIBE_MAX_OBJECTS = 32
# The still is scaled down (never up) to about this many megapixels before the
# model sees it: ~1000 vision tokens, which is plenty for captions and boxes
# and keeps a 12 MP photo from costing 12000.
DESCRIBE_MEGAPIXELS = 1.0
# Qwen3-VL grounds boxes as [x1, y1, x2, y2] from 0 to 1000 of the image's
# width and height.
QWEN3VL_BOX_SCALE = 1000.0
MAX_LABEL_LENGTH = 80
MAX_CAPTION_LENGTH = 1000

_CAPTION_FIELD = '"caption": "<one or two plain sentences describing the whole picture>"'
_OBJECTS_FIELD = '"objects": [{"label": "<short noun phrase>", "bbox_2d": [x1, y1, x2, y2]}]'
_OBJECT_RULES = (
    f"List up to {DESCRIBE_PROMPT_OBJECTS} distinct things someone editing this picture might want to "
    "select: people, animals, objects, text, and regions such as the sky, water, ground or background. "
    "Make each label specific enough to tell similar things apart, e.g. \"man in red jacket\" or "
    "\"woman on the left\". bbox_2d is the tight box around the thing, with coordinates from 0 to 1000 "
    "of the image's width and height.")
_BOX_KEYS = ("bbox_2d", "bbox", "box_2d", "box")
_LABEL_KEYS = ("label", "name", "object")
_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
_CAPTION = re.compile(r'"caption"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_FLAT_OBJECT = re.compile(r"\{[^{}]*\}")


def describe_request(fields):
    """(task, max_length) of an image-describe request."""
    task = (fields.get("task") or DESCRIBE_DEFAULT_TASK).strip().lower()
    if task not in DESCRIBE_TASKS:
        raise ValueError(f"task must be one of: {', '.join(DESCRIBE_TASKS)}.")
    raw = (fields.get("max_length") or "").strip()
    try:
        max_length = int(raw) if raw else DESCRIBE_DEFAULT_MAX_LENGTH
    except ValueError:
        max_length = 0
    if not 1 <= max_length <= DESCRIBE_MAX_LENGTH:
        raise ValueError(f"max_length must be a whole number from 1 to {DESCRIBE_MAX_LENGTH}.")
    return task, max_length


def describe_megapixels(width, height):
    """The megapixel budget the still is scaled to: DESCRIBE_MEGAPIXELS, or
    its own size when smaller (ImageScaleToTotalPixels counts 1024*1024)."""
    own = width * height / (1024 * 1024)
    return round(max(0.01, min(DESCRIBE_MEGAPIXELS, own)), 4)


def describe_prompt(task):
    """The fixed instruction for ``task``: answer with JSON only."""
    fields = {"caption": [_CAPTION_FIELD], "objects": [_OBJECTS_FIELD],
              "both": [_CAPTION_FIELD, _OBJECTS_FIELD]}[task]
    prompt = ("You are labelling a picture for a photo editor. Answer with one JSON object and "
              "nothing else: {" + ", ".join(fields) + "}.")
    return prompt if task == "caption" else f"{prompt} {_OBJECT_RULES}"


def _is_answer(value):
    if isinstance(value, dict):
        return "caption" in value or "objects" in value
    return isinstance(value, list) and any(isinstance(item, dict) for item in value)


def _whole_json(text):
    """The first complete JSON answer in ``text``: an object with a caption or
    objects, or a list of objects. None when there is none."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if _is_answer(value):
            return value
    return None


def _salvaged(text):
    """(caption, objects) from an answer max_length cut off: the caption if
    its string closed, and every object that did."""
    found = _CAPTION.search(text)
    caption = None
    if found:
        try:
            caption = json.loads(f'"{found.group(1)}"')
        except ValueError:
            caption = found.group(1)
    objects = []
    for fragment in _FLAT_OBJECT.findall(text):
        try:
            value = json.loads(fragment)
        except ValueError:
            continue
        if isinstance(value, dict) and any(key in value for key in _BOX_KEYS):
            objects.append(value)
    return caption, objects


def parse_answer(text):
    """(caption, raw objects) of the model's answer, before normalisation.
    The caption is None when the answer has none."""
    text = _THINK.sub("", text or "").strip()
    value = _whole_json(text)
    if isinstance(value, dict):
        objects = value.get("objects")
        return value.get("caption"), objects if isinstance(objects, list) else []
    if isinstance(value, list):
        return None, value
    caption, objects = _salvaged(text)
    if caption is None and not objects and "{" not in text:
        caption = text  # prose instead of JSON: the prose is the caption
    return caption, objects


def _clean_text(value, limit):
    return " ".join(str(value).split())[:limit] if value is not None else ""


def _corners(raw):
    """[x1, y1, x2, y2] numbers of a box value, or None."""
    if isinstance(raw, dict):
        raw = [raw.get(key) for key in ("x1", "y1", "x2", "y2")]
    if (isinstance(raw, list) and len(raw) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw)):
        return [float(v) for v in raw]
    return None


def box_scale(boxes):
    """1 when every coordinate is already a 0-1 fraction, else Qwen3-VL's 1000."""
    return 1.0 if all(0.0 <= c <= 1.0 for box in boxes for c in box) else QWEN3VL_BOX_SCALE


def corners_to_xywh(box, scale):
    """[x1, y1, x2, y2] at ``scale`` -> [x, y, w, h] fractions of the image,
    clipped to it (corners in either order); None when no area is left."""
    def unit(v):
        return min(max(v / scale, 0.0), 1.0)
    x1, x2 = sorted((unit(box[0]), unit(box[2])))
    y1, y2 = sorted((unit(box[1]), unit(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)]


def normalize_objects(raw_objects):
    """[{label, bbox: [x, y, w, h] in 0-1}] from the model's objects: labelled
    and boxed ones only, duplicates dropped, at most DESCRIBE_MAX_OBJECTS."""
    labelled = []
    for item in raw_objects or ():
        if not isinstance(item, dict):
            continue
        label = _clean_text(next((item[k] for k in _LABEL_KEYS if item.get(k)), None), MAX_LABEL_LENGTH)
        corners = _corners(next((item[k] for k in _BOX_KEYS if k in item), None))
        if label and corners:
            labelled.append((label, corners))
    scale = box_scale([corners for _, corners in labelled])
    objects, seen = [], set()
    for label, corners in labelled:
        bbox = corners_to_xywh(corners, scale)
        key = (label.lower(), tuple(bbox or ()))
        if bbox is None or key in seen:
            continue
        seen.add(key)
        objects.append({"label": label, "bbox": bbox})
    return objects[:DESCRIBE_MAX_OBJECTS]


def describe_result(text, tag, images=()):
    """The image-describe JSON answer for the model's ``text``. ``caption`` is
    null when the task was 'objects', ``objects`` empty when it was 'caption'."""
    task = tag.get("task", DESCRIBE_DEFAULT_TASK)
    caption, raw_objects = parse_answer(text)
    return {
        "caption": None if task == "objects" else _clean_text(caption, MAX_CAPTION_LENGTH),
        "objects": [] if task == "caption" else normalize_objects(raw_objects),
    }
