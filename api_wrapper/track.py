"""Requests and answers of the fx-track workflow (SAM 3.1 video tracking).

fx-track follows one object through a clip: frame 0 is the anchor, picked by
an uploaded mask or by the same phrase, boxes or points image-segment takes,
and SAM3_VideoTrack carries it through every frame. These pure functions --
free of comfy imports, like segment.py -- check the form fields and shape the
per-frame boxes GET /wrapper/jobs/{job_id}/text answers with. Coordinates are
pixels of the video's frames.
"""

import json
import os

from api_wrapper import segment

TRACK_MAX_BOXES = 4
TRACK_DEFAULT_DETECT_INTERVAL = 1
TRACK_MAX_DETECT_INTERVAL = 600
# The whole clip sits in host RAM as float32 three times over, as in
# fx-bgremove: the decoded frames (12 bytes per pixel), the matte (4) and the
# matte as RGB frames for the encoder (12), ~28 bytes per pixel-frame. The
# default (2e9 pixel-frames, ~56 GB) takes 30 s of 1080p30 (1.87e9); the
# tracker itself works on 1008x1008 copies, one frame at a time, on the GPU.
TRACK_MAX_PIXEL_FRAMES = max(1, int(float(
    os.environ.get("COMFY_FX_TRACK_MAX_PIXEL_FRAMES", "2000000000"))))


def track_request(fields, width, height, has_mask):
    """Check an fx-track request against the video's frame size and return
    what build_fx_track takes.

    An uploaded frame-0 ``mask`` is the anchor as is (mode 'mask'); the
    selection fields are then ignored. Without one, exactly one of ``prompt``,
    ``boxes`` (at most TRACK_MAX_BOXES) or ``positive_coords`` (with optional
    ``negative_coords``) picks the object on frame 0, as in image-segment;
    whatever they select is tracked as one object."""
    threshold = segment.parse_bounded(fields.get("threshold"), segment.SEGMENT_DEFAULT_THRESHOLD, 0.0, 1.0,
                                      "threshold", float)
    detect_interval = segment.parse_bounded(fields.get("detect_interval"), TRACK_DEFAULT_DETECT_INTERVAL, 1,
                                            TRACK_MAX_DETECT_INTERVAL, "detect_interval", int)
    if has_mask:
        return {"mode": "mask", "threshold": threshold, "detect_interval": detect_interval}
    if not any((fields.get(name) or "").strip() for name in ("prompt", "raw_prompt", "boxes", "positive_coords")):
        raise ValueError("Say what to track on frame 0: upload its 'mask' (PNG, white = object), or send one of "
                         "'prompt' (a phrase), 'boxes' ([[x, y, w, h], ...] in pixels) or 'positive_coords' "
                         "([[x, y], ...] in pixels).")
    request = segment.segment_request({**fields, "max_objects": "1", "individual_masks": "false"}, width, height)
    if request["mode"] == "boxes" and len(request["boxes"]) > TRACK_MAX_BOXES:
        raise ValueError(f"boxes takes at most {TRACK_MAX_BOXES} boxes for a track.")
    request["detect_interval"] = detect_interval
    return request


def track_summary(request):
    """What a request anchors the track on, for the X-Wrapper-Note header."""
    return "the uploaded frame-0 mask" if request["mode"] == "mask" else segment.segment_what(request)


def track_check_budget(width, height, frames):
    """An error detail when the clip would not fit this host's memory budget, else None."""
    used = width * height * frames
    if used <= TRACK_MAX_PIXEL_FRAMES:
        return None
    return (f"{frames} frames of {width}x{height} is {used / 1e6:.0f} megapixel-frames, above this host's "
            f"{TRACK_MAX_PIXEL_FRAMES / 1e6:.0f} budget. Send at most about "
            f"{TRACK_MAX_PIXEL_FRAMES // (width * height)} frames at this size, or a smaller video.")


def _frame(i, boxes):
    """One frame's entry: its best box (the tracked object), or a hidden frame."""
    found = [box for box in boxes if isinstance(box, dict)] if isinstance(boxes, list) else []
    if not found:
        return {"i": i, "bbox": None, "score": 0.0, "area": 0.0}
    best = max(found, key=lambda box: box.get("score", 0.0))
    return {"i": i, "bbox": segment.box_xywh(best), "score": round(float(best.get("score", 0.0)), 4),
            "area": round(float(best.get("area", 0.0)), 6)}


def track_result(text, tag, images):
    """The fx-track JSON answer.

    ``text`` is SAM3_TrackToBBoxes' per-frame boxes as PreviewAny printed them
    (one list per decoded frame), ``tag`` the result node's tag (the video's
    size and fps). Every frame gets an entry: ``bbox`` [x, y, w, h] in pixels,
    or null where the object is not visible; ``score`` the track's confidence
    in the object (0 when hidden); ``area`` its mask's share of the frame. The
    matte itself is the job's video (GET /wrapper/jobs/{job_id}/image)."""
    try:
        frames = json.loads(text) if text else []
    except ValueError:
        frames = []
    if not isinstance(frames, list):
        frames = []
    return {
        "width": tag.get("width"),
        "height": tag.get("height"),
        "fps": tag.get("fps"),
        "frames": [_frame(i, boxes) for i, boxes in enumerate(frames)],
    }
