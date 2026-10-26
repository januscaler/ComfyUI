import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.sam3.tracker import pack_masks  # noqa: E402
from comfy_extras.nodes_sam3 import SAM3_TrackToBBoxes, _mask_boxes, _track_object_indices  # noqa: E402


def _track(masks, orig_size, scores):
    """track_data as SAM3_VideoTrack returns it, from bool masks [N, N_obj, H, W]."""
    return {"packed_masks": pack_masks(masks), "n_frames": masks.shape[0], "scores": scores, "orig_size": orig_size}


def test_mask_boxes_bounds_and_area():
    masks = torch.zeros(2, 8, 16, dtype=torch.bool)
    masks[0, 2:5, 3:11] = True  # 3 rows x 8 cols
    visible, x1, y1, x2, y2, area = (t.tolist() for t in _mask_boxes(masks))
    assert visible == [True, False]
    assert (x1[0], y1[0], x2[0], y2[0]) == (3, 2, 11, 5)
    assert area[0] == 24 / 128 and area[1] == 0


def test_object_indices():
    assert _track_object_indices("", 3) == [0, 1, 2]
    assert _track_object_indices("2, 0, 9, x", 3) == [2, 0]


def test_boxes_are_scaled_to_the_original_frame_and_hidden_frames_are_empty():
    masks = torch.zeros(3, 1, 8, 16, dtype=torch.bool)
    masks[0, 0, 0:4, 0:8] = True
    masks[2, 0, 4:8, 8:16] = True
    frames = SAM3_TrackToBBoxes.execute(_track(masks, (80, 320), [0.9])).result[0]
    assert frames[1] == []
    assert frames[0] == [{"x": 0.0, "y": 0.0, "width": 160.0, "height": 40.0, "score": 0.9, "area": 0.25}]
    assert frames[2][0]["x"] == 160.0 and frames[2][0]["y"] == 40.0


def test_no_object_tracked():
    data = {"packed_masks": None, "n_frames": 2, "scores": [], "orig_size": (4, 4)}
    assert SAM3_TrackToBBoxes.execute(data).result[0] == [[], []]
