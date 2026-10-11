from unittest.mock import patch

import torch

import comfy.utils
from comfy.bg_removal_model import BackgroundRemovalModel


class FakeBiRefNet(torch.nn.Module):
    """Logits from the mean of the preprocessed pixels, and the batch sizes it saw."""

    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, pixel_values):
        self.batches.append(pixel_values.shape[0])
        return pixel_values.mean(dim=1, keepdim=True) * 4 - 2


def _model():
    bg = BackgroundRemovalModel.__new__(BackgroundRemovalModel)
    bg.image_size = 16
    bg.image_mean = [0.0, 0.0, 0.0]
    bg.image_std = [1.0, 1.0, 1.0]
    bg.load_device = torch.device("cpu")
    bg.patcher = None
    bg.model = FakeBiRefNet()
    return bg


def test_encode_image_runs_one_frame_at_a_time_and_reports_progress():
    bg = _model()
    frames = torch.rand(5, 24, 40, 3)
    updates = []
    previous = comfy.utils.PROGRESS_BAR_HOOK
    comfy.utils.set_progress_bar_global_hook(lambda value, total, preview, node_id=None: updates.append((value, total)))
    try:
        with patch("comfy.model_management.load_model_gpu"):
            mask = bg.encode_image(frames)
    finally:
        comfy.utils.set_progress_bar_global_hook(previous)

    assert mask.shape == (5, 24, 40)
    assert bg.model.batches == [1] * 5
    # ProgressBar throttles its hook calls; the total and the final count are what matter.
    assert updates[0] == (1, 5) and updates[-1] == (5, 5)
    # Frame by frame gives each frame's own mask.
    with patch("comfy.model_management.load_model_gpu"):
        single = bg.encode_image(frames[3:4])
    assert torch.allclose(mask[3:4], single)
    assert 0.0 <= mask.min() and mask.max() <= 1.0
