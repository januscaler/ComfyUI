"""Tests for the VAE decode latent-channel guard.

A VAE with batch-norm scaling (e.g. the 128-channel flux2 VAE) raises a clear
error when handed a latent with the wrong channel count instead of failing
with a cryptic tensor-shape mismatch.
"""

import unittest

import torch

from comfy.ldm.models.autoencoder import AutoencodingEngineLegacy


class _FakeBN:
    def __init__(self, channels):
        self.running_mean = torch.zeros(channels)
        self.running_var = torch.ones(channels)


class _SentinelError(Exception):
    pass


def _make_vae(channels=128):
    vae = object.__new__(AutoencodingEngineLegacy)
    vae.bn = _FakeBN(channels)
    vae.bn_eps = 1e-6
    vae.ps = (1, 1)
    vae.max_batch_size = None

    def dummy_post_quant_conv(z):
        raise _SentinelError(f"post_quant_conv reached with {z.shape[1]} channels")

    vae.post_quant_conv = dummy_post_quant_conv
    return vae


class TestVAEDecodeChannelGuard(unittest.TestCase):
    def test_wrong_channel_count_raises_clear_error(self):
        vae = _make_vae(channels=128)
        latent = torch.zeros(1, 16, 8, 8)  # SD3-style 16-channel latent
        with self.assertRaisesRegex(ValueError, "128"):
            vae.decode(latent)

    def test_matching_channel_count_passes_guard(self):
        vae = _make_vae(channels=128)
        latent = torch.zeros(1, 128, 8, 8)  # flux2-style 128-channel latent
        with self.assertRaises(_SentinelError):
            vae.decode(latent)


if __name__ == "__main__":
    unittest.main()
