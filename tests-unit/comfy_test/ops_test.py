"""Tests for comfy.ops fp8 weight handling."""

import unittest

import torch

from comfy.ops import _dequantize_fp8_weight


class TestDequantizeFP8Weight(unittest.TestCase):
    def test_uint8_stored_fp8_weight(self):
        # Legacy on-disk layout: raw fp8 bits stored as uint8.
        w_fp8 = torch.randn(8, 16).to(torch.float8_e4m3fn)
        w_uint8 = w_fp8.view(torch.uint8)
        scale = torch.tensor(0.5, dtype=torch.float32)

        out = _dequantize_fp8_weight(w_uint8, scale, "float8_e4m3fn", torch.bfloat16)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(out.float(), (w_fp8.float() * scale).bfloat16().float(), atol=1e-2))

    def test_native_fp8_weight(self):
        w_fp8 = torch.randn(4, 8).to(torch.float8_e4m3fn)
        scale = torch.tensor(2.0, dtype=torch.float32)
        out = _dequantize_fp8_weight(w_fp8, scale, "float8_e4m3fn", torch.float32)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.allclose(out, w_fp8.float() * scale))

    def test_e5m2_and_missing_scale(self):
        w_fp8 = torch.randn(4, 8).to(torch.float8_e5m2)
        out = _dequantize_fp8_weight(w_fp8, None, "float8_e5m2", torch.float32)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.allclose(out, w_fp8.float()))


if __name__ == "__main__":
    unittest.main()
