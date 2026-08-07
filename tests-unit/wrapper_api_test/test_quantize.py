"""Tests for the fp8 -> nvfp4 checkpoint converter."""

import json
import os
import tempfile
import unittest

import torch

from api_wrapper.quantize import convert_fp8_to_nvfp4, fp4_filename
from comfy.quant_ops import TensorCoreNVFP4Layout
from comfy.utils import load_torch_file


def _make_fp8_checkpoint(tmpdir):
    """Synthetic float8_e4m3fn mixed-precision checkpoint: 2 quantized layers
    + one plain bf16 layer."""
    torch.manual_seed(0)
    w1 = torch.randn(64, 128) * 0.05
    w2 = torch.randn(32, 64) * 0.05
    sd = {
        "double_blocks.0.mlp.0.weight": w1.to(torch.float8_e4m3fn),
        "double_blocks.0.mlp.0.weight_scale": torch.tensor(0.5, dtype=torch.float32),
        "double_blocks.0.mlp.2.weight": w2.to(torch.float8_e4m3fn),
        "double_blocks.0.mlp.2.weight_scale": torch.tensor(0.5, dtype=torch.float32),
        "double_blocks.0.norm.weight": torch.randn(64, dtype=torch.bfloat16),
    }
    metadata = {"_quantization_metadata": json.dumps({"layers": {
        "double_blocks.0.mlp.0": {"format": "float8_e4m3fn"},
        "double_blocks.0.mlp.2": {"format": "float8_e4m3fn"},
    }})}
    src = os.path.join(tmpdir, "test-fp8.safetensors")
    from comfy.utils import save_torch_file
    save_torch_file(sd, src, metadata=metadata)
    return src, {"w1": w1, "w2": w2}


class TestFP4Conversion(unittest.TestCase):
    def test_filename_helper(self):
        self.assertEqual(fp4_filename("flux-2-klein-base-9b-fp8.safetensors"),
                         "flux-2-klein-base-9b-fp4.safetensors")

    def test_convert_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src, refs = _make_fp8_checkpoint(tmpdir)
            dst = os.path.join(tmpdir, "test-fp4.safetensors")

            summary = convert_fp8_to_nvfp4(src, dst)
            self.assertEqual(summary["converted_layers"], 2)
            self.assertEqual(summary["total_layers"], 2)
            self.assertTrue(os.path.getsize(dst) > 0)

            sd, metadata = load_torch_file(dst, safe_load=True, return_metadata=True)
            qm = json.loads(metadata["_quantization_metadata"])
            self.assertEqual(qm["layers"], {
                "double_blocks.0.mlp.0": {"format": "nvfp4"},
                "double_blocks.0.mlp.2": {"format": "nvfp4"},
            })
            # nvfp4 layer keys: uint8 qdata + weight_scale_2 + fp8 block scales
            w = sd["double_blocks.0.mlp.0.weight"]
            self.assertEqual(w.dtype, torch.uint8)
            self.assertIn("double_blocks.0.mlp.0.weight_scale_2", sd)
            self.assertEqual(sd["double_blocks.0.mlp.0.weight_scale"].dtype, torch.float8_e4m3fn)
            # plain layer passed through unchanged
            self.assertEqual(sd["double_blocks.0.norm.weight"].dtype, torch.bfloat16)

            # Round-trip quality: dequantized nvfp4 must stay close to the fp8
            # source (fp4 is coarse, so allow generous relative error).
            params = TensorCoreNVFP4Layout.Params(
                scale=sd["double_blocks.0.mlp.0.weight_scale_2"],
                orig_dtype=torch.float32,
                orig_shape=tuple(refs["w1"].shape),
                block_scale=sd["double_blocks.0.mlp.0.weight_scale"],
            )
            dq = TensorCoreNVFP4Layout.dequantize(w, params)
            ref = refs["w1"].to(torch.float32)
            rel_err = (dq - ref).abs().max().item() / ref.abs().max().item()
            self.assertLess(rel_err, 0.5, f"nvfp4 dequant too far from fp8 source: {rel_err:.3f}")

    def test_converted_checkpoint_loads_through_mixed_precision_ops(self):
        """The CUDA/CPU fp4 path: load the converted state dict through the
        real ops with nvfp4 in the disabled set (eager emulation) and run a
        forward on CPU."""
        import torch.nn.functional as F
        from comfy.ops import mixed_precision_ops

        with tempfile.TemporaryDirectory() as tmpdir:
            src, refs = _make_fp8_checkpoint(tmpdir)
            dst = os.path.join(tmpdir, "test-fp4.safetensors")
            convert_fp8_to_nvfp4(src, dst)

            sd, _ = load_torch_file(dst, safe_load=True, return_metadata=True)
            layer_name = "double_blocks.0.mlp.0"
            state_dict = {k.removeprefix(layer_name + "."): v for k, v in sd.items() if k.startswith(layer_name)}
            state_dict["comfy_quant"] = torch.tensor(
                list(json.dumps({"format": "nvfp4"}).encode("utf-8")), dtype=torch.uint8)

            Ops = mixed_precision_ops({layer_name: {"format": "nvfp4"}},
                                      compute_dtype=torch.bfloat16, disabled=["nvfp4"])
            layer = Ops.Linear(128, 64, bias=False)
            layer.load_state_dict(state_dict, strict=False)
            self.assertIsInstance(layer.weight, torch.nn.Parameter)

            x = torch.randn(4, 128, dtype=torch.bfloat16)
            out = layer(x)
            ref = F.linear(x, refs["w1"].to(torch.bfloat16))
            self.assertEqual(tuple(out.shape), (4, 64))
            self.assertLess((out.float() - ref.float()).abs().max().item(), 0.5)


if __name__ == "__main__":
    unittest.main()
