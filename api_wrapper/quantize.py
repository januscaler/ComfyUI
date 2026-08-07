"""FP8 -> NVFP4 checkpoint conversion for the wrapper API.

The FLUX.2 klein 9B fp8 checkpoint quantizes its linear weights with
float8_e4m3fn + per-tensor scales. NVFP4 (fp4 e2m1, block-scaled) halves the
size of those weights. This module converts the fp8 file into a standard
nvfp4 mixed-precision checkpoint that the normal ComfyUI loading path accepts
(``_quantization_metadata`` -> per-layer ``comfy_quant`` markers).

NVFP4 runs natively on Blackwell (``torch._scaled_mm``), on older CUDA GPUs
and CPU via the eager emulated dequant. It cannot run on MPS: the format
carries fp8 block scales and torch has no fp8 support on MPS, so callers on
Apple Silicon should keep the fp8 file (which loads as bf16 there).
"""

import json
import os

import torch

from comfy.quant_ops import TensorCoreNVFP4Layout
from comfy.utils import load_torch_file, save_torch_file

FP4_SUFFIX = "-fp4.safetensors"


def fp4_filename(fp8_filename):
    """Derive the fp4 filename from the fp8 one (flux-2-klein-base-9b-fp8 -> ...-fp4)."""
    return fp8_filename.replace("-fp8.safetensors", FP4_SUFFIX)


def convert_fp8_to_nvfp4(src_path, dst_path, stochastic_rounding=0):
    """Convert a float8_e4m3fn mixed-precision checkpoint to nvfp4.

    Quantized layers are converted weight-by-weight; every other tensor is
    passed through unchanged. Returns a summary dict.
    """
    sd, metadata = load_torch_file(src_path, safe_load=True, return_metadata=True)
    if "_quantization_metadata" not in metadata:
        raise ValueError(f"{src_path} has no quantization metadata; nothing to convert.")
    layers = json.loads(metadata["_quantization_metadata"])["layers"]

    out_sd = {}
    new_layers = {}
    converted = 0
    for key, tensor in sd.items():
        if key.endswith(".comfy_quant"):
            continue
        layer, _, suffix = key.rpartition(".")
        conf = layers.get(layer)
        if conf is not None and conf.get("format") == "float8_e4m3fn":
            if suffix == "weight":
                if tensor.dtype == torch.uint8:  # legacy uint8-on-disk layout
                    tensor = tensor.view(torch.float8_e4m3fn)
                w = tensor.to(torch.float32)
                qdata, params = TensorCoreNVFP4Layout.quantize(
                    w, stochastic_rounding=stochastic_rounding)
                out_sd[f"{layer}.weight"] = qdata
                out_sd[f"{layer}.weight_scale_2"] = params.scale
                out_sd[f"{layer}.weight_scale"] = params.block_scale
                new_layers[layer] = {"format": "nvfp4"}
                converted += 1
                continue
            if suffix == "weight_scale":
                continue  # replaced by the two nvfp4 scales
        out_sd[key] = tensor
        if layer in layers:
            new_layers[layer] = conf

    metadata_out = dict(metadata)
    metadata_out["_quantization_metadata"] = json.dumps({"layers": new_layers})
    save_torch_file(out_sd, dst_path, metadata=metadata_out)
    return {"converted_layers": converted, "total_layers": len(layers),
            "size_bytes": os.path.getsize(dst_path)}
