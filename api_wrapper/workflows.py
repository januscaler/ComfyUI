"""Workflow builders for the wrapper API.

This module is deliberately free of comfy imports so it stays unit-testable
and the graph structure is easy to audit. It builds ComfyUI API-format prompt
graphs (as accepted by POST /prompt) and declares the models a workflow needs.
"""

import json
import math
import os

from api_wrapper import describe as image_describe
from api_wrapper import segment as image_segment
from api_wrapper import track as object_track
from api_wrapper.prompt_rewriter import MODEL_OPTIONS as MINIMAX_H3_LLM_MODELS
from api_wrapper.results import tag_result

# Models required by the FLUX.2 [klein] 9B image edit workflow, with the exact
# files expected by the loaders in the graph below. The URLs match the
# Comfy-Org/BFL repos the shipped blueprints point at.
FLUX2_KLEIN_9B_MODELS = [
    {
        "folder": "diffusion_models",
        "filename": "flux-2-klein-base-9b-fp8.safetensors",
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/resolve/main/flux-2-klein-base-9b-fp8.safetensors",
    },
    {
        "folder": "text_encoders",
        "filename": "qwen_3_8b_fp8mixed.safetensors",
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
    },
    {
        "folder": "vae",
        "filename": "flux2-vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors",
    },
]

FLUX2_KLEIN_9B_UNET = "flux-2-klein-base-9b-fp8.safetensors"
FLUX2_KLEIN_9B_CLIP = "qwen_3_8b_fp8mixed.safetensors"
FLUX2_KLEIN_9B_VAE = "flux2-vae.safetensors"

# Qwen Image 2.1 (Comfy-Org/Qwen-Image-2.1), text to image and image edit.
QWEN_IMAGE_21_BASE_URL = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"
QWEN_IMAGE_21_VAE = "qwen_image_2.1_vae_bf16.safetensors"
# int8_convrot is what the official "Qwen Image 2.1" templates ship (7 GB UNET
# + 9 GB encoder); bf16 is the full-precision pair (14 GB + 18 GB).
QWEN_IMAGE_21_QUANT_MODELS = {
    "int8": {"unet": "qwen_image_2.1_int8_convrot.safetensors", "clip": "qwen3vl_8b_int8_convrot.safetensors"},
    "bf16": {"unet": "qwen_image_2.1_bf16.safetensors", "clip": "qwen3vl_8b_bf16.safetensors"},
}
QWEN_IMAGE_21_UNET = QWEN_IMAGE_21_QUANT_MODELS["int8"]["unet"]
QWEN_IMAGE_21_CLIP = QWEN_IMAGE_21_QUANT_MODELS["int8"]["clip"]
# The templates' KSampler: euler/simple, 25 steps, cfg 1 (the negative prompt
# only matters once cfg is raised above 1).
QWEN_IMAGE_21_DEFAULT_STEPS = 25
QWEN_IMAGE_21_DEFAULT_CFG = 1.0


def qwen_image_21_models(quantization):
    """The UNET, text encoder and VAE a Qwen Image 2.1 job needs."""
    q = QWEN_IMAGE_21_QUANT_MODELS[quantization]
    return [
        {"folder": "diffusion_models", "filename": q["unet"],
         "url": f"{QWEN_IMAGE_21_BASE_URL}/diffusion_models/{q['unet']}"},
        {"folder": "text_encoders", "filename": q["clip"],
         "url": f"{QWEN_IMAGE_21_BASE_URL}/text_encoders/{q['clip']}"},
        {"folder": "vae", "filename": QWEN_IMAGE_21_VAE,
         "url": f"{QWEN_IMAGE_21_BASE_URL}/vae/{QWEN_IMAGE_21_VAE}"},
    ]


# Models required by the Ideogram 4 text-to-image workflow (fp8 + native nvfp4
# variants). URL layout matches the Comfy-Org repos the shipped "Text to Image
# (Ideogram v4)" blueprint points at.
IDEOGRAM4_UNET = "ideogram4_fp8_scaled.safetensors"
IDEOGRAM4_UNET_UNCONDITIONAL = "ideogram4_unconditional_fp8_scaled.safetensors"
IDEOGRAM4_UNET_FP4 = "ideogram4_nvfp4_mixed.safetensors"
IDEOGRAM4_UNET_UNCONDITIONAL_FP4 = "ideogram4_unconditional_nvfp4_mixed.safetensors"
IDEOGRAM4_CLIP = "qwen3vl_8b_fp8_scaled.safetensors"
IDEOGRAM4_VAE = "flux2-vae.safetensors"

IDEOGRAM4_MODELS = [
    {"folder": "diffusion_models", "filename": IDEOGRAM4_UNET,
     "url": "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_fp8_scaled.safetensors"},
    {"folder": "diffusion_models", "filename": IDEOGRAM4_UNET_UNCONDITIONAL,
     "url": "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"},
    {"folder": "text_encoders", "filename": IDEOGRAM4_CLIP,
     "url": "https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"},
    {"folder": "vae", "filename": IDEOGRAM4_VAE,
     "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"},
]

IDEOGRAM4_FP4_MODELS = [
    {"folder": "diffusion_models", "filename": IDEOGRAM4_UNET_FP4,
     "url": "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_nvfp4_mixed.safetensors"},
    {"folder": "diffusion_models", "filename": IDEOGRAM4_UNET_UNCONDITIONAL_FP4,
     "url": "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors"},
    {"folder": "text_encoders", "filename": IDEOGRAM4_CLIP,
     "url": "https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"},
    {"folder": "vae", "filename": IDEOGRAM4_VAE,
     "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"},
]

# Scheduler presets from the blueprint's preset JSON; `mode` selects one and
# the optional `steps` form field overrides the preset step count.
IDEOGRAM4_PRESETS = {
    "quality": {"steps": 48, "mu": 0.0, "std": 1.5},
    "default": {"steps": 20, "mu": 0.0, "std": 1.75},
    "turbo": {"steps": 12, "mu": 0.5, "std": 1.75},
}

# Example prompt for the docs: Ideogram 4 accepts rich structured JSON prompts
# that describe composition, style and elements with bounding boxes.
IDEOGRAM4_EXAMPLE_PROMPT = """{
  "high_level_description": "A surreal streetwear mixed-media collage poster featuring a relaxed skateboarder mid-air against a vibrant blue sky, backed by giant puffy 3D letters spelling 'COMFY'. The composition blends retro magazine cutout aesthetics with grunge elements like torn paper banners and distressed red stamps, conveying an effortless, cozy vibe.",
  "style_description": {
    "aesthetics": "Retro magazine cutout style, mixed-media digital collage, high-contrast streetwear graphic, featuring rough ripped paper edges and distressed grunge textures.",
    "lighting": "High-contrast flash mixed with harsh midday sunlight on the skater cutout, contrasting with flat, bright graphic lighting on the 3D typography.",
    "photo": "Vintage grainy 35mm film with distressed halftone scan textures and subtle light leaks.",
    "medium": "Mixed-media digital collage",
    "color_palette": ["#1E73BE", "#FDFDFD", "#C82A2A", "#657C9C", "#EFEFEF"]
  },
  "compositional_deconstruction": {
    "background": "A vibrant, clear blue sky layered with a vintage grainy film texture and subtle halftone dot patterns, transitioning down to an implied pale gray concrete ramp at the very bottom edge.",
    "elements": [
      {"type": "obj", "bbox": [128, 149, 354, 810], "desc": "Massive 3D puffy, inflatable white typography spelling 'COMFY' stretching across the upper half of the canvas, acting as a surreal, soft cloud-like backdrop.", "color_palette": ["#FDFDFD", "#E0E0E0", "#D3DBE2"]},
      {"type": "obj", "bbox": [459, 37, 727, 264], "desc": "A cluster of oversized, distressed red stamped circles and dots, applied loosely to the midground like a grunge ink stamp, partially obscuring the bottom left of the text.", "color_palette": ["#C82A2A", "#A11D1D"]},
      {"type": "obj", "bbox": [23, 366, 153, 666], "desc": "A vertically oriented, torn paper side banner pinned to the left edge, displaying the bold stamped text 'STAY COZY' in high-contrast black ink.", "color_palette": ["#EFEFEF", "#1A1A1A", "#C82A2A"]},
      {"type": "obj", "bbox": [287, 210, 756, 819], "desc": "A sharp photographic cutout of a skateboarder mid-air in a relaxed pose, wearing loose-fitting washed denim jeans and a plain white tee, floating effortlessly above the concrete ramp with a distinct white cutout border.", "color_palette": ["#FDFDFD", "#657C9C", "#2B2B2B", "#DCA57D"]},
      {"type": "obj", "bbox": [773, 39, 973, 187], "desc": "A surreal, miniature floating skateboard cutout, positioned playfully in the upper right sky as if defying gravity.", "color_palette": ["#D2A679", "#2B2B2B", "#C82A2A"]},
      {"type": "obj", "bbox": [105, 830, 905, 980], "desc": "A wide, horizontal strip of heavily textured torn paper spanning the lower third of the composition, featuring the bold typographic phrase 'BEYOND THE COMFORT ZONE' intermixed with 'EFFORTLESS RIDE' alongside ripped edges that reveal the background.", "color_palette": ["#EFEFEF", "#1A1A1A", "#999999"]}
    ]
  }
}"""


def build_flux2_klein_9b_img2img(*, prompt, image=None, ref_images=(), negative_prompt="",
                                 seed=0, steps=20, cfg=5.0, megapixels=1.0,
                                 width=None, height=None,
                                 filename_prefix="wrapper/flux2_klein_9b",
                                 unet_name=FLUX2_KLEIN_9B_UNET):
    """Build the FLUX.2 [klein] 9B image edit graph (API format).

    Mirrors the shipped "Image Edit (Flux.2 Klein)" blueprint: the input image
    is scaled to a fixed megapixel budget, its latent is attached to both the
    positive and negative conditioning as a reference latent, and sampling
    runs through the flux2 custom sampler stack (CFG guider, euler,
    flux2 scheduler, empty flux2 latent at the image size).

    Several references (``image`` first, then ``ref_images``) chain one
    reference latent each — FLUX.2's multi-reference edit — so a character
    portrait and a location can shape one still. ``width``/``height`` set the
    output canvas (a video's frame); without them it takes the first
    reference's scaled size, as the blueprint does.

    File names are relative to the ComfyUI input directory. ``unet_name``
    selects the diffusion model file (fp8 by default; pass the converted
    nvfp4 file name for the fp4 path).
    """
    refs = ([image] if image else []) + list(_ref_list(ref_images))
    if not refs:
        raise ValueError("the image edit needs at least one image")
    canvas_w = ["6", 0] if width is None else width
    canvas_h = ["6", 1] if height is None else height
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX2_KLEIN_9B_CLIP, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_KLEIN_9B_VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": refs[0]}},
        # lanczos, not nearest: the budget usually upscales a still, and a
        # blocky reference is a blurred face in everything made from it.
        "5": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 1}},
        "6": {"class_type": "GetImageSize", "inputs": {"image": ["5", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative_prompt}},
        "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "10": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["9", 0]}},
        "11": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["8", 0], "latent": ["9", 0]}},
    }
    positive, negative = ["10", 0], ["11", 0]
    next_id = 20
    for ref in refs[1:]:
        load, scale, encode, pos, neg = (str(next_id + i) for i in range(5))
        graph[load] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        graph[scale] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": [load, 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 1}}
        graph[encode] = {"class_type": "VAEEncode", "inputs": {"pixels": [scale, 0], "vae": ["3", 0]}}
        graph[pos] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive, "latent": [encode, 0]}}
        graph[neg] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": negative, "latent": [encode, 0]}}
        positive, negative = [pos, 0], [neg, 0]
        next_id += 5
    graph.update({
        "12": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": positive, "negative": negative, "cfg": cfg}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": canvas_w, "height": canvas_h}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "16": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": canvas_w, "height": canvas_h, "batch_size": 1}},
        "17": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["12", 0], "sampler": ["13", 0], "sigmas": ["14", 0], "latent_image": ["16", 0]}},
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["3", 0]}},
        "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": filename_prefix}},
    })
    return graph


def build_ideogram4_text2img(*, prompt, seed=0, steps=20, mu=0.0, std=1.75,
                             width=1024, height=1024,
                             filename_prefix="wrapper/ideogram4",
                             unet_name=IDEOGRAM4_UNET,
                             unconditional_unet_name=IDEOGRAM4_UNET_UNCONDITIONAL):
    """Build the Ideogram 4 text-to-image graph (API format).

    Mirrors the shipped "Text to Image (Ideogram v4)" blueprint: the positive
    conditioning is zeroed out as the negative, the main UNET runs through a
    CFG override and a dual-model guider (the unconditional UNET powers the
    negative pass), and sampling uses the ideogram4 scheduler (euler, custom
    sampler stack). Width/height are rounded up to multiples of 16 (min 256)
    exactly like the blueprint's math expressions.

    ``prompt`` is passed to the text encoder as-is: Ideogram 4 natively
    understands rich structured JSON prompts (see IDEOGRAM4_EXAMPLE_PROMPT).
    """
    width = max(((width + 15) // 16) * 16, 256)
    height = max(((height + 15) // 16) * 16, 256)
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": unconditional_unet_name, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": IDEOGRAM4_CLIP, "type": "ideogram4", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": IDEOGRAM4_VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "CFGOverride", "inputs": {"model": ["1", 0], "cfg": 3.0, "start_percent": 0.7, "end_percent": 1.0}},
        "8": {"class_type": "DualModelGuider", "inputs": {"model": ["7", 0], "model_negative": ["2", 0], "positive": ["5", 0], "cfg": 7.0, "negative": ["6", 0]}},
        "9": {"class_type": "Ideogram4Scheduler", "inputs": {"steps": steps, "width": width, "height": height, "mu": mu, "std": std}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "12": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["8", 0], "sampler": ["11", 0], "sigmas": ["9", 0], "latent_image": ["12", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": filename_prefix}},
    }


def build_flux2_klein_9b_text2img(*, prompt, negative_prompt="", seed=0,
                                  steps=20, cfg=5.0, width=1024, height=1024,
                                  filename_prefix="wrapper/flux2klein9b_txt2img",
                                  unet_name=FLUX2_KLEIN_9B_UNET):
    """Build the FLUX.2 [klein] 9B text-to-image graph (API format).

    Same model stack and sampler settings as the image-edit variant (klein
    uses CFG with a real negative prompt), with an empty latent at the
    requested width/height instead of the reference-image path.
    """
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX2_KLEIN_9B_CLIP, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_KLEIN_9B_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative_prompt}},
        "6": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": cfg}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9", 0], "guider": ["6", 0], "sampler": ["7", 0], "sigmas": ["8", 0], "latent_image": ["10", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": filename_prefix}},
    }


def build_qwen_image_21_text2img(*, prompt, negative_prompt="", seed=0,
                                 steps=QWEN_IMAGE_21_DEFAULT_STEPS, cfg=QWEN_IMAGE_21_DEFAULT_CFG,
                                 width=1024, height=1024,
                                 filename_prefix="wrapper/qwenimage21_txt2img",
                                 unet_name=QWEN_IMAGE_21_UNET, clip_name=QWEN_IMAGE_21_CLIP):
    """Build the Qwen Image 2.1 text-to-image graph (API format).

    Mirrors the official "Qwen Image 2.1: Text to Image" template:
    TextEncodeQwenImage21 gives the positive and negative conditioning, and a
    plain KSampler (euler, simple) samples an empty latent at width/height.
    """
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_IMAGE_21_VAE}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0], "prompt": prompt, "negative_prompt": negative_prompt, "resolution": 1024}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "simple", "positive": ["4", 0], "negative": ["4", 1],
            "latent_image": ["5", 0], "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}},
    }


def build_qwen_image_21_edit(*, prompt, image=None, ref_images=(), negative_prompt="",
                             seed=0, steps=QWEN_IMAGE_21_DEFAULT_STEPS, cfg=QWEN_IMAGE_21_DEFAULT_CFG,
                             resolution=1024, width=None, height=None,
                             filename_prefix="wrapper/qwenimage21",
                             unet_name=QWEN_IMAGE_21_UNET, clip_name=QWEN_IMAGE_21_CLIP):
    """Build the Qwen Image 2.1 image edit graph (API format).

    Mirrors the official "Qwen Image 2.1: Image Edit" template. Every image
    (``image`` first, then ``ref_images``) goes to TextEncodeQwenImage21,
    which shows it to the text encoder and splices its VAE latent into the
    sequence; the prompt names them <image1>, <image2>, ... in that order, and
    image 1 is the one edited. ``resolution`` is the images' pixel budget
    (about resolution x resolution, 0 keeps their own size). The output takes
    image 1's resized size unless ``width``/``height`` set a canvas, which
    should stay close to it or the edit shifts.
    """
    refs = ([image] if image else []) + list(_ref_list(ref_images))
    if not refs:
        raise ValueError("the image edit needs at least one image")
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_IMAGE_21_VAE}},
    }
    # Autogrow slots are addressed by their flat dotted id ("images.image_1");
    # a nested dict is dropped without a validation error.
    images = {}
    for i, ref in enumerate(refs, start=1):
        graph[str(9 + i)] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        images[f"images.image_{i}"] = [str(9 + i), 0]
    graph["4"] = {"class_type": "TextEncodeQwenImage21", "inputs": {
        "clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "negative_prompt": negative_prompt,
        "resolution": resolution, **images}}
    latent = ["4", 2]
    if width is not None:
        graph["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        latent = ["5", 0]
    graph.update({
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "simple", "positive": ["4", 0], "negative": ["4", 1],
            "latent_image": latent, "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}},
    })
    return graph


# MiniMax H3 omni-modal video model files (Comfy-Org/MiniMax-H3). The fl2va
# family powers text-to-video and image-to-video; ref2va is the
# reference-to-video variant with its own weights. int8_convrot is the
# canonical template choice; fp8_scaled loads everywhere (bf16 fallback on
# MPS); bf16 is the full-quality option.
MINIMAX_H3_FPS = 24
# The official ComfyUI templates (video_minimax_h3_{t2v,i2v,r2v}) all ship the
# int8_convrot UNET + nvfp4_awq encoder + fp16 video VAE + fp32 audio VAE, so
# that quadruple is what the wrapper reproduces by default. int8 and fp8 UNETs
# are the same size on disk (~21 GB); int8_convrot is the one the templates and
# the model card were validated against.
MINIMAX_H3_DEFAULT_QUANT = "int8"
MINIMAX_H3_UNET_FP8 = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
MINIMAX_H3_UNET_INT8 = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
MINIMAX_H3_UNET_BF16 = "minimax_h3_fl2va_pruned_bf16.safetensors"
MINIMAX_H3_REF2VA_UNET_FP8 = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
MINIMAX_H3_REF2VA_UNET_INT8 = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
MINIMAX_H3_REF2VA_UNET_BF16 = "minimax_h3_ref2va_pruned_bf16.safetensors"
MINIMAX_H3_CLIP = "qwen3vl_32b_minimax_h3_bf16.safetensors"
MINIMAX_H3_CLIP_INT8 = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
MINIMAX_H3_CLIP_NVFP4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
MINIMAX_H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
MINIMAX_H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
MINIMAX_H3_BASE_URL = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
MINIMAX_H3_QUANT_MODELS = {
    "fp8": {"unet": MINIMAX_H3_UNET_FP8, "ref2va": MINIMAX_H3_REF2VA_UNET_FP8, "clip": MINIMAX_H3_CLIP},
    "int8": {"unet": MINIMAX_H3_UNET_INT8, "ref2va": MINIMAX_H3_REF2VA_UNET_INT8, "clip": MINIMAX_H3_CLIP_INT8},
    "bf16": {"unet": MINIMAX_H3_UNET_BF16, "ref2va": MINIMAX_H3_REF2VA_UNET_BF16, "clip": MINIMAX_H3_CLIP},
    # No nvfp4 UNET variant exists; nvfp4 is a CLIP-only option. On Blackwell
    # (sm_120) the CLIP runs natively as fp4 — 16 GB vs 34 GB (int8) / 66 GB
    # (bf16) — which is what lets a 32 GB GPU fit the whole pipeline.
    "nvfp4": {"unet": MINIMAX_H3_UNET_FP8, "ref2va": MINIMAX_H3_REF2VA_UNET_FP8, "clip": MINIMAX_H3_CLIP_NVFP4},
}
MINIMAX_H3_MODEL_NAMES = (
    MINIMAX_H3_UNET_FP8, MINIMAX_H3_UNET_INT8, MINIMAX_H3_UNET_BF16,
    MINIMAX_H3_REF2VA_UNET_FP8, MINIMAX_H3_REF2VA_UNET_INT8, MINIMAX_H3_REF2VA_UNET_BF16,
    MINIMAX_H3_CLIP, MINIMAX_H3_CLIP_INT8, MINIMAX_H3_CLIP_NVFP4,
    MINIMAX_H3_VIDEO_VAE, MINIMAX_H3_AUDIO_VAE,
)


MINIMAX_H3_QUANT_ORDER = ("nvfp4", "int8", "fp8", "bf16")
# Smallest CLIP first; bf16 (66 GB) is only ever used when it is the only
# encoder present. nvfp4 (16 GB) is the canonical template default.
MINIMAX_H3_CLIP_ORDER = (MINIMAX_H3_CLIP_NVFP4, MINIMAX_H3_CLIP_INT8, MINIMAX_H3_CLIP)


def minimax_h3_model_pick(requested, ref2va, is_present):
    """Pick (unet_name, clip_name, note) for a MiniMax H3 job.

    The CLIP is chosen independently of the UNET quantization: the smallest
    encoder already on disk wins (nvfp4 16 GB -> int8 34 GB -> bf16 66 GB),
    so a previously-downloaded bf16 encoder is never preferred over a smaller
    one. The UNET follows the requested quantization when that file exists,
    otherwise the first present variant (nvfp4/int8/fp8 UNETs are ~20 GB;
    bf16 is ~40 GB). ``is_present(folder, filename)`` reports local files."""
    if requested is not None and requested not in MINIMAX_H3_QUANT_MODELS:
        raise ValueError(f"Invalid quantization: {requested}")
    q = requested or MINIMAX_H3_DEFAULT_QUANT
    quants = MINIMAX_H3_QUANT_MODELS
    unet_key = "ref2va" if ref2va else "unet"

    notes = []
    unet_order = [q] + [x for x in MINIMAX_H3_QUANT_ORDER if x != q]
    unet_quant, unet = None, None
    for qq in unet_order:
        cand = quants[qq][unet_key]
        if is_present("diffusion_models", cand):
            unet_quant, unet = qq, cand
            break
    if unet is None:
        unet_quant, unet = q, quants[q][unet_key]
    if unet_quant != q:
        notes.append(f"the quantization={q} UNET is not on disk; using {unet} already present.")

    clip = None
    for cand in MINIMAX_H3_CLIP_ORDER:
        if is_present("text_encoders", cand):
            clip = cand
            break
    if clip is None:
        clip = MINIMAX_H3_CLIP_NVFP4
    if clip != MINIMAX_H3_CLIP_ORDER[0]:
        # Only worth reporting when the preferred (smallest) encoder is absent
        # and a heavier one had to stand in -- that is the case that costs RAM.
        notes.append(f"{MINIMAX_H3_CLIP_ORDER[0]} is not on disk; using the larger {clip} "
                     "text encoder already present.")

    return unet, clip, " ".join(notes) or None


def minimax_h3_models(quantization, ref2va=False, unet_name=None, clip_name=None):
    """The 4 model files a MiniMax H3 task needs for the given quantization.

    ``unet_name``/``clip_name`` override the quantization defaults when the
    wrapper picked a different (already-downloaded) variant."""
    q = MINIMAX_H3_QUANT_MODELS[quantization]
    unet = unet_name or (q["ref2va"] if ref2va else q["unet"])
    clip = clip_name or q["clip"]
    return [
        {"folder": "diffusion_models", "filename": unet,
         "url": f"{MINIMAX_H3_BASE_URL}/diffusion_models/{unet}"},
        {"folder": "text_encoders", "filename": clip,
         "url": f"{MINIMAX_H3_BASE_URL}/text_encoders/{clip}"},
        {"folder": "vae", "filename": MINIMAX_H3_VIDEO_VAE,
         "url": f"{MINIMAX_H3_BASE_URL}/vae/{MINIMAX_H3_VIDEO_VAE}"},
        {"folder": "vae", "filename": MINIMAX_H3_AUDIO_VAE,
         "url": f"{MINIMAX_H3_BASE_URL}/vae/{MINIMAX_H3_AUDIO_VAE}"},
    ]


def minimax_h3_length(duration):
    """Duration (seconds) -> frame count at 24 fps (min 5; the nodes snap it
    to the model's 17k+5 grid)."""
    return max(5, round(duration * MINIMAX_H3_FPS))


def minimax_h3_align_frames(length):
    """Frame count the H3 nodes will actually generate for ``length``.

    Mirrors ``comfy_extras.nodes_minimax_h3.align_frame_count`` (kept local so
    this module stays comfy-import-free): the model works in 17-frame blocks
    plus a 5-frame head, so any length snaps up to the next 17k+5 value."""
    n = max(5, length)
    while n % 17 != 5:
        n += 1
    return n


# H3's native canvas: a 768 px short edge capped at 768*1344 pixels, rounded to
# a multiple of 32. Beyond that the model is out of distribution, so the wrapper
# scales an oversized request down instead of generating something unusable.
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MAX_PIXELS = 1344 * 768

# H3's four checkpoints total ~43 GB (21 GB UNET + 16 GB text encoder + 5.8 GB
# VAEs), so on a 32 GB RAM box every job already runs at 29-31 GB of host RAM
# before a single pixel is generated. What is left decides whether the run
# finishes or the kernel OOM-kills the server (exit 137), which takes the whole
# queue down with it -- so the wrapper refuses jobs it does not expect to
# survive rather than discovering the limit the hard way.
#
# Measured on the target box (RTX 5090, 32 GB VRAM / 32 GB RAM, no swap, no
# --fast-disk) with the int8 UNET + nvfp4 encoder:
#
#     canvas    frames  steps  Mpx-frames  peak RAM  result
#      864x480     124     20        51.4    30.4 GB  ok, 91 s
#      736x736     124      2        67.2    29.7 GB  ok, 33 s
#     1152x640     124     20        91.4    31.4 GB  ok, 175 s
#     1344x768     124      2       128.0    30.4 GB  ok, 57 s
#      960x544     243      2       126.9    30.6 GB  OOM-killed
#     1344x768     124     20       128.0    29.9 GB  OOM-killed after 298 s
#     1344x768     158      2       163.1    30.4 GB  OOM-killed
#     1344x768     192      2       198.3    31.1 GB  OOM-killed
#     1344x768     243      2       250.9    30.9 GB  OOM-killed
#
# Two independent limits fall out of that. Frame count is the harder one: every
# run above 124 frames died regardless of how small the canvas was, because the
# video VAE tiles spatially (256 px) but not temporally. Canvas area matters
# too, but only at a realistic step count -- 1344x768 x 124 survives 2 steps and
# dies at 20, because the longer a job sits at ~30 GB the more likely it is to
# lose the race. So both a frame cap and an area*frames budget are enforced.
#
# The budget is the largest combination that completed (1152x640 x 124), while
# the form defaults below sit at the largest one that completed with real
# margin (864x480 x 124, 1.6 GB of RAM to spare -- 1152x640 finished with only
# 0.6 GB). Both limits are env-tunable: a host with more RAM, with swap, or
# running with --fast-disk (see docker-compose.yml) can raise them without a
# code change.
MINIMAX_H3_MAX_FRAMES = max(5, int(os.environ.get("COMFY_MINIMAX_H3_MAX_FRAMES", "124")))
MINIMAX_H3_MAX_DURATION = round(MINIMAX_H3_MAX_FRAMES / MINIMAX_H3_FPS, 2)
MINIMAX_H3_MAX_PIXEL_FRAMES = max(1, int(float(
    os.environ.get("COMFY_MINIMAX_H3_MAX_PIXEL_FRAMES", str(1152 * 640 * 124)))))
# Defaults: the largest canvas/length that finished a real 20-step job with RAM
# to spare, so a request carrying nothing but a prompt gets the best output the
# host can reliably finish. 864x480 is also the resolution the official H3
# templates' own Resolution Selector ships (0.4 MP, 16:9).
MINIMAX_H3_DEFAULT_WIDTH = 864
MINIMAX_H3_DEFAULT_HEIGHT = 480
MINIMAX_H3_DEFAULT_STEPS = 20


def minimax_h3_check_budget(width, height, frames):
    """Return an error detail when a canvas/length would risk OOM-killing the
    host, or None when the job fits the configured budget."""
    if frames > MINIMAX_H3_MAX_FRAMES:
        return (f"{frames} frames at 24 fps is above this host's {MINIMAX_H3_MAX_FRAMES}-frame limit "
                f"(max duration {MINIMAX_H3_MAX_DURATION}s). Frame count is the hardest limit for "
                "MiniMax H3: the video VAE tiles spatially but not temporally, so a longer clip "
                "costs far more memory than a wider one, and no resolution is small enough to "
                "compensate.")
    budget = MINIMAX_H3_MAX_PIXEL_FRAMES
    used = width * height * frames
    if used > budget:
        max_pixels = budget // frames
        # Largest 32-aligned canvas with this aspect ratio that fits the budget.
        m = MINIMAX_H3_CANVAS_MULTIPLE
        scale = (max_pixels / (width * height)) ** 0.5
        fit_w = max(m, int(width * scale) // m * m)
        fit_h = max(m, int(height * scale) // m * m)
        return (f"{width}x{height} for {frames} frames needs {used / 1e6:.0f} megapixel-frames, "
                f"above this host's {budget / 1e6:.0f} budget. Either drop the canvas to about "
                f"{fit_w}x{fit_h} at this length, or shorten the clip.")
    return None


def minimax_h3_fit_canvas(width, height):
    """Round a requested canvas up to H3's grid, scaling it down (aspect kept)
    when it exceeds the model's 768*1344 pixel budget."""
    m = MINIMAX_H3_CANVAS_MULTIPLE
    width = max(m, -(-width // m) * m)
    height = max(m, -(-height // m) * m)
    if width * height <= MINIMAX_H3_MAX_PIXELS:
        return width, height, None
    # Round the fitted canvas *down* to the grid: rounding up here could push
    # the area back over the cap the scale factor just brought it under.
    scale = (MINIMAX_H3_MAX_PIXELS / (width * height)) ** 0.5
    fitted_w = max(m, int(width * scale) // m * m)
    fitted_h = max(m, int(height * scale) // m * m)
    note = (f"{width}x{height} exceeds MiniMax H3's {MINIMAX_H3_MAX_PIXELS}-pixel native canvas; "
            f"generating at {fitted_w}x{fitted_h} instead.")
    return fitted_w, fitted_h, note


def _minimax_h3_head(prompt, width, height, length, unet_name, clip_name,
                     video_vae_name, audio_vae_name, cond_inputs):
    """Shared model/conditioning half of every MiniMax H3 graph."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae_name}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae_name}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": prompt,
            "width": width, "height": height, "length": length, **cond_inputs}},
    }


def _minimax_h3_tail(model_ref, cond_ref, latent_ref, seed, steps, scheduler, start, filename_prefix):
    """Sampling + joint decode + mux + save half of every MiniMax H3 graph."""
    nodes = {
        str(start): {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": cond_ref}},
        str(start + 1): {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        str(start + 2): {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        str(start + 3): {"class_type": "BasicScheduler", "inputs": {"model": model_ref, "scheduler": scheduler, "steps": steps, "denoise": 1.0}},
        str(start + 4): {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [str(start + 1), 0], "guider": [str(start), 0], "sampler": [str(start + 2), 0],
            "sigmas": [str(start + 3), 0], "latent_image": latent_ref}},
        str(start + 5): {"class_type": "VAEDecode", "inputs": {"samples": [str(start + 4), 0], "vae": ["4", 0]}},
        str(start + 6): {"class_type": "VAEDecodeAudio", "inputs": {"samples": [str(start + 4), 0], "vae": ["5", 0]}},
        str(start + 7): {"class_type": "CreateVideo", "inputs": {
            "images": [str(start + 5), 0], "fps": float(MINIMAX_H3_FPS),
            "audio": [str(start + 6), 0], "bit_depth": 8}},
        str(start + 8): {"class_type": "SaveVideo", "inputs": {
            "video": [str(start + 7), 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"}},
    }
    return nodes


def _minimax_h3_cond_and_tail(prompt, width, height, length, seed, steps, scheduler,
                              unet_name, clip_name, cond_inputs, loader_nodes, next_id,
                              filename_prefix):
    """Head (models + conditioning) + loader nodes + sampling/decode tail."""
    head = _minimax_h3_head(prompt, width, height, length, unet_name, clip_name,
                            MINIMAX_H3_VIDEO_VAE, MINIMAX_H3_AUDIO_VAE, cond_inputs)
    tail = _minimax_h3_tail(["2", 0], ["6", 0], ["6", 1], seed, steps, scheduler,
                            next_id, filename_prefix)
    return {**head, **loader_nodes, **tail}


def build_minimax_h3_text_to_video(*, prompt, seed=0, steps=MINIMAX_H3_DEFAULT_STEPS,
                                   width=MINIMAX_H3_DEFAULT_WIDTH, height=MINIMAX_H3_DEFAULT_HEIGHT,
                                   duration=5.0, scheduler="beta",
                                   filename_prefix="wrapper/minimaxh3_t2v",
                                   unet_name=MINIMAX_H3_UNET_INT8,
                                   clip_name=MINIMAX_H3_CLIP_NVFP4):
    """MiniMax H3 text-to-video: prompt -> joint audio+video MP4."""
    return _minimax_h3_cond_and_tail(prompt, width, height, minimax_h3_length(duration),
                                     seed, steps, scheduler, unet_name, clip_name, {}, {}, 7,
                                     filename_prefix)


def build_minimax_h3_image_to_video(*, prompt, seed=0, steps=MINIMAX_H3_DEFAULT_STEPS,
                                    width=MINIMAX_H3_DEFAULT_WIDTH, height=MINIMAX_H3_DEFAULT_HEIGHT,
                                    duration=5.0, scheduler="beta",
                                    filename_prefix="wrapper/minimaxh3_i2v",
                                    first_frame=None, last_frame=None,
                                    unet_name=MINIMAX_H3_UNET_INT8,
                                    clip_name=MINIMAX_H3_CLIP_NVFP4):
    """MiniMax H3 image-to-video: first (and optional last) frame + prompt ->
    joint audio+video MP4."""
    cond_inputs = {}
    loader_nodes = {}
    next_id = 7
    loaded = []
    for input_name, ref in (("first_frame", first_frame), ("last_frame", last_frame)):
        if ref is None:
            continue
        loader_nodes[str(next_id)] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        loaded.append((input_name, str(next_id)))
        next_id += 1
    for input_name, loader in loaded:
        # Cover-crop to the canvas first. The node stretches a first frame to
        # fit (and cover-crops a last one), so an off-aspect still — a 3:4
        # photo in a 9:16 film, a 1:1 edit — squashed every face in the shot,
        # and a pair from one source warped differently at each end.
        loader_nodes[str(next_id)] = {"class_type": "ImageScale", "inputs": {
            "image": [loader, 0], "upscale_method": "lanczos",
            "width": width, "height": height, "crop": "center"}}
        cond_inputs[input_name] = [str(next_id), 0]
        next_id += 1
    return _minimax_h3_cond_and_tail(prompt, width, height, minimax_h3_length(duration),
                                     seed, steps, scheduler, unet_name, clip_name,
                                     cond_inputs, loader_nodes, next_id, filename_prefix)


def _ref_list(refs):
    """Accept a single ref filename (max=1 uploads arrive as a string) or a
    list/tuple of them."""
    if isinstance(refs, str):
        return (refs,)
    return refs or ()


def build_minimax_h3_reference_to_video(*, prompt, seed=0, steps=MINIMAX_H3_DEFAULT_STEPS,
                                        width=MINIMAX_H3_DEFAULT_WIDTH, height=MINIMAX_H3_DEFAULT_HEIGHT,
                                        duration=5.0, scheduler="beta", ref_image_size="match",
                                        filename_prefix="wrapper/minimaxh3_ref2va",
                                        ref_images=(), ref_videos=(), ref_video_audios=(),
                                        ref_audios=(), first_frame=None, last_frame=None,
                                        unet_name=MINIMAX_H3_REF2VA_UNET_INT8,
                                        clip_name=MINIMAX_H3_CLIP_NVFP4):
    """MiniMax H3 reference-to-video (ref2va): reference images/videos/audio +
    prompt -> joint audio+video MP4. The prompt refers to references by tag
    (<Picture i> / <Video k> / <Audio j>) in the order they were provided.
    Up to 9 images — the node's own limit (the official workflow wires 3) —
    1 reference video (with its optional soundtrack), and 3 standalone audio
    refs. Measured on the 5090 (480p, 5 s, 20 steps, int8): 3 images 90 s, 5
    110 s, 7 120 s, 9 140 s; each image is its own reference latent, so every
    one adds to every step.

    ``first_frame`` / ``last_frame`` pin exact pixels at the video's first and
    last frame (MiniMaxH3AddGuide) on top of the references — the reference
    guide's "keyframe completion". A shot can then continue from the previous
    shot's final frame *and* hold its people to their character sheets in one
    render, which image-to-video (no references) cannot. Measured on the 5090:
    a pinned first frame opens as exactly as image-to-video does (33 dB PSNR to
    the frame), where naming it only in the prompt re-imagines it (22 dB)."""
    length = minimax_h3_length(duration)
    # The node's reference inputs are Autogrow groups, and ComfyUI addresses an
    # Autogrow slot by its flat dotted id ("ref_images.ref_image_0"). A nested
    # {"ref_images": {"ref_image_0": ...}} is not an input the node declares:
    # execution drops it without a validation error and the node runs with
    # ref_images={} — every reference silently ignored.
    ref_inputs = {}
    loader_nodes = {}
    next_id = 7
    for group, slot, loader, field, refs in (
            ("ref_images", "ref_image", "LoadImage", "image", ref_images),
            ("ref_videos", "ref_video", "LoadVideo", "file", ref_videos),
            ("ref_video_audios", "ref_video_audio", "LoadAudio", "audio", ref_video_audios),
            ("ref_audios", "ref_audio", "LoadAudio", "audio", ref_audios)):
        for i, ref in enumerate(_ref_list(refs)):
            loader_nodes[str(next_id)] = {"class_type": loader, "inputs": {field: ref}}
            if loader == "LoadVideo":
                # The slot takes frames (IMAGE); LoadVideo gives a VIDEO.
                loader_nodes[str(next_id + 1)] = {"class_type": "GetVideoComponents",
                                                  "inputs": {"video": [str(next_id), 0]}}
                next_id += 1
            ref_inputs[f"{group}.{slot}_{i}"] = [str(next_id), 0]
            next_id += 1
    # Pinned frames: loaded, cover-cropped to the canvas (the guide node crops
    # too, but the image-to-video path is cropped here, and one rule for both
    # keeps a frame pinned either way identical), then anchored in order.
    cond_ref = ["6", 0]
    for image, frame_idx in ((first_frame, 0), (last_frame, -1)):
        if not image:
            continue
        load, scale, guide = str(next_id), str(next_id + 1), str(next_id + 2)
        loader_nodes[load] = {"class_type": "LoadImage", "inputs": {"image": image}}
        loader_nodes[scale] = {"class_type": "ImageScale", "inputs": {
            "image": [load, 0], "upscale_method": "lanczos",
            "width": width, "height": height, "crop": "center"}}
        loader_nodes[guide] = {"class_type": "MiniMaxH3AddGuide", "inputs": {
            "positive": cond_ref, "vae": ["4", 0], "latent": ["6", 1],
            "image": [scale, 0], "frame_idx": frame_idx}}
        cond_ref = [guide, 0]
        next_id += 3
    head = _minimax_h3_head(prompt, width, height, length, unet_name, clip_name,
                            MINIMAX_H3_VIDEO_VAE, MINIMAX_H3_AUDIO_VAE, {})
    head["6"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
        "prompt": prompt, "width": width, "height": height, "length": length,
        "ref_image_size": ref_image_size, **ref_inputs}}
    tail = _minimax_h3_tail(["2", 0], cond_ref, ["6", 1], seed, steps, scheduler,
                            next_id, filename_prefix)
    return {**head, **loader_nodes, **tail}

# The H3 prompt format (alignment instruction + timed shots +
# overall_soundscape + non_diegetic_music, or six labelled sections in
# reference mode) is a lot to hand-write, so `prompt` takes plain intent and an
# LLM turns it into the real thing using MiniMax's own authoring guide. See
# api_wrapper/prompt_rewriter.py. `raw_prompt` is the escape hatch for callers
# that already have an H3 prompt.
MINIMAX_H3_PROMPT_FORM_EXTRA = {
    "raw_prompt": {"type": "string",
                   "description": "A ready-made MiniMax H3 prompt, used exactly as given. Supplying "
                                  "this skips the LLM rewrite entirely -- nothing is sent to MiMo and "
                                  "no API key is needed. Use it to reproduce an earlier generation "
                                  "verbatim, or when you have written the H3 structure yourself. "
                                  "If both 'prompt' and 'raw_prompt' are sent, 'raw_prompt' wins."},
    "llm_image": {"type": "string", "format": "binary",
                  "description": "Optional image given to the prompt-writing model as visual context, "
                                 "so the H3 prompt it writes describes what is actually in your "
                                 "footage (subjects, clothing, setting, lighting, framing) instead of "
                                 "guessing from the text. Purely an input to the rewrite -- it is not "
                                 "a keyframe and never reaches H3 itself. When omitted, the task's own "
                                 "first image is used instead (the 'image' first frame, or "
                                 "'ref_images' #1), so image- and reference-driven generations get "
                                 "visual grounding for free. Ignored when 'raw_prompt' is set."},
    "llm_model": {"type": "string", "enum": list(MINIMAX_H3_LLM_MODELS), "default": "auto",
                  "description": "Which Xiaomi MiMo model writes the H3 prompt. Leave it on 'auto' "
                                 "unless you have a reason not to: it picks mimo-v2.6-flash, which "
                                 "reads the context image when there is one. mimo-v2.6-pro and "
                                 "mimo-v2.5 read images too, but mimo-v2.5-pro is text-only and the "
                                 "API rejects any request carrying an image. Pinning mimo-v2.5-pro together with an uploaded "
                                 "llm_image is refused with a 400; the context image the wrapper adds "
                                 "on its own (your keyframe or first reference) is instead skipped, "
                                 "and the response says so. Requires MIMO_API_KEY on the server."},
}

MINIMAX_H3_FORM_EXTRA = {
    "width": {"type": "integer", "minimum": 32, "maximum": 8192, "default": MINIMAX_H3_DEFAULT_WIDTH,
              "description": "Output width, rounded up to a multiple of 32. Width*height is capped at "
                             f"H3's native canvas ({MINIMAX_H3_MAX_PIXELS} px = 1344x768) and a larger "
                             "request is scaled down keeping its aspect ratio. Separately, "
                             "width*height*frames must fit this host's memory budget "
                             f"({MINIMAX_H3_MAX_PIXEL_FRAMES // 1000000} megapixel-frames, i.e. up to "
                             f"{MINIMAX_H3_MAX_PIXEL_FRAMES // MINIMAX_H3_MAX_FRAMES // 1000} kilopixels "
                             f"at the full {MINIMAX_H3_MAX_FRAMES} frames) or the request is rejected "
                             f"with a 400. The default {MINIMAX_H3_DEFAULT_WIDTH}x"
                             f"{MINIMAX_H3_DEFAULT_HEIGHT} is the largest canvas that finished a "
                             "20-step job on this box with memory to spare."},
    "height": {"type": "integer", "minimum": 32, "maximum": 8192, "default": MINIMAX_H3_DEFAULT_HEIGHT,
               "description": "Output height, rounded up to a multiple of 32. See width for the caps."},
    "duration": {"type": "number", "minimum": 0.2, "maximum": MINIMAX_H3_MAX_DURATION,
                 "default": MINIMAX_H3_MAX_DURATION,
                 "description": f"Video length in seconds at 24 fps, snapped up to the model's 17k+5 "
                                f"frame grid. This is the parameter that drives memory hardest: the "
                                f"video VAE tiles spatially but not temporally, so no resolution is "
                                f"small enough to buy back a longer clip. The maximum is the longest "
                                f"length verified not to OOM-kill this host "
                                f"({MINIMAX_H3_MAX_FRAMES} frames); raise it with the "
                                f"COMFY_MINIMAX_H3_MAX_FRAMES env var on a host with more RAM or swap. "
                                f"The model itself is trained for 124-362 frames (5-15 s)."},
    "steps": {"type": "integer", "minimum": 1, "maximum": 1000,
              "default": MINIMAX_H3_DEFAULT_STEPS,
              "description": "Sampling steps. Barely affects peak memory, but a longer run spends "
                             "longer at that peak, so very high step counts do make an OOM more "
                             "likely. 20 is what the official MiniMax H3 templates ship and is a "
                             "good quality/time balance; below ~15 motion and audio get mushy, "
                             "above ~30 returns diminish."},
    "scheduler": {"type": "string", "enum": ["beta", "normal", "simple"], "default": "beta",
                   "description": "Sigma scheduler; beta/normal outperform simple for reference-heavy prompts."},
    "quantization": {"type": "string", "default": MINIMAX_H3_DEFAULT_QUANT,
                     "description": "Weight precision for the UNET. 'int8' (default) is the "
                                    "int8_convrot checkpoint the official templates ship; 'fp8' is the "
                                    "same size (~21 GB) fp8_scaled build; 'bf16' is ~40 GB and will not "
                                    "fit this box. The text encoder is picked independently: the "
                                    "smallest one already on disk wins, which on Blackwell is the "
                                    "nvfp4_awq build (~16 GB vs 27 GB int8 / 52 GB bf16)."},
    **MINIMAX_H3_PROMPT_FORM_EXTRA,
}
MINIMAX_H3_REF_FORM_EXTRA = {
    **MINIMAX_H3_FORM_EXTRA,
    "image": {"type": "string", "format": "binary",
              "description": "Optional first frame: these exact pixels open the video "
                             "(anchored with MiniMaxH3AddGuide, cover-cropped to the canvas), "
                             "on top of the references -- keyframe completion. Send the same "
                             "image as ref_images #1 too, so the prompt can name it as "
                             "<Picture 1>, 'the first frame of [Shot 1]'."},
    "last_frame": {"type": "string", "format": "binary",
                   "description": "Optional last frame the video must end on, anchored the "
                                  "same way at its final frame."},
    "ref_image_size": {"type": "string", "enum": ["match", "max"], "default": "match",
                        "description": "Reference image sizing: 'match' downscales refs to the generation's pixel area (faster); 'max' keeps a 2048px short edge for stronger identity fidelity (slower). Reference tokens ride through every sampling step, so 'max' with several large refs costs both time and memory on top of the canvas budget -- keep 'match' unless identity fidelity is the priority."},
}


QWEN_IMAGE_21_FORM_EXTRA = {
    "steps": {"type": "integer", "minimum": 1, "maximum": 4096, "default": QWEN_IMAGE_21_DEFAULT_STEPS,
              "description": "Sampling steps (euler). The official templates ship 25; Qwen's own pipeline uses about 40-50."},
    "cfg": {"type": "number", "minimum": 0, "maximum": 100, "default": QWEN_IMAGE_21_DEFAULT_CFG,
            "description": "CFG scale. 1 is the official setting and ignores negative_prompt; raise it only to use one."},
    "quantization": {"type": "string", "enum": list(QWEN_IMAGE_21_QUANT_MODELS), "default": "int8",
                     "description": "Weight precision of the UNET and text encoder: int8 (the templates' "
                                    "int8_convrot, ~17 GB) or bf16 (~32 GB)."},
}


# Frame interpolation (slow motion) with the core FILM / RIFE nodes
# (comfy_extras/nodes_frame_interpolation.py). The checkpoints are the ones the
# shipped "Frame Interpolation" blueprint points at (Comfy-Org/frame_interpolation),
# and they are small: FILM 69 MB, RIFE 4.26 23 MB, both in models/frame_interpolation/.
FX_INTERP_BASE_URL = "https://huggingface.co/Comfy-Org/frame_interpolation/resolve/main/frame_interpolation"
FX_INTERP_MODELS = {
    # FILM is what the blueprint ships: slower, but it holds up on large motion.
    "film": "film_net_fp16.safetensors",
    # RIFE 4.26 is several times faster and close in quality on ordinary footage.
    "rife": "rife_v4.26.safetensors",
}
FX_INTERP_DEFAULT_MODEL = "film"
FX_INTERP_FACTORS = (2, 4)
# FrameInterpolate keeps every output frame in host RAM as float32 (12 bytes
# per pixel) on top of the decoded source, and a factor-4 1080p clip grows
# fast: 5 s at 30 fps is ~600 output frames, ~15 GB. On the rig's 32 GB box an
# over-large job would OOM-kill the server (and the queue with it), so the
# wrapper refuses anything whose output frames * pixels pass this budget.
# The default (1e9 pixel-frames, ~12 GB of output) is e.g. 4 s of 1080p30 at
# 4x, 8 s at 2x, or 18 s of 720p30 at 2x.
FX_INTERP_MAX_PIXEL_FRAMES = max(1, int(float(
    os.environ.get("COMFY_FX_INTERP_MAX_PIXEL_FRAMES", "1000000000"))))


def fx_interp_models(model):
    """The one checkpoint an fx-interp job needs (``model`` is 'film' or 'rife')."""
    filename = FX_INTERP_MODELS[model]
    return [{"folder": "frame_interpolation", "filename": filename,
             "url": f"{FX_INTERP_BASE_URL}/{filename}"}]


def fx_interp_output_frames(frames, factor):
    """Frames FrameInterpolate returns: ``factor`` per source gap plus the last."""
    if frames < 2:
        return frames
    return (frames - 1) * factor + 1


def fx_interp_check_budget(width, height, frames, factor):
    """An error detail when the interpolated clip would not fit this host's
    memory budget, else None."""
    out_frames = fx_interp_output_frames(frames, factor)
    used = width * height * out_frames
    if used <= FX_INTERP_MAX_PIXEL_FRAMES:
        return None
    max_source = FX_INTERP_MAX_PIXEL_FRAMES // (width * height * factor) + 1
    return (f"{frames} frames of {width}x{height} at {factor}x makes {out_frames} frames "
            f"({used / 1e6:.0f} megapixel-frames), above this host's "
            f"{FX_INTERP_MAX_PIXEL_FRAMES / 1e6:.0f} budget. Send at most about {max_source} "
            f"source frames at this size and factor, a smaller video, or factor 2.")


def build_fx_interp(*, video, factor=2, model_name=FX_INTERP_MODELS[FX_INTERP_DEFAULT_MODEL],
                    frames=None, filename_prefix="wrapper/fx_interp"):
    """Frame interpolation (slow motion): video -> MP4 with ``factor``x the
    frames at the source frame rate, so it plays ``factor``x slower. No audio
    (a slowed soundtrack would be pitched down; the caller keeps the original).

    The source fps is linked straight from GetVideoComponents into CreateVideo,
    so the output is retimed without the wrapper knowing the rate. ``frames``
    (the source frame count, when the caller probed it) is recorded as the
    interpolation node's expected step total, so the job's progress bar
    reserves the interpolation passes from the start instead of racing over
    the loaders (see api_wrapper/progress.py)."""
    if factor not in FX_INTERP_FACTORS:
        raise ValueError(f"factor must be one of {FX_INTERP_FACTORS}")
    interpolate = {"class_type": "FrameInterpolate", "inputs": {
        "interp_model": ["3", 0], "images": ["2", 0], "multiplier": factor}}
    if frames and frames >= 2:
        # FrameInterpolate's own ProgressBar total: (pairs) * (new frames per pair).
        interpolate["_meta"] = {"expected_steps": (frames - 1) * (factor - 1)}
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        # outputs: images, audio, fps, bit_depth, color_space
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": model_name}},
        "4": interpolate,
        # The source fps, not fps * factor: that is what makes it slow motion.
        # No audio input, so the MP4 has no audio track.
        "5": {"class_type": "CreateVideo", "inputs": {"images": ["4", 0], "fps": ["2", 2], "bit_depth": 8}},
        "6": {"class_type": "SaveVideo", "inputs": {
            "video": ["5", 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"}},
    }


FX_INTERP_FORM_EXTRA = {
    "video": {"type": "string", "format": "binary",
              "description": "The clip to slow down (mp4, webm, mov, mkv, avi or gif). Its audio is "
                             "dropped. The output keeps its size and frame rate."},
    "factor": {"type": "integer", "enum": list(FX_INTERP_FACTORS), "default": 2,
               "description": "Frames made per source frame: 2 or 4. The output has (frames-1)*factor+1 "
                              "frames played at the source fps, so it runs factor times slower."},
    "model": {"type": "string", "enum": list(FX_INTERP_MODELS), "default": FX_INTERP_DEFAULT_MODEL,
              "description": "Interpolation model: 'film' (FILM, the ComfyUI blueprint's choice; best on "
                             "large motion) or 'rife' (RIFE 4.26; several times faster). Downloaded on "
                             "first use (69 MB / 23 MB)."},
}


# Background removal for FloStudio's "Remove background": the core BiRefNet
# nodes (comfy_extras/nodes_bg_removal.py) turn every frame into a foreground
# mask, and the masks are encoded as a grayscale matte video (white = subject)
# that Flow composites itself. The checkpoints are the ones the shipped
# "Remove Background (BiRefNet)" blueprint points at (Comfy-Org/BiRefNet, MIT).
FX_BGREMOVE_BASE_URL = "https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal"
FX_BGREMOVE_MODELS = {
    # BiRefNet (ZhengPeng7/BiRefNet, MIT), 444 MB: the blueprint's model, general subjects.
    "birefnet": "birefnet.safetensors",
    # Lucida (egeorcun/lucida, MIT), 885 MB: a BiRefNet_HR fine-tune that keeps
    # fine text, illustration edges and camouflaged subjects.
    "lucida": "lucida.safetensors",
}
FX_BGREMOVE_DEFAULT_MODEL = "birefnet"
# The whole clip sits in host RAM as float32 three times over: the decoded
# frames (12 bytes per pixel), their masks (4) and the masks as RGB frames for
# the encoder (12), ~28 bytes per pixel-frame. The default budget (1e9
# pixel-frames, ~28 GB) is e.g. 16 s of 1080p30 or 36 s of 720p30.
FX_BGREMOVE_MAX_PIXEL_FRAMES = max(1, int(float(
    os.environ.get("COMFY_FX_BGREMOVE_MAX_PIXEL_FRAMES", "1000000000"))))


def fx_bgremove_models(model):
    """The one checkpoint an fx-bgremove job needs (``model`` is 'birefnet' or 'lucida')."""
    filename = FX_BGREMOVE_MODELS[model]
    return [{"folder": "background_removal", "filename": filename,
             "url": f"{FX_BGREMOVE_BASE_URL}/{filename}"}]


def fx_bgremove_check_budget(width, height, frames):
    """An error detail when the clip would not fit this host's memory budget, else None."""
    used = width * height * frames
    if used <= FX_BGREMOVE_MAX_PIXEL_FRAMES:
        return None
    return (f"{frames} frames of {width}x{height} is {used / 1e6:.0f} megapixel-frames, above this "
            f"host's {FX_BGREMOVE_MAX_PIXEL_FRAMES / 1e6:.0f} budget. Send at most about "
            f"{FX_BGREMOVE_MAX_PIXEL_FRAMES // (width * height)} frames at this size, or a smaller video.")


def build_fx_bgremove(*, video, model_name=FX_BGREMOVE_MODELS[FX_BGREMOVE_DEFAULT_MODEL],
                      frames=None, filename_prefix="wrapper/fx_bgremove"):
    """Background removal: video -> grayscale matte MP4 (white = subject) with
    the source's frames, size and frame rate, no audio.

    ``frames`` (the probed source frame count) is recorded as RemoveBackground's
    expected step total, one per frame, so the progress bar reserves the model
    passes from the start (see api_wrapper/progress.py)."""
    remove = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["3", 0], "image": ["2", 0]}}
    if frames and frames >= 2:
        remove["_meta"] = {"expected_steps": frames}
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        # outputs: images, audio, fps, bit_depth, color_space
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": model_name}},
        "4": remove,
        "5": {"class_type": "MaskToImage", "inputs": {"mask": ["4", 0]}},
        "6": {"class_type": "CreateVideo", "inputs": {"images": ["5", 0], "fps": ["2", 2], "bit_depth": 8}},
        "7": {"class_type": "SaveVideo", "inputs": {
            "video": ["6", 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"}},
    }


FX_BGREMOVE_FORM_EXTRA = {
    "video": {"type": "string", "format": "binary",
              "description": "The clip to cut out (mp4, webm, mov, mkv, avi or gif), with an even width "
                             "and height."},
    "model": {"type": "string", "enum": list(FX_BGREMOVE_MODELS), "default": FX_BGREMOVE_DEFAULT_MODEL,
              "description": "Segmentation model: 'birefnet' (BiRefNet, the ComfyUI blueprint's choice) or "
                             "'lucida' (a BiRefNet_HR fine-tune for fine text, illustration and camouflage). "
                             "Downloaded on first use (444 MB / 885 MB)."},
}


# Upscale for FloStudio's "Enhance resolution", to a short side of at least
# ``target_short_side`` and at most 4x the source.
# - esrgan: Real-ESRGAN (xinntao/Real-ESRGAN, BSD-3-Clause) through the core
#   upscale-model nodes (comfy_extras/nodes_upscale_model.py). The x2 model
#   runs when the job needs at most 2x and the x4 one above that, and the
#   result is area-scaled to the exact size. x4plus is the repackaged file the
#   shipped "Video Upscale(GAN x4)" blueprint uses; x2plus has no repackaged
#   copy, so it comes from the official release (a weights-only .pth).
# - seedvr2: the SeedVR2 3B one-step video restorer (ByteDance-Seed/SeedVR2,
#   Apache-2.0) through comfy_extras/nodes_seedvr.py, wired like the shipped
#   "Upscale Video (SeedVR2 3B Int8)" blueprint: lanczos-resize to the target,
#   then restore detail at that size. The latent is split into overlapping
#   temporal chunks sized to the free VRAM.
FX_UPSCALE_MODELS = ("esrgan", "seedvr2")
FX_UPSCALE_DEFAULT_MODEL = "esrgan"
FX_UPSCALE_MAX_FACTOR = 4
FX_UPSCALE_DEFAULT_SHORT_SIDE = 1080
FX_UPSCALE_MAX_SHORT_SIDE = 4320
FX_UPSCALE_ESRGAN = {
    2: {"folder": "upscale_models", "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"},
    4: {"folder": "upscale_models", "filename": "RealESRGAN_x4plus.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged/resolve/main/RealESRGAN_x4plus.safetensors"},
}
FX_UPSCALE_SEEDVR2_UNET = "seedvr2_3b_int8_convrot.safetensors"
FX_UPSCALE_SEEDVR2_VAE = "seedvr2_ema_vae_fp16.safetensors"
FX_UPSCALE_SEEDVR2_MODELS = [
    {"folder": "diffusion_models", "filename": FX_UPSCALE_SEEDVR2_UNET,
     "url": f"https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/{FX_UPSCALE_SEEDVR2_UNET}"},
    {"folder": "vae", "filename": FX_UPSCALE_SEEDVR2_VAE,
     "url": f"https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/vae/{FX_UPSCALE_SEEDVR2_VAE}"},
]
# Latent frames (4 pixel frames each) shared and crossfaded between SeedVR2's
# temporal chunks, so chunk seams don't pop.
FX_UPSCALE_SEEDVR2_TEMPORAL_OVERLAP = 2
# ImageUpscaleWithModel's tiling (comfy_extras/nodes_upscale_model.py), for its progress total.
FX_UPSCALE_TILE = 512
FX_UPSCALE_TILE_OVERLAP = 32
# Host RAM: esrgan holds the model's full-scale output as float32 (12 bytes per
# pixel) and briefly a clamped copy of it, so its cost is the model-scale
# pixel-frames; seedvr2 holds about four float32 copies of the clip at the
# target size (resized, padded, decoded, colour-corrected), counted as two
# pixel-frames per output pixel-frame. The default budget (1.5e9, ~18 GB per
# copy) is e.g. 13 s of 360p30 or 720p30 to 1080p with esrgan, 24 s of 540p30,
# or 12 s of anything to 1080p30 with seedvr2.
FX_UPSCALE_MAX_PIXEL_FRAMES = max(1, int(float(
    os.environ.get("COMFY_FX_UPSCALE_MAX_PIXEL_FRAMES", "1500000000"))))


def fx_upscale_size(width, height, target_short_side):
    """(out_width, out_height, factor) for a source upscaled until its short
    side reaches ``target_short_side``, by at least 1x and at most 4x. Output
    sides are even (H.264 4:2:0); the short side is rounded up so it never
    falls below the target."""
    short, long = min(width, height), max(width, height)
    factor = min(float(FX_UPSCALE_MAX_FACTOR), max(1.0, target_short_side / short))
    out_short = 2 * math.ceil(short * factor / 2)
    out_long = 2 * round(long * factor / 2)
    if width <= height:
        return out_short, out_long, factor
    return out_long, out_short, factor


def fx_upscale_esrgan_scale(factor):
    """The Real-ESRGAN model scale (2 or 4) for an upscale of ``factor``."""
    return 2 if factor <= 2 else 4


def fx_upscale_models(model, factor):
    """The checkpoints an fx-upscale job needs."""
    if model == "seedvr2":
        return list(FX_UPSCALE_SEEDVR2_MODELS)
    return [FX_UPSCALE_ESRGAN[fx_upscale_esrgan_scale(factor)]]


def fx_upscale_check_budget(model, width, height, frames, out_width, out_height, factor):
    """An error detail when the upscale would not fit this host's memory budget, else None."""
    if model == "seedvr2":
        per_frame = 2 * out_width * out_height
        work = f"seedvr2 at {out_width}x{out_height}"
    else:
        scale = fx_upscale_esrgan_scale(factor)
        per_frame = width * scale * height * scale
        work = f"the esrgan x{scale} model ({width * scale}x{height * scale} before scaling)"
    used = per_frame * frames
    if used <= FX_UPSCALE_MAX_PIXEL_FRAMES:
        return None
    return (f"{frames} frames of {width}x{height} through {work} is {used / 1e6:.0f} megapixel-frames, "
            f"above this host's {FX_UPSCALE_MAX_PIXEL_FRAMES / 1e6:.0f} budget. Send at most about "
            f"{FX_UPSCALE_MAX_PIXEL_FRAMES // per_frame} frames at this size, a smaller video, or a lower "
            f"target_short_side.")


def fx_upscale_tile_steps(width, height):
    """ImageUpscaleWithModel's progress total per frame (comfy.utils.get_tiled_scale_steps)."""
    step = FX_UPSCALE_TILE - FX_UPSCALE_TILE_OVERLAP
    rows = 1 if height <= FX_UPSCALE_TILE else math.ceil((height - FX_UPSCALE_TILE_OVERLAP) / step)
    cols = 1 if width <= FX_UPSCALE_TILE else math.ceil((width - FX_UPSCALE_TILE_OVERLAP) / step)
    return rows * cols


def build_fx_upscale(*, video, width, height, model=FX_UPSCALE_DEFAULT_MODEL,
                     upscale_model_name=FX_UPSCALE_ESRGAN[4]["filename"],
                     unet_name=FX_UPSCALE_SEEDVR2_UNET, vae_name=FX_UPSCALE_SEEDVR2_VAE,
                     seed=0, frames=None, source_size=None, filename_prefix="wrapper/fx_upscale"):
    """Upscale: video -> H.264 MP4 of exactly ``width`` x ``height`` with the
    source's frames and frame rate, no audio.

    esrgan runs ``upscale_model_name`` and area-scales its output to the size;
    seedvr2 resizes to the size and restores it with ``unet_name`` /
    ``vae_name`` (``seed`` drives its one sampling step). ``frames`` and
    ``source_size`` (the probe) give esrgan's upscale node its expected step
    total, frames x tiles, for the progress bar."""
    if model not in FX_UPSCALE_MODELS:
        raise ValueError(f"model must be one of {FX_UPSCALE_MODELS}")
    graph = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        # outputs: images, audio, fps, bit_depth, color_space
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
    }
    if model == "esrgan":
        upscale = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["3", 0], "image": ["2", 0]}}
        if frames and source_size:
            upscale["_meta"] = {"expected_steps": frames * fx_upscale_tile_steps(*source_size)}
        graph.update({
            "3": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": upscale_model_name}},
            "4": upscale,
            "5": {"class_type": "ImageScale", "inputs": {
                "image": ["4", 0], "upscale_method": "area", "width": width, "height": height, "crop": "disabled"}},
        })
        images = ["5", 0]
    else:
        tiled = {"tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8}
        graph.update({
            "3": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "5": {"class_type": "ImageScale", "inputs": {
                "image": ["2", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"}},
            # Pads to a multiple of 16 and to 4n+1 frames; post-processing crops both back.
            "6": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["5", 0]}},
            "7": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["6", 0], "vae": ["4", 0], **tiled}},
            # A list of latent chunks: conditioning and sampling run once per chunk.
            "8": {"class_type": "SeedVR2TemporalChunk", "inputs": {
                "latent": ["7", 0], "temporal_overlap": FX_UPSCALE_SEEDVR2_TEMPORAL_OVERLAP,
                "chunking_mode": "auto"}},
            "9": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["3", 0], "vae_conditioning": ["8", 0]}},
            "10": {"class_type": "KSampler", "inputs": {
                "model": ["3", 0], "positive": ["9", 0], "negative": ["9", 1], "latent_image": ["8", 0],
                "seed": seed, "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                "denoise": 1.0}},
            "11": {"class_type": "SeedVR2TemporalMerge", "inputs": {"latents": ["10", 0], "temporal_overlap": ["8", 1]}},
            "12": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["11", 0], "vae": ["4", 0], **tiled}},
            "13": {"class_type": "SeedVR2PostProcessing", "inputs": {
                "images": ["12", 0], "original_resized_images": ["5", 0], "color_correction_method": "lab"}},
        })
        images = ["13", 0]
    create, save = str(len(graph) + 1), str(len(graph) + 2)
    graph[create] = {"class_type": "CreateVideo", "inputs": {"images": images, "fps": ["2", 2], "bit_depth": 8}}
    graph[save] = {"class_type": "SaveVideo", "inputs": {
        "video": [create, 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"}}
    return graph


# Image mode: a still in ``image`` comes back as a PNG (LoadImage ... SaveImage),
# so it never passes through H.264 (yuv420 and a one-frame clip's encode). The
# output is ``width`` x ``height`` exactly (any sides; no 4:2:0 even rule), up to
# FX_UPSCALE_IMAGE_MAX_SIDE a side and FX_UPSCALE_IMAGE_MAX_FACTOR times the
# source; seedvr2 is the default model for a still.
#
# ``refine=tiled`` (for 8K-class outputs): the model upscales to at most
# FX_UPSCALE_REFINE_BASE_LONG_SIDE on the long side (SeedVR2's sweet spot), the
# picture is lanczos-scaled to the output, and FLUX.2 [klein] 4B (Apache-2.0)
# redraws it in overlapping tiles at a low denoise, so the last 2x is new texture
# rather than a smear: Ultimate SD Upscale's idea, with the core
# SplitImageToTileList / ImageMergeTileList nodes (sine-feathered blend).
FX_UPSCALE_IMAGE_DEFAULT_MODEL = "seedvr2"
FX_UPSCALE_IMAGE_MAX_SIDE = 8192
FX_UPSCALE_IMAGE_MAX_FACTOR = 8
FX_UPSCALE_REFINES = ("none", "tiled")
FX_UPSCALE_REFINE_BASE_LONG_SIDE = 3840
FX_UPSCALE_REFINE_TILE = 1024
FX_UPSCALE_REFINE_OVERLAP = 128
FX_UPSCALE_REFINE_DEFAULT_DENOISE = 0.3
FX_UPSCALE_REFINE_STEPS = 8
FX_UPSCALE_REFINE_DEFAULT_PROMPT = "a sharp, highly detailed photograph with fine natural texture"
FX_UPSCALE_REFINE_UNET = "flux-2-klein-4b-fp8.safetensors"
FX_UPSCALE_REFINE_CLIP = "qwen_3_4b.safetensors"
FX_UPSCALE_REFINE_MODELS = [
    {"folder": "diffusion_models", "filename": FX_UPSCALE_REFINE_UNET,
     "url": f"https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/{FX_UPSCALE_REFINE_UNET}"},
    {"folder": "text_encoders", "filename": FX_UPSCALE_REFINE_CLIP,
     "url": f"https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/{FX_UPSCALE_REFINE_CLIP}"},
    {"folder": "vae", "filename": FLUX2_KLEIN_9B_VAE,
     "url": f"https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/{FLUX2_KLEIN_9B_VAE}"},
]


def fx_upscale_image_size(width, height, fields):
    """(out_width, out_height) of an image-mode upscale: ``width`` and
    ``height`` when both are sent, else the source scaled until its short side
    reaches ``target_short_side`` (rounded, aspect kept). Raises ValueError
    with the reason when the size is not one this mode makes."""
    def whole(name):
        value = fields.get(name)
        if value in (None, ""):
            return None
        try:
            number = int(str(value).strip())
        except ValueError:
            raise ValueError(f"{name} must be a whole number.") from None
        if not 16 <= number <= FX_UPSCALE_IMAGE_MAX_SIDE:
            raise ValueError(f"{name} must be from 16 to {FX_UPSCALE_IMAGE_MAX_SIDE}.")
        return number

    out_w, out_h, short = whole("width"), whole("height"), whole("target_short_side")
    if (out_w is None) != (out_h is None):
        raise ValueError("Send width and height together.")
    if out_w is None:
        target = short or FX_UPSCALE_DEFAULT_SHORT_SIDE
        k = target / min(width, height)
        out_w, out_h = max(1, round(width * k)), max(1, round(height * k))
        if max(out_w, out_h) > FX_UPSCALE_IMAGE_MAX_SIDE:
            raise ValueError(f"That would be {out_w}x{out_h}; at most {FX_UPSCALE_IMAGE_MAX_SIDE} a side.")
    factor = max(out_w / width, out_h / height)
    if factor > FX_UPSCALE_IMAGE_MAX_FACTOR:
        raise ValueError(f"{width}x{height} to {out_w}x{out_h} is {factor:.1f}x; at most "
                         f"{FX_UPSCALE_IMAGE_MAX_FACTOR}x.")
    return out_w, out_h


def fx_upscale_refine_base(out_width, out_height, width, height):
    """The size the model upscales to before a tiled refine: the output scaled
    down to FX_UPSCALE_REFINE_BASE_LONG_SIDE on its long side, never below the
    source."""
    k = min(1.0, FX_UPSCALE_REFINE_BASE_LONG_SIDE / max(out_width, out_height))
    k = max(k, max(width / out_width, height / out_height))
    return max(1, round(out_width * k)), max(1, round(out_height * k))


def fx_upscale_refine_tile_size(width, height):
    """The refine's (tile_width, tile_height): FX_UPSCALE_REFINE_TILE, or a side
    shorter than that floored to a multiple of 16. A tile taller than the
    picture would be cut to its height, which the VAE then trims to a multiple
    of 16, and ImageMergeTileList leaves the trimmed rows black."""
    def side(length):
        return min(FX_UPSCALE_REFINE_TILE, max(64, length - length % 16))

    return side(width), side(height)


def fx_upscale_refine_tiles(width, height):
    """How many tiles the refine redraws (SplitImageToTileList's grid)."""
    overlap = FX_UPSCALE_REFINE_OVERLAP

    def count(side, tile):
        stride = round(max(tile * 0.25, tile - overlap))
        return 1 if side <= tile else 1 + math.ceil((side - tile) / stride)

    tile_w, tile_h = fx_upscale_refine_tile_size(width, height)
    return count(width, tile_w) * count(height, tile_h)


def _fx_upscale_model_stage(graph, image, model, width, height, *, upscale_model_name, unet_name, vae_name, seed,
                            source_size=None, first_id):
    """Add one esrgan or seedvr2 upscale of ``image`` to ``width`` x ``height``
    to ``graph`` (node ids from ``first_id``); returns the output's link."""
    ids = [str(first_id + i) for i in range(11)]
    if model == "esrgan":
        upscale = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": [ids[0], 0], "image": image}}
        if source_size:
            upscale["_meta"] = {"expected_steps": fx_upscale_tile_steps(*source_size)}
        graph.update({
            ids[0]: {"class_type": "UpscaleModelLoader", "inputs": {"model_name": upscale_model_name}},
            ids[1]: upscale,
            ids[2]: {"class_type": "ImageScale", "inputs": {
                "image": [ids[1], 0], "upscale_method": "area", "width": width, "height": height, "crop": "disabled"}},
        })
        return [ids[2], 0]
    tiled = {"tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8}
    graph.update({
        ids[0]: {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        ids[1]: {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        ids[2]: {"class_type": "ImageScale", "inputs": {
            "image": image, "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"}},
        ids[3]: {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": [ids[2], 0]}},
        ids[4]: {"class_type": "VAEEncodeTiled", "inputs": {"pixels": [ids[3], 0], "vae": [ids[1], 0], **tiled}},
        ids[5]: {"class_type": "SeedVR2TemporalChunk", "inputs": {
            "latent": [ids[4], 0], "temporal_overlap": FX_UPSCALE_SEEDVR2_TEMPORAL_OVERLAP, "chunking_mode": "auto"}},
        ids[6]: {"class_type": "SeedVR2Conditioning", "inputs": {"model": [ids[0], 0], "vae_conditioning": [ids[5], 0]}},
        ids[7]: {"class_type": "KSampler", "inputs": {
            "model": [ids[0], 0], "positive": [ids[6], 0], "negative": [ids[6], 1], "latent_image": [ids[5], 0],
            "seed": seed, "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        ids[8]: {"class_type": "SeedVR2TemporalMerge", "inputs": {"latents": [ids[7], 0], "temporal_overlap": [ids[5], 1]}},
        ids[9]: {"class_type": "VAEDecodeTiled", "inputs": {"samples": [ids[8], 0], "vae": [ids[1], 0], **tiled}},
        ids[10]: {"class_type": "SeedVR2PostProcessing", "inputs": {
            "images": [ids[9], 0], "original_resized_images": [ids[2], 0], "color_correction_method": "lab"}},
    })
    return [ids[10], 0]


def build_fx_upscale_image(*, image, width, height, model=FX_UPSCALE_IMAGE_DEFAULT_MODEL, refine="none",
                           upscale_model_name=FX_UPSCALE_ESRGAN[4]["filename"],
                           unet_name=FX_UPSCALE_SEEDVR2_UNET, vae_name=FX_UPSCALE_SEEDVR2_VAE,
                           seed=0, source_size=None, base_size=None, prompt=FX_UPSCALE_REFINE_DEFAULT_PROMPT,
                           denoise=FX_UPSCALE_REFINE_DEFAULT_DENOISE, filename_prefix="wrapper/fx_upscale"):
    """Upscale a still: image -> PNG of exactly ``width`` x ``height``.

    ``refine='none'``: the model (esrgan or seedvr2, as in build_fx_upscale)
    makes the output size directly. ``refine='tiled'``: the model makes
    ``base_size`` (fx_upscale_refine_base), lanczos takes it to the output, and
    klein 4B redraws FX_UPSCALE_REFINE_TILE tiles at ``denoise`` from ``prompt``
    (each tile's own latent is also its reference, so it keeps the picture),
    merged back with a sine-feathered overlap."""
    if model not in FX_UPSCALE_MODELS:
        raise ValueError(f"model must be one of {FX_UPSCALE_MODELS}")
    if refine not in FX_UPSCALE_REFINES:
        raise ValueError(f"refine must be one of {FX_UPSCALE_REFINES}")
    graph = {"1": {"class_type": "LoadImage", "inputs": {"image": image}}}
    stage_w, stage_h = (base_size or (width, height)) if refine == "tiled" else (width, height)
    out = _fx_upscale_model_stage(graph, ["1", 0], model, stage_w, stage_h, upscale_model_name=upscale_model_name,
                                  unet_name=unet_name, vae_name=vae_name, seed=seed, source_size=source_size,
                                  first_id=2)
    if refine == "tiled":
        tile_w, tile_h = fx_upscale_refine_tile_size(width, height)
        overlap = FX_UPSCALE_REFINE_OVERLAP
        graph.update({
            "20": {"class_type": "ImageScale", "inputs": {
                "image": out, "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"}},
            "21": {"class_type": "UNETLoader", "inputs": {"unet_name": FX_UPSCALE_REFINE_UNET, "weight_dtype": "default"}},
            "22": {"class_type": "CLIPLoader", "inputs": {"clip_name": FX_UPSCALE_REFINE_CLIP, "type": "flux2", "device": "default"}},
            "23": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_KLEIN_9B_VAE}},
            "24": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["22", 0], "text": prompt or FX_UPSCALE_REFINE_DEFAULT_PROMPT}},
            # A list of tiles: everything downstream runs once per tile.
            "25": {"class_type": "SplitImageToTileList", "inputs": {
                "image": ["20", 0], "tile_width": tile_w, "tile_height": tile_h, "overlap": overlap}},
            "26": {"class_type": "VAEEncode", "inputs": {"pixels": ["25", 0], "vae": ["23", 0]}},
            "27": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["24", 0], "latent": ["26", 0]}},
            "28": {"class_type": "CFGGuider", "inputs": {"model": ["21", 0], "positive": ["27", 0], "negative": ["27", 0], "cfg": 1.0}},
            "29": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "30": {"class_type": "Flux2Scheduler", "inputs": {"steps": FX_UPSCALE_REFINE_STEPS, "width": tile_w, "height": tile_h}},
            "31": {"class_type": "SplitSigmasDenoise", "inputs": {"sigmas": ["30", 0], "denoise": denoise}},
            "32": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed % (2 ** 63)}},
            "33": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["32", 0], "guider": ["28", 0], "sampler": ["29", 0], "sigmas": ["31", 1], "latent_image": ["26", 0]}},
            "34": {"class_type": "VAEDecode", "inputs": {"samples": ["33", 0], "vae": ["23", 0]}},
            "35": {"class_type": "ImageMergeTileList", "inputs": {
                "image_list": ["34", 0], "final_width": width, "final_height": height, "overlap": overlap}},
        })
        out = ["35", 0]
    graph["40"] = {"class_type": "SaveImage", "inputs": {"images": out, "filename_prefix": filename_prefix}}
    return graph


def build_fx_upscale_any(*, video=None, image=None, **kwargs):
    """fx-upscale's builder: image mode for a still in ``image``, else the clip."""
    if image is not None:
        return build_fx_upscale_image(image=image, **{k: v for k, v in kwargs.items() if k != "frames"})
    image_only = ("refine", "base_size", "prompt", "denoise")
    return build_fx_upscale(video=video, **{k: v for k, v in kwargs.items() if k not in image_only})


FX_UPSCALE_FORM_EXTRA = {
    "video": {"type": "string", "format": "binary",
              "description": "The clip to upscale (mp4, webm, mov, mkv, avi or gif). Its audio is dropped. "
                             "Send this or 'image'."},
    "image": {"type": "string", "format": "binary",
              "description": "Image mode: a still (png, jpg, webp, bmp or gif) upscaled to a PNG, never through a "
                             "video codec. Send this or 'video'."},
    "width": {"type": "integer", "minimum": 16, "maximum": FX_UPSCALE_IMAGE_MAX_SIDE,
              "description": "Image mode: the exact output width (with height). Omit both to use "
                             "target_short_side."},
    "height": {"type": "integer", "minimum": 16, "maximum": FX_UPSCALE_IMAGE_MAX_SIDE,
               "description": "Image mode: the exact output height (with width)."},
    "refine": {"type": "string", "enum": list(FX_UPSCALE_REFINES), "default": "none",
               "description": "Image mode: 'tiled' upscales with the model to at most "
                              f"{FX_UPSCALE_REFINE_BASE_LONG_SIDE} px long, then redraws the full-size picture in "
                              f"{FX_UPSCALE_REFINE_TILE} px tiles with FLUX.2 [klein] 4B at a low denoise (new "
                              "texture for 8K outputs). 'none' upscales with the model alone."},
    "denoise": {"type": "number", "minimum": 0.05, "maximum": 0.6, "default": FX_UPSCALE_REFINE_DEFAULT_DENOISE,
                "description": "Image mode, refine=tiled: how much of each tile is redrawn."},
    "prompt": {"type": "string",
               "description": "Image mode, refine=tiled: what the picture shows (guides the redrawn texture)."},
    "model": {"type": "string", "enum": list(FX_UPSCALE_MODELS), "default": FX_UPSCALE_DEFAULT_MODEL,
              "description": "'esrgan' (Real-ESRGAN x2/x4, fast; 67 MB each) or 'seedvr2' (SeedVR2 3B int8 "
                             "video restoration, slower and much sharper on real footage; 3.5 GB + 0.5 GB "
                             "VAE). Downloaded on first use."},
    "target_short_side": {"type": "integer", "minimum": 16, "maximum": FX_UPSCALE_MAX_SHORT_SIDE,
                          "default": FX_UPSCALE_DEFAULT_SHORT_SIDE,
                          "description": "Upscale until the short side is at least this many pixels (e.g. "
                                         "the project canvas's short side), by at least 1x and at most 4x. "
                                         "Output sides are even."},
}


def upright_size(width, height, orientation):
    """A still's (width, height) once its EXIF orientation is applied, the way
    LoadImage (ImageOps.exif_transpose) hands it to the graph: orientations 5-8
    turn it a quarter, which swaps the sides."""
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)


def _first_frame(image):
    """LoadImage + the first frame only: a GIF loads as a batch of frames, and
    these workflows answer for one picture."""
    return {
        "2": {"class_type": "LoadImage", "inputs": {"image": image}},
        "3": {"class_type": "ImageFromBatch", "inputs": {"image": ["2", 0], "batch_index": 0, "length": 1}},
    }


# Image Studio's "select by name": SAM 3.1 (facebook/sam3.1, SAM License)
# repackaged as one checkpoint -- detector, tracker and its own CLIP text
# encoder -- the one the shipped "Image Segmentation (SAM3)" blueprint loads.
IMAGE_SEGMENT_CKPT = "sam3.1_multiplex_fp16.safetensors"
IMAGE_SEGMENT_MODELS = [
    {"folder": "checkpoints", "filename": IMAGE_SEGMENT_CKPT,
     "url": f"https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/{IMAGE_SEGMENT_CKPT}"},
]
# SAM3_Detect's own default: decoder refinement passes per mask.
IMAGE_SEGMENT_REFINE_ITERATIONS = 2


def _sam3_detect_inputs(graph, request, image, text_node):
    """SAM3_Detect's inputs for a segment_request() on ``image``, with the SAM
    3.1 checkpoint in node "1": one mask, the union of what is selected. A
    phrase adds its text encoding to ``graph`` as node ``text_node``."""
    detect = {"model": ["1", 0], "image": image, "threshold": request["threshold"],
              "refine_iterations": IMAGE_SEGMENT_REFINE_ITERATIONS, "individual_masks": False}
    if request["mode"] == "prompt":
        # SAM 3's own text encoder, from the checkpoint's CLIP output.
        graph[text_node] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": request["text"]}}
        detect["conditioning"] = [text_node, 0]
    elif request["mode"] == "boxes":
        # Wrapped: a bare list is read as a link to another node.
        detect["bboxes"] = {"__value__": request["boxes"]}
    else:
        detect["positive_coords"] = json.dumps(request["positive"])
        if request["negative"]:
            detect["negative_coords"] = json.dumps(request["negative"])
    return detect


def build_image_segment(*, image, request, width, height, ckpt_name=IMAGE_SEGMENT_CKPT,
                        filename_prefix="wrapper/image_segment"):
    """Image segmentation: still + a phrase, boxes or points -> grayscale PNG
    mask(s) at the still's upright size, white = selected.

    ``request`` is api_wrapper.segment.segment_request()'s output. The PNGs
    are one batch: the union of everything selected first, then (with
    individual_masks) one per object. SAM3_Detect gives either the union or
    the per-object masks, not both, so that case runs it twice -- the second
    pass is a fraction of a second -- and batches the union in front, which
    also keeps a request that finds nothing a single black mask rather than
    no file. The detections reach the job's history through a PreviewAny
    tagged as the job's JSON answer (GET /wrapper/jobs/{job_id}/text)."""
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
             **_first_frame(image)}
    detect = _sam3_detect_inputs(graph, request, ["3", 0], "4")
    mode = request["mode"]
    tag = {"mode": mode, "width": width, "height": height, "individual_masks": request["individual_masks"]}
    if mode == "boxes":
        tag["boxes"] = [image_segment.box_xywh(box) for box in request["boxes"]]
    graph["5"] = {"class_type": "SAM3_Detect", "inputs": detect}
    masks, detections = ["5", 0], ["5", 1]
    if request["individual_masks"]:
        graph["6"] = {"class_type": "SAM3_Detect", "inputs": {**detect, "individual_masks": True}}
        graph["7"] = {"class_type": "BatchMasksNode", "inputs": {"masks.mask0": ["5", 0], "masks.mask1": ["6", 0]}}
        masks, detections = ["7", 0], ["6", 1]
    graph["8"] = {"class_type": "MaskToImage", "inputs": {"mask": masks}}
    graph["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}}
    graph["10"] = tag_result({"class_type": "PreviewAny", "inputs": {"source": detections}}, "image-segment", **tag)
    return graph


IMAGE_SEGMENT_FORM_EXTRA = {
    "image": {"type": "string", "format": "binary",
              "description": "The picture (png, jpg, webp, bmp or gif; a GIF's first frame). Every coordinate "
                             "below, and the masks, use its upright size: after its EXIF orientation."},
    "prompt": {"type": "string",
               "description": "What to select, as a short noun phrase: 'the sky', 'person', 'red car'. Commas "
                              f"separate phrases (up to {image_segment.SEGMENT_MAX_PHRASES}), each found "
                              "separately. SAM 3 finds every instance of a concept; it does not resolve "
                              "relations like 'the person on the left' -- send that object's box instead. "
                              "Send exactly one of prompt, boxes or positive_coords."},
    "boxes": {"type": "string",
              "description": "JSON list of pixel boxes [[x, y, w, h], ...] (up to "
                             f"{image_segment.SEGMENT_MAX_OBJECTS}); each is segmented on its own. A box past "
                             "the edge is clipped to the picture."},
    "positive_coords": {"type": "string",
                        "description": "JSON list of pixel points [[x, y], ...] on the thing to select "
                                       "(tap-to-select). Points make one mask."},
    "negative_coords": {"type": "string",
                        "description": "JSON list of pixel points [[x, y], ...] to keep out of the selection; "
                                       "only with positive_coords."},
    "threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": image_segment.SEGMENT_DEFAULT_THRESHOLD,
                  "description": "Detection confidence a phrase's object needs to be selected."},
    "max_objects": {"type": "integer", "minimum": 1, "maximum": image_segment.SEGMENT_MAX_OBJECTS,
                    "default": image_segment.SEGMENT_DEFAULT_MAX_OBJECTS,
                    "description": "Most objects kept per phrase, best first."},
    "individual_masks": {"type": "boolean", "default": False,
                         "description": "Also save one mask per object (phrases and boxes), after the union. "
                                        "The response body stays the union; GET /api/wrapper/jobs/{job_id}/text "
                                        "pairs each object with its mask's URL."},
}


# FloStudio's "Track & replace", step 2: follow the object a member picked
# through the shot. Frame 0 of the clip is the anchor (Flow cuts each pass so,
# reversing backward ones). SAM3_VideoTrack (comfy_extras/nodes_sam3.py)
# propagates the frame-0 mask with SAM 3.1's memory tracker -- the checkpoint
# image-segment already loads -- and the result is a grey matte (white =
# object) plus per-frame boxes as the job's JSON answer (api_wrapper/track.py).
FX_TRACK_MODELS = IMAGE_SEGMENT_MODELS


def build_fx_track(*, video, request, width, height, fps, mask=None, frames=None,
                   ckpt_name=IMAGE_SEGMENT_CKPT, filename_prefix="wrapper/fx_track"):
    """Object tracking: video (+ a frame-0 mask PNG, or a phrase, boxes or
    points on frame 0) -> grayscale matte MP4 with the source's frames, size
    and frame rate, no audio; the per-frame boxes reach the job's history
    through a PreviewAny tagged as its JSON answer.

    ``request`` is api_wrapper.track.track_request()'s output; ``width``,
    ``height`` and ``fps`` (the probed video) go into the answer. Whatever
    frame 0's selection holds is tracked as one object. ``frames`` (the probed
    frame count) is the tracker's expected step total, one per frame (see
    api_wrapper/progress.py)."""
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
             "4": {"class_type": "LoadVideo", "inputs": {"file": video}},
             # outputs: images, audio, fps, bit_depth, color_space
             "5": {"class_type": "GetVideoComponents", "inputs": {"video": ["4", 0]}}}
    if request["mode"] == "mask":
        if not mask:
            raise ValueError("a mask-anchored track needs the mask")
        graph.update(_first_frame(mask))
        graph["8"] = {"class_type": "ImageToMask", "inputs": {"image": ["3", 0], "channel": "red"}}
    else:
        graph["6"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["5", 0], "batch_index": 0, "length": 1}}
        graph["8"] = {"class_type": "SAM3_Detect", "inputs": _sam3_detect_inputs(graph, request, ["6", 0], "7")}
    track = {"class_type": "SAM3_VideoTrack", "inputs": {
        "images": ["5", 0], "model": ["1", 0], "initial_mask": ["8", 0],
        "detection_threshold": request["threshold"], "max_objects": 1,
        "detect_interval": request["detect_interval"]}}
    if frames:
        track["_meta"] = {"expected_steps": frames}
    graph.update({
        "9": track,
        "10": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["9", 0], "object_indices": ""}},
        "11": {"class_type": "MaskToImage", "inputs": {"mask": ["10", 0]}},
        "12": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "fps": ["5", 2], "bit_depth": 8}},
        "13": {"class_type": "SaveVideo", "inputs": {
            "video": ["12", 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"}},
        "14": {"class_type": "SAM3_TrackToBBoxes", "inputs": {"track_data": ["9", 0], "object_indices": ""}},
        "15": tag_result({"class_type": "PreviewAny", "inputs": {"source": ["14", 0]}}, "fx-track",
                         width=width, height=height, fps=fps),
    })
    return graph


FX_TRACK_FORM_EXTRA = {
    "video": {"type": "string", "format": "binary",
              "description": "The pass to track (mp4, webm, mov, mkv, avi or gif), with an even width and height. "
                             "Frame 0 is the anchor: track backward by sending the clip reversed."},
    "mask": {"type": "string", "format": "binary",
             "description": "Frame 0's selection as a PNG at the video's size, white = object (e.g. "
                            "image-segment's mask). Preferred; with it the fields below are ignored."},
    "prompt": {"type": "string",
               "description": "Without a mask: what to track as a short phrase, found on frame 0 (its best match). "
                              "Send exactly one of prompt, boxes or positive_coords."},
    "boxes": {"type": "string",
              "description": "Without a mask: JSON list of pixel boxes [[x, y, w, h], ...] on frame 0 (up to "
                             f"{object_track.TRACK_MAX_BOXES}), tracked together as one object."},
    "positive_coords": {"type": "string",
                        "description": "Without a mask: JSON list of pixel points [[x, y], ...] on the object in "
                                       f"frame 0 (up to {image_segment.SEGMENT_MAX_POINTS})."},
    "negative_coords": {"type": "string",
                        "description": "JSON list of pixel points [[x, y], ...] to keep out of frame 0's "
                                       "selection; only with positive_coords."},
    "threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": image_segment.SEGMENT_DEFAULT_THRESHOLD,
                  "description": "Detection confidence a phrase's match needs on frame 0."},
    "detect_interval": {"type": "integer", "minimum": 1, "maximum": object_track.TRACK_MAX_DETECT_INTERVAL,
                        "default": object_track.TRACK_DEFAULT_DETECT_INTERVAL,
                        "description": "SAM3_VideoTrack's detector interval in frames (1 = every frame)."},
}


# FloStudio's "Track & replace", AI mode: regenerate the tracked object inside
# its (dilated) matte from a photo, with Wan 2.1 VACE 14B
# (Wan-AI/Wan2.1-VACE-14B, Apache-2.0) through the core WanVaceToVideo node,
# wired like the shipped "Video Inpainting (Wan2.1 VACE)" blueprint: the
# blueprint's files (Comfy-Org's repackage and Kijai's CausVid speed LoRA at
# 0.3), the fp16 UNET loaded as fp8_e4m3fn_fast, 6 uni_pc steps at cfg 1 and
# shift 5. No int8_convrot or fp8 build of VACE 14B is published by Comfy-Org,
# so the 35 GB fp16 file is fetched and cast on load (~17 GB of VRAM).
FX_REPLACE_BASE_URL = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files"
FX_REPLACE_UNET = "wan2.1_vace_14B_fp16.safetensors"
FX_REPLACE_WEIGHT_DTYPE = "fp8_e4m3fn_fast"
FX_REPLACE_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
FX_REPLACE_VAE = "wan_2.1_vae.safetensors"
FX_REPLACE_LORA = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
FX_REPLACE_MODELS = [
    {"folder": "diffusion_models", "filename": FX_REPLACE_UNET,
     "url": f"{FX_REPLACE_BASE_URL}/diffusion_models/{FX_REPLACE_UNET}"},
    {"folder": "text_encoders", "filename": FX_REPLACE_CLIP,
     "url": f"{FX_REPLACE_BASE_URL}/text_encoders/{FX_REPLACE_CLIP}"},
    {"folder": "vae", "filename": FX_REPLACE_VAE, "url": f"{FX_REPLACE_BASE_URL}/vae/{FX_REPLACE_VAE}"},
    {"folder": "loras", "filename": FX_REPLACE_LORA,
     "url": f"https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/{FX_REPLACE_LORA}"},
]
FX_REPLACE_LORA_STRENGTH = 0.3
FX_REPLACE_SHIFT = 5.0
FX_REPLACE_DEFAULT_STEPS = 6
FX_REPLACE_CFG = 1.0
# The quick tier swaps CausVid (6 steps, strength 0.3) for lightx2v's 4-step
# distillation of Wan 2.1 14B (Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2,
# Kijai/WanVideo_comfy, rank 32): its documented use is 4 steps, cfg 1, strength 1.
FX_REPLACE_QUICK_LORA = "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors"
FX_REPLACE_QUICK_LORA_STRENGTH = 1.0
FX_REPLACE_QUICK_STEPS = 4
FX_REPLACE_MODELS.append({
    "folder": "loras", "filename": FX_REPLACE_QUICK_LORA,
    "url": f"https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/{FX_REPLACE_QUICK_LORA}"})
# Generation canvases by tier, as pixel areas: 'standard' and 'hd' are Wan
# 2.1's 480p and 720p; 'quick' is a quarter of a megapixel.
FX_REPLACE_TIERS = {"quick": 640 * 384, "standard": 832 * 480, "hd": 1280 * 720}
FX_REPLACE_DEFAULT_TIER = "standard"
# What each tier samples with: the speed LoRA, its strength and the steps.
FX_REPLACE_TIER_SETUP = {
    "quick": {"lora": FX_REPLACE_QUICK_LORA, "lora_strength": FX_REPLACE_QUICK_LORA_STRENGTH,
              "steps": FX_REPLACE_QUICK_STEPS},
    "standard": {"lora": FX_REPLACE_LORA, "lora_strength": FX_REPLACE_LORA_STRENGTH,
                 "steps": FX_REPLACE_DEFAULT_STEPS},
    "hd": {"lora": FX_REPLACE_LORA, "lora_strength": FX_REPLACE_LORA_STRENGTH,
           "steps": FX_REPLACE_DEFAULT_STEPS},
}
# ``interpolate``: frames made per generated gap (1 = none), so a clip generated
# at half the frame rate is played back at the full one.
FX_REPLACE_INTERPOLATE = (1, 2)
# The exact generation canvas a caller may ask for (``canvas``, 'WxH').
FX_REPLACE_CANVAS_SIDE = (64, 2048)
# Wan 2.1 is trained on 81-frame clips; longer ones sample in overlapping
# windows of that length (WanContextWindowsManual), blended where they meet.
FX_REPLACE_WINDOW = 81
FX_REPLACE_WINDOW_OVERLAP = 30
# What WanVaceToVideo reads as "no picture here": the object's pixels are
# painted this grey so the model never sees what it replaces.
FX_REPLACE_GREY = 0x808080
# Sampling time and VRAM grow with the generated pixel-frames. The default
# (3e8) takes 10 s of 30 fps at hd (1280x720x301) or 20 s at standard.
FX_REPLACE_MAX_PIXEL_FRAMES = max(1, int(float(
    os.environ.get("COMFY_FX_REPLACE_MAX_PIXEL_FRAMES", "300000000"))))


def fx_replace_models(tier=FX_REPLACE_DEFAULT_TIER, interpolate=1):
    """The files one fx-replace job at ``tier`` needs: the VACE stack, the
    tier's speed LoRA, and the interpolation checkpoint when it interpolates."""
    wanted = FX_REPLACE_TIER_SETUP[tier]["lora"]
    models = [m for m in FX_REPLACE_MODELS if m["folder"] != "loras" or m["filename"] == wanted]
    if interpolate > 1:
        models += fx_interp_models(FX_INTERP_DEFAULT_MODEL)
    return models


def fx_replace_all_models():
    """Every file any tier needs (what ``prefetch fx-replace`` fetches)."""
    return FX_REPLACE_MODELS + fx_interp_models(FX_INTERP_DEFAULT_MODEL)


def fx_replace_size(width, height, tier):
    """The generation canvas for a width x height clip: the tier's pixel area
    at the clip's aspect, each side a multiple of 16."""
    scale = math.sqrt(FX_REPLACE_TIERS[tier] / (width * height))
    return max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16)


def fx_replace_canvas(text):
    """A ``canvas`` form value ('WxH', multiples of 16) as ``(width, height)``;
    ``ValueError`` (with what is wrong) for anything else."""
    parts = str(text).lower().replace("*", "x").split("x")
    try:
        width, height = (int(p) for p in parts) if len(parts) == 2 else (0, 0)
    except ValueError:
        width = height = 0
    low, high = FX_REPLACE_CANVAS_SIDE
    if not (low <= width <= high and low <= height <= high) or width % 16 or height % 16:
        raise ValueError(f"canvas must be 'WxH' with each side a multiple of 16 from {low} to {high}.")
    return width, height


def fx_replace_length(frames):
    """Frames Wan generates for a ``frames`` clip: the next 4n+1 (its VAE packs
    four frames per latent after the first); the extra ones are trimmed off."""
    return (frames + 2) // 4 * 4 + 1


def fx_replace_check_budget(width, height, length):
    """An error detail when the generation would not fit this host's budget, else None."""
    used = width * height * length
    if used <= FX_REPLACE_MAX_PIXEL_FRAMES:
        return None
    return (f"{length} frames at {width}x{height} is {used / 1e6:.0f} megapixel-frames, above this host's "
            f"{FX_REPLACE_MAX_PIXEL_FRAMES / 1e6:.0f} budget. Send at most about "
            f"{FX_REPLACE_MAX_PIXEL_FRAMES // (width * height)} frames at this tier, a shorter range, or tier "
            "'standard'.")


def build_fx_replace(*, video, mask, reference, width, height, frames, keyframe=None, prompt="", seed=0,
                     steps=FX_REPLACE_DEFAULT_STEPS, unet_name=FX_REPLACE_UNET, clip_name=FX_REPLACE_CLIP,
                     vae_name=FX_REPLACE_VAE, lora_name=FX_REPLACE_LORA,
                     lora_strength=FX_REPLACE_LORA_STRENGTH, interpolate=1, fps=None,
                     interp_model_name=FX_INTERP_MODELS[FX_INTERP_DEFAULT_MODEL],
                     filename_prefix="wrapper/fx_replace"):
    """Generative replace inside a mask: video + its grey mask video (white =
    replace) + a reference photo (+ an approved edited frame 0) -> MP4 at the
    width x height canvas with the source's ``frames`` and frame rate, no
    audio. With ``interpolate`` 2 the answer has ``(frames - 1) * 2 + 1``
    frames at twice the rate (the same duration): FILM fills in between the
    generated ones, so a clip sent at half the frame rate comes back at it
    (``fps`` is then the sent clip's probed rate, which the answer doubles).

    The clip and its mask are scaled to the canvas, the masked pixels painted
    grey, and WanVaceToVideo regenerates them from the reference (padded onto
    white, so all of it stays in view) and the prompt. A ``keyframe`` replaces
    frame 0 of the control video, unmasked, so the result starts from it.
    Generation runs for fx_replace_length(frames) frames, in context windows
    past FX_REPLACE_WINDOW, and the padding is trimmed off the decoded clip."""
    if frames < 2:
        raise ValueError("fx-replace needs at least two frames")
    if interpolate not in FX_REPLACE_INTERPOLATE:
        raise ValueError(f"interpolate must be one of {FX_REPLACE_INTERPOLATE}")
    if interpolate > 1 and not fps:
        raise ValueError("interpolating needs the clip's frame rate (fps)")
    length = fx_replace_length(frames)
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": FX_REPLACE_WEIGHT_DTYPE}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "lora_name": lora_name, "strength_model": lora_strength}},
        "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": FX_REPLACE_SHIFT}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "wan", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": prompt}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": ""}},
        "10": {"class_type": "LoadVideo", "inputs": {"file": video}},
        # outputs: images, audio, fps, bit_depth, color_space
        "11": {"class_type": "GetVideoComponents", "inputs": {"video": ["10", 0]}},
        "12": {"class_type": "ImageScale", "inputs": {
            "image": ["11", 0], "upscale_method": "area", "width": width, "height": height, "crop": "center"}},
        "13": {"class_type": "LoadVideo", "inputs": {"file": mask}},
        "14": {"class_type": "GetVideoComponents", "inputs": {"video": ["13", 0]}},
        "15": {"class_type": "ImageScale", "inputs": {
            "image": ["14", 0], "upscale_method": "bilinear", "width": width, "height": height, "crop": "center"}},
        "16": {"class_type": "ImageToMask", "inputs": {"image": ["15", 0], "channel": "red"}},
        "17": {"class_type": "EmptyImage", "inputs": {"width": width, "height": height, "batch_size": 1,
                                                      "color": FX_REPLACE_GREY}},
        "18": {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": ["12", 0], "source": ["17", 0], "x": 0, "y": 0, "resize_source": False,
            "mask": ["16", 0]}},
        "30": {"class_type": "LoadImage", "inputs": {"image": reference}},
        "31": {"class_type": "ResizeAndPadImage", "inputs": {
            "image": ["30", 0], "target_width": width, "target_height": height, "padding_color": "white",
            "interpolation": "area"}},
    }
    model, control, control_masks = ["3", 0], ["18", 0], ["16", 0]
    if length > FX_REPLACE_WINDOW:
        graph["4"] = {"class_type": "WanContextWindowsManual", "inputs": {
            "model": ["3", 0], "context_length": FX_REPLACE_WINDOW, "context_overlap": FX_REPLACE_WINDOW_OVERLAP,
            "context_schedule": "standard_static", "context_stride": 1, "closed_loop": False,
            "fuse_method": "pyramid", "freenoise": True, "retain_first_frame": False,
            "split_conds_to_windows": False}}
        model = ["4", 0]
    if keyframe:
        # Frame 0 becomes the keyframe, kept as is (a black mask frame).
        graph.update({
            "20": {"class_type": "LoadImage", "inputs": {"image": keyframe}},
            "21": {"class_type": "ImageScale", "inputs": {
                "image": ["20", 0], "upscale_method": "area", "width": width, "height": height, "crop": "center"}},
            "22": {"class_type": "ImageFromBatch", "inputs": {"image": ["18", 0], "batch_index": 1, "length": frames - 1}},
            "23": {"class_type": "BatchImagesNode", "inputs": {"images.image0": ["21", 0], "images.image1": ["22", 0]}},
            "24": {"class_type": "EmptyImage", "inputs": {"width": width, "height": height, "batch_size": 1, "color": 0}},
            "25": {"class_type": "ImageFromBatch", "inputs": {"image": ["15", 0], "batch_index": 1, "length": frames - 1}},
            "26": {"class_type": "BatchImagesNode", "inputs": {"images.image0": ["24", 0], "images.image1": ["25", 0]}},
            "27": {"class_type": "ImageToMask", "inputs": {"image": ["26", 0], "channel": "red"}},
        })
        control, control_masks = ["23", 0], ["27", 0]
    graph.update({
        "32": {"class_type": "WanVaceToVideo", "inputs": {
            "positive": ["7", 0], "negative": ["8", 0], "vae": ["6", 0], "width": width, "height": height,
            "length": length, "batch_size": 1, "strength": 1.0, "control_video": control,
            "control_masks": control_masks, "reference_image": ["31", 0]}},
        "33": {"class_type": "KSampler", "inputs": {
            "model": model, "seed": seed, "steps": steps, "cfg": FX_REPLACE_CFG, "sampler_name": "uni_pc",
            "scheduler": "simple", "positive": ["32", 0], "negative": ["32", 1], "latent_image": ["32", 2],
            "denoise": 1.0}},
        # Drop the reference's latent frame, then the 4n+1 padding.
        "34": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["33", 0], "trim_amount": ["32", 3]}},
        "35": {"class_type": "VAEDecode", "inputs": {"samples": ["34", 0], "vae": ["6", 0]}},
        "36": {"class_type": "ImageFromBatch", "inputs": {"image": ["35", 0], "batch_index": 0, "length": frames}},
        "37": {"class_type": "CreateVideo", "inputs": {"images": ["36", 0], "fps": ["11", 2], "bit_depth": 8}},
        "38": {"class_type": "SaveVideo", "inputs": {
            "video": ["37", 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"}},
    })
    if interpolate > 1:
        graph.update({
            "39": {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": interp_model_name}},
            "40": {"class_type": "FrameInterpolate", "inputs": {
                "interp_model": ["39", 0], "images": ["36", 0], "multiplier": interpolate},
                   "_meta": {"expected_steps": (frames - 1) * (interpolate - 1)}},
        })
        graph["37"]["inputs"].update({"images": ["40", 0], "fps": float(fps) * interpolate})
    return graph


FX_REPLACE_FORM_EXTRA = {
    "video": {"type": "string", "format": "binary",
              "description": "The range to regenerate (mp4, webm, mov, mkv, avi or gif), at least two frames."},
    "mask": {"type": "string", "format": "binary",
             "description": "Grey mask video with the same frames (and aspect) as 'video', white = replace; "
                            "used as is, so dilate it first."},
    "reference": {"type": "string", "format": "binary",
                  "description": "Photo of the new object (png, jpg, webp, bmp or gif)."},
    "keyframe": {"type": "string", "format": "binary",
                 "description": "Optional approved edit of frame 0 (e.g. a still preview); the video starts from it."},
    "prompt": {"type": "string", "description": "Optional words about the new object, e.g. 'a red soda can'."},
    "tier": {"type": "string", "enum": list(FX_REPLACE_TIERS), "default": FX_REPLACE_DEFAULT_TIER,
             "description": "The model setup and, without 'canvas', the generation size at the clip's aspect "
                            "(sides multiples of 16): 'quick' (about 640x384; lightx2v's 4-step LoRA), "
                            "'standard' (about 832x480; CausVid, 6 steps) or 'hd' (about 1280x720; the same "
                            "setup as standard). The output stays at the canvas size."},
    "canvas": {"type": "string", "example": "336x512",
               "description": "The exact generation size, 'WxH' with each side a multiple of 16 (64 to 2048); "
                              "overrides the tier's. A caller that sends only the crop around the object sends "
                              "it already at this size."},
    "steps": {"type": "integer", "minimum": 1, "maximum": 50,
              "description": "Sampling steps; the speed LoRA is tuned for the tier's default (quick 4, "
                             f"standard and hd {FX_REPLACE_DEFAULT_STEPS})."},
    "interpolate": {"type": "integer", "enum": list(FX_REPLACE_INTERPOLATE), "default": 1,
                    "description": "2: FILM makes a frame between each pair generated, so the answer has "
                                   "(frames - 1) * 2 + 1 frames at twice the video's frame rate (the same "
                                   "duration). For a video sent at half the rate it is to play at."},
}


# Image Studio's "look": the Qwen3-VL 8B text encoder Qwen Image 2.1 loads
# (its int8_convrot file carries the language-model head), so describing a
# picture needs no download of its own.
IMAGE_DESCRIBE_CLIP = QWEN_IMAGE_21_CLIP
IMAGE_DESCRIBE_MODELS = [m for m in qwen_image_21_models("int8") if m["folder"] == "text_encoders"]


def build_image_describe(*, image, task=image_describe.DESCRIBE_DEFAULT_TASK,
                         max_length=image_describe.DESCRIBE_DEFAULT_MAX_LENGTH,
                         megapixels=image_describe.DESCRIBE_MEGAPIXELS, clip_name=IMAGE_DESCRIBE_CLIP,
                         filename_prefix=None):
    """Image description: still -> Qwen3-VL's JSON answer to a fixed prompt
    (api_wrapper/describe.py), as the job's JSON answer. Nothing is saved to
    disk, so ``filename_prefix`` is unused.

    The CLIP loads as a plain Qwen3-VL ('stable_diffusion' type: no image
    model's template around the prompt) and decodes greedily, so one picture
    always gets the same answer. ``megapixels`` is the still's budget
    (describe_megapixels: scaled down, never up)."""
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "stable_diffusion", "device": "default"}},
        **_first_frame(image),
        "4": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["3", 0], "upscale_method": "area", "megapixels": megapixels, "resolution_steps": 1}},
        "5": {"class_type": "TextGenerate", "inputs": {
            "clip": ["1", 0], "prompt": image_describe.describe_prompt(task), "image": ["4", 0],
            "max_length": max_length, "sampling_mode": "off"}},
        "6": tag_result({"class_type": "PreviewAny", "inputs": {"source": ["5", 0]}}, "image-describe", task=task),
    }


IMAGE_DESCRIBE_FORM_EXTRA = {
    "image": {"type": "string", "format": "binary",
              "description": "The picture (png, jpg, webp, bmp or gif; a GIF's first frame)."},
    "task": {"type": "string", "enum": list(image_describe.DESCRIBE_TASKS),
             "default": image_describe.DESCRIBE_DEFAULT_TASK,
             "description": "'caption' (one or two sentences), 'objects' (labels with boxes) or 'both'."},
    "max_length": {"type": "integer", "minimum": 1, "maximum": image_describe.DESCRIBE_MAX_LENGTH,
                   "default": image_describe.DESCRIBE_DEFAULT_MAX_LENGTH,
                   "description": "Most tokens the model writes. An answer cut short keeps the objects "
                                  "completed before the cut."},
}


"""The wrapper API's workflow registry.

Each entry describes one dedicated workflow API: ``build`` constructs the
API-format prompt graph, ``requires_image`` tells the generic generate
handler whether an uploaded image is mandatory, and ``uses`` lists which of
the shared form parameters the builder consumes (the rest are ignored).
Entries may declare ``tasks`` (e.g. MiniMax H3's text/image/reference
variants) to get one endpoint per task under /api/wrapper/{name}/{task}.
Route-level setup (model downloads, quantization) lives in ``api_wrapper.
routes`` keyed by the same names. Adding a new workflow = adding one entry
here plus one setup handler.

A workflow that takes no prompt sets ``requires_prompt: False``;
``required_uploads`` names upload fields that must be present (any kind, not
just images); ``setup_uses_uploads`` hands the saved upload refs to the setup
handler (e.g. to probe an uploaded video before queueing).

A workflow that answers with data declares ``text_result``, a pure parser
``(text, tag, images) -> dict`` of the node its graph tags with
api_wrapper.results.tag_result; GET /wrapper/jobs/{job_id}/text serves it.
With ``output_type: "json"`` that answer is also the generate response body,
for workflows that save no file.
"""

# --- Lightricks LTX-2.5 (22B distilled audio+video) -------------------------------
# The official two-stage pipeline (blueprints "Text to Video (LTX-2.5)" and
# "Image to Video (LTX-2.5)"): an 8-step pass at half the canvas, a 2x latent
# upscale, a 3-step refine at the full canvas. The distilled transformer runs at
# CFG 1, so there is no steps/cfg/scheduler to tune. Weights ~44 GB:
# Lightricks/LTX-2.5 (LTX-2 Community License: free under USD 10M annual revenue).
LTX25_BASE_URL = "https://huggingface.co/Lightricks/LTX-2.5/resolve/main"
LTX25_UNET = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
LTX25_CLIP = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
LTX25_VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
LTX25_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
LTX25_UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
LTX25_MODELS = [
    {"folder": "diffusion_models", "filename": LTX25_UNET, "url": f"{LTX25_BASE_URL}/diffusion_models/{LTX25_UNET}"},
    {"folder": "text_encoders", "filename": LTX25_CLIP, "url": f"{LTX25_BASE_URL}/text_encoders/{LTX25_CLIP}"},
    {"folder": "vae", "filename": LTX25_VIDEO_VAE, "url": f"{LTX25_BASE_URL}/vae/{LTX25_VIDEO_VAE}"},
    {"folder": "vae", "filename": LTX25_AUDIO_VAE, "url": f"{LTX25_BASE_URL}/vae/{LTX25_AUDIO_VAE}"},
    {"folder": "latent_upscale_models", "filename": LTX25_UPSCALER,
     "url": f"{LTX25_BASE_URL}/latent_upscale_models/{LTX25_UPSCALER}"},
]
LTX25_FPS = 24
# The two stages work on a half-size canvas that must sit on the 32 px latent
# grid, so the output canvas is a multiple of 64.
LTX25_CANVAS_MULTIPLE = 64
LTX25_DEFAULT_WIDTH = 1280
LTX25_DEFAULT_HEIGHT = 704
LTX25_MAX_PIXELS = 1920 * 1088
LTX25_MAX_FRAMES = max(9, int(os.environ.get("COMFY_LTX25_MAX_FRAMES", "241")))
LTX25_MAX_DURATION = round((LTX25_MAX_FRAMES - 1) / LTX25_FPS, 2)
LTX25_NEGATIVE_PROMPT = "pc game, console game, video game, cartoon, childish, ugly"
LTX25_STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX25_STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"


def ltx25_length(duration):
    """Duration (seconds) -> frame count at 24 fps on LTX's 8k+1 grid."""
    n = max(9, round(duration * LTX25_FPS))
    return ((n - 1 + 7) // 8) * 8 + 1


def ltx25_fit_canvas(width, height):
    """Round a requested canvas to LTX's 64 px grid (nearest, never below 64),
    scaling it down (aspect kept) when it exceeds 1920x1088 pixels. Returns
    (width, height, note)."""
    m = LTX25_CANVAS_MULTIPLE
    snap = lambda v: max(m, int(round(v / m)) * m)
    fitted_w, fitted_h = snap(width), snap(height)
    note = None
    if fitted_w * fitted_h > LTX25_MAX_PIXELS:
        scale = (LTX25_MAX_PIXELS / (fitted_w * fitted_h)) ** 0.5
        fitted_w, fitted_h = max(m, int(fitted_w * scale) // m * m), max(m, int(fitted_h * scale) // m * m)
        note = (f"{width}x{height} exceeds this model's {LTX25_MAX_PIXELS}-pixel canvas; "
                f"generating at {fitted_w}x{fitted_h} instead.")
    elif (fitted_w, fitted_h) != (width, height):
        note = f"{width}x{height} generated at {fitted_w}x{fitted_h} (this model works on a 64 px grid)."
    return fitted_w, fitted_h, note


def _ltx25_graph(prompt, width, height, length, seed, filename_prefix, *, first_frame=None,
                 unet_name=LTX25_UNET, clip_name=LTX25_CLIP):
    """Shared two-stage graph. ``first_frame`` (an uploaded image name) switches
    the low-res and full-res latents to image-conditioned ones."""
    half_w, half_h = width // 2, height // 2
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": LTX25_VIDEO_VAE}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": LTX25_AUDIO_VAE}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "ltxv", "device": "default"}},
        "5": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": LTX25_UPSCALER}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": LTX25_NEGATIVE_PROMPT, "clip": ["4", 0]}},
        "8": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "frame_rate": float(LTX25_FPS)}},
        "9": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": half_w, "height": half_h, "length": length, "batch_size": 1}},
        "10": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": length, "frame_rate": LTX25_FPS, "batch_size": 1, "audio_vae": ["3", 0]}},
    }
    video_latent = ["9", 0]
    image = None
    if first_frame is not None:
        # Cover-crop the still to the canvas, light compression like the
        # official template, then pin it as the first frame of both stages.
        g["11"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        g["12"] = {"class_type": "ImageScale", "inputs": {
            "image": ["11", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
        g["13"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["12", 0], "img_compression": 18}}
        g["14"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["2", 0], "image": ["13", 0], "latent": ["9", 0], "strength": 0.7, "bypass": False}}
        video_latent = ["14", 0]
        image = ["13", 0]
    g.update({
        "20": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": video_latent, "audio_latent": ["10", 0]}},
        "21": {"class_type": "LTXVDualCFGGuider", "inputs": {
            "model": ["1", 0], "positive": ["8", 0], "negative": ["8", 1], "video_cfg": 1.0, "audio_cfg": 1.0}},
        "22": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "23": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "24": {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX25_STAGE1_SIGMAS}},
        "25": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["22", 0], "guider": ["21", 0], "sampler": ["23", 0], "sigmas": ["24", 0],
            "latent_image": ["20", 0]}},
        "26": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["25", 0]}},
        "27": {"class_type": "LTXVLatentUpsampler", "inputs": {
            "samples": ["26", 0], "upscale_model": ["5", 0], "vae": ["2", 0]}},
    })
    upscaled = ["27", 0]
    if image is not None:
        g["28"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["2", 0], "image": image, "latent": ["27", 0], "strength": 1.0, "bypass": False}}
        upscaled = ["28", 0]
    g.update({
        "30": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": upscaled, "audio_latent": ["26", 1]}},
        "31": {"class_type": "LTXVDualCFGGuider", "inputs": {
            "model": ["1", 0], "positive": ["8", 0], "negative": ["8", 1], "video_cfg": 1.0, "audio_cfg": 1.0}},
        "32": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "33": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "34": {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX25_STAGE2_SIGMAS}},
        "35": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["32", 0], "guider": ["31", 0], "sampler": ["33", 0], "sigmas": ["34", 0],
            "latent_image": ["30", 0]}},
        "36": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["35", 0]}},
        "37": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["36", 0], "vae": ["2", 0], "tile_size": 512, "overlap": 64,
            "temporal_size": 64, "temporal_overlap": 16}},
        "38": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["36", 1], "audio_vae": ["3", 0]}},
        "39": {"class_type": "CreateVideo", "inputs": {
            "images": ["37", 0], "fps": float(LTX25_FPS), "audio": ["38", 0], "bit_depth": 8}},
        "40": {"class_type": "SaveVideo", "inputs": {
            "video": ["39", 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"}},
    })
    return g


def build_ltx25_text_to_video(*, prompt, seed=0, width=LTX25_DEFAULT_WIDTH, height=LTX25_DEFAULT_HEIGHT,
                              duration=5.0, filename_prefix="wrapper/ltx25_t2v",
                              unet_name=LTX25_UNET, clip_name=LTX25_CLIP):
    """LTX-2.5 text-to-video: prompt -> joint audio+video MP4."""
    return _ltx25_graph(prompt, width, height, ltx25_length(duration), seed, filename_prefix,
                        unet_name=unet_name, clip_name=clip_name)


def build_ltx25_image_to_video(*, prompt, seed=0, width=LTX25_DEFAULT_WIDTH, height=LTX25_DEFAULT_HEIGHT,
                               duration=5.0, filename_prefix="wrapper/ltx25_i2v", first_frame=None,
                               unet_name=LTX25_UNET, clip_name=LTX25_CLIP):
    """LTX-2.5 image-to-video: a first frame + prompt -> joint audio+video MP4."""
    return _ltx25_graph(prompt, width, height, ltx25_length(duration), seed, filename_prefix,
                        first_frame=first_frame, unet_name=unet_name, clip_name=clip_name)


LTX25_FORM_EXTRA = {
    "width": {"type": "integer", "minimum": 64, "maximum": 4096, "default": LTX25_DEFAULT_WIDTH,
              "description": "Output width, rounded to a multiple of 64 (the two stages work on a half-size "
                             f"canvas on a 32 px latent grid). Capped at {LTX25_MAX_PIXELS} pixels; a larger "
                             "request is scaled down keeping its aspect ratio."},
    "height": {"type": "integer", "minimum": 64, "maximum": 4096, "default": LTX25_DEFAULT_HEIGHT,
               "description": "Output height, rounded to a multiple of 64. 720 becomes 704."},
    "duration": {"type": "number", "minimum": 0.4, "maximum": LTX25_MAX_DURATION, "default": 5.0,
                 "description": "Video length in seconds at 24 fps, snapped up to the model's 8k+1 frame grid. "
                                f"The maximum ({LTX25_MAX_FRAMES} frames) is set by COMFY_LTX25_MAX_FRAMES."},
}


WORKFLOWS = {
    "flux2klein9b": {
        "title": "FLUX.2 [klein] 9B image edit",
        "requires_image": True,
        # Either field satisfies requires_image: `image` (one) and/or up to
        # three `ref_images`, each its own reference latent.
        "image_fields": ["image", "ref_images"],
        "uploads": {"image": {"ext": "image", "max": 1}, "ref_images": {"ext": "image", "max": 3}},
        "uses": ["prompt", "negative_prompt", "seed", "steps", "cfg", "megapixels"],
        "form": ["prompt", "image", "ref_images", "negative_prompt", "seed", "steps", "cfg",
                 "megapixels", "width", "height"],
        "extra_form_properties": {
            "width": {"type": "integer", "minimum": 256, "maximum": 8192,
                      "description": "Output width. Omit to take the first image's size."},
            "height": {"type": "integer", "minimum": 256, "maximum": 8192,
                       "description": "Output height. Omit to take the first image's size."},
        },
        "build": build_flux2_klein_9b_img2img,
    },
    "flux2klein9b-txt2img": {
        "title": "FLUX.2 [klein] 9B text to image",
        "requires_image": False,
        "uses": ["prompt", "negative_prompt", "seed", "steps", "cfg"],
        "form": ["prompt", "negative_prompt", "seed", "steps", "cfg", "width", "height"],
        "extra_form_properties": {
            "width": {"type": "integer", "minimum": 256, "maximum": 8192, "default": 1024,
                      "description": "Output width."},
            "height": {"type": "integer", "minimum": 256, "maximum": 8192, "default": 1024,
                       "description": "Output height."},
        },
        "build": build_flux2_klein_9b_text2img,
    },
    "qwenimage21": {
        "title": "Qwen Image 2.1 image edit",
        "requires_image": True,
        # Image 1 is `image` (else the first of `ref_images`); the node takes
        # 16 images in all.
        "image_fields": ["image", "ref_images"],
        "uploads": {"image": {"ext": "image", "max": 1}, "ref_images": {"ext": "image", "max": 15}},
        "uses": ["prompt", "negative_prompt", "seed"],
        "form": ["prompt", "image", "ref_images", "negative_prompt", "seed", "steps", "cfg",
                 "resolution", "width", "height"],
        "extra_form_properties": {
            **QWEN_IMAGE_21_FORM_EXTRA,
            "prompt": {"type": "string",
                       "description": "The edit instruction. The images are <image1>, <image2>, ... in upload "
                                      "order (image, then ref_images) and <image1> is the one edited, e.g. "
                                      "'Put the denim shirt from <image2> on the person in <image1>, keep "
                                      "the pose, face and background.'"},
            "resolution": {"type": "integer", "minimum": 0, "maximum": 4096, "default": 1024,
                           "description": "Each image is resized to about resolution x resolution pixels "
                                          "(aspect kept, multiples of 32) before encoding; 0 keeps its own "
                                          "size. The output takes image 1's resized size."},
            "width": {"type": "integer", "minimum": 256, "maximum": 8192,
                      "description": "Output width, sent with height. Omit both to take image 1's size; a "
                                     "canvas far from it shifts the edit."},
            "height": {"type": "integer", "minimum": 256, "maximum": 8192,
                       "description": "Output height, sent with width."},
        },
        "build": build_qwen_image_21_edit,
    },
    "qwenimage21-txt2img": {
        "title": "Qwen Image 2.1 text to image",
        "requires_image": False,
        "uses": ["prompt", "negative_prompt", "seed"],
        "form": ["prompt", "negative_prompt", "seed", "steps", "cfg", "width", "height"],
        "extra_form_properties": {
            **QWEN_IMAGE_21_FORM_EXTRA,
            "width": {"type": "integer", "minimum": 256, "maximum": 8192, "default": 1024,
                      "description": "Output width. Qwen Image 2.1 is native up to 2048x2048; prefer multiples of 32."},
            "height": {"type": "integer", "minimum": 256, "maximum": 8192, "default": 1024,
                       "description": "Output height."},
        },
        "build": build_qwen_image_21_text2img,
    },
    "ideogram4": {
        "title": "Ideogram 4 text to image",
        "requires_image": False,
        "uses": ["prompt", "seed"],
        "form": ["prompt", "seed", "steps", "mode", "width", "height"],
        "example_prompt": IDEOGRAM4_EXAMPLE_PROMPT,
        "extra_form_properties": {
            "mode": {"type": "string", "enum": ["default", "quality", "turbo"], "default": "default",
                     "description": "Scheduler preset: default (20 steps), quality (48 steps, lower std), turbo (12 steps, mu 0.5)."},
            "width": {"type": "integer", "minimum": 256, "maximum": 8192, "default": 1024,
                      "description": "Output width; rounded up to a multiple of 16."},
            "height": {"type": "integer", "minimum": 256, "maximum": 8192, "default": 1024,
                       "description": "Output height; rounded up to a multiple of 16."},
        },
        "build": build_ideogram4_text2img,
    },
    "minimaxh3": {
        "title": "MiniMax H3 omni-modal video",
        "output_type": "video",
        "tasks": {
            "text": {
                "title": "Text to video",
                "requires_image": False,
                "uploads": {"llm_image": {"ext": "image", "max": 1}},
                # llm_image feeds the prompt rewriter only; None keeps the
                # generic handler from passing it to the graph builder.
                "upload_params": {"llm_image": None},
                "prompt_rewrite": True,
                "uses": ["prompt", "seed"],
                "form": ["prompt", "raw_prompt", "llm_image", "llm_model", "seed", "steps", "width", "height", "duration", "scheduler"],
                "extra_form_properties": MINIMAX_H3_FORM_EXTRA,
                "quantization_options": ["fp8", "int8", "bf16", "nvfp4"],
                "build": build_minimax_h3_text_to_video,
            },
            "image": {
                "title": "Image to video",
                "requires_image": True,
                "uploads": {"image": {"ext": "image", "max": 1},
                             "last_frame": {"ext": "image", "max": 1},
                             "llm_image": {"ext": "image", "max": 1}},
                "upload_params": {"image": "first_frame", "llm_image": None},
                "prompt_rewrite": True,
                "uses": ["prompt", "seed"],
                "form": ["prompt", "raw_prompt", "image", "last_frame", "llm_image", "llm_model", "seed", "steps", "width", "height", "duration", "scheduler"],
                "extra_form_properties": MINIMAX_H3_FORM_EXTRA,
                "quantization_options": ["fp8", "int8", "bf16", "nvfp4"],
                "build": build_minimax_h3_image_to_video,
            },
            "reference": {
                "title": "Reference to video (ref2va)",
                "requires_image": False,
                # The node's own limit: nine reference images (each a <Picture i>).
                "uploads": {"ref_images": {"ext": "image", "max": 9},
                             "ref_videos": {"ext": "video", "max": 1},
                             "ref_video_audios": {"ext": "audio", "max": 1},
                             # The node's own limit: three voice-timbre clips (a
                             # cast of speakers, or two and the scene's sound).
                             "ref_audios": {"ext": "audio", "max": 3},
                             # Pinned first / last frame (keyframe completion).
                             "image": {"ext": "image", "max": 1},
                             "last_frame": {"ext": "image", "max": 1},
                             "llm_image": {"ext": "image", "max": 1}},
                "upload_params": {"image": "first_frame", "llm_image": None},
                "prompt_rewrite": True,
                "uses": ["prompt", "seed"],
                "form": ["prompt", "raw_prompt", "image", "last_frame", "llm_image", "llm_model", "seed", "steps", "width", "height", "duration", "scheduler", "ref_image_size"],
                "extra_form_properties": MINIMAX_H3_REF_FORM_EXTRA,
                "quantization_options": ["fp8", "int8", "bf16", "nvfp4"],
                "example_prompt": (
                    "Show <Picture 1> skateboarding down a sunlit street, the camera follows "
                    "from the side, <Audio 1> with the sound of wheels rolling on asphalt, "
                    "cinematic 24fps handheld shot."),
                "build": build_minimax_h3_reference_to_video,
            },
        },
    },
    "ltx25": {
        "title": "LTX-2.5 video (audio + video, two-stage distilled)",
        "output_type": "video",
        "tasks": {
            "text": {
                "title": "Text to video",
                "requires_image": False,
                "uses": ["prompt", "seed"],
                "form": ["prompt", "seed", "width", "height", "duration"],
                "extra_form_properties": LTX25_FORM_EXTRA,
                "build": build_ltx25_text_to_video,
            },
            "image": {
                "title": "Image to video",
                "requires_image": True,
                "uploads": {"image": {"ext": "image", "max": 1}},
                "upload_params": {"image": "first_frame"},
                "uses": ["prompt", "seed"],
                "form": ["prompt", "image", "seed", "width", "height", "duration"],
                "extra_form_properties": LTX25_FORM_EXTRA,
                "build": build_ltx25_image_to_video,
            },
        },
    },
    "fx-interp": {
        "title": "Frame interpolation (slow motion, FILM/RIFE)",
        "output_type": "video",
        "requires_image": False,
        "requires_prompt": False,
        "required_uploads": ["video"],
        "setup_uses_uploads": True,
        "uploads": {"video": {"ext": "video", "max": 1}},
        "uses": [],
        "form": ["video", "factor", "model"],
        "extra_form_properties": FX_INTERP_FORM_EXTRA,
        "build": build_fx_interp,
    },
    "fx-bgremove": {
        "title": "Background removal matte (BiRefNet)",
        "output_type": "video",
        "requires_image": False,
        "requires_prompt": False,
        "required_uploads": ["video"],
        "setup_uses_uploads": True,
        "uploads": {"video": {"ext": "video", "max": 1}},
        "uses": [],
        "form": ["video", "model"],
        "extra_form_properties": FX_BGREMOVE_FORM_EXTRA,
        "build": build_fx_bgremove,
    },
    "fx-upscale": {
        "title": "Video or image upscale (Real-ESRGAN / SeedVR2, tiled FLUX.2 klein refine)",
        "output_type": "video",
        "requires_image": False,
        "requires_prompt": False,
        # 'video' (a clip -> MP4) or 'image' (a still -> PNG); the setup refuses neither.
        "required_uploads": [],
        "setup_uses_uploads": True,
        "uploads": {"video": {"ext": "video", "max": 1}, "image": {"ext": "image", "max": 1}},
        # The setup adds a seed for seedvr2 only (esrgan doesn't sample); the
        # image mode's refine prompt comes through its own setup too.
        "uses": [],
        "form": ["video", "image", "model", "target_short_side", "width", "height", "refine", "denoise",
                 "prompt", "seed"],
        "extra_form_properties": FX_UPSCALE_FORM_EXTRA,
        "build": build_fx_upscale_any,
    },
    "fx-track": {
        "title": "Object tracking matte and boxes (SAM 3.1)",
        "output_type": "video",
        "requires_image": False,
        # The phrase is one of four ways to say what to track (setup checks).
        "requires_prompt": False,
        "required_uploads": ["video"],
        "setup_uses_uploads": True,
        "uploads": {"video": {"ext": "video", "max": 1}, "mask": {"ext": "image", "max": 1}},
        "uses": [],
        "form": ["video", "mask", "prompt", "boxes", "positive_coords", "negative_coords", "threshold",
                 "detect_interval"],
        "extra_form_properties": FX_TRACK_FORM_EXTRA,
        "text_result": object_track.track_result,
        "build": build_fx_track,
    },
    "fx-replace": {
        "title": "Generative object replace inside a mask (Wan 2.1 VACE 14B)",
        "output_type": "video",
        "requires_image": False,
        "requires_prompt": False,
        "required_uploads": ["video", "mask", "reference"],
        "setup_uses_uploads": True,
        "uploads": {"video": {"ext": "video", "max": 1}, "mask": {"ext": "video", "max": 1},
                    "reference": {"ext": "image", "max": 1}, "keyframe": {"ext": "image", "max": 1}},
        # steps comes through the setup (the speed LoRA's default, not 20).
        "uses": ["prompt", "seed"],
        "form": ["video", "mask", "reference", "keyframe", "prompt", "tier", "seed", "steps"],
        "extra_form_properties": FX_REPLACE_FORM_EXTRA,
        "build": build_fx_replace,
    },
    "image-segment": {
        "title": "Image segmentation by phrase, box or point (SAM 3)",
        "requires_image": True,
        # The phrase is one of three ways to say what to select (setup checks).
        "requires_prompt": False,
        "setup_uses_uploads": True,
        "uploads": {"image": {"ext": "image", "max": 1}},
        "uses": [],
        "form": ["image", "prompt", "boxes", "positive_coords", "negative_coords", "threshold",
                 "max_objects", "individual_masks"],
        "extra_form_properties": IMAGE_SEGMENT_FORM_EXTRA,
        "text_result": image_segment.segment_result,
        "build": build_image_segment,
    },
    "image-describe": {
        "title": "Image caption and objects (Qwen3-VL)",
        "output_type": "json",
        "result_schema": "ImageDescribeResult",
        "requires_image": True,
        "requires_prompt": False,
        "setup_uses_uploads": True,
        "uploads": {"image": {"ext": "image", "max": 1}},
        "uses": [],
        "form": ["image", "task", "max_length"],
        "extra_form_properties": IMAGE_DESCRIBE_FORM_EXTRA,
        "text_result": image_describe.describe_result,
        "build": build_image_describe,
    },
}
