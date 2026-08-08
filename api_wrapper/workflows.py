"""Workflow builders for the wrapper API.

This module is deliberately free of comfy imports so it stays unit-testable
and the graph structure is easy to audit. It builds ComfyUI API-format prompt
graphs (as accepted by POST /prompt) and declares the models a workflow needs.
"""

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


def build_flux2_klein_9b_img2img(*, prompt, image, negative_prompt="", seed=0,
                                 steps=20, cfg=5.0, megapixels=1.0,
                                 filename_prefix="wrapper/flux2_klein_9b",
                                 unet_name=FLUX2_KLEIN_9B_UNET):
    """Build the FLUX.2 [klein] 9B image edit graph (API format).

    Mirrors the shipped "Image Edit (Flux.2 Klein)" blueprint: the input image
    is scaled to a fixed megapixel budget, its latent is attached to both the
    positive and negative conditioning as a reference latent, and sampling
    runs through the flux2 custom sampler stack (CFG guider, euler,
    flux2 scheduler, empty flux2 latent at the image size).

    ``image`` is a file name relative to the ComfyUI input directory.
    ``unet_name`` selects the diffusion model file (fp8 by default; pass the
    converted nvfp4 file name for the fp4 path).
    """
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX2_KLEIN_9B_CLIP, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_KLEIN_9B_VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image}},
        "5": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["4", 0], "upscale_method": "nearest-exact", "megapixels": megapixels, "resolution_steps": 1}},
        "6": {"class_type": "GetImageSize", "inputs": {"image": ["5", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative_prompt}},
        "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "10": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["9", 0]}},
        "11": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["8", 0], "latent": ["9", 0]}},
        "12": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["10", 0], "negative": ["11", 0], "cfg": cfg}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": ["6", 0], "height": ["6", 1]}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "16": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["6", 0], "height": ["6", 1], "batch_size": 1}},
        "17": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["12", 0], "sampler": ["13", 0], "sigmas": ["14", 0], "latent_image": ["16", 0]}},
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["3", 0]}},
        "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": filename_prefix}},
    }


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


# MiniMax H3 omni-modal video model files (Comfy-Org/MiniMax-H3). The fl2va
# family powers text-to-video and image-to-video; ref2va is the
# reference-to-video variant with its own weights. int8_convrot is the
# canonical template choice; fp8_scaled loads everywhere (bf16 fallback on
# MPS); bf16 is the full-quality option.
MINIMAX_H3_FPS = 24
MINIMAX_H3_UNET_FP8 = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
MINIMAX_H3_UNET_INT8 = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
MINIMAX_H3_UNET_BF16 = "minimax_h3_fl2va_pruned_bf16.safetensors"
MINIMAX_H3_REF2VA_UNET_FP8 = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
MINIMAX_H3_REF2VA_UNET_INT8 = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
MINIMAX_H3_REF2VA_UNET_BF16 = "minimax_h3_ref2va_pruned_bf16.safetensors"
MINIMAX_H3_CLIP = "qwen3vl_32b_minimax_h3_bf16.safetensors"
MINIMAX_H3_CLIP_INT8 = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
MINIMAX_H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
MINIMAX_H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
MINIMAX_H3_BASE_URL = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
MINIMAX_H3_QUANT_MODELS = {
    "fp8": {"unet": MINIMAX_H3_UNET_FP8, "ref2va": MINIMAX_H3_REF2VA_UNET_FP8, "clip": MINIMAX_H3_CLIP},
    "int8": {"unet": MINIMAX_H3_UNET_INT8, "ref2va": MINIMAX_H3_REF2VA_UNET_INT8, "clip": MINIMAX_H3_CLIP_INT8},
    "bf16": {"unet": MINIMAX_H3_UNET_BF16, "ref2va": MINIMAX_H3_REF2VA_UNET_BF16, "clip": MINIMAX_H3_CLIP},
}
MINIMAX_H3_MODEL_NAMES = (
    MINIMAX_H3_UNET_FP8, MINIMAX_H3_UNET_INT8, MINIMAX_H3_UNET_BF16,
    MINIMAX_H3_REF2VA_UNET_FP8, MINIMAX_H3_REF2VA_UNET_INT8, MINIMAX_H3_REF2VA_UNET_BF16,
    MINIMAX_H3_CLIP, MINIMAX_H3_CLIP_INT8,
    MINIMAX_H3_VIDEO_VAE, MINIMAX_H3_AUDIO_VAE,
)


def minimax_h3_models(quantization, ref2va=False):
    """The 4 model files a MiniMax H3 task needs for the given quantization."""
    q = MINIMAX_H3_QUANT_MODELS[quantization]
    unet = q["ref2va"] if ref2va else q["unet"]
    files = [unet, q["clip"], MINIMAX_H3_VIDEO_VAE, MINIMAX_H3_AUDIO_VAE]
    return [
        {"folder": "diffusion_models", "filename": unet,
         "url": f"{MINIMAX_H3_BASE_URL}/diffusion_models/{unet}"},
        {"folder": "text_encoders", "filename": q["clip"],
         "url": f"{MINIMAX_H3_BASE_URL}/text_encoders/{q['clip']}"},
        {"folder": "vae", "filename": MINIMAX_H3_VIDEO_VAE,
         "url": f"{MINIMAX_H3_BASE_URL}/vae/{MINIMAX_H3_VIDEO_VAE}"},
        {"folder": "vae", "filename": MINIMAX_H3_AUDIO_VAE,
         "url": f"{MINIMAX_H3_BASE_URL}/vae/{MINIMAX_H3_AUDIO_VAE}"},
    ]


def minimax_h3_length(duration):
    """Duration (seconds) -> frame count at 24 fps (min 5; the nodes snap it
    to the model's 17k+5 grid)."""
    return max(5, round(duration * MINIMAX_H3_FPS))


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


def build_minimax_h3_text_to_video(*, prompt, seed=0, steps=50, width=1344, height=768,
                                   duration=5.0, scheduler="beta",
                                   filename_prefix="wrapper/minimaxh3_t2v",
                                   unet_name=MINIMAX_H3_UNET_FP8,
                                   clip_name=MINIMAX_H3_CLIP):
    """MiniMax H3 text-to-video: prompt -> joint audio+video MP4."""
    return _minimax_h3_cond_and_tail(prompt, width, height, minimax_h3_length(duration),
                                     seed, steps, scheduler, unet_name, clip_name, {}, {}, 7,
                                     filename_prefix)


def build_minimax_h3_image_to_video(*, prompt, seed=0, steps=50, width=1344, height=768,
                                    duration=5.0, scheduler="beta",
                                    filename_prefix="wrapper/minimaxh3_i2v",
                                    first_frame=None, last_frame=None,
                                    unet_name=MINIMAX_H3_UNET_FP8,
                                    clip_name=MINIMAX_H3_CLIP):
    """MiniMax H3 image-to-video: first (and optional last) frame + prompt ->
    joint audio+video MP4."""
    cond_inputs = {}
    loader_nodes = {}
    next_id = 7
    for input_name, ref in (("first_frame", first_frame), ("last_frame", last_frame)):
        if ref is None:
            continue
        loader_nodes[str(next_id)] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        cond_inputs[input_name] = [str(next_id), 0]
        next_id += 1
    return _minimax_h3_cond_and_tail(prompt, width, height, minimax_h3_length(duration),
                                     seed, steps, scheduler, unet_name, clip_name,
                                     cond_inputs, loader_nodes, next_id, filename_prefix)


def build_minimax_h3_reference_to_video(*, prompt, seed=0, steps=50, width=1344, height=768,
                                        duration=5.0, scheduler="beta", ref_image_size="match",
                                        filename_prefix="wrapper/minimaxh3_ref2va",
                                        ref_images=(), ref_videos=(), ref_audios=(),
                                        unet_name=MINIMAX_H3_REF2VA_UNET_FP8,
                                        clip_name=MINIMAX_H3_CLIP):
    """MiniMax H3 reference-to-video (ref2va): reference images/videos/audio +
    prompt -> joint audio+video MP4. The prompt refers to references by tag
    (<Picture i> / <Video k> / <Audio j>) in the order they were provided."""
    length = minimax_h3_length(duration)
    ref_inputs = {}
    loader_nodes = {}
    next_id = 7
    for i, ref in enumerate(ref_images):
        ref_inputs.setdefault("ref_images", {})[f"ref_image_{i}"] = [str(next_id), 0]
        loader_nodes[str(next_id)] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        next_id += 1
    for i, ref in enumerate(ref_videos):
        ref_inputs.setdefault("ref_videos", {})[f"ref_video_{i}"] = [str(next_id), 0]
        loader_nodes[str(next_id)] = {"class_type": "LoadVideo", "inputs": {"file": ref}}
        next_id += 1
    for i, ref in enumerate(ref_audios):
        ref_inputs.setdefault("ref_audios", {})[f"ref_audio_{i}"] = [str(next_id), 0]
        loader_nodes[str(next_id)] = {"class_type": "LoadAudio", "inputs": {"audio": ref}}
        next_id += 1
    head = _minimax_h3_head(prompt, width, height, length, unet_name, clip_name,
                            MINIMAX_H3_VIDEO_VAE, MINIMAX_H3_AUDIO_VAE, {})
    head["6"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
        "prompt": prompt, "width": width, "height": height, "length": length,
        "ref_image_size": ref_image_size, **ref_inputs}}
    tail = _minimax_h3_tail(["2", 0], ["6", 0], ["6", 1], seed, steps, scheduler,
                            next_id, filename_prefix)
    return {**head, **loader_nodes, **tail}

MINIMAX_H3_FORM_EXTRA = {
    "width": {"type": "integer", "minimum": 32, "maximum": 8192, "default": 1344,
              "description": "Output width (rounded to a multiple of 32)."},
    "height": {"type": "integer", "minimum": 32, "maximum": 8192, "default": 768,
               "description": "Output height (rounded to a multiple of 32)."},
    "duration": {"type": "number", "minimum": 0.2, "maximum": 150, "default": 5.0,
                  "description": "Video length in seconds at 24 fps (snapped to the model's frame grid; trained range ~124-362 frames = 5-15s)."},
    "scheduler": {"type": "string", "enum": ["beta", "normal", "simple"], "default": "beta",
                   "description": "Sigma scheduler; beta/normal outperform simple for reference-heavy prompts."},
}
MINIMAX_H3_REF_FORM_EXTRA = {
    **MINIMAX_H3_FORM_EXTRA,
    "ref_image_size": {"type": "string", "enum": ["match", "max"], "default": "match",
                        "description": "Reference image sizing: 'match' downscales refs to the generation's pixel area (faster); 'max' keeps a 2048px short edge for stronger identity fidelity (slower)."},
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
"""

WORKFLOWS = {
    "flux2klein9b": {
        "title": "FLUX.2 [klein] 9B image edit",
        "requires_image": True,
        "uploads": {"image": {"ext": "image", "max": 1}},
        "uses": ["prompt", "negative_prompt", "seed", "steps", "cfg", "megapixels"],
        "form": ["prompt", "image", "negative_prompt", "seed", "steps", "cfg", "megapixels"],
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
                "uses": ["prompt", "seed"],
                "form": ["prompt", "seed", "steps", "width", "height", "duration", "scheduler"],
                "extra_form_properties": MINIMAX_H3_FORM_EXTRA,
                "quantization_options": ["fp8", "int8", "bf16"],
                "build": build_minimax_h3_text_to_video,
            },
            "image": {
                "title": "Image to video",
                "requires_image": True,
                "uploads": {"image": {"ext": "image", "max": 1},
                             "last_frame": {"ext": "image", "max": 1}},
                "upload_params": {"image": "first_frame"},
                "uses": ["prompt", "seed"],
                "form": ["prompt", "image", "last_frame", "seed", "steps", "width", "height", "duration", "scheduler"],
                "extra_form_properties": MINIMAX_H3_FORM_EXTRA,
                "quantization_options": ["fp8", "int8", "bf16"],
                "build": build_minimax_h3_image_to_video,
            },
            "reference": {
                "title": "Reference to video (ref2va)",
                "requires_image": False,
                "uploads": {"ref_images": {"ext": "image", "max": 9},
                             "ref_videos": {"ext": "video", "max": 3},
                             "ref_audios": {"ext": "audio", "max": 3}},
                "uses": ["prompt", "seed"],
                "form": ["prompt", "seed", "steps", "width", "height", "duration", "scheduler", "ref_image_size"],
                "extra_form_properties": MINIMAX_H3_REF_FORM_EXTRA,
                "quantization_options": ["fp8", "int8", "bf16"],
                "example_prompt": (
                    "Show <Picture 1> skateboarding down a sunlit street, the camera follows "
                    "from the side, <Audio 1> with the sound of wheels rolling on asphalt, "
                    "cinematic 24fps handheld shot."),
                "build": build_minimax_h3_reference_to_video,
            },
        },
    },
}
