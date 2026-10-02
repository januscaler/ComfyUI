"""Rank 0 of a CPU sequence-parallel run, for test_minimax_h3_sequence_parallel.py.

  python sp_harness.py make <dir>           write a tiny random MiniMax H3 to <dir>/tiny_h3.safetensors
  python sp_harness.py run <ckpt>           load it as ComfyUI does (COMFY_SP_* env set by the test) and
                                            print one JSON line per check

`run` compares the model's forward with the sequence-parallel wrapper (rank 0 plus
the followers the test started) against the same model's dense forward.
"""

import json
import os
import sys
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
COMMAND, TARGET = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], "--cpu"]  # ComfyUI's own flags, as the followers get them

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401
import torch  # noqa: E402

LATENT_T, LAT_H, LAT_W, AUDIO_T, TEXT_LEN = 2, 8, 12, 9, 16


def make(directory):
    import comfy.ops
    import comfy.utils
    from comfy.ldm.minimax.model import MiniMaxH3Model

    torch.manual_seed(0)
    model = MiniMaxH3Model(hidden_size=256, num_layers=3, token_refiner_num_layers=1, num_attention_heads=4,
                           attention_head_dim=128, ffn_hidden_size=512, text_dim=64, time_embed_hidden_size=256,
                           time_embed_dim=128, operations=comfy.ops.manual_cast, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() == 1 and "norm" in name:
                param.fill_(1.0)
            else:
                param.normal_(0.0, 0.05)
        model.rope.inv_freq.copy_(1.0 / (10000.0 ** (torch.arange(16, dtype=torch.float32) / 16.0)))
    path = os.path.join(directory, "tiny_h3.safetensors")
    comfy.utils.save_torch_file({k: v.contiguous() for k, v in model.state_dict().items()}, path)
    print(path)  # noqa: T201


def report(check, **fields):
    print(json.dumps({"check": check, **fields}), flush=True)  # noqa: T201


def run(ckpt):
    import comfy.model_management as mm
    import comfy.patcher_extension
    import comfy.sd
    from comfy.ldm.minimax import sequence_parallel
    from comfy.ldm.minimax.model import PackedLayout

    patcher = comfy.sd.load_diffusion_model(ckpt)
    mm.load_models_gpu([patcher])
    model = patcher.model.diffusion_model
    group = sequence_parallel.get_group()
    # As comfy.sampler_helpers does before sampling: the patcher's wrappers ride in transformer_options.
    sampling_options = comfy.patcher_extension.copy_nested_dicts(patcher.model_options["transformer_options"])
    comfy.patcher_extension.merge_nested_dicts(sampling_options.setdefault("wrappers", {}), patcher.wrappers, copy_dict1=False)
    wrappers = comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, sampling_options)
    report("attached", active=bool(group and group.active), wrappers=len(wrappers))

    gen = torch.Generator().manual_seed(3)
    video = torch.randn(1, 24, LATENT_T, LAT_H, LAT_W, generator=gen)
    audio = torch.randn(1, 32, 2, AUDIO_T, generator=gen)
    context = torch.randn(1, TEXT_LEN, 64, generator=gen) * 0.5
    sigmas = torch.linspace(1.0, 0.0, 4)
    mask = torch.ones(1, 1, LATENT_T, LAT_H, LAT_W)
    mask[..., :4, :] = 0.25
    mask[..., 4:6, :] = 0.6
    payload = {"layout": PackedLayout(TEXT_LEN, LATENT_T, LAT_H, LAT_W, AUDIO_T), "audio_scale": 1.0, "seed": 0}

    def forward(split, sigma, **extra):
        options = {"sample_sigmas": sigmas}
        if split:
            options = {**comfy.patcher_extension.copy_nested_dicts(sampling_options), **options}
        with torch.inference_mode():
            return model([video, audio], torch.tensor([sigma * 1000.0]), context, transformer_options=options,
                         minimax_payload=payload, **extra)

    def diff(a, b):
        return max(float((x - y).abs().max()) for x, y in zip(a, b))

    # A split forward leaves ComfyUI's allocation compiler (it cannot plan the split blocks).
    import comfy.model_prefetch
    ends = []
    original_end = comfy.model_prefetch.malloc_graph_end
    comfy.model_prefetch.malloc_graph_end = lambda: (ends.append(1), original_end())[1]
    sent = group.seq if group else 0
    split = [forward(True, s) for s in (0.9, 0.5, 0.1)]
    graph_ends = len(ends)
    comfy.model_prefetch.malloc_graph_end = original_end
    report("forward", diff=max(diff(out, forward(False, s)) for out, s in zip(split, (0.9, 0.5, 0.1))),
           posted=(group.seq if group else 0) - sent, malloc_graph_ends=graph_ends)

    report("masked", diff=diff(forward(True, 0.7, denoise_mask=mask), forward(False, 0.7, denoise_mask=mask)))

    # A LoRA'd model: the followers hold the plain weights, so rank 0 renders alone.
    clean = patcher.model.current_weight_patches_uuid
    patcher.model.current_weight_patches_uuid = uuid.uuid4()
    sent = group.seq if group else 0
    lora = forward(True, 0.5)
    report("patched", diff=diff(lora, forward(False, 0.5)), posted=(group.seq if group else 0) - sent)
    patcher.model.current_weight_patches_uuid = clean

    if not (group and group.active):
        return  # a dense forward stops at a cancel at once, as always
    # A cancel during a split forward waits for its end.
    mm.interrupt_current_processing(True)
    finished = forward(True, 0.5)
    try:
        mm.throw_exception_if_processing_interrupted()
        raised = False
    except mm.InterruptProcessingException:
        raised = True
    report("interrupt", diff=diff(finished, forward(False, 0.5)), raised_after=raised)


if __name__ == "__main__":
    {"make": make, "run": run}[COMMAND](TARGET)
