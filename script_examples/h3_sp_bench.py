"""Time one MiniMax H3 render's DiT on 1 GPU against N GPUs with sequence parallelism.

Drives the block patches of comfy/ldm/minimax/sequence_parallel.py directly, one
torchrun rank per GPU (a render instead runs ComfyUI plus docker/sp_follower.py).
Every run uses the same seeded inputs (noise latents and random text states), so
outputs can be compared.

  # one GPU, the dense model as it renders today
  python script_examples/h3_sp_bench.py --ckpt <unet.safetensors> --width 640 --height 1120 --frames 124 --out ref.pt
  # four GPUs, one forward split across them
  torchrun --nproc-per-node 4 script_examples/h3_sp_bench.py --ckpt <unet.safetensors> --width 640 --height 1120 --frames 124 --out sp4.pt
  # how far apart the outputs are
  python script_examples/h3_sp_bench.py --compare ref.pt sp4.pt
  # correctness on CPU with a tiny random model (no weights, no GPU)
  torchrun --nproc-per-node 2 script_examples/h3_sp_bench.py --tiny --cpu --out tiny2.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FPS = 24
AUDIO_LATENT_FPS = 40


def align_frame_count(n: int) -> int:
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", help="MiniMax H3 diffusion model (e.g. minimax_h3_fl2va_pruned_int8_convrot.safetensors)")
    p.add_argument("--tiny", action="store_true", help="a tiny random H3 instead of a checkpoint (correctness only)")
    p.add_argument("--cpu", action="store_true", help="run on CPU (gloo); with --tiny")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=1120)
    p.add_argument("--frames", type=int, default=124, help="pixel frames at 24 fps, snapped to 17k+5 (124 = 5.17 s, 362 = 15.08 s)")
    p.add_argument("--text-len", type=int, default=256)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=4, help="timed forwards (one forward = one sampling step)")
    p.add_argument("--steps", type=int, default=3, help="euler steps from noise after timing, to compare whole trajectories (0 = skip)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--attention", choices=["default", "sage", "flash", "pytorch"], default="default")
    p.add_argument("--no-sp", action="store_true", help="with several ranks, run the dense model on each (sanity)")
    p.add_argument("--load", choices=["production", "full"], default="production",
                   help="production: ComfyUI's own memory estimate decides how much of the DiT stays on the GPU, as a render does "
                        "(each SP rank asks for 1/N of the activations); full: the whole DiT on the GPU")
    p.add_argument("--out", help="save outputs and timings here (rank 0)")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two saved runs and exit")
    return p.parse_args()


def compare(a_path: str, b_path: str) -> None:
    import torch

    a, b = torch.load(a_path), torch.load(b_path)
    report = {}
    for key in ("forward_video", "forward_audio", "final_video", "final_audio"):
        if key not in a or key not in b:
            continue
        x, y = a[key].float(), b[key].float()
        report[key] = {
            "max_abs": float((x - y).abs().max()),
            "rel_l2": float((x - y).norm() / x.norm().clamp(min=1e-12)),
            "cosine": float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)),
        }
    for name, run in (("A", a), ("B", b)):
        report[name] = {k: run[k] for k in ("world", "tokens", "step_seconds", "peak_gb", "attention") if k in run}
    if "step_seconds" in a and "step_seconds" in b:
        report["speedup"] = round(a["step_seconds"] / b["step_seconds"], 3)
    print(json.dumps(report, indent=2))  # noqa: T201


def tiny_model(device, dtype):
    import torch
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model

    torch.manual_seed(0)
    model = MiniMaxH3Model(hidden_size=256, num_layers=3, token_refiner_num_layers=1, num_attention_heads=4,
                           attention_head_dim=128, ffn_hidden_size=512, text_dim=64, time_embed_hidden_size=256,
                           time_embed_dim=128, operations=comfy.ops.manual_cast, dtype=dtype, device=device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() == 1 and "norm" in name:
                param.fill_(1.0)
            else:
                param.normal_(0.0, 0.05)
        model.rope.inv_freq.copy_(1.0 / (10000.0 ** (torch.arange(16, dtype=torch.float32) / 16.0)))
    # as a loaded checkpoint: the in-place rope kernel refuses weights that track gradients
    return model.requires_grad_(False).eval()


def run_all(args, forward, sync, video, audio, sigmas):
    """Warm-up, timed forwards at sigma 0.9, then a few euler steps from the same noise."""
    for _ in range(args.warmup):
        forward(video, audio, 0.9)
    sync()
    times = []
    first = None
    for _ in range(args.iters):
        started = time.perf_counter()
        out = forward(video, audio, 0.9)
        sync()
        times.append(time.perf_counter() - started)
        if first is None:
            first = out

    final_video, final_audio = video.clone(), audio.clone()
    for i in range(args.steps):
        v = forward(final_video, final_audio, float(sigmas[i]))
        dt = float(sigmas[i + 1] - sigmas[i])
        final_video = final_video + dt * v[0]
        final_audio = final_audio + dt * v[1]
    sync()
    return times, first, final_video, final_audio


def main() -> None:
    args = parse_args()
    if args.compare:
        compare(*args.compare)
        return

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    import torch
    import torch.distributed as dist

    if not args.cpu:
        torch.cuda.set_device(local_rank)
    import comfy.cli_args

    comfy.cli_args.args.cpu = args.cpu
    if args.attention == "sage":
        comfy.cli_args.args.use_sage_attention = True
    elif args.attention == "flash":
        comfy.cli_args.args.use_flash_attention = True
    elif args.attention == "pytorch":
        comfy.cli_args.args.use_pytorch_cross_attention = True
    if world > 1:
        if args.cpu:
            dist.init_process_group("gloo")
        else:
            dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    import comfy.model_management as mm
    import comfy.sd

    device = mm.get_torch_device()
    if args.tiny:
        dtype = torch.float32 if args.cpu else torch.bfloat16
        model = tiny_model(device, dtype)

    gen = torch.Generator("cpu").manual_seed(args.seed)
    frames = align_frame_count(args.frames)
    latent_t = video_latent_t(frames)
    audio_t = round(frames / FPS * AUDIO_LATENT_FPS)
    lat_h, lat_w = args.height // 16, args.width // 16
    sp_ranks = world if world > 1 and not args.no_sp else 1
    if not args.tiny:
        patcher = comfy.sd.load_diffusion_model(args.ckpt)
        if args.load == "full":
            mm.load_models_gpu([patcher], force_full_load=True)
        else:
            # as the sampler does: activations by ComfyUI's estimate, plus its inference reserve
            needed = patcher.model.memory_required([1, 24, latent_t, lat_h, lat_w]) / sp_ranks
            mm.load_models_gpu([patcher], memory_required=needed + mm.minimum_inference_memory())
        model = patcher.model.diffusion_model
        dtype = patcher.model.get_dtype_inference()
    video = torch.randn(1, 24, latent_t, lat_h, lat_w, generator=gen).to(device)
    audio = torch.randn(1, 32, 2, audio_t, generator=gen).to(device)
    text_dim = model.condition_proj.in_features
    context = (torch.randn(1, args.text_len, text_dim, generator=gen) * 0.5).to(device=device, dtype=dtype)
    tokens = args.text_len + audio_t * 2 + latent_t * (lat_h // 2) * (lat_w // 2)
    sigmas = torch.linspace(1.0, 0.0, max(args.steps, 1) + 1)

    base = {"sample_sigmas": sigmas.to(device)}
    if world > 1 and not args.no_sp:
        from comfy.ldm.minimax.sequence_parallel import H3SequenceParallel

        base["patches_replace"] = H3SequenceParallel(model).patches()

    def forward(video_x, audio_x, sigma):
        options = dict(base)  # _forward writes into it; the patches stay shared
        timestep = torch.tensor([float(sigma) * 1000.0], device=device)
        with torch.inference_mode():
            return model([video_x, audio_x], timestep, context, transformer_options=options, minimax_payload={})

    def sync():
        if not args.cpu:
            torch.cuda.synchronize()
        if world > 1:
            dist.barrier()

    if not args.cpu:
        torch.cuda.reset_peak_memory_stats()
    try:
        times, first, final_video, final_audio = run_all(args, forward, sync, video, audio, sigmas)
    except torch.cuda.OutOfMemoryError as error:
        print(json.dumps({"world": world, "rank": rank, "tokens": tokens, "canvas": f"{args.width}x{args.height}", "frames": frames,  # noqa: T201
                          "load": args.load, "oom": str(error).splitlines()[0][:200]}))
        if world > 1:
            dist.destroy_process_group()
        return

    peak = torch.cuda.max_memory_allocated() / 1e9 if not args.cpu else 0.0
    step_seconds = sorted(times)[len(times) // 2]
    loaded = None
    if not args.tiny:
        loaded = round(patcher.loaded_size() / 1e9, 2) if hasattr(patcher, "loaded_size") else None
    summary = {"world": world, "sp": world > 1 and not args.no_sp, "tokens": tokens, "frames": frames, "load": args.load, "dit_on_gpu_gb": loaded,
               "canvas": f"{args.width}x{args.height}", "step_seconds": round(step_seconds, 4),
               "all_step_seconds": [round(t, 4) for t in times], "peak_gb": round(peak, 2), "attention": args.attention}
    if world > 1:
        peaks = [None] * world
        dist.all_gather_object(peaks, summary["peak_gb"])
        summary["peak_gb_per_rank"] = peaks
    if rank == 0:
        print(json.dumps(summary))  # noqa: T201
        if args.out:
            torch.save({**summary, "forward_video": first[0].float().cpu(), "forward_audio": first[1].float().cpu(),
                        "final_video": final_video.float().cpu(), "final_audio": final_audio.float().cpu()}, args.out)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
