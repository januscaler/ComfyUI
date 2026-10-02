"""Checks that a sequence-parallel group's GPUs can move tensors to each other, before ComfyUI starts.

NCCL can open a group on a host whose GPU-to-GPU path carries no data, and then
hang in the first collective. On RunPod (2x RTX PRO 6000, 2026-10-03) a render's
first broadcast never completed; the watchdog ended it after
COMFY_SP_TIMEOUT_SECONDS (600 s), and the render failed. docker/entrypoint.sh
runs this once per boot with the GPUs it is about to give the group:

    python docker/sp_probe.py GPU [GPU ...]

One process per GPU, each seeing only its own GPU (as ComfyUI and the followers
do), joins a group. It runs the collectives a split forward uses (a broadcast
from rank 0, an all-to-all, an all-gather) on 64 MB buffers and checks every
value that arrived. Exit 0 when every rank did so within COMFY_SP_PROBE_SECONDS
(default 90). Otherwise exit 1, after killing any rank still running: a hung
collective ends with its process.

NCCL reads its settings (NCCL_P2P_DISABLE and the like) from the environment
when a group starts, so the entrypoint tries another setting in a new probe.

Env: COMFY_SP_PROBE_SECONDS, and COMFY_SP_BACKEND (nccl with CUDA; gloo for the CPU tests).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from datetime import timedelta

DEFAULT_SECONDS = 90
# int32 values per collective: 64 MB, about a 720p forward's broadcast (40 MB) and all-to-alls.
ELEMENTS = 16 * 1024 * 1024


def log(message: str) -> None:
    sys.stderr.write(f"sp-probe: {message}\n")
    sys.stderr.flush()


def probe_seconds() -> float:
    try:
        seconds = float(os.environ.get("COMFY_SP_PROBE_SECONDS", "") or DEFAULT_SECONDS)
    except ValueError:
        return DEFAULT_SECONDS
    return seconds if seconds > 0 else DEFAULT_SECONDS


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rank_command(rank: int, world: int, port: int) -> list[str]:
    return [sys.executable, os.path.abspath(__file__), "--rank", str(rank), str(world), str(port)]


def probe(gpus: list[str], seconds: float, launch=subprocess.Popen) -> str | None:
    """None when every rank passed within `seconds`; otherwise what went wrong. No rank outlives it."""
    world = len(gpus)
    port = free_port()
    procs = []
    try:
        for rank, gpu in enumerate(gpus):
            procs.append(launch(rank_command(rank, world, port), env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}))
        deadline = time.monotonic() + seconds
        while True:
            codes = [proc.poll() for proc in procs]
            failed = [(rank, code) for rank, code in enumerate(codes) if code not in (None, 0)]
            if failed:
                # The others may wait for this rank forever.
                return "; ".join(f"rank {rank} (GPU {gpus[rank]}) exited with status {code}" for rank, code in failed)
            if all(code == 0 for code in codes):
                return None
            if time.monotonic() >= deadline:
                waiting = ", ".join(f"rank {rank} (GPU {gpus[rank]})" for rank, code in enumerate(codes) if code is None)
                return f"{waiting} did not finish within {seconds:.0f}s"
            time.sleep(0.1)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


# --- one rank -------------------------------------------------------------------

def exchange(rank: int, world: int, device) -> None:
    """The collectives of a split forward, each checked value by value; raises on the first wrong one."""
    import torch
    import torch.distributed as dist

    chunk = ELEMENTS // world
    total = chunk * world

    sent = torch.arange(total, dtype=torch.int32, device=device) * 7 + 3
    buf = sent.clone() if rank == 0 else torch.zeros(total, dtype=torch.int32, device=device)
    dist.broadcast(buf, src=0)
    if not torch.equal(buf, sent):
        raise RuntimeError("the broadcast from rank 0 arrived with wrong values")

    # Chunk j goes to rank j, carrying (sender, receiver).
    ranks = torch.arange(world, dtype=torch.int32, device=device)
    send = (rank * world + ranks).repeat_interleave(chunk)
    recv = torch.empty(total, dtype=torch.int32, device=device)
    dist.all_to_all_single(recv, send)
    if not torch.equal(recv, (ranks * world + rank).repeat_interleave(chunk)):
        raise RuntimeError("the all-to-all arrived with wrong values")

    part = torch.full((chunk,), rank, dtype=torch.int32, device=device)
    gathered = torch.empty(total, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(gathered, part)
    if not torch.equal(gathered, ranks.repeat_interleave(chunk)):
        raise RuntimeError("the all-gather arrived with wrong values")


def run_rank(rank: int, world: int, port: int, seconds: float) -> None:
    import warnings

    import torch
    import torch.distributed as dist

    # The probe makes the calls sequence_parallel.py makes, deprecated or not; one warning per rank is noise.
    warnings.filterwarnings("ignore", message=".*all_gather_into_tensor.* is deprecated", category=FutureWarning)

    backend = os.environ.get("COMFY_SP_BACKEND", "").strip().lower() or ("nccl" if torch.cuda.is_available() else "gloo")
    if backend == "nccl":
        torch.cuda.set_device(0)  # this process sees only its own GPU
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    timeout = timedelta(seconds=seconds)
    store = dist.TCPStore("127.0.0.1", port, world, is_master=rank == 0, timeout=timeout)
    dist.init_process_group(backend=backend, store=store, rank=rank, world_size=world, timeout=timeout,
                            device_id=device if backend == "nccl" else None)
    exchange(rank, world, device)
    dist.destroy_process_group()
    store.set(f"done/{rank}", "1")
    if rank == 0:
        store.wait([f"done/{other}" for other in range(world)])  # the store's host leaves last


def main(argv: list[str]) -> int:
    if len(argv) == 5 and argv[1] == "--rank":
        rank, world, port = (int(value) for value in argv[2:])
        try:
            run_rank(rank, world, port, probe_seconds())
        except Exception as error:
            log(f"rank {rank}: {type(error).__name__}: {error}")
            return 1
        return 0
    gpus = argv[1:]
    if len(gpus) < 2:
        log("usage: sp_probe.py GPU GPU [GPU ...]")
        return 2
    setting = f" with NCCL_P2P_DISABLE={os.environ['NCCL_P2P_DISABLE']}" if "NCCL_P2P_DISABLE" in os.environ else ""
    started = time.monotonic()
    problem = probe(gpus, probe_seconds())
    if problem:
        log(f"GPUs {','.join(gpus)} could not exchange tensors{setting}: {problem}")
        return 1
    log(f"GPUs {','.join(gpus)} exchanged tensors{setting} (broadcast, all-to-all, all-gather) in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
