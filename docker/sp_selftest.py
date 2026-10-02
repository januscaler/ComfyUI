"""One rank of the sequence-parallel transport self-test.

docker/entrypoint.sh runs one copy per GPU before ComfyUI starts, laid out as the
real group is (each copy sees only its own GPU), with a hard time limit. Only when
every copy exits 0 does the container split renders over its GPUs; otherwise it
renders on one, instead of hanging inside a customer's render.

Why it exists: on a RunPod 2x RTX PRO 6000 worker (2026-10-02) the group's first
broadcast never completed: both ranks waited 600 s in it, the NCCL watchdog
aborted them, and the render was lost. A transport can fail that way on one host
and not another (peer-to-peer advertised but blocked by the host, a /dev/shm too
small for NCCL's buffers), and only running it tells.

The test is the render's own traffic in miniature: a broadcast of the size one
forward's inputs take, an all-to-all like attention's, and an all-gather like the
last block's, each checked for the right values.

Env: COMFY_SP_RANK, COMFY_SP_WORLD, COMFY_SP_PORT (a store of its own, not the
real group's), CUDA_VISIBLE_DEVICES; COMFY_SP_BACKEND=gloo for a CPU run.
"""

import logging
import os
import sys
import time
from datetime import timedelta

#: About one 720p 15 s forward's inputs, the first thing a render sends.
BROADCAST_BYTES = 48 * 1024 * 1024
#: Per peer, for the all-to-all.
EXCHANGE_FLOATS = 1024 * 1024


def _connect(port: int, world: int, rank: int, deadline: float):
    import torch.distributed as dist

    while True:
        try:
            return dist.TCPStore("127.0.0.1", port, world, is_master=rank == 0,
                                 wait_for_workers=False, timeout=timedelta(seconds=30))
        except dist.DistNetworkError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)


def main() -> int:
    started = time.monotonic()
    rank = int(os.environ["COMFY_SP_RANK"])
    world = int(os.environ["COMFY_SP_WORLD"])
    port = int(os.environ["COMFY_SP_PORT"])

    import torch
    import torch.distributed as dist

    backend = os.environ.get("COMFY_SP_BACKEND") or ("nccl" if torch.cuda.is_available() else "gloo")
    if backend == "nccl":
        torch.cuda.set_device(0)  # this process sees only its own GPU
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    store = _connect(port, world, rank, started + 60)
    dist.init_process_group(backend=backend, store=store, rank=rank, world_size=world,
                            timeout=timedelta(seconds=60),
                            device_id=device if backend == "nccl" else None)

    data = torch.full((BROADCAST_BYTES,), 7 if rank == 0 else 0, dtype=torch.uint8, device=device)
    dist.broadcast(data, src=0)
    ok = bool((data == 7).all())

    send = torch.full((world * EXCHANGE_FLOATS,), float(rank), device=device)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    expected = torch.arange(world, dtype=send.dtype).repeat_interleave(EXCHANGE_FLOATS)
    ok = ok and torch.equal(recv.cpu(), expected)

    gathered = torch.empty(world, device=device)
    dist.all_gather_into_tensor(gathered, torch.tensor([float(rank)], device=device))
    ok = ok and torch.equal(gathered.cpu(), torch.arange(world, dtype=gathered.dtype))

    if backend == "nccl":
        torch.cuda.synchronize()
    dist.destroy_process_group()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    logging.info("sp-selftest %d/%d: %s (%s, %.1fs)", rank, world, "ok" if ok else "WRONG DATA", backend,
                 time.monotonic() - started)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
