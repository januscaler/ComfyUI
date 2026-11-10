#!/usr/bin/env python3
"""Exit 0 when the CUDA driver sees a GPU; otherwise print why and exit 1.

docker/idle_gate.py runs this while ComfyUI is stopped (--down-check), so a GPU
the driver has lost reads as unhealthy before a render is sent to it. On
2026-11-11 the rig's RTX 5090 hung in the driver (nvidia-smi: ERR!): every
start of ComfyUI died at once with "No CUDA GPUs are available", while the
gate's health answer still said the worker was fine.

cuInit and cuDeviceGetCount are what torch's device count rests on. Neither one
creates a context, so the check holds no VRAM and takes well under a second.
Standard library only.
"""

from __future__ import annotations

import ctypes
import sys

LIBRARIES = ("libcuda.so.1", "libcuda.so")


def describe(lib, status: int) -> str:
    text = ctypes.c_char_p()
    try:
        if lib.cuGetErrorString(status, ctypes.byref(text)) == 0 and text.value:
            return f"CUDA error {status} ({text.value.decode('utf-8', 'replace')})"
    except AttributeError:
        pass
    return f"CUDA error {status}"


def check(load=ctypes.CDLL) -> str | None:
    """None when the driver sees at least one GPU, else why not."""
    lib = None
    for name in LIBRARIES:
        try:
            lib = load(name)
            break
        except OSError:
            continue
    if lib is None:
        return f"the CUDA driver ({LIBRARIES[0]}) could not be loaded"
    status = lib.cuInit(0)
    if status != 0:
        return f"cuInit failed: {describe(lib, status)}"
    count = ctypes.c_int(0)
    status = lib.cuDeviceGetCount(ctypes.byref(count))
    if status != 0:
        return f"cuDeviceGetCount failed: {describe(lib, status)}"
    if count.value < 1:
        return "the CUDA driver sees no GPU"
    return None


def main() -> int:
    problem = check()
    if problem:
        sys.stdout.write(f"GPU check: {problem}\n")
        sys.stdout.flush()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
