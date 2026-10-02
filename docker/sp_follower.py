"""A sequence-parallel follower: one extra GPU of a MiniMax H3 render.

docker/entrypoint.sh starts one per extra GPU, next to ComfyUI, with the same
ComfyUI flags (so the same attention and kernel choices) and:

  CUDA_VISIBLE_DEVICES  this follower's GPU only
  COMFY_SP_RANK         1..N-1
  COMFY_SP_WORLD        N (ComfyUI is rank 0)
  COMFY_SP_PORT         the TCP store ComfyUI hosts

It waits for ComfyUI to open the group, then runs its share of every H3 forward
ComfyUI posts, and exits when ComfyUI does. See comfy/ldm/minimax/sequence_parallel.py.
"""

import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# As main.py: ComfyUI's own flags from argv, then the CUDA allocator choice,
# which must happen before torch first touches CUDA.
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
from comfy.cli_args import args  # noqa: E402,F401
import cuda_malloc  # noqa: E402,F401

rank = os.environ.get("COMFY_SP_RANK", "?")
logging.basicConfig(level=logging.INFO, format=f"sp-follower {rank}: %(message)s", stream=sys.stderr)

from comfy.ldm.minimax import sequence_parallel  # noqa: E402

if __name__ == "__main__":
    sequence_parallel.follower_main()
