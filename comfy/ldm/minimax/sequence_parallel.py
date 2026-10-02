"""Sequence parallelism (DeepSpeed-Ulysses): one MiniMax H3 render's DiT on N GPUs.

ComfyUI is one process on one GPU. On a box with N GPUs, docker/entrypoint.sh
starts N-1 follower processes next to it (docker/sp_follower.py, each seeing only
its own GPU), and ComfyUI drives them as rank 0:

- When rank 0 loads an H3 diffusion model (comfy.sd.load_diffusion_model),
  `attach` adds a DIFFUSION_MODEL wrapper to its patcher.
- Each DiT forward then runs on every rank: rank 0 posts the call on a local TCP
  store (pickled without its tensors), broadcasts the tensors in one collective,
  and every rank runs the same `_forward` with block patches that hand it 1/N of
  the packed sequence. Inside a block everything is per token except attention,
  which trades the sequence split for a head split with one all-to-all, runs
  dense over the whole sequence for H/N heads, and trades back. The last block
  gathers the sequence again, so every rank ends with the whole output; the
  followers drop theirs.
- Followers load the checkpoint rank 0 loaded (its `cached_patcher_init`), the
  first time a forward needs it, and keep it on their GPU.

The math is the dense model's: each head still attends over the full sequence and
per-token ops see the same rows. On 4x RTX 5090 the outputs were bit-identical to
one GPU, and a DiT step ran 2.6x (480p 2.3 s), 3.3x (720p 5 s) and 3.6x (720p 15 s)
faster. The text encoder and the VAEs stay on rank 0.

A forward the followers could not reproduce exactly runs on rank 0 alone, as
without this module: weight patches (a LoRA), another diffusion-model wrapper,
block or attention patches, or an attention override.

The head count must divide by N (H3: 56 heads, so 2, 4, 7 or 8 GPUs).

Env (docker/entrypoint.sh sets it):
  COMFY_SP_WORLD            GPUs in the group, rank 0 included; unset or 1 = off
  COMFY_SP_RANK             a follower's rank, 1..N-1 (ComfyUI is rank 0)
  COMFY_SP_PORT             the local TCP store's port (default 29511)
  COMFY_SP_BACKEND          nccl (the default with CUDA) or gloo (CPU tests)
  COMFY_SP_TIMEOUT_SECONDS  how long a collective may take (default 600)
  COMFY_SP_LOAD_SECONDS     how long followers may take to load a model (default 1800)
  COMFY_SP_JOIN_SECONDS     how long rank 0 waits for its followers (default 300)
"""

from __future__ import annotations

import gc
import hashlib
import io
import logging
import math
import os
import pickle
import threading
import weakref
from datetime import timedelta

import torch
import torch.distributed as dist

import comfy.model_management
import comfy.patcher_extension
import comfy.quant_ops
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

WRAPPER_KEY = "h3_sequence_parallel"
DEFAULT_PORT = 29511
# What a follower's forward reads from rank 0's transformer_options (MiniMaxH3Model._forward).
FOLLOWER_OPTION_KEYS = ("minimax_h3_sigma_shift_video", "minimax_h3_sigma_shift_audio", "sample_sigmas")
# Tensors travel as one byte buffer; each starts on this boundary so it can be viewed as its dtype.
ALIGN = 16
# A follower's store wait: long, since each timed-out wait logs a c10d warning.
POLL_SECONDS = 3600


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def configured_world() -> int:
    return max(1, _env_int("COMFY_SP_WORLD", 1))


def _backend() -> str:
    backend = os.environ.get("COMFY_SP_BACKEND", "").strip().lower()
    if backend:
        return backend
    return "nccl" if torch.cuda.is_available() else "gloo"


def _comm_device(backend: str) -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")


def split_sizes(total: int, world: int) -> list[int]:
    """Rows per rank: as even as possible, the first ranks taking the remainder."""
    base, extra = divmod(total, world)
    return [base + (1 if rank < extra else 0) for rank in range(world)]


def local_segments(segments, start: int, stop: int):
    """The block's mod segments (start, stop, row) clipped to [start, stop) and rebased to 0.

    `row` is a mod-row index, or a per-token LongTensor of them (masked rows), which
    is sliced along with the rows.
    """
    out = []
    for a, b, row in segments:
        lo, hi = max(a, start), min(b, stop)
        if lo >= hi:
            continue
        if torch.is_tensor(row) and row.dim() > 0:
            row = row[lo - a:hi - a]
        out.append((lo - start, hi - start, row))
    return out


# --- moving one forward's arguments to the followers ---------------------------

class _TensorPickler(pickle.Pickler):
    """Pickles everything but tensors, which it collects (at any depth) in `tensors`."""

    def __init__(self, file, tensors: list):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.tensors = tensors

    def persistent_id(self, obj):
        if isinstance(obj, torch.Tensor):
            self.tensors.append(obj)
            return len(self.tensors) - 1
        return None


class _TensorUnpickler(pickle.Unpickler):
    def __init__(self, file, tensors: list):
        super().__init__(file)
        self.tensors = tensors

    def persistent_load(self, pid):
        return self.tensors[pid]


def pack(obj) -> tuple[bytes, list[torch.Tensor], list[tuple]]:
    """`obj` without its tensors, the tensors, and a (shape, dtype, device type) spec per tensor."""
    tensors: list[torch.Tensor] = []
    buf = io.BytesIO()
    _TensorPickler(buf, tensors).dump(obj)
    return buf.getvalue(), tensors, [(tuple(t.shape), t.dtype, t.device.type) for t in tensors]


def unpack(data: bytes, tensors: list[torch.Tensor]):
    return _TensorUnpickler(io.BytesIO(data), tensors).load()


def _span(shape, dtype) -> int:
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    return (nbytes + ALIGN - 1) // ALIGN * ALIGN


def flatten(tensors: list[torch.Tensor], specs: list[tuple], device: torch.device) -> torch.Tensor:
    """The tensors' bytes in one uint8 buffer on `device`, each at an ALIGN boundary."""
    buf = torch.zeros(sum(_span(shape, dtype) for shape, dtype, _ in specs), dtype=torch.uint8, device=device)
    offset = 0
    for t, (shape, dtype, _) in zip(tensors, specs):
        raw = t.detach().reshape(-1).contiguous().view(torch.uint8)
        buf[offset:offset + raw.numel()].copy_(raw)
        offset += _span(shape, dtype)
    return buf


def unflatten(buf: torch.Tensor, specs: list[tuple], device: torch.device) -> list[torch.Tensor]:
    """`flatten` undone: rank 0's CUDA tensors land on `device`, its CPU tensors on the CPU."""
    out = []
    offset = 0
    for shape, dtype, kind in specs:
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        t = buf[offset:offset + nbytes].view(dtype).reshape(shape)
        out.append(t.cpu() if kind == "cpu" else t.to(device))
        offset += _span(shape, dtype)
    return out


def buffer_size(specs: list[tuple]) -> int:
    return sum(_span(shape, dtype) for shape, dtype, _ in specs)


# --- the block patches ----------------------------------------------------------

class _Pass:
    """What one forward needs on this rank, set at block 0 and dropped after the last."""

    __slots__ = ("sizes", "start", "stop", "segments", "rope")

    def __init__(self, sizes, start, stop, segments, rope):
        self.sizes = sizes
        self.start = start
        self.stop = stop
        self.segments = segments
        self.rope = rope


class H3SequenceParallel:
    """Ulysses block patches for one MiniMax H3 diffusion model, on the default process group."""

    def __init__(self, diffusion_model):
        self.model = diffusion_model
        self.heads = diffusion_model.blocks[0].attn.heads
        self._pass: _Pass | None = None
        self._patches: dict | None = None
        self.world = 1
        self.rank = 0

    def patches(self) -> dict:
        """`patches_replace` for one forward: every block hands this rank only its rows."""
        if self._patches is None:
            self.world = dist.get_world_size()
            self.rank = dist.get_rank()
            if self.heads % self.world:
                raise ValueError(f"MiniMax H3 has {self.heads} attention heads, which do not split evenly over {self.world} GPUs")
            count = len(self.model.blocks)
            self._patches = {("double_block", index): self._block(index, count) for index in range(count)}
        return {"dit": self._patches}

    def _block(self, index: int, count: int):
        attention = self._attention(self.model.blocks[index].attn)

        def run(args, extra):
            h = args["img"]
            if index == 0:
                self._begin(h, args)
                h = h[self._pass.start:self._pass.stop].clone()
            state = self._pass
            out = extra["original_block"]({**args, "img": h, "mod_segments": state.segments, "rope_freqs": state.rope, "attention": attention})["img"]
            if index == count - 1:
                out = self._gather(out)
                self._pass = None
            return {"img": out}

        return run

    def _begin(self, h: torch.Tensor, args: dict) -> None:
        sizes = split_sizes(h.shape[0], self.world)
        start = sum(sizes[:self.rank])
        stop = start + sizes[self.rank]
        rope = args["rope_freqs"]
        if rope is not None:
            rope = rope[:, start:stop].contiguous()
        self._pass = _Pass(sizes, start, stop, local_segments(args["mod_segments"], start, stop), rope)

    def _gather(self, h: torch.Tensor) -> torch.Tensor:
        sizes = self._pass.sizes
        widest = max(sizes)
        padded = h.new_zeros(widest, h.shape[1])
        padded[:h.shape[0]] = h
        gathered = h.new_empty(self.world * widest, h.shape[1])
        dist.all_gather_into_tensor(gathered, padded)
        return torch.cat([gathered[rank * widest:rank * widest + size] for rank, size in enumerate(sizes)])

    def _to_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[s_local, H, D] -> [S, H/N, D]: this rank's head group over the whole sequence."""
        s, heads, dim = t.shape
        group_heads = heads // self.world
        send = t.reshape(s, self.world, group_heads, dim).transpose(0, 1).contiguous().view(self.world * s, group_heads, dim)
        recv = t.new_empty(sum(self._pass.sizes), group_heads, dim)
        dist.all_to_all_single(recv, send, output_split_sizes=self._pass.sizes, input_split_sizes=[s] * self.world)
        return recv

    def _to_tokens(self, t: torch.Tensor, s: int) -> torch.Tensor:
        """[S, H/N, D] -> [s_local, H*D]: every head again, for this rank's rows."""
        _, group_heads, dim = t.shape
        recv = t.new_empty(self.world * s, group_heads, dim)
        dist.all_to_all_single(recv, t.contiguous(), output_split_sizes=[s] * self.world, input_split_sizes=self._pass.sizes)
        return recv.view(self.world, s, group_heads, dim).transpose(0, 1).reshape(s, self.world * group_heads * dim)

    def _attention(self, attn):
        """`Attention.forward` for this rank's rows: the same qkv, norms and rope, attention by head group."""

        def run(x, rope_freqs=None, transformer_options={}):
            s = x.shape[0]
            heads, dim = attn.heads, attn.head_dim
            q, k, v = attn.qkv_proj(x).split(heads * dim, dim=-1)
            v = v.view(s, heads, dim)
            if rope_freqs is not None:
                q = q.view(1, s, heads, dim)
                k = k.view(1, s, heads, dim)
                qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
                kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
                comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
                q = q[0]
                k = k[0]
            else:
                q = attn.q_norm(q.view(s, heads, dim))
                k = attn.k_norm(k.view(s, heads, dim))

            q, k, v = self._to_heads(q), self._to_heads(k), self._to_heads(v)
            group_heads = heads // self.world
            out = optimized_attention(
                AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0)),
                AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0)),
                AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0)),
                group_heads, preferred_attention=attn.comfy_attention, mask=None, skip_reshape=True,
                transformer_options=transformer_options)
            out = out.view(-1, group_heads, dim)
            return attn.out_proj(self._to_tokens(out, s))

        return run


# --- rank 0: ComfyUI ------------------------------------------------------------

def _init_process_group(store, rank: int, world: int, backend: str, timeout: int) -> None:
    device_id = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else None
    dist.init_process_group(backend=backend, store=dist.PrefixStore("pg", store), rank=rank, world_size=world,
                            timeout=timedelta(seconds=timeout), device_id=device_id)


class SequenceParallelGroup:
    """Rank 0's side: the store, the process group, and the commands to the followers."""

    def __init__(self, world: int):
        self.world = world
        self.port = _env_int("COMFY_SP_PORT", DEFAULT_PORT)
        self.backend = _backend()
        self.timeout = _env_int("COMFY_SP_TIMEOUT_SECONDS", 600)
        self.load_timeout = _env_int("COMFY_SP_LOAD_SECONDS", 1800)
        self.join_timeout = _env_int("COMFY_SP_JOIN_SECONDS", 300)
        self.lock = threading.Lock()
        self.store = None
        self.active = False
        self.seq = 0
        self.loaded_key: str | None = None

    def start(self) -> None:
        """Opens the store, waits for every follower, and joins the process group; on failure stays off."""
        try:
            self.store = dist.TCPStore("127.0.0.1", self.port, self.world, is_master=True, wait_for_workers=False,
                                       timeout=timedelta(seconds=self.timeout))
            self.store.wait([f"hello/{rank}" for rank in range(1, self.world)], timedelta(seconds=self.join_timeout))
            self.store.set("start", "1")
            _init_process_group(self.store, 0, self.world, self.backend, self.timeout)
            self.active = True
            logging.info("H3 sequence parallelism: %d GPUs (%s)", self.world, self.backend)
        except Exception as error:
            logging.warning("H3 sequence parallelism is off: its %d followers did not join (%s); renders use one GPU",
                            self.world - 1, error)

    def broken(self, error: BaseException) -> None:
        """A forward failed mid-collective: the group is out of step for good, so stop using it.

        The followers time out in their collective (COMFY_SP_TIMEOUT_SECONDS) and exit,
        and docker/entrypoint.sh then restarts the container with a fresh group.
        """
        self.active = False
        logging.error("H3 sequence parallelism stopped after a failed forward (%s); renders use one GPU until a restart", error)

    def _post(self, command: dict) -> int:
        seq = self.seq
        self.seq += 1
        self.store.set(f"cmd/{seq}", pickle.dumps(command, protocol=pickle.HIGHEST_PROTOCOL))
        return seq

    def ensure_model(self, key: str, path: str, options: dict) -> None:
        """The followers hold the model `key` (loaded from `path` with `options`) on their GPU."""
        if self.loaded_key == key:
            return
        self.loaded_key = None
        seq = self._post({"op": "load", "key": key, "path": path, "options": options})
        keys = [f"ack/{seq}/{rank}" for rank in range(1, self.world)]
        self.store.wait(keys, timedelta(seconds=self.load_timeout))
        errors = []
        for ack_key in keys:
            ack = pickle.loads(self.store.get(ack_key))
            self.store.delete_key(ack_key)
            if ack is not True:
                errors.append(f"{ack_key.rsplit('/', 1)[1]}: {ack}")
        self.store.delete_key(f"cmd/{seq}")
        if errors:
            raise RuntimeError("followers could not load " + os.path.basename(path) + ": " + "; ".join(errors))
        self.loaded_key = key

    def send_forward(self, key: str, packed: tuple) -> int:
        """Posts one forward (`pack`ed arguments) and broadcasts its tensors; the caller runs its own share next."""
        data, tensors, specs = packed
        seq = self._post({"op": "forward", "key": key, "data": data, "specs": specs})
        dist.broadcast(flatten(tensors, specs, _comm_device(self.backend)), src=0)
        return seq

    def forward_done(self, seq: int) -> None:
        # Every follower read the command before it joined the collectives that just finished.
        self.store.delete_key(f"cmd/{seq}")


_GROUP: SequenceParallelGroup | None = None
_GROUP_LOCK = threading.Lock()


def get_group() -> SequenceParallelGroup | None:
    """Rank 0's group, started on first use; None when this process runs without followers."""
    global _GROUP
    world = configured_world()
    if world < 2 or os.environ.get("COMFY_SP_RANK", "0") not in ("", "0"):
        return None
    with _GROUP_LOCK:
        if _GROUP is None:
            _GROUP = SequenceParallelGroup(world)
            _GROUP.start()
    return _GROUP


class _Attached:
    """An H3 diffusion model rank 0 can split: where the followers load it from, and its patches."""

    def __init__(self, base_model, path: str, options: dict, key: str, clean_patches_uuid):
        self.base_model = weakref.ref(base_model)
        self.path = path
        self.options = options
        self.key = key
        self.clean_patches_uuid = clean_patches_uuid
        self.sequence_parallel = H3SequenceParallel(base_model.diffusion_model)
        self.disabled: str | None = None
        self.reported: set[str] = set()


_ATTACHED: "weakref.WeakKeyDictionary[torch.nn.Module, _Attached]" = weakref.WeakKeyDictionary()


def ineligible(info: _Attached, transformer_options: dict) -> str | None:
    """Why the followers could not reproduce this forward exactly, or None when they can."""
    if info.disabled:
        return info.disabled
    base = info.base_model()
    applied = getattr(base, "current_weight_patches_uuid", None)
    if applied is not None and applied != info.clean_patches_uuid:
        return "weight patches (a LoRA or a compute-dtype change) are applied"
    if any(transformer_options.get("patches_replace", {}).values()):
        return "block patches are set"
    if any(transformer_options.get("patches", {}).values()):
        return "attention patches are set"
    if "optimized_attention_override" in transformer_options:
        return "an attention override is set"
    wrappers = comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
    if len(wrappers) != 1:
        return "another diffusion-model wrapper is set"
    return None


def _wrapper(executor, x, timestep, context, transformer_options, **kwargs):
    info = _ATTACHED.get(executor.class_obj)
    group = get_group() if info is not None else None
    if info is None or group is None or not group.active:
        return executor(x, timestep, context, transformer_options, **kwargs)
    reason = ineligible(info, transformer_options)
    if reason:
        if reason not in info.reported:
            info.reported.add(reason)
            logging.info("H3 sequence parallelism: this render runs on one GPU, since %s", reason)
        return executor(x, timestep, context, transformer_options, **kwargs)
    with group.lock:
        try:
            group.ensure_model(info.key, info.path, info.options)
        except Exception as error:
            info.disabled = f"the followers could not load the model ({error})"
            logging.warning("H3 sequence parallelism: %s", info.disabled)
            return executor(x, timestep, context, transformer_options, **kwargs)
        follower_options = {k: transformer_options[k] for k in FOLLOWER_OPTION_KEYS if k in transformer_options}
        try:
            packed = pack(((x, timestep, context, follower_options), kwargs))
        except Exception as error:
            reason = f"its arguments cannot reach the followers ({type(error).__name__}: {error})"
            if reason not in info.reported:
                info.reported.add(reason)
                logging.info("H3 sequence parallelism: this render runs on one GPU, since %s", reason)
            return executor(x, timestep, context, transformer_options, **kwargs)
        # From the post on, every follower is in this forward: rank 0 must finish it
        # too, so a cancel waits for the end of the step (the sampler checks again).
        with comfy.model_management.defer_processing_interrupt():
            try:
                seq = group.send_forward(info.key, packed)
                local = dict(transformer_options)
                local["patches_replace"] = info.sequence_parallel.patches()
                out = executor(x, timestep, context, local, **kwargs)
            except BaseException as error:
                group.broken(error)
                raise
        group.forward_done(seq)
        return out


def _scale_memory_estimate(base_model) -> None:
    """Each rank holds 1/N of the activations, so rank 0 reserves 1/N of the dense estimate.

    Without it ComfyUI would keep part of the DiT off the GPU to make room for
    activations it never allocates, and every rank would wait on rank 0's streaming.
    """
    if getattr(base_model, "_sequence_parallel_memory", False):
        return
    original = base_model.memory_required

    def memory_required(*args, **kwargs):
        need = original(*args, **kwargs)
        group = _GROUP
        return need / group.world if group is not None and group.active else need

    base_model.memory_required = memory_required
    base_model._sequence_parallel_memory = True


def attach(patcher) -> None:
    """Lets rank 0 split this H3 model's forwards over the followers (comfy.sd.load_diffusion_model calls it)."""
    world = configured_world()
    if world < 2:
        return
    from comfy.ldm.minimax.model import MiniMaxH3Model

    diffusion_model = getattr(patcher.model, "diffusion_model", None)
    if not isinstance(diffusion_model, MiniMaxH3Model):
        return
    init = getattr(patcher, "cached_patcher_init", None)
    if not init or getattr(init[0], "__name__", "") != "load_diffusion_model" or len(init) > 2:
        return
    path, options = init[1][0], dict(init[1][1] or {})
    heads = diffusion_model.blocks[0].attn.heads
    if heads % world:
        logging.warning("H3 sequence parallelism is off for this model: %d heads do not split over %d GPUs", heads, world)
        return
    try:
        blob = pickle.dumps(options, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        logging.warning("H3 sequence parallelism is off for this model: its load options cannot reach the followers (%s)", error)
        return
    if get_group() is None or not get_group().active:
        return
    key = hashlib.sha256(os.path.abspath(path).encode() + b"\0" + blob).hexdigest()[:16]
    _ATTACHED[diffusion_model] = _Attached(patcher.model, path, options, key, patcher.patches_uuid)
    patcher.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, _wrapper)
    _scale_memory_estimate(patcher.model)


# --- followers ------------------------------------------------------------------

def _wait(store, key: str) -> bytes:
    """`key`'s value once rank 0 sets it; exits when rank 0 (the store's host) is gone."""
    while True:
        try:
            store.wait([key], timedelta(seconds=POLL_SECONDS))
            return store.get(key)
        except dist.DistStoreError:
            continue  # a timed-out wait: rank 0 is idle
        except dist.DistNetworkError:
            logging.info("sequence-parallel follower: rank 0 is gone, exiting")
            raise SystemExit(0)


def _connect(port: int, world: int, timeout: int):
    """The store rank 0 hosts; ComfyUI opens it when it first loads an H3 model, so keep trying."""
    while True:
        try:
            return dist.TCPStore("127.0.0.1", port, world, is_master=False, timeout=timedelta(seconds=timeout))
        except dist.DistNetworkError:
            continue


def _load(command: dict, backend: str):
    import comfy.sd

    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache()
    # The plain patcher, fully on this GPU: nothing else runs here, and a dynamic
    # patcher would pin a host copy of the weights in every follower.
    patcher = comfy.sd.load_diffusion_model(command["path"], command["options"], disable_dynamic=True)
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    return patcher, H3SequenceParallel(patcher.model.diffusion_model)


def follower_main() -> None:
    """One follower: joins rank 0's group, then runs every forward rank 0 posts, until rank 0 exits."""
    rank = _env_int("COMFY_SP_RANK", 0)
    world = configured_world()
    if not 1 <= rank < world:
        raise SystemExit(f"sequence-parallel follower: COMFY_SP_RANK={rank} is not in 1..{world - 1}")
    backend = _backend()
    timeout = _env_int("COMFY_SP_TIMEOUT_SECONDS", 600)
    if backend == "nccl":
        torch.cuda.set_device(0)  # this process sees only its own GPU
    device = comfy.model_management.get_torch_device()
    store = _connect(_env_int("COMFY_SP_PORT", DEFAULT_PORT), world, timeout)
    store.set(f"hello/{rank}", str(os.getpid()))
    _wait(store, "start")
    try:
        _init_process_group(store, rank, world, backend, timeout)
    except Exception as error:
        # Rank 0 failed the same way and renders on one GPU; a clean exit keeps it running.
        logging.error("sequence-parallel follower %d: the group could not start (%s); exiting", rank, error)
        raise SystemExit(0)
    logging.info("sequence-parallel follower %d/%d joined (%s, %s)", rank, world, backend, device)

    loaded: dict[str, tuple] = {}
    seq = 0
    while True:
        command = pickle.loads(_wait(store, f"cmd/{seq}"))
        if command["op"] == "load":
            loaded.clear()
            try:
                loaded[command["key"]] = _load(command, backend)
                ack = True
            except Exception as error:
                logging.exception("sequence-parallel follower %d: loading %s failed", rank, command["path"])
                ack = f"{type(error).__name__}: {error}"
            store.set(f"ack/{seq}/{rank}", pickle.dumps(ack))
        elif command["op"] == "forward":
            patcher, sequence_parallel = loaded[command["key"]]
            buf = torch.empty(buffer_size(command["specs"]), dtype=torch.uint8, device=_comm_device(backend))
            dist.broadcast(buf, src=0)
            args, kwargs = unpack(command["data"], unflatten(buf, command["specs"], device))
            del buf
            x, timestep, context, options = args
            options["patches_replace"] = sequence_parallel.patches()
            with torch.inference_mode():
                patcher.model.diffusion_model._forward(x, timestep, context, transformer_options=options, **kwargs)
        elif command["op"] == "exit":
            return
        seq += 1
