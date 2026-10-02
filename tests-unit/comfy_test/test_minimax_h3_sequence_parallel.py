"""MiniMax H3 sequence parallelism (comfy/ldm/minimax/sequence_parallel.py).

The unit tests cover the pieces; the two-process tests run rank 0 (sp_harness.py)
and a follower (docker/sp_follower.py) on the CPU over gloo, with a tiny random H3
loaded from a file through comfy.sd.load_diffusion_model, as a render loads it.
"""

import json
import os
import socket
import subprocess
import sys
import time
import uuid

import pytest
import torch

import comfy.model_management
import comfy.patcher_extension
from comfy.ldm.minimax import sequence_parallel as sp
from comfy.ldm.minimax.model import PackedLayout

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HARNESS = os.path.join(os.path.dirname(__file__), "sp_harness.py")


def test_split_sizes_cover_the_sequence_evenly():
    assert sp.split_sizes(10, 4) == [3, 3, 2, 2]
    assert sp.split_sizes(8, 4) == [2, 2, 2, 2]
    assert sum(sp.split_sizes(76362, 8)) == 76362


def test_local_segments_clip_rebase_and_slice_per_token_rows():
    rows = torch.arange(10, 20)
    segments = [(0, 4, 1), (4, 14, rows), (14, 20, 2)]
    assert [(a, b, r if not torch.is_tensor(r) else r.tolist()) for a, b, r in sp.local_segments(segments, 3, 9)] == [
        (0, 1, 1), (1, 6, [10, 11, 12, 13, 14])]
    assert sp.local_segments(segments, 14, 20) == [(0, 6, 2)]


def test_pack_moves_every_tensor_out_of_band_and_back():
    layout = PackedLayout(4, 2, 4, 6, 3)
    obj = (
        [torch.randn(1, 2, 3), torch.tensor([7.5], dtype=torch.float64)],
        {"minimax_payload": {"layout": layout, "audio_scale": 1.0, "tags": torch.tensor([1, 0, 1]),
                             "mask": torch.tensor([True, False, True]), "half": torch.ones(3, dtype=torch.bfloat16),
                             "scalar": torch.tensor(3)}},
    )
    data, tensors, specs = sp.pack(obj)
    assert len(tensors) >= 6 + 3  # layout carries its own tensors
    assert isinstance(data, bytes)
    buf = sp.flatten(tensors, specs, torch.device("cpu"))
    assert buf.numel() == sp.buffer_size(specs)
    back = sp.unpack(data, sp.unflatten(buf, specs, torch.device("cpu")))
    torch.testing.assert_close(back[0][0], obj[0][0])
    assert back[0][1].dtype == torch.float64 and float(back[0][1]) == 7.5
    payload = back[1]["minimax_payload"]
    assert payload["mask"].tolist() == [True, False, True] and payload["scalar"].dim() == 0
    assert payload["half"].dtype == torch.bfloat16
    assert payload["layout"].signature == layout.signature
    assert payload["layout"].segments == layout.segments
    torch.testing.assert_close(payload["layout"].position_ids, layout.position_ids)


class _Base:
    current_weight_patches_uuid = None


def _info(clean):
    info = sp._Attached.__new__(sp._Attached)
    base = _Base()
    info.base_model = lambda: base
    info.clean_patches_uuid = clean
    info.disabled = None
    return info, base


def _options(**extra):
    options = {"wrappers": {comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL: {sp.WRAPPER_KEY: [sp._wrapper]}}}
    options.update(extra)
    return options


def test_only_forwards_the_followers_can_reproduce_are_split():
    clean = uuid.uuid4()
    info, base = _info(clean)
    assert sp.ineligible(info, _options()) is None
    base.current_weight_patches_uuid = clean
    assert sp.ineligible(info, _options(patches={}, patches_replace={})) is None, "empty patch dicts are fine"
    base.current_weight_patches_uuid = uuid.uuid4()
    assert "weight patches" in sp.ineligible(info, _options())
    base.current_weight_patches_uuid = clean
    assert "block patches" in sp.ineligible(info, _options(patches_replace={"dit": {("double_block", 0): object()}}))
    assert "attention patches" in sp.ineligible(info, _options(patches={"attn1_patch": [object()]}))
    assert "attention override" in sp.ineligible(info, _options(optimized_attention_override=object()))
    two = _options()
    two["wrappers"][comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL]["cache"] = [object()]
    assert "another diffusion-model wrapper" in sp.ineligible(info, two)
    info.disabled = "the followers could not load the model"
    assert sp.ineligible(info, _options()) == info.disabled


def test_an_interrupt_is_held_until_the_deferred_block_ends():
    comfy.model_management.interrupt_current_processing(True)
    try:
        with comfy.model_management.defer_processing_interrupt():
            comfy.model_management.throw_exception_if_processing_interrupted()  # held
            with comfy.model_management.defer_processing_interrupt():
                comfy.model_management.throw_exception_if_processing_interrupted()
            comfy.model_management.throw_exception_if_processing_interrupted()  # still held after the inner block
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            comfy.model_management.throw_exception_if_processing_interrupted()
    finally:
        comfy.model_management.interrupt_current_processing(False)


def test_attach_is_a_no_op_without_followers(monkeypatch):
    monkeypatch.delenv("COMFY_SP_WORLD", raising=False)
    sp.attach(object())  # never touches the patcher
    assert sp.get_group() is None


# --- two processes on the CPU ---------------------------------------------------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("sp")
    out = subprocess.run([sys.executable, HARNESS, "make", str(directory)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def _run(checkpoint, followers, join_seconds="60"):
    port = _free_port()
    env = {**os.environ, "COMFY_SP_WORLD": "2", "COMFY_SP_BACKEND": "gloo", "COMFY_SP_PORT": str(port),
           "COMFY_SP_JOIN_SECONDS": join_seconds, "COMFY_SP_TIMEOUT_SECONDS": "120", "TORCH_CPP_LOG_LEVEL": "ERROR"}
    procs = [subprocess.Popen([sys.executable, os.path.join(ROOT, "docker", "sp_follower.py"), "--cpu"],
                              env={**env, "COMFY_SP_RANK": "1"}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for _ in range(followers)]
    try:
        rank0 = subprocess.run([sys.executable, HARNESS, "run", checkpoint], env=env, capture_output=True, text=True, timeout=600)
        follower_logs = []
        for proc in procs:
            out, _ = proc.communicate(timeout=60)
            follower_logs.append((proc.returncode, out))
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
    assert rank0.returncode == 0, rank0.stderr[-3000:]
    checks = {line["check"]: line for line in map(json.loads, (l for l in rank0.stdout.splitlines() if l.startswith("{")))}
    return checks, follower_logs, rank0.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="gloo process groups")
def test_two_processes_split_a_forward_with_the_dense_output(tiny_checkpoint):
    checks, followers, _ = _run(tiny_checkpoint, followers=1)
    assert checks["attached"] == {"check": "attached", "active": True, "wrappers": 1}
    # 1 load + 3 forwards reached the follower; the outputs match the dense model's.
    assert checks["forward"]["posted"] == 4
    assert checks["forward"]["diff"] <= 1e-5
    # Each split forward left the allocation compiler: on CUDA it fails to compile the split blocks.
    assert checks["forward"]["malloc_graph_ends"] == 3
    assert checks["masked"]["diff"] <= 1e-5, "per-token mod rows (a denoise mask) split with their rows"
    # A LoRA'd model renders on rank 0 alone, and nothing is posted.
    assert checks["patched"] == {"check": "patched", "diff": 0.0, "posted": 0}
    # A cancel during a split forward lets it finish, then raises at the next check.
    assert checks["interrupt"]["diff"] <= 1e-5 and checks["interrupt"]["raised_after"] is True
    code, log = followers[0]
    assert code == 0, log[-2000:]
    assert "follower 1/2 joined (gloo" in log and "rank 0 is gone, exiting" in log


@pytest.mark.skipif(sys.platform == "win32", reason="gloo process groups")
def test_without_followers_renders_stay_on_one_gpu(tiny_checkpoint):
    checks, _, stderr = _run(tiny_checkpoint, followers=0, join_seconds="2")
    assert checks["attached"] == {"check": "attached", "active": False, "wrappers": 0}
    assert checks["forward"]["diff"] == 0.0 and checks["forward"]["posted"] == 0
    assert checks["forward"]["malloc_graph_ends"] == 0
    assert "did not join" in stderr


# --- docker/sp_probe.py: the group's GPUs tried before ComfyUI starts -----------

def _probe_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("sp_probe", os.path.join(ROOT, "docker", "sp_probe.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(sys.platform == "win32", reason="gloo process groups")
def test_the_probe_passes_when_every_rank_exchanges_tensors():
    env = {**os.environ, "COMFY_SP_BACKEND": "gloo", "TORCH_CPP_LOG_LEVEL": "ERROR", "COMFY_SP_PROBE_SECONDS": "120"}
    for gpus in (["0", "1"], ["0", "1", "2", "3"]):
        out = subprocess.run([sys.executable, os.path.join(ROOT, "docker", "sp_probe.py"), *gpus], env=env,
                             capture_output=True, text=True, timeout=180)
        assert out.returncode == 0, out.stderr[-3000:]
        assert f"GPUs {','.join(gpus)} exchanged tensors (broadcast, all-to-all, all-gather)" in out.stderr
        assert "deprecated" not in out.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="signals")
def test_a_rank_that_hangs_fails_the_probe_at_its_deadline_and_is_killed():
    probe = _probe_module()
    launched = []

    def launch(command, env):
        launched.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]))
        return launched[-1]

    started = time.monotonic()
    problem = probe.probe(["3", "6"], 1, launch=launch)
    assert problem == "rank 0 (GPU 3), rank 1 (GPU 6) did not finish within 1s"
    assert time.monotonic() - started < 15
    assert all(proc.poll() is not None for proc in launched), "no rank outlives the probe"


@pytest.mark.skipif(sys.platform == "win32", reason="signals")
def test_a_rank_that_fails_ends_the_probe_without_waiting_for_the_deadline():
    probe = _probe_module()
    launched = []

    def launch(command, env):
        rank = int(command[command.index("--rank") + 1])
        assert env["CUDA_VISIBLE_DEVICES"] == ["4", "5"][rank], "each rank sees only its own GPU"
        code = "import sys; sys.exit(3)" if rank == 1 else "import time; time.sleep(60)"
        launched.append(subprocess.Popen([sys.executable, "-c", code]))
        return launched[-1]

    started = time.monotonic()
    problem = probe.probe(["4", "5"], 60, launch=launch)
    assert problem == "rank 1 (GPU 5) exited with status 3"
    assert time.monotonic() - started < 15
    assert launched[0].poll() is not None, "rank 0, left waiting for rank 1, is killed"


def test_values_that_arrive_wrong_fail_the_exchange(monkeypatch):
    probe = _probe_module()
    import torch.distributed as dist

    monkeypatch.setattr(dist, "broadcast", lambda tensor, src: None)  # nothing arrives at rank 1
    with pytest.raises(RuntimeError, match="broadcast"):
        probe.exchange(1, 2, torch.device("cpu"))
    # Rank 0 keeps its own broadcast; an all-to-all that hands back what was sent is wrong.
    monkeypatch.setattr(dist, "all_to_all_single", lambda recv, send: recv.copy_(send))
    with pytest.raises(RuntimeError, match="all-to-all"):
        probe.exchange(0, 2, torch.device("cpu"))
    monkeypatch.setattr(dist, "all_to_all_single", lambda recv, send: recv.copy_(torch.tensor([0, 2], dtype=torch.int32).repeat_interleave(recv.numel() // 2)))
    monkeypatch.setattr(dist, "all_gather_into_tensor", lambda out, part: out.zero_())
    with pytest.raises(RuntimeError, match="all-gather"):
        probe.exchange(0, 2, torch.device("cpu"))
