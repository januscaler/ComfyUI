#!/usr/bin/env python3
"""Run a GPU server only while it is used, so that an idle one holds no VRAM.

Unloading every model still leaves a process's CUDA context on the card (about
600 MB on an RTX 5090), and nothing gives that back while the process lives. So
an idle server holds no VRAM only if its process does not exist. This gate owns
the server's port, starts the server on the first request that needs it,
forwards traffic to it, and stops it after --idle-seconds without use.

Requests that only check on the server (health probes, queue polls) are
configured as quiet: they never count as use, and while the server is down the
gate answers them itself, either with a fixed answer (--stub) or with the
server's last answer (--cache). A quiet request with neither starts the server.

Those answers are health, so they must not say "fine" for a server that cannot
run. After a start fails, the gate answers them with a 503 carrying the
failure instead, until a later start succeeds. It tries again on the first
quiet request after --start-backoff seconds (doubled after each failed try, up
to 10 minutes), so a probe that keeps asking finds out when the server is back.
That probe, and every probe while that start runs, is answered at once with a
503 that says "starting" ("loading": true): probes give up after 10 s, and a
model can take minutes to load. A start fails as soon as the server's
--ready-path answer says it did (a JSON "failed" with "loading" not true), not
after --boot-timeout: Maya's server stays up after its model load runs out of
GPU memory, and on 2026-11-19 that held every voice line for 20 minutes.
(--fail-health-seconds, VoxMin's first take on this, is an older name for
--start-backoff.) With --down-check, the gate also runs that command (a cheap
check that the GPU answers, say) before answering for a stopped server, at most
once every --down-check-seconds; a failed check is answered with a 503 too. The
503 also says "running": false, as the stubs do. On 2026-11-11
the rig's GPU hung in the driver: every start died at once, and the stub still
read healthy, so every render was sent to it and failed.

The server is told about one connection per request (the gate asks it to close
each one), so every request passes through the gate's accounting; WebSockets
are forwarded as they are. X-Forwarded-For is set to the real client, replacing
whatever the client sent.

Two ways to stay warm longer than --idle-seconds, both off unless asked for, so that a
first request after a quiet spell does not pay a cold start (Maya's takes 22-32 s):

* A lease (refused, like start=1, while --warm-while fails: only while nothing else wants the
  GPU). Whoever knows work is queued for this server (a voice-over job, a lane's
  dispatcher) asks POST /__idle-gate/keep-warm?seconds=N: the gate does not stop the
  server before N seconds are up (at most --lease-max-seconds, 0 turns the route off),
  and with start=1 it also starts a stopped server in the background, so the model is
  loaded by the time the work arrives. DELETE drops the lease. The route is never
  forwarded and never counts as use; with --bearer-env it needs the token.
* A wider window while nobody else wants the GPU. With --warm-idle-seconds above
  --idle-seconds and --warm-while COMMAND, a server idle for --idle-seconds is kept
  until --warm-idle-seconds as long as COMMAND exits 0 (docker/gpu_used_check.py: the
  card's used memory is below a ceiling, i.e. no render or other engine is on it). The
  moment it fails the server stops as it always did. Re-checked every
  --warm-check-seconds.

Handing the GPU back is synchronous: POST /__idle-gate/release?block=N&wait=M (what a caller
does before a render that needs the card) drops the lease, waits up to M seconds for requests
in flight to end, stops the server and answers only once it is down; for N seconds no lease,
start=1 or wide window applies. 409 when requests were still running after M seconds.

Standard library only, so it runs in any image with Python 3.9+. A copy lives
in each repository whose image runs it (ComfyUI docker/, VoxMin docker/); keep
them identical.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import http
import json
import math
import os
import shlex
import signal
import sys
import time

HEAD_LIMIT = 64 * 1024
HEAD_TIMEOUT = 30.0
CACHE_LIMIT = 1024 * 1024
PROBE_TIMEOUT = 10.0
START_FAILURE_HOLD = 10.0
START_BACKOFF_MAX = 600.0
# Below the 10 s that health probes (voxmin-backend, the compose healthcheck) wait.
DOWN_CHECK_TIMEOUT = 8.0
CHUNK = 64 * 1024
LEASE_ROUTE = "/__idle-gate/keep-warm"
RELEASE_ROUTE = "/__idle-gate/release"
# Hop-by-hop or client-asserted headers the gate replaces rather than passes on.
DROPPED_HEADERS = {"connection", "keep-alive", "proxy-connection", "x-forwarded-for", "x-forwarded-proto"}


def log(message: str) -> None:
    sys.stderr.write(f"idle-gate: {message}\n")
    sys.stderr.flush()


def parse_address(value: str) -> tuple[str, int]:
    host, sep, port = value.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"expected HOST:PORT, got {value!r}")
    return host.strip("[]"), int(port)


def parse_route(value: str) -> tuple[str, str]:
    parts = value.split()
    if len(parts) != 2 or not parts[1].startswith("/"):
        raise argparse.ArgumentTypeError(f"expected 'METHOD /path', got {value!r}")
    return parts[0].upper(), parts[1]


def parse_stub(value: str) -> tuple[tuple[str, str], tuple[int, bytes]]:
    parts = value.split(None, 3)
    if len(parts) < 3 or not parts[2].isdigit():
        raise argparse.ArgumentTypeError(f"expected 'METHOD /path STATUS [JSON]', got {value!r}")
    body = parts[3] if len(parts) == 4 else ""
    if body:
        try:
            json.loads(body)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"stub body for {value!r} is not JSON: {exc}") from exc
    return parse_route(" ".join(parts[:2])), (int(parts[2]), body.encode())


def parse_busy(value: str) -> tuple[tuple[str, str], list[str]]:
    parts = value.split(None, 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'METHOD /path KEY[,KEY...]', got {value!r}")
    return parse_route(" ".join(parts[:2])), [key for key in parts[2].split(",") if key]


def parse_head(raw: bytes) -> tuple[str, str, str, list[tuple[str, str]]]:
    lines = raw.decode("latin-1").split("\r\n")
    request_line = lines[0].split(" ")
    if len(request_line) != 3 or not request_line[2].startswith("HTTP/"):
        raise ValueError("bad request line")
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep or not name or name != name.strip():
            raise ValueError("bad header line")
        headers.append((name, value.strip()))
    return request_line[0], request_line[1], request_line[2], headers


def header(headers: list[tuple[str, str]], name: str) -> str:
    name = name.lower()
    return ", ".join(value for key, value in headers if key.lower() == name)


def request_body(headers: list[tuple[str, str]]) -> tuple[int | None, bool]:
    """(Content-Length, chunked) of a request: how the gate knows its body arrived whole."""
    chunked = "chunked" in header(headers, "transfer-encoding").lower()
    raw = header(headers, "content-length").strip()
    try:
        length = int(raw) if raw and not chunked else None
    except ValueError:
        length = None
    return (max(0, length) if length is not None else None), chunked


def is_upgrade(headers: list[tuple[str, str]]) -> bool:
    connection = {token.strip().lower() for token in header(headers, "connection").split(",")}
    return "upgrade" in connection and bool(header(headers, "upgrade"))


def response(status: int, body: bytes = b"", *, head_only: bool = False, extra: str = "") -> bytes:
    try:
        reason = http.HTTPStatus(status).phrase
    except ValueError:
        reason = "Status"
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"X-Idle-Gate: answered\r\n{extra}\r\n"
    )
    return head.encode("latin-1") + (b"" if head_only else body)


def error(status: int, message: str, *, head_only: bool = False, retry_after: int = 5) -> bytes:
    return response(status, json.dumps({"error": message}).encode(), head_only=head_only,
                    extra=f"Retry-After: {max(1, retry_after)}\r\n")


def unhealthy(message: str, *, head_only: bool = False, retry_after: int = 5) -> bytes:
    """The 503 a stubbed or cached route gets for a server that cannot run. Besides
    the error it says the server is not running, as the stubs do: VoxMin's Maya
    health reads that to tell "failed" from "loading"."""
    body = json.dumps({"status": "error", "running": False, "error": message}).encode()
    return response(503, body, head_only=head_only, extra=f"Retry-After: {max(1, retry_after)}\r\n")


def starting(*, head_only: bool = False) -> bytes:
    """The 503 a stubbed or cached route gets while a start is under way. No
    "running": false, so VoxMin's Maya health reads it as loading, not failed."""
    body = json.dumps({"status": "starting", "loading": True}).encode()
    return response(503, body, head_only=head_only, extra="Retry-After: 5\r\n")


def reported_failure(body: bytes) -> str | None:
    """Why a starting server says it cannot start, if its ready answer says so:
    a JSON object with "failed" set and "loading" not true. Maya's server keeps
    running after its model load fails (out of memory, say) and answers that way
    for good; waiting out --boot-timeout on it held every request for 20 minutes."""
    try:
        state = json.loads(body)
    except ValueError:
        return None
    if not isinstance(state, dict) or not state.get("failed") or state.get("loading") is True:
        return None
    failed = state["failed"]
    return (failed if isinstance(failed, str) else json.dumps(failed))[:300]


def dechunk(body: bytes) -> bytes:
    out = bytearray()
    while body:
        size_line, _, rest = body.partition(b"\r\n")
        size = int(size_line.split(b";")[0], 16)
        if size == 0:
            break
        out += rest[:size]
        body = rest[size + 2 :]
    return bytes(out)


class Gate:
    def __init__(self, args: argparse.Namespace) -> None:
        self.command: list[str] = args.command
        self.upstream_host, self.upstream_port = args.upstream
        self.idle_seconds: float = args.idle_seconds
        self.boot_timeout: float = args.boot_timeout
        self.stop_timeout: float = args.stop_timeout
        self.forward_idle: float = args.forward_idle_seconds
        self.ready_path: str = args.ready_path
        self.stubs: dict[tuple[str, str], tuple[int, bytes]] = dict(args.stub)
        self.cache_routes: set[tuple[str, str]] = set(args.cache)
        self.quiet: set[tuple[str, str]] = set(args.quiet) | set(self.stubs) | self.cache_routes
        self.busy = args.busy
        self.token = os.environ.get(args.bearer_env, "") if args.bearer_env else ""
        self.cache: dict[tuple[str, str], bytes] = {}
        self.proc: asyncio.subprocess.Process | None = None
        self.watcher: asyncio.Future | None = None
        # A start a health probe asked for after the backoff, run without the probe waiting on it.
        self.start_task: asyncio.Future | None = None
        self.up = False
        self.stopping = False
        self.lock = asyncio.Lock()
        self.active = 0
        self.last_used = time.monotonic()
        self.busy_probe_failing = False
        self.start_failure: tuple[float, str] | None = None
        # Starts that failed since the last one that worked, and when the gate tries again.
        self.start_backoff: float = args.start_backoff
        self.failed_starts = 0
        self.retry_at = 0.0
        self.down_check: list[str] = shlex.split(args.down_check) if args.down_check else []
        self.down_check_seconds: float = args.down_check_seconds
        self.down_check_result: tuple[float, str | None] | None = None
        self.down_check_task: asyncio.Future | None = None
        self.down_check_stuck: asyncio.subprocess.Process | None = None
        # Staying warm: a lease somebody took, and the wider window while the GPU is free.
        self.lease_max: float = args.lease_max_seconds
        self.lease_until = 0.0
        self.warm_idle: float = args.warm_idle_seconds
        self.warm_while: list[str] = shlex.split(args.warm_while) if args.warm_while else []
        self.warm_check_seconds: float = args.warm_check_seconds
        self.warm_result: tuple[float, bool] | None = None
        # After a release: no lease, start=1 or wide window until then.
        self.blocked_until = 0.0

    # -- the server process -------------------------------------------------

    async def ensure_up(self, reason: str) -> None:
        async with self.lock:
            if self.up:
                return
            # Requests queued behind a start that just failed fail with it, rather
            # than each paying for another start (no longer than its backoff, which
            # is when a health probe tries again).
            if self.start_failure and time.monotonic() < min(self.start_failure[0] + START_FAILURE_HOLD, self.retry_at):
                raise RuntimeError(self.start_failure[1])
            started = time.monotonic()
            log(f"starting the server for {reason}")
            # Its own process group, so stopping it also stops whatever it started.
            proc = await asyncio.create_subprocess_exec(*self.command, start_new_session=True)
            self.proc = proc
            try:
                await self._wait_ready(proc)
            except BaseException as exc:
                self.start_failure = (time.monotonic(), str(exc))
                self.failed_starts += 1
                backoff = min(START_BACKOFF_MAX, self.start_backoff * 2 ** (self.failed_starts - 1))
                self.retry_at = time.monotonic() + backoff
                log(f"the start failed ({self.failed_starts} in a row); health reads unhealthy, "
                    f"next try in {backoff:g}s at the earliest")
                await self._terminate(proc)
                self.proc = None
                raise
            if self.failed_starts:
                log(f"the server started after {self.failed_starts} failed starts; health reads healthy again")
            self.start_failure = None
            self.failed_starts = 0
            self.down_check_result = None
            self.up = True
            self.last_used = time.monotonic()
            self.watcher = asyncio.ensure_future(self._watch(proc))
            log(f"server ready in {time.monotonic() - started:.1f}s (pid {proc.pid})")

    def _start_under_way(self) -> bool:
        return self.lock.locked() or (self.start_task is not None and not self.start_task.done())

    def _start_in_background(self, reason: str) -> None:
        async def start() -> None:
            try:
                await self.ensure_up(reason)
            except Exception as exc:
                log(f"could not start the server: {exc}")

        self.start_task = asyncio.ensure_future(start())

    async def _wait_ready(self, proc: asyncio.subprocess.Process) -> None:
        deadline = time.monotonic() + self.boot_timeout
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                raise RuntimeError(f"the server exited with status {proc.returncode} while starting")
            try:
                # Some servers listen before they can serve and say so with a
                # 503; any answer below 500 (a 401 too) means they route.
                status, body = await self._get(self.ready_path)
                if 0 < status < 500:
                    return
            except (OSError, asyncio.TimeoutError, ValueError):
                pass
            else:
                failure = reported_failure(body)
                if failure:
                    raise RuntimeError(f"the server failed to start: {failure}")
            await asyncio.sleep(0.25)
        raise RuntimeError(f"the server was not ready after {self.boot_timeout:.0f}s")

    async def _watch(self, proc: asyncio.subprocess.Process) -> None:
        status = await proc.wait()
        if self.proc is proc and not self.stopping:
            log(f"the server exited on its own with status {status}; the next request starts it again")
            self.proc = None
            self.up = False
            self._signal_group(proc, signal.SIGKILL)

    @staticmethod
    def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is None:
            self._signal_group(proc, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), self.stop_timeout)
            except asyncio.TimeoutError:
                log(f"the server did not stop within {self.stop_timeout:.0f}s; killing it")
                self._signal_group(proc, signal.SIGKILL)
                await proc.wait()
        # Anything it left behind in its group would still hold the card.
        self._signal_group(proc, signal.SIGKILL)

    async def stop(self, why: str) -> None:
        async with self.lock:
            proc = self.proc
            if not self.up or proc is None or (self.active and why == "idle"):
                return
            log(f"stopping the server ({why})")
            self.stopping = True
            self.up = False
            try:
                await self._terminate(proc)
            finally:
                self.proc = None
                self.stopping = False
            log("server stopped")

    async def idle_loop(self) -> None:
        tick = max(0.1, min(5.0, self.idle_seconds / 4))
        if self.warm_idle > self.idle_seconds and self.warm_while:
            tick = min(tick, 1.0)       # a render that wants the card is noticed within a second
        while True:
            await asyncio.sleep(tick)
            if not self.up or self.active:
                continue
            if time.monotonic() < self.lease_until and await self._gpu_free():
                continue
            if time.monotonic() - self.last_used < self.idle_seconds:
                continue
            if time.monotonic() - self.last_used < await self._idle_limit():
                continue
            if await self._is_busy():
                # The idle window starts when the work ends, not when it was asked for.
                self.last_used = time.monotonic()
                continue
            await self.stop(f"{self.idle_seconds:g}s without use")

    async def _gpu_free(self) -> bool:
        """Whether nothing else wants the GPU (--warm-while passes, and no release is in force).
        True when no --warm-while is set: the lease then works as it always has."""
        if time.monotonic() < self.blocked_until:
            return False
        if not self.warm_while:
            return True
        now = time.monotonic()
        result = self.warm_result
        if result is None or now - result[0] >= self.warm_check_seconds:
            env = {**os.environ, "IDLE_GATE_SERVER_UP": "1" if self.up else "0"}
            problem = await self._spawn_check(self.warm_while, os.path.basename(self.warm_while[0]), env)
            if problem and (result is None or result[1]):
                log(f"not staying warm: {problem}")
            elif not problem and result is not None and not result[1]:
                log("staying warm again: the GPU is free")
            result = (now, problem is None)
            self.warm_result = result
        return result[1]

    async def _idle_limit(self) -> float:
        """Seconds without use after which the server stops: --idle-seconds, or the wider
        --warm-idle-seconds while the GPU is free (--warm-while passes)."""
        if self.warm_idle <= self.idle_seconds or not self.warm_while:
            return self.idle_seconds
        return self.warm_idle if await self._gpu_free() else self.idle_seconds

    def keep_warm(self, seconds: float) -> float:
        """Do not stop the server for ``seconds`` (at most --lease-max-seconds); the seconds
        granted. A lease only ever grows: a shorter ask never cuts a longer one short."""
        granted = max(0.0, min(seconds, self.lease_max))
        self.lease_until = max(self.lease_until, time.monotonic() + granted)
        return granted

    def drop_lease(self) -> None:
        self.lease_until = 0.0
        self.last_used = time.monotonic()

    async def _lease(self, method: str, target: str, headers, writer) -> None:
        """The gate's own route for staying warm (never forwarded, never use)."""
        if not self._authorized(headers):
            writer.write(error(401, "a bearer token is required"))
            return
        query = dict(part.partition("=")[::2] for part in target.partition("?")[2].split("&") if part)
        if method == "DELETE":
            self.drop_lease()
        elif method == "POST":
            try:
                seconds = float(query.get("seconds", "0"))
            except ValueError:
                writer.write(error(400, "seconds must be a number"))
                return
            if not await self._gpu_free():
                # A lease or a start would load or hold the engine beside a render: refused.
                writer.write(response(409, json.dumps({
                    "keep_warm_seconds": 0.0, "running": self.up,
                    "error": "the GPU is wanted by something else"}).encode()))
                await writer.drain()
                return
            self.keep_warm(seconds)
            if query.get("start") in ("1", "true") and not self.up and not self._start_under_way() \
                    and not self.failed_starts:
                self._start_in_background("a keep-warm request")
        else:
            writer.write(error(405, "POST or DELETE"))
            return
        left = max(0.0, self.lease_until - time.monotonic())
        writer.write(response(200, json.dumps({"keep_warm_seconds": round(left, 1),
                                               "running": self.up}).encode()))
        await writer.drain()

    async def _release(self, target: str, headers, writer) -> None:
        """POST /__idle-gate/release?block=N&wait=M: hand the GPU back before a render. Drops the
        lease, waits (up to M s) for requests in flight, stops the server, and answers once it is
        down; for N s nothing keeps it or starts it warm. 409 when requests outlasted M."""
        if not self._authorized(headers):
            writer.write(error(401, "a bearer token is required"))
            return
        query = dict(part.partition("=")[::2] for part in target.partition("?")[2].split("&") if part)
        try:
            block, wait = float(query.get("block", "300")), float(query.get("wait", "30"))
        except ValueError:
            writer.write(error(400, "block and wait must be numbers"))
            return
        self.blocked_until = max(self.blocked_until, time.monotonic() + max(0.0, block))
        self.lease_until = 0.0
        deadline = time.monotonic() + max(0.0, wait)
        while self.active and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if self.active:
            writer.write(response(409, json.dumps({"running": self.up, "error": "requests are still running"}).encode()))
            return
        await self.stop("released for a render")
        writer.write(response(200, json.dumps({"running": self.up, "blocked_seconds": round(max(0.0, self.blocked_until - time.monotonic()), 1)}).encode()))
        await writer.drain()

    async def _is_busy(self) -> bool:
        if not self.busy:
            return False
        (_, path), keys = self.busy
        try:
            status, body = await self._get(path)
            state = json.loads(body)
            if status != 200 or not isinstance(state, dict):
                raise ValueError(f"status {status}")
        except (OSError, asyncio.TimeoutError, ValueError) as exc:
            if not self.busy_probe_failing:
                log(f"could not read {path} ({exc}); keeping the server up until it answers")
                self.busy_probe_failing = True
            return True
        self.busy_probe_failing = False
        return any(state.get(key) for key in keys)

    async def _get(self, path: str) -> tuple[int, bytes]:
        """GET *path* on the server; (status, body). Status 0: not HTTP yet."""

        async def request() -> tuple[int, bytes]:
            reader, writer = await asyncio.open_connection(self.upstream_host, self.upstream_port)
            try:
                auth = f"Authorization: Bearer {self.token}\r\n" if self.token else ""
                writer.write(
                    f"GET {path} HTTP/1.1\r\nHost: {self.upstream_host}:{self.upstream_port}\r\n"
                    f"{auth}Connection: close\r\n\r\n".encode("latin-1")
                )
                await writer.drain()
                raw = b""
                while len(raw) < CACHE_LIMIT:
                    more = await reader.read(CACHE_LIMIT - len(raw))
                    if not more:
                        break
                    raw += more
            finally:
                writer.close()
            head, _, body = raw.partition(b"\r\n\r\n")
            status_line = head.split(b"\r\n", 1)[0].split(b" ")
            if not status_line[0].startswith(b"HTTP/") or len(status_line) < 2:
                return 0, b""
            if b"transfer-encoding: chunked" in head.lower():
                body = dechunk(body)
            return int(status_line[1]), body

        return await asyncio.wait_for(request(), PROBE_TIMEOUT)

    # -- clients ------------------------------------------------------------

    def _authorized(self, headers: list[tuple[str, str]]) -> bool:
        if not self.token:
            return True
        return hmac.compare_digest(header(headers, "authorization").encode(), f"Bearer {self.token}".encode())

    async def _answer_while_down(self, route: tuple[str, str], head_only: bool) -> bytes | None:
        """The gate's own answer to a quiet request while the server is down, or
        None to hand the request to the server (starting it)."""
        cached = self.cache.get(route) if not head_only else None
        if route not in self.stubs and cached is None:
            return None
        if self.failed_starts:
            wait = self.retry_at - time.monotonic()
            if wait > 0:
                reason = self.start_failure[1] if self.start_failure else "it failed to start"
                return unhealthy(f"the server could not be started: {reason}", head_only=head_only,
                                 retry_after=math.ceil(wait))
        problem = await self._run_down_check()
        if problem:
            return unhealthy(f"the server cannot run here: {problem}", head_only=head_only)
        if self.failed_starts:
            # Its backoff is over and the check passes: try a start, but do not hold
            # the probe on it (they wait 10 s; a model can take minutes to load).
            if not self._start_under_way():
                self._start_in_background(f"a {route[0]} {route[1]} after the backoff")
            return starting(head_only=head_only)
        if route in self.stubs:
            status, body = self.stubs[route]
            return response(status, body, head_only=head_only)
        return cached

    async def _run_down_check(self) -> str | None:
        """None when --down-check passes (or there is none), else why it failed.
        One run at a time; its result is reused for --down-check-seconds."""
        if not self.down_check:
            return None
        result = self.down_check_result
        if result and time.monotonic() - result[0] < self.down_check_seconds:
            return result[1]
        if self.down_check_task is None or self.down_check_task.done():
            self.down_check_task = asyncio.ensure_future(self._down_check_once())
        # Shielded: a client that goes does not cancel the check others wait on.
        return await asyncio.shield(self.down_check_task)

    async def _down_check_once(self) -> str | None:
        what = os.path.basename(self.down_check[0])
        stuck = self.down_check_stuck
        if stuck is not None and stuck.returncode is None:
            # A check hung in a wedged driver may not even die when killed: do not pile up more.
            problem = f"an earlier {what} (pid {stuck.pid}) has still not exited"
        else:
            self.down_check_stuck = None
            problem = await self._spawn_down_check(what)
        previous = self.down_check_result
        if problem and (previous is None or previous[1] is None):
            log(f"the down check failed: {problem}")
        elif not problem and previous is not None and previous[1]:
            log("the down check passes again")
        self.down_check_result = (time.monotonic(), problem)
        return problem

    async def _spawn_down_check(self, what: str) -> str | None:
        return await self._spawn_check(self.down_check, what)

    async def _spawn_check(self, command: list[str], what: str, env: dict | None = None) -> str | None:
        """Run ``command`` (a cheap check): None when it exits 0, else why not."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *command, env=env, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            return f"could not run {what}: {exc}"
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), DOWN_CHECK_TIMEOUT)
        except asyncio.TimeoutError:
            self._signal_group(proc, signal.SIGKILL)
            self.down_check_stuck = proc
            asyncio.ensure_future(proc.wait())  # reaped whenever it does exit
            return f"{what} did not finish within {DOWN_CHECK_TIMEOUT:g}s"
        if proc.returncode == 0:
            return None
        lines = [line.strip() for line in out.decode("utf-8", "replace").splitlines() if line.strip()]
        return lines[-1][:300] if lines else f"{what} exited with status {proc.returncode}"

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._handle(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT)
            method, target, version, headers = parse_head(raw[:-4])
        except asyncio.LimitOverrunError:
            writer.write(error(431, "request head too large"))
            return
        except ValueError:
            writer.write(error(400, "malformed request"))
            return
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            return
        path = target.split("?", 1)[0]
        head_only = method == "HEAD"
        route = ("GET" if head_only else method, path)
        quiet = route in self.quiet

        if path == RELEASE_ROUTE and method == "POST":
            await self._release(target, headers, writer)
            return
        if path == LEASE_ROUTE and self.lease_max > 0:
            await self._lease(method, target, headers, writer)
            return
        if quiet and not self.up and self._authorized(headers):
            answer = await self._answer_while_down(route, head_only)
            if answer is not None:
                writer.write(answer)
                await writer.drain()
                return

        if not quiet:
            self.active += 1
        try:
            if not self.up:
                try:
                    await self.ensure_up(f"{method} {path}")
                except Exception as exc:
                    log(f"could not start the server: {exc}")
                    writer.write(error(503, f"the server could not be started: {exc}"))
                    return
                if reader.at_eof() or writer.transport.is_closing():
                    return  # the client gave up while the server started; do not run its request
            peer = writer.get_extra_info("peername")
            await self._forward(reader, writer, method, target, version, headers, route, peer[0] if peer else "")
        finally:
            if not quiet:
                self.active -= 1
                self.last_used = time.monotonic()

    async def _forward(self, reader, writer, method, target, version, headers, route, client_ip) -> None:
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(self.upstream_host, self.upstream_port), PROBE_TIMEOUT)
        except (OSError, asyncio.TimeoutError) as exc:
            writer.write(error(502, f"the server is not reachable: {exc or 'timed out'}"))
            return
        upgrade = is_upgrade(headers)
        forwarded = [(k, v) for k, v in headers if k.lower() not in DROPPED_HEADERS]
        forwarded.append(("Connection", header(headers, "connection") if upgrade else "close"))
        if client_ip:
            forwarded.append(("X-Forwarded-For", client_ip))
        head = f"{method} {target} {version}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in forwarded) + "\r\n"
        up_writer.write(head.encode("latin-1"))
        capture = bytearray() if route in self.cache_routes and method == "GET" else None

        # When anything last moved, either way: what --forward-idle-seconds measures.
        moved = [time.monotonic()]
        to_client = asyncio.ensure_future(self._pipe(up_reader, writer, capture, moved))
        if upgrade:
            to_server = asyncio.ensure_future(self._pipe(reader, up_writer, None, moved))
        else:
            to_server = asyncio.ensure_future(self._send_body(reader, up_writer, request_body(headers), moved))
        try:
            if upgrade:
                await asyncio.wait({to_server, to_client}, return_when=asyncio.FIRST_COMPLETED)
            else:
                await self._await_answer(to_server, to_client, moved, writer, f"{method} {route[1]}")
        finally:
            for task in (to_server, to_client):
                task.cancel()
            # Closing it is what tells the server a client that went is gone: a
            # half-sent upload is otherwise waited on forever, and the request
            # stays counted as use, so the server is never stopped.
            up_writer.close()
        if capture is not None and capture.startswith((b"HTTP/1.1 200", b"HTTP/1.0 200")) and len(capture) < CACHE_LIMIT:
            self.cache[route] = bytes(capture)

    async def _await_answer(self, to_server, to_client, moved, writer, what: str) -> None:
        """Wait for the server's answer. A client that stops sending after its
        whole request (or half-closes) still gets the whole answer; one that
        goes before its request arrived whole does not hold the server; and a
        request where nothing moves for --forward-idle-seconds is dropped."""
        tick = 1.0 if not self.forward_idle else max(0.05, min(1.0, self.forward_idle / 4))
        while True:
            pending = {to_client} if to_server.done() else {to_server, to_client}
            await asyncio.wait(pending, timeout=tick, return_when=asyncio.FIRST_COMPLETED)
            if to_client.done():
                return
            if to_server.done() and not to_server.result():
                log(f"the client went before its {what} request arrived whole; dropped it")
                return
            if self.forward_idle and time.monotonic() - moved[0] >= self.forward_idle:
                log(f"nothing moved on {what} for {self.forward_idle:g}s; dropped it")
                try:
                    writer.write(error(504, f"nothing moved for {self.forward_idle:g}s"))
                except (ConnectionError, OSError, RuntimeError):
                    pass
                return

    @staticmethod
    async def _send_body(src: asyncio.StreamReader, dst: asyncio.StreamWriter,
                         body: tuple[int | None, bool], moved: list[float]) -> bool:
        """Client -> server, for a plain request. True once the request's body has
        arrived whole (whatever the client does after); False when the client
        went before it did."""
        length, chunked = body
        whole = not chunked and not length
        got, tail = 0, b""
        try:
            while True:
                data = await src.read(CHUNK)
                if not data:
                    return whole
                moved[0] = time.monotonic()
                dst.write(data)
                await dst.drain()
                if whole:
                    continue
                if chunked:
                    tail = (tail + data)[-7:]
                    whole = tail.endswith(b"\r\n0\r\n\r\n") or tail == b"0\r\n\r\n"
                else:
                    got += len(data)
                    whole = got >= length
        except (ConnectionError, OSError):
            return whole

    @staticmethod
    async def _pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter, capture: bytearray | None = None,
                    moved: list[float] | None = None) -> None:
        try:
            while True:
                data = await src.read(CHUNK)
                if not data:
                    return
                if moved is not None:
                    moved[0] = time.monotonic()
                dst.write(data)
                await dst.drain()
                if capture is not None and len(capture) < CACHE_LIMIT:
                    capture += data
        except (ConnectionError, OSError):
            return


async def serve(args: argparse.Namespace) -> None:
    gate = Gate(args)
    host, port = args.listen
    server = await asyncio.start_server(gate.handle, host, port, limit=HEAD_LIMIT, reuse_address=True)
    if args.idle_seconds:
        log(f"listening on {host}:{port}; the server starts on first use and stops after {args.idle_seconds:g}s without use")
    else:
        log(f"listening on {host}:{port}; the server starts on first use and then stays up (--idle-seconds 0)")
    done = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, done.set)
    idle = asyncio.ensure_future(gate.idle_loop() if args.idle_seconds else done.wait())
    await done.wait()
    idle.cancel()
    server.close()
    await gate.stop("the gate is shutting down")


def parse_args(argv: list[str]) -> argparse.Namespace:
    if "--" not in argv:
        sys.exit("idle-gate: usage: idle_gate.py [options] -- SERVER COMMAND...")
    split = argv.index("--")
    parser = argparse.ArgumentParser(prog="idle_gate.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--listen", type=parse_address, required=True, help="HOST:PORT clients connect to")
    parser.add_argument("--upstream", type=parse_address, required=True, help="HOST:PORT the server command listens on")
    parser.add_argument("--idle-seconds", type=float, default=120.0,
                        help="stop the server after this long without use; 0: never stop it once started")
    parser.add_argument("--boot-timeout", type=float, default=600.0, help="give up on a start after this long")
    parser.add_argument("--stop-timeout", type=float, default=30.0, help="SIGTERM grace before SIGKILL")
    parser.add_argument("--forward-idle-seconds", type=float, default=900.0,
                        help="drop a forwarded request when nothing moves either way for this long; 0: never")
    parser.add_argument("--ready-path", default="/", help="the server is up once a GET here answers below 500")
    parser.add_argument("--quiet", type=parse_route, action="append", default=[], metavar="'METHOD /path'",
                        help="a request that never counts as use")
    parser.add_argument("--stub", type=parse_stub, action="append", default=[], metavar="'METHOD /path STATUS [JSON]'",
                        help="a quiet request, answered with this while the server is down")
    parser.add_argument("--cache", type=parse_route, action="append", default=[], metavar="'METHOD /path'",
                        help="a quiet request, answered with the server's last 200 while it is down")
    parser.add_argument("--busy", type=parse_busy, metavar="'METHOD /path KEY[,KEY...]'",
                        help="before stopping, GET this JSON; the server is busy while any KEY is non-empty")
    parser.add_argument("--bearer-env", metavar="NAME",
                        help="env var with the server's bearer token: sent on the gate's own requests, and "
                             "required before the gate answers for the server")
    parser.add_argument("--start-backoff", type=float, metavar="SECONDS",
                        help="after a failed start, quiet requests answer 503 for this long, then one tries a "
                             "start again; doubled after each failed try, up to 600s (default 30)")
    parser.add_argument("--fail-health-seconds", type=float, default=0.0, metavar="SECONDS",
                        help="the older name of --start-backoff, which wins when both are given; 0: the default")
    parser.add_argument("--lease-max-seconds", type=float, default=600.0, metavar="SECONDS",
                        help="the longest POST /__idle-gate/keep-warm may keep the server up past its idle "
                             "window; 0: no such route")
    parser.add_argument("--warm-idle-seconds", type=float, default=0.0, metavar="SECONDS",
                        help="stop after this long without use instead of --idle-seconds while --warm-while "
                             "passes; 0 or below --idle-seconds: off")
    parser.add_argument("--warm-while", metavar="'COMMAND [ARGS]'",
                        help="exit 0 while the wider --warm-idle-seconds window may apply (nothing else "
                             "wants the GPU)")
    parser.add_argument("--warm-check-seconds", type=float, default=2.0,
                        help="reuse a --warm-while result for this long")
    parser.add_argument("--down-check", metavar="'COMMAND [ARGS]'",
                        help="run before answering for a stopped server; a non-zero exit answers 503 with its "
                             "last line of output (a GPU check, say)")
    parser.add_argument("--down-check-seconds", type=float, default=30.0,
                        help="reuse a --down-check result for this long")
    args = parser.parse_args(argv[:split])
    args.command = argv[split + 1 :]
    if not args.command:
        parser.error("no server command after --")
    if args.idle_seconds < 0:
        parser.error("--idle-seconds cannot be negative")
    if args.forward_idle_seconds < 0:
        parser.error("--forward-idle-seconds cannot be negative")
    if (args.start_backoff or 0) < 0 or args.fail_health_seconds < 0 or args.down_check_seconds < 0:
        parser.error("--start-backoff, --fail-health-seconds and --down-check-seconds cannot be negative")
    if args.start_backoff is None:
        args.start_backoff = args.fail_health_seconds or 30.0
    if args.lease_max_seconds < 0 or args.warm_idle_seconds < 0 or args.warm_check_seconds < 0:
        parser.error("--lease-max-seconds, --warm-idle-seconds and --warm-check-seconds cannot be negative")
    if args.warm_while is not None and not shlex.split(args.warm_while):
        parser.error("--warm-while needs a command")
    if args.down_check is not None and not shlex.split(args.down_check):
        parser.error("--down-check needs a command")
    return args


def main(argv: list[str] | None = None) -> None:
    asyncio.run(serve(parse_args(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    main()
