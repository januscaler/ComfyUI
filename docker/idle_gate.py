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

The server is told about one connection per request (the gate asks it to close
each one), so every request passes through the gate's accounting; WebSockets
are forwarded as they are. X-Forwarded-For is set to the real client, replacing
whatever the client sent.

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
import os
import signal
import sys
import time

HEAD_LIMIT = 64 * 1024
HEAD_TIMEOUT = 30.0
CACHE_LIMIT = 1024 * 1024
PROBE_TIMEOUT = 10.0
START_FAILURE_HOLD = 10.0
CHUNK = 64 * 1024
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


def error(status: int, message: str) -> bytes:
    return response(status, json.dumps({"error": message}).encode(), extra="Retry-After: 5\r\n")


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
        self.ready_path: str = args.ready_path
        self.stubs: dict[tuple[str, str], tuple[int, bytes]] = dict(args.stub)
        self.cache_routes: set[tuple[str, str]] = set(args.cache)
        self.quiet: set[tuple[str, str]] = set(args.quiet) | set(self.stubs) | self.cache_routes
        self.busy = args.busy
        self.token = os.environ.get(args.bearer_env, "") if args.bearer_env else ""
        self.cache: dict[tuple[str, str], bytes] = {}
        self.proc: asyncio.subprocess.Process | None = None
        self.watcher: asyncio.Future | None = None
        self.up = False
        self.stopping = False
        self.lock = asyncio.Lock()
        self.active = 0
        self.last_used = time.monotonic()
        self.busy_probe_failing = False
        self.start_failure: tuple[float, str] | None = None

    # -- the server process -------------------------------------------------

    async def ensure_up(self, reason: str) -> None:
        async with self.lock:
            if self.up:
                return
            # Requests queued behind a start that just failed fail with it, rather
            # than each paying for another start.
            if self.start_failure and time.monotonic() - self.start_failure[0] < START_FAILURE_HOLD:
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
                await self._terminate(proc)
                self.proc = None
                raise
            self.start_failure = None
            self.up = True
            self.last_used = time.monotonic()
            self.watcher = asyncio.ensure_future(self._watch(proc))
            log(f"server ready in {time.monotonic() - started:.1f}s (pid {proc.pid})")

    async def _wait_ready(self, proc: asyncio.subprocess.Process) -> None:
        deadline = time.monotonic() + self.boot_timeout
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                raise RuntimeError(f"the server exited with status {proc.returncode} while starting")
            try:
                # Some servers listen before they can serve and say so with a
                # 503; any answer below 500 (a 401 too) means they route.
                status, _ = await self._get(self.ready_path)
                if 0 < status < 500:
                    return
            except (OSError, asyncio.TimeoutError, ValueError):
                pass
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
        while True:
            await asyncio.sleep(tick)
            if not self.up or self.active:
                continue
            if time.monotonic() - self.last_used < self.idle_seconds:
                continue
            if await self._is_busy():
                # The idle window starts when the work ends, not when it was asked for.
                self.last_used = time.monotonic()
                continue
            await self.stop(f"{self.idle_seconds:g}s without use")

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

    def _answer_while_down(self, route: tuple[str, str], head_only: bool) -> bytes | None:
        if route in self.stubs:
            status, body = self.stubs[route]
            return response(status, body, head_only=head_only)
        cached = self.cache.get(route)
        if cached is not None and not head_only:
            return cached
        return None

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

        if quiet and not self.up and self._authorized(headers):
            answer = self._answer_while_down(route, head_only)
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
            up_reader, up_writer = await asyncio.open_connection(self.upstream_host, self.upstream_port)
        except OSError as exc:
            writer.write(error(502, f"the server is not reachable: {exc}"))
            return
        upgrade = is_upgrade(headers)
        forwarded = [(k, v) for k, v in headers if k.lower() not in DROPPED_HEADERS]
        forwarded.append(("Connection", header(headers, "connection") if upgrade else "close"))
        if client_ip:
            forwarded.append(("X-Forwarded-For", client_ip))
        head = f"{method} {target} {version}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in forwarded) + "\r\n"
        up_writer.write(head.encode("latin-1"))
        capture = bytearray() if route in self.cache_routes and method == "GET" else None

        to_server = asyncio.ensure_future(self._pipe(reader, up_writer))
        to_client = asyncio.ensure_future(self._pipe(up_reader, writer, capture))
        try:
            if upgrade:
                await asyncio.wait({to_server, to_client}, return_when=asyncio.FIRST_COMPLETED)
            else:
                # The request is answered when the server closes; a client that stops
                # sending early (or half-closes) still gets the whole answer.
                await to_client
        finally:
            for task in (to_server, to_client):
                task.cancel()
            up_writer.close()
        if capture is not None and capture.startswith((b"HTTP/1.1 200", b"HTTP/1.0 200")) and len(capture) < CACHE_LIMIT:
            self.cache[route] = bytes(capture)

    @staticmethod
    async def _pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter, capture: bytearray | None = None) -> None:
        try:
            while True:
                data = await src.read(CHUNK)
                if not data:
                    return
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
    args = parser.parse_args(argv[:split])
    args.command = argv[split + 1 :]
    if not args.command:
        parser.error("no server command after --")
    if args.idle_seconds < 0:
        parser.error("--idle-seconds cannot be negative")
    return args


def main(argv: list[str] | None = None) -> None:
    asyncio.run(serve(parse_args(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    main()
