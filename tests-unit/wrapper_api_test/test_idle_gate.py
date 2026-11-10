"""Tests for docker/idle_gate.py: a GPU server runs only while it is used.

Each test starts the real gate in front of a small stand-in server (written to
a temp dir) and talks to it over TCP, so the process handling (start, stop,
process groups, signals) is exercised as it runs in the image.
"""

import http.client
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GATE = os.path.join(ROOT, "docker", "idle_gate.py")

# The stand-in: records each start, answers a few routes, echoes on /ws.
FAKE_SERVER = textwrap.dedent(
    '''
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    port, starts, busy = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    with open(starts + ".attempts", "a") as f:
        f.write("1\\n")
    if os.environ.get("FAKE_FAIL") or os.path.exists(busy + ".fail"):
        # As ComfyUI did on a GPU the driver had lost: "No CUDA GPUs are available".
        sys.exit(3)
    started = time.monotonic()
    warmup = float(os.environ.get("FAKE_WARMUP", "0"))
    with open(starts, "a") as f:
        f.write(f"{os.getpid()}\\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            warming = time.monotonic() - started < warmup
            if warming:
                # Listening, but not serving yet: every route says so.
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif path == "/ws":
                self.send_response(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.end_headers()
                self.wfile.flush()
                while True:
                    data = self.connection.recv(1024)
                    if not data:
                        break
                    self.connection.sendall(data)
                self.close_connection = True
            elif path == "/queue":
                self.answer({"queue_running": [1] if os.path.exists(busy) else [], "queue_pending": []})
            elif path == "/slow":
                time.sleep(float(self.path.split("s=")[1]))
                self.answer({"pid": os.getpid()})
            else:
                self.answer({"pid": os.getpid(), "path": path, "headers": dict(self.headers)})

        def do_POST(self):
            # An upload, as the wrapper's /generate reads one: the whole body, then
            # the answer. What arrived is recorded beside the start log.
            length = int(self.headers.get("Content-Length", "0"))
            got = len(self.rfile.read(length))
            with open(starts + ".posts", "a") as f:
                f.write(f"{'whole' if got == length else 'short'} {got}/{length}\\n")
            if got == length:
                self.answer({"pid": os.getpid(), "got": got})

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    '''
)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for(predicate, timeout=10.0, step=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


class IdleGateTest(unittest.TestCase):
    def start_gate(self, *options, idle="1", env=None):
        self.tmp = tempfile.mkdtemp()
        self.starts = os.path.join(self.tmp, "starts")
        self.busy = os.path.join(self.tmp, "busy")
        fake = os.path.join(self.tmp, "fake_server.py")
        with open(fake, "w") as f:
            f.write(FAKE_SERVER)
        self.port, upstream = free_port(), free_port()
        self.gate = subprocess.Popen(
            [
                sys.executable, GATE,
                "--listen", f"127.0.0.1:{self.port}",
                "--upstream", f"127.0.0.1:{upstream}",
                "--idle-seconds", idle,
                "--ready-path", "/ready",
                *options,
                "--", sys.executable, fake, str(upstream), self.starts, self.busy,
            ],
            env={**os.environ, **(env or {})},
        )
        self.addCleanup(self.stop_gate)
        self.assertTrue(wait_for(self.listening), "the gate never listened")

    def stop_gate(self):
        if self.gate.poll() is None:
            self.gate.send_signal(signal.SIGTERM)
            self.gate.wait(15)
        for pid in self.pids():
            if alive(pid):
                os.kill(pid, signal.SIGKILL)

    def listening(self):
        try:
            socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
            return True
        except OSError:
            return False

    def pids(self):
        if not os.path.exists(self.starts):
            return []
        with open(self.starts) as f:
            return [int(line) for line in f if line.strip()]

    def get(self, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("GET", path, headers=headers or {})
            res = conn.getresponse()
            return res.status, dict(res.getheaders()), res.read()
        finally:
            conn.close()

    def posts(self):
        path = self.starts + ".posts"
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [line.split()[0] for line in f if line.strip()]

    def post_head(self, length):
        return (f"POST /api/wrapper/qwenimage21/generate HTTP/1.1\r\nHost: x\r\n"
                f"Content-Type: application/octet-stream\r\nContent-Length: {length}\r\n\r\n").encode()

    def attempts(self):
        path = self.starts + ".attempts"
        if not os.path.exists(path):
            return 0
        with open(path) as f:
            return sum(1 for line in f if line.strip())

    def failing(self, on):
        """Make every later start of the stand-in fail (or work again)."""
        flag = self.busy + ".fail"
        if on:
            open(flag, "w").close()
        elif os.path.exists(flag):
            os.remove(flag)

    def down_check(self, flag, runs):
        """A --down-check that fails while *flag* exists and counts its runs in *runs*."""
        code = (f"import os, sys; open({runs!r}, 'a').write('1\\n'); "
                f"os.path.exists({flag!r}) and (print('GPU check: cuInit failed: CUDA error 100'), sys.exit(1))")
        return shlex.join([sys.executable, "-c", code])

    def server_stopped(self):
        pids = self.pids()
        return bool(pids) and not alive(pids[-1])

    def test_quiet_stub_answers_without_starting_the_server(self):
        self.start_gate("--stub", 'GET /health 200 {"status": "healthy"}')
        status, headers, body = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "healthy"})
        self.assertEqual(headers.get("X-Idle-Gate"), "answered")
        self.assertEqual(self.pids(), [], "a health probe must not start the server")

    def test_a_request_starts_the_server_and_reaches_it(self):
        self.start_gate()
        status, _, body = self.get("/anything?x=1", headers={"X-Forwarded-For": "6.6.6.6"})
        self.assertEqual(status, 200)
        answer = json.loads(body)
        self.assertEqual(answer["path"], "/anything")
        self.assertEqual(answer["pid"], self.pids()[0])
        self.assertEqual(answer["headers"]["X-Forwarded-For"], "127.0.0.1", "the client cannot claim another address")
        self.assertEqual(answer["headers"]["Connection"], "close")
        status, _, body = self.get("/again")
        self.assertEqual(json.loads(body)["pid"], self.pids()[0])
        self.assertEqual(len(self.pids()), 1, "an up server is reused")

    def test_an_idle_server_is_stopped_and_started_again_on_demand(self):
        self.start_gate("--stub", "GET /health 200 {}")
        self.get("/work")
        self.assertTrue(wait_for(self.server_stopped), "the idle server was not stopped")
        self.assertEqual(self.get("/health")[2], b"{}", "the stub answers again once it is down")
        self.assertEqual(len(self.pids()), 1)
        status, _, body = self.get("/work")
        self.assertEqual(status, 200)
        self.assertEqual(len(self.pids()), 2)
        self.assertEqual(json.loads(body)["pid"], self.pids()[1])

    def test_a_server_that_listens_before_it_serves_is_waited_for(self):
        # OmniVoice answers 503 on every route for its first few seconds.
        self.start_gate(env={"FAKE_WARMUP": "1.5"})
        status, _, body = self.get("/work")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["path"], "/work")

    def test_quiet_requests_do_not_keep_the_server_up(self):
        self.start_gate("--quiet", "GET /health")
        self.get("/work")
        deadline = time.monotonic() + 10
        while not self.server_stopped() and time.monotonic() < deadline:
            self.get("/health")
            time.sleep(0.2)
        self.assertTrue(self.server_stopped(), "polling a quiet route kept the server up")

    def test_a_request_longer_than_the_idle_window_is_not_cut(self):
        self.start_gate()
        status, _, body = self.get("/slow?s=2.5")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["pid"], self.pids()[0])

    def test_a_busy_server_is_kept_up_until_its_work_ends(self):
        self.start_gate("--busy", "GET /queue queue_running,queue_pending")
        open(self.busy, "w").close()
        self.get("/work")
        time.sleep(2.5)
        self.assertTrue(alive(self.pids()[0]), "a server with queued work was stopped")
        os.remove(self.busy)
        self.assertTrue(wait_for(self.server_stopped), "the server was not stopped once its work ended")

    def test_a_cached_route_is_answered_from_the_last_answer_while_down(self):
        self.start_gate("--cache", "GET /workflows")
        status, _, first = self.get("/workflows")
        self.assertEqual(status, 200)
        self.assertEqual(len(self.pids()), 1, "with nothing cached yet, the server answers")
        self.assertTrue(wait_for(self.server_stopped))
        status, _, again = self.get("/workflows")
        self.assertEqual(status, 200)
        self.assertEqual(again, first)
        self.assertEqual(len(self.pids()), 1, "the cached answer must not start the server")

    def test_the_gate_answers_for_the_server_only_with_its_token(self):
        self.start_gate("--stub", "GET /health 200 {}", "--bearer-env", "GATE_TOKEN", env={"GATE_TOKEN": "s3cret"})
        status, headers, _ = self.get("/health", headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(headers.get("X-Idle-Gate"), "answered")
        self.assertEqual(self.pids(), [])
        status, headers, body = self.get("/health", headers={"Authorization": "Bearer wrong"})
        self.assertIsNone(headers.get("X-Idle-Gate"), "without the token the server itself decides")
        self.assertEqual(json.loads(body)["pid"], self.pids()[0])

    def test_a_websocket_is_piped_both_ways(self):
        self.start_gate()
        with socket.create_connection(("127.0.0.1", self.port), timeout=15) as s:
            s.sendall(b"GET /ws HTTP/1.1\r\nHost: x\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\n")
            head = b""
            while b"\r\n\r\n" not in head:
                head += s.recv(1024)
            self.assertTrue(head.startswith(b"HTTP/1.0 101"), head)
            s.sendall(b"ping")
            self.assertEqual(s.recv(1024), b"ping")

    def test_a_server_that_cannot_start_answers_503(self):
        self.start_gate(env={"FAKE_FAIL": "1"})
        status, headers, body = self.get("/work")
        self.assertEqual(status, 503)
        self.assertIn("exited with status 3", json.loads(body)["error"])
        self.assertEqual(headers.get("Retry-After"), "5")

    def test_after_a_failed_start_health_reads_unhealthy_until_a_start_works(self):
        # 2026-11-11: the rig's GPU hung in the driver, every start died at once, and
        # the stubbed and cached health still said the worker was fine, so voxmin-backend
        # kept sending every render to it.
        self.start_gate("--stub", "GET /health 200 {}", "--cache", "GET /api/wrapper/workflows",
                        "--start-backoff", "1.5")
        self.assertEqual(self.get("/api/wrapper/workflows")[0], 200)
        self.assertTrue(wait_for(self.server_stopped))
        self.assertEqual(self.get("/health")[0], 200, "a stopped server that starts fine is healthy")
        self.failing(True)
        status, _, body = self.get("/work")
        self.assertEqual(status, 503)
        self.assertIn("exited with status 3", json.loads(body)["error"])
        attempts = self.attempts()
        for path in ("/health", "/api/wrapper/workflows"):
            status, headers, body = self.get(path)
            self.assertEqual(status, 503, f"{path} must not read healthy after a failed start")
            self.assertEqual(headers.get("X-Idle-Gate"), "answered")
            self.assertIn("could not be started: the server exited with status 3", json.loads(body)["error"])
            self.assertIn(headers.get("Retry-After"), ("1", "2"))
        self.assertEqual(self.attempts(), attempts, "health probes inside the backoff do not start it")
        self.failing(False)
        time.sleep(1.6)
        status, headers, body = self.get("/health")
        self.assertEqual(status, 200, "after the backoff a probe tries a start, which works")
        self.assertIsNone(headers.get("X-Idle-Gate"), "the server itself answered")
        self.assertEqual(json.loads(body)["path"], "/health")
        self.assertEqual(self.attempts(), attempts + 1)
        self.assertTrue(wait_for(self.server_stopped), "a probe's start is not use: it stops again")
        status, headers, _ = self.get("/health")
        self.assertEqual((status, headers.get("X-Idle-Gate")), (200, "answered"), "healthy and stubbed again")

    def test_failed_tries_back_off(self):
        self.start_gate("--stub", "GET /health 200 {}", "--start-backoff", "0.6")
        self.failing(True)
        self.assertEqual(self.get("/work")[0], 503)
        self.assertEqual(self.attempts(), 1)
        time.sleep(0.7)
        self.assertEqual(self.get("/health")[0], 503, "the try after the backoff failed too")
        self.assertEqual(self.attempts(), 2)
        time.sleep(0.7)
        self.assertEqual(self.get("/health")[0], 503)
        self.assertEqual(self.attempts(), 2, "the second backoff is twice as long")
        time.sleep(0.7)
        self.assertEqual(self.get("/health")[0], 503)
        self.assertEqual(self.attempts(), 3)

    def test_a_failing_down_check_reads_unhealthy_without_starting_the_server(self):
        flag, runs = os.path.join(tempfile.mkdtemp(), "gpu-lost"), os.path.join(tempfile.mkdtemp(), "runs")
        self.start_gate("--stub", "GET /health 200 {}", "--down-check", self.down_check(flag, runs),
                        "--down-check-seconds", "0")
        self.assertEqual(self.get("/health")[0], 200)
        open(flag, "w").close()
        status, headers, body = self.get("/health")
        self.assertEqual(status, 503)
        self.assertEqual(headers.get("X-Idle-Gate"), "answered")
        self.assertEqual(json.loads(body)["error"], "the server cannot run here: GPU check: cuInit failed: CUDA error 100")
        self.assertEqual(self.attempts(), 0, "the check alone decides; nothing was started")
        os.remove(flag)
        self.assertEqual(self.get("/health")[0], 200, "healthy again once the check passes")

    def test_the_down_check_runs_once_per_window_and_not_while_the_server_is_up(self):
        flag, runs = os.path.join(tempfile.mkdtemp(), "gpu-lost"), os.path.join(tempfile.mkdtemp(), "runs")
        self.start_gate("--stub", "GET /health 200 {}",
                        "--down-check", self.down_check(flag, runs), "--down-check-seconds", "60", idle="600")
        for _ in range(3):
            self.assertEqual(self.get("/health")[0], 200)
        with open(runs) as f:
            self.assertEqual(len(f.read().split()), 1, "one run serves the whole window")
        open(flag, "w").close()
        self.assertEqual(self.get("/health")[0], 200, "inside the window the last result stands")
        self.get("/work")
        os.remove(runs)
        self.assertEqual(self.get("/health")[0], 200, "an up server answers for itself")
        self.assertFalse(os.path.exists(runs), "no check while the server is up")

    def test_idle_seconds_zero_starts_on_demand_and_never_stops(self):
        self.start_gate(idle="0")
        self.assertEqual(self.pids(), [])
        self.get("/work")
        time.sleep(1.5)
        self.assertTrue(alive(self.pids()[0]))

    def test_stopping_the_gate_stops_the_server(self):
        self.start_gate(idle="600")
        self.get("/work")
        pid = self.pids()[0]
        self.gate.send_signal(signal.SIGTERM)
        self.assertEqual(self.gate.wait(15), 0)
        self.assertFalse(alive(pid))

    def test_a_client_that_goes_mid_upload_does_not_hold_the_server(self):
        # 2026-10-24: the backend gave up on a fill whose upload was still arriving
        # (60 s); the gate kept the half-sent upload open to ComfyUI forever, so
        # five orphans piled up and the idle server was never stopped again.
        self.start_gate()
        with socket.create_connection(("127.0.0.1", self.port), timeout=15) as s:
            s.sendall(self.post_head(100_000) + b"x" * 1000)
            self.assertTrue(wait_for(lambda: bool(self.pids())), "the upload never reached the server")
            time.sleep(0.3)
        self.assertTrue(wait_for(lambda: self.posts() == ["short"]), "the server still waits on the upload")
        self.assertTrue(wait_for(self.server_stopped), "the request still counts as use; the server never stops")

    def test_a_whole_upload_is_answered_after_the_client_half_closes(self):
        self.start_gate()
        with socket.create_connection(("127.0.0.1", self.port), timeout=15) as s:
            s.sendall(self.post_head(5000) + b"x" * 5000)
            s.shutdown(socket.SHUT_WR)
            answer = b""
            while True:
                data = s.recv(65536)
                if not data:
                    break
                answer += data
        self.assertTrue(answer.startswith(b"HTTP/1.0 200"), answer[:80])
        self.assertEqual(json.loads(answer.split(b"\r\n\r\n", 1)[1])["got"], 5000)
        self.assertEqual(self.posts(), ["whole"])

    def test_a_request_where_nothing_moves_is_dropped(self):
        self.start_gate("--forward-idle-seconds", "1", idle="600")
        started = time.monotonic()
        status, _, body = self.get("/slow?s=6")
        self.assertEqual(status, 504)
        self.assertIn("nothing moved", json.loads(body)["error"])
        self.assertLess(time.monotonic() - started, 4.5)

    def test_a_long_answer_with_traffic_is_not_dropped(self):
        self.start_gate("--forward-idle-seconds", "1.5")
        self.assertEqual(self.get("/slow?s=0.5")[0], 200)

    def test_a_malformed_request_is_refused(self):
        self.start_gate()
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as s:
            s.sendall(b"nonsense\r\n\r\n")
            self.assertTrue(s.recv(1024).startswith(b"HTTP/1.1 400"))
        self.assertEqual(self.pids(), [])


if __name__ == "__main__":
    unittest.main()
