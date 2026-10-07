"""Tests for docker/idle_gate.py: a GPU server runs only while it is used.

Each test starts the real gate in front of a small stand-in server (written to
a temp dir) and talks to it over TCP, so the process handling (start, stop,
process groups, signals) is exercised as it runs in the image.
"""

import http.client
import json
import os
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
    if os.environ.get("FAKE_FAIL"):
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

    def test_a_malformed_request_is_refused(self):
        self.start_gate()
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as s:
            s.sendall(b"nonsense\r\n\r\n")
            self.assertTrue(s.recv(1024).startswith(b"HTTP/1.1 400"))
        self.assertEqual(self.pids(), [])


if __name__ == "__main__":
    unittest.main()
