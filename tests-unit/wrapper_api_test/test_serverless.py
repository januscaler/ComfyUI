"""docker/serverless/jobs.py: the job a RunPod or vast.ai serverless worker runs.

run_job drives a fake wrapper over real HTTP: the sync reply, the 504 + poll
path with its /image redirect, failures, the deadline's cancel, uploads and
file inputs. The platform SDKs are not needed.
"""

import base64
import email.parser
import http.server
import importlib.util
import json
import os
import sys
import threading
import time
import unittest
import urllib.parse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JOB_ID = "0b8d2f5e-6a3c-4f1e-9d7a-2c4b6e8f0a1d"
VIDEO = b"\x00\x00\x00\x18ftypmp42 fake video bytes"


def _load():
    spec = importlib.util.spec_from_file_location("serverless_jobs", os.path.join(ROOT, "docker", "serverless", "jobs.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jobs = _load()


class FakeWrapper(http.server.BaseHTTPRequestHandler):
    """The wrapper routes run_job uses; `state` scripts the answers and records the calls."""

    state: dict = {}

    def log_message(self, *args):
        pass

    def _send(self, status, body=b"", headers=None):
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, value):
        self._send(status, json.dumps(value).encode(), {"Content-Type": "application/json"})

    def _authorized(self):
        self.state["calls"].append((self.command, self.path))
        if self.headers.get("Authorization") != "Bearer tok":
            self._json(401, {"error": {"message": "missing or invalid bearer token"}})
            return False
        return True

    def do_POST(self):
        if not self._authorized():
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path.endswith("/cancel"):
            self.state["cancelled"] = True
            return self._json(200, {})
        # requests sends a text-only form urlencoded and one with files as multipart.
        fields, files = {}, []
        if self.headers["Content-Type"].startswith("multipart/form-data"):
            message = email.parser.BytesParser().parsebytes(
                b"MIME-Version: 1.0\r\nContent-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
            for part in message.get_payload():
                name = part.get_param("name", header="content-disposition")
                if part.get_filename():
                    files.append((name, part.get_filename(), part.get_payload(decode=True)))
                else:
                    fields[name] = part.get_payload(decode=True).decode()
        else:
            fields = dict(urllib.parse.parse_qsl(body.decode()))
        self.state["submits"].append({"path": self.path, "fields": fields, "files": files})
        time.sleep(self.state.get("submit_delay", 0))
        status, value = self.state["submit"]
        if status == 200:
            return self._send(200, VIDEO, {"Content-Type": "video/mp4", "X-Wrapper-Job-Id": JOB_ID})
        self._json(status, value)

    def do_GET(self):
        if not self._authorized():
            return
        if self.path == f"/api/wrapper/jobs/{JOB_ID}":
            statuses = self.state["statuses"]
            record = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            return self._json(200, record)
        if self.path == f"/api/wrapper/jobs/{JOB_ID}/image":
            return self._send(302, headers={"Location": "/view?filename=clip.mp4&subfolder=&type=output"})
        if self.path.startswith("/view"):
            return self._send(200, VIDEO, {"Content-Type": "video/mp4"})
        if self.path == "/api/wrapper/workflows":
            return self._json(200, {"workflows": [{"name": "minimaxh3"}]})
        self._json(404, {"error": {"message": "not found"}})

    def do_DELETE(self):
        if self._authorized():
            self.state["deleted"].append(self.path)
            self._json(200, {})


COMPLETED = {"id": JOB_ID, "status": "completed", "images": [{"filename": "clip.mp4"}],
             "provenance": {"seed": 7, "models": ["h3.safetensors"]}}


class TestRunJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeWrapper)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeWrapper.state = {"calls": [], "submits": [], "deleted": [], "cancelled": False,
                             "submit": (200, None), "statuses": [COMPLETED]}
        self.wrapper = jobs.Wrapper("tok", base_url=self.base_url)
        self.now = [0.0]
        self.updates = []

    def run_job(self, job, **kwargs):
        return jobs.run_job(job, self.wrapper, progress=self.updates.append, clock=lambda: self.now[0],
                            sleep=lambda seconds: self.now.__setitem__(0, self.now[0] + seconds), **kwargs)

    def test_sync_reply_is_returned_inline_and_the_job_deleted(self):
        result = self.run_job({"prompt": "A red fox", "fields": {"seed": 7, "duration": 2.5}})
        submit = FakeWrapper.state["submits"][0]
        self.assertEqual(submit["path"], "/api/wrapper/minimaxh3/text/generate")
        self.assertEqual(submit["fields"], {"raw_prompt": "A red fox", "seed": "7", "duration": "2.5",
                                            "free_vram": "false", "timeout": str(jobs.SUBMIT_WAIT)})
        self.assertEqual(base64.b64decode(result["base64"]), VIDEO)
        self.assertEqual((result["job_id"], result["content_type"], result["filename"], result["bytes"]),
                         (JOB_ID, "video/mp4", "clip.mp4", len(VIDEO)))
        self.assertEqual(result["provenance"], COMPLETED["provenance"])
        self.assertEqual(FakeWrapper.state["deleted"], [f"/api/wrapper/jobs/{JOB_ID}"])

    def test_slow_render_is_polled_with_progress_then_read_through_the_redirect(self):
        FakeWrapper.state["submit"] = (504, {"error": {"type": "timeout"}, "job_id": JOB_ID})
        FakeWrapper.state["statuses"] = [
            {"status": "pending", "progress": {"percent": 0, "queue_position": 0}},
            {"status": "in_progress", "progress": {"percent": 40, "step": 8, "node_class": "SamplerCustomAdvanced"}},
            {"status": "in_progress", "progress": {"percent": 40, "step": 8, "node_class": "SamplerCustomAdvanced"}},
            COMPLETED,
        ]
        result = self.run_job({"prompt": "A red fox"})
        self.assertEqual(base64.b64decode(result["base64"]), VIDEO)
        self.assertEqual(self.updates, [
            {"status": "pending", "percent": 0, "step": None, "node": None},
            {"status": "in_progress", "percent": 40, "step": 8, "node": "SamplerCustomAdvanced"},
        ])
        self.assertIn(("GET", "/view?filename=clip.mp4&subfolder=&type=output"), FakeWrapper.state["calls"])

    def test_rewrite_sends_prompt_and_a_given_workflow_has_no_default_task(self):
        self.run_job({"workflow": "flux2klein9b-txt2img", "prompt": "A red fox", "rewrite": True,
                      "fields": {"free_vram": True}})
        submit = FakeWrapper.state["submits"][0]
        self.assertEqual(submit["path"], "/api/wrapper/flux2klein9b-txt2img/generate")
        self.assertEqual(submit["fields"]["prompt"], "A red fox")
        self.assertNotIn("raw_prompt", submit["fields"])
        self.assertEqual(submit["fields"]["free_vram"], "true")

    def test_refused_render_reports_the_wrapper_error(self):
        FakeWrapper.state["submit"] = (400, {"error": {"message": "Unknown workflow", "details": "Available workflows: minimaxh3"}})
        with self.assertRaisesRegex(jobs.JobError, r"HTTP 400\): Unknown workflow — Available workflows: minimaxh3"):
            self.run_job({"workflow": "nope", "prompt": "x"})
        self.assertEqual(FakeWrapper.state["deleted"], [])

    def test_failed_render_reports_the_exception_and_cleans_up(self):
        FakeWrapper.state["submit"] = (504, {"job_id": JOB_ID})
        FakeWrapper.state["statuses"] = [{"status": "failed", "execution_error": {"exception_message": "CUDA out of memory"}}]
        with self.assertRaisesRegex(jobs.JobError, "the render failed: CUDA out of memory"):
            self.run_job({"prompt": "x"})
        self.assertEqual(FakeWrapper.state["deleted"], [f"/api/wrapper/jobs/{JOB_ID}"])

    def test_render_past_its_timeout_is_cancelled(self):
        FakeWrapper.state["submit"] = (504, {"job_id": JOB_ID})
        FakeWrapper.state["statuses"] = [{"status": "in_progress", "progress": {"percent": 10}}]
        with self.assertRaisesRegex(jobs.JobError, "did not finish in time"):
            self.run_job({"prompt": "x", "timeout": 30})
        self.assertTrue(FakeWrapper.state["cancelled"])
        self.assertIn(("POST", f"/api/jobs/{JOB_ID}/cancel"), FakeWrapper.state["calls"])

    def test_submit_waits_as_long_as_the_job_may_for_model_downloads(self):
        # The wrapper downloads missing checkpoints before it answers the submit.
        FakeWrapper.state["submit_delay"] = 1.5
        self.run_job({"prompt": "x", "timeout": 5})
        with self.assertRaisesRegex(jobs.JobError, "not queued within 1s"):
            self.run_job({"prompt": "x", "timeout": 1})

    def test_upload_url_gets_the_file_instead_of_inline(self):
        uploads = []

        class Uploaded:
            status_code = 200

        def put(url, data, headers, timeout):
            uploads.append((url, data, headers["Content-Type"]))
            return Uploaded()

        result = self.run_job({"prompt": "x", "upload_url": "https://bucket.example/clip.mp4?sig=1"}, put=put)
        self.assertEqual(uploads, [("https://bucket.example/clip.mp4?sig=1", VIDEO, "video/mp4")])
        self.assertTrue(result["uploaded"])
        self.assertNotIn("base64", result)

    def test_output_over_the_inline_limit_asks_for_an_upload_url(self):
        with self.assertRaisesRegex(jobs.JobError, "upload_url"):
            self.run_job({"prompt": "x"}, max_inline_bytes=len(VIDEO) - 1)

    def test_files_from_base64_and_url_become_multipart_uploads(self):
        class Fetched:
            status_code = 200
            content = b"url image"

        first = base64.encodebytes(b"first frame" * 10).decode()  # wrapped, as GNU base64 prints it
        self.run_job({"task": "image", "prompt": "x", "files": [
            {"field": "image", "base64": first, "filename": "first.png"},
            {"field": "last_frame", "url": "https://cdn.example/last.png?x=1"},
        ]}, fetch=lambda url, timeout: Fetched())
        submit = FakeWrapper.state["submits"][0]
        self.assertEqual(submit["path"], "/api/wrapper/minimaxh3/image/generate")
        self.assertEqual(submit["files"], [("image", "first.png", b"first frame" * 10), ("last_frame", "last.png", b"url image")])

    def test_bad_inputs_fail_before_submitting(self):
        for job, message in [
            ({"prompt": "  "}, "non-empty"),
            ({"prompt": "x", "fields": [1]}, "'fields' must be an object"),
            ({"prompt": "x", "files": [{"field": "image", "base64": "aGk="}]}, "needs a 'filename'"),
            ({"prompt": "x", "files": [{"field": "image", "base64": "not base64!", "filename": "a.png"}]}, "not valid base64"),
            ({"prompt": "x", "files": [{"field": "image"}]}, "'url' or 'base64'"),
            ({"prompt": "x", "timeout": "soon"}, "'timeout' must be"),
        ]:
            with self.subTest(job=job), self.assertRaisesRegex(jobs.JobError, message):
                self.run_job(job)
        self.assertEqual(FakeWrapper.state["submits"], [])

    def test_async_submit_then_state_reports_progress_then_the_result(self):
        FakeWrapper.state["submit"] = (504, {"job_id": JOB_ID})
        FakeWrapper.state["statuses"] = [
            {"status": "in_progress", "progress": {"percent": 40, "step": {"value": 8, "max": 20}, "node_class": "SamplerCustomAdvanced"}},
            {**COMPLETED, "execution_start_time": 1_000, "execution_end_time": 43_500},
        ]
        self.assertEqual(jobs.submit_job({"prompt": "A red fox"}, self.wrapper), JOB_ID)
        self.assertEqual(jobs.job_state(JOB_ID, self.wrapper), {
            "status": "in_progress", "progress": {"percent": 40, "step": {"value": 8, "max": 20}, "node": "SamplerCustomAdvanced"}})
        self.assertEqual(FakeWrapper.state["deleted"], [])
        state = jobs.job_state(JOB_ID, self.wrapper)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(base64.b64decode(state["output"]["base64"]), VIDEO)
        self.assertEqual((state["output"]["seconds"], state["output"]["provenance"]), (42.5, COMPLETED["provenance"]))
        self.assertEqual(FakeWrapper.state["deleted"], [f"/api/wrapper/jobs/{JOB_ID}"])

    def test_async_fast_render_is_read_through_its_job(self):
        # A render finished within the submit wait still reports through job_state (its file stays on the worker).
        self.assertEqual(jobs.submit_job({"prompt": "x"}, self.wrapper), JOB_ID)
        self.assertEqual(FakeWrapper.state["deleted"], [])
        self.assertEqual(jobs.job_state(JOB_ID, self.wrapper)["status"], "completed")

    def test_async_failure_and_cancel(self):
        FakeWrapper.state["statuses"] = [{"status": "failed", "execution_error": {"exception_message": "CUDA out of memory"}}]
        self.assertEqual(jobs.job_state(JOB_ID, self.wrapper), {"status": "failed", "error": "the render failed: CUDA out of memory"})
        jobs.cancel_job(JOB_ID, self.wrapper)
        self.assertTrue(FakeWrapper.state["cancelled"])

    def test_async_failed_upload_keeps_the_job_for_a_retry(self):
        class Refused:
            status_code = 403

        with self.assertRaisesRegex(jobs.JobError, "upload_url answered HTTP 403"):
            jobs.job_state(JOB_ID, self.wrapper, "https://bucket.example/out.mp4", put=lambda *args, **kwargs: Refused())
        self.assertEqual(FakeWrapper.state["deleted"], [])

    def test_ready_needs_the_token(self):
        self.assertTrue(self.wrapper.ready())
        self.assertFalse(jobs.Wrapper("wrong", base_url=self.base_url).ready())
        self.assertFalse(jobs.Wrapper("tok", base_url="http://127.0.0.1:9").ready())


if __name__ == "__main__":
    unittest.main()
