"""One render per serverless request, shared by the RunPod handler
(runpod_handler.py) and the vast.ai PyWorker (vast_worker.py).

The worker starts ComfyUI + wrapper through the image's entrypoint on
127.0.0.1, with a bearer token only it knows, and forwards each job to the
wrapper API, so a serverless render is the same render a pod makes:

    POST /api/wrapper/{workflow}[/{task}]/generate   (multipart; answers 504 {job_id} after SUBMIT_WAIT s)
    GET  /api/wrapper/jobs/{id}                      (polled for status and progress)
    GET  /api/wrapper/jobs/{id}/image                (the output file)
    DELETE /api/wrapper/jobs/{id}                    (the worker's copy, once sent)

A job (only the prompt is needed):

    {
      "workflow": "minimaxh3",          # any wrapper workflow; with "task" omitted,
      "task": "text",                   #   minimaxh3 + text is the default
      "prompt": "A red fox ...",        # sent as raw_prompt (no LLM rewrite), unless
      "rewrite": true,                  #   this is set: then as prompt (needs MIMO_API_KEY)
      "fields": {"width": 864, "height": 480, "duration": 5, "steps": 20, "seed": 7},
      "files": [{"field": "image", "url": "https://..."},
                {"field": "image", "base64": "...", "filename": "first.png"}],
      "upload_url": "https://...",      # a presigned PUT: the output goes there, not inline
      "timeout": 1800                   # seconds the render may take before it is cancelled
    }

`fields` are the wrapper's form fields as they are. free_vram defaults to false
here: a serverless worker renders for one endpoint, so the models stay loaded
for its next job.

The result: {"job_id", "content_type", "filename", "bytes", "seconds",
"provenance"} plus "base64" (the file) or "uploaded": true.
"""

from __future__ import annotations

import base64
import os
import secrets
import subprocess
import sys
import time
from typing import Callable, Optional

import requests

COMFY_ROOT = "/opt/ComfyUI"
PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
# The first boot of a worker may copy the model weights before ComfyUI starts.
BOOT_TIMEOUT = int(os.environ.get("SERVERLESS_BOOT_TIMEOUT", "1800"))
SUBMIT_WAIT = 10
POLL_SECONDS = 2
DEFAULT_TIMEOUT = 1800


class JobError(Exception):
    """A job the worker could not complete; its message goes back to the caller."""


def log(message: str) -> None:
    sys.stderr.write(f"serverless: {message}\n")
    sys.stderr.flush()


def form_fields(job: dict) -> dict:
    fields = job.get("fields") or {}
    if not isinstance(fields, dict):
        raise JobError("'fields' must be an object")
    fields = {"free_vram": False, **fields}
    prompt = job.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise JobError("'prompt' must be a non-empty string")
        fields["prompt" if job.get("rewrite") else "raw_prompt"] = prompt
    fields["timeout"] = SUBMIT_WAIT
    return {key: _form_value(value) for key, value in fields.items() if value is not None}


def _form_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def form_files(job: dict, fetch: Callable[..., requests.Response] = requests.get) -> list:
    """`files` as multipart parts, each from a URL or inline base64."""
    files = job.get("files") or []
    if not isinstance(files, list):
        raise JobError("'files' must be a list")
    parts = []
    for index, item in enumerate(files):
        if not isinstance(item, dict) or not item.get("field"):
            raise JobError(f"files[{index}] needs a 'field' (e.g. 'image')")
        if item.get("base64"):
            if not item.get("filename"):
                raise JobError(f"files[{index}] needs a 'filename' with its base64 (the extension picks the file type)")
            try:
                # Whitespace is dropped: GNU base64 wraps its output every 76 columns.
                body = base64.b64decode("".join(str(item["base64"]).split()), validate=True)
            except ValueError as exc:
                raise JobError(f"files[{index}].base64 is not valid base64") from exc
            name = item["filename"]
        elif item.get("url"):
            try:
                response = fetch(item["url"], timeout=120)
            except requests.RequestException as exc:
                raise JobError(f"files[{index}].url could not be fetched: {exc}") from exc
            if response.status_code != 200:
                raise JobError(f"files[{index}].url answered HTTP {response.status_code}")
            body = response.content
            name = item.get("filename") or os.path.basename(item["url"].split("?")[0])
        else:
            raise JobError(f"files[{index}] needs a 'url' or 'base64'")
        parts.append((item["field"], (name, body, item.get("content_type") or "application/octet-stream")))
    return parts


def error_text(response: requests.Response) -> str:
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        return response.text[:500] or f"HTTP {response.status_code}"
    if isinstance(error, dict):
        return " — ".join(str(part) for part in (error.get("message"), error.get("details")) if part)
    return str(error)[:500]


class Wrapper:
    """The wrapper API of this worker's ComfyUI."""

    def __init__(self, token: str, base_url: str = f"http://127.0.0.1:{PORT}"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def ready(self) -> bool:
        try:
            return self.session.get(self.base_url + "/api/wrapper/workflows", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def submit(self, workflow: str, task: Optional[str], fields: dict, files: list, timeout: float) -> requests.Response:
        # The wrapper downloads a workflow's missing checkpoints before it queues
        # the job, so the reply may take as long as the whole job may.
        path = "/api/wrapper/" + requests.utils.quote(str(workflow), safe="")
        if task:
            path += "/" + requests.utils.quote(str(task), safe="")
        return self.session.post(self.base_url + path + "/generate", data=fields, files=files or None, timeout=timeout)

    def job(self, job_id: str) -> dict:
        response = self.session.get(f"{self.base_url}/api/wrapper/jobs/{job_id}", timeout=30)
        if response.status_code != 200:
            raise JobError(f"the worker lost job {job_id} (HTTP {response.status_code})")
        return response.json()

    def output(self, job_id: str) -> requests.Response:
        response = self.session.get(f"{self.base_url}/api/wrapper/jobs/{job_id}/image", timeout=300)
        if response.status_code != 200:
            raise JobError(f"could not read the output (HTTP {response.status_code}: {error_text(response)})")
        return response

    def cancel(self, job_id: str) -> None:
        self.session.post(f"{self.base_url}/api/jobs/{job_id}/cancel", timeout=30)

    def delete(self, job_id: str) -> None:
        # A job that is still running answers 409 and is left for ComfyUI's own history cap.
        self.session.delete(f"{self.base_url}/api/wrapper/jobs/{job_id}", timeout=30)


def wait_for_job(wrapper: Wrapper, job_id: str, deadline: float, progress: Callable[[dict], None],
                 clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
    last = None
    while True:
        record = wrapper.job(job_id)
        status = record.get("status")
        if status == "completed":
            return
        if status == "failed":
            error = record.get("execution_error") or {}
            raise JobError(f"the render failed: {error.get('exception_message') or 'ComfyUI reported no message'}")
        if status == "cancelled":
            raise JobError("the render was cancelled on the worker")
        if clock() > deadline:
            wrapper.cancel(job_id)
            raise JobError("the render did not finish in time and was cancelled")
        info = record.get("progress") or {}
        update = {"status": status, "percent": info.get("percent"), "step": info.get("step"), "node": info.get("node_class")}
        if update != last:
            progress(update)
            last = update
        sleep(POLL_SECONDS)


def _queue(job: dict, wrapper: Wrapper, fetch: Callable[..., requests.Response]) -> tuple[requests.Response, str, float]:
    """Send one job to the wrapper: its reply (200 with the file, or 504 while it renders), its job id and timeout."""
    if not isinstance(job, dict):
        raise JobError("the job input must be an object")
    if "workflow" in job:
        workflow, task = job["workflow"], job.get("task")
    else:
        workflow, task = "minimaxh3", job.get("task", "text")
    try:
        timeout = float(job.get("timeout") or DEFAULT_TIMEOUT)
    except (TypeError, ValueError) as exc:
        raise JobError("'timeout' must be a number of seconds") from exc
    fields, files = form_fields(job), form_files(job, fetch)
    try:
        response = wrapper.submit(workflow, task, fields, files, timeout)
    except requests.Timeout as exc:
        raise JobError(f"the render was not queued within {timeout:.0f}s (its checkpoints were still downloading?)") from exc
    if response.status_code == 504:
        return response, response.json()["job_id"], timeout
    if response.status_code == 200:
        return response, response.headers["X-Wrapper-Job-Id"], timeout
    raise JobError(f"the render was refused (HTTP {response.status_code}): {error_text(response)}")


def _deliver(job_id: str, response: requests.Response, record: dict, seconds: Optional[float], upload_url: Optional[str],
             max_inline_bytes: Optional[int], put: Callable[..., requests.Response]) -> dict:
    """The result of a finished job: its file inline (base64) or PUT to upload_url."""
    body = response.content
    result = {
        "job_id": job_id,
        "content_type": response.headers.get("Content-Type", "application/octet-stream").split(";")[0],
        "filename": ((record.get("images") or [{}])[0]).get("filename"),
        "bytes": len(body),
        "seconds": seconds,
        "provenance": record.get("provenance"),
    }
    if upload_url:
        try:
            uploaded = put(upload_url, data=body, headers={"Content-Type": result["content_type"]}, timeout=600)
        except requests.RequestException as exc:
            raise JobError(f"the upload to upload_url failed: {exc}") from exc
        if uploaded.status_code not in (200, 201, 204):
            raise JobError(f"upload_url answered HTTP {uploaded.status_code}")
        result["uploaded"] = True
    elif max_inline_bytes is not None and len(body) > max_inline_bytes:
        raise JobError(f"the output is {len(body)} bytes, over the {max_inline_bytes}-byte inline limit: "
                       "send an 'upload_url' (a presigned PUT) for it")
    else:
        result["base64"] = base64.b64encode(body).decode("ascii")
    return result


def run_job(job: dict, wrapper: Wrapper, progress: Callable[[dict], None] = lambda update: None,
            max_inline_bytes: Optional[int] = None, put: Callable[..., requests.Response] = requests.put,
            fetch: Callable[..., requests.Response] = requests.get, clock: Callable[[], float] = time.monotonic,
            sleep: Callable[[float], None] = time.sleep) -> dict:
    """Submit one job, wait for it, and hand the output back inline or to its upload_url.

    max_inline_bytes: the caller's limit on an inline (base64) result.
    """
    started = clock()
    response, job_id, timeout = _queue(job, wrapper, fetch)
    try:
        if response.status_code == 504:
            wait_for_job(wrapper, job_id, started + timeout, progress, clock, sleep)
            response = wrapper.output(job_id)
        record = wrapper.job(job_id)
        return _deliver(job_id, response, record, round(clock() - started, 1), job.get("upload_url"), max_inline_bytes, put)
    finally:
        wrapper.delete(job_id)


# Asynchronous jobs (vast.ai's /submit, /status and /cancel): the caller polls
# instead of holding one request open for the whole render.

def submit_job(job: dict, wrapper: Wrapper, fetch: Callable[..., requests.Response] = requests.get) -> str:
    """Queue one job and return its job id; the render goes on in ComfyUI (see job_state)."""
    return _queue(job, wrapper, fetch)[1]


def job_state(job_id: str, wrapper: Wrapper, upload_url: Optional[str] = None,
              put: Callable[..., requests.Response] = requests.put) -> dict:
    """Where a submitted job is: pending / in_progress (with progress), or its end.

    completed carries the result (as run_job returns it) under "output"; failed and
    cancelled carry an "error". A job is deleted from the worker once it is
    reported finished; one whose upload failed stays, so a later call can retry it.
    """
    record = wrapper.job(job_id)
    status = record.get("status")
    if status in ("pending", "in_progress"):
        info = record.get("progress") or {}
        return {"status": status, "progress": {"percent": info.get("percent"), "step": info.get("step"), "node": info.get("node_class")}}
    if status == "failed":
        error = record.get("execution_error") or {}
        state = {"status": "failed", "error": f"the render failed: {error.get('exception_message') or 'ComfyUI reported no message'}"}
    elif status == "cancelled":
        state = {"status": "cancelled", "error": "the render was cancelled on the worker"}
    else:
        start, end = record.get("execution_start_time"), record.get("execution_end_time")
        seconds = round((end - start) / 1000, 1) if isinstance(start, (int, float)) and isinstance(end, (int, float)) else None
        state = {"status": "completed", "output": _deliver(job_id, wrapper.output(job_id), record, seconds, upload_url, None, put)}
    wrapper.delete(job_id)
    return state


def cancel_job(job_id: str, wrapper: Wrapper) -> None:
    wrapper.cancel(job_id)


def start_comfyui(data_dir: Optional[str], hidden_env: tuple) -> tuple[subprocess.Popen, str]:
    """ComfyUI + wrapper on 127.0.0.1 through docker/entrypoint.sh; returns the process and its token.

    hidden_env: the platform's worker credentials, kept out of ComfyUI's environment.
    """
    token = secrets.token_urlsafe(24)
    env = {key: value for key, value in os.environ.items() if key not in hidden_env}
    env.update(COMFYUI_LISTEN="127.0.0.1", WRAPPER_AUTH_TOKEN=token)
    if data_dir:
        env["COMFYUI_DATA_DIR"] = data_dir
    process = subprocess.Popen([os.path.join(COMFY_ROOT, "docker", "entrypoint.sh")], env=env, cwd=COMFY_ROOT)
    return process, token


def wait_ready(wrapper: Wrapper, process: subprocess.Popen, timeout: int = BOOT_TIMEOUT) -> float:
    """Seconds until the wrapper answered; raises if ComfyUI exits or never answers."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited during boot (code {process.returncode})")
        if wrapper.ready():
            return time.monotonic() - started
        time.sleep(2)
    raise RuntimeError(f"ComfyUI did not answer within {timeout}s")
