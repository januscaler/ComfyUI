"""vast.ai Serverless PyWorker (image target `vast` of docker/serverless/Dockerfile).

A vast.ai serverless worker is an instance made from the workergroup's
template. This PyWorker (the vastai SDK's Worker) serves POST /generate on
WORKER_PORT over TLS, with a certificate vast signs for the instance. The
router hands each client a worker URL and a signature, the PyWorker checks the
signature and runs the request's payload as one job (jobs.py), answering
{"result": <the job result>}. Errors come back as {"result": {"error": ...}}.

A render longer than one request should be held open runs asynchronously, on
one worker pinned by a vast session (/session/create): POST /submit with the
job answers {"job_id"}; POST /status {"job_id", "upload_url"?} answers
{"status": pending | in_progress (+ "progress") | completed (+ "output", the
result) | failed | cancelled (+ "error")}; POST /cancel {"job_id"} stops it.
These carry no workload of their own (the session holds the render's cost)
and never queue behind a /generate.

The lifecycle starts ComfyUI + wrapper and waits for it; the SDK then runs
the benchmark job twice (a warm-up that loads the models, then a timed run)
and only after that marks the worker ready. If ComfyUI exits later, the worker
reports itself errored and vast replaces it.

There is no network volume: COMFYUI_DATA_DIR (default /workspace) is on the
instance disk. The entrypoint copies the weights from R2 there before ComfyUI
starts (COMFY_MODELS_S3_*, docker/s3_models_sync.py) and they survive a
stopped (cold) worker, so only a new worker copies them.
"""

import asyncio
import ipaddress
import json
import os
import threading
import time

import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from vastai import BenchmarkConfig, HandlerConfig, Worker, WorkerConfig

from jobs import JobError, Wrapper, cancel_job, job_state, log, run_job, start_comfyui, submit_job, wait_ready

SIGN_CERT_URL = "https://console.vast.ai/api/v0/sign_cert/"
CERT_FILE, KEY_FILE = "/etc/instance.crt", "/etc/instance.key"
HIDDEN_ENV = ("CONTAINER_API_KEY", "MASTER_TOKEN", "VAST_API_KEY")
# Replace with SERVERLESS_BENCHMARK_JOB (JSON) on an endpoint that serves another workflow.
BENCHMARK_JOB = {
    "workflow": "minimaxh3",
    "task": "text",
    "prompt": "A cute anime girl with massive fennec ears, a big fluffy tail, long blonde wavy hair and "
              "blue eyes waves at the camera in a sunny meadow, soft wind in the grass.",
}


def sign_certificate(container_id: str) -> None:
    """The TLS certificate the PyWorker serves, signed by vast for this instance (as vast's start_server.sh does)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    csr = (x509.CertificateSigningRequestBuilder()
           .subject_name(x509.Name([x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                                    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                                    x509.NameAttribute(NameOID.COMMON_NAME, "pyworker.vast.ai")]))
           .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("0.0.0.0"))]), critical=False)
           .sign(key, hashes.SHA256()))
    for attempt in range(1, 6):
        try:
            response = requests.post(SIGN_CERT_URL, params={"instance_id": container_id},
                                     data=csr.public_bytes(serialization.Encoding.PEM),
                                     headers={"Content-Type": "application/octet-stream"}, timeout=30)
            if response.ok:
                break
            failure = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            failure = str(exc)
        log(f"signing the TLS certificate failed (attempt {attempt}/5): {failure}")
        time.sleep(2 ** attempt)
    else:
        raise RuntimeError("vast.ai did not sign the worker's TLS certificate")
    with open(os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    with open(CERT_FILE, "wb") as f:
        f.write(response.content)


class ComfyUI:
    """The PyWorker lifecycle: ComfyUI is up before the benchmark, and stopped with the worker."""

    def __init__(self):
        self.process = None
        self.wrapper = None
        # The Worker's error report, set once the Worker exists.
        self.on_exit = None

    async def __aenter__(self):
        self.process, token = start_comfyui(os.environ.get("COMFYUI_DATA_DIR") or "/workspace", HIDDEN_ENV)
        self.wrapper = Wrapper(token)
        seconds = await asyncio.to_thread(wait_ready, self.wrapper, self.process)
        log(f"ComfyUI ready after {seconds:.0f}s; benchmarking")
        threading.Thread(target=self._report_exit, daemon=True).start()
        return self

    async def __aexit__(self, *exc_info):
        self.process.terminate()

    def _report_exit(self):
        code = self.process.wait()
        log(f"ComfyUI exited (code {code}); this worker cannot render any more")
        self.on_exit(f"ComfyUI exited (code {code})")


def main() -> None:
    os.environ.setdefault("WORKER_PORT", "3000")
    os.environ.setdefault("REPORT_ADDR", "https://run.vast.ai")
    os.environ.setdefault("USE_SSL", "true")
    if os.environ["USE_SSL"] == "true":
        sign_certificate(os.environ["CONTAINER_ID"])
    benchmark_job = json.loads(os.environ.get("SERVERLESS_BENCHMARK_JOB") or "null") or BENCHMARK_JOB
    comfy = ComfyUI()

    async def generate(benchmark: bool = False, **job):
        try:
            return await asyncio.to_thread(run_job, job, comfy.wrapper)
        except (JobError, requests.RequestException) as exc:
            if benchmark:
                raise  # a failed benchmark marks the worker errored instead of ready
            if comfy.process.poll() is not None:
                return {"error": f"ComfyUI stopped on this worker (exit code {comfy.process.returncode})"}
            return {"error": str(exc)}

    def answer(call):
        """A route's reply: its result, or {"error"} (a clear reason rather than an empty 500)."""
        async def route(**payload):
            try:
                return await asyncio.to_thread(call, **payload)
            except (JobError, requests.RequestException, TypeError, KeyError) as exc:
                if comfy.process.poll() is not None:
                    return {"error": f"ComfyUI stopped on this worker (exit code {comfy.process.returncode})"}
                return {"error": str(exc)}
        return route

    submit = answer(lambda **job: {"job_id": submit_job(job, comfy.wrapper)})
    status = answer(lambda job_id, upload_url=None: job_state(job_id, comfy.wrapper, upload_url))
    cancel = answer(lambda job_id: cancel_job(job_id, comfy.wrapper) or {"cancelled": True})

    worker = Worker(WorkerConfig(
        handlers=[
            HandlerConfig(
                route="/generate",
                remote_function=generate,
                allow_parallel_requests=False,
                benchmark_config=BenchmarkConfig(dataset=[{**benchmark_job, "benchmark": True}], runs=1),
            ),
            *(HandlerConfig(route=route, remote_function=call, allow_parallel_requests=True, max_queue_time=None,
                            workload_calculator=lambda payload: 0.0)
              for route, call in (("/submit", submit), ("/status", status), ("/cancel", cancel))),
        ],
        lifecycle=comfy,
    ))
    comfy.on_exit = worker.backend.backend_errored
    worker.run()


if __name__ == "__main__":
    main()
