"""RunPod Serverless worker (image target `runpod` of docker/serverless/Dockerfile).

Each worker starts ComfyUI + wrapper in the background (jobs.py), waits for it
to answer and only then takes jobs from the endpoint's queue, one at a time.
A job's `input` is the job described in jobs.py; its output is the result.

The endpoint's network volume, mounted at /runpod-volume, is the data dir: the
models, Hugging Face cache and Triton kernels in the layout the pods keep on
/workspace, so the pods' volume works as it is. Without a volume the wrapper
downloads the checkpoints on the first job of every worker.
"""

import os

import requests
import runpod

from jobs import JobError, Wrapper, log, run_job, start_comfyui, wait_ready

VOLUME = "/runpod-volume"
# RunPod keeps job results small; a bigger output needs the job's upload_url.
MAX_INLINE_BYTES = 15 * 1024 * 1024
HIDDEN_ENV = ("RUNPOD_AI_API_KEY", "RUNPOD_WEBHOOK_GET_JOB", "RUNPOD_WEBHOOK_POST_OUTPUT",
              "RUNPOD_WEBHOOK_POST_STREAM", "RUNPOD_WEBHOOK_PING")


def main() -> None:
    data_dir = VOLUME if os.path.isdir(VOLUME) else None
    if data_dir is None:
        log(f"no network volume at {VOLUME}: the checkpoints download on each worker's first job")
    process, token = start_comfyui(data_dir, HIDDEN_ENV)
    wrapper = Wrapper(token)
    log(f"ComfyUI ready after {wait_ready(wrapper, process):.0f}s; taking jobs")

    def handler(job: dict) -> dict:
        try:
            return run_job(job["input"], wrapper, progress=lambda update: runpod.serverless.progress_update(job, update),
                           max_inline_bytes=MAX_INLINE_BYTES)
        except JobError as exc:
            return {"error": str(exc)}
        except requests.RequestException as exc:
            if process.poll() is not None:
                # This worker cannot render any more: RunPod replaces it.
                return {"error": f"ComfyUI stopped on this worker (exit code {process.returncode})", "refresh_worker": True}
            return {"error": f"could not reach ComfyUI on this worker: {exc}"}

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
