# Serverless worker images (published as shivanshtalwar0/comfyui:serverless-*):
# the full image (target `comfyui` of the root Dockerfile, torch baked in) plus
# a worker that takes jobs from the platform and renders each one through the
# wrapper API of a ComfyUI it starts on 127.0.0.1 (jobs.py).
#
#   runpod  RunPod Serverless queue worker (runpod_handler.py)
#   vast    vast.ai Serverless PyWorker    (vast_worker.py)
#
# The worker and its platform SDK get their own venv (/opt/worker-env), so the
# SDK's pins (vastai pins pillow and cryptography) never touch the packages
# ComfyUI imports; ComfyUI runs on the image's own Python as before.
#
# Build context is this directory. CI builds on the full image of the same
# commit:
#   docker build --target runpod --build-arg BASE_IMAGE=shivanshtalwar0/comfyui:sha-<short> docker/serverless
ARG BASE_IMAGE=shivanshtalwar0/comfyui:latest
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.2

FROM ${UV_IMAGE} AS uv

FROM ${BASE_IMAGE} AS runpod

ARG RUNPOD_VERSION=1.12.0

RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    export UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 && \
    uv venv --python "$(command -v python)" /opt/worker-env && \
    uv pip install --no-cache --python /opt/worker-env/bin/python "runpod==${RUNPOD_VERSION}"

COPY jobs.py runpod_handler.py /opt/ComfyUI/docker/serverless/

# The base healthcheck needs WRAPPER_AUTH_TOKEN, which the worker now makes
# per boot; RunPod watches the worker through its queue instead.
HEALTHCHECK NONE
ENTRYPOINT ["tini", "--", "/opt/worker-env/bin/python", "/opt/ComfyUI/docker/serverless/runpod_handler.py"]
CMD []

FROM ${BASE_IMAGE} AS vast

ARG VASTAI_VERSION=1.8.2

RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    export UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 && \
    uv venv --python "$(command -v python)" /opt/worker-env && \
    uv pip install --no-cache --python /opt/worker-env/bin/python "vastai==${VASTAI_VERSION}"

COPY jobs.py vast_worker.py /opt/ComfyUI/docker/serverless/

# The PyWorker's port; the template maps it (-p 3000:3000).
ENV WORKER_PORT=3000
EXPOSE 3000
# vast.ai tracks the worker through the PyWorker's own reports.
HEALTHCHECK NONE
ENTRYPOINT ["tini", "--", "/opt/worker-env/bin/python", "/opt/ComfyUI/docker/serverless/vast_worker.py"]
CMD []
