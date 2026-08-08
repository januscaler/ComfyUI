# Build stages are ordered so Docker caches aggressively:
#   - `base`: system packages + pip requirements. Rebuilt ONLY when
#     requirements*.txt change.
#   - `comfyui`: the ComfyUI sources on top of base. Rebuilt ONLY when the
#     source tree changes (models/, output/ etc. are excluded via .dockerignore
#     and mounted as volumes at runtime).
# ComfyUI's PyPI torch wheels bundle the CUDA runtime, so this slim image uses
# the host GPU through the NVIDIA container toolkit (see docker-compose.yml).
# To build on an explicit CUDA base instead:
#   docker compose build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
ARG BASE_IMAGE=python:3.12-slim

FROM ${BASE_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/ComfyUI

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        gcc \
        g++ \
        git \
        libglib2.0-0 \
        libgl1 \
        make \
        tini && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt manager_requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

FROM base AS comfyui

COPY . .

RUN mkdir -p input output temp user models api_server/workflows

EXPOSE 8188 8000

ENTRYPOINT ["tini", "--"]
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
