FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/ComfyUI

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libglib2.0-0 \
        libgl1 \
        tini && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt manager_requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY . .

RUN mkdir -p input output temp user models api_server/workflows

EXPOSE 8188 8000

ENTRYPOINT ["tini", "--"]
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
