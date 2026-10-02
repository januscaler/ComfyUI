<div></div>
<div align="center">

# ComfyUI
**The most powerful and modular AI engine for content creation.**

## RUN

```
export HF_TOKEN="your_hf_token"
python3 main.py --preview-method auto --auto-download-models
```

[![Website][website-shield]][website-url]
[![Dynamic JSON Badge][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![][github-release-shield]][github-release-link]
[![][github-release-date-shield]][github-release-link]
[![][github-downloads-shield]][github-downloads-link]
[![][github-downloads-latest-shield]][github-downloads-link]

[matrix-shield]: https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white
[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[website-shield]: https://img.shields.io/badge/ComfyOrg-4285F4?style=flat
[website-url]: https://www.comfy.org/
<!-- Workaround to display total user from https://github.com/badges/shields/issues/4500#issuecomment-2060079995 -->
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total
[discord-url]: https://discord.com/invite/comfyorg
[twitter-shield]: https://img.shields.io/twitter/follow/ComfyUI
[twitter-url]: https://x.com/ComfyUI

[github-release-shield]: https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-release-date-shield]: https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat
[github-downloads-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

<img width="1590" height="795" alt="ComfyUI Screenshot" src="https://github.com/user-attachments/assets/36e065e0-bfae-4456-8c7f-8369d5ea48a2" />
<br>
</div>

ComfyUI is the AI creation engine for visual professionals who demand control over every model, every parameter, and every output. Its powerful and modular node graph interface empowers creatives to generate images, videos, 3D models, audio, and more...
- ComfyUI natively supports the latest open-source state of the art models.
- It is by far the most optimized inference engine for diffusion models in the world with a focus on local consumer hardware.
- Can run even the biggest open source models on as low as 4GB vram + 8GB ram relatively quickly (saturating your GPU compute) using our state of the art asynchronous weight streaming technology.
- [Partner nodes](https://docs.comfy.org/tutorials/partner-nodes/overview#partner-nodes) provide access to the best closed source models such as Nano Banana, Seedance, Hunyuan3D, etc.
- It is available on Windows, Linux, and macOS, locally with our [desktop application](https://www.comfy.org/download), our [portable install](#installing) or on our [cloud](https://www.comfy.org/cloud).
- The most sophisticated workflows can be exposed through a simple UI thanks to App Mode.
- It integrates seamlessly into production pipelines with our API endpoints.

## Get Started

### Quickstart (uv)

The fastest local setup is the included `start_uv.sh` launcher, which uses [uv](https://docs.astral.sh/uv/) to create the virtualenv and install all dependencies automatically on first run:

```bash
./start_uv.sh                  # first run: creates .venv, installs requirements.txt, starts on :8188
./start_uv.sh --port 8588 --disable-auto-launch   # extra args are passed to main.py
```

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv` on macOS). Later runs skip installation and start the server immediately.

### Local

#### [Desktop Application](https://www.comfy.org/download)
- The easiest way to get started.
- Available on Windows & macOS.

#### [Manual Install](#manual-install-windows-linux)
Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

### Cloud

#### [Comfy Cloud](https://www.comfy.org/cloud)
- Our official paid cloud version for those who can't afford local hardware.

## Examples
See what ComfyUI can do with the [newer template workflows](https://comfy.org/workflows) or old [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Wrapper API

A simplified REST API on top of the regular ComfyUI API (`api_wrapper/`). It hides the node graph and model setup: each workflow has a dedicated synchronous endpoint that returns the final artifact (image, video, audio, 3D asset, ...) directly as a downloadable file. Missing model files for the requested workflow are downloaded automatically on first use (requires starting ComfyUI with `--auto-download-models`; gated models need `HF_TOKEN`).

- `POST /api/wrapper/{workflow}/generate` — multipart form with `prompt` and `image` (optional `negative_prompt`, `seed`, `steps`, `cfg`, `megapixels`, `timeout`, `free_vram`, `quantization`). Runs the workflow to completion and returns the final file as a download (`Content-Disposition: attachment`). By default the request waits as long as the job takes; set `timeout` (5–86400 seconds) if you prefer a cap — the timeout response includes the `job_id` to pick the result up via `/jobs/{job_id}`. `GET /api/wrapper/workflows` lists the available `{workflow}` names.
- `GET /api/wrapper/jobs/{job_id}` — job status (`pending` / `in_progress` / `completed` / `failed` / `cancelled`) with output image URLs; also useful to pick up a generate call that timed out (the timeout response includes the `job_id`).
- `GET /api/wrapper/jobs/{job_id}/image` — redirects to the generated image.
- `DELETE /api/wrapper/jobs/{job_id}` — worker cleanup once you have stored the result (or after a failure/cancel): deletes the job's output files, the uploads saved for it and its history entry, and answers `{"deleted": {"outputs": n, "inputs": m, "history": true|false}, "refused": 0}`. Idempotent (unknown job → zeros), `409` while the job is pending/running, and it never deletes anything outside ComfyUI's output/input/temp directories (real paths are checked; symlinks pointing out are refused). Each job's uploads and outputs live in their own `wrapper/<job_id>/` folder under `input/` and `output/`.
- `POST /api/wrapper/free` — releases the GPU VRAM and RAM used by model execution (unloads all models, empties torch caches). When a job is running/queued the release happens automatically right after that job finishes.
- `GET /api/wrapper/docs` — Swagger UI for the wrapper API (spec at `/api/wrapper/openapi.json`).

Example:

```bash
curl -o result.png -X POST http://127.0.0.1:8188/api/wrapper/flux2klein9b/generate \
  -F "prompt=make it snow" -F "image=@input.png" -F "steps=20"
```

**Memory:** by default the wrapper releases VRAM and RAM as soon as a job finishes — models are unloaded and caches emptied automatically, so the server stays resource-efficient between requests. Send `free_vram=false` on a generate request to keep models loaded across consecutive jobs (e.g. batch runs).

**Precision:** `quantization=fp8` (default) uses the shipped fp8 checkpoint — on Apple Silicon it loads as bf16 automatically. `quantization=fp4` converts the checkpoint to NVFP4 on first use (~half the file size, best on CUDA: native on Blackwell, emulated on older GPUs and CPU); on MPS the wrapper automatically falls back to fp8 since NVFP4 needs fp8 block scales that MPS cannot handle. MiniMax H3 has its own set (`fp8` / `int8` / `bf16` / `nvfp4`, default `int8`) — see below. Every generate response reports the checkpoints that actually ran in the `X-Wrapper-Models` header and the resolved parameters in `X-Wrapper-Settings`, so you can confirm what produced a file without reading the server log.

**MiniMax H3 prompt rewriting:** H3 does not want a one-line description — it wants an alignment instruction, `integrated_multimodal_description` split into timed shots, `overall_soundscape` and `non_diegetic_music` (or six labelled sections in reference mode). So the H3 endpoints take `prompt` as a plain description in your own words ("two wrestlers in a gym, one chokeslams the other onto a crash mat, heavy metal soundtrack") and have a Xiaomi MiMo model rewrite it into the real thing, following [MiniMax's own H3 prompt-writing guide](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing) (vendored in `api_wrapper/h3_prompt_skill/`) and this request's actual mode, duration, canvas and reference labels. Set `MIMO_API_KEY` to enable it.

- `raw_prompt` — a ready-made H3 prompt, used verbatim. Skips the rewrite entirely and needs no API key, so existing callers keep working by renaming `prompt` to `raw_prompt`.
- `llm_image` — an image given to the rewriter as visual context, so it describes what is actually in your footage. It never reaches H3 itself. When omitted, the task's own first image (the `image` first frame, or `ref_images` #1) is used, so image- and reference-driven jobs get visual grounding for free.
- `llm_model` — `auto` (default) sends every rewrite to `mimo-v2.6-flash`, an omnimodal model that reads the context image when there is one. `mimo-v2.6-pro`, `mimo-v2.5` and `mimo-v2.5-pro` can be pinned instead. **`mimo-v2.5-pro` cannot read images** — the API rejects any request carrying one (`HTTP 404: No endpoints found that support image input`), so pinning it alongside an uploaded `llm_image` returns a 400, while the context image the wrapper adds on its own is simply skipped and reported. Leave it on `auto` unless you have a reason not to.
- `POST /api/wrapper/minimaxh3/prompt` turns a free-form prompt into an H3 prompt and returns it as JSON — no GPU work, no render. The task is inferred from what you attach (reference assets → ref2va, a keyframe → image-to-video, nothing → text-to-video), so a prompt on its own is a complete request; `POST /api/wrapper/minimaxh3/{task}/prompt` pins one. Iterate there, then send the result back to `/generate` as `raw_prompt`. Accepts multipart, form-urlencoded or JSON.

```bash
curl -s --max-time 60 -X POST http://127.0.0.1:8188/api/wrapper/minimaxh3/prompt \
  -d 'prompt=two wrestlers in a gym, one chokeslams the other onto a crash mat, heavy metal soundtrack'
```

A rewrite takes 10–20 s (it is one call to a large model), so allow at least 60 s — a short client timeout aborts mid-flight and looks like a missing endpoint. `/generate` is synchronous and streams the MP4 back in the response body, so it needs `-o out.mp4` and a timeout in minutes, not seconds.

**Generated prompts are recorded.** Because the H3 prompt is written by a model rather than supplied, every generation logs the full prompt it ran with to the server log (job id, mode, model, and the free-form input it came from), and saves a `<video>.prompt.json` sidecar next to the output — input prompt, generated prompt, model, mode, settings and checkpoints. The `X-Wrapper-Prompt-File` response header gives its path. `raw_prompt` runs are recorded the same way, marked `"source": "raw_prompt"`, so every video in the output directory can be traced back to the exact prompt that made it.

**MiniMax H3 memory limits:** H3's four checkpoints total ~43 GB (21 GB UNET + 16 GB text encoder + 5.8 GB VAEs), so on a 32 GB RAM box every job runs within ~2 GB of the ceiling and an over-large request does not fail gracefully — it OOM-kills the server and takes the queue with it. The wrapper therefore checks clip length and canvas×frames *before* queueing and returns a 400 for anything past the host's envelope. Frame count is the limit that bites: the video VAE tiles spatially but not temporally, so a longer clip costs far more memory than a wider one and no resolution is small enough to compensate. Defaults (`864x480`, 5.17 s, 20 steps, `int8`) are the largest values measured to complete with memory to spare on a 32 GB RTX 5090; raise `COMFY_MINIMAX_H3_MAX_FRAMES` / `COMFY_MINIMAX_H3_MAX_PIXEL_FRAMES` on a host with more RAM, swap, or running with `--fast-disk`. See `.env.example` for the deployment side.

**Interactive docs:** with the server running, open `http://127.0.0.1:8188/api/wrapper/docs` in a browser to browse and try every endpoint (the "Try it out" button works — you can upload an image and generate right from the docs page). The raw OpenAPI spec is at `http://127.0.0.1:8188/api/wrapper/openapi.json` for code generation; swap the port if you started ComfyUI elsewhere.

The first workflow shipped is `flux2klein9b`, a FLUX.2 [klein] 9B image edit: the input image is scaled to a megapixel budget, attached to the conditioning as a reference latent, and sampled with the flux2 custom sampler stack. `qwenimage21` is the Qwen Image 2.1 image edit and `qwenimage21-txt2img` its text-to-image variant, both built like the official Qwen Image 2.1 templates (int8_convrot UNET and text encoder, `quantization=bf16` for the full-precision pair; 25 steps, cfg 1): the edit takes `image` and/or up to 15 `ref_images`, named `<image1>`, `<image2>`, ... in the prompt, and edits image 1. `ideogram4` is a text-to-image workflow (no input image needed) using the Ideogram 4 dual-model CFG stack — it natively understands rich structured JSON prompts, e.g. `curl -F "prompt=$(cat prompt.json)"` with a JSON prompt describing composition, style and elements; the full example (with bounding boxes per element) is shown in the interactive docs. `minimaxh3` is an omni-modal video workflow with three task endpoints under `/api/wrapper/minimaxh3/{task}`: `text` (text-to-video), `image` (image-to-video, first/last frame uploads) and `reference` (ref2va: up to 9 reference images, 3 videos and 3 audio clips, referenced in the prompt as `<Picture i>` / `<Video k>` / `<Audio j>`); it returns a synchronized audio+video MP4. To add another workflow (e.g. a 3D model generator), add its graph builder to the `WORKFLOWS` registry in `api_wrapper/workflows.py`, a model-setup handler in `api_wrapper/routes.py`, and it immediately gets its own `/api/wrapper/{name}/generate` endpoint — output files (video/audio/3D) are detected automatically by their node output type. All endpoints are also available without the `/api` prefix.

### Authentication

Set `WRAPPER_AUTH_TOKEN` to require `Authorization: Bearer <token>` on **every** HTTP route of the server — the wrapper endpoints and the native ComfyUI ones alike (`/prompt`, `/queue`, `/history`, `/view`, `/object_info`, `/ws`, `/internal/...`, the docs pages and the static frontend). It is meant for publicly reachable boxes such as RunPod pods, where each pod is started with its own random token (e.g. `WRAPPER_AUTH_TOKEN=$(openssl rand -hex 32)`). Unset or empty, nothing changes: the server stays open, as before.

- A missing or wrong token gets a `401` with `WWW-Authenticate: Bearer` and the body `{"error": {"type": "unauthorized", "message": "missing or invalid bearer token"}}`. The token is compared in constant time.
- Browsers cannot set headers on a WebSocket, so `/ws` (and its `/api/ws` alias) also accepts the token as a query parameter: `ws://host:8188/ws?token=<token>&clientId=...`. No other route accepts `?token=`.
- `OPTIONS` requests pass without a token: CORS preflights never carry credentials, and all they get back is an empty reply.
- The token is read once at startup, so changing it needs a restart. The stock ComfyUI web UI does not send the header, so with auth on, the server is for API clients rather than for opening the frontend in a browser. The `developer-api` compose service does not send it either.

```bash
curl -o result.png -X POST http://127.0.0.1:8188/api/wrapper/flux2klein9b/generate \
  -H "Authorization: Bearer $WRAPPER_AUTH_TOKEN" \
  -F "prompt=make it snow" -F "image=@input.png"
```

### Docker image (local GPU box, RunPod and vast.ai)

The worker is published to Docker Hub as **`shivanshtalwar0/comfyui`** by [`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml), in two flavours built from the same [`Dockerfile`](Dockerfile):

- **full** (target `comfyui`, the default): torch + requirements baked in, a ~5.2 GB compressed pull. The local GPU box and vast.ai pods run it; it also carries pinned `cloudflared` and `s5cmd` binaries for vast.ai (see [vast.ai](#vastai-no-volume-no-proxy)).
- **runpod** (target `runpod`): the same system packages and sources, no Python dependencies, a ~0.38 GB pull. The first pod on a network volume installs the Python env onto the volume and every later pod reuses it (see [RunPod variant](#runpod-variant-runpod)).

| Trigger | full image tags | runpod image tags |
|---|---|---|
| push to `master` (source or Docker files changed) | `latest`, `sha-<short>` | `runpod`, `runpod-sha-<short>` |
| tag `vX.Y.Z` | `X.Y.Z`, `X.Y`, `sha-<short>` | `runpod-X.Y.Z`, `runpod-X.Y`, `runpod-sha-<short>` |
| manual run (Actions → Docker image → Run workflow) | `<branch>`, `sha-<short>` | `runpod-<branch>`, `runpod-sha-<short>` |
| pull request touching Docker files | nothing: build + smoke test only | nothing: build + smoke test only |

Every build is smoke-tested before anything is pushed. Full image: torch has CUDA, `prefetch --dry-run` resolves every wrapper checkpoint, `cloudflared` and `s5cmd` run, a bucket sync without credentials stops the container, and the server boots (CPU mode) and answers `/api/wrapper/workflows` with 200 with the bearer token and 401 without it. Runpod image: no torch in the image, it refuses to start without a volume, a first container on a fresh docker volume builds the real CUDA env on it and passes the same 200/401 check, and a second container on that volume reuses the env (no rebuild) and comes up faster. The two flavours build in parallel jobs with separate registry build caches (`:buildcache`, `:buildcache-runpod`). Publishing needs the repository secret **`DOCKERHUB_TOKEN`** (a Docker Hub access token with Read & Write). `DOCKERHUB_USERNAME` / `DOCKERHUB_IMAGE` repository variables override the defaults.

Both images are **CUDA only**: `python:3.12-slim` plus the PyTorch **cu128** wheels (Blackwell / RTX 5090 kernels, host driver ≥ 570). The full image's build fails if any dependency leaves it with a CPU torch; the runpod image's env build on the volume fails the same way. Nothing model-sized is baked in. [`docker/entrypoint.sh`](docker/entrypoint.sh) configures everything from env, so the image runs the same with or without a compose file:

| Env | Default | Meaning |
|---|---|---|
| `WRAPPER_AUTH_TOKEN` | unset | Bearer token on every route (see Authentication). **Always set it on public boxes.** |
| `COMFYUI_DATA_DIR` | `/workspace` if present | Persistent store: `models/` is linked to `<dir>/models`, `HF_HOME` to `<dir>/huggingface`, Triton's kernel cache to `<dir>/.triton`. |
| `COMFYUI_MODELS_DIR` | `<data dir>/models` | Model store on its own. A bind mount on `/opt/ComfyUI/models` always wins. |
| `COMFYUI_PORT` | `8188` | Container port. |
| `AUTO_DOWNLOAD_MODELS` | `1` | Fetch a workflow's checkpoints on first use. |
| `VRAM_HEADROOM_GB`, `CACHE_RAM_GB`, `ASYNC_OFFLOAD_STREAMS`, `FAST_DISK` | `2`, `2 8`, `0`, `1` | Memory tuning, as in `.env.example`. |
| `COMFYUI_ARGS` | empty | Extra ComfyUI flags. |
| `COMFY_SEQUENCE_PARALLEL_GPUS` | `auto` | Several GPUs: how many split each MiniMax H3 forward (see below). `auto` takes the largest group the box has (8, 7, 4 or 2), `1` turns it off, `N` asks for exactly N. |
| `COMFY_HF_DOWNLOAD_MODE` | `direct` with a data dir, else `cache` | `direct` downloads Hugging Face weights next to the models dir and moves them in (one copy). `cache` keeps the HF cache plus a copy, which is useful when the HF cache is shared between projects. |
| `MIMO_API_KEY`, `HF_TOKEN` | unset | H3 prompt rewrite; gated FLUX.2 [klein] weights. |
| `COMFY_ENV_WAIT_SECONDS` | `1800` | Runpod image: how long a pod waits for another pod that is building the env before it gives up. |
| `COMFY_ENV_LOCK_STALE_SECONDS` | `120` | Runpod image: an env lock whose heartbeat has not moved for this long belongs to a dead pod and is taken over. |
| `COMFY_ENVS_DIR` | `<data dir>/envs` | Runpod image: where the envs live. |
| `CF_TUNNEL_TOKEN` | unset | vast.ai: token of the pod's remotely-managed Cloudflare tunnel. Runs `cloudflared tunnel --no-autoupdate --metrics 127.0.0.1:20241 run` beside ComfyUI (server mode only), restarted with backoff. Never printed. |
| `COMFY_MODELS_S3_BUCKET` | unset | vast.ai: copy the weights from this bucket into the models dir before ComfyUI starts (server mode only). |
| `COMFY_MODELS_S3_ENDPOINT` | unset (AWS S3) | e.g. `https://<account>.r2.cloudflarestorage.com`. |
| `COMFY_MODELS_S3_PREFIX` | `models/` (also when empty) | Key prefix mapped onto the models dir: `models/vae/x.safetensors` → `<models dir>/vae/x.safetensors`. `/` maps the bucket root. |
| `COMFY_MODELS_S3_INCLUDE` | everything | Comma-separated globs on the path under the prefix (`*` also matches `/`), e.g. `diffusion_models/*int8*,text_encoders/*,vae/*`. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | unset, unset, `auto` | The bucket's credentials (required with a bucket, never printed); `auto` is R2's region. |
| `COMFY_MODELS_PREFETCH` | unset | Without a bucket: `1` runs `prefetch` (the default checkpoints, from Hugging Face) before ComfyUI starts. Empty, `0`, `false`, `no` and `off` leave it off. |

**Local GPU box**: `cp .env.example .env`, then `docker compose pull && docker compose up -d` (or `docker compose up -d --build` to build this tree). `COMFYUI_HOST_PORT=8282` publishes it where the FloStudio rig is reached over WireGuard.

**RunPod (FloStudio burst pods)**: voxmin-backend creates the pods itself through the RunPod API, with a network volume attached at `/workspace` (Python env, models, HF cache, Triton kernels). The backend gives every pod its own `WRAPPER_AUTH_TOKEN` and reaches it on port 8188 through the RunPod proxy; put `MIMO_API_KEY` / `HF_TOKEN` in the `runpod` ExternalApi row's `podEnv`. The fastest way to start a pod is **not to pull an image at all** (see [RunPod: no image pull](#runpod-no-image-pull)): pulls from Docker Hub ran at about 10 Mbps on EU-RO-1 hosts, over 9 minutes for the full `:latest` image (~5.2 GB) and 4-5 minutes even for the slim `:runpod` one, while those same hosts fetched Python wheels at ~700 Mbps.

**Opening a pod's ComfyUI in a browser**: every route needs the pod's `WRAPPER_AUTH_TOKEN`, which a browser cannot send as a header. Whoever holds the token signs a short-lived login link instead, `https://<podId>-8188.proxy.runpod.net/?comfy_login=<exp>.<HMAC-SHA256(token, "comfy-login:<exp>")>` (at most 10 minutes ahead). Opening it sets an HttpOnly session cookie (12 h, signed the same way) and redirects to the clean URL; the UI, its API calls and its WebSocket then use the cookie, and the token itself never reaches the browser. voxmin-backend's admin pods panel builds these links ("Open ComfyUI"). See `middleware/wrapper_auth.py`.

**Fill a network volume once** so the first pod doesn't spend its boot downloading ~40 GB of weights (and, with the runpod image, installing the Python env). Run this on a cheap CPU pod with the volume attached, or locally against the model dir:

```bash
# runpod image: builds the Python env on the volume, then downloads the weights
docker run --rm -e HF_TOKEN=... -v /workspace:/workspace shivanshtalwar0/comfyui:runpod prefetch
docker run --rm -v /workspace:/workspace shivanshtalwar0/comfyui:runpod prefetch --env-only   # env only
# full image: weights only
docker run --rm -e HF_TOKEN=... -v /workspace:/workspace shivanshtalwar0/comfyui prefetch
# specs: minimaxh3[:nvfp4|int8|fp8|bf16]  minimaxh3-ref[:quant]  flux2klein9b  qwenimage21[:int8|bf16]
docker run --rm -v "$PWD/models:/opt/ComfyUI/models" shivanshtalwar0/comfyui prefetch minimaxh3:int8 --dry-run
```

With no specs it fetches `minimaxh3:int8 minimaxh3-ref:int8 flux2klein9b` (override with `PREFETCH_MODELS`). That set is the INT8 H3 UNETs (the weights the FloStudio RTX 5090 rig runs, so a shot looks the same on a pod as on the rig), the 16 GB NVFP4 text encoder that a 32 GB Blackwell card needs, and the FLUX.2 [klein] stills model. An H3 spec always comes with the encoder the wrapper loads at render time (the smallest on disk), never the 34 GB INT8 or 66 GB BF16 one.

#### RunPod: no image pull

RunPod keeps its own PyTorch image cached on its machines. A pod on `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (the image of RunPod's "Runpod Pytorch 2.8.0" template) was running 2 s after it was created, where ours took minutes to pull. That image already has Python 3.12, gcc/g++/make, git, ffmpeg and `uv`. So a FloStudio pod runs RunPod's image with [`docker/runpod-bootstrap.sh`](docker/runpod-bootstrap.sh) as its start command, and everything of ours lives on the network volume:

```
/workspace/code/ComfyUI-<commit>.tar.gz   this repository at COMFY_CODE_REF (~13 MB), fetched once per commit
/workspace/envs/<key>.tar                 the Python env (torch cu128 + requirements.txt), archived once
/workspace/envs/.lock-<key>/              present only while a pod is building <key>
/workspace/models, huggingface, .triton   weights and caches, as with any image
```

On every boot the bootstrap unpacks the code to `/opt/ComfyUI` on the container disk (seconds) and hands over to `docker/entrypoint.sh`, which unpacks the env archive to `/opt/comfy-env/<key>` and starts ComfyUI. `COMFY_CODE_REF` must be a commit sha (a branch name would be cached under that name and never updated); voxmin-backend's `scripts/flo-runpod-template.mjs` puts the bootstrap, pinned to a commit, into a RunPod template that System Config → FloStudio → RunPod → template points at. Rolling out new code = re-running that script with the new commit.

#### RunPod variant (`:runpod`)

`docker.io/shivanshtalwar0/comfyui:runpod` is `python:3.12-slim` plus the apt packages (ffmpeg, git, and the gcc/g++/make that Triton's JIT needs at runtime), the `uv` binary and the ComfyUI sources: about **0.38 GB compressed** (1.1 GB unpacked) against **5.15 GB compressed** for the full image, 4.8 GB of which is the torch + requirements layers. It uses the same env archive on the volume as the bootstrap does.

- **The key** (`docker/env-key`, exported as `COMFY_ENV_KEY`) is a hash of `requirements.txt`, the torch index, the base interpreter, the CPU arch and the OS release, printed by `docker/env-inputs.sh` (the hashed text is in `docker/env-inputs`). The slim image bakes it; code fetched by the bootstrap computes it at boot with the same script. Code-only changes reuse the env on the volume; a dependency change builds a new one next to it.
- **Why an archive and not the env directory on the volume**: RunPod volumes are MooseFS, which streams a big file fast (~800 MB/s measured reading a checkpoint) but creates small files slowly. The CUDA env is ~7.7 GB in ~42k files; written in place onto the volume it was still being written after 16 minutes, and every import from it would pay a network round trip per file. So it is built on the container disk and published as one tar.
- **First boot on a volume** (or after a dependency change): the pod builds the env on its container disk with `uv` (torch from the cu128 index first, then `requirements.txt`, bytecode precompiled), checks that torch is a CUDA build, writes the completion marker, and publishes `<key>.tar` to the volume under a temporary name renamed into place last. The container disk needs room for the env plus `uv`'s cache while it builds.
- **Every later boot** unpacks the archive next to `/opt/comfy-env/<key>` and renames it into place. The log says which it was: `python env <key> unpacked from ... in <n>s, reusing it` or `... built in <n>s` (with the torch, requirements.txt and archive times). A restarted container reuses its unpacked env.
- **Several pods at once**: only one builds. The lock is a directory (`mkdir` is atomic on a network filesystem, where `flock` may not be). The builder keeps a heartbeat file in it; the other pods wait (up to `COMFY_ENV_WAIT_SECONDS`) and then unpack the published archive. If the heartbeat stops moving for `COMFY_ENV_LOCK_STALE_SECONDS` (a pod killed mid-build), a waiting pod takes the lock over and builds again. Only the lock owner publishes, and only a finished archive.
- **No volume, no start**: without `/workspace` (or `COMFYUI_DATA_DIR`) the runpod variant exits at once with an error. Use the full image anywhere without a volume. `prefetch` and any other command (`bash`, `uvicorn ...`) also run inside the env.

A first boot also has to fit in voxmin-backend's FloStudio boot timeout (900 s by default), so build the env once on a new volume (`prefetch --env-only`, or a one-off pod) rather than letting a customer's pod do it.

Envs and code archives are never deleted automatically, because a pod still running older code may need them. Once none does, free the space from a pod on the volume:

```bash
cd /workspace/envs && ls | grep -vx "$(cat /opt/ComfyUI/docker/env-key).tar" | xargs -r rm -rf
```

#### vast.ai: no volume, no proxy

vast.ai pods run the full image with `COMFYUI_DATA_DIR=/workspace` on the container disk. There is no shared volume and no HTTPS proxy, so voxmin-backend gives each pod its own Cloudflare tunnel (`CF_TUNNEL_TOKEN`; the tunnel's ingress routes the pod's hostname to `http://127.0.0.1:8188`) and a bucket to copy the weights from (`COMFY_MODELS_S3_*` and the R2 credentials). In server mode the entrypoint then:

- starts `cloudflared` in the background with the token in `TUNNEL_TOKEN` (not argv, so `ps` never shows it), prefixes its output `cloudflared:` with the token masked, restarts it with backoff (1 s doubling to 60 s) whenever it exits, and logs `tunnel ready` once its `/ready` metrics endpoint answers 200;
- copies the weights with [`docker/s3_models_sync.py`](docker/s3_models_sync.py) before ComfyUI starts: `s5cmd` fetches 4 files at once, each as 16 parallel 64 MiB ranged GETs. A file already on disk with the object's size is kept, so a stopped-then-started container downloads nothing; each download lands as `<name>.s3-partial` and is renamed only once it has the object's full size (leftovers of a killed sync are deleted on the next run). The disk must hold the download plus a 5 GB margin. A download still running after 10 minutes and slower than its share of a 20 MB/s link (5 MB/s with 4 at once) is killed. Each file gets 3 attempts; if one still fails the container exits non-zero and never becomes ready, and the backend recycles it. A `progress` line every 30 s and the last line report files, bytes, seconds and MB/s.

Without a bucket, `COMFY_MODELS_PREFETCH=1` fetches the default checkpoints from Hugging Face (`prefetch`) before starting instead. To see what a pod would download, with the credentials exported in your shell (`-e NAME` with no value passes them through, so they stay out of your shell history):

```bash
docker run --rm -e COMFY_MODELS_S3_BUCKET=<bucket> -e COMFY_MODELS_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  --entrypoint python shivanshtalwar0/comfyui docker/s3_models_sync.py models --dry-run
```

ComfyUI itself starts without the bucket credentials and the tunnel token in its environment: the entrypoint unsets them first.

#### Several GPUs: one H3 render on all of them

On a box with several GPUs, each MiniMax H3 forward is split over all of them with sequence parallelism (DeepSpeed-Ulysses, [`comfy/ldm/minimax/sequence_parallel.py`](comfy/ldm/minimax/sequence_parallel.py)). One render gets faster; the box still renders one job at a time. The text encoder and the VAEs stay on the first GPU, so a whole render gains less than one DiT step.

| Shape (DiT int8, cu130, 4x RTX 5090) | 1 GPU | 2 GPUs | 4 GPUs |
|---|---|---|---|
| 480p, 2.3 s (7.3k tokens) | 0.98 s/step | 0.68 | 0.38 (2.6x) |
| 720p, 5 s (26.6k tokens) | 7.40 s/step | 4.23 | 2.24 (3.3x) |
| 720p, 15 s (76.4k tokens) | 47.7 s/step | 25.8 | 13.4 (3.6x) |

Whole renders through the wrapper (`POST /api/wrapper/minimaxh3/text/generate`, int8, 20 steps, the image's entrypoint on a vast.ai 4x RTX 5090 host, 2026-10-02):

| Render | 1 GPU | 4 GPUs |
|---|---|---|
| 720p, 5 s | 164.1 s (7.50 s/step) | 57.5 s (2.19 s/step): 2.85x |
| 720p, 15 s | about 16 min (47.7 s/step, the DiT benchmark) | 298.9 s (13.27 s/step) |

The text encoder and VAE decode took about 14 s (5 s clip) and 33 s (15 s clip) of those totals, on the first GPU either way. Each GPU holds the whole DiT: at 720p 15 s the first GPU peaked at 28.2 GB and the followers at 26.6–28.0 GB, and the host used 62 GB of RAM. Measured on PCIe 5.0 x16 boards without P2P; boards with slower links scale worse.

Given the same inputs the split DiT's output is bit-identical to one GPU's. Whole renders are not bit-identical run to run even on one GPU (repeats differ by about 41 dB PSNR, or not at all), and one-GPU against four-GPU renders fall in the same range; the audio came out identical.

How it runs:

- The entrypoint gives ComfyUI the first GPU and starts one follower per other GPU ([`docker/sp_follower.py`](docker/sp_follower.py), with ComfyUI's own flags and only its own GPU visible). H3 has 56 attention heads, so a group is 8, 7, 4 or 2 GPUs; `COMFY_SEQUENCE_PARALLEL_GPUS` overrides the size (`1` turns it off).
- When ComfyUI loads an H3 diffusion model it opens the group on `127.0.0.1:29511` (`COMFY_SP_PORT`), and the followers load the same file the first time a forward needs it.
- Each step, ComfyUI posts the forward's arguments on that TCP store and broadcasts their tensors over NCCL. Every GPU runs 1/N of the packed sequence; attention exchanges heads for rows with two all-to-alls; the last block gathers the whole output.
- A forward the followers could not reproduce exactly runs on the first GPU alone: weight patches (a LoRA), another diffusion-model wrapper, block or attention patches, an attention override.
- A cancel waits for the current step to finish (the GPUs run each forward in lockstep): a cancel at step 8 of 20 returned 3 s later, and the next render ran split again.
- A split forward runs outside ComfyUI's allocation compiler (the comfy_aimdo malloc graph), which cannot plan its first and last blocks.
- A follower that exits stops ComfyUI too, and the container restarts with a fresh group. If the followers never join (`COMFY_SP_JOIN_SECONDS`, 300 s), renders use one GPU.

To time one DiT step on 1 against N GPUs without ComfyUI, with seeded inputs (`--compare` reports how far apart two runs' outputs are):

```bash
torchrun --nproc-per-node 4 script_examples/h3_sp_bench.py --ckpt models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --width 1120 --height 640 --frames 124 --out sp4.pt
```

### Serverless API (RunPod and vast.ai)

Serverless workers render MiniMax H3 videos (or any other wrapper workflow) on GPUs that run only while there is work. [`docker/serverless/Dockerfile`](docker/serverless/Dockerfile) builds two worker images on top of the full image:

| Image | Platform | Worker |
|---|---|---|
| `shivanshtalwar0/comfyui:serverless-runpod` | RunPod Serverless, queue endpoint | [`runpod_handler.py`](docker/serverless/runpod_handler.py) |
| `shivanshtalwar0/comfyui:serverless-vast` | vast.ai Serverless, PyWorker | [`vast_worker.py`](docker/serverless/vast_worker.py) |

Each worker starts ComfyUI + wrapper on `127.0.0.1`, with a bearer token it makes at boot so nothing else can reach it, and runs each request as one wrapper render ([`docker/serverless/jobs.py`](docker/serverless/jobs.py)). The platform SDK lives in its own venv (`/opt/worker-env`), away from ComfyUI's packages. A job means the same on both platforms; only the envelope differs. CI builds both images on the full image of the same commit, smoke-tests them in CPU mode and pushes them as `serverless-<platform>` and `serverless-<platform>-sha-<short>` (plus `-X.Y.Z` for version tags, `-<branch>` for manual runs).

#### Quick start (RunPod)

```bash
export RUNPOD_API_KEY=...   # RunPod console → Settings → API Keys
export ENDPOINT_ID=...      # RunPod console → Serverless → the endpoint's id

curl -s "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input": {"prompt": "A red fox trots across fresh snow at sunrise, its breath misting in the cold air.", "fields": {"duration": 3}}}' \
  > reply.json
jq '{status, error, seconds: .output.seconds, settings: .output.provenance.settings}' reply.json
jq -r .output.base64 reply.json | base64 --decode > fox.mp4
```

A warm RTX PRO 6000 worker renders that 3 s, 864×480 clip in about 70 s; a default clip (5.17 s) on a freshly started worker took 259 s, loading the models included. `/runsync` waits about 90 s. A job that takes longer (a cold worker, a first-use model download, a longer clip) comes back as `IN_QUEUE` or `IN_PROGRESS` with its `id`: poll `/status/<id>`, as the clients below do.

#### The job

The job is RunPod's `input` and vast.ai's `payload`. Only `prompt` is required; the defaults render a MiniMax H3 text-to-video clip.

| Field | Default | Meaning |
|---|---|---|
| `prompt` | required | What to render. Sent to the wrapper as `raw_prompt`, which H3 gets verbatim. |
| `rewrite` | `false` | `true` sends `prompt` as a plain description that a MiMo LLM rewrites into H3's structured prompt format (needs `MIMO_API_KEY` in the endpoint's env). |
| `workflow` | `minimaxh3` | Any wrapper workflow: `minimaxh3`, `flux2klein9b`, `flux2klein9b-txt2img`, `qwenimage21`, `qwenimage21-txt2img`, `ideogram4`. |
| `task` | `text` | The H3 task: `text`, `image` or `reference`. Without `workflow` it defaults to `text`; with another workflow, only workflows that have tasks take one. |
| `fields` | `{}` | The wrapper's form fields, as they are (below). |
| `files` | `[]` | Uploads (below). |
| `upload_url` | none | A presigned `PUT` URL: the output is uploaded there instead of returned inline. |
| `timeout` | `1800` | Seconds the job may take, including a first-use checkpoint download. A render still running then is cancelled. |

H3 `fields`, all optional:

| Field | Default | Meaning |
|---|---|---|
| `width`, `height` | `864`, `480` | Rounded up to a multiple of 32. A canvas larger than H3's native 1344×768 is scaled down, keeping the aspect ratio. |
| `duration` | `5.17` | Seconds at 24 fps. The wrapper caps a clip at 124 frames (5.17 s) and width×height×frames at 1152×640×124. Those are the limits of a 32 GB-RAM box: raise them on bigger workers with `COMFY_MINIMAX_H3_MAX_FRAMES` / `COMFY_MINIMAX_H3_MAX_PIXEL_FRAMES` in the endpoint's env. The model is trained for 5-15 s. A request past a cap fails with a message naming it. |
| `steps` | `20` | Below ~15, motion and audio get mushy; above ~30, the returns diminish. |
| `seed` | random | Reported back in `provenance.seed`. |
| `scheduler` | `beta` | `beta`, `normal` or `simple`. |
| `quantization` | `int8` | UNET precision: `int8` or `fp8` (~21 GB each), `bf16` (~40 GB), `nvfp4`. Anything but `int8` downloads its checkpoint on first use. |
| `ref_image_size` | `match` | `reference` task: `match` sizes references to the video's pixel area; `max` keeps a 2048 px short edge (stronger identity, slower). |
| `llm_model` | `auto` | With `rewrite`: which MiMo model writes the prompt. |
| `free_vram` | `false` | `true` unloads the models after the job, so the next job reloads them. |

`files` items are `{"field": "image", "url": "https://..."}`, which the worker downloads (the file name comes from the URL unless `filename` is given), or `{"field": "image", "base64": "...", "filename": "first.png"}`, where `filename` is required because its extension picks the file type. RunPod limits the size of a request body, so send anything bigger than a small image as a URL. To attach several files to one field, repeat the field; they are numbered in the order given.

| Task | Upload fields |
|---|---|
| `text` | `llm_image`: visual context for `rewrite` only; never reaches H3 |
| `image` | `image`: first frame, required; `last_frame`: optional end frame; `llm_image` |
| `reference` | `ref_images` (up to 3), `ref_videos` (1), `ref_video_audios` (1), `ref_audios` (up to 3), `image` / `last_frame` (pinned first / last frame), `llm_image` |

Images are `.png .jpg .jpeg .webp .bmp .gif` (up to 64 MB), videos `.mp4 .webm .mov .mkv .avi .gif` (512 MB), audio `.wav .mp3 .flac .ogg .m4a .aac .wma` (256 MB). In a `reference` prompt the assets are `<Picture 1>`, `<Video 1>` and `<Audio 1>`, numbered in upload order. The first reference job downloads the 21 GB ref2va UNET.

**Prompts.** H3 was trained on a structured prompt: an alignment instruction, `integrated_multimodal_description` split into timed shots, `overall_soundscape` and `non_diegetic_music` (see [MiniMax's prompt guide](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing)). A plain description sent verbatim still renders, just with less control. To get the structured version, either set `"rewrite": true`, or draft it once with the wrapper's `POST /api/wrapper/minimaxh3/prompt` on any box that has `MIMO_API_KEY` and send the result back as `prompt`.

#### Example jobs

These go in `{"input": ...}` on RunPod and as the `payload` on vast.ai.

```jsonc
// Text to video, with settings
{"prompt": "A lighthouse keeper climbs a spiral staircase at night, lantern swinging, wind howling outside.",
 "fields": {"width": 768, "height": 768, "duration": 4, "steps": 24, "seed": 42}}

// Image to video: the photo is the first frame
{"task": "image",
 "prompt": "The woman in the photo turns toward the camera and smiles as the wind lifts her hair.",
 "files": [{"field": "image", "url": "https://example.com/portrait.jpg"}]}

// First and last frame, inline
{"task": "image", "prompt": "The empty room fills with furniture as the camera slowly pushes in.",
 "files": [{"field": "image", "base64": "<base64 of first.png>", "filename": "first.png"},
           {"field": "last_frame", "base64": "<base64 of last.png>", "filename": "last.png"}]}

// Reference to video: a character and a voice
{"task": "reference",
 "prompt": "Show <Picture 1> skateboarding down a sunlit street, the camera follows from the side, <Audio 1> with the sound of wheels rolling on asphalt, cinematic 24fps handheld shot.",
 "files": [{"field": "ref_images", "url": "https://example.com/skater.png"},
           {"field": "ref_audios", "url": "https://example.com/voice.wav"}]}

// Plain description, rewritten by the LLM (needs MIMO_API_KEY on the endpoint)
{"prompt": "two wrestlers in a gym, one chokeslams the other onto a crash mat, heavy metal soundtrack", "rewrite": true}

// A still image from another workflow
{"workflow": "flux2klein9b-txt2img", "prompt": "A cozy reading nook with a cat asleep on a stack of books, warm lamp light"}

// Output to your own bucket instead of inline
{"prompt": "Ocean waves roll onto a quiet beach at golden hour.", "upload_url": "https://<bucket>.r2.cloudflarestorage.com/renders/waves.mp4?X-Amz-..."}
```

To build a base64 file item from the shell: `jq -n --arg b64 "$(base64 < first.png)" '{field: "image", base64: $b64, filename: "first.png"}'`. The worker ignores line breaks in base64.

For `upload_url`, the worker `PUT`s the file with `Content-Type` set to the output's type (`video/mp4` for H3), so presign for that type. Any S3-compatible store works; for example, with boto3 against R2:

```python
import boto3

s3 = boto3.client("s3", endpoint_url="https://<account>.r2.cloudflarestorage.com")
upload_url = s3.generate_presigned_url("put_object", ExpiresIn=3600,
                                       Params={"Bucket": "renders", "Key": "waves.mp4", "ContentType": "video/mp4"})
```

#### The result

RunPod's `/runsync` or `/status/<id>` for a finished job (vast.ai returns the same object as `{"result": ...}`):

```json
{
  "id": "d7efb7a5-18b9-4e73-9e59-9c75e1078e7c-e2",
  "status": "COMPLETED",
  "delayTime": 65598,
  "executionTime": 108305,
  "workerId": "ytuk0vik4wxvlw",
  "output": {
    "job_id": "17bfa0db-dcac-4a51-97a4-1ee9e5568578",
    "content_type": "video/mp4",
    "filename": "minimaxh3_00001_.mp4",
    "bytes": 398263,
    "seconds": 107.9,
    "provenance": {
      "mode": "T2VA",
      "seed": 7,
      "models": ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
      "settings": {"width": 864, "height": 480, "duration": 2, "steps": 20, "scheduler": "beta"},
      "prompt_source": "raw_prompt",
      "note": "raw_prompt used verbatim; no LLM rewrite"
    },
    "base64": "AAAAIGZ0eXBpc29tAAACAGlzb21pc28y..."
  }
}
```

With `upload_url` the output has `"uploaded": true` and no `base64`. RunPod keeps results small, so an output over 15 MB needs `upload_url`. `delayTime` (queue plus worker start) and `executionTime` are milliseconds; `seconds` is the render as the worker timed it. `provenance` records what actually ran; send its `seed` and settings back to reproduce a clip.

While a job runs, `/status/<id>` has `"status": "IN_PROGRESS"` and the render's progress as `output`:

```json
{"status": "in_progress", "percent": 64, "step": {"value": 12, "max": 20}, "node": "SamplerCustomAdvanced"}
```

A job that cannot be rendered fails with the wrapper's reason, for example on RunPod:

```json
{"status": "FAILED", "error": "the render was refused (HTTP 400): Unknown workflow — Available workflows: flux2klein9b, flux2klein9b-txt2img, ideogram4, minimaxh3, qwenimage21, qwenimage21-txt2img"}
```

#### RunPod API

All routes are under `https://api.runpod.ai/v2/<endpoint_id>` with `Authorization: Bearer <RUNPOD_API_KEY>`:

| Route | Use |
|---|---|
| `POST /runsync` `{"input": job}` | Waits about 90 s, then answers with the result or with `id` + `IN_QUEUE` / `IN_PROGRESS`. The result is kept for 1 minute. |
| `POST /run` `{"input": job, "webhook": "https://..."}` | Answers `{"id", "status": "IN_QUEUE"}` at once. The result is kept for 30 minutes. With `webhook`, RunPod also POSTs the final status object there. |
| `GET /status/<id>` | `IN_QUEUE`, `IN_PROGRESS` (with progress), then `COMPLETED`, `FAILED`, `CANCELLED` or `TIMED_OUT`. |
| `POST /cancel/<id>` | Cancels a queued or running job. |
| `GET /health` | Jobs by state and workers by state. `throttled` workers mean RunPod has no GPU of the allowed types free in the endpoint's data center at the moment; queued jobs start once one frees up. |

A shell loop around `/run`:

```bash
id=$(curl -s "https://api.runpod.ai/v2/$ENDPOINT_ID/run" -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' -d '{"input": {"prompt": "A lighthouse keeper climbs a spiral staircase at night."}}' | jq -r .id)
while :; do
  curl -s "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$id" -H "Authorization: Bearer $RUNPOD_API_KEY" > status.json
  status=$(jq -r .status status.json)
  echo "$status $(jq -r '.output.percent // empty' status.json)"
  case $status in COMPLETED | FAILED | CANCELLED | TIMED_OUT) break ;; esac
  sleep 5
done
jq -r .output.base64 status.json | base64 --decode > lighthouse.mp4
```

Python (`requests`):

```python
import base64
import os
import time

import requests

API = f"https://api.runpod.ai/v2/{os.environ['ENDPOINT_ID']}"
HEADERS = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}


def render(job: dict, poll_seconds: float = 5) -> dict:
    """Run one job and return its output; raises with the worker's reason if it fails."""
    job_id = requests.post(f"{API}/run", json={"input": job}, headers=HEADERS, timeout=30).json()["id"]
    while True:
        status = requests.get(f"{API}/status/{job_id}", headers=HEADERS, timeout=30).json()
        if status["status"] == "COMPLETED":
            return status["output"]
        if status["status"] in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(status.get("error") or status["status"])
        if status.get("output"):
            print(f"{status['output'].get('percent')}% {status['output'].get('node')}")
        time.sleep(poll_seconds)


output = render({"prompt": "A red fox trots across fresh snow at sunrise.", "fields": {"duration": 3, "seed": 7}})
with open("fox.mp4", "wb") as f:
    f.write(base64.b64decode(output["base64"]))
print(output["seconds"], "s", output["provenance"]["settings"])
```

TypeScript (Node 18+, `fetch`):

```ts
import { writeFile } from "node:fs/promises";

const API = `https://api.runpod.ai/v2/${process.env.ENDPOINT_ID}`;
const headers = { Authorization: `Bearer ${process.env.RUNPOD_API_KEY}`, "Content-Type": "application/json" };

type RenderOutput = {
  job_id: string; content_type: string; filename: string; bytes: number; seconds: number;
  provenance: Record<string, unknown>; base64?: string; uploaded?: boolean;
};

export async function render(job: Record<string, unknown>, pollMs = 5000): Promise<RenderOutput> {
  const { id } = await (await fetch(`${API}/run`, { method: "POST", headers, body: JSON.stringify({ input: job }) })).json();
  for (;;) {
    const status = await (await fetch(`${API}/status/${id}`, { headers })).json();
    if (status.status === "COMPLETED") return status.output;
    if (["FAILED", "CANCELLED", "TIMED_OUT"].includes(status.status)) throw new Error(status.error ?? status.status);
    await new Promise((resolve) => setTimeout(resolve, pollMs));
  }
}

const output = await render({ prompt: "A red fox trots across fresh snow at sunrise.", fields: { duration: 3 } });
await writeFile("fox.mp4", Buffer.from(output.base64!, "base64"));
```

#### vast.ai API

vast.ai has no job queue of its own. A client asks vast's router (`https://run.vast.ai/route/`) for a ready worker, then POSTs `{"auth_data": <the router's reply>, "payload": job}` to that worker's `/generate` over TLS. The worker checks the router's signature, renders, and answers `{"result": <the result>}` (errors as `{"result": {"error": "..."}}`). The `vastai` SDK does all of that, including waiting while the endpoint starts a worker:

```python
import asyncio
import base64

from vastai import Serverless  # pip install vastai


async def main():
    async with Serverless() as client:  # reads VAST_API_KEY
        endpoint = await client.get_endpoint(name="comfyui-serverless-h3")
        reply = await endpoint.request("/generate", {"prompt": "A red fox trots across fresh snow at sunrise.",
                                                     "fields": {"duration": 3}})
        result = reply["response"]["result"]
        if "error" in result:
            raise RuntimeError(result["error"])
        with open("fox.mp4", "wb") as f:
            f.write(base64.b64decode(result["base64"]))


asyncio.run(main())
```

The SDK gives a worker 600 s to answer, which covers H3 clips at the default settings; send longer jobs with an `upload_url`. The same flow by hand (the router answers without a `url` while no worker is ready; ask again after a few seconds):

```bash
curl -s -o jvastai_root.cer https://console.vast.ai/static/jvastai_root.cer   # vast's CA for worker certificates
route=$(curl -s https://run.vast.ai/route/ -H "Authorization: Bearer $VAST_API_KEY" -H 'Content-Type: application/json' \
  -d "{\"endpoint\": \"comfyui-serverless-h3\", \"api_key\": \"$VAST_API_KEY\", \"cost\": 100}")
jq -n --argjson auth "$route" '{auth_data: $auth, payload: {prompt: "A red fox trots across fresh snow at sunrise."}}' \
  | curl -s --cacert jvastai_root.cer "$(jq -r .url <<<"$route")/generate" -H 'Content-Type: application/json' -d @- > reply.json
jq -r .result.base64 reply.json | base64 --decode > fox.mp4
```

A render longer than one request should stay open (or one that must survive a client restart) runs asynchronously on one worker, pinned by a vast session. Route once, then send to that worker's URL, each body carrying the router's reply as `auth_data`:

| Worker route | Body | Answer |
|---|---|---|
| `POST /session/create` | `{"auth_data", "payload": {"lifetime": 900}}` | `{"session_id"}`; every later request extends the session by `lifetime` |
| `POST /submit` | `{"auth_data", "session_id", "payload": job}` | `{"result": {"job_id"}}`, or `{"result": {"error"}}` for a job the wrapper refuses |
| `POST /status` | `{"auth_data", "session_id", "payload": {"job_id", "upload_url"?}}` | `{"result": {"status": "pending" \| "in_progress" \| "completed" \| "failed" \| "cancelled", "progress"?, "output"?, "error"?}}` |
| `POST /cancel` | `{"auth_data", "session_id", "payload": {"job_id"}}` | `{"result": {"cancelled": true}}` |
| `POST /session/end` | `{"auth_data", "session_id"}` | |

`output` is the same result as above and is reported once; the worker then deletes its copy. A `410` means the worker restarted and lost the session: submit again. These routes carry no workload of their own, so they never queue behind a `/generate`.

#### Deploying

**RunPod.** A serverless template on `docker.io/shivanshtalwar0/comfyui:serverless-runpod` (pin a `serverless-runpod-sha-<short>` tag or a digest so workers roll deliberately), 20 GB container disk. A queue endpoint on it:

```json
{"name": "comfyui-serverless-h3", "templateId": "<template id>", "networkVolumeId": "<volume id>",
 "gpuTypeIds": ["NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"],
 "minCudaVersion": "12.8", "workersMin": 0, "workersMax": 2, "idleTimeout": 60, "executionTimeoutMs": 3600000, "flashboot": true}
```

(`POST https://rest.runpod.io/v1/endpoints`.) The network volume (80 GB holds every H3 task plus Qwen Image 2.1) is mounted at `/runpod-volume` and becomes the data dir, in the layout the pods use on `/workspace`. Without one, every new worker downloads the checkpoints onto its container disk (about 43 GB for H3, 17 GB for Qwen Image 2.1), so give it 80 GB instead of 20. The first job on an empty volume downloads that task's checkpoints onto it, and every later worker reuses them. The volume pins the endpoint to its data center, so list every GPU type that can run H3 there: a data center out of those GPUs shows `throttled` workers until one frees up. `workersMin: 1` keeps one worker warm (billed while idle); `idleTimeout` is how long a worker stays up after its last job. Put `MIMO_API_KEY` (for `rewrite`) and `HF_TOKEN` (for gated FLUX.2 weights) in the template's env.

**vast.ai.** A private template on `shivanshtalwar0/comfyui:serverless-vast` in entrypoint mode (neither `--ssh` nor `--jupyter`) with `-p 3000:3000` for the PyWorker and a disk that holds the weights. A workergroup without a template starts its instances in SSH mode, which replaces the image's entrypoint, so the PyWorker never runs. The weights come either from R2, through the vast.ai pod env (`COMFY_MODELS_S3_*` + R2 credentials; keep the template private), or from Hugging Face (`COMFY_MODELS_PREFETCH=1 PREFETCH_MODELS=minimaxh3:int8`). Then an endpoint and a workergroup:

```bash
vastai create template --name comfyui-serverless-vast --image shivanshtalwar0/comfyui --image_tag serverless-vast \
  --env '-p 3000:3000 -e COMFY_MODELS_PREFETCH=1 -e PREFETCH_MODELS=minimaxh3:int8' --disk_space 100 \
  --search_params 'gpu_name=RTX_5090 num_gpus=1 verified=true rentable=true direct_port_count>=2 cuda_max_good>=12.8 disk_space>=100 reliability>=0.98 inet_down>=1000 cpu_ram>=60 disk_bw>=1500 dlperf>=150 storage_cost<=0.2'
vastai create endpoint --endpoint_name comfyui-serverless-h3 --cold_workers 1 --max_workers 2 --min_load 0 --inactivity_timeout 600
vastai create workergroup --endpoint_name comfyui-serverless-h3 --template_hash <hash from create template>
```

At boot a vast.ai worker has vast sign its TLS certificate, copies the weights, starts ComfyUI, and renders the benchmark job twice (a warm-up that loads the models, then a timed run that tells vast how fast the worker is) before it takes traffic. A stopped (cold) worker keeps its weights and its benchmark result, so it restarts much faster. An endpoint that serves another workflow sets `SERVERLESS_BENCHMARK_JOB` to a job of that workflow (JSON). Creating an endpoint needs $5 of vast.ai credit.

A cold worker still pays for its disk. Hosts price disk from about $0.07 to $1.00 per GB per month, so cap it in the search (`storage_cost<=0.2`: 80 GB costs at most $16 a month). `--cold_workers 0` costs nothing idle, but every render after a quiet spell then waits for a new worker: about 7 minutes to copy the 42.5 GB of H3 weights, plus the two benchmark renders. Hosts also vary a lot in speed beyond the GPU. One RTX 5090 host whose container was limited to 43.5 GB of RAM (less than the weights) rendered a 2 s clip at 2.5 s a step but a 5 s clip at 26 s a step, several times slower than RunPod's 5090s: the weights no longer fit in RAM next to the activations. The floors above (`inet_down`, `cpu_ram`, `disk_bw`, `dlperf`) are the ones FloStudio's vast.ai pods use; that host fails `inet_down>=1000`.

| Worker env | Default | Meaning |
|---|---|---|
| `SERVERLESS_BOOT_TIMEOUT` | `1800` | Seconds a worker may take to boot (weights copy included) before it gives up. |
| `SERVERLESS_BENCHMARK_JOB` | an H3 text job | vast.ai: the job the worker benchmarks with. |
| `COMFY_MINIMAX_H3_MAX_FRAMES`, `COMFY_MINIMAX_H3_MAX_PIXEL_FRAMES` | `124`, `1152*640*124` | The H3 length and canvas×frames caps, for workers with more RAM than the 32 GB box they were measured on. |
| `MIMO_API_KEY`, `HF_TOKEN` | unset | `rewrite`; gated FLUX.2 [klein] weights. |
| `COMFYUI_ARGS`, `VRAM_HEADROOM_GB`, ... | | As in the [Docker image](#docker-image-local-gpu-box-runpod-and-vastai) table. |

## Features
- A visual node graph for building and reusing image, video, audio, 3D, and text workflows without code.
- Reusable subgraphs, workflow templates, App Mode, and a local API for integrating workflows into applications.
- Efficient local execution with asynchronous queueing, partial graph re-execution, smart VRAM and RAM management, model offloading, and support for quantized models.
- Broad native model support. This is a representative list; browse the [workflow library](https://comfy.org/workflows/) for maintained, ready-to-run templates.
  - [Image generation](https://comfy.org/workflows/tag/text-to-image/): Stable Diffusion 1.5, SDXL, SD3.5, Flux.1, Flux.2, Qwen Image, Z-Image, Hunyuan Image 2.1, HiDream, Lumina Image 2.0, Chroma, Anima, LongCat Image, Ideogram 4, Krea 2, MageFlow, Microsoft Lens, PixelDiT, Kandinsky 5, and Ernie Image.
  - [Image editing](https://comfy.org/workflows/tag/image-edit/): Flux Kontext, Flux.2 Klein, Qwen Image Edit, HiDream E1.1 and O1, OmniGen2, Boogu, JoyImage Edit, MageFlow Edit, and LongCat Image Edit.
  - [Video generation](https://comfy.org/workflows/tag/video-generation/): Wan 2.1 and 2.2, LTX-Video 2 and 2.3, HunyuanVideo 1.5, Kandinsky 5 Video, CogVideoX, Cosmos Predict2, Bernini-R, SCAIL 2, and Mochi.
  - [Audio and video generation](https://comfy.org/workflows/): MiniMax H3 and LTX-AV.
  - [Audio generation](https://comfy.org/workflows/tag/text-to-audio/): ACE-Step 1.5, Stable Audio 3, MiniMax Music 3 and Yue 2.
  - [3D and vision](https://comfy.org/workflows/): Hunyuan3D 2.1, TripoSplat, SeedVR2, SUPIR, Depth Anything 3, MoGe, SAM 3 and 3.1, RT-DETRv4, and BiRefNet.
  - [Text generation](https://comfy.org/workflows/tag/text-generation/): Gemma 3 and 4, Qwen3, Qwen3.5, and Qwen3-VL, including multimodal inputs.
- Load complete checkpoints or separate diffusion models, VAEs, text encoders, LoRAs, ControlNets, adapters, and upscalers from supported model formats.
- Built-in tools for inpainting, outpainting, reference conditioning, masks and compositing, model merging, upscaling, frame interpolation, segmentation, depth estimation, and media processing.
- Save and load workflows as JSON, or recover complete workflows and seeds from supported generated media.
- Runs fully offline: core does not download anything unless you request it. Use `--disable-api-nodes` to disable the optional paid [Comfy API nodes](https://docs.comfy.org/tutorials/api-nodes/overview) and force all built-in functionality to stay offline.
- Extend ComfyUI with custom nodes
- Configure additional model locations with [`extra_model_paths.yaml`](extra_model_paths.yaml.example).
- Support for saving and loading high bit depth images and videos: 16 bit PNG images, 32 bit EXR, 10 bit AVIF are supported and more.
- Support for saving and loading HDR videos and images in various formats.


## Release Process

ComfyUI follows a weekly release cycle targeting Monday but this regularly changes because of model releases or large changes to the codebase. There are three interconnected repositories:

1. **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**
   - Releases a new major stable version (e.g., v0.7.0) roughly every 2 weeks.
   - Starting from v0.4.0 patch versions will be used for fixes backported onto the current stable release.
   - Minor versions will be used for releases off the master branch.
   - Patch versions may still be used for releases on the master branch in cases where a backport would not make sense.
   - Commits outside of the stable release tags may be very unstable and break many custom nodes.
   - Serves as the foundation for the desktop release

2. **[Comfy Desktop](https://github.com/Comfy-Org/Comfy-Desktop)**
   - Builds a new release using the latest stable core version

3. **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**
   - Every 2+ weeks frontend updates are merged into the core repository
   - Features are frozen for the upcoming core release
   - Development continues for the next release cycle

## Shortcuts

| Keybind                            | Explanation                                                                                                        |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | Queue up current graph for generation                                                                              |
| `Ctrl` + `Shift` + `Enter`              | Queue up current graph as first for generation                                                                     |
| `Ctrl` + `Alt` + `Enter`                | Cancel current generation                                                                                          |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | Undo/Redo                                                                                                          |
| `Ctrl` + `S`                          | Save workflow                                                                                                      |
| `Ctrl` + `O`                          | Load workflow                                                                                                      |
| `Ctrl` + `A`                          | Select all nodes                                                                                                   |
| `Alt `+ `C`                           | Collapse/uncollapse selected nodes                                                                                 |
| `Ctrl` + `M`                          | Mute/unmute selected nodes                                                                                         |
| `Ctrl` + `B`                           | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)            |
| `Delete`/`Backspace`                   | Delete selected nodes                                                                                              |
| `Ctrl` + `Backspace`                   | Delete the current graph                                                                                           |
| `Space`                              | Move the canvas around when held and moving the cursor                                                             |
| `Ctrl`/`Shift` + `Click`                 | Add clicked node to selection                                                                                      |
| `Ctrl` + `C`/`Ctrl` + `V`                  | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                     |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes) |
| `Shift` + `Drag`                       | Move multiple selected nodes at the same time                                                                      |
| `Ctrl` + `D`                           | Load default graph                                                                                                 |
| `Alt` + `+`                          | Canvas Zoom in                                                                                                     |
| `Alt` + `-`                          | Canvas Zoom out                                                                                                    |
| `Ctrl` + `Shift` + LMB + Vertical drag | Canvas Zoom in/out                                                                                                 |
| `P`                                  | Pin/Unpin selected nodes                                                                                           |
| `Ctrl` + `G`                           | Group selected nodes                                                                                               |
| `Q`                                 | Toggle visibility of the queue                                                                                     |
| `H`                                  | Toggle visibility of history                                                                                       |
| `R`                                  | Refresh graph                                                                                                      |
| `F`                                  | Show/Hide menu                                                                                                      |
| `.`                                  | Fit view to selection (Whole graph when nothing is selected)                                                        |
| Double-Click LMB                   | Open node quick search palette                                                                                     |
| `Shift` + Drag                       | Move multiple wires at once                                                                                        |
| `Ctrl` + `Alt` + LMB                   | Disconnect all wires from clicked slot                                                                             |

`Ctrl` can also be replaced with `Cmd` instead for macOS users

# Installing

## Windows and Mac

We highly recommend using the [desktop app](https://comfy.org/download):

### [Link to Download](https://comfy.org/download)

The desktop app is the easiest and best way to use ComfyUI for new users.

## Windows Portable

There is a portable standalone build for Windows that should work for running on Nvidia GPUs or for running on your CPU only. It is not recommended for regular users. Regular users should use the desktop app above.

[Direct link to download (nvidia)](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

Simply download, extract with [7-Zip](https://7-zip.org) or with the windows explorer on recent windows versions and run. For smaller models you normally only need to put the checkpoints (the huge ckpt/safetensors files) in: ComfyUI\models\checkpoints but many of the larger models have multiple files. Make sure to follow the instructions to know which subfolder to put them in ComfyUI\models\

If you have trouble extracting it, right click the file -> properties -> unblock

The portable above currently comes with python 3.13 and pytorch cuda 13.0. Update your Nvidia drivers if it doesn't start.

#### All Official Portable Downloads:

[Portable for AMD GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_amd.7z)

[Portable for Intel GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_intel.7z)

[Portable for Nvidia GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z) (supports 20 series and above).

[Portable for Nvidia GPUs with pytorch cuda 12.6 and python 3.12](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia_cu126.7z) (Supports Nvidia 10 series and older GPUs, DO NOT USE THIS ON NEWER 20 SERIES AND ABOVE GPUS).

#### How do I share models between another UI and ComfyUI?

See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. In the standalone windows build you can find this file in the ComfyUI directory. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.


## [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

You can install and start ComfyUI using comfy-cli:
```bash
pip install comfy-cli
comfy install
```

## Manual Install (Windows, Linux)

Python 3.14 works but some custom nodes may have issues. The free threaded variant works but some dependencies will enable the GIL so it's not fully supported.

Python 3.13 is very well supported. If you have trouble with some custom node dependencies on 3.13 you can try 3.12

torch 2.7 is minimally supported but using a newer version is extremely recommended. Using a cu130 or above version of pytorch is required on Nvidia 20 series and above. Some features and optimizations might only work on newer versions. We generally recommend using the latest major version of pytorch with the latest cuda version unless it is less than 2 weeks old. If your pytorch is more than 6 months old, please update it.

### Instructions:

Git clone this repo.

Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints

Put your VAE in: models/vae


### AMD GPUs (Linux)

AMD users can install rocm and pytorch with pip if you don't have it already installed, this is the command to install the stable version:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2```

This is the command to install the nightly with ROCm 7.2 which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2```


### AMD GPUs (Windows, ROCm 10.0)

Use AMD's [multi-architecture PyTorch packages](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). The `device-*` extras install your GPU's kernels and the matching ROCm runtime automatically; a separate HIP SDK installation is not needed.

Use Windows 11, a current [AMD graphics driver](https://www.amd.com/en/support/download/drivers.html), and 64-bit Python 3.13.

The install command below uses `device-all` to install kernels for all supported GPUs. To reduce download size and disk usage, optionally replace **both** occurrences of `device-all` with the target for your GPU:

| GPU | Device extra |
| --- | --- |
| RX 9070 / XT, Radeon AI PRO R9700 | `device-gfx1201` |
| RX 9060 / XT | `device-gfx1200` |
| RX 7900 XT / XTX | `device-gfx1100` |
| RX 7700 XT / 7800 XT | `device-gfx1101` |
| RX 7600 / XT | `device-gfx1102` |
| Ryzen AI Max / Max+ (Strix Halo) | `device-gfx1151` |

**Note:** This table only lists examples. A GPU missing from it may still be supported: supported architectures include RDNA 2, RDNA 3, RDNA 3.5, and RDNA 4. Keep `device-all` to install kernels for all supported targets. For other models, see AMD's [GPU target table](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#gfx-target-lookup-table) and [ROCm compatibility matrix](https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html).

**ROCm 10.0.0 with PyTorch 2.13:**

```bat
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-all]==2.13.0+rocm10.0.0" "torchvision[device-all]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
```

### Intel GPUs (Windows and Linux)

Intel Arc GPU users can install native PyTorch with torch.xpu support using pip. More information can be found [here](https://pytorch.org/docs/main/notes/get_start_xpu.html)

1. To install PyTorch xpu, use the following command:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu```

This is the command to install the Pytorch xpu nightly which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu```

### NVIDIA

Nvidia users should install stable pytorch using this command:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130```

This is the command to install pytorch nightly instead which might have performance improvements.

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132```

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:

```pip uninstall torch```

And install it again with the command above.

### Dependencies

Install the dependencies by opening your terminal inside the ComfyUI folder and:

```pip install -r requirements.txt```

After this you should have everything installed and can proceed to running ComfyUI.

### Others:

#### Apple Mac silicon

You can install ComfyUI in Apple Mac silicon (M1, M2, M3 or M4) with any recent macOS version.

1. Install pytorch nightly. For instructions, read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide (make sure to install the latest pytorch nightly).
1. Follow the [ComfyUI manual installation](#manual-install-windows-linux) instructions for Windows and Linux.
1. Install the ComfyUI [dependencies](#dependencies). If you have another Stable Diffusion UI [you might be able to reuse the dependencies](#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies).
1. Launch ComfyUI by running `python main.py`

> **Note**: Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy folders, as discussed in [ComfyUI manual installation](#manual-install-windows-linux).

#### Ascend NPUs

For models compatible with Ascend Extension for PyTorch (torch_npu). To get started, ensure your environment meets the prerequisites outlined on the [installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) page. Here's a step-by-step guide tailored to your platform and installation method:

1. Begin by installing the recommended or newer kernel version for Linux as specified in the Installation page of torch-npu, if necessary.
2. Proceed with the installation of Ascend Basekit, which includes the driver, firmware, and CANN, following the instructions provided for your specific platform.
3. Next, install the necessary packages for torch-npu by adhering to the platform-specific instructions on the [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch) page.
4. Finally, adhere to the [ComfyUI manual installation](#manual-install-windows-linux) guide for Linux. Once all components are installed, you can run ComfyUI as described earlier.

#### Cambricon MLUs

For models compatible with Cambricon Extension for PyTorch (torch_mlu). Here's a step-by-step guide tailored to your platform and installation method:

1. Install the Cambricon CNToolkit by adhering to the platform-specific instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
2. Next, install the PyTorch(torch_mlu) following the instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html)
3. Launch ComfyUI by running `python main.py`

#### Iluvatar Corex

For models compatible with Iluvatar Extension for PyTorch. Here's a step-by-step guide tailored to your platform and installation method:

1. Install the Iluvatar Corex Toolkit by adhering to the platform-specific instructions on the [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536)
2. Launch ComfyUI by running `python main.py`


## [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager/tree/manager-v4)

**ComfyUI-Manager** is an extension that allows you to easily install, update, and manage custom nodes for ComfyUI.

### Setup

1. Install the manager dependencies:
   ```bash
   pip install -r manager_requirements.txt
   ```

2. Enable the manager with the `--enable-manager` flag when running ComfyUI:
   ```bash
   python main.py --enable-manager
   ```

### Command Line Options

| Flag | Description |
|------|-------------|
| `--enable-manager` | Enable ComfyUI-Manager |
| `--enable-manager-legacy-ui` | Use the legacy manager UI instead of the new UI (implies `--enable-manager`) |
| `--disable-manager-ui` | Disable the manager UI and endpoints while keeping background features like security checks and scheduled installation completion (requires `--enable-manager`) |


# Running

```python main.py```

### For AMD cards not officially supported by ROCm

Try running it with this command if you have issues:

For 6700, 6600 and maybe other RDNA2 or older: ```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

For AMD 7600 and maybe other RDNA3 cards: ```HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py```

### AMD ROCm Tips

You can try setting this env variable `PYTORCH_TUNABLEOP_ENABLED=1` which might speed things up at the cost of a very slow initial run.

# Notes

Only parts of the graph that have an output with all the correct inputs will be executed.

Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.

Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.

You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).

You can use {day|night}, for wildcard/dynamic prompts. With this syntax "{wild|card|test}" will be randomly replaced by either "wild", "card" or "test" by the frontend every time you queue the prompt. To use {} characters in your actual prompt escape them like: \\{ or \\}.

Dynamic prompts also support C-style comments, like `// comment` or `/* comment */`.

To use a textual inversion concepts/embeddings in a text prompt put them in the models/embeddings directory and use them in the CLIPTextEncode node like this (you can omit the .pt extension):

```embedding:embedding_filename.pt```


## How to show high-quality previews?

Use ```--preview-method auto``` to enable previews.

The default installation includes a fast latent preview method that's low-resolution. To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download the [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth](https://github.com/madebyollin/taesd/) and place them in the `models/vae_approx` folder. Once they're installed, restart ComfyUI and launch it with `--preview-method taesd` to enable high-quality previews.

## How to use TLS/SSL?
Generate a self-signed certificate (not appropriate for shared/production use) and key by running the command: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL, the app will now be accessible with `https://...` instead of `http://...`.

> Note: Windows users can use [alexisrolland/docker-openssl](https://github.com/alexisrolland/docker-openssl) or one of the [3rd party binary distributions](https://wiki.openssl.org/index.php/Binaries) to run the command example above.
<br/><br/>If you use a container, note that the volume mount `-v` can be a relative path so `... -v ".\:/openssl-certs" ...` would create the key & cert files in the current directory of your command prompt or powershell terminal.

## Support and dev channel

[Discord](https://comfy.org/discord): Try the #help or #feedback channels.

[Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) (it's like discord but open source).

See also: [https://www.comfy.org/](https://www.comfy.org/)

> _psst — we're hiring!_ Help build ComfyUI: [comfy.org/careers](https://www.comfy.org/careers)

## Frontend Development

As of August 15, 2024, we have transitioned to a new frontend, which is now hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend). The compiled JS files (from TS/Vue) are published to [pypi](https://pypi.org/project/comfyui-frontend-package) and installed as a dependency in ComfyUI.

### Reporting Issues and Requesting Features

For any bugs, issues, or feature requests related to the frontend, please use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend). This will help us manage and address frontend-specific concerns more efficiently.

### Using the Latest Frontend

The new frontend is now the default for ComfyUI. However, please note:

1. The frontend in the main ComfyUI repository is updated fortnightly.
2. Daily releases are available in the separate frontend repository.

To use the most up-to-date frontend version:

1. For the latest daily release, launch ComfyUI with this command line argument:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@latest
   ```

2. For a specific version, replace `latest` with the desired version number:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
   ```

This approach allows you to easily switch between the stable fortnightly release and the cutting-edge daily updates, or even specific versions for testing purposes.

# QA

### Which GPU should I buy for this?

[See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)
