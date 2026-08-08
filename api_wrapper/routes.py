"""HTTP handlers for the wrapper API.

Endpoints are registered on the PromptServer route table so they live next to
the core API and get the same /api-prefixed aliases. Every workflow gets a
dedicated synchronous endpoint that returns the final artifact (image, video,
audio, 3D asset, ...) as a downloadable file:
- POST   /wrapper/{workflow}/generate    (also /api/wrapper/{workflow}/generate)
- GET    /wrapper/workflows
- GET    /wrapper/jobs/{job_id}          (also /api/wrapper/jobs/{job_id})
- GET    /wrapper/jobs/{job_id}/image
- POST   /wrapper/free
- GET    /wrapper/openapi.json
- GET    /wrapper/docs
"""

import asyncio
import gc
import os
import random
import time
import uuid

from aiohttp import web

import comfy.samplers as comfy_samplers
import execution
import folder_paths
from comfy import model_downloader, model_management
from comfy_execution import jobs as comfy_jobs

from api_wrapper import workflows as wrapper_workflows
from api_wrapper.openapi import WRAPPER_SWAGGER_HTML, spec_with_workflows
from api_wrapper.quantize import convert_fp8_to_nvfp4, fp4_filename

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
UPLOAD_EXTENSIONS = {
    "image": ALLOWED_IMAGE_EXTENSIONS,
    "video": {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"},
    "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"},
}
MAX_UPLOAD_BYTES = {"image": 64 * 1024 * 1024, "video": 512 * 1024 * 1024, "audio": 256 * 1024 * 1024}
OUTPUT_FILE_KEYS = ("images", "videos", "video", "gifs", "audio", "3d", "files")
MIME_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif",
    ".mp4": "video/mp4", ".webm": "video/webm",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".obj": "model/obj", ".glb": "model/gltf-binary", ".gltf": "model/gltf+json",
    ".stl": "model/stl", ".fbx": "model/fbx",
    ".usd": "model/vnd.usd+zip", ".usdz": "model/vnd.usdz+zip",
    ".json": "application/json",
}

_queue_number = 0


class _SetupError(Exception):
    """Raised by a workflow's model setup; carries the fields of the
    wrapper error response (missing model downloads etc.)."""

    def __init__(self, message, details="", missing=None):
        super().__init__(message)
        self.message = message
        self.details = details
        self.missing = missing or []


def _queue_snapshots(prompt_queue):
    """Running/queued items with the sensitive element removed, matching what
    the core /api/jobs endpoints pass to comfy_execution.jobs.get_job."""
    running = [item[:5] for item in prompt_queue.currently_running.values()]
    queued = [item[:5] for item in prompt_queue.queue]
    return running, queued


def free_vram(prompt_queue):
    """Unload all models and empty caches.

    The prompt worker consumes the queue flags after the next job finishes, so
    setting them here frees VRAM as soon as nothing is executing. When the
    queue is idle we free immediately, holding the queue mutex so the worker
    cannot consume the flags and unload concurrently (that race crashed on
    double-unload). When a job is in flight the worker does it right after
    that job completes (unloading mid-job would crash it).
    """
    prompt_queue.set_flag("unload_models", True)
    prompt_queue.set_flag("free_memory", True)
    with prompt_queue.mutex:
        deferred = bool(prompt_queue.currently_running) or bool(prompt_queue.queue)
        if not deferred:
            model_management.unload_all_models()
            gc.collect()
            model_management.soft_empty_cache()
    return {"status": "ok", "deferred": deferred}


def _parse_timeout(raw):
    """timeout form value -> seconds or None for no limit.

    '0' / '' / missing mean wait until the workflow finishes, however long
    that takes (the sync endpoint exists for exactly that). An explicit value
    caps the wait in seconds before returning 504.
    """
    if raw is None or raw == "" or raw == "0":
        return None
    value = int(raw)
    if not 5 <= value <= 86400:
        raise ValueError("timeout must be 0 (no limit) or between 5 and 86400 seconds.")
    return value


async def _wait_for_prompt(prompt_queue, prompt_id, timeout=None, sleep_interval=0.5):
    """Wait until a prompt finishes and return its history entry.

    Returns None when the prompt did not finish: either ``timeout`` seconds
    elapsed or the prompt vanished from the queue without completing (e.g. a
    queued item deleted via the /queue endpoint). Without ``timeout`` there is
    no deadline — only a vanished job ends the wait.
    """
    start = time.time()
    while True:
        history = prompt_queue.get_history(prompt_id=prompt_id)
        if history and prompt_id in history:
            return history[prompt_id]
        if timeout is not None and time.time() - start >= timeout:
            return None
        running = list(prompt_queue.currently_running.values())
        queued = list(prompt_queue.queue)
        if not any(str(item[1]) == prompt_id for item in running + queued):
            return None  # removed from the queue without completing
        await asyncio.sleep(sleep_interval)


def _error_response(message, details="", missing=None, status=400):
    body = {
        "error": {
            "type": "wrapper_api_error",
            "message": message,
            "details": details,
            "extra_info": {},
        },
        "missing": missing or [],
    }
    return web.json_response(body, status=status)


def _collect_output_files(history_entry):
    """Resolve every output file a finished workflow saved to disk."""
    output_dirs = {
        "output": folder_paths.get_output_directory(),
        "input": folder_paths.get_input_directory(),
        "temp": folder_paths.get_temp_directory(),
    }
    paths = []
    for node_outputs in (history_entry.get("outputs") or {}).values():
        for key in OUTPUT_FILE_KEYS:
            for item in node_outputs.get(key, []):
                filename = item.get("filename")
                if not filename:
                    continue
                base = output_dirs.get(item.get("type", "output"), output_dirs["output"])
                path = os.path.join(base, item.get("subfolder", ""), filename)
                if os.path.isfile(path):
                    paths.append(path)
    return paths


async def _download_models(models, downloaded):
    """Download any missing model files (off the event loop). Returns the
    list of failures; successfully downloaded files are appended to
    ``downloaded``."""
    missing = []
    for model in models:
        if folder_paths.get_full_path(model["folder"], model["filename"]):
            continue
        # Downloads can take a while; keep them off the event loop.
        path, error = await asyncio.to_thread(
            model_downloader.download_model, model["folder"], model["filename"], model["url"]
        )
        if path is None:
            missing.append({"folder": model["folder"], "filename": model["filename"], "error": error})
        else:
            downloaded.append(f'{model["folder"]}/{model["filename"]}')
    return missing


def _raise_if_missing(missing):
    if not missing:
        return
    hint = ("Model auto-download is disabled; start ComfyUI with --auto-download-models."
            if any("disabled" in (m.get("error") or "") for m in missing)
            else "Check that Hugging Face login is set up (HF_TOKEN) for gated models.")
    raise _SetupError("Required models could not be downloaded", hint, missing=missing)


async def _setup_flux2klein9b(fields, downloaded):
    """Model setup for the FLUX.2 [klein] 9B workflow: auto-download the fp8
    checkpoint + text encoder + VAE, and convert to fp4 on request (CUDA/CPU;
    MPS falls back to fp8 because nvfp4 needs fp8 block scales)."""
    quantization = fields.get("quantization", "fp8").lower()
    if quantization not in ("fp8", "fp4"):
        raise _SetupError("Invalid quantization", "quantization must be 'fp8' or 'fp4'.")

    _raise_if_missing(await _download_models(wrapper_workflows.FLUX2_KLEIN_9B_MODELS, downloaded))

    unet_name = wrapper_workflows.FLUX2_KLEIN_9B_UNET
    note = None
    if quantization == "fp4":
        if model_management.get_torch_device().type == "mps":
            note = "fp4 is not supported on MPS; using the fp8 model instead."
        else:
            fp4_name = fp4_filename(wrapper_workflows.FLUX2_KLEIN_9B_UNET)
            if folder_paths.get_full_path("diffusion_models", fp4_name) is None:
                src_path = folder_paths.get_full_path("diffusion_models", wrapper_workflows.FLUX2_KLEIN_9B_UNET)
                dst_path = os.path.join(folder_paths.get_folder_paths("diffusion_models")[0], fp4_name)
                try:
                    summary = await asyncio.to_thread(convert_fp8_to_nvfp4, src_path, dst_path)
                except Exception as e:
                    raise _SetupError("FP4 conversion failed", str(e)) from e
                downloaded.append(f"diffusion_models/{fp4_name} (converted, {summary['converted_layers']} layers)")
            unet_name = fp4_name

    return {"unet_name": unet_name}, note


async def _setup_ideogram4(fields, downloaded):
    """Model setup for the Ideogram 4 text-to-image workflow: resolve the
    scheduler preset (mode/steps), validate width/height, and auto-download
    the fp8 or native nvfp4 checkpoint pair + text encoder + VAE. fp4 (nvfp4)
    is not supported on MPS and falls back to fp8 there."""
    mode = fields.get("mode", "default").lower()
    if mode not in wrapper_workflows.IDEOGRAM4_PRESETS:
        raise _SetupError("Invalid mode", "mode must be one of: default, quality, turbo.")
    preset = wrapper_workflows.IDEOGRAM4_PRESETS[mode]
    try:
        steps = int(fields.get("steps", preset["steps"]))
        width = int(fields.get("width", 1024))
        height = int(fields.get("height", 1024))
    except ValueError:
        raise _SetupError("Invalid parameter value", "steps/width/height must be numbers.") from None
    if not 1 <= steps <= 200:
        raise _SetupError("Invalid steps", "steps must be between 1 and 200.")
    if not 256 <= width <= 8192 or not 256 <= height <= 8192:
        raise _SetupError("Invalid size", "width/height must be between 256 and 8192.")

    quantization = fields.get("quantization", "fp8").lower()
    if quantization not in ("fp8", "fp4"):
        raise _SetupError("Invalid quantization", "quantization must be 'fp8' or 'fp4'.")

    use_fp4 = quantization == "fp4" and model_management.get_torch_device().type != "mps"
    note = None
    if quantization == "fp4" and not use_fp4:
        note = "fp4 is not supported on MPS; using the fp8 model instead."
    models = (wrapper_workflows.IDEOGRAM4_FP4_MODELS if use_fp4
              else wrapper_workflows.IDEOGRAM4_MODELS)
    _raise_if_missing(await _download_models(models, downloaded))

    if use_fp4:
        unet_name, unconditional_unet_name = (wrapper_workflows.IDEOGRAM4_UNET_FP4,
                                              wrapper_workflows.IDEOGRAM4_UNET_UNCONDITIONAL_FP4)
    else:
        unet_name, unconditional_unet_name = (wrapper_workflows.IDEOGRAM4_UNET,
                                              wrapper_workflows.IDEOGRAM4_UNET_UNCONDITIONAL)
    return {
        "width": width,
        "height": height,
        "steps": steps,
        "mu": preset["mu"],
        "std": preset["std"],
        "unet_name": unet_name,
        "unconditional_unet_name": unconditional_unet_name,
    }, note


async def _setup_flux2klein9b_txt2img(fields, downloaded):
    """Model setup for the FLUX.2 [klein] 9B text-to-image workflow: same
    models as the image-edit variant, plus width/height validation."""
    build_kwargs, note = await _setup_flux2klein9b(fields, downloaded)
    try:
        width = int(fields.get("width", 1024))
        height = int(fields.get("height", 1024))
    except ValueError:
        raise _SetupError("Invalid parameter value", "width/height must be numbers.") from None
    if not 256 <= width <= 8192 or not 256 <= height <= 8192:
        raise _SetupError("Invalid size", "width/height must be between 256 and 8192.")
    build_kwargs["width"] = width
    build_kwargs["height"] = height
    return build_kwargs, note


async def _setup_minimax_h3(fields, downloaded, ref2va):
    """Shared MiniMax H3 setup: validate the sampler params and pick the model
    set — reusing quantizations already on disk (int8/nvfp4 are the canonical
    template defaults) before auto-downloading anything else."""
    raw_quantization = fields.get("quantization", "").strip().lower() or None
    if raw_quantization is not None and raw_quantization not in wrapper_workflows.MINIMAX_H3_QUANT_MODELS:
        raise _SetupError("Invalid quantization", "quantization must be one of: fp8, int8, bf16, nvfp4.")
    try:
        quantization, note = wrapper_workflows.minimax_h3_quantization_preference(
            raw_quantization, ref2va,
            lambda q: all(folder_paths.get_full_path(m["folder"], m["filename"])
                          for m in wrapper_workflows.minimax_h3_models(q, ref2va=ref2va)))
    except ValueError as e:
        raise _SetupError("Invalid quantization", str(e)) from None
    try:
        steps = int(fields.get("steps", 50))
        width = int(fields.get("width", 1344))
        height = int(fields.get("height", 768))
        duration = float(fields.get("duration", 5.0))
    except ValueError:
        raise _SetupError("Invalid parameter value", "steps/width/height/duration must be numbers.") from None
    if not 1 <= steps <= 1000:
        raise _SetupError("Invalid steps", "steps must be between 1 and 1000.")
    if not 32 <= width <= 8192 or not 32 <= height <= 8192:
        raise _SetupError("Invalid size", "width/height must be between 32 and 8192.")
    width = max(32, (width + 31) // 32 * 32)
    height = max(32, (height + 31) // 32 * 32)
    if not 0.2 <= duration <= 150:
        raise _SetupError("Invalid duration", "duration must be between 0.2 and 150 seconds.")
    scheduler = fields.get("scheduler", "beta")
    if scheduler not in comfy_samplers.SCHEDULER_NAMES:
        raise _SetupError("Invalid scheduler",
                          f"scheduler must be one of: {', '.join(comfy_samplers.SCHEDULER_NAMES)}.")
    ref_image_size = fields.get("ref_image_size", "match")
    if ref_image_size not in ("match", "max"):
        raise _SetupError("Invalid ref_image_size", "ref_image_size must be 'match' or 'max'.")

    _raise_if_missing(await _download_models(
        wrapper_workflows.minimax_h3_models(quantization, ref2va=ref2va), downloaded))

    if model_management.get_torch_device().type == "mps":
        mps_note = ("MiniMax H3 is a very large omni-modal model; on MPS the fp8 weights "
                    "load as bf16 and the run may exceed available memory.")
        note = f"{note} {mps_note}".strip() if note else mps_note
    q = wrapper_workflows.MINIMAX_H3_QUANT_MODELS[quantization]
    kwargs = {
        "steps": steps,
        "width": width,
        "height": height,
        "duration": duration,
        "scheduler": scheduler,
        "unet_name": q["ref2va"] if ref2va else q["unet"],
        "clip_name": q["clip"],
    }
    if ref2va:
        # Only the reference builder consumes this; the text/image builders
        # would reject it as an unexpected keyword argument.
        kwargs["ref_image_size"] = ref_image_size
    return kwargs, note


async def _setup_minimax_h3_text(fields, downloaded):
    return await _setup_minimax_h3(fields, downloaded, ref2va=False)


async def _setup_minimax_h3_image(fields, downloaded):
    return await _setup_minimax_h3(fields, downloaded, ref2va=False)


async def _setup_minimax_h3_reference(fields, downloaded):
    return await _setup_minimax_h3(fields, downloaded, ref2va=True)


_WORKFLOW_SETUPS = {"flux2klein9b": _setup_flux2klein9b,
                    "flux2klein9b-txt2img": _setup_flux2klein9b_txt2img,
                    "ideogram4": _setup_ideogram4,
                    "minimaxh3": {"text": _setup_minimax_h3_text,
                                  "image": _setup_minimax_h3_image,
                                  "reference": _setup_minimax_h3_reference}}


def register_wrapper_routes(routes, prompt_server):
    """Attach the wrapper handlers to a PromptServer route table."""
    prompt_queue = prompt_server.prompt_queue

    # Registration order matters: decorators apply bottom-up, and aiohttp
    # resolves the first matching pattern. The flat route must win for
    # /{workflow}/generate, so it is registered first (bottom decorator).
    @routes.post("/wrapper/{workflow}/{task}/generate")
    @routes.post("/wrapper/{workflow}/generate")
    async def generate(request):
        """Run one workflow (or workflow task) synchronously and return the
        final file. Task workflows get /wrapper/{name}/{task}/generate; flat
        workflows use /wrapper/{name}/generate."""
        global _queue_number

        workflow_name = request.match_info["workflow"].lower()
        task_name = request.match_info.get("task")
        workflow = wrapper_workflows.WORKFLOWS.get(workflow_name)
        if workflow is None:
            return _error_response(
                "Unknown workflow",
                f"Available workflows: {', '.join(sorted(wrapper_workflows.WORKFLOWS))}",
            )
        task = workflow
        if "tasks" in workflow:
            task = workflow["tasks"].get(task_name)
            if task is None:
                return _error_response(
                    "Unknown task",
                    f"Available tasks for {workflow_name}: {', '.join(sorted(workflow['tasks']))}",
                )
        elif task_name is not None:
            return _error_response("Unknown task", f"Workflow {workflow_name} has no task variants.")

        reader = None
        if request.headers.get("Content-Type", "").startswith("multipart/form-data"):
            reader = await request.multipart()
        fields = {}
        uploads = {}
        while reader is not None:
            part = await reader.next()
            if part is None:
                break
            data = await part.read()
            if part.filename:
                uploads.setdefault(part.name, []).append((part.filename, data))
            else:
                fields[part.name] = data.decode("utf-8", "replace").strip()

        prompt = fields.get("prompt", "")
        if not prompt:
            return _error_response("No prompt provided", "The 'prompt' form field is required.")
        if task.get("requires_image") and not uploads.get("image"):
            return _error_response("No image provided", "The 'image' form field with the input image is required.")

        try:
            seed = int(fields["seed"]) if "seed" in fields else random.randrange(0, 2 ** 64)
            steps = int(fields.get("steps", 20))
            cfg = float(fields.get("cfg", 5.0))
            megapixels = float(fields.get("megapixels", 1.0))
        except ValueError:
            return _error_response("Invalid parameter value", "seed/steps/cfg/megapixels must be numbers.")
        try:
            timeout = _parse_timeout(fields.get("timeout"))
        except ValueError as e:
            return _error_response("Invalid timeout", str(e))
        if not 1 <= steps <= 4096:
            return _error_response("Invalid steps", "steps must be between 1 and 4096.")
        if not 0 <= cfg <= 100:
            return _error_response("Invalid cfg", "cfg must be between 0 and 100.")
        if not 0.01 <= megapixels <= 16.0:
            return _error_response("Invalid megapixels", "megapixels must be between 0.01 and 16.")

        # Save uploaded files (validated against the task's upload spec) and
        # map them onto builder parameters.
        upload_spec = task.get("uploads", {})
        upload_refs = {}
        for name, items in uploads.items():
            spec = upload_spec.get(name)
            if spec is None:
                return _error_response("Unexpected upload field", f"No '{name}' upload is accepted by this workflow.")
            if len(items) > spec["max"]:
                return _error_response("Too many files", f"'{name}' accepts at most {spec['max']} file(s).")
            allowed = UPLOAD_EXTENSIONS[spec["ext"]]
            limit = MAX_UPLOAD_BYTES[spec["ext"]]
            wrapper_dir = os.path.join(folder_paths.get_input_directory(), "wrapper")
            os.makedirs(wrapper_dir, exist_ok=True)
            refs = []
            for filename, data in items:
                ext = os.path.splitext(filename or "")[1].lower()
                if ext not in allowed:
                    return _error_response(
                        "Unsupported file format", f"'{name}' must be one of: {sorted(allowed)}")
                if len(data) > limit:
                    return _error_response("File too large", f"'{name}' files must be at most {limit // (1024 * 1024)} MB.")
                saved = f"{uuid.uuid4().hex}{ext}"
                with open(os.path.join(wrapper_dir, saved), "wb") as f:
                    f.write(data)
                refs.append(f"wrapper/{saved}")
            upload_refs[name] = refs

        downloaded = []
        setups = _WORKFLOW_SETUPS[workflow_name]
        setup = setups[task_name] if isinstance(setups, dict) else setups
        try:
            build_kwargs, note = await setup(fields, downloaded)
        except _SetupError as e:
            return _error_response(e.message, e.details, missing=e.missing)

        # Pass only the shared form params the workflow's builder consumes
        # (workflow-specific params arrive via build_kwargs from the setup).
        common_params = {
            "prompt": prompt,
            "negative_prompt": fields.get("negative_prompt", ""),
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "megapixels": megapixels,
        }
        uses = task.get("uses", tuple(common_params))
        build_kwargs.update({k: v for k, v in common_params.items() if k in uses})
        upload_params = task.get("upload_params", {})
        for name, refs in upload_refs.items():
            param = upload_params.get(name, name)
            build_kwargs[param] = refs if upload_spec[name]["max"] != 1 else refs[0]
        build_kwargs["filename_prefix"] = f"wrapper/{workflow_name}"

        prompt_id = str(uuid.uuid4())
        graph = task["build"](**build_kwargs)

        valid = await execution.validate_prompt(prompt_id, graph, None)
        if not valid[0]:
            return _error_response("Workflow validation failed", str(valid[1]))

        extra_data = {"create_time": int(time.time() * 1000)}

        # Per-request VRAM management: auto lets ComfyUI's dynamic VRAM decide
        # (it streams weights to/from RAM so models larger than the GPU don't
        # OOM); low/normal/high force the legacy vram state for this job and
        # restore the previous state once the job finishes.
        vram_state_map = {"low": model_management.VRAMState.LOW_VRAM,
                          "normal": model_management.VRAMState.NORMAL_VRAM,
                          "high": model_management.VRAMState.HIGH_VRAM}
        previous_vram_state = None
        vram_requested = fields.get("vram", "auto").lower()
        if vram_requested != "auto":
            if vram_requested not in vram_state_map:
                return _error_response("Invalid vram", "vram must be one of: auto, low, normal, high.")
            previous_vram_state = model_management.vram_state
            model_management.vram_state = vram_state_map[vram_requested]

        try:
            _queue_number += 1
            prompt_queue.put((_queue_number, prompt_id, graph, extra_data, valid[2], {}))

            # Default: free VRAM as soon as this job finishes. The worker consumes
            # these flags after the next prompt completes and unloads all models
            # plus empty caches. Set free_vram=false to keep models loaded between
            # consecutive jobs (e.g. batch runs).
            if fields.get("free_vram", "true").lower() in ("1", "true", "yes", "on"):
                prompt_queue.set_flag("unload_models", True)
                prompt_queue.set_flag("free_memory", True)

            # Synchronous wait: block this request until the job finishes. By
            # default there is no deadline; an explicit timeout field caps the wait
            # (the returned job_id can then be polled via /jobs/{job_id}).
            history_entry = await _wait_for_prompt(prompt_queue, prompt_id, timeout)
        finally:
            if previous_vram_state is not None:
                model_management.vram_state = previous_vram_state

        if history_entry is None:
            if timeout is None:
                message = "The job was removed from the queue before finishing."
            else:
                message = f"Workflow did not finish within {timeout}s."
            return web.json_response({
                "error": {
                    "type": "timeout" if timeout is not None else "job_removed",
                    "message": message,
                    "details": f"job_id={prompt_id}",
                    "extra_info": {},
                },
                "job_id": prompt_id,
            }, status=504 if timeout is not None else 500)
        status = history_entry.get("status") or {}
        status_str = status.get("status_str") if isinstance(status, dict) else status
        if status_str != "success":
            err = history_entry.get("execution_error") or {}
            return web.json_response({
                "error": {
                    "type": "execution_error",
                    "message": err.get("message", "Workflow execution failed."),
                    "details": f"job_id={prompt_id}",
                    "extra_info": {"status": status_str},
                },
                "job_id": prompt_id,
            }, status=500)

        files = _collect_output_files(history_entry)
        if not files:
            return _error_response("No output produced", "The workflow finished without saving any file.", status=500)

        first = files[0]
        ext = os.path.splitext(first)[1].lower()
        headers = {"Content-Type": MIME_TYPES.get(ext, "application/octet-stream"),
                   "Content-Disposition": f'attachment; filename="{os.path.basename(first)}"'}
        return web.FileResponse(first, headers=headers)

    @routes.get("/wrapper/workflows")
    async def list_workflows(request):
        entries = []
        for name, workflow in wrapper_workflows.WORKFLOWS.items():
            entry = {"name": name, "title": workflow["title"],
                     "requires_image": workflow.get("requires_image", False)}
            if "tasks" in workflow:
                entry["tasks"] = list(workflow["tasks"])
            entries.append(entry)
        return web.json_response({"workflows": entries})

    @routes.get("/wrapper/jobs/{job_id}")
    async def get_job(request):
        try:
            job_id = comfy_jobs.validate_job_id(request.match_info["job_id"])
        except ValueError:
            return _error_response("Invalid job id", "job_id must be a canonical UUID.", status=400)

        job = comfy_jobs.get_job(
            job_id,
            *_queue_snapshots(prompt_queue),
            prompt_queue.get_history(),
        )
        if job is None:
            return _error_response("Job not found", f"No job with id {job_id}", status=404)

        images = []
        for node_outputs in (job.get("outputs") or {}).values():
            images.extend(node_outputs.get("images", []))
        for image in images:
            image["url"] = "/view?filename={}&subfolder={}&type={}".format(
                image["filename"], image.get("subfolder", ""), image.get("type", "output")
            )
        job["images"] = images
        return web.json_response(job)

    @routes.get("/wrapper/jobs/{job_id}/image")
    async def get_job_image(request):
        try:
            job_id = comfy_jobs.validate_job_id(request.match_info["job_id"])
        except ValueError:
            return _error_response("Invalid job id", "job_id must be a canonical UUID.", status=400)

        job = comfy_jobs.get_job(
            job_id,
            *_queue_snapshots(prompt_queue),
            prompt_queue.get_history(),
        )
        if job is None:
            return _error_response("Job not found", f"No job with id {job_id}", status=404)
        if job["status"] != "completed":
            return _error_response("Job not finished", f"Job status is '{job['status']}'.", status=409)

        for node_outputs in (job.get("outputs") or {}).values():
            for image in node_outputs.get("images", []):
                location = "/view?filename={}&subfolder={}&type={}".format(
                    image["filename"], image.get("subfolder", ""), image.get("type", "output")
                )
                raise web.HTTPFound(location)
        return _error_response("Job has no image output", status=404)

    @routes.post("/wrapper/free")
    async def free_memory(request):
        return web.json_response(free_vram(prompt_queue))

    @routes.get("/wrapper/openapi.json")
    async def wrapper_openapi(request):
        spec = spec_with_workflows(wrapper_workflows.WORKFLOWS)
        return web.json_response(spec)

    @routes.get("/wrapper/docs")
    async def wrapper_docs(request):
        return web.Response(text=WRAPPER_SWAGGER_HTML, content_type="text/html")
