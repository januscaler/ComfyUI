import copy
import json
import mimetypes
import os
import re
import tempfile
import time
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask


COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
WORKFLOW_DIR = Path(os.environ.get("COMFYUI_WORKFLOW_DIR", Path(__file__).with_name("workflows"))).resolve()
POLL_INTERVAL_SECONDS = float(os.environ.get("COMFYUI_API_POLL_INTERVAL_SECONDS", "1.0"))
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("COMFYUI_API_TIMEOUT_SECONDS", "900"))
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("COMFYUI_API_REQUEST_TIMEOUT_SECONDS", "30"))

ARTIFACT_KEYS = ("images", "audio", "videos", "gifs", "files", "ui")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


app = FastAPI(
    title="ComfyUI Synchronous Developer API",
    version="1.0.0",
    description=(
        "Blocking REST endpoints for running ComfyUI API workflows. Each generation "
        "request queues a workflow, waits for completion, and returns the generated "
        "artifact directly. Swagger UI is available at /docs."
    ),
    contact={"name": "ComfyUI"},
)


class WorkflowSummary(BaseModel):
    id: str = Field(description="Workflow id used in generation routes.")
    filename: str = Field(description="Workflow JSON filename.")


class WorkflowListResponse(BaseModel):
    workflows: list[WorkflowSummary]


class HealthResponse(BaseModel):
    status: str
    comfyui_base_url: str
    workflow_dir: str


class Artifact(BaseModel):
    filename: str
    subfolder: str = ""
    type: str = "output"


def _comfy_url(path: str) -> str:
    return f"{COMFYUI_BASE_URL}{path}"


def _workflow_id_from_path(path: Path) -> str:
    return path.relative_to(WORKFLOW_DIR).with_suffix("").as_posix()


def _workflow_path(workflow_id: str) -> Path:
    if not workflow_id or not SAFE_ID_RE.match(workflow_id.replace("/", ".")):
        raise HTTPException(status_code=400, detail="workflow_id may only contain letters, numbers, slash, dot, dash, and underscore")

    path = (WORKFLOW_DIR / f"{workflow_id}.json").resolve()
    if WORKFLOW_DIR not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' was not found")
    return path


def _load_workflow(workflow_id: str) -> dict[str, Any]:
    path = _workflow_path(workflow_id)
    with path.open("r", encoding="utf-8") as f:
        workflow = json.load(f)

    if not isinstance(workflow, dict):
        raise HTTPException(status_code=400, detail="Workflow JSON must be an object")
    if "nodes" in workflow and "links" in workflow:
        raise HTTPException(
            status_code=400,
            detail="This endpoint requires ComfyUI API workflow JSON. Use File -> Export (API), not the frontend workflow format.",
        )
    return workflow


def _parse_json_object(raw: str | None, field_name: str) -> dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"{field_name} must be valid JSON: {e.msg}") from e
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON object")
    return data


def _set_node_input(workflow: dict[str, Any], node_id: str, input_name: str, value: Any) -> bool:
    node = workflow.get(str(node_id))
    if not isinstance(node, dict):
        return False
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return False
    inputs[input_name] = value
    return True


def _node_has_text_input(node: Any) -> bool:
    return isinstance(node, dict) and isinstance(node.get("inputs"), dict) and isinstance(node["inputs"].get("text"), str)


def _linked_text_node_id(workflow: dict[str, Any], input_name: str) -> str | None:
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        value = inputs.get(input_name)
        if isinstance(value, list) and value:
            linked_node_id = str(value[0])
            if _node_has_text_input(workflow.get(linked_node_id)):
                return linked_node_id
    return None


def _patch_text_inputs(
    workflow: dict[str, Any],
    prompt: str | None,
    negative_prompt: str | None,
    prompt_node_id: str | None,
    negative_prompt_node_id: str | None,
) -> None:
    if prompt is not None:
        patched = False
        if prompt_node_id:
            patched = _set_node_input(workflow, prompt_node_id, "text", prompt)
        else:
            positive_node_id = _linked_text_node_id(workflow, "positive")
            if positive_node_id:
                patched = _set_node_input(workflow, positive_node_id, "text", prompt)
            else:
                for node_id, node in workflow.items():
                    if _node_has_text_input(node):
                        patched = _set_node_input(workflow, node_id, "text", prompt)
                        break
        if not patched:
            raise HTTPException(status_code=400, detail="Could not find a text input to patch; pass prompt_node_id or overrides")

    if negative_prompt is not None:
        patched = False
        if negative_prompt_node_id:
            patched = _set_node_input(workflow, negative_prompt_node_id, "text", negative_prompt)
        else:
            negative_node_id = _linked_text_node_id(workflow, "negative")
            if negative_node_id:
                patched = _set_node_input(workflow, negative_node_id, "text", negative_prompt)
        if not patched:
            raise HTTPException(status_code=400, detail="Could not find a negative text input to patch; pass negative_prompt_node_id or overrides")


def _patch_media_input(
    workflow: dict[str, Any],
    media_ref: str,
    explicit_node_id: str | None,
    explicit_input_name: str | None,
    class_types: set[str],
    fallback_inputs: tuple[str, ...],
) -> None:
    if explicit_node_id:
        input_name = explicit_input_name or fallback_inputs[0]
        if not _set_node_input(workflow, explicit_node_id, input_name, media_ref):
            raise HTTPException(status_code=400, detail=f"Could not patch node '{explicit_node_id}' input '{input_name}'")
        return

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        class_type = node.get("class_type")
        if class_type not in class_types:
            continue
        for input_name in fallback_inputs:
            if input_name in inputs:
                inputs[input_name] = media_ref
                return

    raise HTTPException(status_code=400, detail="Could not find a media loader to patch; pass image_node_id/video_node_id or overrides")


def _apply_overrides(workflow: dict[str, Any], overrides: dict[str, Any]) -> None:
    for node_id, values in overrides.items():
        node = workflow.get(str(node_id))
        if not isinstance(node, dict):
            raise HTTPException(status_code=400, detail=f"Override references missing node '{node_id}'")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise HTTPException(status_code=400, detail=f"Node '{node_id}' has no inputs to override")
        if not isinstance(values, dict):
            raise HTTPException(status_code=400, detail=f"Override for node '{node_id}' must be an object")
        node_inputs = values.get("inputs", values)
        if not isinstance(node_inputs, dict):
            raise HTTPException(status_code=400, detail=f"Override inputs for node '{node_id}' must be an object")
        inputs.update(node_inputs)


def _safe_upload_name(original_name: str | None, input_id: str) -> str:
    base_name = os.path.basename(original_name or "upload")
    stem, ext = os.path.splitext(base_name)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "upload"
    ext = re.sub(r"[^A-Za-z0-9.]+", "", ext)[:16]
    return f"{input_id}_{stem[:80]}{ext}"


def _upload_media(file: UploadFile, input_id: str) -> str:
    upload_name = _safe_upload_name(file.filename, input_id)
    data = {
        "type": "input",
        "subfolder": f"api/{input_id}",
        "overwrite": "true",
    }
    files = {"image": (upload_name, file.file, file.content_type or "application/octet-stream")}
    try:
        response = requests.post(_comfy_url("/upload/image"), data=data, files=files, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not upload input media to ComfyUI: {e}") from e
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"ComfyUI rejected media upload with status {response.status_code}")

    payload = response.json()
    name = payload.get("name")
    subfolder = payload.get("subfolder", "")
    if not name:
        raise HTTPException(status_code=502, detail="ComfyUI upload response did not include a filename")
    return f"{subfolder}/{name}" if subfolder else name


def _queue_workflow(workflow: dict[str, Any], input_id: str, extra_data: dict[str, Any]) -> None:
    payload = {
        "prompt": workflow,
        "prompt_id": input_id,
        "extra_data": extra_data,
    }
    try:
        response = requests.post(_comfy_url("/prompt"), json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not queue workflow in ComfyUI: {e}") from e
    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)


def _wait_for_history(input_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = requests.get(_comfy_url(f"/history/{input_id}"), timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Could not read ComfyUI history: {e}") from e
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"ComfyUI history returned status {response.status_code}")
        history = response.json()
        if input_id in history:
            result = history[input_id]
            status = result.get("status") or {}
            if status.get("status_str") == "error":
                raise HTTPException(status_code=502, detail={"message": "Workflow execution failed", "status": status})
            return result
        time.sleep(POLL_INTERVAL_SECONDS)
    raise HTTPException(status_code=504, detail=f"Workflow did not finish within {timeout_seconds} seconds")


def _walk_artifacts(value: Any) -> Iterator[Artifact]:
    if isinstance(value, dict):
        if "filename" in value and "type" in value:
            yield Artifact(
                filename=str(value["filename"]),
                subfolder=str(value.get("subfolder") or ""),
                type=str(value.get("type") or "output"),
            )
        for nested in value.values():
            yield from _walk_artifacts(nested)
    elif isinstance(value, list) or isinstance(value, tuple):
        for nested in value:
            yield from _walk_artifacts(nested)


def _extract_artifacts(history_result: dict[str, Any]) -> list[Artifact]:
    outputs = history_result.get("outputs")
    if not isinstance(outputs, dict):
        return []

    artifacts: list[Artifact] = []
    seen: set[tuple[str, str, str]] = set()
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        candidates = {key: node_output[key] for key in ARTIFACT_KEYS if key in node_output}
        if not candidates:
            candidates = node_output
        for artifact in _walk_artifacts(candidates):
            key = (artifact.type, artifact.subfolder, artifact.filename)
            if artifact.type in ("output", "temp") and key not in seen:
                seen.add(key)
                artifacts.append(artifact)
    return artifacts


def _artifact_response_url(artifact: Artifact) -> str:
    return _comfy_url("/view")


def _stream_artifact(artifact: Artifact, input_id: str) -> StreamingResponse:
    params = {
        "filename": artifact.filename,
        "subfolder": artifact.subfolder,
        "type": artifact.type,
    }

    try:
        response = requests.get(_artifact_response_url(artifact), params=params, stream=True, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch generated file from ComfyUI: {e}") from e
    if response.status_code >= 400:
        response.close()
        raise HTTPException(status_code=502, detail=f"ComfyUI could not serve generated file with status {response.status_code}")

    def body() -> Iterator[bytes]:
        try:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    yield chunk
        finally:
            response.close()

    filename = os.path.basename(artifact.filename)
    media_type = response.headers.get("Content-Type") or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-ComfyUI-Prompt-ID": input_id,
    }
    return StreamingResponse(body(), media_type=media_type, headers=headers)


def _zip_artifacts(artifacts: list[Artifact], input_id: str) -> FileResponse:
    temp = tempfile.NamedTemporaryFile(prefix=f"comfyui_{input_id}_", suffix=".zip", delete=False)
    temp.close()
    try:
        with zipfile.ZipFile(temp.name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            used_names: set[str] = set()
            for artifact in artifacts:
                params = {
                    "filename": artifact.filename,
                    "subfolder": artifact.subfolder,
                    "type": artifact.type,
                }
                try:
                    response = requests.get(_artifact_response_url(artifact), params=params, stream=True, timeout=REQUEST_TIMEOUT_SECONDS)
                except requests.RequestException as e:
                    raise HTTPException(status_code=502, detail=f"Could not fetch generated file from ComfyUI: {e}") from e
                with response:
                    if response.status_code >= 400:
                        raise HTTPException(status_code=502, detail=f"ComfyUI could not serve generated file with status {response.status_code}")
                    archive_name = os.path.basename(artifact.filename)
                    if artifact.subfolder:
                        archive_name = f"{artifact.subfolder.strip('/')}/{archive_name}"
                    original_name = archive_name
                    suffix = 1
                    while archive_name in used_names:
                        stem, ext = os.path.splitext(original_name)
                        archive_name = f"{stem}_{suffix}{ext}"
                        suffix += 1
                    used_names.add(archive_name)
                    with archive.open(archive_name, "w") as archive_file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                archive_file.write(chunk)
    except Exception:
        os.unlink(temp.name)
        raise

    filename = f"comfyui_{input_id}.zip"
    headers = {"X-ComfyUI-Prompt-ID": input_id}
    return FileResponse(
        temp.name,
        media_type="application/zip",
        filename=filename,
        headers=headers,
        background=BackgroundTask(lambda path: os.path.exists(path) and os.unlink(path), temp.name),
    )


def _return_artifacts(history_result: dict[str, Any], input_id: str):
    artifacts = _extract_artifacts(history_result)
    if not artifacts:
        raise HTTPException(status_code=502, detail="Workflow completed but did not produce a downloadable file artifact")
    if len(artifacts) == 1:
        return _stream_artifact(artifacts[0], input_id)
    return _zip_artifacts(artifacts, input_id)


def _prepare_workflow(
    workflow_id: str,
    prompt: str | None,
    negative_prompt: str | None,
    prompt_node_id: str | None,
    negative_prompt_node_id: str | None,
    overrides: str | None,
    media_ref: str | None,
    media_kind: str | None,
    media_node_id: str | None,
    media_input_name: str | None,
) -> dict[str, Any]:
    workflow = copy.deepcopy(_load_workflow(workflow_id))
    _patch_text_inputs(workflow, prompt, negative_prompt, prompt_node_id, negative_prompt_node_id)

    if media_ref and media_kind == "image":
        _patch_media_input(workflow, media_ref, media_node_id, media_input_name, {"LoadImage", "LoadImageMask", "LoadImageOutput"}, ("image",))
    elif media_ref and media_kind == "video":
        _patch_media_input(workflow, media_ref, media_node_id, media_input_name, {"LoadVideo"}, ("file",))

    _apply_overrides(workflow, _parse_json_object(overrides, "overrides"))
    return workflow


def _run_sync_workflow(
    workflow_id: str,
    prompt: str | None,
    negative_prompt: str | None,
    input_id: str | None,
    timeout_seconds: int,
    prompt_node_id: str | None,
    negative_prompt_node_id: str | None,
    overrides: str | None,
    extra_data: str | None,
    media_file: UploadFile | None = None,
    media_kind: str | None = None,
    media_node_id: str | None = None,
    media_input_name: str | None = None,
):
    run_id = input_id or str(uuid.uuid4())
    try:
        run_id = str(uuid.UUID(run_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail="input_id must be a valid UUID") from e

    media_ref = _upload_media(media_file, run_id) if media_file is not None else None
    workflow = _prepare_workflow(
        workflow_id,
        prompt,
        negative_prompt,
        prompt_node_id,
        negative_prompt_node_id,
        overrides,
        media_ref,
        media_kind,
        media_node_id,
        media_input_name,
    )
    api_extra_data = _parse_json_object(extra_data, "extra_data")
    api_extra_data.setdefault("sync_developer_api", True)
    api_extra_data.setdefault("workflow_id", workflow_id)

    _queue_workflow(workflow, run_id, api_extra_data)
    history_result = _wait_for_history(run_id, timeout_seconds)
    return _return_artifacts(history_result, run_id)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    try:
        response = requests.get(_comfy_url("/system_stats"), timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"ComfyUI is not reachable: {e}") from e
    return HealthResponse(status="ok", comfyui_base_url=COMFYUI_BASE_URL, workflow_dir=str(WORKFLOW_DIR))


@app.get("/v1/workflows", response_model=WorkflowListResponse, tags=["Workflows"])
def list_workflows() -> WorkflowListResponse:
    if not WORKFLOW_DIR.exists():
        return WorkflowListResponse(workflows=[])

    workflows = [
        WorkflowSummary(id=_workflow_id_from_path(path), filename=path.name)
        for path in sorted(WORKFLOW_DIR.rglob("*.json"))
        if path.is_file()
    ]
    return WorkflowListResponse(workflows=workflows)


@app.post("/v1/text-to-image/{workflow_id}", tags=["Generation"])
def text_to_image(
    workflow_id: str,
    prompt: str = Form(..., description="Positive prompt text."),
    negative_prompt: str | None = Form(None, description="Negative prompt text."),
    input_id: str | None = Form(None, description="Optional UUID to use as the ComfyUI prompt id."),
    timeout_seconds: int = Form(DEFAULT_TIMEOUT_SECONDS, ge=1, description="How long to wait for the workflow to finish."),
    prompt_node_id: str | None = Form(None, description="Optional node id whose text input should receive prompt."),
    negative_prompt_node_id: str | None = Form(None, description="Optional node id whose text input should receive negative_prompt."),
    overrides: str | None = Form(None, description='JSON object of node input overrides, e.g. {"3":{"seed":123}}.'),
    extra_data: str | None = Form(None, description="Optional JSON object sent through ComfyUI extra_data."),
):
    return _run_sync_workflow(workflow_id, prompt, negative_prompt, input_id, timeout_seconds, prompt_node_id, negative_prompt_node_id, overrides, extra_data)


@app.post("/v1/image-to-image/{workflow_id}", tags=["Generation"])
def image_to_image(
    workflow_id: str,
    image: UploadFile = File(..., description="Input image file."),
    prompt: str | None = Form(None, description="Optional positive prompt text."),
    negative_prompt: str | None = Form(None, description="Optional negative prompt text."),
    input_id: str | None = Form(None, description="Optional UUID to use as the ComfyUI prompt id."),
    timeout_seconds: int = Form(DEFAULT_TIMEOUT_SECONDS, ge=1, description="How long to wait for the workflow to finish."),
    image_node_id: str | None = Form(None, description="Optional LoadImage node id to receive the uploaded image."),
    image_input_name: str | None = Form(None, description="Optional image input name. Defaults to image."),
    prompt_node_id: str | None = Form(None, description="Optional node id whose text input should receive prompt."),
    negative_prompt_node_id: str | None = Form(None, description="Optional node id whose text input should receive negative_prompt."),
    overrides: str | None = Form(None, description='JSON object of node input overrides, e.g. {"10":{"denoise":0.45}}.'),
    extra_data: str | None = Form(None, description="Optional JSON object sent through ComfyUI extra_data."),
):
    return _run_sync_workflow(
        workflow_id,
        prompt,
        negative_prompt,
        input_id,
        timeout_seconds,
        prompt_node_id,
        negative_prompt_node_id,
        overrides,
        extra_data,
        media_file=image,
        media_kind="image",
        media_node_id=image_node_id,
        media_input_name=image_input_name,
    )


@app.post("/v1/text-to-video/{workflow_id}", tags=["Generation"])
def text_to_video(
    workflow_id: str,
    prompt: str = Form(..., description="Positive prompt text."),
    negative_prompt: str | None = Form(None, description="Negative prompt text."),
    input_id: str | None = Form(None, description="Optional UUID to use as the ComfyUI prompt id."),
    timeout_seconds: int = Form(DEFAULT_TIMEOUT_SECONDS, ge=1, description="How long to wait for the workflow to finish."),
    prompt_node_id: str | None = Form(None, description="Optional node id whose text input should receive prompt."),
    negative_prompt_node_id: str | None = Form(None, description="Optional node id whose text input should receive negative_prompt."),
    overrides: str | None = Form(None, description='JSON object of node input overrides, e.g. {"12":{"length":81}}.'),
    extra_data: str | None = Form(None, description="Optional JSON object sent through ComfyUI extra_data."),
):
    return _run_sync_workflow(workflow_id, prompt, negative_prompt, input_id, timeout_seconds, prompt_node_id, negative_prompt_node_id, overrides, extra_data)


@app.post("/v1/video-to-video/{workflow_id}", tags=["Generation"])
def video_to_video(
    workflow_id: str,
    video: UploadFile = File(..., description="Input video file."),
    prompt: str | None = Form(None, description="Optional positive prompt text."),
    negative_prompt: str | None = Form(None, description="Optional negative prompt text."),
    input_id: str | None = Form(None, description="Optional UUID to use as the ComfyUI prompt id."),
    timeout_seconds: int = Form(DEFAULT_TIMEOUT_SECONDS, ge=1, description="How long to wait for the workflow to finish."),
    video_node_id: str | None = Form(None, description="Optional LoadVideo node id to receive the uploaded video."),
    video_input_name: str | None = Form(None, description="Optional video loader input name. Defaults to file."),
    prompt_node_id: str | None = Form(None, description="Optional node id whose text input should receive prompt."),
    negative_prompt_node_id: str | None = Form(None, description="Optional node id whose text input should receive negative_prompt."),
    overrides: str | None = Form(None, description='JSON object of node input overrides, e.g. {"15":{"duration":3.0}}.'),
    extra_data: str | None = Form(None, description="Optional JSON object sent through ComfyUI extra_data."),
):
    return _run_sync_workflow(
        workflow_id,
        prompt,
        negative_prompt,
        input_id,
        timeout_seconds,
        prompt_node_id,
        negative_prompt_node_id,
        overrides,
        extra_data,
        media_file=video,
        media_kind="video",
        media_node_id=video_node_id,
        media_input_name=video_input_name,
    )


@app.post("/v1/image-to-video/{workflow_id}", tags=["Generation"])
def image_to_video(
    workflow_id: str,
    image: UploadFile = File(..., description="Input image file."),
    prompt: str | None = Form(None, description="Optional positive prompt text."),
    negative_prompt: str | None = Form(None, description="Optional negative prompt text."),
    input_id: str | None = Form(None, description="Optional UUID to use as the ComfyUI prompt id."),
    timeout_seconds: int = Form(DEFAULT_TIMEOUT_SECONDS, ge=1, description="How long to wait for the workflow to finish."),
    image_node_id: str | None = Form(None, description="Optional LoadImage node id to receive the uploaded image."),
    image_input_name: str | None = Form(None, description="Optional image input name. Defaults to image."),
    prompt_node_id: str | None = Form(None, description="Optional node id whose text input should receive prompt."),
    negative_prompt_node_id: str | None = Form(None, description="Optional node id whose text input should receive negative_prompt."),
    overrides: str | None = Form(None, description='JSON object of node input overrides, e.g. {"22":{"length":81}}.'),
    extra_data: str | None = Form(None, description="Optional JSON object sent through ComfyUI extra_data."),
):
    return _run_sync_workflow(
        workflow_id,
        prompt,
        negative_prompt,
        input_id,
        timeout_seconds,
        prompt_node_id,
        negative_prompt_node_id,
        overrides,
        extra_data,
        media_file=image,
        media_kind="image",
        media_node_id=image_node_id,
        media_input_name=image_input_name,
    )
