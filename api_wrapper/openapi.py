"""OpenAPI spec and Swagger UI for the wrapper API.

The wrapper exposes its own docs so consumers never have to learn the
internal ComfyUI endpoints. Served at /api/wrapper/openapi.json and
/api/wrapper/docs.
"""

WRAPPER_OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "ComfyUI Wrapper API",
        "description": "Simplified generation API on top of ComfyUI. Submit a prompt and an input image, and the wrapper takes care of the internal workflow: it builds the FLUX.2 [klein] 9B image edit graph, downloads any model files that are missing, queues the job on the ComfyUI execution server and tracks it until the generated image is ready. VRAM and RAM are released automatically as soon as a job finishes; POST /api/wrapper/free releases them on demand.",
        "version": "0.1.0",
    },
    "servers": [{"url": "/"}],
    "paths": {
        "/api/wrapper/{workflow}/generate": {
            "post": {
                "summary": "Run a workflow synchronously and download the result",
                "description": "Uploads the input image, ensures the workflow's model files are present (downloading them if missing), runs the workflow to completion and returns the final artifact (image, video, audio, 3D asset, ...) directly as the file download. Model downloads or the fp4 conversion can take a while on first use.",
                "operationId": "generate",
                "parameters": [
                    {"name": "workflow", "in": "path", "required": True, "schema": {"type": "string", "enum": ["flux2klein9b"]}, "description": "Workflow to run; see GET /api/wrapper/workflows."},
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["prompt", "image"],
                                "properties": {
                                    "prompt": {"type": "string", "description": "The edit instruction, e.g. 'make it snow'."},
                                    "image": {"type": "string", "format": "binary", "description": "Input image (png/jpg/webp)."},
                                    "negative_prompt": {"type": "string", "default": "", "description": "Things to avoid in the output."},
                                    "seed": {"type": "integer", "minimum": 0, "description": "Random seed; a random one is used when omitted."},
                                    "steps": {"type": "integer", "minimum": 1, "maximum": 4096, "default": 20},
                                    "cfg": {"type": "number", "minimum": 0, "maximum": 100, "default": 5.0, "description": "CFG scale used by the guider."},
                                    "megapixels": {"type": "number", "minimum": 0.01, "maximum": 16.0, "default": 1.0, "description": "Input image is scaled to roughly this many megapixels before editing."},
                                    "timeout": {"type": "integer", "minimum": 0, "maximum": 86400, "default": 0, "description": "Seconds to wait before returning 504. 0 (default) waits until the workflow finishes, however long that takes; the timeout response includes the job_id to poll via /jobs/{job_id}."},
                                    "free_vram": {"type": "boolean", "default": True, "description": "Release VRAM and RAM as soon as this job finishes (set to false to keep models loaded between jobs, e.g. batch runs)."},
                                    "quantization": {"type": "string", "enum": ["fp8", "fp4"], "default": "fp8", "description": "Weight precision. fp8 uses the shipped fp8 checkpoint (loaded as bf16 on MPS). fp4 converts the checkpoint to NVFP4 (~half the size) for CUDA/CPU; on MPS it automatically falls back to fp8."},
                                    "vram": {"type": "string", "enum": ["auto", "low", "normal", "high"], "default": "auto", "description": "VRAM management for this job: auto (default) lets ComfyUI's dynamic VRAM decide, streaming model weights to/from RAM so a model larger than the GPU never OOMs. low/normal/high force the legacy vram state for this job (advisory while dynamic VRAM is active)."},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "The final output image (image/png), shown inline in the docs.",
                        "content": {
                            "image/png": {"schema": {"type": "string", "format": "binary"}},
                            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
                        },
                    },
                    "400": {
                        "description": "Invalid request: unknown workflow, missing prompt/image, bad parameter value, or a required model could not be downloaded (e.g. gated model without HF_TOKEN).",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
                    },
                    "500": {
                        "description": "The workflow ran but failed (execution error) or produced no output file.",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
                    },
                    "504": {
                        "description": "The workflow did not finish within the timeout; poll /api/wrapper/jobs/{job_id} with the returned id.",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TimeoutResponse"}}},
                    },
                },
            },
        },
        "/api/wrapper/workflows": {
            "get": {
                "summary": "List the available workflows",
                "operationId": "listWorkflows",
                "responses": {
                    "200": {
                        "description": "Workflow names usable in /api/wrapper/{workflow}/generate.",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/WorkflowList"}}},
                    },
                },
            },
        },
        "/api/wrapper/free": {
            "post": {
                "summary": "Free VRAM and RAM used by model execution",
                "description": "Unloads every loaded model (releasing GPU VRAM and RAM) and empties the torch caches. If a job is currently running or queued the free happens automatically right after that job finishes (unloading mid-job would crash it); when idle it happens immediately.",
                "operationId": "free",
                "responses": {
                    "200": {
                        "description": "Free scheduled/completed.",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/FreeResponse"}}},
                    },
                },
            },
        },
        "/api/wrapper/jobs/{job_id}": {
            "get": {
                "summary": "Get the status and result of a generation job",
                "operationId": "getJob",
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}},
                ],
                "responses": {
                    "200": {"description": "Job details.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Job"}}}},
                    "404": {"description": "Unknown job id.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                },
            },
        },
        "/api/wrapper/jobs/{job_id}/image": {
            "get": {
                "summary": "Get the generated image of a completed job",
                "description": "Redirects to the ComfyUI /view endpoint serving the image bytes.",
                "operationId": "getJobImage",
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}},
                ],
                "responses": {
                    "302": {"description": "Redirect to the image."},
                    "404": {"description": "Unknown job id.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                    "409": {"description": "Job has not completed yet.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                },
            },
        },
    },
    "components": {
        "schemas": {
            "WorkflowList": {
                "type": "object",
                "properties": {
                    "workflows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "title": {"type": "string"},
                                "requires_image": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
            "TimeoutResponse": {
                "type": "object",
                "properties": {
                    "error": {"type": "object", "properties": {"type": {"type": "string"}, "message": {"type": "string"}, "details": {"type": "string"}}},
                    "job_id": {"type": "string", "format": "uuid", "description": "Poll /api/wrapper/jobs/{job_id} with this id."},
                },
            },
            "FreeResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["ok"]},
                    "deferred": {"type": "boolean", "description": "True when a job is running/queued and the free happens after it finishes; false means memory was released immediately."},
                },
            },
            "Job": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "format": "uuid"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed", "cancelled"]},
                    "images": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "subfolder": {"type": "string"},
                                "type": {"type": "string"},
                                "url": {"type": "string", "description": "Relative URL to fetch the image."},
                            },
                        },
                        "description": "Generated images (only present once the job completed).",
                    },
                    "execution_error": {"type": "object", "nullable": True, "description": "Details when the job failed."},
                },
            },
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "object", "properties": {"type": {"type": "string"}, "message": {"type": "string"}, "details": {"type": "string"}}},
                    "missing": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"folder": {"type": "string"}, "filename": {"type": "string"}, "error": {"type": "string"}}},
                        "description": "Models that could not be downloaded.",
                    },
                },
            },
        },
    },
}

OUTPUT_MEDIA_TYPES = {"video": "video/mp4", "audio": "audio/wav"}


def _expanded_operation(template_operation, workflow, task, operation_suffix):
    """One concrete generate operation for a workflow/task: unique operationId,
    per-task form fields, required uploads, response media type."""
    import copy

    operation = copy.deepcopy(template_operation)
    operation["parameters"] = [p for p in operation.get("parameters", [])
                               if p.get("name") != "workflow"]
    operation["operationId"] = operation["operationId"] + "".join(
        part.capitalize() for part in operation_suffix.replace("-", "_").split("_"))
    schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    form = task.get("form")
    if form:
        universal = ("timeout", "free_vram", "quantization", "vram")
        schema["properties"] = {k: v for k, v in schema["properties"].items()
                                if k in form or k in universal}
        schema["properties"].update(task.get("extra_form_properties") or {})
        for up_name, up_spec in (task.get("uploads") or {}).items():
            if up_name not in schema["properties"]:
                schema["properties"][up_name] = {
                    "type": "string", "format": "binary",
                    "description": f"Upload ({up_spec['ext']}; up to {up_spec['max']})."}
        schema["required"] = ["prompt"] + (["image"] if task.get("requires_image") else [])
        if task.get("quantization_options"):
            schema["properties"]["quantization"]["enum"] = task["quantization_options"]
    if workflow.get("example_prompt"):
        prompt_prop = schema["properties"]["prompt"]
        prompt_prop["description"] = (
            f"{prompt_prop['description']} This workflow natively understands "
            f"structured JSON prompts; example:\n\n```json\n{workflow['example_prompt']}\n```"
        )
    if task.get("example_prompt"):
        schema["properties"]["prompt"]["description"] = (
            f"{schema['properties']['prompt']['description']} Example:\n\n{task['example_prompt']}"
        )
    media_type = OUTPUT_MEDIA_TYPES.get(workflow.get("output_type"), "image/png")
    operation["responses"]["200"]["content"] = {
        media_type: {"schema": {"type": "string", "format": "binary"}},
        "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
    }
    return operation


def spec_with_workflows(workflows):
    """Return a copy of the spec with one concrete /api/wrapper/{name}/generate
    path per registered workflow (plus /api/wrapper/{name}/{task} paths for
    task workflows), so the docs show the real endpoints."""
    import copy

    spec = copy.deepcopy(WRAPPER_OPENAPI_SPEC)
    paths = spec["paths"]
    template = paths.pop("/api/wrapper/{workflow}/generate")
    template_operation = template["post"]
    for name, workflow in workflows.items():
        tasks = workflow.get("tasks")
        if tasks:
            for task_name, task in tasks.items():
                operation = _expanded_operation(template_operation, workflow, task,
                                                f"{name}_{task_name}")
                paths[f"/api/wrapper/{name}/{task_name}/generate"] = {"post": operation}
        else:
            operation = _expanded_operation(template_operation, workflow, workflow, name)
            paths[f"/api/wrapper/{name}/generate"] = {"post": operation}
    return spec


WRAPPER_SWAGGER_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>ComfyUI Wrapper API Docs</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin:0; background: #fafafa; }
        .topbar { display: none; }
        .swagger-ui .info { margin: 20px 0; }
        .swagger-ui .info .title { font-size: 28px; }
    </style>
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
<script>
window.onload = function() {
  SwaggerUIBundle({
    url: "/api/wrapper/openapi.json",
    dom_id: '#swagger-ui',
    deepLinking: true,
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
    layout: "StandaloneLayout",
    defaultModelsExpandDepth: -1,
    docExpansion: "list",
    filter: true,
    tryItOutEnabled: true,
  })
}
</script>
</body>
</html>"""
