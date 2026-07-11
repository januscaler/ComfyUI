OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "ComfyUI API",
        "description": "REST API for ComfyUI — queue prompts, check status, manage models and history.",
        "version": "0.27.0",
    },
    "servers": [{"url": "/"}],
    "paths": {
        "/prompt": {
            "post": {
                "summary": "Queue a prompt",
                "description": "Submit a workflow for execution.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/PromptRequest"}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Prompt queued",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PromptResponse"}
                            }
                        }
                    },
                    "400": {
                        "description": "Validation error",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}
                    }
                }
            },
            "get": {
                "summary": "Get queue info",
                "description": "Returns current prompt queue (running + pending).",
                "responses": {
                    "200": {
                        "description": "Queue info",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/QueueResponse"}}}
                    }
                }
            }
        },
        "/queue": {
            "get": {
                "summary": "Get full queue",
                "description": "Returns current queue including running and pending items.",
                "responses": {
                    "200": {
                        "description": "Queue data",
                        "content": {"application/json": {"schema": {"type": "object"}}}
                    }
                }
            },
            "post": {
                "summary": "Clear queue",
                "description": "Clear all items from the queue, or delete specific items.",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "clear": {"type": "boolean"},
                                    "delete": {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "OK"}}
            }
        },
        "/history": {
            "get": {
                "summary": "Get execution history",
                "description": "Returns completed/failed prompt history. Optional query: ?prompt_id=UUID or ?max_items=N.",
                "parameters": [
                    {"name": "prompt_id", "in": "query", "schema": {"type": "string"}},
                    {"name": "max_items", "in": "query", "schema": {"type": "integer"}}
                ],
                "responses": {
                    "200": {
                        "description": "History entries",
                        "content": {"application/json": {"schema": {"type": "object"}}}
                    }
                }
            },
            "post": {
                "summary": "Clear history",
                "description": "Clear history entries.",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "clear": {"type": "boolean"},
                                    "delete": {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "OK"}}
            }
        },
        "/history/{prompt_id}": {
            "get": {
                "summary": "Get single history entry",
                "parameters": [
                    {"name": "prompt_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {"description": "History entry", "content": {"application/json": {"schema": {"type": "object"}}}},
                    "404": {"description": "Not found"}
                }
            }
        },
        "/interrupt": {
            "post": {
                "summary": "Interrupt execution",
                "description": "Interrupt currently running prompt. Optional: ?prompt_id=UUID.",
                "parameters": [
                    {"name": "prompt_id", "in": "query", "schema": {"type": "string"}}
                ],
                "responses": {"200": {"description": "Interrupted"}}
            }
        },
        "/object_info": {
            "get": {
                "summary": "Get node type schemas",
                "description": "Returns all registered node types with their INPUT_TYPES, RETURN_TYPES, and CATEGORY.",
                "responses": {
                    "200": {
                        "description": "Node schemas",
                        "content": {"application/json": {"schema": {"type": "object"}}}
                    }
                }
            }
        },
        "/system_stats": {
            "get": {
                "summary": "System info",
                "description": "Returns OS, Python, PyTorch version, device info, VRAM.",
                "responses": {
                    "200": {
                        "description": "System stats",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SystemStats"}}}
                    }
                }
            }
        },
        "/models": {
            "get": {
                "summary": "List model folder types",
                "responses": {
                    "200": {
                        "description": "Array of folder type names",
                        "content": {"application/json": {"schema": {"type": "array", "items": {"type": "string"}}}}
                    }
                }
            }
        },
        "/models/{folder}": {
            "get": {
                "summary": "List models in folder",
                "description": "Returns all model filenames in the given folder (e.g. checkpoints, loras, vae).",
                "parameters": [
                    {"name": "folder", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {
                        "description": "Model filenames",
                        "content": {"application/json": {"schema": {"type": "array", "items": {"type": "string"}}}}
                    }
                }
            }
        },
        "/api/download-model": {
            "post": {
                "summary": "Download a model",
                "description": "Downloads a model to the correct folder. If URL omitted, looks up model_download_urls.yaml.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["folder_type", "filename"],
                                "properties": {
                                    "folder_type": {"type": "string", "description": "Model folder type (checkpoints, loras, vae, etc.)"},
                                    "filename": {"type": "string", "description": "Target filename"},
                                    "url": {"type": "string", "description": "Optional: direct download URL or HuggingFace resolve URL"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Download started",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/DownloadResponse"}}}
                    },
                    "400": {"description": "Bad request"}
                }
            }
        },
        "/api/download-status/{key}": {
            "get": {
                "summary": "Check download status",
                "parameters": [
                    {"name": "key", "in": "path", "required": True, "schema": {"type": "string"},
                     "description": "Download key from /api/download-model response"}
                ],
                "responses": {
                    "200": {
                        "description": "Download status",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/DownloadStatus"}}}
                    }
                }
            }
        },
        "/embeddings": {
            "get": {
                "summary": "List text embeddings",
                "responses": {
                    "200": {
                        "description": "Embedding filenames",
                        "content": {"application/json": {"schema": {"type": "array", "items": {"type": "string"}}}}
                    }
                }
            }
        },
        "/extensions": {
            "get": {
                "summary": "List web extensions",
                "responses": {
                    "200": {"description": "Extension list", "content": {"application/json": {"schema": {"type": "array", "items": {"type": "string"}}}}}
                }
            }
        },
        "/features": {
            "get": {
                "summary": "Server feature flags",
                "responses": {
                    "200": {"description": "Feature flags", "content": {"application/json": {"schema": {"type": "object"}}}}
                }
            }
        },
        "/ws": {
            "get": {
                "summary": "WebSocket connection",
                "description": "Real-time status, progress, and execution events.",
                "responses": {"101": {"description": "Switching protocols"}}
            }
        },
        "/api/sync-run": {
            "post": {
                "summary": "Run workflow synchronously",
                "description": "Upload images/video + workflow JSON, runs to completion, returns output files. Multipart form: 'workflow' (JSON), 'images'/'videos' (files), 'timeout' (int), 'response_type' ('files'|'json').",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["workflow"],
                                "properties": {
                                    "workflow": {"type": "string", "format": "binary", "description": "Workflow JSON file or string"},
                                    "images": {"type": "string", "format": "binary", "description": "Reference image(s) — mapped to LoadImage nodes in order"},
                                    "videos": {"type": "string", "format": "binary", "description": "Reference video(s) — mapped to LoadVideo nodes in order"},
                                    "timeout": {"type": "integer", "default": 300, "description": "Max wait seconds"},
                                    "response_type": {"type": "string", "enum": ["files", "json"], "default": "files", "description": "'files' returns multipart/mixed with output files, 'json' returns base64-encoded outputs"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Output files (multipart/mixed or JSON with base64)",
                        "content": {
                            "multipart/mixed": {"schema": {"type": "string", "format": "binary"}},
                            "application/json": {"schema": {"$ref": "#/components/schemas/SyncRunJsonResponse"}}
                        }
                    },
                    "400": {"description": "Invalid workflow or missing files"},
                    "504": {"description": "Timeout"}
                }
            }
        },
    },
    "components": {
        "schemas": {
            "PromptRequest": {
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "object", "description": "Workflow node graph"},
                    "client_id": {"type": "string", "description": "WebSocket client ID"},
                    "prompt_id": {"type": "string", "description": "Optional UUID for idempotent queuing"},
                    "extra_data": {"type": "object", "description": "Extra metadata for the prompt"},
                    "front": {"type": "boolean", "description": "Queue at front"}
                }
            },
            "PromptResponse": {
                "type": "object",
                "properties": {
                    "prompt_id": {"type": "string"},
                    "number": {"type": "number"},
                    "node_errors": {"type": "object"},
                    "model_downloads": {
                        "type": "object",
                        "properties": {
                            "downloaded": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                            "missing": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
                        }
                    }
                }
            },
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "error": {"$ref": "#/components/schemas/ErrorDetail"},
                    "node_errors": {"type": "object"},
                    "model_downloads": {"$ref": "#/components/schemas/ModelDownloads"}
                }
            },
            "ErrorDetail": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "message": {"type": "string"},
                    "details": {"type": "string"},
                    "extra_info": {"type": "object"}
                }
            },
            "ModelDownloads": {
                "type": "object",
                "properties": {
                    "downloaded": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                    "missing": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
                }
            },
            "QueueResponse": {
                "type": "object",
                "properties": {
                    "queue_running": {"type": "array", "items": {"type": "array"}},
                    "queue_pending": {"type": "array", "items": {"type": "array"}}
                }
            },
            "SystemStats": {
                "type": "object",
                "properties": {
                    "system": {"type": "object"},
                    "devices": {"type": "array", "items": {"type": "object"}}
                }
            },
            "DownloadResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["started", "downloaded", "downloading", "failed"]},
                    "key": {"type": "string", "description": "Use with /api/download-status/{key} to poll progress"},
                    "reason": {"type": "string"}
                }
            },
            "DownloadStatus": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["downloaded", "downloading", "failed", "unknown"]},
                    "downloaded_bytes": {"type": "integer"},
                    "total_bytes": {"type": "integer"},
                    "reason": {"type": "string"}
                }
            },
            "SyncRunJsonResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "completed"},
                    "outputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "type": {"type": "string", "example": "images"},
                                "base64": {"type": "string", "description": "Base64-encoded file contents"}
                            }
                        }
                    }
                }
            }
        }
    }
}

SWAGGER_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>ComfyUI API Docs</title>
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
    url: "/openapi.json",
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
