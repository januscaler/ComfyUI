# Synchronous Developer API Workflows

Place ComfyUI API workflow JSON files in this directory.

Use **File -> Export (API)** in the ComfyUI interface. The regular frontend
workflow JSON format is not accepted by the synchronous FastAPI wrapper because
the wrapper submits workflows directly to ComfyUI's existing `/prompt` queue.

Each file becomes a workflow id:

- `text_to_image.json` -> `text_to_image`
- `video/wan_edit.json` -> `video/wan_edit`

The REST endpoints support explicit node overrides so one workflow can serve
multiple models or settings without editing the saved JSON:

```json
{
  "3": { "seed": 1234, "steps": 30 },
  "4": { "ckpt_name": "model.safetensors" }
}
```

For media workflows, the wrapper uploads files into ComfyUI's `input/api/<id>/`
folder and patches `LoadImage` or `LoadVideo` automatically. If a workflow has
multiple loader nodes, pass `image_node_id` or `video_node_id` in the request.
