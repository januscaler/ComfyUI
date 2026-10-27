"""The JSON answer of a job (api_wrapper/results.py) and GET /wrapper/jobs/{job_id}/text.

The lookup is pure and tested on plain dicts. The route is the real one,
registered by routes.register_wrapper_routes on a bare aiohttp app (mirrored
under /api the way server.py does) over a fake prompt queue whose history
holds what a finished ComfyUI job leaves: its graph and its node outputs.
"""

import json
import threading
import uuid

import pytest
from aiohttp import web

from api_wrapper import results
from api_wrapper import workflows as wf

SEGMENT_TEXT = json.dumps([[{"x": 0.0, "y": 0.0, "width": 640.0, "height": 200.0, "score": 0.9}]], indent=4)
DESCRIBE_TEXT = '{"caption": "A beach.", "objects": [{"label": "sea", "bbox_2d": [0, 400, 1000, 1000]}]}'


def segment_graph(individual=True):
    request = {"mode": "prompt", "threshold": 0.5, "individual_masks": individual,
               "text": "sky:8", "max_detections": 8}
    return wf.build_image_segment(image="wrapper/j/in.png", request=request, width=640, height=480,
                                  filename_prefix="wrapper/j/image-segment")


def segment_outputs(text=SEGMENT_TEXT):
    return {
        "9": {"images": [{"filename": f"image-segment_0000{i}_.png", "subfolder": "wrapper/j", "type": "output"}
                         for i in (1, 2)]},
        "10": {"text": (text,)},
    }


def describe_graph():
    return wf.build_image_describe(image="wrapper/j/in.png", task="both")


class TestTextResult:
    def test_segment_answer_pairs_objects_with_their_masks(self):
        answer = results.text_result(segment_graph(), segment_outputs(), wf.WORKFLOWS)
        assert answer == {
            "mode": "prompt", "width": 640, "height": 480,
            "mask": "/view?filename=image-segment_00001_.png&subfolder=wrapper/j&type=output",
            "objects": [{"score": 0.9, "bbox": [0.0, 0.0, 640.0, 200.0],
                         "mask": "/view?filename=image-segment_00002_.png&subfolder=wrapper/j&type=output"}],
        }

    def test_describe_answer(self):
        answer = results.text_result(describe_graph(), {"6": {"text": [DESCRIBE_TEXT]}}, wf.WORKFLOWS)
        assert answer == {"caption": "A beach.", "objects": [{"label": "sea", "bbox": [0.0, 0.4, 1.0, 0.6]}]}

    def test_untagged_and_unknown_workflows_have_none(self):
        assert results.text_result(wf.build_fx_bgremove(video="v.mp4"), {}, wf.WORKFLOWS) is None
        graph = {"1": results.tag_result({"class_type": "PreviewAny", "inputs": {}}, "no-such-workflow")}
        assert results.text_result(graph, {"1": {"text": ["x"]}}, wf.WORKFLOWS) is None
        assert results.text_result(None, None, wf.WORKFLOWS) is None

    def test_a_missing_text_output_still_answers(self):
        """A history entry without the node's text (it never ran) parses as nothing found."""
        assert results.text_result(describe_graph(), {}, wf.WORKFLOWS) == {"caption": "", "objects": []}

    def test_saved_images_skip_temp_and_input_entries(self):
        outputs = {"1": {"images": [{"filename": "up.png", "type": "input"}]},
                   "2": {"images": [{"filename": "p.png", "type": "temp"}, {"filename": "a.png"}]},
                   "3": {"text": ["x"]}, "4": "not a dict"}
        assert results.saved_images(outputs) == ["/view?filename=a.png&subfolder=&type=output"]


    def test_first_image_is_a_saved_one_never_an_inputs_preview(self):
        """A graph that loads its video through LoadVideo lists the input in its history
        (node ids 10 and 13 before the SaveVideo's 38): the job's answer is the saved file."""
        outputs = {"10": {"images": [{"filename": "in.mp4", "subfolder": "wrapper/j", "type": "input"}]},
                   "13": {"images": [{"filename": "mask.mp4", "subfolder": "wrapper/j", "type": "input"}]},
                   "38": {"images": [{"filename": "fx-replace_00001_.mp4", "subfolder": "wrapper/j",
                                      "type": "output"}]}}
        assert results.first_image(outputs)["filename"] == "fx-replace_00001_.mp4"
        # With nothing saved, the first of any kind (the old behaviour); with nothing, None.
        only_inputs = {"10": outputs["10"], "13": outputs["13"]}
        assert results.first_image(only_inputs)["filename"] == "in.mp4"
        assert results.first_image({"3": {"text": ["x"]}, "4": "not a dict"}) is None
        assert results.first_image(None) is None


class FakePromptQueue:
    def __init__(self):
        self.mutex = threading.RLock()
        self.currently_running = {}
        self.queue = []
        self.history = {}

    def get_history(self, prompt_id=None):
        if prompt_id is None:
            return dict(self.history)
        return {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}


def finished(queue, graph, outputs, status="success"):
    job_id = str(uuid.uuid4())
    queue.history[job_id] = {"prompt": (1, job_id, graph, {}, []), "outputs": outputs,
                             "status": {"status_str": status, "completed": status == "success", "messages": []}}
    return job_id


@pytest.fixture
def queue():
    return FakePromptQueue()


def make_app(prompt_queue):
    """Every wrapper route on a bare app, mirrored under /api like server.py does."""
    from api_wrapper import routes as wrapper_routes

    class Server:
        pass

    server = Server()
    server.prompt_queue = prompt_queue
    routes = web.RouteTableDef()
    wrapper_routes.register_wrapper_routes(routes, server)
    api_routes = web.RouteTableDef()
    for route in routes:
        api_routes.route(route.method, "/api" + route.path)(route.handler, **route.kwargs)
    app = web.Application()
    app.add_routes(api_routes)
    app.add_routes(routes)
    return app


@pytest.mark.asyncio
class TestTextRoute:
    async def test_describe_job(self, aiohttp_client, queue):
        job_id = finished(queue, describe_graph(), {"6": {"text": [DESCRIBE_TEXT]}})
        client = await aiohttp_client(make_app(queue))
        response = await client.get(f"/api/wrapper/jobs/{job_id}/text")
        assert response.status == 200
        assert await response.json() == {"caption": "A beach.",
                                         "objects": [{"label": "sea", "bbox": [0.0, 0.4, 1.0, 0.6]}]}

    async def test_segment_job_without_the_api_prefix(self, aiohttp_client, queue):
        job_id = finished(queue, segment_graph(individual=False), segment_outputs())
        client = await aiohttp_client(make_app(queue))
        response = await client.get(f"/wrapper/jobs/{job_id}/text")
        assert response.status == 200
        body = await response.json()
        assert body["mask"].endswith("image-segment_00001_.png&subfolder=wrapper/j&type=output")
        assert body["objects"] == [{"score": 0.9, "bbox": [0.0, 0.0, 640.0, 200.0]}]

    async def test_refusals(self, aiohttp_client, queue):
        http = await aiohttp_client(make_app(queue))
        assert (await http.get("/api/wrapper/jobs/not-a-uuid/text")).status == 400
        assert (await http.get(f"/api/wrapper/jobs/{uuid.uuid4()}/text")).status == 404
        video_job = finished(queue, wf.build_fx_bgremove(video="v.mp4"), {})
        response = await http.get(f"/api/wrapper/jobs/{video_job}/text")
        assert response.status == 404
        assert (await response.json())["error"]["message"] == "Job has no text output"
        failed = finished(queue, describe_graph(), {}, status="error")
        assert (await http.get(f"/api/wrapper/jobs/{failed}/text")).status == 409
        pending = str(uuid.uuid4())
        queue.queue.append((2, pending, describe_graph(), {}, [], {}))
        assert (await http.get(f"/api/wrapper/jobs/{pending}/text")).status == 409

    async def test_image_route_still_redirects_to_the_first_mask(self, aiohttp_client, queue):
        job_id = finished(queue, segment_graph(), segment_outputs())
        client = await aiohttp_client(make_app(queue))
        response = await client.get(f"/api/wrapper/jobs/{job_id}/image", allow_redirects=False)
        assert response.status == 302
        assert response.headers["Location"] == "/view?filename=image-segment_00001_.png&subfolder=wrapper/j&type=output"

    async def test_image_route_redirects_to_the_saved_file_not_the_loaded_input(self, aiohttp_client, queue):
        outputs = {"10": {"images": [{"filename": "in.mp4", "subfolder": "wrapper/j", "type": "input"}]},
                   "38": {"images": [{"filename": "fx-replace_00001_.mp4", "subfolder": "wrapper/j",
                                      "type": "output"}]}}
        job_id = finished(queue, wf.build_fx_bgremove(video="v.mp4"), outputs)
        client = await aiohttp_client(make_app(queue))
        response = await client.get(f"/api/wrapper/jobs/{job_id}/image", allow_redirects=False)
        assert response.status == 302
        assert response.headers["Location"] == (
            "/view?filename=fx-replace_00001_.mp4&subfolder=wrapper/j&type=output")
