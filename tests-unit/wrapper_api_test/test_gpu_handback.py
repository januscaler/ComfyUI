"""A video render takes the GPU back from warm voice engines before it is queued.

The engines sit behind idle gates that may keep a model loaded while nothing else uses the card; an
H3 render started beside a warm Maya (~12 GB of 32) is the one that runs out of memory. The wrapper
asks each gate to hand the GPU back first (bounded, and never fatal to the render).
"""

import asyncio
import threading
import time

import pytest
from aiohttp import web

from api_wrapper import gpu_handback
from api_wrapper import routes as wrapper_routes


class Gate:
    """A stand-in idle gate: records the releases it gets, answers as told."""

    def __init__(self, status=200, delay=0.0):
        self.status, self.delay, self.calls = status, delay, []

    async def release(self, request):
        self.calls.append(dict(request.query))
        await asyncio.sleep(self.delay)
        return web.json_response({"running": False}, status=self.status)


async def serve(aiohttp_server, gate):
    app = web.Application()
    app.router.add_post("/__idle-gate/release", gate.release)
    server = await aiohttp_server(app)
    return str(server.make_url("")).rstrip("/")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("WRAPPER_RELEASE_URLS", "WRAPPER_RELEASE_WORKFLOWS", "WRAPPER_RELEASE_BLOCK_S",
                 "WRAPPER_RELEASE_WAIT_S"):
        monkeypatch.delenv(name, raising=False)


def test_nothing_is_asked_unless_the_deployment_names_the_gates(monkeypatch):
    assert not gpu_handback.wanted("minimaxh3")
    assert asyncio.run(gpu_handback.release_all()) == {}
    monkeypatch.setenv("WRAPPER_RELEASE_URLS", "http://maya:8080, http://omnivoice:3900/")
    assert gpu_handback.urls() == ["http://maya:8080", "http://omnivoice:3900"]


def test_only_the_video_workflows_need_the_whole_card_unless_the_deployment_says_otherwise(monkeypatch):
    monkeypatch.setenv("WRAPPER_RELEASE_URLS", "http://maya:8080")
    assert gpu_handback.wanted("minimaxh3") and gpu_handback.wanted("LTX25")
    assert not gpu_handback.wanted("qwenimage21") and not gpu_handback.wanted("fx-upscale")
    monkeypatch.setenv("WRAPPER_RELEASE_WORKFLOWS", "qwenimage21")
    assert gpu_handback.wanted("qwenimage21") and not gpu_handback.wanted("minimaxh3")


@pytest.mark.asyncio
async def test_every_gate_is_asked_to_release_with_the_block_and_wait(aiohttp_server, monkeypatch):
    maya, voice = Gate(), Gate()
    monkeypatch.setenv("WRAPPER_RELEASE_URLS",
                       f"{await serve(aiohttp_server, maya)},{await serve(aiohttp_server, voice)}")
    monkeypatch.setenv("WRAPPER_RELEASE_BLOCK_S", "120")
    monkeypatch.setenv("WRAPPER_RELEASE_WAIT_S", "7")
    outcomes = await gpu_handback.release_all()
    assert list(outcomes.values()) == ["released", "released"]
    assert maya.calls == voice.calls == [{"block": "120", "wait": "7"}]


@pytest.mark.asyncio
async def test_an_unreachable_gate_is_logged_and_the_others_still_release(
        aiohttp_server, monkeypatch, caplog):
    good = Gate()
    dead = "http://127.0.0.1:9"                         # nothing listens
    monkeypatch.setenv("WRAPPER_RELEASE_URLS", f"{dead},{await serve(aiohttp_server, good)}")
    outcomes = await gpu_handback.release_all()
    assert outcomes[dead] != "released" and list(outcomes.values())[1] == "released"
    assert good.calls and "did not hand the GPU back" in caplog.text


@pytest.mark.asyncio
async def test_a_gate_that_still_has_lines_running_is_logged_not_waited_for(
        aiohttp_server, monkeypatch, caplog):
    busy = Gate(status=409)
    monkeypatch.setenv("WRAPPER_RELEASE_URLS", await serve(aiohttp_server, busy))
    outcomes = await gpu_handback.release_all()
    assert list(outcomes.values()) == ["answered 409"] and "rendering anyway" in caplog.text


@pytest.mark.asyncio
async def test_a_gate_that_never_answers_costs_a_bounded_wait_and_nothing_else(
        aiohttp_server, monkeypatch, caplog):
    stuck = Gate(delay=10)
    monkeypatch.setenv("WRAPPER_RELEASE_URLS", await serve(aiohttp_server, stuck))
    monkeypatch.setenv("WRAPPER_RELEASE_WAIT_S", "0.2")
    monkeypatch.setattr(gpu_handback, "SLACK_S", 0.3)
    started = time.monotonic()
    outcomes = await gpu_handback.release_all()
    assert time.monotonic() - started < 2
    assert list(outcomes.values()) != ["released"] and "rendering anyway" in caplog.text


# --- the call site: before the prompt is queued -----------------------------------------------


class RecordingQueue:
    def __init__(self, events):
        self.mutex = threading.RLock()
        self.currently_running, self.queue, self.flags = {}, [], {}
        self.events = events

    def get_history(self, *args, **kwargs):
        return {}

    def set_flag(self, name, data):
        self.flags[name] = data

    def put(self, item):
        self.events.append("queued")
        raise RuntimeError("stop here: the order is what is tested")


class Server:
    def __init__(self, events):
        self.prompt_queue = RecordingQueue(events)


def app_of(server):
    routes = web.RouteTableDef()
    wrapper_routes.register_wrapper_routes(routes, server)
    app = web.Application()
    app.add_routes(routes)
    return app


@pytest.fixture
def fake_workflows(monkeypatch):
    from api_wrapper import workflows

    def entry():
        return {"title": "t", "requires_image": False, "uses": [], "form": [],
                "build": lambda **kw: {}}

    monkeypatch.setitem(workflows.WORKFLOWS, "minimaxh3", entry())
    monkeypatch.setitem(workflows.WORKFLOWS, "qwenimage21-txt2img", entry())

    async def setup(fields, downloaded, *rest):
        return {}, ""

    for name in ("minimaxh3", "qwenimage21-txt2img"):
        monkeypatch.setitem(wrapper_routes._WORKFLOW_SETUPS, name, setup)

    async def valid(*args, **kwargs):
        return (True, None, [])

    monkeypatch.setattr(wrapper_routes.execution, "validate_prompt", valid)


async def generate(aiohttp_client, events, workflow):
    client = await aiohttp_client(app_of(Server(events)))
    await client.post(f"/wrapper/{workflow}/generate", data={"prompt": "a mug"})


@pytest.mark.asyncio
async def test_a_video_prompt_is_queued_only_after_the_gates_released(
        aiohttp_client, aiohttp_server, monkeypatch, fake_workflows):
    events = []

    class Ordered(Gate):
        async def release(self, request):
            events.append("released")
            return await super().release(request)

    monkeypatch.setenv("WRAPPER_RELEASE_URLS", await serve(aiohttp_server, Ordered()))
    await generate(aiohttp_client, events, "minimaxh3")
    assert events == ["released", "queued"]


@pytest.mark.asyncio
async def test_an_image_prompt_never_asks_the_gates(aiohttp_client, aiohttp_server, monkeypatch,
                                                    fake_workflows):
    events, gate = [], Gate()
    monkeypatch.setenv("WRAPPER_RELEASE_URLS", await serve(aiohttp_server, gate))
    await generate(aiohttp_client, events, "qwenimage21-txt2img")
    assert events == ["queued"] and gate.calls == []


@pytest.mark.asyncio
async def test_a_dead_gate_never_stops_the_render_being_queued(aiohttp_client, monkeypatch,
                                                               fake_workflows):
    events = []
    monkeypatch.setenv("WRAPPER_RELEASE_URLS", "http://127.0.0.1:9")
    await generate(aiohttp_client, events, "minimaxh3")
    assert events == ["queued"]
