"""A submit that is retried runs once, and an upload that does not arrive is let go.

2026-10-24: Image Studio fills whose uploads crawled over a saturated link timed
out at the caller (60 s) and were sent again, up to three times; each copy was a
new job id, and a stalled upload held its handler with no deadline. A caller's
own job id (X-Client-Job-Id) makes the retry find the job instead of queueing
it again, and the body has a deadline.
"""

import asyncio
import threading
import uuid

import pytest
from aiohttp import web

from api_wrapper import cleanup as wrapper_cleanup
from api_wrapper import routes as wrapper_routes


class FakePromptQueue:
    def __init__(self):
        self.mutex = threading.RLock()
        self.currently_running = {}
        self.queue = []
        self.flags = {}

    def get_history(self, *args, **kwargs):
        return {}

    def set_flag(self, name, data):
        self.flags[name] = data


class FakePromptServer:
    def __init__(self):
        self.prompt_queue = FakePromptQueue()


@pytest.fixture
def server():
    return FakePromptServer()


def make_app(server):
    routes = web.RouteTableDef()
    wrapper_routes.register_wrapper_routes(routes, server)
    app = web.Application()
    app.add_routes(routes)
    return app


URL = "/wrapper/qwenimage21/generate"


@pytest.mark.asyncio
async def test_a_retried_submit_finds_its_queued_job_instead_of_queueing_it_again(aiohttp_client, server):
    client = await aiohttp_client(make_app(server))
    job = str(uuid.uuid4())
    server.prompt_queue.queue.append((1, job, {}, {"create_time": 1}, []))
    res = await client.post(URL, headers={wrapper_routes.CLIENT_JOB_HEADER: job}, data=b"never read")
    assert res.status == 504
    body = await res.json()
    assert body["job_id"] == job
    assert body["error"]["type"] == "duplicate"
    assert len(server.prompt_queue.queue) == 1


@pytest.mark.asyncio
async def test_a_second_copy_while_the_first_is_still_arriving_is_refused(aiohttp_client, server):
    client = await aiohttp_client(make_app(server))
    job = str(uuid.uuid4())
    with wrapper_cleanup.generate_scope(job):
        res = await client.post(URL, headers={wrapper_routes.CLIENT_JOB_HEADER: job}, data=b"x")
        assert res.status == 409
        assert (await res.json())["error"]["type"] == "in_progress"


@pytest.mark.asyncio
async def test_a_client_job_id_must_be_a_canonical_uuid(aiohttp_client, server):
    client = await aiohttp_client(make_app(server))
    res = await client.post(URL, headers={wrapper_routes.CLIENT_JOB_HEADER: "not-a-uuid"}, data=b"x")
    assert res.status == 400


@pytest.mark.asyncio
async def test_an_upload_that_does_not_arrive_whole_in_time_is_let_go(aiohttp_client, server, monkeypatch):
    client = await aiohttp_client(make_app(server))
    monkeypatch.setattr(wrapper_routes, "_BODY_TIMEOUT", 0.3)

    async def crawling():
        yield b"--b\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\na mug\r\n"
        await asyncio.sleep(5)
        yield b"--b--\r\n"

    loop = asyncio.get_running_loop()
    started = loop.time()
    res = await client.post(URL, data=crawling(), headers={"Content-Type": "multipart/form-data; boundary=b"})
    assert res.status == 408
    assert (await res.json())["error"]["message"] == "Upload too slow"
    assert loop.time() - started < 3


@pytest.mark.asyncio
async def test_a_caller_whose_submit_timed_out_finds_a_job_still_arriving(aiohttp_client, server):
    client = await aiohttp_client(make_app(server))
    job = str(uuid.uuid4())
    assert (await client.get(f"/wrapper/jobs/{job}")).status == 404
    with wrapper_cleanup.generate_scope(job):
        res = await client.get(f"/wrapper/jobs/{job}")
        assert res.status == 200
        assert (await res.json())["status"] == "pending"
    assert (await client.get(f"/wrapper/jobs/{job}")).status == 404
