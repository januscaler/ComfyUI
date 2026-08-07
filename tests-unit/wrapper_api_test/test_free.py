"""Tests for the wrapper VRAM/RAM freeing logic."""

import unittest
from unittest import mock

from api_wrapper.routes import _queue_snapshots, free_vram


class FakeQueue:
    def __init__(self, running=0, queued=0):
        import threading

        self.mutex = threading.RLock()
        self.currently_running = {i: None for i in range(running)}
        self.queue = [None] * queued
        self.flags = {}

    def set_flag(self, name, data):
        self.flags[name] = data


class TestFreeVram(unittest.TestCase):
    def test_idle_queue_frees_immediately(self):
        q = FakeQueue()
        with mock.patch("api_wrapper.routes.model_management.unload_all_models") as unload, \
             mock.patch("api_wrapper.routes.gc.collect") as gc, \
             mock.patch("api_wrapper.routes.model_management.soft_empty_cache") as empty:
            result = free_vram(q)
        self.assertEqual(result, {"status": "ok", "deferred": False})
        unload.assert_called_once()
        gc.assert_called_once()
        empty.assert_called_once()
        self.assertEqual(q.flags, {"unload_models": True, "free_memory": True})

    def test_running_job_defers_free(self):
        q = FakeQueue(running=1)
        with mock.patch("api_wrapper.routes.model_management.unload_all_models") as unload, \
             mock.patch("api_wrapper.routes.gc.collect") as gc, \
             mock.patch("api_wrapper.routes.model_management.soft_empty_cache") as empty:
            result = free_vram(q)
        self.assertEqual(result, {"status": "ok", "deferred": True})
        unload.assert_not_called()
        gc.assert_not_called()
        empty.assert_not_called()
        # Flags stay set so the prompt worker releases memory after the job.
        self.assertEqual(q.flags, {"unload_models": True, "free_memory": True})

    def test_queued_job_defers_free(self):
        q = FakeQueue(queued=1)
        with mock.patch("api_wrapper.routes.model_management.unload_all_models") as unload, \
             mock.patch("api_wrapper.routes.model_management.soft_empty_cache") as empty:
            result = free_vram(q)
        self.assertEqual(result["deferred"], True)
        unload.assert_not_called()
        empty.assert_not_called()


class TestQueueSnapshots(unittest.TestCase):
    def test_strips_sensitive_element(self):
        class Q:
            currently_running = {0: (1, "run-id", {}, {"create_time": 1}, [], "SECRET")}
            queue = [(2, "q-id", {}, {"create_time": 2}, [], "SECRET")]

        running, queued = _queue_snapshots(Q())
        self.assertEqual(running, [(1, "run-id", {}, {"create_time": 1}, [])])
        self.assertEqual(queued, [(2, "q-id", {}, {"create_time": 2}, [])])

    def test_empty_queue(self):
        class Q:
            currently_running = {}
            queue = []

        running, queued = _queue_snapshots(Q())
        self.assertEqual((running, queued), ([], []))


if __name__ == "__main__":
    unittest.main()
