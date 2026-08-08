"""Tests for the wrapper route helpers (output file collection, setup errors)."""

import asyncio
import os
import time
import unittest

from api_wrapper import routes as wrapper_routes
from api_wrapper import workflows as wrapper_workflows


class TestCollectOutputFiles(unittest.TestCase):
    def _touch(self, *parts):
        path = os.path.join(*parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"test")
        return path

    def test_collects_images_videos_audio_3d(self):
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        img = self._touch(output_dir, "wrapper", "a.png")
        vid = self._touch(output_dir, "wrapper", "b.mp4")
        aud = self._touch(output_dir, "wrapper", "c.wav")
        obj = self._touch(output_dir, "wrapper", "d.obj")
        entry = {
            "outputs": {
                "10": {
                    "images": [{"filename": "a.png", "subfolder": "wrapper", "type": "output"}],
                    "videos": [{"filename": "b.mp4", "subfolder": "wrapper", "type": "output"}],
                },
                "11": {
                    "audio": [{"filename": "c.wav", "subfolder": "wrapper", "type": "output"}],
                    "3d": [{"filename": "d.obj", "subfolder": "wrapper", "type": "output"}],
                },
            }
        }
        files = wrapper_routes._collect_output_files(entry)
        self.assertEqual([os.path.basename(f) for f in files], ["a.png", "b.mp4", "c.wav", "d.obj"])
        for path in (img, vid, aud, obj):
            os.remove(path)

    def test_skips_missing_and_foreign_keys(self):
        entry = {
            "outputs": {
                "10": {
                    "images": [{"filename": "missing.png", "subfolder": "", "type": "output"}],
                    "latents": [{"whatever": True}],
                }
            }
        }
        self.assertEqual(wrapper_routes._collect_output_files(entry), [])


class TestSetupErrors(unittest.TestCase):
    def test_setup_error_carries_missing_models(self):
        err = wrapper_routes._SetupError(
            "Required models could not be downloaded",
            "hint",
            missing=[{"folder": "diffusion_models", "filename": "x.safetensors", "error": "401"}],
        )
        self.assertEqual(err.message, "Required models could not be downloaded")
        self.assertEqual(err.missing[0]["filename"], "x.safetensors")
        self.assertEqual(err.details, "hint")

    def test_setup_error_defaults(self):
        err = wrapper_routes._SetupError("boom")
        self.assertEqual(err.missing, [])
        self.assertEqual(err.details, "")


class TestTimeoutParsing(unittest.TestCase):
    def test_missing_or_zero_means_no_limit(self):
        self.assertIsNone(wrapper_routes._parse_timeout(None))
        self.assertIsNone(wrapper_routes._parse_timeout(""))
        self.assertIsNone(wrapper_routes._parse_timeout("0"))

    def test_explicit_cap(self):
        self.assertEqual(wrapper_routes._parse_timeout("120"), 120)

    def test_out_of_range_rejected(self):
        for bad in ("1", "100000", "abc"):
            with self.assertRaises(ValueError):
                wrapper_routes._parse_timeout(bad)


class _FakePromptQueue:
    """Minimal stand-in: history dict + queue lists, mutable from the test."""

    def __init__(self):
        self.history = {}
        self.currently_running = {}
        self.queue = []

    def get_history(self, prompt_id=None):
        if prompt_id is None:
            return self.history
        return {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}


class TestWaitForPrompt(unittest.IsolatedAsyncioTestCase):
    async def test_returns_when_history_appears(self):
        queue = _FakePromptQueue()
        queue.currently_running = {0: (1, "job-1", {}, {}, {}, {})}

        async def finish_later():
            await asyncio.sleep(0.05)
            queue.history["job-1"] = {"status": {"status_str": "success", "completed": True, "messages": []}}

        await asyncio.gather(wrapper_routes._wait_for_prompt(queue, "job-1", sleep_interval=0.01), finish_later())
        self.assertEqual(queue.history["job-1"]["status"]["status_str"], "success")

    async def test_no_timeout_waits_past_any_deadline(self):
        queue = _FakePromptQueue()
        queue.currently_running = {0: (1, "job-1", {}, {}, {}, {})}

        async def finish_later():
            await asyncio.sleep(0.2)  # longer than a naive 0.1s cap
            queue.history["job-1"] = {"status": {"status_str": "success", "completed": True, "messages": []}}

        entry = await asyncio.gather(wrapper_routes._wait_for_prompt(queue, "job-1", sleep_interval=0.01), finish_later())
        self.assertIsNotNone(entry[0])

    async def test_timeout_returns_none(self):
        queue = _FakePromptQueue()
        queue.currently_running = {0: (1, "job-1", {}, {}, {}, {})}
        entry = await wrapper_routes._wait_for_prompt(queue, "job-1", timeout=0.05, sleep_interval=0.01)
        self.assertIsNone(entry)

    async def test_vanished_job_returns_none(self):
        queue = _FakePromptQueue()  # job never queued: not running, not queued
        entry = await wrapper_routes._wait_for_prompt(queue, "job-1", sleep_interval=0.01)
        self.assertIsNone(entry)

    async def test_running_job_without_timeout_keeps_waiting(self):
        queue = _FakePromptQueue()
        queue.currently_running = {0: (1, "job-1", {}, {}, {}, {})}
        start = time.time()
        entry = await wrapper_routes._wait_for_prompt(queue, "job-1", timeout=0.2, sleep_interval=0.01)
        # job still running after the cap -> None (timed out)
        self.assertIsNone(entry)
        self.assertGreaterEqual(time.time() - start, 0.2)


class TestMiniMaxSetupContract(unittest.TestCase):
    """The per-task setups must return exactly the kwargs their builder
    accepts (a mismatch crashed the text task with an unexpected
    'ref_image_size' argument), and own their own defaults (steps default 50,
    not the shared handler default 20)."""

    def setUp(self):
        self._download_models = wrapper_routes._download_models
        self._raise_if_missing = wrapper_routes._raise_if_missing
        self._get_torch_device = wrapper_routes.model_management.get_torch_device
        import types

        async def no_download(*args, **kwargs):
            return []

        wrapper_routes._download_models = no_download
        wrapper_routes._raise_if_missing = lambda missing: None
        wrapper_routes.model_management.get_torch_device = lambda: types.SimpleNamespace(type="cpu")

    def tearDown(self):
        wrapper_routes._download_models = self._download_models
        wrapper_routes._raise_if_missing = self._raise_if_missing
        wrapper_routes.model_management.get_torch_device = self._get_torch_device

    def test_text_setup_kwargs_build(self):
        kwargs, _ = asyncio.run(wrapper_routes._setup_minimax_h3_text({}, []))
        self.assertNotIn("ref_image_size", kwargs)
        self.assertEqual(kwargs["steps"], 50)  # setup default, not the handler's 20
        graph = wrapper_workflows.build_minimax_h3_text_to_video(prompt="x", **kwargs)
        self.assertIn("1", graph)

    def test_image_setup_kwargs_build(self):
        kwargs, _ = asyncio.run(wrapper_routes._setup_minimax_h3_image({}, []))
        self.assertNotIn("ref_image_size", kwargs)
        kwargs["first_frame"] = "wrapper/a.png"
        graph = wrapper_workflows.build_minimax_h3_image_to_video(prompt="x", **kwargs)
        self.assertIn("1", graph)

    def test_reference_setup_kwargs_build(self):
        kwargs, _ = asyncio.run(wrapper_routes._setup_minimax_h3_reference({}, []))
        self.assertEqual(kwargs["ref_image_size"], "match")
        graph = wrapper_workflows.build_minimax_h3_reference_to_video(prompt="x", **kwargs)
        self.assertIn("1", graph)

    def test_steps_field_forwarded(self):
        kwargs, _ = asyncio.run(wrapper_routes._setup_minimax_h3_text({"steps": "20"}, []))
        self.assertEqual(kwargs["steps"], 20)


if __name__ == "__main__":
    unittest.main()
