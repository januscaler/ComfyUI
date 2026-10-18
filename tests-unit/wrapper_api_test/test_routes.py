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

    def test_saved_output_wins_over_an_upload_preview(self):
        """LoadVideo reports the upload itself (type "input") and runs before
        SaveVideo; the response must be the saved video, not the upload."""
        import folder_paths

        upload = self._touch(folder_paths.get_input_directory(), "wrapper", "job", "upload.mp4")
        saved = self._touch(folder_paths.get_output_directory(), "wrapper", "job", "fx-interp_00001_.mp4")
        entry = {
            "outputs": {
                "1": {"images": [{"filename": "upload.mp4", "subfolder": "wrapper/job", "type": "input"}]},
                "6": {"images": [{"filename": "fx-interp_00001_.mp4", "subfolder": "wrapper/job", "type": "output"}]},
            }
        }
        try:
            files = wrapper_routes._collect_output_files(entry)
            self.assertEqual(files, [saved, upload])
        finally:
            os.remove(upload)
            os.remove(saved)

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
    'ref_image_size' argument), and own their own defaults (H3's canvas/length
    defaults, not the shared handler's)."""

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
        self.assertEqual(kwargs["steps"], wrapper_workflows.MINIMAX_H3_DEFAULT_STEPS)
        self.assertEqual(kwargs["width"], wrapper_workflows.MINIMAX_H3_DEFAULT_WIDTH)
        self.assertEqual(kwargs["height"], wrapper_workflows.MINIMAX_H3_DEFAULT_HEIGHT)
        self.assertEqual(kwargs["unet_name"], wrapper_workflows.MINIMAX_H3_UNET_INT8)
        self.assertEqual(kwargs["clip_name"], wrapper_workflows.MINIMAX_H3_CLIP_NVFP4)
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


class TestQwenImage21SetupContract(unittest.TestCase):
    """Qwen Image 2.1 runs at the templates' 25 steps / cfg 1, not the shared
    handler's 20 / 5, and each setup's kwargs build its graph."""

    def setUp(self):
        self._download_models = wrapper_routes._download_models
        self.requested = []

        async def no_download(models, downloaded):
            self.requested.extend(m["filename"] for m in models)
            return []

        wrapper_routes._download_models = no_download

    def tearDown(self):
        wrapper_routes._download_models = self._download_models

    def test_txt2img_defaults_are_the_templates(self):
        kwargs, note = asyncio.run(wrapper_routes._setup_qwenimage21_txt2img({}, []))
        self.assertIsNone(note)
        self.assertEqual((kwargs["steps"], kwargs["cfg"]), (25, 1.0))
        self.assertEqual((kwargs["width"], kwargs["height"]), (1024, 1024))
        self.assertEqual(kwargs["unet_name"], wrapper_workflows.QWEN_IMAGE_21_UNET)
        self.assertEqual(kwargs["clip_name"], wrapper_workflows.QWEN_IMAGE_21_CLIP)
        self.assertIn(wrapper_workflows.QWEN_IMAGE_21_VAE, self.requested)
        wrapper_workflows.build_qwen_image_21_text2img(prompt="x", **kwargs)

    def test_edit_takes_resolution_canvas_and_bf16(self):
        kwargs, _ = asyncio.run(wrapper_routes._setup_qwenimage21_edit(
            {"quantization": "bf16", "steps": "40", "cfg": "4", "resolution": "0",
             "width": "1024", "height": "576"}, []))
        self.assertEqual((kwargs["steps"], kwargs["cfg"], kwargs["resolution"]), (40, 4.0, 0))
        self.assertEqual((kwargs["width"], kwargs["height"]), (1024, 576))
        self.assertEqual(kwargs["unet_name"], "qwen_image_2.1_bf16.safetensors")
        self.assertEqual(kwargs["clip_name"], "qwen3vl_8b_bf16.safetensors")
        wrapper_workflows.build_qwen_image_21_edit(prompt="x", image="wrapper/a.png", **kwargs)

    def test_edit_canvas_is_both_or_neither(self):
        kwargs, _ = asyncio.run(wrapper_routes._setup_qwenimage21_edit({}, []))
        self.assertNotIn("width", kwargs)
        with self.assertRaises(wrapper_routes._SetupError):
            asyncio.run(wrapper_routes._setup_qwenimage21_edit({"width": "1024"}, []))

    def test_bad_values_fail_before_any_download(self):
        for fields in ({"quantization": "fp8"}, {"resolution": "5000"}, {"width": "10", "height": "10"}):
            with self.assertRaises(wrapper_routes._SetupError):
                asyncio.run(wrapper_routes._setup_qwenimage21_edit(fields, []))
        self.assertEqual(self.requested, [])


class TestFxInterpSetup(unittest.TestCase):
    """fx-interp probes the upload (mocked here: no real video or model) and
    refuses bad params and over-budget clips before anything is downloaded."""

    def setUp(self):
        self._download_models = wrapper_routes._download_models
        self._probe_video = wrapper_routes._probe_video
        self.requested = []
        self.probe = (1280, 720, 61, 30.0)

        async def no_download(models, downloaded):
            self.requested.extend((m["folder"], m["filename"]) for m in models)
            return []

        wrapper_routes._download_models = no_download
        wrapper_routes._probe_video = lambda path: self.probe

    def tearDown(self):
        wrapper_routes._download_models = self._download_models
        wrapper_routes._probe_video = self._probe_video

    def _setup(self, fields, refs=None):
        refs = {"video": ["wrapper/job/clip.mp4"]} if refs is None else refs
        return asyncio.run(wrapper_routes._setup_fx_interp(fields, [], refs))

    def test_defaults_are_factor_2_with_film_and_build(self):
        kwargs, note = self._setup({})
        self.assertEqual(kwargs, {"factor": 2, "model_name": "film_net_fp16.safetensors", "frames": 61})
        self.assertEqual(self.requested, [("frame_interpolation", "film_net_fp16.safetensors")])
        self.assertIn("61 frames of 1280x720 -> 121 frames", note)
        self.assertIn("30 fps (2x slower, no audio)", note)
        graph = wrapper_workflows.build_fx_interp(video="wrapper/job/clip.mp4", **kwargs)
        self.assertEqual(graph["4"]["inputs"]["multiplier"], 2)

    def test_factor_4_with_rife(self):
        kwargs, _ = self._setup({"factor": "4", "model": "RIFE"})
        self.assertEqual((kwargs["factor"], kwargs["model_name"]), (4, "rife_v4.26.safetensors"))
        self.assertEqual(self.requested, [("frame_interpolation", "rife_v4.26.safetensors")])

    def test_bad_params_fail_before_any_download(self):
        for fields in ({"factor": "3"}, {"factor": "8"}, {"factor": "two"}, {"model": "dain"}):
            with self.assertRaises(wrapper_routes._SetupError):
                self._setup(fields)
        self.assertEqual(self.requested, [])

    def test_unreadable_short_and_missing_videos_are_refused(self):
        def broken(path):
            raise ValueError("No video stream found")

        wrapper_routes._probe_video = broken
        with self.assertRaises(wrapper_routes._SetupError) as ctx:
            self._setup({})
        self.assertEqual(ctx.exception.message, "Unreadable video")
        wrapper_routes._probe_video = lambda path: (640, 360, 1, 30.0)
        with self.assertRaises(wrapper_routes._SetupError) as ctx:
            self._setup({})
        self.assertEqual(ctx.exception.message, "Video too short")
        with self.assertRaises(wrapper_routes._SetupError):
            self._setup({}, refs={})
        self.assertEqual(self.requested, [])

    def test_over_budget_clip_is_refused(self):
        # 10 s of 1080p30 at 4x is ~1197 frames, ~2.5 gigapixel-frames.
        self.probe = (1920, 1080, 300, 30.0)
        with self.assertRaises(wrapper_routes._SetupError) as ctx:
            self._setup({"factor": "4"})
        self.assertIn("COMFY_FX_INTERP_MAX_PIXEL_FRAMES", ctx.exception.details)
        self.assertEqual(self.requested, [])

    def test_factor_shows_in_the_settings_header(self):
        headers = wrapper_routes._resolution_headers(
            {"factor": 4, "model_name": "rife_v4.26.safetensors", "frames": 61}, "n")
        self.assertEqual(headers["X-Wrapper-Settings"], "factor=4")
        self.assertEqual(headers["X-Wrapper-Models"], "rife_v4.26.safetensors")


class _ProbedVideoSetup(unittest.TestCase):
    """Setup handlers that probe the uploaded video: the probe and the
    downloads are mocked (no real video or model)."""

    probe = (1280, 720, 61, 30.0)

    def setUp(self):
        self._download_models = wrapper_routes._download_models
        self._probe_video = wrapper_routes._probe_video
        self.requested = []

        async def no_download(models, downloaded):
            self.requested.extend((m["folder"], m["filename"]) for m in models)
            return []

        wrapper_routes._download_models = no_download
        wrapper_routes._probe_video = lambda path: self.probe

    def tearDown(self):
        wrapper_routes._download_models = self._download_models
        wrapper_routes._probe_video = self._probe_video

    def _run(self, setup, fields, refs=None):
        refs = {"video": ["wrapper/job/clip.mp4"]} if refs is None else refs
        return asyncio.run(setup(fields, [], refs))

    def _refused(self, setup, fields, refs=None):
        with self.assertRaises(wrapper_routes._SetupError) as ctx:
            self._run(setup, fields, refs)
        return ctx.exception


class TestFxBgRemoveSetup(_ProbedVideoSetup):
    def _setup(self, fields, refs=None):
        return self._run(wrapper_routes._setup_fx_bgremove, fields, refs)

    def test_defaults_to_birefnet_and_builds(self):
        kwargs, note = self._setup({})
        self.assertEqual(kwargs, {"model_name": "birefnet.safetensors", "frames": 61})
        self.assertEqual(self.requested, [("background_removal", "birefnet.safetensors")])
        self.assertIn("61 frames of 1280x720 -> grayscale matte (white = subject) with birefnet", note)
        self.assertIn("30 fps, no audio", note)
        graph = wrapper_workflows.build_fx_bgremove(video="wrapper/job/clip.mp4", **kwargs)
        self.assertEqual(graph["4"]["_meta"]["expected_steps"], 61)
        headers = wrapper_routes._resolution_headers(kwargs, note)
        self.assertEqual(headers["X-Wrapper-Models"], "birefnet.safetensors")

    def test_lucida(self):
        kwargs, _ = self._setup({"model": " Lucida "})
        self.assertEqual(kwargs["model_name"], "lucida.safetensors")
        self.assertEqual(self.requested, [("background_removal", "lucida.safetensors")])

    def test_refusals_happen_before_any_download(self):
        self.assertEqual(self._refused(wrapper_routes._setup_fx_bgremove, {"model": "u2net"}).message, "Invalid model")
        self.assertEqual(self._refused(wrapper_routes._setup_fx_bgremove, {}, refs={}).message, "No video provided")
        self.probe = (641, 360, 30, 30.0)
        self.assertEqual(self._refused(wrapper_routes._setup_fx_bgremove, {}).message, "Unsupported video size")
        self.probe = (640, 360, 0, 30.0)
        self.assertEqual(self._refused(wrapper_routes._setup_fx_bgremove, {}).message, "Video too short")

        def broken(path):
            raise ValueError("No video stream found")

        wrapper_routes._probe_video = broken
        self.assertEqual(self._refused(wrapper_routes._setup_fx_bgremove, {}).message, "Unreadable video")
        self.assertEqual(self.requested, [])

    def test_over_budget_clip_is_refused(self):
        # 30 s of 1080p30.
        self.probe = (1920, 1080, 900, 30.0)
        error = self._refused(wrapper_routes._setup_fx_bgremove, {})
        self.assertIn("COMFY_FX_BGREMOVE_MAX_PIXEL_FRAMES", error.details)
        self.assertEqual(self.requested, [])


class TestFxUpscaleSetup(_ProbedVideoSetup):
    probe = (640, 360, 61, 30.0)

    def _setup(self, fields, refs=None):
        return self._run(wrapper_routes._setup_fx_upscale, fields, refs)

    def test_defaults_are_esrgan_to_1080(self):
        kwargs, note = self._setup({})
        self.assertEqual(kwargs, {"model": "esrgan", "width": 1920, "height": 1080, "frames": 61,
                                  "upscale_model_name": "RealESRGAN_x4plus.safetensors",
                                  "source_size": (640, 360)})
        self.assertEqual(self.requested, [("upscale_models", "RealESRGAN_x4plus.safetensors")])
        self.assertIn("61 frames of 640x360 -> 1920x1080 (3.00x) with esrgan x4 at the source 30 fps", note)
        graph = wrapper_workflows.build_fx_upscale(video="wrapper/job/clip.mp4", **kwargs)
        self.assertEqual(graph["5"]["inputs"]["width"], 1920)
        headers = wrapper_routes._resolution_headers(kwargs, note)
        self.assertEqual(headers["X-Wrapper-Models"], "RealESRGAN_x4plus.safetensors")
        self.assertEqual(headers["X-Wrapper-Settings"], "width=1920 height=1080")

    def test_small_factor_uses_the_x2_model(self):
        self.probe = (1280, 720, 61, 25.0)
        kwargs, note = self._setup({"target_short_side": "1080"})
        self.assertEqual(kwargs["upscale_model_name"], "RealESRGAN_x2plus.pth")
        self.assertEqual(self.requested, [("upscale_models", "RealESRGAN_x2plus.pth")])
        self.assertIn("(1.50x) with esrgan x2", note)

    def test_seedvr2(self):
        kwargs, note = self._setup({"model": "SeedVR2", "target_short_side": "720", "seed": "42"})
        self.assertEqual(kwargs, {"model": "seedvr2", "width": 1280, "height": 720, "frames": 61,
                                  "unet_name": "seedvr2_3b_int8_convrot.safetensors",
                                  "vae_name": "seedvr2_ema_vae_fp16.safetensors", "seed": 42})
        self.assertEqual(self.requested, [("diffusion_models", "seedvr2_3b_int8_convrot.safetensors"),
                                          ("vae", "seedvr2_ema_vae_fp16.safetensors")])
        self.assertIn("with seedvr2", note)
        headers = wrapper_routes._resolution_headers(kwargs, note)
        self.assertEqual(headers["X-Wrapper-Settings"], "width=1280 height=720 seed=42")
        # Without one, a random seed.
        self.assertIsInstance(self._setup({"model": "seedvr2"})[0]["seed"], int)
        self.assertEqual(headers["X-Wrapper-Models"], "seedvr2_3b_int8_convrot.safetensors, "
                                                      "seedvr2_ema_vae_fp16.safetensors")

    def test_capped_at_4x_says_so(self):
        self.probe = (320, 180, 61, 30.0)
        kwargs, note = self._setup({"target_short_side": "1080"})
        self.assertEqual((kwargs["width"], kwargs["height"]), (1280, 720))
        self.assertIn("capped at 4x, short of 1080", note)

    def test_refusals_happen_before_any_download(self):
        setup = wrapper_routes._setup_fx_upscale
        self.assertEqual(self._refused(setup, {"model": "waifu2x"}).message, "Invalid model")
        for target in ("0", "15", "5000", "big", "1080.5"):
            self.assertEqual(self._refused(setup, {"target_short_side": target}).message,
                             "Invalid target_short_side")
        self.assertEqual(self._refused(setup, {}, refs={}).message, "No video provided")
        self.probe = (640, 360, 0, 30.0)
        self.assertEqual(self._refused(setup, {}).message, "Video too short")
        self.assertEqual(self.requested, [])

    def test_over_budget_clip_is_refused(self):
        # 30 s of 360p30 to 1080p through the x4 model is 3.3e9 pixel-frames.
        self.probe = (640, 360, 900, 30.0)
        for model in ("esrgan", "seedvr2"):
            error = self._refused(wrapper_routes._setup_fx_upscale, {"model": model})
            self.assertIn("COMFY_FX_UPSCALE_MAX_PIXEL_FRAMES", error.details)
        self.assertEqual(self.requested, [])


class TestProbeImage(unittest.TestCase):
    def test_exif_orientation_turns_the_size(self):
        """LoadImage applies EXIF orientation, so coordinates and masks use the upright size."""
        import tempfile

        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "photo.jpg")
            exif = Image.Exif()
            exif[0x0112] = 6  # rotate 90 degrees clockwise
            Image.new("RGB", (40, 30)).save(path, exif=exif)
            self.assertEqual(wrapper_routes._probe_image(path), (30, 40))
            Image.new("RGB", (40, 30)).save(path)
            self.assertEqual(wrapper_routes._probe_image(path), (40, 30))


class _ProbedImageSetup(unittest.TestCase):
    """Setup handlers that probe the uploaded still: the probe and the
    downloads are mocked (no real image or model)."""

    probe = (1024, 768)

    def setUp(self):
        self._download_models = wrapper_routes._download_models
        self._probe_image = wrapper_routes._probe_image
        self.requested = []

        async def no_download(models, downloaded):
            self.requested.extend((m["folder"], m["filename"]) for m in models)
            return []

        wrapper_routes._download_models = no_download
        wrapper_routes._probe_image = lambda path: self.probe

    def tearDown(self):
        wrapper_routes._download_models = self._download_models
        wrapper_routes._probe_image = self._probe_image

    def _run(self, fields, refs=None):
        refs = {"image": ["wrapper/job/in.png"]} if refs is None else refs
        return asyncio.run(self.setup(fields, [], refs))

    def _refused(self, fields, refs=None):
        with self.assertRaises(wrapper_routes._SetupError) as ctx:
            self._run(fields, refs)
        return ctx.exception


class TestImageSegmentSetup(_ProbedImageSetup):
    setup = staticmethod(wrapper_routes._setup_image_segment)

    def test_phrase_fetches_sam3_and_builds(self):
        kwargs, note = self._run({"prompt": "the sky"})
        self.assertEqual(self.requested, [("checkpoints", "sam3.1_multiplex_fp16.safetensors")])
        self.assertEqual((kwargs["width"], kwargs["height"], kwargs["ckpt_name"]),
                         (1024, 768, "sam3.1_multiplex_fp16.safetensors"))
        self.assertEqual(kwargs["request"]["text"], "the sky:8")
        self.assertEqual(note, "1024x768 segmented by 'the sky:8'; saves the union mask")
        graph = wrapper_workflows.build_image_segment(image="wrapper/job/in.png", **kwargs)
        self.assertEqual(graph["4"]["inputs"]["text"], "the sky:8")
        headers = wrapper_routes._resolution_headers(kwargs, note)
        self.assertEqual(headers["X-Wrapper-Models"], "sam3.1_multiplex_fp16.safetensors")
        self.assertEqual(headers["X-Wrapper-Settings"], "width=1024 height=768")

    def test_boxes_are_checked_against_the_probed_size(self):
        kwargs, _ = self._run({"boxes": "[[1000, 700, 100, 100]]"})
        self.assertEqual(kwargs["request"]["boxes"], [{"x": 1000.0, "y": 700.0, "width": 24.0, "height": 68.0}])
        self.probe = (64, 64)
        error = self._refused({"boxes": "[[100, 100, 10, 10]]"})
        self.assertEqual(error.message, "Invalid segment request")
        self.assertIn("64x64", error.details)

    def test_refusals_happen_before_any_download(self):
        self.assertEqual(self._refused({}).message, "Invalid segment request")
        self.assertEqual(self._refused({"prompt": "sky", "boxes": "[[0,0,1,1]]"}).message, "Invalid segment request")
        self.assertEqual(self._refused({"prompt": "sky"}, refs={}).message, "No image provided")

        def broken(path):
            raise OSError("cannot identify image file")

        wrapper_routes._probe_image = broken
        error = self._refused({"prompt": "sky"})
        self.assertEqual(error.message, "Unreadable image")
        self.assertIn("cannot identify image file", error.details)
        self.assertEqual(self.requested, [])

    def test_over_budget_masks_are_refused(self):
        self.probe = (8000, 6000)  # 48 MP: fine for the union, not for 1 + 32 masks
        self._run({"prompt": "sky"})
        error = self._refused({"prompt": "sky", "individual_masks": "true", "max_objects": "32"})
        self.assertIn("COMFY_IMAGE_SEGMENT_MAX_PIXEL_MASKS", error.details)


class TestImageDescribeSetup(_ProbedImageSetup):
    setup = staticmethod(wrapper_routes._setup_image_describe)

    def test_defaults_reuse_the_qwen_image_21_encoder(self):
        kwargs, note = self._run({})
        self.assertEqual(kwargs, {"task": "both", "max_length": 1024, "megapixels": 0.75,
                                  "clip_name": wrapper_workflows.QWEN_IMAGE_21_CLIP})
        self.assertEqual(self.requested, [("text_encoders", wrapper_workflows.QWEN_IMAGE_21_CLIP)])
        self.assertEqual(note, "1024x768 described (both) at 0.75 MP")
        graph = wrapper_workflows.build_image_describe(image="wrapper/job/in.png", **kwargs)
        self.assertEqual(graph["5"]["inputs"]["max_length"], 1024)

    def test_large_stills_are_scaled_down(self):
        self.probe = (4000, 3000)
        kwargs, _ = self._run({"task": "caption", "max_length": "200"})
        self.assertEqual((kwargs["task"], kwargs["max_length"], kwargs["megapixels"]), ("caption", 200, 1.0))

    def test_refusals_happen_before_any_download(self):
        self.assertEqual(self._refused({"task": "tags"}).message, "Invalid describe request")
        self.assertEqual(self._refused({"max_length": "0"}).message, "Invalid describe request")
        self.assertEqual(self._refused({}, refs={}).message, "No image provided")
        self.assertEqual(self.requested, [])


class TestPromptRewriteHook(unittest.TestCase):
    """The glue between an incoming request and the rewriter: which mode it
    reports, which labels it declares, and which upload it uses for visual
    context."""

    def setUp(self):
        from api_wrapper import prompt_rewriter

        self.calls = []
        self._rewrite = prompt_rewriter.rewrite
        self._data_url = prompt_rewriter.data_url_from_path

        async def fake_rewrite(**kwargs):
            self.calls.append(kwargs)
            return "REWRITTEN", "mimo-v2.5-pro"

        prompt_rewriter.rewrite = fake_rewrite
        prompt_rewriter.data_url_from_path = lambda path: f"data:image/png;base64,#{path}"

    def tearDown(self):
        from api_wrapper import prompt_rewriter

        prompt_rewriter.rewrite = self._rewrite
        prompt_rewriter.data_url_from_path = self._data_url

    def _run(self, task_name, build_kwargs, upload_refs, fields=None):
        base = {"prompt": "a wrestler chokeslams another", "duration": 5.0,
                "width": 864, "height": 480}
        base.update(build_kwargs)
        return asyncio.run(wrapper_routes._rewrite_prompt(
            task_name, fields or {}, base, upload_refs))

    def test_image_task_reports_i2va_and_grounds_on_the_first_frame(self):
        prompt, note, record = self._run("image", {"first_frame": "wrapper/a.png"},
                                         {"image": ["wrapper/a.png"]})
        self.assertEqual(record["prompt"], "REWRITTEN")
        self.assertEqual(record["source"], "mimo-v2.5-pro")
        self.assertEqual(record["mode"], "I2VA")
        self.assertEqual(record["input_prompt"], "a wrestler chokeslams another")
        self.assertEqual(prompt, "REWRITTEN")
        call = self.calls[0]
        self.assertEqual(call["mode"], "I2VA")
        self.assertEqual(call["frames"], 124)  # 5.0s snapped to the H3 grid
        self.assertAlmostEqual(call["duration"], 124 / 24, places=2)
        # no llm_image, so the keyframe itself is the visual context
        self.assertIn("wrapper/a.png", call["image"])
        self.assertIn("visual context", note)

    def test_llm_image_wins_over_the_keyframe(self):
        self._run("image", {"first_frame": "wrapper/a.png", "last_frame": "wrapper/b.png"},
                  {"image": ["wrapper/a.png"], "last_frame": ["wrapper/b.png"],
                   "llm_image": ["wrapper/ctx.png"]})
        call = self.calls[0]
        self.assertEqual(call["mode"], "FL2VA")  # both keyframes present
        self.assertIn("wrapper/ctx.png", call["image"])

    def test_text_task_without_uploads_sends_no_image(self):
        _, note, record = self._run("text", {}, {})
        self.assertIsNone(record["context_image"])
        call = self.calls[0]
        self.assertEqual(call["mode"], "T2VA")
        self.assertIsNone(call["image"])
        self.assertNotIn("visual context", note)

    def test_reference_counts_include_the_video_soundtrack_label(self):
        self._run("reference", {}, {
            "ref_images": ["wrapper/1.png", "wrapper/2.png"],
            "ref_videos": ["wrapper/v.mp4"],
            "ref_video_audios": ["wrapper/va.wav"],
            "ref_audios": ["wrapper/a1.wav", "wrapper/a2.wav"],
        })
        call = self.calls[0]
        self.assertEqual(call["mode"], "Ref2VA")
        # a reference video's soundtrack gets its own <Audio j>, so 1 + 2 = 3
        self.assertEqual(call["reference_counts"], {"images": 2, "videos": 1, "audios": 3})
        self.assertIn("wrapper/1.png", call["image"])

    def test_llm_model_field_is_forwarded(self):
        self._run("text", {}, {}, fields={"llm_model": "mimo-v2.5"})
        self.assertEqual(self.calls[0]["model"], "mimo-v2.5")


class TestPromptRecord(unittest.TestCase):
    """The H3 prompt is generated rather than supplied, so the only record of
    what a video was actually made from is the one the wrapper writes."""

    def setUp(self):
        import folder_paths

        self.video = os.path.join(folder_paths.get_output_directory(), "wrapper", "record_test.mp4")
        os.makedirs(os.path.dirname(self.video), exist_ok=True)
        with open(self.video, "wb") as f:
            f.write(b"MP4")
        self.sidecar = os.path.splitext(self.video)[0] + ".prompt.json"

    def tearDown(self):
        for path in (self.video, self.sidecar):
            if os.path.exists(path):
                os.remove(path)

    def test_sidecar_lands_next_to_the_video_and_round_trips(self):
        import json

        record = {"source": "mimo-v2.5", "mode": "I2VA", "job_id": "abc",
                  "input_prompt": "two wrestlers", "prompt": "integrated_multimodal_description: ...",
                  "settings": {"width": 864, "duration": 5.17}}
        path = wrapper_routes._save_prompt_record(self.video, record)
        self.assertEqual(path, self.sidecar)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), record)

    def test_non_ascii_prompt_survives(self):
        """H3 keeps dialogue in its original language, so the record must not
        be ascii-escaped into something unreadable."""
        import json

        record = {"prompt": '一个男人说: <d>[Chinese] 我们走吧。</d>', "source": "mimo-v2.5"}
        with open(wrapper_routes._save_prompt_record(self.video, record), encoding="utf-8") as f:
            self.assertIn("我们走吧", f.read())
        with open(self.sidecar, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["prompt"], record["prompt"])

    def test_unwritable_target_does_not_lose_the_video(self):
        missing = os.path.join(self.video, "no", "such", "dir", "x.mp4")
        self.assertIsNone(wrapper_routes._save_prompt_record(missing, {"prompt": "x"}))

    def test_logging_includes_the_generated_prompt(self):
        record = {"source": "mimo-v2.5", "mode": "Ref2VA",
                  "input_prompt": "a cat", "prompt": "THE GENERATED H3 PROMPT"}
        with self.assertLogs(level="INFO") as logs:
            wrapper_routes._log_prompt_record(record, "job-1", "minimaxh3", "reference")
        joined = "\n".join(logs.output)
        self.assertIn("THE GENERATED H3 PROMPT", joined)
        self.assertIn("mimo-v2.5", joined)
        self.assertIn("Ref2VA", joined)
        self.assertIn("job-1", joined)
        self.assertIn("a cat", joined)

    def test_verbatim_raw_prompt_is_not_logged_as_a_rewrite(self):
        record = {"source": "raw_prompt", "mode": "I2VA",
                  "input_prompt": None, "prompt": "MY OWN PROMPT"}
        with self.assertLogs(level="INFO") as logs:
            wrapper_routes._log_prompt_record(record, "job-2", "minimaxh3", "image")
        joined = "\n".join(logs.output)
        self.assertIn("MY OWN PROMPT", joined)
        self.assertNotIn("rewritten from", joined)


if __name__ == "__main__":
    unittest.main()


class TestJobProvenance(unittest.TestCase):
    """What a job ran with, for a caller that polls and never sees the headers."""

    def test_raw_prompt_jobs_report_seed_models_and_settings(self):
        provenance = wrapper_routes._job_provenance(
            {"seed": 42, "unet_name": "h3_ref2va_int8.safetensors", "clip_name": "enc.safetensors",
             "width": 864, "height": 480, "steps": 20, "prompt": "subject_definitions: ..."},
            "raw_prompt used verbatim; no LLM rewrite", {"source": "raw_prompt", "mode": "Ref2VA"})
        self.assertEqual(provenance["seed"], 42)
        self.assertEqual(provenance["models"], ["h3_ref2va_int8.safetensors", "enc.safetensors"])
        self.assertEqual(provenance["settings"], {"width": 864, "height": 480, "steps": 20})
        self.assertEqual((provenance["prompt_source"], provenance["mode"]), ("raw_prompt", "Ref2VA"))
        self.assertNotIn("prompt", provenance)  # the caller sent it; no need to echo it

    def test_a_rewritten_prompt_is_kept(self):
        provenance = wrapper_routes._job_provenance(
            {"seed": 1, "prompt": "integrated_multimodal_description: [Shot 1] ..."}, None,
            {"source": "mimo", "mode": "T2VA"})
        self.assertEqual(provenance["prompt"], "integrated_multimodal_description: [Shot 1] ...")

    def test_the_record_is_bounded_newest_first(self):
        saved = dict(wrapper_routes._PROVENANCE)
        try:
            wrapper_routes._PROVENANCE.clear()
            for i in range(wrapper_routes._PROVENANCE_MAX + 5):
                wrapper_routes._remember_provenance(f"job-{i}", {"seed": i})
            self.assertEqual(len(wrapper_routes._PROVENANCE), wrapper_routes._PROVENANCE_MAX)
            self.assertNotIn("job-0", wrapper_routes._PROVENANCE)
            self.assertIn(f"job-{wrapper_routes._PROVENANCE_MAX + 4}", wrapper_routes._PROVENANCE)
        finally:
            wrapper_routes._PROVENANCE.clear()
            wrapper_routes._PROVENANCE.update(saved)
