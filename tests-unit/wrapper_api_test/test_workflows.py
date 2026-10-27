"""Unit tests for the wrapper API workflow builder, model requirements and
OpenAPI spec. The graph validation test at the end only runs when torch is
available (it loads the real node registry)."""

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api_wrapper import openapi as wrapper_openapi
from api_wrapper import workflows as wrapper_workflows
from api_wrapper.workflows import (
    FLUX2_KLEIN_9B_CLIP,
    FLUX2_KLEIN_9B_MODELS,
    FLUX2_KLEIN_9B_UNET,
    FLUX2_KLEIN_9B_VAE,
    build_flux2_klein_9b_img2img,
)

EXPECTED_CLASS_TYPES = {
    "UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "ImageScaleToTotalPixels",
    "GetImageSize", "CLIPTextEncode", "VAEEncode", "ReferenceLatent", "CFGGuider",
    "KSamplerSelect", "Flux2Scheduler", "RandomNoise", "EmptyFlux2LatentImage",
    "SamplerCustomAdvanced", "VAEDecode", "SaveImage",
}


def _sample_graph():
    return build_flux2_klein_9b_img2img(
        prompt="make it snow",
        image="wrapper/abc123.png",
        negative_prompt="blurry",
        seed=42,
        steps=25,
        cfg=4.5,
        megapixels=1.5,
    )


class TestWorkflowGraph(unittest.TestCase):
    def test_node_count_and_class_types(self):
        graph = _sample_graph()
        self.assertEqual(len(graph), 19)
        self.assertEqual({node["class_type"] for node in graph.values()}, EXPECTED_CLASS_TYPES)

    def test_links_reference_existing_nodes(self):
        graph = _sample_graph()
        for node_id, node in graph.items():
            for input_name, value in node["inputs"].items():
                if isinstance(value, list):
                    src_id, src_index = value
                    self.assertIn(src_id, graph, f"{node_id}.{input_name} links to unknown node {src_id}")
                    self.assertIsInstance(src_index, int)
                    self.assertGreaterEqual(src_index, 0)

    def test_output_node_present(self):
        graph = _sample_graph()
        output_nodes = [n for n in graph.values() if n["class_type"] == "SaveImage"]
        self.assertEqual(len(output_nodes), 1)
        self.assertIsInstance(output_nodes[0]["inputs"]["images"], list)

    def test_loader_filenames_match_model_requirements(self):
        graph = _sample_graph()
        unet = next(n for n in graph.values() if n["class_type"] == "UNETLoader")
        clip = next(n for n in graph.values() if n["class_type"] == "CLIPLoader")
        vae = next(n for n in graph.values() if n["class_type"] == "VAELoader")
        self.assertEqual(unet["inputs"]["unet_name"], FLUX2_KLEIN_9B_UNET)
        self.assertEqual(clip["inputs"]["clip_name"], FLUX2_KLEIN_9B_CLIP)
        self.assertEqual(clip["inputs"]["type"], "flux2")
        self.assertEqual(vae["inputs"]["vae_name"], FLUX2_KLEIN_9B_VAE)
        filenames = {m["filename"] for m in FLUX2_KLEIN_9B_MODELS}
        self.assertEqual(filenames, {FLUX2_KLEIN_9B_UNET, FLUX2_KLEIN_9B_CLIP, FLUX2_KLEIN_9B_VAE})

    def test_sampler_settings(self):
        graph = _sample_graph()
        guider = next(n for n in graph.values() if n["class_type"] == "CFGGuider")
        scheduler = next(n for n in graph.values() if n["class_type"] == "Flux2Scheduler")
        noise = next(n for n in graph.values() if n["class_type"] == "RandomNoise")
        self.assertEqual(guider["inputs"]["cfg"], 4.5)
        self.assertEqual(scheduler["inputs"]["steps"], 25)
        self.assertEqual(noise["inputs"]["noise_seed"], 42)


class TestKleinEditReferences(unittest.TestCase):
    def test_several_references_chain_one_reference_latent_each(self):
        graph = build_flux2_klein_9b_img2img(
            prompt="the same man, at the bus stop", ref_images=["wrapper/face.png", "wrapper/stop.png"],
            width=1024, height=576)
        self.assertEqual(graph["4"]["inputs"]["image"], "wrapper/face.png")
        self.assertEqual(graph["20"]["inputs"]["image"], "wrapper/stop.png")
        # face -> stop, on both sides of the guidance
        self.assertEqual(graph["23"]["inputs"], {"conditioning": ["10", 0], "latent": ["22", 0]})
        self.assertEqual(graph["24"]["inputs"], {"conditioning": ["11", 0], "latent": ["22", 0]})
        self.assertEqual(graph["12"]["inputs"]["positive"], ["23", 0])
        self.assertEqual(graph["12"]["inputs"]["negative"], ["24", 0])
        # the canvas is the one asked for, not the first reference's
        self.assertEqual(graph["16"]["inputs"]["width"], 1024)
        self.assertEqual(graph["14"]["inputs"]["height"], 576)
        self.assertEqual(graph["5"]["inputs"]["upscale_method"], "lanczos")

    def test_image_and_ref_images_combine_image_first(self):
        graph = build_flux2_klein_9b_img2img(prompt="x", image="wrapper/a.png", ref_images="wrapper/b.png")
        self.assertEqual(graph["4"]["inputs"]["image"], "wrapper/a.png")
        self.assertEqual(graph["20"]["inputs"]["image"], "wrapper/b.png")
        self.assertEqual(graph["16"]["inputs"]["width"], ["6", 0])  # no canvas: first image's size

    def test_no_image_at_all_is_refused(self):
        with self.assertRaises(ValueError):
            build_flux2_klein_9b_img2img(prompt="x")

    def test_either_upload_field_satisfies_the_edit(self):
        entry = wrapper_workflows.WORKFLOWS["flux2klein9b"]
        self.assertEqual(entry["image_fields"], ["image", "ref_images"])
        self.assertEqual(entry["uploads"]["ref_images"]["max"], 3)


class TestQwenImage21Graphs(unittest.TestCase):
    def test_txt2img_mirrors_the_template(self):
        graph = wrapper_workflows.build_qwen_image_21_text2img(
            prompt="a cat", negative_prompt="blurry", seed=7, width=1344, height=768)
        self.assertEqual(graph["2"]["inputs"]["type"], "qwen_image")
        self.assertEqual(graph["4"]["inputs"]["prompt"], "a cat")
        self.assertEqual(graph["4"]["inputs"]["negative_prompt"], "blurry")
        self.assertEqual((graph["5"]["inputs"]["width"], graph["5"]["inputs"]["height"]), (1344, 768))
        sampler = graph["6"]["inputs"]
        self.assertEqual((sampler["seed"], sampler["steps"], sampler["cfg"]), (7, 25, 1.0))
        self.assertEqual((sampler["sampler_name"], sampler["scheduler"]), ("euler", "simple"))
        self.assertEqual((sampler["positive"], sampler["negative"]), (["4", 0], ["4", 1]))
        self.assertEqual(graph["8"]["class_type"], "SaveImage")

    def test_edit_images_are_flat_autogrow_slots_image_first(self):
        graph = wrapper_workflows.build_qwen_image_21_edit(
            prompt="<image1> wears the shirt from <image2>", image="wrapper/a.png",
            ref_images=["wrapper/b.png", "wrapper/c.png"])
        encode = graph["4"]["inputs"]
        self.assertEqual(encode["images.image_1"], ["10", 0])
        self.assertEqual(encode["images.image_3"], ["12", 0])
        self.assertNotIn("images", encode)
        self.assertEqual([graph[n]["inputs"]["image"] for n in ("10", "11", "12")],
                         ["wrapper/a.png", "wrapper/b.png", "wrapper/c.png"])
        self.assertEqual(encode["vae"], ["3", 0])
        self.assertEqual(encode["resolution"], 1024)
        # no canvas: the encoder's latent, on image 1's size
        self.assertNotIn("5", graph)
        self.assertEqual(graph["6"]["inputs"]["latent_image"], ["4", 2])

    def test_edit_canvas_uses_an_empty_latent(self):
        graph = wrapper_workflows.build_qwen_image_21_edit(
            prompt="x", ref_images="wrapper/a.png", width=1024, height=576)
        self.assertEqual(graph["5"]["inputs"]["width"], 1024)
        self.assertEqual(graph["6"]["inputs"]["latent_image"], ["5", 0])

    def test_edit_without_an_image_is_refused(self):
        with self.assertRaises(ValueError):
            wrapper_workflows.build_qwen_image_21_edit(prompt="x")

    def test_registered_like_flux2klein9b(self):
        edit = wrapper_workflows.WORKFLOWS["qwenimage21"]
        self.assertIs(edit["build"], wrapper_workflows.build_qwen_image_21_edit)
        self.assertEqual(edit["image_fields"], ["image", "ref_images"])
        # TextEncodeQwenImage21 takes 16 images
        self.assertEqual(edit["uploads"]["image"]["max"] + edit["uploads"]["ref_images"]["max"], 16)
        txt2img = wrapper_workflows.WORKFLOWS["qwenimage21-txt2img"]
        self.assertFalse(txt2img["requires_image"])
        self.assertIs(txt2img["build"], wrapper_workflows.build_qwen_image_21_text2img)

    def test_models_come_from_the_qwen_image_21_repo(self):
        for quant in ("int8", "bf16"):
            models = wrapper_workflows.qwen_image_21_models(quant)
            self.assertEqual([m["folder"] for m in models], ["diffusion_models", "text_encoders", "vae"])
            for model in models:
                self.assertEqual(model["url"], f"{wrapper_workflows.QWEN_IMAGE_21_BASE_URL}/{model['folder']}/{model['filename']}")

    def test_openapi_documents_both_endpoints(self):
        spec = wrapper_openapi.spec_with_workflows(wrapper_workflows.WORKFLOWS)
        edit = spec["paths"]["/api/wrapper/qwenimage21/generate"]["post"]
        schema = edit["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(schema["required"], ["prompt"])  # image or ref_images
        self.assertIn("<image1>", schema["properties"]["prompt"]["description"])
        self.assertEqual(schema["properties"]["quantization"]["enum"], ["int8", "bf16"])
        self.assertEqual(schema["properties"]["steps"]["default"], 25)
        self.assertEqual(schema["properties"]["cfg"]["default"], 1.0)
        self.assertNotIn("megapixels", schema["properties"])
        for prop in ("image", "ref_images", "resolution", "width", "height"):
            self.assertIn(prop, schema["properties"])
        txt2img = spec["paths"]["/api/wrapper/qwenimage21-txt2img/generate"]["post"]
        schema = txt2img["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertNotIn("image", schema["properties"])
        self.assertEqual(schema["properties"]["width"]["default"], 1024)


class TestModelRequirements(unittest.TestCase):
    def test_three_models_with_folders_and_urls(self):
        self.assertEqual(len(FLUX2_KLEIN_9B_MODELS), 3)
        for model in FLUX2_KLEIN_9B_MODELS:
            self.assertIn(model["folder"], {"diffusion_models", "text_encoders", "vae"})
            self.assertTrue(model["filename"].endswith(".safetensors"))
            self.assertTrue(model["url"].startswith("https://huggingface.co/"))


class TestWorkflowRegistry(unittest.TestCase):
    def test_flux2klein9b_registered(self):
        from api_wrapper import workflows as workflows_module
        self.assertIn("flux2klein9b", workflows_module.WORKFLOWS)
        entry = workflows_module.WORKFLOWS["flux2klein9b"]
        self.assertTrue(entry["requires_image"])
        self.assertIs(entry["build"], workflows_module.build_flux2_klein_9b_img2img)

    def test_ideogram4_registered(self):
        from api_wrapper import workflows as workflows_module
        self.assertIn("ideogram4", workflows_module.WORKFLOWS)
        entry = workflows_module.WORKFLOWS["ideogram4"]
        self.assertFalse(entry["requires_image"])
        self.assertIs(entry["build"], workflows_module.build_ideogram4_text2img)
        self.assertIn("high_level_description", entry["example_prompt"])
        self.assertIn("mode", entry["extra_form_properties"])

    def test_minimaxh3_registered(self):
        from api_wrapper import workflows as workflows_module
        self.assertIn("minimaxh3", workflows_module.WORKFLOWS)
        entry = workflows_module.WORKFLOWS["minimaxh3"]
        self.assertEqual(sorted(entry["tasks"]), ["image", "reference", "text"])
        self.assertEqual(entry["output_type"], "video")
        self.assertIn("uploads", entry["tasks"]["reference"])
        # The node's own limit (Autogrow max=9), not the official workflow's 3.
        self.assertEqual(entry["tasks"]["reference"]["uploads"]["ref_images"]["max"], 9)
        self.assertEqual(entry["tasks"]["reference"]["uploads"]["ref_videos"]["max"], 1)
        self.assertEqual(entry["tasks"]["reference"]["uploads"]["ref_video_audios"]["max"], 1)
        self.assertEqual(entry["tasks"]["reference"]["uploads"]["ref_audios"]["max"], 3)

    def test_klein_txt2img_registered(self):
        from api_wrapper import workflows as workflows_module
        self.assertIn("flux2klein9b-txt2img", workflows_module.WORKFLOWS)
        entry = workflows_module.WORKFLOWS["flux2klein9b-txt2img"]
        self.assertFalse(entry["requires_image"])
        self.assertIs(entry["build"], workflows_module.build_flux2_klein_9b_text2img)
        self.assertIn("width", entry["extra_form_properties"])


class TestOpenAPISpec(unittest.TestCase):
    def test_spec_has_wrapper_paths(self):
        spec = wrapper_openapi.WRAPPER_OPENAPI_SPEC
        self.assertEqual(spec["openapi"], "3.0.3")
        paths = spec["paths"]
        self.assertIn("/api/wrapper/{workflow}/generate", paths)
        self.assertIn("/api/wrapper/workflows", paths)
        self.assertIn("/api/wrapper/jobs/{job_id}", paths)
        self.assertIn("/api/wrapper/jobs/{job_id}/image", paths)
        generate = paths["/api/wrapper/{workflow}/generate"]["post"]
        self.assertEqual(generate["requestBody"]["content"]["multipart/form-data"]["schema"]["required"],
                         ["prompt", "image"])
        form = generate["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]
        self.assertIn("vram", form)
        self.assertEqual(form["vram"]["enum"], ["auto", "low", "normal", "high"])

    def test_minimax_nvfp4_in_expanded_spec(self):
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows(workflows_module.WORKFLOWS)
        minimax = spec["paths"]["/api/wrapper/minimaxh3/text/generate"]["post"]
        quantization = minimax["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]["quantization"]
        self.assertEqual(quantization["enum"], ["fp8", "int8", "bf16", "nvfp4"])
        # the vram field survives per-workflow filtering for every workflow
        for path in ("/api/wrapper/flux2klein9b/generate", "/api/wrapper/ideogram4/generate",
                     "/api/wrapper/minimaxh3/image/generate"):
            props = spec["paths"][path]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]
            self.assertIn("vram", props)

    def test_spec_expands_to_concrete_workflow_paths(self):
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows({"flux2klein9b": workflows_module.WORKFLOWS["flux2klein9b"],
                                                    "video9b": {"build": None, "title": "x"}})
        paths = spec["paths"]
        self.assertNotIn("/api/wrapper/{workflow}/generate", paths)
        self.assertIn("/api/wrapper/flux2klein9b/generate", paths)
        self.assertIn("/api/wrapper/video9b/generate", paths)
        flux2 = paths["/api/wrapper/flux2klein9b/generate"]["post"]
        self.assertEqual([p.get("name") for p in flux2.get("parameters", [])], [])  # no leftover {workflow} param

    def test_spec_folds_workflow_example_and_extra_fields(self):
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows({"ideogram4": workflows_module.WORKFLOWS["ideogram4"]})
        operation = spec["paths"]["/api/wrapper/ideogram4/generate"]["post"]
        schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertIn("high_level_description", schema["properties"]["prompt"]["description"])
        self.assertIn("mode", schema["properties"])
        self.assertEqual(schema["properties"]["mode"]["enum"], ["default", "quality", "turbo"])
        # ideogram4 is text-to-image: no image field, image not required
        self.assertNotIn("image", schema["properties"])
        self.assertEqual(schema["required"], ["prompt"])

    def test_spec_keeps_image_required_only_for_image_workflows(self):
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows(workflows_module.WORKFLOWS)
        flux2 = spec["paths"]["/api/wrapper/flux2klein9b/generate"]["post"]
        flux2_schema = flux2["requestBody"]["content"]["multipart/form-data"]["schema"]
        # image OR ref_images: one-of, so neither is required on its own
        self.assertEqual(flux2_schema["required"], ["prompt"])
        self.assertIn("megapixels", flux2_schema["properties"])
        self.assertIn("ref_images", flux2_schema["properties"])
        i2v = spec["paths"]["/api/wrapper/minimaxh3/image/generate"]["post"]
        i2v_schema = i2v["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertIn("image", i2v_schema["required"])  # a single image field stays required

    def test_spec_emits_task_paths_and_video_response(self):
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows(workflows_module.WORKFLOWS)
        for path in ("/api/wrapper/minimaxh3/text/generate",
                     "/api/wrapper/minimaxh3/image/generate",
                     "/api/wrapper/minimaxh3/reference/generate"):
            self.assertIn(path, spec["paths"], f"missing {path}")
        # the bare /{name}/{task} alias was removed as redundant
        self.assertNotIn("/api/wrapper/minimaxh3/reference", spec["paths"])
        op = spec["paths"]["/api/wrapper/minimaxh3/reference/generate"]["post"]
        content = op["responses"]["200"]["content"]
        self.assertEqual(list(content)[0], "video/mp4")
        schema = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        # prompt/raw_prompt are one-of on a rewriting task, so neither is required
        self.assertEqual(schema["required"], [])
        for prop in ("prompt", "raw_prompt", "llm_image", "llm_model",
                     "ref_images", "ref_videos", "ref_video_audios", "ref_audios",
                     "ref_image_size", "duration", "image", "last_frame"):
            self.assertIn(prop, schema["properties"])
        # the pinned first frame is optional: only the prompt pair is "one of"
        self.assertNotIn("image", schema["required"])
        self.assertIn("first frame", schema["properties"]["image"]["description"])
        self.assertEqual(schema["properties"]["quantization"]["enum"], ["fp8", "int8", "bf16", "nvfp4"])
        # the image task requires an image upload and returns video
        op = spec["paths"]["/api/wrapper/minimaxh3/image/generate"]["post"]
        schema = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(schema["required"], ["image"])
        self.assertEqual(list(op["responses"]["200"]["content"])[0], "video/mp4")

    def test_generate_response_is_an_image(self):
        """The 200 response must list the workflow's media type first so
        Swagger UI shows the artifact inline instead of a generic download."""
        from api_wrapper import workflows as workflows_module
        from api_wrapper.openapi import OUTPUT_MEDIA_TYPES
        spec = wrapper_openapi.spec_with_workflows(workflows_module.WORKFLOWS)
        for path, path_item in spec["paths"].items():
            if not path.endswith("/generate") or "post" not in path_item:
                continue
            workflow_name = path.split("/")[3]
            entry = workflows_module.WORKFLOWS[workflow_name]
            expected = OUTPUT_MEDIA_TYPES.get(entry.get("output_type"), "image/png")
            content = path_item["post"]["responses"]["200"]["content"]
            self.assertEqual(list(content)[0], expected,
                             f"{path}: expected {expected} first, got {list(content)}")

    def test_spec_paths_are_valid_openapi_shape(self):
        """Paths must map to HTTP-method-keyed path items (Swagger UI silently
        drops operations that are not wrapped in a method key)."""
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows(workflows_module.WORKFLOWS)
        operation_ids = []
        for path, path_item in spec["paths"].items():
            for method in ("get", "post"):
                if method in path_item:
                    operation = path_item[method]
                    operation_ids.append(operation["operationId"])
                    # concrete paths must not declare the removed {workflow} param
                    self.assertNotIn("workflow", [p.get("name") for p in operation.get("parameters", [])])
        self.assertEqual(len(operation_ids), len(set(operation_ids)),
                         f"duplicate operationIds: {operation_ids}")
        self.assertIn("generateFlux2klein9b", operation_ids)
        self.assertIn("generateIdeogram4", operation_ids)

    def test_swagger_html_points_at_wrapper_spec(self):
        self.assertIn("/api/wrapper/openapi.json", wrapper_openapi.WRAPPER_SWAGGER_HTML)
        self.assertIn("swagger-ui", wrapper_openapi.WRAPPER_SWAGGER_HTML)


class TestKleinTxt2imgGraph(unittest.TestCase):
    def test_structure_and_sampler(self):
        graph = wrapper_workflows.build_flux2_klein_9b_text2img(
            prompt="a cat", negative_prompt="blurry", seed=7, steps=25, cfg=4.5,
            width=512, height=384)
        self.assertEqual(len(graph), 13)
        ids = set(graph)
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], ids)
        self.assertNotIn("LoadImage", [n["class_type"] for n in graph.values()])
        self.assertEqual(graph["4"]["inputs"]["text"], "a cat")
        self.assertEqual(graph["5"]["inputs"]["text"], "blurry")
        self.assertEqual(graph["6"]["inputs"]["cfg"], 4.5)
        self.assertEqual(graph["8"]["inputs"]["steps"], 25)
        self.assertEqual(graph["8"]["inputs"]["width"], 512)
        self.assertEqual(graph["10"]["inputs"]["height"], 384)
        self.assertEqual(graph["9"]["inputs"]["noise_seed"], 7)

    def test_defaults(self):
        graph = wrapper_workflows.build_flux2_klein_9b_text2img(prompt="x")
        self.assertEqual(graph["8"]["inputs"]["width"], 1024)
        self.assertEqual(graph["10"]["inputs"]["height"], 1024)
        self.assertEqual(graph["13"]["inputs"]["filename_prefix"], "wrapper/flux2klein9b_txt2img")


class TestMiniMaxH3Graph(unittest.TestCase):
    def test_text_to_video_structure(self):
        graph = wrapper_workflows.build_minimax_h3_text_to_video(
            prompt="a cat walks", seed=3, steps=40, width=1344, height=768,
            duration=5.0, scheduler="beta")
        ids = set(graph)
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], ids)
        self.assertEqual(graph["1"]["class_type"], "UNETLoader")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], wrapper_workflows.MINIMAX_H3_UNET_INT8)
        self.assertEqual(graph["3"]["inputs"]["type"], "minimax")
        self.assertEqual(graph["6"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(graph["6"]["inputs"]["length"], 120)  # 5s * 24fps
        self.assertEqual(graph["7"]["class_type"], "BasicGuider")
        self.assertEqual(graph["9"]["class_type"], "KSamplerSelect")
        self.assertEqual(graph["9"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(graph["10"]["inputs"]["steps"], 40)
        self.assertEqual(graph["13"]["class_type"], "VAEDecodeAudio")
        self.assertEqual(graph["14"]["class_type"], "CreateVideo")
        self.assertEqual(graph["14"]["inputs"]["fps"], 24.0)
        self.assertEqual(graph["14"]["inputs"]["audio"], ["13", 0])
        self.assertEqual(graph["15"]["class_type"], "SaveVideo")
        self.assertEqual(graph["15"]["inputs"]["video"], ["14", 0])
        self.assertEqual(graph["15"]["inputs"]["filename_prefix"], "wrapper/minimaxh3_t2v")

    def test_image_to_video_wires_frames(self):
        graph = wrapper_workflows.build_minimax_h3_image_to_video(
            prompt="x", first_frame="wrapper/a.png", last_frame="wrapper/b.png")
        self.assertEqual(graph["7"]["class_type"], "LoadImage")
        self.assertEqual(graph["8"]["class_type"], "LoadImage")
        # each frame is cover-cropped to the canvas before the node sees it
        self.assertEqual(graph["6"]["inputs"]["first_frame"], ["9", 0])
        self.assertEqual(graph["6"]["inputs"]["last_frame"], ["10", 0])
        for scale, loader in (("9", "7"), ("10", "8")):
            self.assertEqual(graph[scale]["class_type"], "ImageScale")
            self.assertEqual(graph[scale]["inputs"]["image"], [loader, 0])
            self.assertEqual(graph[scale]["inputs"]["crop"], "center")
            self.assertEqual((graph[scale]["inputs"]["width"], graph[scale]["inputs"]["height"]),
                             (graph["6"]["inputs"]["width"], graph["6"]["inputs"]["height"]))

    def test_reference_to_video_takes_three_voices(self):
        # A cast of three speakers, each with its voice-timbre clip: the node's
        # own limit, and what Flow sends once its cap is raised to match.
        graph = wrapper_workflows.build_minimax_h3_reference_to_video(
            prompt="<Audio 1> <Audio 2> <Audio 3>",
            ref_audios=["wrapper/a1.wav", "wrapper/a2.wav", "wrapper/a3.wav"])
        audios = sorted(k for k in graph["6"]["inputs"] if k.startswith("ref_audios."))
        self.assertEqual(audios, ["ref_audios.ref_audio_0", "ref_audios.ref_audio_1",
                                  "ref_audios.ref_audio_2"])
        for key in audios:
            node = graph[graph["6"]["inputs"][key][0]]
            self.assertEqual(node["class_type"], "LoadAudio")

    def test_reference_to_video_wires_nine_images(self):
        images = [f"wrapper/i{i}.png" for i in range(9)]
        graph = wrapper_workflows.build_minimax_h3_reference_to_video(
            prompt=" ".join(f"<Picture {i + 1}>" for i in range(9)), ref_images=images)
        slots = {k: v for k, v in graph["6"]["inputs"].items() if k.startswith("ref_images.")}
        self.assertEqual(sorted(slots), [f"ref_images.ref_image_{i}" for i in range(9)])
        self.assertEqual([graph[slots[f"ref_images.ref_image_{i}"][0]]["inputs"]["image"]
                          for i in range(9)], images)

    def test_reference_to_video_wires_refs(self):
        graph = wrapper_workflows.build_minimax_h3_reference_to_video(
            prompt="<Picture 1> <Video 1> <Audio 1>", ref_image_size="max",
            ref_images=["wrapper/i1.png", "wrapper/i2.png", "wrapper/i3.png"],
            ref_videos=["wrapper/v1.mp4"], ref_video_audios=["wrapper/v1.wav"],
            ref_audios=["wrapper/a1.wav", "wrapper/a2.wav"])
        self.assertEqual(graph["6"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(graph["6"]["inputs"]["ref_image_size"], "max")
        # Autogrow slots are addressed by flat dotted ids; a nested dict is
        # dropped by execution (see test_reference_inputs_reach_the_node).
        refs = {k: v for k, v in graph["6"]["inputs"].items() if k.startswith("ref_") and "." in k}
        self.assertEqual(refs, {
            "ref_images.ref_image_0": ["7", 0], "ref_images.ref_image_1": ["8", 0],
            "ref_images.ref_image_2": ["9", 0], "ref_videos.ref_video_0": ["11", 0],
            "ref_video_audios.ref_video_audio_0": ["12", 0],
            "ref_audios.ref_audio_0": ["13", 0], "ref_audios.ref_audio_1": ["14", 0]})
        self.assertNotIn("ref_images", graph["6"]["inputs"])
        self.assertEqual(graph["7"]["class_type"], "LoadImage")
        self.assertEqual(graph["10"]["class_type"], "LoadVideo")
        # the slot takes frames, so the video goes through GetVideoComponents
        self.assertEqual(graph["11"], {"class_type": "GetVideoComponents", "inputs": {"video": ["10", 0]}})
        self.assertEqual(graph["12"]["class_type"], "LoadAudio")  # video soundtrack
        self.assertEqual(graph["13"]["class_type"], "LoadAudio")
        # single-ref uploads (max=1) arrive as a bare filename string
        graph3 = wrapper_workflows.build_minimax_h3_reference_to_video(
            prompt="x", ref_videos="wrapper/v1.mp4", ref_video_audios="wrapper/v1.wav")
        self.assertEqual(graph3["6"]["inputs"]["ref_videos.ref_video_0"], ["8", 0])
        self.assertEqual(graph3["6"]["inputs"]["ref_video_audios.ref_video_audio_0"], ["9", 0])
        # no refs -> no loader nodes, tail starts at 7
        graph2 = wrapper_workflows.build_minimax_h3_reference_to_video(prompt="x")
        self.assertEqual(graph2["7"]["class_type"], "BasicGuider")
        self.assertFalse([k for k in graph2["6"]["inputs"] if k.startswith("ref_") and k != "ref_image_size"])

    def test_reference_to_video_pins_first_and_last_frames(self):
        """Keyframe completion: the frames are anchored with MiniMaxH3AddGuide
        on the reference conditioning, cover-cropped to the canvas, and the
        sampler is guided by the last guide in the chain."""
        graph = wrapper_workflows.build_minimax_h3_reference_to_video(
            prompt="<Picture 1> <Picture 2>", width=864, height=480,
            ref_images=["wrapper/anchor.png", "wrapper/sheet.png"],
            first_frame="wrapper/anchor.png", last_frame="wrapper/end.png")
        guides = {nid: n for nid, n in graph.items() if n["class_type"] == "MiniMaxH3AddGuide"}
        self.assertEqual(sorted(g["inputs"]["frame_idx"] for g in guides.values()), [-1, 0])
        first = next(nid for nid, g in guides.items() if g["inputs"]["frame_idx"] == 0)
        last = next(nid for nid, g in guides.items() if g["inputs"]["frame_idx"] == -1)
        self.assertEqual(guides[first]["inputs"]["positive"], ["6", 0])
        self.assertEqual(guides[last]["inputs"]["positive"], [first, 0])
        for guide in guides.values():
            self.assertEqual(guide["inputs"]["latent"], ["6", 1])
            self.assertEqual(guide["inputs"]["vae"], ["4", 0])
            scale = graph[guide["inputs"]["image"][0]]
            self.assertEqual(scale["class_type"], "ImageScale")
            self.assertEqual((scale["inputs"]["width"], scale["inputs"]["height"], scale["inputs"]["crop"]),
                             (864, 480, "center"))
        self.assertEqual(graph[graph[guides[first]["inputs"]["image"][0]]["inputs"]["image"][0]]["inputs"],
                         {"image": "wrapper/anchor.png"})
        guider = next(n for n in graph.values() if n["class_type"] == "BasicGuider")
        self.assertEqual(guider["inputs"]["conditioning"], [last, 0])
        # the sampler still starts from the reference node's latent
        sampler = next(n for n in graph.values() if n["class_type"] == "SamplerCustomAdvanced")
        self.assertEqual(sampler["inputs"]["latent_image"], ["6", 1])
        # ids never collide with the references' loaders
        self.assertEqual(len(graph), len(set(graph)))
        # no pins: the guider reads the reference node directly, as before
        plain = wrapper_workflows.build_minimax_h3_reference_to_video(prompt="x", ref_images=["wrapper/a.png"])
        self.assertFalse([n for n in plain.values() if n["class_type"] == "MiniMaxH3AddGuide"])
        self.assertEqual(next(n for n in plain.values() if n["class_type"] == "BasicGuider")["inputs"]["conditioning"],
                         ["6", 0])

    def test_reference_task_accepts_the_pinned_frames(self):
        task = wrapper_workflows.WORKFLOWS["minimaxh3"]["tasks"]["reference"]
        self.assertEqual(task["uploads"]["image"], {"ext": "image", "max": 1})
        self.assertEqual(task["uploads"]["last_frame"], {"ext": "image", "max": 1})
        self.assertEqual(task["upload_params"]["image"], "first_frame")
        self.assertIn("image", task["form"])
        self.assertFalse(task["requires_image"])

    def test_decode_nodes_use_the_correct_vae(self):
        # The joint AV latent is a NestedTensor (video [B,24,T,H,W], audio
        # [B,32,2,T]). VAEDecode must unbind the video part -> the VIDEO vae,
        # VAEDecodeAudio the audio part -> the AUDIO vae; swapping them makes
        # the video VAE's 5D memory estimator crash with "tuple index out of
        # range" on the 4D audio latent (sd.py estimate_decode_memory).
        for task_name in ("text", "image"):
            task = wrapper_workflows.WORKFLOWS["minimaxh3"]["tasks"][task_name]
            graph = task["build"](
                prompt="x",
                **({"first_frame": "wrapper/t1.png"} if task_name == "image" else {}),
                seed=1, steps=20, width=672, height=384,
                duration=5.0, scheduler="beta", filename_prefix="wrapper/minimaxh3")
            vae_loaders = {n["inputs"]["vae_name"]: nid for nid, n in graph.items()
                           if n["class_type"] == "VAELoader"}
            decode = {n["class_type"]: n for n in graph.values()}
            self.assertEqual(decode["VAEDecode"]["inputs"]["vae"][0],
                             vae_loaders[wrapper_workflows.MINIMAX_H3_VIDEO_VAE])
            self.assertEqual(decode["VAEDecodeAudio"]["inputs"]["vae"][0],
                             vae_loaders[wrapper_workflows.MINIMAX_H3_AUDIO_VAE])
            # both decoders read the same sampler output (the joint AV latent)
            sampler_out = decode["VAEDecode"]["inputs"]["samples"]
            self.assertEqual(decode["VAEDecodeAudio"]["inputs"]["samples"], sampler_out)

    def test_model_pick_prefers_smallest_present_clip(self):
        w = wrapper_workflows
        disk = set()

        def is_present(folder, filename):
            return filename in disk

        # nvfp4 CLIP on disk -> never touches bf16 even though fp8 is requested,
        # and the preferred encoder needs no note
        disk = {w.MINIMAX_H3_UNET_FP8, w.MINIMAX_H3_CLIP_NVFP4}
        unet, clip, note = w.minimax_h3_model_pick("fp8", ref2va=False, is_present=is_present)
        self.assertEqual(unet, w.MINIMAX_H3_UNET_FP8)
        self.assertEqual(clip, w.MINIMAX_H3_CLIP_NVFP4)
        self.assertIsNone(note)

        # bf16 CLIP is the only encoder present (old downloads) -> used, and the
        # note calls out that the smaller preferred encoder is missing
        disk = {w.MINIMAX_H3_UNET_FP8, w.MINIMAX_H3_CLIP}
        unet, clip, note = w.minimax_h3_model_pick("fp8", ref2va=False, is_present=is_present)
        self.assertEqual(clip, w.MINIMAX_H3_CLIP)
        self.assertIn(w.MINIMAX_H3_CLIP_NVFP4, note)

        # requested UNET missing -> first present variant (fp8) used with note
        disk = {w.MINIMAX_H3_REF2VA_UNET_FP8, w.MINIMAX_H3_CLIP_NVFP4}
        unet, clip, note = w.minimax_h3_model_pick("bf16", ref2va=True, is_present=is_present)
        self.assertEqual(unet, w.MINIMAX_H3_REF2VA_UNET_FP8)
        self.assertEqual(clip, w.MINIMAX_H3_CLIP_NVFP4)
        self.assertIn("not on disk", note)

        # nothing on disk -> requested (or fp8) UNET + nvfp4 CLIP are downloaded
        disk = set()
        unet, clip, note = w.minimax_h3_model_pick("bf16", ref2va=False, is_present=is_present)
        self.assertEqual(unet, w.MINIMAX_H3_UNET_BF16)
        self.assertEqual(clip, w.MINIMAX_H3_CLIP_NVFP4)
        # no explicit quantization -> the templates' int8_convrot + nvfp4 pair
        unet, clip, _ = w.minimax_h3_model_pick(None, ref2va=False, is_present=is_present)
        self.assertEqual(unet, w.MINIMAX_H3_UNET_INT8)
        self.assertEqual(clip, w.MINIMAX_H3_CLIP_NVFP4)
        with self.assertRaises(ValueError):
            w.minimax_h3_model_pick("fp16", False, is_present)

        # models() honors explicit overrides for the download list
        models = w.minimax_h3_models("fp8", ref2va=True, unet_name="x.safetensors", clip_name="y.safetensors")
        self.assertEqual(models[0]["filename"], "x.safetensors")
        self.assertEqual(models[1]["filename"], "y.safetensors")

    def test_models_and_length_helper(self):
        models = wrapper_workflows.minimax_h3_models("int8", ref2va=True)
        self.assertEqual(models[0]["filename"], wrapper_workflows.MINIMAX_H3_REF2VA_UNET_INT8)
        self.assertEqual([m["filename"] for m in models],
                         [wrapper_workflows.MINIMAX_H3_REF2VA_UNET_INT8,
                          wrapper_workflows.MINIMAX_H3_CLIP_INT8,
                          wrapper_workflows.MINIMAX_H3_VIDEO_VAE,
                          wrapper_workflows.MINIMAX_H3_AUDIO_VAE])
        self.assertEqual(wrapper_workflows.minimax_h3_length(5.0), 120)
        self.assertEqual(wrapper_workflows.minimax_h3_length(0.1), 5)  # min 5

    def test_nvfp4_quantization_uses_nvfp4_clip_with_fp8_unet(self):
        # No nvfp4 UNET exists; nvfp4 is a CLIP-only option that drops the text
        # encoder from 66 GB (bf16) / 34 GB (int8) to 16 GB, letting the whole
        # pipeline fit on a 32 GB GPU.
        for ref2va in (False, True):
            models = wrapper_workflows.minimax_h3_models("nvfp4", ref2va=ref2va)
            unet = wrapper_workflows.MINIMAX_H3_REF2VA_UNET_FP8 if ref2va else wrapper_workflows.MINIMAX_H3_UNET_FP8
            self.assertEqual(models[0]["filename"], unet)
            self.assertIn(wrapper_workflows.MINIMAX_H3_CLIP_NVFP4, [m["filename"] for m in models])
            self.assertNotIn(wrapper_workflows.MINIMAX_H3_CLIP, [m["filename"] for m in models])
        self.assertTrue(all(
            q in wrapper_workflows.MINIMAX_H3_QUANT_MODELS for q in ("fp8", "int8", "bf16", "nvfp4")))

    def test_fit_canvas_rounds_up_and_caps_area(self):
        w = wrapper_workflows
        # already on the grid and inside the cap -> untouched, nothing to report
        self.assertEqual(w.minimax_h3_fit_canvas(1344, 768), (1344, 768, None))
        # off-grid rounds up, never below what was asked for
        self.assertEqual(w.minimax_h3_fit_canvas(720, 720)[:2], (736, 736))
        # over the native canvas -> scaled down, aspect kept, note explains it
        for width, height in ((1920, 1080), (3840, 2160), (1080, 1920), (4000, 1000)):
            fitted_w, fitted_h, note = w.minimax_h3_fit_canvas(width, height)
            self.assertEqual((fitted_w % 32, fitted_h % 32), (0, 0))
            self.assertLessEqual(fitted_w * fitted_h, w.MINIMAX_H3_MAX_PIXELS)
            self.assertIn("native canvas", note)
            self.assertAlmostEqual(fitted_w / fitted_h, width / height, delta=0.15)

    def test_budget_matches_measured_envelope(self):
        """The guard has to agree with what the reference box actually did:
        letting an over-budget job through OOM-kills the whole server."""
        w = wrapper_workflows
        frames = lambda seconds: w.minimax_h3_align_frames(w.minimax_h3_length(seconds))
        self.assertEqual(frames(5), 124)  # 5s snaps up to the 17k+5 grid

        for width, height, seconds in ((864, 480, 5), (1152, 640, 5), (736, 736, 5)):
            self.assertIsNone(w.minimax_h3_check_budget(width, height, frames(seconds)),
                              f"{width}x{height} {seconds}s completed on the reference box")
        for width, height, seconds in ((1344, 768, 5), (960, 544, 10), (1344, 768, 8),
                                       (1344, 768, 10)):
            self.assertIsNotNone(w.minimax_h3_check_budget(width, height, frames(seconds)),
                                 f"{width}x{height} {seconds}s OOM-killed the reference box")

        # the form defaults must be inside the budget, or every default request 400s
        self.assertIsNone(w.minimax_h3_check_budget(
            w.MINIMAX_H3_DEFAULT_WIDTH, w.MINIMAX_H3_DEFAULT_HEIGHT, w.MINIMAX_H3_MAX_FRAMES))
        # and the advertised maximum duration must not itself be over the cap
        self.assertLessEqual(frames(w.MINIMAX_H3_MAX_DURATION), w.MINIMAX_H3_MAX_FRAMES)

    def test_over_length_is_rejected_at_any_resolution(self):
        """Frame count is the hard limit -- shrinking the canvas must not buy
        back a longer clip, because the video VAE does not tile temporally."""
        w = wrapper_workflows
        too_many = w.MINIMAX_H3_MAX_FRAMES + 17
        for width, height in ((1344, 768), (864, 480), (64, 64)):
            note = w.minimax_h3_check_budget(width, height, too_many)
            self.assertIsNotNone(note)
            self.assertIn("frames", note)


class TestIdeogram4Graph(unittest.TestCase):
    def test_structure(self):
        graph = wrapper_workflows.build_ideogram4_text2img(prompt="test", width=1000, height=700, seed=42)
        self.assertEqual(len(graph), 15)
        class_types = [node["class_type"] for node in graph.values()]
        for expected in ("UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode",
                         "ConditioningZeroOut", "CFGOverride", "DualModelGuider",
                         "Ideogram4Scheduler", "RandomNoise", "KSamplerSelect",
                         "EmptyFlux2LatentImage", "SamplerCustomAdvanced", "VAEDecode", "SaveImage"):
            self.assertIn(expected, class_types)
        # link integrity: every ["id", slot] reference exists
        ids = set(graph)
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], ids)

    def test_size_rounding_and_presets(self):
        graph = wrapper_workflows.build_ideogram4_text2img(
            prompt="test", width=1000, height=700, steps=12, mu=0.5, std=1.75)
        scheduler = graph["9"]["inputs"]
        self.assertEqual(scheduler["width"], 1008)  # (1000+15)//16*16
        self.assertEqual(scheduler["height"], 704)  # (700+15)//16*16
        self.assertEqual(scheduler["steps"], 12)
        self.assertEqual(scheduler["mu"], 0.5)
        latent = graph["12"]["inputs"]
        self.assertEqual((latent["width"], latent["height"]), (1008, 704))

    def test_minimum_size(self):
        graph = wrapper_workflows.build_ideogram4_text2img(prompt="test", width=10, height=10)
        self.assertEqual(graph["9"]["inputs"]["width"], 256)
        self.assertEqual(graph["9"]["inputs"]["height"], 256)

    def test_model_names(self):
        graph = wrapper_workflows.build_ideogram4_text2img(prompt="test")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], wrapper_workflows.IDEOGRAM4_UNET)
        self.assertEqual(graph["2"]["inputs"]["unet_name"], wrapper_workflows.IDEOGRAM4_UNET_UNCONDITIONAL)
        self.assertEqual(graph["3"]["inputs"]["clip_name"], wrapper_workflows.IDEOGRAM4_CLIP)
        self.assertEqual(graph["4"]["inputs"]["vae_name"], wrapper_workflows.IDEOGRAM4_VAE)


class TestFxInterpGraph(unittest.TestCase):
    def test_structure_retimes_at_source_fps_without_audio(self):
        graph = wrapper_workflows.build_fx_interp(video="wrapper/job/clip.mp4", factor=4)
        self.assertEqual([graph[k]["class_type"] for k in sorted(graph, key=int)], [
            "LoadVideo", "GetVideoComponents", "FrameInterpolationModelLoader",
            "FrameInterpolate", "CreateVideo", "SaveVideo"])
        self.assertEqual(graph["1"]["inputs"]["file"], "wrapper/job/clip.mp4")
        self.assertEqual(graph["4"]["inputs"], {"interp_model": ["3", 0], "images": ["2", 0], "multiplier": 4})
        create = graph["5"]["inputs"]
        # fps is GetVideoComponents' fps output (index 2): the source rate, so
        # factor x the frames play factor x slower.
        self.assertEqual(create["fps"], ["2", 2])
        self.assertEqual(create["images"], ["4", 0])
        self.assertNotIn("audio", create)
        self.assertEqual(graph["6"]["inputs"]["video"], ["5", 0])
        for node_id, node in graph.items():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], graph, f"{node_id} links to unknown node {value[0]}")

    def test_default_model_is_film_and_rife_is_selectable(self):
        graph = wrapper_workflows.build_fx_interp(video="v.mp4")
        self.assertEqual(graph["3"]["inputs"]["model_name"], "film_net_fp16.safetensors")
        self.assertEqual(graph["4"]["inputs"]["multiplier"], 2)
        graph = wrapper_workflows.build_fx_interp(video="v.mp4", model_name="rife_v4.26.safetensors")
        self.assertEqual(graph["3"]["inputs"]["model_name"], "rife_v4.26.safetensors")

    def test_only_factors_2_and_4(self):
        for factor in (1, 3, 8):
            with self.assertRaises(ValueError):
                wrapper_workflows.build_fx_interp(video="v.mp4", factor=factor)

    def test_expected_steps_match_the_nodes_progress_total(self):
        graph = wrapper_workflows.build_fx_interp(video="v.mp4", factor=4, frames=121)
        self.assertEqual(graph["4"]["_meta"]["expected_steps"], 120 * 3)
        self.assertNotIn("_meta", wrapper_workflows.build_fx_interp(video="v.mp4")["4"])

    def test_models_come_from_the_blueprint_repo(self):
        for model, filename in (("film", "film_net_fp16.safetensors"), ("rife", "rife_v4.26.safetensors")):
            (entry,) = wrapper_workflows.fx_interp_models(model)
            self.assertEqual(entry["folder"], "frame_interpolation")
            self.assertEqual(entry["filename"], filename)
            self.assertEqual(entry["url"], "https://huggingface.co/Comfy-Org/frame_interpolation/resolve/main/"
                                           f"frame_interpolation/{filename}")

    def test_output_frames_and_budget(self):
        self.assertEqual(wrapper_workflows.fx_interp_output_frames(121, 2), 241)
        self.assertEqual(wrapper_workflows.fx_interp_output_frames(121, 4), 481)
        self.assertEqual(wrapper_workflows.fx_interp_output_frames(1, 4), 1)
        # 4 s of 1080p30 at 4x fits the default budget; 10 s does not.
        self.assertIsNone(wrapper_workflows.fx_interp_check_budget(1920, 1080, 120, 4))
        detail = wrapper_workflows.fx_interp_check_budget(1920, 1080, 300, 4)
        self.assertIn("1197 frames", detail)
        self.assertIn("factor 2", detail)

    def test_registered_without_a_prompt(self):
        entry = wrapper_workflows.WORKFLOWS["fx-interp"]
        self.assertEqual(entry["output_type"], "video")
        self.assertFalse(entry["requires_prompt"])
        self.assertEqual(entry["required_uploads"], ["video"])
        self.assertEqual(entry["uploads"], {"video": {"ext": "video", "max": 1}})
        self.assertEqual(entry["uses"], [])
        self.assertIs(entry["build"], wrapper_workflows.build_fx_interp)

    def test_openapi_documents_the_endpoint(self):
        spec = wrapper_openapi.spec_with_workflows(wrapper_workflows.WORKFLOWS)
        op = spec["paths"]["/api/wrapper/fx-interp/generate"]["post"]
        schema = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(schema["required"], ["video"])
        self.assertNotIn("prompt", schema["properties"])
        self.assertEqual(schema["properties"]["factor"]["enum"], [2, 4])
        self.assertEqual(schema["properties"]["model"]["enum"], ["film", "rife"])
        self.assertEqual(schema["properties"]["video"]["format"], "binary")
        self.assertEqual(list(op["responses"]["200"]["content"])[0], "video/mp4")
        self.assertEqual(op["operationId"], "generateFxInterp")


def _assert_links_resolve(test, graph):
    for node_id, node in graph.items():
        for value in node["inputs"].values():
            if isinstance(value, list):
                test.assertIn(value[0], graph, f"{node_id} links to unknown node {value[0]}")


class TestFxBgRemoveGraph(unittest.TestCase):
    def test_structure_is_a_matte_at_source_fps_without_audio(self):
        graph = wrapper_workflows.build_fx_bgremove(video="wrapper/job/clip.mp4")
        self.assertEqual([graph[k]["class_type"] for k in sorted(graph, key=int)], [
            "LoadVideo", "GetVideoComponents", "LoadBackgroundRemovalModel", "RemoveBackground",
            "MaskToImage", "CreateVideo", "SaveVideo"])
        self.assertEqual(graph["1"]["inputs"]["file"], "wrapper/job/clip.mp4")
        self.assertEqual(graph["3"]["inputs"]["bg_removal_name"], "birefnet.safetensors")
        self.assertEqual(graph["4"]["inputs"], {"bg_removal_model": ["3", 0], "image": ["2", 0]})
        # The mask itself, white = subject (no InvertMask, unlike the blueprint's cut-out).
        self.assertEqual(graph["5"]["inputs"], {"mask": ["4", 0]})
        create = graph["6"]["inputs"]
        self.assertEqual(create["images"], ["5", 0])
        self.assertEqual(create["fps"], ["2", 2])
        self.assertNotIn("audio", create)
        self.assertEqual(graph["7"]["inputs"]["video"], ["6", 0])
        _assert_links_resolve(self, graph)

    def test_lucida_is_selectable(self):
        graph = wrapper_workflows.build_fx_bgremove(video="v.mp4", model_name="lucida.safetensors")
        self.assertEqual(graph["3"]["inputs"]["bg_removal_name"], "lucida.safetensors")

    def test_expected_steps_are_one_per_frame(self):
        graph = wrapper_workflows.build_fx_bgremove(video="v.mp4", frames=90)
        self.assertEqual(graph["4"]["_meta"]["expected_steps"], 90)
        self.assertNotIn("_meta", wrapper_workflows.build_fx_bgremove(video="v.mp4")["4"])

    def test_models_come_from_the_blueprint_repo(self):
        for model, filename in (("birefnet", "birefnet.safetensors"), ("lucida", "lucida.safetensors")):
            (entry,) = wrapper_workflows.fx_bgremove_models(model)
            self.assertEqual(entry["folder"], "background_removal")
            self.assertEqual(entry["filename"], filename)
            self.assertEqual(entry["url"], "https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/"
                                           f"background_removal/{filename}")

    def test_budget(self):
        # 15 s of 1080p30 fits the default budget; 20 s does not.
        self.assertIsNone(wrapper_workflows.fx_bgremove_check_budget(1920, 1080, 450))
        detail = wrapper_workflows.fx_bgremove_check_budget(1920, 1080, 600)
        self.assertIn("600 frames of 1920x1080", detail)
        self.assertIn("at most about 482 frames", detail)

    def test_registered_without_a_prompt(self):
        entry = wrapper_workflows.WORKFLOWS["fx-bgremove"]
        self.assertEqual(entry["output_type"], "video")
        self.assertFalse(entry["requires_prompt"])
        self.assertEqual(entry["required_uploads"], ["video"])
        self.assertTrue(entry["setup_uses_uploads"])
        self.assertEqual(entry["uses"], [])
        self.assertIs(entry["build"], wrapper_workflows.build_fx_bgremove)

    def test_openapi_documents_the_endpoint(self):
        spec = wrapper_openapi.spec_with_workflows(wrapper_workflows.WORKFLOWS)
        op = spec["paths"]["/api/wrapper/fx-bgremove/generate"]["post"]
        schema = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(schema["required"], ["video"])
        self.assertNotIn("prompt", schema["properties"])
        self.assertEqual(schema["properties"]["model"]["enum"], ["birefnet", "lucida"])
        self.assertEqual(list(op["responses"]["200"]["content"])[0], "video/mp4")
        self.assertEqual(op["operationId"], "generateFxBgremove")


class TestFxUpscaleGraph(unittest.TestCase):
    def test_size_reaches_the_target_short_side_within_1x_to_4x(self):
        size = wrapper_workflows.fx_upscale_size
        self.assertEqual(size(640, 360, 1080), (1920, 1080, 3.0))
        self.assertEqual(size(1280, 720, 1080), (1920, 1080, 1.5))
        self.assertEqual(size(360, 640, 1080), (1080, 1920, 3.0))  # portrait
        # Capped at 4x: short of the target.
        self.assertEqual(size(320, 180, 1080), (1280, 720, 4.0))
        # Already big enough: same size (still enhanced, never downscaled).
        self.assertEqual(size(1920, 1080, 720), (1920, 1080, 1.0))
        # Odd-ratio sources: even sides, the short side rounded up past the target.
        width, height, _ = size(853, 481, 1080)
        self.assertEqual((width % 2, height % 2), (0, 0))
        self.assertGreaterEqual(height, 1080)

    def test_esrgan_picks_x2_up_to_2x_and_x4_above(self):
        self.assertEqual(wrapper_workflows.fx_upscale_esrgan_scale(1.0), 2)
        self.assertEqual(wrapper_workflows.fx_upscale_esrgan_scale(2.0), 2)
        self.assertEqual(wrapper_workflows.fx_upscale_esrgan_scale(2.25), 4)
        (x2,) = wrapper_workflows.fx_upscale_models("esrgan", 1.5)
        self.assertEqual((x2["folder"], x2["filename"]), ("upscale_models", "RealESRGAN_x2plus.pth"))
        self.assertTrue(x2["url"].startswith("https://github.com/xinntao/Real-ESRGAN/releases/download/"))
        (x4,) = wrapper_workflows.fx_upscale_models("esrgan", 3.0)
        self.assertEqual(x4["url"], "https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged/resolve/main/"
                                    "RealESRGAN_x4plus.safetensors")
        self.assertEqual([(m["folder"], m["filename"]) for m in wrapper_workflows.fx_upscale_models("seedvr2", 3.0)],
                         [("diffusion_models", "seedvr2_3b_int8_convrot.safetensors"),
                          ("vae", "seedvr2_ema_vae_fp16.safetensors")])

    def test_esrgan_graph_upscales_then_scales_to_the_exact_size(self):
        graph = wrapper_workflows.build_fx_upscale(
            video="wrapper/job/clip.mp4", width=1920, height=1080, upscale_model_name="RealESRGAN_x4plus.safetensors",
            frames=31, source_size=(640, 360))
        self.assertEqual([graph[k]["class_type"] for k in sorted(graph, key=int)], [
            "LoadVideo", "GetVideoComponents", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale",
            "CreateVideo", "SaveVideo"])
        self.assertEqual(graph["3"]["inputs"]["model_name"], "RealESRGAN_x4plus.safetensors")
        self.assertEqual(graph["4"]["inputs"], {"upscale_model": ["3", 0], "image": ["2", 0]})
        # 640x360 is 2x1 tiles of 512 with 32 overlap: 2 steps per frame.
        self.assertEqual(graph["4"]["_meta"]["expected_steps"], 62)
        self.assertEqual(graph["5"]["inputs"], {"image": ["4", 0], "upscale_method": "area",
                                                "width": 1920, "height": 1080, "crop": "disabled"})
        create = graph["6"]["inputs"]
        self.assertEqual((create["images"], create["fps"]), (["5", 0], ["2", 2]))
        self.assertNotIn("audio", create)
        self.assertEqual(graph["7"]["inputs"]["video"], ["6", 0])
        _assert_links_resolve(self, graph)

    def test_seedvr2_graph_follows_the_blueprint_with_temporal_chunks(self):
        graph = wrapper_workflows.build_fx_upscale(
            video="wrapper/job/clip.mp4", width=1920, height=1080, model="seedvr2", seed=7)
        self.assertEqual([graph[k]["class_type"] for k in sorted(graph, key=int)], [
            "LoadVideo", "GetVideoComponents", "UNETLoader", "VAELoader", "ImageScale", "SeedVR2Preprocess",
            "VAEEncodeTiled", "SeedVR2TemporalChunk", "SeedVR2Conditioning", "KSampler", "SeedVR2TemporalMerge",
            "VAEDecodeTiled", "SeedVR2PostProcessing", "CreateVideo", "SaveVideo"])
        self.assertEqual(graph["3"]["inputs"]["unet_name"], "seedvr2_3b_int8_convrot.safetensors")
        self.assertEqual(graph["4"]["inputs"]["vae_name"], "seedvr2_ema_vae_fp16.safetensors")
        self.assertEqual((graph["5"]["inputs"]["width"], graph["5"]["inputs"]["height"]), (1920, 1080))
        self.assertEqual(graph["8"]["inputs"]["chunking_mode"], "auto")
        sampler = graph["10"]["inputs"]
        # Conditioning and the sampler both take the chunk list; the merge gets
        # the chunker's effective overlap.
        self.assertEqual(graph["9"]["inputs"]["vae_conditioning"], ["8", 0])
        self.assertEqual(sampler["latent_image"], ["8", 0])
        self.assertEqual((sampler["positive"], sampler["negative"]), (["9", 0], ["9", 1]))
        self.assertEqual((sampler["seed"], sampler["steps"], sampler["cfg"]), (7, 1, 1.0))
        self.assertEqual(graph["11"]["inputs"], {"latents": ["10", 0], "temporal_overlap": ["8", 1]})
        # Post-processing crops back to the resized frames (size and frame count).
        self.assertEqual(graph["13"]["inputs"]["original_resized_images"], ["5", 0])
        create = graph["14"]["inputs"]
        self.assertEqual((create["images"], create["fps"]), (["13", 0], ["2", 2]))
        self.assertNotIn("audio", create)
        self.assertEqual(graph["15"]["inputs"]["video"], ["14", 0])
        _assert_links_resolve(self, graph)

    def test_unknown_model_is_refused(self):
        with self.assertRaises(ValueError):
            wrapper_workflows.build_fx_upscale(video="v.mp4", width=64, height=64, model="waifu2x")

    def test_tile_steps_match_comfy(self):
        try:
            from comfy.utils import get_tiled_scale_steps
        except ImportError:
            self.skipTest("comfy (torch) not importable")

        for width, height in ((320, 180), (640, 360), (1280, 720), (1920, 1080), (513, 2000)):
            self.assertEqual(wrapper_workflows.fx_upscale_tile_steps(width, height),
                             get_tiled_scale_steps(width, height, 512, 512, 32))

    def test_budget_counts_the_model_scale_or_seedvr2_copies(self):
        check = wrapper_workflows.fx_upscale_check_budget
        # 10 s of 360p30 to 1080p: esrgan x4 makes 1440p frames (1.1e9) and fits.
        self.assertIsNone(check("esrgan", 640, 360, 300, 1920, 1080, 3.0))
        detail = check("esrgan", 640, 360, 500, 1920, 1080, 3.0)
        self.assertIn("esrgan x4 model (2560x1440", detail)
        self.assertIn("at most about 406 frames", detail)
        # seedvr2 weighs two pixel-frames per output pixel-frame.
        self.assertIsNone(check("seedvr2", 640, 360, 300, 1920, 1080, 3.0))
        self.assertIn("seedvr2 at 1920x1080", check("seedvr2", 640, 360, 400, 1920, 1080, 3.0))

    def test_image_size_is_exact_or_from_the_short_side(self):
        size = wrapper_workflows.fx_upscale_image_size
        self.assertEqual(size(2048, 1152, {"width": "7680", "height": "4320"}), (7680, 4320))
        # Odd sides are fine: a PNG has no 4:2:0 rule.
        self.assertEqual(size(1001, 667, {"width": "3003", "height": "2001"}), (3003, 2001))
        self.assertEqual(size(1024, 576, {"target_short_side": "2160"}), (3840, 2160))
        for src, fields, why in (((1024, 1024), {"width": "4000"}, "together"),
                                 ((1024, 1024), {"width": "9000", "height": "100"}, "from 16 to 8192"),
                                 ((1024, 1024), {"width": "x", "height": "1"}, "whole number"),
                                 ((512, 512), {"width": "8192", "height": "8192"}, "at most 8x"),
                                 ((1024, 576), {"target_short_side": "8000"}, "at most 8192 a side")):
            with self.assertRaises(ValueError) as ctx:
                size(*src, fields)
            self.assertIn(why, str(ctx.exception))

    def test_refine_base_and_tiles(self):
        base = wrapper_workflows.fx_upscale_refine_base
        self.assertEqual(base(7680, 4320, 2048, 1152), (3840, 2160))
        self.assertEqual(base(7680, 3072, 1600, 640), (3840, 1536))
        # Never below the source.
        self.assertEqual(base(7680, 4320, 4096, 2304), (4096, 2304))
        self.assertEqual(base(3000, 2000, 1500, 1000), (3000, 2000))
        tiles = wrapper_workflows.fx_upscale_refine_tiles
        self.assertEqual(tiles(1024, 1024), 1)
        # 7680 = 1024 + 8 strides of 896 (rounded up); 4320 = 1024 + 4 strides.
        self.assertEqual(tiles(7680, 4320), 9 * 5)
        # A side shorter than a tile: tiles of that side floored to 16 (else the merge leaves black rows).
        self.assertEqual(wrapper_workflows.fx_upscale_refine_tile_size(7680, 750), (1024, 736))
        self.assertEqual(tiles(7680, 750), 9 * 2)
        graph = wrapper_workflows.build_fx_upscale_image(image="a.png", width=7680, height=750, refine="tiled", base_size=(3840, 375))
        self.assertEqual((graph["25"]["inputs"]["tile_width"], graph["25"]["inputs"]["tile_height"]), (1024, 736))

    def test_image_graph_is_png_in_png_out(self):
        graph = wrapper_workflows.build_fx_upscale_image(image="wrapper/job/in.png", width=3840, height=2160, seed=5)
        classes = [node["class_type"] for node in graph.values()]
        self.assertEqual(classes[0], "LoadImage")
        self.assertEqual(graph["40"]["class_type"], "SaveImage")
        self.assertNotIn("CreateVideo", classes)
        self.assertNotIn("LoadVideo", classes)
        self.assertIn("SeedVR2Conditioning", classes)
        self.assertNotIn("SplitImageToTileList", classes)

    def test_tiled_refine_graph(self):
        graph = wrapper_workflows.build_fx_upscale_image(
            image="wrapper/job/in.png", width=7680, height=4320, refine="tiled", base_size=(3840, 2160),
            prompt="a fisherman", denoise=0.25, seed=2 ** 64 - 1)
        scale = next(node for node in graph.values() if node["class_type"] == "ImageScale")
        self.assertEqual((scale["inputs"]["width"], scale["inputs"]["height"]), (3840, 2160))
        self.assertEqual((graph["20"]["inputs"]["width"], graph["20"]["inputs"]["height"]), (7680, 4320))
        self.assertEqual(graph["21"]["inputs"]["unet_name"], "flux-2-klein-4b-fp8.safetensors")
        self.assertEqual(graph["24"]["inputs"]["text"], "a fisherman")
        self.assertEqual(graph["31"]["inputs"]["denoise"], 0.25)
        self.assertLess(graph["32"]["inputs"]["noise_seed"], 2 ** 63)
        self.assertEqual(graph["35"]["inputs"]["final_width"], 7680)
        self.assertEqual(graph["40"]["inputs"]["images"], ["35", 0])
        with self.assertRaises(ValueError):
            wrapper_workflows.build_fx_upscale_image(image="a.png", width=64, height=64, refine="soft")

    def test_any_builder_picks_the_mode(self):
        still = wrapper_workflows.build_fx_upscale_any(image="a.png", width=64, height=64, model="seedvr2", frames=1,
                                                       refine="none", seed=1)
        self.assertEqual(still["1"]["class_type"], "LoadImage")
        clip = wrapper_workflows.build_fx_upscale_any(video="a.mp4", width=64, height=64, model="esrgan", frames=3,
                                                      refine="none")
        self.assertEqual(clip["1"]["class_type"], "LoadVideo")

    def test_registered_without_a_prompt(self):
        entry = wrapper_workflows.WORKFLOWS["fx-upscale"]
        self.assertFalse(entry["requires_prompt"])
        # A clip in 'video' or a still in 'image' (the setup refuses neither).
        self.assertEqual(entry["required_uploads"], [])
        self.assertEqual(set(entry["uploads"]), {"video", "image"})
        self.assertTrue(entry["setup_uses_uploads"])
        self.assertEqual(entry["uses"], [])
        self.assertIs(entry["build"], wrapper_workflows.build_fx_upscale_any)

    def test_openapi_documents_the_endpoint(self):
        spec = wrapper_openapi.spec_with_workflows(wrapper_workflows.WORKFLOWS)
        op = spec["paths"]["/api/wrapper/fx-upscale/generate"]["post"]
        schema = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertNotIn("video", schema.get("required", []))
        self.assertEqual(schema["properties"]["refine"]["enum"], ["none", "tiled"])
        self.assertEqual(schema["properties"]["model"]["enum"], ["esrgan", "seedvr2"])
        self.assertEqual(schema["properties"]["target_short_side"]["default"], 1080)
        self.assertIn("seed", schema["properties"])
        self.assertEqual(op["operationId"], "generateFxUpscale")


def _segment_graph(width=640, height=480, **fields):
    from api_wrapper import segment
    request = segment.segment_request(fields, width, height)
    return wrapper_workflows.build_image_segment(
        image="wrapper/test_input.png", request=request, width=width, height=height,
        filename_prefix="wrapper/job/image-segment")


def _assert_links_resolve(case, graph):
    for node_id, node in graph.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                case.assertIn(value[0], graph, f"node {node_id}.{name} links to missing node {value[0]}")


class TestImageSegmentGraph(unittest.TestCase):
    def test_phrase_runs_sam3_on_its_own_text_encoder(self):
        graph = _segment_graph(prompt="the sky")
        self.assertEqual({k: v["class_type"] for k, v in graph.items()}, {
            "1": "CheckpointLoaderSimple", "2": "LoadImage", "3": "ImageFromBatch", "4": "CLIPTextEncode",
            "5": "SAM3_Detect", "8": "MaskToImage", "9": "SaveImage", "10": "PreviewAny"})
        self.assertEqual(graph["1"]["inputs"]["ckpt_name"], "sam3.1_multiplex_fp16.safetensors")
        self.assertEqual(graph["3"]["inputs"], {"image": ["2", 0], "batch_index": 0, "length": 1})
        self.assertEqual(graph["4"]["inputs"], {"clip": ["1", 1], "text": "the sky:8"})
        self.assertEqual(graph["5"]["inputs"], {
            "model": ["1", 0], "image": ["3", 0], "conditioning": ["4", 0], "threshold": 0.5,
            "refine_iterations": 2, "individual_masks": False})
        self.assertEqual(graph["8"]["inputs"]["mask"], ["5", 0])
        self.assertEqual(graph["9"]["inputs"]["filename_prefix"], "wrapper/job/image-segment")
        self.assertEqual(graph["10"]["inputs"], {"source": ["5", 1]})
        self.assertEqual(graph["10"]["_meta"], {"wrapper_result": {
            "workflow": "image-segment", "mode": "prompt", "width": 640, "height": 480, "individual_masks": False}})
        _assert_links_resolve(self, graph)

    def test_individual_masks_batch_the_union_first(self):
        graph = _segment_graph(prompt="person", individual_masks="true", max_objects="3", threshold="0.4")
        self.assertEqual(graph["4"]["inputs"]["text"], "person:3")
        self.assertEqual(graph["6"]["class_type"], "SAM3_Detect")
        self.assertEqual(graph["6"]["inputs"], {**graph["5"]["inputs"], "individual_masks": True})
        self.assertEqual(graph["5"]["inputs"]["threshold"], 0.4)
        self.assertEqual(graph["7"], {"class_type": "BatchMasksNode",
                                      "inputs": {"masks.mask0": ["5", 0], "masks.mask1": ["6", 0]}})
        self.assertEqual(graph["8"]["inputs"]["mask"], ["7", 0])
        # the detections come from the pass whose masks follow the union
        self.assertEqual(graph["10"]["inputs"]["source"], ["6", 1])
        self.assertTrue(graph["10"]["_meta"]["wrapper_result"]["individual_masks"])
        _assert_links_resolve(self, graph)

    def test_boxes_are_a_wrapped_literal(self):
        """A bare list input is read as a link; {"__value__": ...} is a value."""
        graph = _segment_graph(boxes="[[10, 20, 30, 40]]")
        self.assertNotIn("4", graph)
        self.assertNotIn("conditioning", graph["5"]["inputs"])
        self.assertEqual(graph["5"]["inputs"]["bboxes"],
                         {"__value__": [{"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}]})
        self.assertEqual(graph["10"]["_meta"]["wrapper_result"]["boxes"], [[10.0, 20.0, 30.0, 40.0]])

    def test_points_are_the_nodes_json_strings(self):
        graph = _segment_graph(positive_coords="[[1, 2]]")
        self.assertEqual(json.loads(graph["5"]["inputs"]["positive_coords"]), [{"x": 1.0, "y": 2.0}])
        self.assertNotIn("negative_coords", graph["5"]["inputs"])
        graph = _segment_graph(positive_coords="[[1, 2]]", negative_coords="[[3, 4]]")
        self.assertEqual(json.loads(graph["5"]["inputs"]["negative_coords"]), [{"x": 3.0, "y": 4.0}])

    def test_upright_size_follows_exif_orientation(self):
        self.assertEqual(wrapper_workflows.upright_size(4000, 3000, 6), (3000, 4000))
        self.assertEqual(wrapper_workflows.upright_size(4000, 3000, 1), (4000, 3000))
        self.assertEqual(wrapper_workflows.upright_size(4000, 3000, None), (4000, 3000))

    def test_model_is_the_blueprints_sam31_checkpoint(self):
        (model,) = wrapper_workflows.IMAGE_SEGMENT_MODELS
        self.assertEqual(model["folder"], "checkpoints")
        self.assertEqual(model["url"], "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/"
                                       "sam3.1_multiplex_fp16.safetensors")

    def test_registered_with_a_text_answer(self):
        from api_wrapper import segment
        entry = wrapper_workflows.WORKFLOWS["image-segment"]
        self.assertNotIn("output_type", entry)  # the body is the union mask PNG
        self.assertTrue(entry["requires_image"])
        self.assertFalse(entry["requires_prompt"])
        self.assertTrue(entry["setup_uses_uploads"])
        self.assertEqual(entry["uses"], [])
        self.assertIs(entry["text_result"], segment.segment_result)
        self.assertIs(entry["build"], wrapper_workflows.build_image_segment)

    def test_openapi_documents_the_endpoint_and_the_text_route(self):
        spec = wrapper_openapi.spec_with_workflows(wrapper_workflows.WORKFLOWS)
        op = spec["paths"]["/api/wrapper/image-segment/generate"]["post"]
        schema = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(schema["required"], ["image"])
        for name in ("prompt", "boxes", "positive_coords", "negative_coords", "threshold", "max_objects",
                     "individual_masks"):
            self.assertIn(name, schema["properties"])
        self.assertIn("the person on the left", schema["properties"]["prompt"]["description"])
        self.assertEqual(list(op["responses"]["200"]["content"])[0], "image/png")
        self.assertEqual(op["operationId"], "generateImageSegment")
        text = spec["paths"]["/api/wrapper/jobs/{job_id}/text"]["get"]
        self.assertEqual(text["operationId"], "getJobText")
        self.assertIn("ImageSegmentResult", spec["components"]["schemas"])


class TestImageDescribeGraph(unittest.TestCase):
    def test_structure(self):
        from api_wrapper import describe
        graph = wrapper_workflows.build_image_describe(image="wrapper/test_input.png", task="objects",
                                                       max_length=300, megapixels=0.5)
        self.assertEqual(graph["1"]["inputs"], {"clip_name": "qwen3vl_8b_int8_convrot.safetensors",
                                                "type": "stable_diffusion", "device": "default"})
        self.assertEqual(graph["3"]["class_type"], "ImageFromBatch")
        self.assertEqual(graph["4"]["inputs"], {"image": ["3", 0], "upscale_method": "area",
                                                "megapixels": 0.5, "resolution_steps": 1})
        self.assertEqual(graph["5"], {"class_type": "TextGenerate", "inputs": {
            "clip": ["1", 0], "prompt": describe.describe_prompt("objects"), "image": ["4", 0],
            "max_length": 300, "sampling_mode": "off"}})
        self.assertEqual(graph["6"]["inputs"], {"source": ["5", 0]})
        self.assertEqual(graph["6"]["_meta"], {"wrapper_result": {"workflow": "image-describe", "task": "objects"}})
        self.assertNotIn("SaveImage", {node["class_type"] for node in graph.values()})
        _assert_links_resolve(self, graph)

    def test_reuses_the_qwen_image_21_encoder(self):
        self.assertEqual(wrapper_workflows.IMAGE_DESCRIBE_CLIP, wrapper_workflows.QWEN_IMAGE_21_CLIP)
        self.assertEqual(wrapper_workflows.IMAGE_DESCRIBE_MODELS,
                         [wrapper_workflows.qwen_image_21_models("int8")[1]])

    def test_registered_as_a_json_answer(self):
        from api_wrapper import describe
        from api_wrapper.openapi import OUTPUT_MEDIA_TYPES
        entry = wrapper_workflows.WORKFLOWS["image-describe"]
        self.assertEqual(OUTPUT_MEDIA_TYPES[entry["output_type"]], "application/json")
        self.assertIs(entry["text_result"], describe.describe_result)
        spec = wrapper_openapi.spec_with_workflows(wrapper_workflows.WORKFLOWS)
        op = spec["paths"]["/api/wrapper/image-describe/generate"]["post"]
        self.assertEqual(op["responses"]["200"]["content"],
                         {"application/json": {"schema": {"$ref": "#/components/schemas/ImageDescribeResult"}}})
        schema = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(schema["required"], ["image"])
        self.assertEqual(schema["properties"]["task"]["enum"], ["caption", "objects", "both"])
        self.assertIn("ImageDescribeResult", spec["components"]["schemas"])


try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _track_graph(width=640, height=360, has_mask=False, **fields):
    from api_wrapper import track
    request = track.track_request(fields, width, height, has_mask)
    return wrapper_workflows.build_fx_track(
        video="wrapper/test_input.mp4", request=request, width=width, height=height, fps=29.97,
        mask="wrapper/test_input.png" if has_mask else None, frames=12, filename_prefix="wrapper/job/fx-track")


class TestFxTrackGraph(unittest.TestCase):
    def test_uploaded_mask_anchors_the_tracker(self):
        graph = _track_graph(has_mask=True, prompt="ignored")
        self.assertEqual({k: v["class_type"] for k, v in graph.items()}, {
            "1": "CheckpointLoaderSimple", "2": "LoadImage", "3": "ImageFromBatch", "4": "LoadVideo",
            "5": "GetVideoComponents", "8": "ImageToMask", "9": "SAM3_VideoTrack", "10": "SAM3_TrackToMask",
            "11": "MaskToImage", "12": "CreateVideo", "13": "SaveVideo", "14": "SAM3_TrackToBBoxes",
            "15": "PreviewAny"})
        self.assertEqual(graph["8"]["inputs"], {"image": ["3", 0], "channel": "red"})
        self.assertEqual(graph["9"]["inputs"], {
            "images": ["5", 0], "model": ["1", 0], "initial_mask": ["8", 0], "detection_threshold": 0.5,
            "max_objects": 1, "detect_interval": 1})
        self.assertEqual(graph["9"]["_meta"], {"expected_steps": 12})
        _assert_links_resolve(self, graph)

    def test_matte_keeps_the_source_fps_without_audio(self):
        graph = _track_graph(has_mask=True)
        self.assertEqual(graph["10"]["inputs"], {"track_data": ["9", 0], "object_indices": ""})
        self.assertEqual(graph["11"]["inputs"], {"mask": ["10", 0]})
        self.assertEqual(graph["12"]["inputs"], {"images": ["11", 0], "fps": ["5", 2], "bit_depth": 8})
        self.assertEqual(graph["13"]["inputs"]["filename_prefix"], "wrapper/job/fx-track")

    def test_boxes_are_the_json_answer(self):
        graph = _track_graph(has_mask=True)
        self.assertEqual(graph["14"]["inputs"], {"track_data": ["9", 0], "object_indices": ""})
        self.assertEqual(graph["15"]["inputs"], {"source": ["14", 0]})
        self.assertEqual(graph["15"]["_meta"], {"wrapper_result": {
            "workflow": "fx-track", "width": 640, "height": 360, "fps": 29.97}})

    def test_phrase_detects_on_frame_0(self):
        graph = _track_graph(prompt="logo", threshold="0.3", detect_interval="4")
        self.assertNotIn("2", graph)
        self.assertEqual(graph["6"]["inputs"], {"image": ["5", 0], "batch_index": 0, "length": 1})
        self.assertEqual(graph["7"]["inputs"], {"clip": ["1", 1], "text": "logo:1"})
        self.assertEqual(graph["8"], {"class_type": "SAM3_Detect", "inputs": {
            "model": ["1", 0], "image": ["6", 0], "conditioning": ["7", 0], "threshold": 0.3,
            "refine_iterations": 2, "individual_masks": False}})
        self.assertEqual((graph["9"]["inputs"]["detection_threshold"], graph["9"]["inputs"]["detect_interval"]),
                         (0.3, 4))
        _assert_links_resolve(self, graph)

    def test_points_and_boxes(self):
        graph = _track_graph(positive_coords="[[10, 20]]", negative_coords="[[30, 40]]")
        self.assertNotIn("7", graph)
        self.assertEqual(json.loads(graph["8"]["inputs"]["positive_coords"]), [{"x": 10.0, "y": 20.0}])
        self.assertEqual(json.loads(graph["8"]["inputs"]["negative_coords"]), [{"x": 30.0, "y": 40.0}])
        graph = _track_graph(boxes="[[0, 0, 50, 50]]")
        self.assertEqual(graph["8"]["inputs"]["bboxes"],
                         {"__value__": [{"x": 0.0, "y": 0.0, "width": 50.0, "height": 50.0}]})

    def test_mask_mode_needs_the_mask(self):
        from api_wrapper import track
        with self.assertRaises(ValueError):
            wrapper_workflows.build_fx_track(video="v.mp4", request=track.track_request({}, 64, 64, True),
                                             width=64, height=64, fps=30.0)

    def test_registered_with_a_text_answer(self):
        from api_wrapper import track
        entry = wrapper_workflows.WORKFLOWS["fx-track"]
        self.assertEqual(entry["output_type"], "video")
        self.assertFalse(entry["requires_prompt"])
        self.assertEqual(entry["required_uploads"], ["video"])
        self.assertEqual(entry["uploads"], {"video": {"ext": "video", "max": 1}, "mask": {"ext": "image", "max": 1}})
        self.assertTrue(entry["setup_uses_uploads"])
        self.assertEqual(entry["uses"], [])
        self.assertIs(entry["text_result"], track.track_result)
        self.assertIs(entry["build"], wrapper_workflows.build_fx_track)
        self.assertEqual(wrapper_workflows.FX_TRACK_MODELS, wrapper_workflows.IMAGE_SEGMENT_MODELS)

    def test_openapi_documents_the_endpoint_and_its_answer(self):
        spec = wrapper_openapi.spec_with_workflows(wrapper_workflows.WORKFLOWS)
        op = spec["paths"]["/api/wrapper/fx-track/generate"]["post"]
        schema = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(schema["required"], ["video"])
        for name in ("mask", "prompt", "boxes", "positive_coords", "negative_coords", "threshold", "detect_interval"):
            self.assertIn(name, schema["properties"])
        self.assertEqual(list(op["responses"]["200"]["content"])[0], "video/mp4")
        self.assertEqual(op["operationId"], "generateFxTrack")
        answer = spec["components"]["schemas"]["FxTrackResult"]
        self.assertEqual(sorted(answer["properties"]), ["fps", "frames", "height", "width"])
        self.assertEqual(sorted(answer["properties"]["frames"]["items"]["properties"]), ["area", "bbox", "i", "score"])


def _replace_graph(frames=40, keyframe=None, **kwargs):
    return wrapper_workflows.build_fx_replace(
        video="wrapper/test_input.mp4", mask="wrapper/test_input.mp4", reference="wrapper/test_input.png",
        keyframe=keyframe, width=832, height=480, frames=frames, prompt="a red can", seed=7,
        filename_prefix="wrapper/job/fx-replace", **kwargs)


class TestFxReplaceGraph(unittest.TestCase):
    def test_blueprint_models_and_sampler(self):
        graph = _replace_graph()
        self.assertEqual(graph["1"]["inputs"], {"unet_name": "wan2.1_vace_14B_fp16.safetensors",
                                                "weight_dtype": "fp8_e4m3fn_fast"})
        self.assertEqual(graph["2"]["inputs"], {"model": ["1", 0], "strength_model": 0.3,
                                                "lora_name": "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"})
        self.assertEqual(graph["3"]["inputs"], {"model": ["2", 0], "shift": 5.0})
        self.assertEqual(graph["5"]["inputs"], {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                                                "type": "wan", "device": "default"})
        self.assertEqual(graph["7"]["inputs"]["text"], "a red can")
        sampler = graph["33"]["inputs"]
        self.assertEqual((sampler["model"], sampler["seed"], sampler["steps"], sampler["cfg"],
                          sampler["sampler_name"], sampler["scheduler"]), (["3", 0], 7, 6, 1.0, "uni_pc", "simple"))
        _assert_links_resolve(self, graph)

    def test_masked_pixels_are_painted_grey_and_the_reference_padded(self):
        graph = _replace_graph()
        self.assertEqual(graph["12"]["inputs"], {"image": ["11", 0], "upscale_method": "area", "width": 832,
                                                 "height": 480, "crop": "center"})
        self.assertEqual(graph["15"]["inputs"]["image"], ["14", 0])
        self.assertEqual(graph["16"]["inputs"], {"image": ["15", 0], "channel": "red"})
        self.assertEqual(graph["17"]["inputs"]["color"], 0x808080)
        self.assertEqual(graph["18"]["inputs"], {"destination": ["12", 0], "source": ["17", 0], "x": 0, "y": 0,
                                                 "resize_source": False, "mask": ["16", 0]})
        self.assertEqual(graph["31"]["inputs"]["padding_color"], "white")
        vace = graph["32"]["inputs"]
        self.assertEqual((vace["control_video"], vace["control_masks"], vace["reference_image"]),
                         (["18", 0], ["16", 0], ["31", 0]))
        self.assertEqual((vace["width"], vace["height"], vace["length"]), (832, 480, 41))

    def test_padding_is_trimmed_and_the_source_fps_kept(self):
        graph = _replace_graph(frames=40)
        self.assertEqual(graph["34"]["inputs"], {"samples": ["33", 0], "trim_amount": ["32", 3]})
        self.assertEqual(graph["36"]["inputs"], {"image": ["35", 0], "batch_index": 0, "length": 40})
        self.assertEqual(graph["37"]["inputs"], {"images": ["36", 0], "fps": ["11", 2], "bit_depth": 8})
        self.assertEqual(graph["38"]["inputs"]["filename_prefix"], "wrapper/job/fx-replace")

    def test_keyframe_is_frame_0_unmasked(self):
        graph = _replace_graph(keyframe="wrapper/key.png")
        self.assertEqual(graph["22"]["inputs"], {"image": ["18", 0], "batch_index": 1, "length": 39})
        self.assertEqual(graph["23"]["inputs"], {"images.image0": ["21", 0], "images.image1": ["22", 0]})
        self.assertEqual(graph["24"]["inputs"]["color"], 0)
        self.assertEqual(graph["26"]["inputs"], {"images.image0": ["24", 0], "images.image1": ["25", 0]})
        vace = graph["32"]["inputs"]
        self.assertEqual((vace["control_video"], vace["control_masks"]), (["23", 0], ["27", 0]))
        self.assertNotIn("20", _replace_graph())
        _assert_links_resolve(self, graph)

    def test_long_ranges_sample_in_context_windows(self):
        self.assertNotIn("4", _replace_graph(frames=81))
        graph = _replace_graph(frames=82)
        self.assertEqual(graph["32"]["inputs"]["length"], 85)
        windows = graph["4"]["inputs"]
        self.assertEqual((windows["model"], windows["context_length"], windows["context_overlap"],
                          windows["context_schedule"]), (["3", 0], 81, 30, "standard_static"))
        self.assertEqual(graph["33"]["inputs"]["model"], ["4", 0])

    def test_one_frame_is_refused(self):
        with self.assertRaises(ValueError):
            _replace_graph(frames=1)

    def test_length_is_the_next_4n_plus_1(self):
        self.assertEqual([wrapper_workflows.fx_replace_length(n) for n in (2, 5, 6, 9, 80, 81, 82, 300)],
                         [5, 5, 9, 9, 81, 81, 85, 301])

    def test_tier_sizes_keep_the_aspect_in_multiples_of_16(self):
        size = wrapper_workflows.fx_replace_size
        self.assertEqual(size(1920, 1080, "standard"), (848, 480))
        self.assertEqual(size(1920, 1080, "hd"), (1280, 720))
        self.assertEqual(size(1080, 1920, "standard"), (480, 848))
        self.assertEqual(size(1000, 1000, "hd"), (960, 960))
        self.assertEqual(size(640, 360, "standard"), (848, 480))
        self.assertEqual(size(1080, 1920, "quick"), (368, 656))

    def test_budget(self):
        # 10 s of 30 fps at hd fits the default budget; 20 s does not.
        self.assertIsNone(wrapper_workflows.fx_replace_check_budget(1280, 720, 301))
        detail = wrapper_workflows.fx_replace_check_budget(1280, 720, 601)
        self.assertIn("601 frames at 1280x720", detail)
        self.assertIn("at most about 325 frames", detail)

    def test_models_come_from_the_blueprints_sources(self):
        urls = {m["filename"]: m["url"] for m in wrapper_workflows.FX_REPLACE_MODELS}
        base = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files"
        self.assertEqual(urls, {
            "wan2.1_vace_14B_fp16.safetensors": f"{base}/diffusion_models/wan2.1_vace_14B_fp16.safetensors",
            "umt5_xxl_fp8_e4m3fn_scaled.safetensors": f"{base}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "wan_2.1_vae.safetensors": f"{base}/vae/wan_2.1_vae.safetensors",
            "Wan21_CausVid_14B_T2V_lora_rank32.safetensors":
                "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
            "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors":
                "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/"
                "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors",
        })
        self.assertEqual([m["folder"] for m in wrapper_workflows.FX_REPLACE_MODELS],
                         ["diffusion_models", "text_encoders", "vae", "loras", "loras"])

    def test_each_tier_fetches_only_its_speed_lora(self):
        names = lambda tier, interpolate=1: [m["filename"] for m in  # noqa: E731
                                              wrapper_workflows.fx_replace_models(tier, interpolate)]
        self.assertIn("Wan21_CausVid_14B_T2V_lora_rank32.safetensors", names("standard"))
        self.assertNotIn("lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors", names("hd"))
        quick = names("quick", 2)
        self.assertIn("lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors", quick)
        self.assertNotIn("Wan21_CausVid_14B_T2V_lora_rank32.safetensors", quick)
        self.assertIn("film_net_fp16.safetensors", quick)          # the interpolation checkpoint
        self.assertNotIn("film_net_fp16.safetensors", names("quick"))
        # What `prefetch fx-replace` fetches is every file any tier needs.
        self.assertEqual({m["filename"] for m in wrapper_workflows.fx_replace_all_models()},
                         set(quick) | set(names("standard")))

    def test_tiers_and_their_setups(self):
        self.assertEqual(list(wrapper_workflows.FX_REPLACE_TIERS), ["quick", "standard", "hd"])
        areas = list(wrapper_workflows.FX_REPLACE_TIERS.values())
        self.assertEqual(areas, sorted(areas))
        setup = wrapper_workflows.FX_REPLACE_TIER_SETUP
        self.assertEqual(set(setup), set(wrapper_workflows.FX_REPLACE_TIERS))
        self.assertEqual((setup["quick"]["steps"], setup["quick"]["lora_strength"]), (4, 1.0))
        self.assertEqual((setup["standard"]["steps"], setup["hd"]["steps"]), (6, 6))
        # The default tier is the one the endpoint always had.
        self.assertEqual(wrapper_workflows.FX_REPLACE_DEFAULT_TIER, "standard")

    def test_canvas_form_value(self):
        canvas = wrapper_workflows.fx_replace_canvas
        self.assertEqual(canvas("336x512"), (336, 512))
        self.assertEqual(canvas(" 64X2048 "), (64, 2048))
        for bad in ("", "336", "336x", "axb", "336x500", "32x512", "4096x512", "336x512x3", "0x0"):
            with self.assertRaises(ValueError, msg=bad):
                canvas(bad)

    def test_lora_strength_and_interpolation(self):
        quick = _replace_graph(frames=40, lora_name="lightx2v.safetensors", lora_strength=1.0, steps=4,
                               interpolate=2, fps=16.0)
        self.assertEqual(quick["2"]["inputs"]["strength_model"], 1.0)
        self.assertEqual(quick["33"]["inputs"]["steps"], 4)
        self.assertEqual(quick["39"]["inputs"], {"model_name": "film_net_fp16.safetensors"})
        self.assertEqual(quick["40"]["inputs"], {"interp_model": ["39", 0], "images": ["36", 0], "multiplier": 2})
        self.assertEqual(quick["40"]["_meta"], {"expected_steps": 39})
        # The answer plays at twice the sent rate: the same duration, ``(n-1)*2+1`` frames.
        self.assertEqual(quick["37"]["inputs"], {"images": ["40", 0], "fps": 32.0, "bit_depth": 8})
        _assert_links_resolve(self, quick)
        plain = _replace_graph(frames=40)
        self.assertNotIn("40", plain)
        self.assertEqual(plain["37"]["inputs"]["images"], ["36", 0])
        self.assertEqual(plain["2"]["inputs"]["strength_model"], 0.3)
        with self.assertRaises(ValueError):
            _replace_graph(interpolate=3, fps=16.0)
        with self.assertRaises(ValueError):
            _replace_graph(interpolate=2)       # the answer's rate needs the sent clip's

    def test_registered_with_its_uploads(self):
        entry = wrapper_workflows.WORKFLOWS["fx-replace"]
        self.assertEqual(entry["output_type"], "video")
        self.assertFalse(entry["requires_prompt"])
        self.assertEqual(entry["required_uploads"], ["video", "mask", "reference"])
        self.assertEqual(sorted(entry["uploads"]), ["keyframe", "mask", "reference", "video"])
        self.assertEqual(entry["uses"], ["prompt", "seed"])
        self.assertIs(entry["build"], wrapper_workflows.build_fx_replace)

    def test_openapi_documents_the_endpoint(self):
        spec = wrapper_openapi.spec_with_workflows(wrapper_workflows.WORKFLOWS)
        op = spec["paths"]["/api/wrapper/fx-replace/generate"]["post"]
        schema = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(sorted(schema["required"]), ["mask", "reference", "video"])
        self.assertEqual(schema["properties"]["tier"]["enum"], ["quick", "standard", "hd"])
        self.assertEqual(schema["properties"]["interpolate"]["enum"], [1, 2])
        self.assertIn("canvas", schema["properties"])
        self.assertEqual(list(op["responses"]["200"]["content"])[0], "video/mp4")
        self.assertEqual(op["operationId"], "generateFxReplace")


def _prepare_env_for_comfy_import():
    """Allow comfy to import on environments whose torch predates the
    comfy-kitchen requirement (torch < 2.4 lacks torch.library.custom_op).

    A stub module is enough: comfy.quant_ops guards the tensor sub-import with
    `except ImportError` and degrades to eager ops, so only the initial
    `import comfy_kitchen` must succeed.
    """
    try:
        import comfy_kitchen  # noqa: F401
    except Exception:
        import sys
        import types

        stub = types.ModuleType("comfy_kitchen")
        stub.registry = types.SimpleNamespace(
            disable=lambda *args, **kwargs: None,
            enable=lambda *args, **kwargs: None,
        )
        stub.list_backends = lambda: {}
        sys.modules["comfy_kitchen"] = stub

    # torch.serialization.add_safe_globals only exists on torch >= 2.4; the
    # test never loads checkpoints, so a no-op preserves the import path.
    if not hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals = lambda *args, **kwargs: None


@unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
class TestGraphValidation(unittest.TestCase):
    TEST_IMAGE = "wrapper/test_input.png"
    # 1x1 transparent PNG
    TEST_IMAGE_BYTES = (
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    def setUp(self):
        import folder_paths

        image_path = os.path.join(folder_paths.get_input_directory(), "wrapper", "test_input.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(self.TEST_IMAGE_BYTES)

    def tearDown(self):
        import folder_paths

        image_path = os.path.join(folder_paths.get_input_directory(), "wrapper", "test_input.png")
        if os.path.isfile(image_path):
            os.remove(image_path)

    def test_graph_passes_core_validation(self):
        _prepare_env_for_comfy_import()
        import nodes  # noqa: F401
        import api_wrapper.routes  # noqa: F401  (imports cleanly in a real env)

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        import execution

        graph = _sample_graph()
        graph["4"]["inputs"]["image"] = self.TEST_IMAGE
        valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt(
            "3f4e2c1a-9b8d-4a5e-8c6f-1d2e3f4a5b6c", graph, None))
        self.assertTrue(valid, f"graph rejected: {error}\nnode_errors: {node_errors}")
        self.assertTrue(outputs)

    def test_ideogram4_graph_passes_core_validation(self):
        _prepare_env_for_comfy_import()
        import nodes  # noqa: F401
        import api_wrapper.routes  # noqa: F401

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        import execution

        graph = wrapper_workflows.build_ideogram4_text2img(prompt="a test prompt")
        valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt(
            "7f9e3c2b-1a4d-4b5e-9c6f-2e3f4a5b6c7d", graph, None))
        self.assertTrue(valid, f"ideogram4 graph rejected: {error}\nnode_errors: {node_errors}")
        self.assertTrue(outputs)

    def test_klein_txt2img_graph_passes_core_validation(self):
        _prepare_env_for_comfy_import()
        import nodes  # noqa: F401
        import api_wrapper.routes  # noqa: F401

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        import execution

        graph = wrapper_workflows.build_flux2_klein_9b_text2img(prompt="a test prompt")
        valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt(
            "8a0f4d3c-2b5e-4c6f-8d70-3f4a5b6c7d8e", graph, None))
        self.assertTrue(valid, f"klein txt2img graph rejected: {error}\nnode_errors: {node_errors}")
        self.assertTrue(outputs)


    def test_minimax_h3_graphs_pass_core_validation(self):
        _prepare_env_for_comfy_import()
        import nodes  # noqa: F401
        import api_wrapper.routes  # noqa: F401
        import folder_paths

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        # The loader combos only accept files that exist; dummy placeholders
        # satisfy the graph validation without downloading the real models.
        placeholder_paths = []
        for folder, name in (
            ("diffusion_models", wrapper_workflows.MINIMAX_H3_UNET_INT8),
            ("diffusion_models", wrapper_workflows.MINIMAX_H3_REF2VA_UNET_INT8),
            ("text_encoders", wrapper_workflows.MINIMAX_H3_CLIP_NVFP4),
            ("vae", wrapper_workflows.MINIMAX_H3_VIDEO_VAE),
            ("vae", wrapper_workflows.MINIMAX_H3_AUDIO_VAE),
        ):
            path = os.path.join(folder_paths.get_folder_paths(folder)[0], name)
            with open(path, "wb") as f:
                f.write(b"placeholder")
            placeholder_paths.append(path)
        for name in ("test_input.mp4", "test_input.wav"):
            path = os.path.join(folder_paths.get_input_directory(), "wrapper", name)
            with open(path, "wb") as f:
                f.write(b"placeholder")
            placeholder_paths.append(path)
        try:
            import execution

            for label, graph in (
                ("t2v", wrapper_workflows.build_minimax_h3_text_to_video(prompt="a test prompt")),
                ("i2v", wrapper_workflows.build_minimax_h3_image_to_video(
                    prompt="a test prompt", first_frame="wrapper/test_input.png")),
                ("ref2va", wrapper_workflows.build_minimax_h3_reference_to_video(
                    prompt="a test prompt",
                    ref_images=["wrapper/test_input.png"],
                    ref_videos=["wrapper/test_input.mp4"],
                    ref_audios=["wrapper/test_input.wav"])),
                ("ref2va-keyframes", wrapper_workflows.build_minimax_h3_reference_to_video(
                    prompt="a test prompt", ref_images=["wrapper/test_input.png"],
                    first_frame="wrapper/test_input.png", last_frame="wrapper/test_input.png")),
            ):
                valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt(
                    f"9b1a5e4d-{abs(hash(label)) % 100000:05d}-4d7e-8f90-4a5b6c7d8e9f", graph, None))
                self.assertTrue(valid, f"minimax {label} graph rejected: {error}\nnode_errors: {node_errors}")
                self.assertTrue(outputs, f"minimax {label} produced no output nodes")
        finally:
            for path in placeholder_paths:
                os.remove(path)

    def test_reference_inputs_reach_the_node(self):
        """Validation passes whether or not the references are wired, so this
        runs the resolution execution does before calling the node: what the
        node receives as ref_images / ref_audios. A nested dict used to come
        through as {} — every Ref2VA render was text-only."""
        _prepare_env_for_comfy_import()
        import nodes
        from comfy_api.latest import _io

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        import execution

        graph = wrapper_workflows.build_minimax_h3_reference_to_video(
            prompt="<Picture 1> <Picture 2> <Audio 1>",
            ref_images=["wrapper/a.png", "wrapper/b.png"], ref_audios=["wrapper/c.wav"])
        node_class = nodes.NODE_CLASS_MAPPINGS[graph["6"]["class_type"]]
        received, _missing, v3_data = execution.get_input_data(graph["6"]["inputs"], node_class, "6")
        received = _io.build_nested_inputs(received, v3_data)
        self.assertEqual(sorted(received["ref_images"]), ["ref_image_0", "ref_image_1"])
        self.assertEqual(sorted(received["ref_audios"]), ["ref_audio_0"])
        self.assertEqual(received.get("ref_videos"), {})

    def test_qwen_image_21_graphs_pass_core_validation(self):
        _prepare_env_for_comfy_import()
        import nodes  # noqa: F401
        import api_wrapper.routes  # noqa: F401
        import folder_paths

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        placeholder_paths = []
        for model in wrapper_workflows.qwen_image_21_models("int8"):
            path = os.path.join(folder_paths.get_folder_paths(model["folder"])[0], model["filename"])
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(b"placeholder")
                placeholder_paths.append(path)
        try:
            import execution

            for label, graph in (
                ("txt2img", wrapper_workflows.build_qwen_image_21_text2img(prompt="a test prompt")),
                ("edit", wrapper_workflows.build_qwen_image_21_edit(
                    prompt="a test prompt", image=self.TEST_IMAGE, ref_images=[self.TEST_IMAGE])),
                ("edit-canvas", wrapper_workflows.build_qwen_image_21_edit(
                    prompt="a test prompt", image=self.TEST_IMAGE, width=1024, height=576)),
            ):
                valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt(
                    f"5c2d8e1f-{abs(hash(label)) % 100000:05d}-4a3b-9c8d-7e6f5a4b3c2d", graph, None))
                self.assertTrue(valid, f"qwen {label} graph rejected: {error}\nnode_errors: {node_errors}")
                self.assertTrue(outputs)
        finally:
            for path in placeholder_paths:
                os.remove(path)

    def test_fx_interp_graph_passes_core_validation(self):
        _prepare_env_for_comfy_import()
        import nodes  # noqa: F401
        import api_wrapper.routes  # noqa: F401
        import folder_paths

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        placeholder_paths = []
        for model in ("film", "rife"):
            (entry,) = wrapper_workflows.fx_interp_models(model)
            path = os.path.join(folder_paths.get_folder_paths(entry["folder"])[0], entry["filename"])
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(b"placeholder")
                placeholder_paths.append(path)
        video = os.path.join(folder_paths.get_input_directory(), "wrapper", "test_input.mp4")
        with open(video, "wb") as f:
            f.write(b"placeholder")
        placeholder_paths.append(video)
        try:
            import execution

            for factor, model in ((2, "film"), (4, "rife")):
                graph = wrapper_workflows.build_fx_interp(
                    video="wrapper/test_input.mp4", factor=factor,
                    model_name=wrapper_workflows.FX_INTERP_MODELS[model], frames=10)
                valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt(
                    f"6d3e9f2a-{factor}{len(model):04d}-4b5c-8d9e-0f1a2b3c4d5e", graph, None))
                self.assertTrue(valid, f"fx-interp {model} graph rejected: {error}\nnode_errors: {node_errors}")
                self.assertEqual(outputs, ["6"])
        finally:
            for path in placeholder_paths:
                os.remove(path)

    def _validate_with_placeholders(self, models, graphs):
        """Core validate_prompt on each (label, graph, expected outputs), with
        placeholder files for ``models`` and the input video."""
        _prepare_env_for_comfy_import()
        import nodes  # noqa: F401
        import api_wrapper.routes  # noqa: F401
        import folder_paths

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        placeholder_paths = []
        for entry in models:
            path = os.path.join(folder_paths.get_folder_paths(entry["folder"])[0], entry["filename"])
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(b"placeholder")
                placeholder_paths.append(path)
        video = os.path.join(folder_paths.get_input_directory(), "wrapper", "test_input.mp4")
        with open(video, "wb") as f:
            f.write(b"placeholder")
        placeholder_paths.append(video)
        try:
            import execution

            for index, (label, graph, expected) in enumerate(graphs):
                valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt(
                    f"5c2d8e1f-{index:04d}-4a6b-9c7d-8e9f0a1b2c3d", graph, None))
                self.assertTrue(valid, f"{label} graph rejected: {error}\nnode_errors: {node_errors}")
                # validate_prompt collects the output nodes in a set.
                self.assertEqual(sorted(outputs), sorted(expected))
        finally:
            for path in placeholder_paths:
                os.remove(path)

    def test_image_segment_graphs_pass_core_validation(self):
        graphs = [
            ("phrase", _segment_graph(prompt="the sky")),
            ("phrase, individual masks", _segment_graph(prompt="person, dog", individual_masks="true")),
            ("boxes, individual masks", _segment_graph(boxes="[[0, 0, 10, 10], [5, 5, 20, 20]]",
                                                       individual_masks="true")),
            ("points", _segment_graph(positive_coords="[[1, 1]]", negative_coords="[[5, 5]]")),
        ]
        self._validate_with_placeholders(wrapper_workflows.IMAGE_SEGMENT_MODELS,
                                         [(f"image-segment {label}", graph, ["9", "10"]) for label, graph in graphs])
        # Validation unwraps the literal boxes in place, which is what execution reads.
        self.assertEqual(graphs[2][1]["5"]["inputs"]["bboxes"],
                         [{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0},
                          {"x": 5.0, "y": 5.0, "width": 20.0, "height": 20.0}])

    def test_image_describe_graphs_pass_core_validation(self):
        self._validate_with_placeholders(wrapper_workflows.IMAGE_DESCRIBE_MODELS, [
            (f"image-describe {task}", wrapper_workflows.build_image_describe(
                image="wrapper/test_input.png", task=task), ["6"])
            for task in ("caption", "objects", "both")])

    def test_image_segment_masks_reach_the_batch_node(self):
        _prepare_env_for_comfy_import()
        import nodes
        from comfy_api.latest import _io

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        import execution

        graph = _segment_graph(prompt="person", individual_masks="true")
        node_class = nodes.NODE_CLASS_MAPPINGS[graph["7"]["class_type"]]
        received, _missing, v3_data = execution.get_input_data(graph["7"]["inputs"], node_class, "7")
        received = _io.build_nested_inputs(received, v3_data)
        self.assertEqual(list(received["masks"]), ["mask0", "mask1"])

    def test_fx_bgremove_graph_passes_core_validation(self):
        models = [m for name in wrapper_workflows.FX_BGREMOVE_MODELS for m in wrapper_workflows.fx_bgremove_models(name)]
        self._validate_with_placeholders(models, [
            (f"fx-bgremove {name}", wrapper_workflows.build_fx_bgremove(
                video="wrapper/test_input.mp4", model_name=filename, frames=10), ["7"])
            for name, filename in wrapper_workflows.FX_BGREMOVE_MODELS.items()])

    def test_fx_upscale_graphs_pass_core_validation(self):
        models = (wrapper_workflows.fx_upscale_models("esrgan", 2) + wrapper_workflows.fx_upscale_models("esrgan", 4)
                  + wrapper_workflows.fx_upscale_models("seedvr2", 4))
        self._validate_with_placeholders(models, [
            ("fx-upscale esrgan x2", wrapper_workflows.build_fx_upscale(
                video="wrapper/test_input.mp4", width=1920, height=1080,
                upscale_model_name="RealESRGAN_x2plus.pth", frames=10, source_size=(1280, 720)), ["7"]),
            ("fx-upscale esrgan x4", wrapper_workflows.build_fx_upscale(
                video="wrapper/test_input.mp4", width=1920, height=1080), ["7"]),
            ("fx-upscale seedvr2", wrapper_workflows.build_fx_upscale(
                video="wrapper/test_input.mp4", width=1920, height=1080, model="seedvr2", seed=2 ** 63), ["15"]),
        ])

    def test_fx_upscale_image_graphs_pass_core_validation(self):
        models = (wrapper_workflows.fx_upscale_models("esrgan", 4) + wrapper_workflows.fx_upscale_models("seedvr2", 4)
                  + wrapper_workflows.FX_UPSCALE_REFINE_MODELS)
        self._validate_with_placeholders(models, [
            ("fx-upscale image seedvr2", wrapper_workflows.build_fx_upscale_image(
                image="wrapper/test_input.png", width=3840, height=2160, seed=2 ** 63), ["40"]),
            ("fx-upscale image esrgan", wrapper_workflows.build_fx_upscale_image(
                image="wrapper/test_input.png", width=4000, height=2250, model="esrgan", source_size=(1000, 563)), ["40"]),
            ("fx-upscale image tiled", wrapper_workflows.build_fx_upscale_image(
                image="wrapper/test_input.png", width=7680, height=4320, refine="tiled", base_size=(3840, 2160),
                seed=2 ** 63, prompt="a pier", denoise=0.25), ["40"]),
        ])

    def test_fx_track_graphs_pass_core_validation(self):
        graphs = [
            ("mask", _track_graph(has_mask=True)),
            ("phrase", _track_graph(prompt="logo")),
            ("boxes", _track_graph(boxes="[[0, 0, 10, 10]]")),
            ("points", _track_graph(positive_coords="[[1, 1]]", negative_coords="[[5, 5]]")),
        ]
        self._validate_with_placeholders(wrapper_workflows.FX_TRACK_MODELS,
                                         [(f"fx-track {label}", graph, ["13", "15"]) for label, graph in graphs])

    def test_fx_replace_graphs_pass_core_validation(self):
        self._validate_with_placeholders(wrapper_workflows.fx_replace_all_models(), [
            ("fx-replace", _replace_graph(), ["38"]),
            ("fx-replace keyframe", _replace_graph(keyframe="wrapper/test_input.png"), ["38"]),
            ("fx-replace windows", _replace_graph(frames=120), ["38"]),
            ("fx-replace quick", _replace_graph(
                lora_name=wrapper_workflows.FX_REPLACE_QUICK_LORA, lora_strength=1.0, steps=4,
                interpolate=2, fps=16.0, keyframe="wrapper/test_input.png"), ["38"]),
        ])

    def test_fx_replace_keyframe_batches_reach_the_node(self):
        _prepare_env_for_comfy_import()
        import nodes
        from comfy_api.latest import _io

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        import execution

        graph = _replace_graph(keyframe="wrapper/key.png")
        node_class = nodes.NODE_CLASS_MAPPINGS[graph["23"]["class_type"]]
        received, _missing, v3_data = execution.get_input_data(graph["23"]["inputs"], node_class, "23")
        received = _io.build_nested_inputs(received, v3_data)
        self.assertEqual(list(received["images"]), ["image0", "image1"])

    def test_qwen_image_21_edit_images_reach_the_node(self):
        _prepare_env_for_comfy_import()
        import nodes
        from comfy_api.latest import _io

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        import execution

        graph = wrapper_workflows.build_qwen_image_21_edit(
            prompt="<image1> <image2>", image="wrapper/a.png", ref_images=["wrapper/b.png"])
        node_class = nodes.NODE_CLASS_MAPPINGS[graph["4"]["class_type"]]
        received, _missing, v3_data = execution.get_input_data(graph["4"]["inputs"], node_class, "4")
        received = _io.build_nested_inputs(received, v3_data)
        self.assertEqual(sorted(received["images"]), ["image_1", "image_2"])


if __name__ == "__main__":
    unittest.main()
