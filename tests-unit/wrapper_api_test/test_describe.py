"""image-describe's prompt and answer parsing (api_wrapper/describe.py): pure, no comfy."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api_wrapper import describe as desc  # noqa: E402

BOTH = {"task": "both"}


class TestDescribeRequest(unittest.TestCase):
    def test_defaults_and_values(self):
        self.assertEqual(desc.describe_request({}), ("both", 1024))
        self.assertEqual(desc.describe_request({"task": " Caption ", "max_length": "200"}), ("caption", 200))

    def test_refusals(self):
        for fields in ({"task": "tags"}, {"max_length": "0"}, {"max_length": "4097"}, {"max_length": "lots"}):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    desc.describe_request(fields)

    def test_scaled_down_never_up(self):
        self.assertEqual(desc.describe_megapixels(4000, 3000), 1.0)
        self.assertEqual(desc.describe_megapixels(512, 512), 0.25)
        self.assertEqual(desc.describe_megapixels(10, 10), 0.01)  # the node's minimum


class TestPrompt(unittest.TestCase):
    def test_each_task_asks_for_json_with_its_fields(self):
        caption, objects, both = (desc.describe_prompt(task) for task in ("caption", "objects", "both"))
        for prompt in (caption, objects, both):
            self.assertIn("one JSON object and nothing else", prompt)
        self.assertIn('"caption"', caption)
        self.assertNotIn('"objects"', caption)
        self.assertNotIn('"caption"', objects)
        self.assertIn("0 to 1000", objects)
        self.assertIn('"caption"', both)
        self.assertIn('"bbox_2d"', both)


class TestParsing(unittest.TestCase):
    ANSWER = {"caption": "A lighthouse on a cliff at sunset.",
              "objects": [{"label": "sky", "bbox_2d": [0, 0, 1000, 450]},
                          {"label": "lighthouse", "bbox_2d": [600, 200, 700, 800]}]}
    EXPECTED = {"caption": "A lighthouse on a cliff at sunset.",
                "objects": [{"label": "sky", "bbox": [0.0, 0.0, 1.0, 0.45]},
                            {"label": "lighthouse", "bbox": [0.6, 0.2, 0.1, 0.6]}]}

    def test_qwen3vl_scale_becomes_fractions_xywh(self):
        self.assertEqual(desc.describe_result(json.dumps(self.ANSWER), BOTH), self.EXPECTED)

    def test_prose_fences_and_a_think_block_around_the_json(self):
        text = ("<think>\nthe user wants json\n</think>\nSure! Here it is:\n```json\n"
                + json.dumps(self.ANSWER, indent=2) + "\n```\nLet me know if you need more.")
        self.assertEqual(desc.describe_result(text, BOTH), self.EXPECTED)

    def test_fraction_scale_is_kept(self):
        answer = {"caption": "c", "objects": [{"label": "dog", "bbox": [0.1, 0.2, 0.5, 0.9]}]}
        self.assertEqual(desc.describe_result(json.dumps(answer), BOTH)["objects"],
                         [{"label": "dog", "bbox": [0.1, 0.2, 0.4, 0.7]}])

    def test_boxes_are_clipped_reordered_and_degenerate_ones_dropped(self):
        answer = {"objects": [{"label": "a", "bbox_2d": [900, 900, 1200, 1100]},  # past the edge
                              {"label": "b", "bbox_2d": [500, 600, 100, 200]},    # corners swapped
                              {"label": "c", "bbox_2d": [10, 10, 10, 50]},        # no width
                              {"label": "", "bbox_2d": [1, 1, 2, 2]},             # no label
                              {"label": "d"},                                     # no box
                              {"label": "e", "bbox_2d": [1, 2, "3", 4]},          # not numbers
                              "sky"]}
        self.assertEqual(desc.describe_result(json.dumps(answer), BOTH)["objects"], [
            {"label": "a", "bbox": [0.9, 0.9, 0.1, 0.1]},
            {"label": "b", "bbox": [0.1, 0.2, 0.4, 0.4]}])

    def test_duplicates_and_long_labels(self):
        item = {"label": "  red   car " + "x" * 200, "bbox_2d": [0, 0, 500, 500]}
        objects = desc.describe_result(json.dumps({"objects": [item, item]}), BOTH)["objects"]
        self.assertEqual(len(objects), 1)
        self.assertEqual(len(objects[0]["label"]), desc.MAX_LABEL_LENGTH)
        self.assertTrue(objects[0]["label"].startswith("red car x"))

    def test_a_bare_list_of_objects(self):
        text = 'Objects: [{"label": "tree", "bbox_2d": [0, 500, 250, 1000]}]'
        self.assertEqual(desc.describe_result(text, BOTH),
                         {"caption": "", "objects": [{"label": "tree", "bbox": [0.0, 0.5, 0.25, 0.5]}]})

    def test_answer_cut_off_by_max_length_keeps_what_closed(self):
        text = json.dumps(self.ANSWER)[:-30]
        self.assertEqual(desc.describe_result(text, BOTH), {
            "caption": "A lighthouse on a cliff at sunset.",
            "objects": [{"label": "sky", "bbox": [0.0, 0.0, 1.0, 0.45]}]})

    def test_escaped_caption_survives_a_cut(self):
        text = '{"caption": "A \\"quoted\\" sign", "objects": [{"label": "sign", "bbox_2d": [1, 1'
        self.assertEqual(desc.describe_result(text, BOTH), {"caption": 'A "quoted" sign', "objects": []})

    def test_prose_only_answer_is_the_caption(self):
        self.assertEqual(desc.describe_result("  A cat  asleep on a sofa. ", {"task": "caption"}),
                         {"caption": "A cat asleep on a sofa.", "objects": []})
        self.assertEqual(desc.describe_result("", BOTH), {"caption": "", "objects": []})

    def test_task_shapes_the_answer(self):
        text = json.dumps(self.ANSWER)
        self.assertEqual(desc.describe_result(text, {"task": "caption"}),
                         {"caption": self.EXPECTED["caption"], "objects": []})
        self.assertEqual(desc.describe_result(text, {"task": "objects"}),
                         {"caption": None, "objects": self.EXPECTED["objects"]})

    def test_object_cap(self):
        many = {"objects": [{"label": f"thing {i}", "bbox_2d": [i, i, i + 10, i + 10]} for i in range(40)]}
        self.assertEqual(len(desc.describe_result(json.dumps(many), BOTH)["objects"]), desc.DESCRIBE_MAX_OBJECTS)


if __name__ == "__main__":
    unittest.main()
