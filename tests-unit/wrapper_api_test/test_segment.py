"""image-segment's request checks and answer (api_wrapper/segment.py): pure, no comfy."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api_wrapper import segment as seg  # noqa: E402


class TestPhrases(unittest.TestCase):
    def test_each_phrase_gets_the_object_cap(self):
        """SAM 3 keeps one detection per phrase unless it ends in ':N'."""
        self.assertEqual(seg.sam3_text_prompt(seg.sam3_phrases("the sky"), 8), "the sky:8")
        self.assertEqual(seg.sam3_text_prompt(seg.sam3_phrases(" sky ,  red  car,"), 3), "sky:3, red car:3")

    def test_tokenizer_syntax_is_stripped(self):
        """':N' and parentheses mean something to SAM 3's tokenizer; max_objects owns the cap."""
        self.assertEqual(seg.sam3_phrases("person: 3, (dog)"), ["person 3", "dog"])
        self.assertEqual(seg.sam3_phrases(" , ,"), [])


class TestSegmentRequest(unittest.TestCase):
    W, H = 1024, 768

    def request(self, **fields):
        return seg.segment_request(fields, self.W, self.H)

    def refused(self, **fields):
        with self.assertRaises(ValueError) as ctx:
            self.request(**fields)
        return str(ctx.exception)

    def test_phrase_defaults(self):
        self.assertEqual(self.request(prompt="the sky"), {
            "mode": "prompt", "threshold": 0.5, "individual_masks": False,
            "text": "the sky:8", "max_detections": 8})

    def test_raw_prompt_wins_like_in_every_workflow(self):
        """A serverless job's top-level prompt reaches the wrapper as raw_prompt."""
        self.assertEqual(self.request(raw_prompt="dog", prompt="cat")["text"], "dog:8")
        self.assertIn("exactly one", self.refused(raw_prompt="dog", boxes="[[0, 0, 5, 5]]"))

    def test_phrase_options(self):
        request = self.request(prompt="sky, sea", threshold="0.3", max_objects="2", individual_masks="true")
        self.assertEqual((request["text"], request["threshold"], request["individual_masks"], request["max_detections"]),
                         ("sky:2, sea:2", 0.3, True, 4))
        self.assertEqual(seg.segment_mask_count(request), 5)

    def test_exactly_one_selector(self):
        self.assertIn("exactly one", self.refused())
        self.assertIn("exactly one", self.refused(prompt="sky", boxes="[[0, 0, 10, 10]]"))
        self.assertIn("exactly one", self.refused(prompt="sky", positive_coords="[[1, 1]]"))
        self.assertIn("positive point", self.refused(negative_coords="[[1, 1]]"))
        self.assertIn("names nothing", self.refused(prompt=" ( ), "))
        self.assertIn("at most 8", self.refused(prompt=",".join("abcdefghi")))

    def test_bad_numbers_and_flags(self):
        for fields in ({"threshold": "1.5"}, {"threshold": "nan"}, {"threshold": "high"},
                       {"max_objects": "0"}, {"max_objects": "33"}, {"max_objects": "2.5"},
                       {"individual_masks": "maybe"}):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    self.request(prompt="sky", **fields)

    def test_boxes_are_pixels_clipped_to_the_image(self):
        request = self.request(boxes="[[10, 20, 100, 50], [1000, 700, 100, 100]]", individual_masks="1")
        self.assertEqual(request["boxes"], [
            {"x": 10.0, "y": 20.0, "width": 100.0, "height": 50.0},
            {"x": 1000.0, "y": 700.0, "width": 24.0, "height": 68.0}])
        self.assertEqual(seg.segment_mask_count(request), 3)
        # one box on its own needs no outer list
        self.assertEqual(len(self.request(boxes="[0, 0, 5, 5]")["boxes"]), 1)

    def test_bad_boxes(self):
        for boxes in ("not json", "[]", "{}", "[[1, 2, 3]]", "[[0, 0, 0, 10]]", "[[2000, 0, 10, 10]]",
                      '[["a", 0, 1, 1]]', "[[0, 0, true, 1]]", json.dumps([[0, 0, 1, 1]] * 33)):
            with self.subTest(boxes=boxes):
                with self.assertRaises(ValueError):
                    self.request(boxes=boxes)

    def test_points_in_either_format(self):
        request = self.request(positive_coords="[[10, 20], {\"x\": 1024, \"y\": 768}]",
                               negative_coords="[5, 5]", individual_masks="true")
        self.assertEqual(request["positive"], [{"x": 10.0, "y": 20.0}, {"x": 1023.0, "y": 767.0}])
        self.assertEqual(request["negative"], [{"x": 5.0, "y": 5.0}])
        # points always make one mask
        self.assertFalse(request["individual_masks"])
        self.assertEqual(seg.segment_mask_count(request), 1)

    def test_points_outside_the_image_are_refused(self):
        self.assertIn("outside the 1024x768 image", self.refused(positive_coords="[[1025, 10]]"))
        self.assertIn("outside", self.refused(positive_coords="[[1, 1]]", negative_coords="[[-1, 0]]"))

    def test_summary(self):
        self.assertEqual(seg.segment_summary(self.request(prompt="sky")), "'sky:8'; saves the union mask")
        self.assertIn("2 box(es); saves the union mask, then one per object",
                      seg.segment_summary(self.request(boxes="[[0,0,5,5],[5,5,5,5]]", individual_masks="on")))


class TestBudget(unittest.TestCase):
    def test_budget(self):
        self.assertIsNone(seg.segment_check_budget(4000, 3000, 33))  # 12 MP, 32 objects
        error = seg.segment_check_budget(10000, 10000, 9)
        self.assertIn("900 megapixel-masks", error)
        self.assertIn("individual_masks=false", error)


class TestSegmentResult(unittest.TestCase):
    MASKS = ["/view?filename=u.png&subfolder=wrapper/j&type=output",
             "/view?filename=a.png&subfolder=wrapper/j&type=output",
             "/view?filename=b.png&subfolder=wrapper/j&type=output"]
    DETECTIONS = json.dumps([[{"x": 0.0, "y": 0.0, "width": 1024.0, "height": 300.456, "score": 0.93127},
                              {"x": 5.04, "y": 6.0, "width": 7.0, "height": 8.0, "score": 0.61}]], indent=4)

    def test_phrase_detections_with_their_masks(self):
        tag = {"mode": "prompt", "width": 1024, "height": 768, "individual_masks": True}
        self.assertEqual(seg.segment_result(self.DETECTIONS, tag, self.MASKS), {
            "mode": "prompt", "width": 1024, "height": 768, "mask": self.MASKS[0],
            "objects": [{"score": 0.9313, "bbox": [0.0, 0.0, 1024.0, 300.5], "mask": self.MASKS[1]},
                        {"score": 0.61, "bbox": [5.0, 6.0, 7.0, 8.0], "mask": self.MASKS[2]}]})

    def test_union_only_has_no_per_object_masks(self):
        tag = {"mode": "prompt", "width": 1024, "height": 768, "individual_masks": False}
        result = seg.segment_result(self.DETECTIONS, tag, self.MASKS[:1])
        self.assertEqual(result["mask"], self.MASKS[0])
        self.assertNotIn("mask", result["objects"][0])

    def test_nothing_found(self):
        tag = {"mode": "prompt", "width": 8, "height": 8, "individual_masks": True}
        result = seg.segment_result("[\n    []\n]", tag, self.MASKS[:1])
        self.assertEqual((result["mask"], result["objects"]), (self.MASKS[0], []))
        self.assertEqual(seg.segment_result("", tag, [])["mask"], None)
        self.assertEqual(seg.segment_result("garbage", tag, [])["objects"], [])

    def test_boxes_are_the_objects(self):
        tag = {"mode": "boxes", "width": 64, "height": 64, "individual_masks": True,
               "boxes": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]}
        result = seg.segment_result("[[]]", tag, self.MASKS)
        self.assertEqual(result["objects"], [{"score": None, "bbox": [1.0, 2.0, 3.0, 4.0], "mask": self.MASKS[1]},
                                             {"score": None, "bbox": [5.0, 6.0, 7.0, 8.0], "mask": self.MASKS[2]}])

    def test_points_have_no_objects(self):
        tag = {"mode": "points", "width": 64, "height": 64, "individual_masks": False}
        self.assertEqual(seg.segment_result("[[]]", tag, self.MASKS[:1])["objects"], [])


if __name__ == "__main__":
    unittest.main()
