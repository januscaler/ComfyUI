"""fx-track requests and answers (api_wrapper/track.py): pure, no comfy."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api_wrapper import results, track  # noqa: E402
from api_wrapper import workflows as wf  # noqa: E402


class TestTrackRequest(unittest.TestCase):
    def test_an_uploaded_mask_wins_over_the_fields(self):
        request = track.track_request({"prompt": "logo", "threshold": "0.3", "detect_interval": "5"}, 640, 360, True)
        self.assertEqual(request, {"mode": "mask", "threshold": 0.3, "detect_interval": 5})

    def test_a_phrase_keeps_its_best_match(self):
        request = track.track_request({"prompt": "logo on his shirt"}, 640, 360, False)
        self.assertEqual((request["mode"], request["text"], request["individual_masks"], request["detect_interval"]),
                         ("prompt", "logo on his shirt:1", False, 1))

    def test_points_and_boxes_are_checked_against_the_frame(self):
        request = track.track_request({"positive_coords": "[[10, 20]]", "negative_coords": "[[1, 1]]"}, 640, 360, False)
        self.assertEqual((request["positive"], request["negative"]), ([{"x": 10.0, "y": 20.0}], [{"x": 1.0, "y": 1.0}]))
        with self.assertRaises(ValueError):
            track.track_request({"positive_coords": "[[700, 20]]"}, 640, 360, False)
        request = track.track_request({"boxes": json.dumps([[0, 0, 10, 10]] * 4)}, 640, 360, False)
        self.assertEqual(len(request["boxes"]), 4)
        with self.assertRaises(ValueError) as ctx:
            track.track_request({"boxes": json.dumps([[0, 0, 10, 10]] * 5)}, 640, 360, False)
        self.assertIn("at most 4", str(ctx.exception))

    def test_nothing_to_track_names_the_mask_too(self):
        with self.assertRaises(ValueError) as ctx:
            track.track_request({"threshold": "0.5"}, 640, 360, False)
        self.assertIn("'mask'", str(ctx.exception))
        with self.assertRaises(ValueError):
            track.track_request({"prompt": "a", "boxes": "[[0, 0, 5, 5]]"}, 640, 360, False)

    def test_bounds(self):
        for fields in ({"threshold": "1.5"}, {"detect_interval": "0"}, {"detect_interval": "2.5"},
                       {"detect_interval": "601"}):
            with self.assertRaises(ValueError):
                track.track_request(fields, 640, 360, True)

    def test_summary(self):
        self.assertEqual(track.track_summary({"mode": "mask"}), "the uploaded frame-0 mask")
        self.assertEqual(track.track_summary(track.track_request({"prompt": "cup"}, 64, 64, False)), "'cup:1'")


class TestTrackBudget(unittest.TestCase):
    def test_30_s_of_1080p30_fits_the_default(self):
        self.assertIsNone(track.track_check_budget(1920, 1080, 900))
        detail = track.track_check_budget(1920, 1080, 1200)
        self.assertIn("1200 frames of 1920x1080", detail)
        self.assertIn("at most about 964 frames", detail)


class TestTrackResult(unittest.TestCase):
    TAG = {"workflow": "fx-track", "width": 640, "height": 360, "fps": 29.97}

    def test_every_frame_gets_an_entry(self):
        boxes = [[{"x": 10.04, "y": 20, "width": 30, "height": 40, "score": 1.0, "area": 0.0052083333}],
                 [],
                 [{"x": 1, "y": 2, "width": 3, "height": 4, "score": 0.2, "area": 0.1},
                  {"x": 5, "y": 6, "width": 7, "height": 8, "score": 0.9, "area": 0.2}]]
        self.assertEqual(track.track_result(json.dumps(boxes), self.TAG, ["/view?filename=fx-track.mp4"]), {
            "width": 640, "height": 360, "fps": 29.97,
            "frames": [
                {"i": 0, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 1.0, "area": 0.005208},
                {"i": 1, "bbox": None, "score": 0.0, "area": 0.0},
                {"i": 2, "bbox": [5.0, 6.0, 7.0, 8.0], "score": 0.9, "area": 0.2},
            ]})

    def test_unreadable_text_is_no_frames(self):
        for text in ("", "not json", "{}"):
            self.assertEqual(track.track_result(text, self.TAG, [])["frames"], [])

    def test_served_from_the_tagged_node(self):
        graph = {"15": results.tag_result({"class_type": "PreviewAny", "inputs": {}}, "fx-track",
                                          width=8, height=8, fps=25.0)}
        outputs = {"15": {"text": [json.dumps([[], [{"x": 0, "y": 0, "width": 8, "height": 8, "score": 1.0,
                                                     "area": 1.0}]])]}}
        answer = results.text_result(graph, outputs, wf.WORKFLOWS)
        self.assertEqual(answer["fps"], 25.0)
        self.assertEqual(answer["frames"][1], {"i": 1, "bbox": [0.0, 0.0, 8.0, 8.0], "score": 1.0, "area": 1.0})


if __name__ == "__main__":
    unittest.main()
