import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
from onnx import TensorProto, helper

from core.deployment import (
    check_tolerances,
    compare_detections,
    letterbox,
    match_detections,
    onnx_precision,
    postprocess,
    select_parity_images,
)
from core.evaluate import evaluation_filename, resolve_model_path


def detection(class_id, confidence, xyxy):
    return {"class_id": class_id, "confidence": confidence, "xyxy": list(xyxy)}


def raw_output(rows, class_count=2):
    """Build a 1x(4+classes)xN tensor from (cx, cy, w, h, class_id, score) rows."""
    output = np.zeros((1, 4 + class_count, len(rows)), dtype=np.float32)
    for index, (cx, cy, width, height, class_id, score) in enumerate(rows):
        output[0, :4, index] = (cx, cy, width, height)
        output[0, 4 + class_id, index] = score
    return output


def save_model(path, initializer_type=TensorProto.FLOAT, quantized=False):
    weight = helper.make_tensor(
        "w", initializer_type, [1], [1.0] if initializer_type == TensorProto.FLOAT else [15360]
    )
    if quantized:
        scale = helper.make_tensor("s", TensorProto.FLOAT, [], [1.0])
        nodes = [
            helper.make_node("QuantizeLinear", ["x", "s"], ["q"]),
            helper.make_node("DequantizeLinear", ["q", "s"], ["y"]),
        ]
        initializers = [scale]
    else:
        nodes = [helper.make_node("Identity", ["x"], ["y"])]
        initializers = [weight]
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        initializers,
    )
    onnx.save(helper.make_model(graph), str(path))


class LetterboxTests(unittest.TestCase):
    def test_matches_ultralytics_square_letterbox(self):
        from ultralytics.data.augment import LetterBox

        rng = np.random.default_rng(0)
        for shape in ((480, 640, 3), (720, 1280, 3), (811, 1080, 3), (64, 64, 3)):
            image = rng.integers(0, 255, size=shape, dtype=np.uint8)
            expected = LetterBox(new_shape=(320, 320), auto=False)(image=image)
            np.testing.assert_array_equal(letterbox(image, 320), expected)


class PostprocessTests(unittest.TestCase):
    def test_nms_is_per_class_and_boxes_return_to_image_pixels(self):
        # A 200x100 image letterboxed to 100x100: gain 0.5, 25 px of padding top and bottom.
        output = raw_output([
            (50, 50, 20, 20, 0, 0.9),
            (51, 50, 20, 20, 0, 0.8),  # same class, overlaps: suppressed
            (51, 50, 20, 20, 1, 0.7),  # other class, overlaps: kept
            (10, 30, 4, 4, 1, 0.1),    # below confidence
        ])
        detections = postprocess(output, 100, (100, 200), confidence=0.25, iou=0.7)
        self.assertEqual([item["class_id"] for item in detections], [0, 1])
        np.testing.assert_allclose(detections[0]["xyxy"], [80, 30, 120, 70])
        self.assertAlmostEqual(detections[0]["confidence"], 0.9, places=6)

    def test_boxes_are_clipped_to_the_image(self):
        output = raw_output([(2, 50, 10, 10, 0, 0.9)])
        (box,) = postprocess(output, 100, (100, 100), confidence=0.25, iou=0.7)
        self.assertEqual(box["xyxy"][0], 0.0)


class MatchingTests(unittest.TestCase):
    def test_greedy_matching_ignores_class_and_prefers_highest_iou(self):
        reference = [detection(0, 0.9, (0, 0, 10, 10)), detection(1, 0.9, (50, 50, 60, 60))]
        candidate = [detection(1, 0.9, (1, 0, 11, 10)), detection(0, 0.9, (0, 0, 10, 10))]
        matches, missing, extra = match_detections(reference, candidate)
        self.assertEqual(matches, [(0, 1)])
        self.assertEqual(missing, [1])
        self.assertEqual(extra, [0])

    def test_scores_that_straddle_the_threshold_are_compared_not_counted_missing(self):
        reference = {"a.jpg": [detection(0, 0.26, (0, 0, 10, 10))]}
        candidate = {"a.jpg": [detection(0, 0.24, (0, 0, 10, 10))]}
        summary = compare_detections(reference, candidate, confidence=0.25)
        self.assertEqual(summary["only_in_reference"], 0)
        self.assertAlmostEqual(summary["confidence_difference"]["max"], 0.02)

    def test_reports_deviation_class_flips_and_detections_in_one_model(self):
        reference = {
            "a.jpg": [detection(0, 0.9, (0, 0, 10, 10)), detection(1, 0.8, (40, 40, 50, 50))],
            "b.jpg": [detection(0, 0.9, (0, 0, 10, 10))],
        }
        candidate = {
            "a.jpg": [detection(0, 0.88, (0, 0, 10, 12)), detection(0, 0.8, (40, 40, 50, 50))],
            "b.jpg": [detection(1, 0.5, (70, 70, 80, 80))],
        }
        summary = compare_detections(reference, candidate, confidence=0.25)
        self.assertEqual(summary["matched_detections"], 2)
        self.assertEqual(summary["box_deviation_px"]["max"], 2.0)
        self.assertEqual(summary["class_disagreements"], 1)
        self.assertEqual(summary["only_in_reference"], 1)
        self.assertEqual(summary["only_in_candidate"], 1)
        self.assertAlmostEqual(summary["unmatched_rate"], 2 / 3)
        self.assertEqual(summary["images_with_issues"], [
            {"image": "b.jpg", "issues": 2},
            {"image": "a.jpg", "issues": 1},
        ])


class ToleranceTests(unittest.TestCase):
    def summary(self, **values):
        base = compare_detections(
            {"a.jpg": [detection(0, 0.9, (0, 0, 10, 10))]},
            {"a.jpg": [detection(0, 0.9, (0, 0, 10, 10))]},
            confidence=0.25,
        )
        base.update(values)
        return base

    def test_pass_and_fail_against_limits(self):
        checks = check_tolerances(
            self.summary(box_deviation_px={"max": 1.5, "mean": 1.0, "p95": 1.5}),
            {"max_box_deviation_px": 1.0, "max_unmatched_rate": 0.0},
        )
        self.assertEqual({check["name"]: check["passed"] for check in checks}, {
            "max_box_deviation_px": False,
            "max_unmatched_rate": True,
        })

    def test_unknown_tolerance_is_an_error(self):
        with self.assertRaises(ValueError):
            check_tolerances(self.summary(), {"max_box_drift": 1.0})

    def test_no_reference_detections_cannot_pass(self):
        checks = check_tolerances(self.summary(reference_detections=0), {"max_unmatched_rate": 1.0})
        self.assertFalse(checks[0]["passed"])


class SelectionAndPathTests(unittest.TestCase):
    def test_parity_images_are_sorted_and_evenly_spaced(self):
        members = [{"image": f"images/test/{index:02d}.jpg"} for index in reversed(range(10))]
        self.assertEqual(
            select_parity_images(members, 3),
            ["images/test/00.jpg", "images/test/03.jpg", "images/test/06.jpg"],
        )
        self.assertEqual(len(select_parity_images(members, 0)), 10)

    def test_checkpoint_keeps_the_name_compare_reads(self):
        self.assertEqual(evaluation_filename("test", Path("best.pt")), "test.json")
        self.assertEqual(evaluation_filename("test", Path("best_int8.onnx")), "test_best_int8_onnx.json")

    def test_model_path_resolves_relative_to_the_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "weights" / "best.onnx"
            model.parent.mkdir()
            model.touch()
            self.assertEqual(resolve_model_path(root, model="weights/best.onnx"), model.resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_model_path(root, model="weights/missing.onnx")


class PrecisionTests(unittest.TestCase):
    def test_precision_is_read_from_the_graph(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_model(root / "fp32.onnx")
            save_model(root / "fp16.onnx", initializer_type=TensorProto.FLOAT16)
            save_model(root / "int8.onnx", quantized=True)
            self.assertEqual(onnx_precision(root / "fp32.onnx"), "fp32")
            self.assertEqual(onnx_precision(root / "fp16.onnx"), "fp16")
            self.assertEqual(onnx_precision(root / "int8.onnx"), "int8")


class ExportFailureTests(unittest.TestCase):
    def test_failed_export_raises_instead_of_printing(self):
        from core.trainers.ultralytics import UltralyticsTrainer

        trainer = UltralyticsTrainer(args={"imgsz": 32})
        with mock.patch("core.trainers.ultralytics.YOLO") as yolo:
            yolo.return_value.export.side_effect = RuntimeError("opset unsupported")
            with self.assertRaisesRegex(RuntimeError, "ONNX export failed.*opset unsupported"):
                trainer.export_checkpoint_to_onnx(Path("best.pt"))
            yolo.return_value.export.side_effect = None
            yolo.return_value.export.return_value = "missing.onnx"
            with self.assertRaisesRegex(RuntimeError, "reported no file"):
                trainer.export_checkpoint_to_onnx(Path("best.pt"))


if __name__ == "__main__":
    unittest.main()
