import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, numpy_helper

from core.deployment import onnx_precision, preprocess
from core.quantization import (
    build_quantization_report,
    convert_fp16,
    load_study_manifest,
    quantization_markdown,
    quantize_int8,
    quantized_path,
    select_calibration_images,
)


def dataset_info(train, val=(), test=()):
    return {
        "fingerprint": "split-a",
        "manifest": {"splits": {
            "train": [{"image": name} for name in train],
            "val": [{"image": name} for name in val],
            "test": [{"image": name} for name in test],
        }},
    }


def evaluation(map50_95, context="ctx-a", fingerprint="split-a", recall=0.5):
    return {
        "dataset_fingerprint": fingerprint,
        "evaluation_context_id": context,
        "runtime": {"name": "onnxruntime"},
        "metrics": {"map50": 0.9, "map50_95": map50_95, "precision": 0.8, "recall": recall},
        "per_class": [{"class_name": "blue_cone", "map50_95": map50_95, "recall": recall}],
    }


def save_conv_model(path, size=16):
    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(rng.normal(size=(4, 3, 3, 3)).astype(np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["images", "w"], ["output0"], pads=[1, 1, 1, 1])],
        "conv",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4, size, size])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))


class CalibrationSelectionTests(unittest.TestCase):
    def test_uses_train_only(self):
        info = dataset_info([f"images/train/{index}.jpg" for index in range(10)], test=["images/test/0.jpg"])
        selected = select_calibration_images(info, 4)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(name.startswith("images/train/") for name in selected))

    def test_refuses_images_that_are_also_held_out(self):
        info = dataset_info(["images/shared.jpg"], test=["images/shared.jpg"])
        with self.assertRaisesRegex(RuntimeError, "also appear in val or test"):
            select_calibration_images(info, 4)

    def test_refuses_an_empty_train_split(self):
        with self.assertRaises(ValueError):
            select_calibration_images(dataset_info([]), 4)

    def test_variant_names(self):
        self.assertEqual(quantized_path(Path("w/best.onnx"), "int8"), Path("w/best_int8.onnx"))
        self.assertEqual(quantized_path(Path("w/best.onnx"), "fp32"), Path("w/best.onnx"))


class ReportTests(unittest.TestCase):
    def test_deltas_against_fp32_and_non_comparable_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for name in ("best.onnx", "best_fp16.onnx", "best_int8.onnx"):
                path = root / name
                path.write_bytes(name.encode())
                paths.append(path)
            slices = {"small_cones": {
                "evaluation_context_id": "slice-ctx", "image_count": 3, "cone_count": 9,
                "metrics": {"map50": 0.5, "map50_95": 0.30, "precision": 0.6, "recall": 0.40},
            }, "ordinary": {"status": "empty"}}
            int8_slices = {"small_cones": {
                **slices["small_cones"],
                "metrics": {"map50": 0.4, "map50_95": 0.25, "precision": 0.6, "recall": 0.35},
            }, "ordinary": {"status": "empty"}}
            report = build_quantization_report(root, [
                ("fp32", paths[0], evaluation(0.50), slices, None),
                ("fp16", paths[1], evaluation(0.49, context="ctx-b", fingerprint="split-b"), None, None),
                ("int8", paths[2], evaluation(0.45, recall=0.45), int8_slices, {
                    "image_count": 2, "images_sha256": "abc123abc123ff",
                }),
            ], "Slices from fixture.")
        fp32, fp16, int8 = report["rows"]
        self.assertEqual(fp32["delta"]["map50_95"], 0.0)
        self.assertFalse(fp16["accuracy_comparable"])
        self.assertIsNone(fp16["delta"]["map50_95"])
        self.assertAlmostEqual(int8["delta"]["map50_95"], -0.05)
        self.assertAlmostEqual(int8["per_class_delta"]["blue_cone"]["recall"], -0.05)
        self.assertAlmostEqual(int8["slices"]["small_cones"]["delta"]["recall"], -0.05)
        self.assertEqual(int8["slices"]["ordinary"], {"status": "empty"})
        markdown = quantization_markdown(report)
        self.assertIn("| int8 | small_cones |", markdown)
        self.assertIn("| fp16 | test |", markdown)
        self.assertIn("calibration: 2 train images", markdown)

    def test_fp32_baseline_must_come_first(self):
        with self.assertRaises(ValueError):
            build_quantization_report(Path("."), [("int8", Path("x"), evaluation(0.4), None, None)], None)


class StudyManifestTests(unittest.TestCase):
    def test_only_a_clean_manifest_for_the_same_split_is_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, note = load_study_manifest(root, "split-a")
            self.assertIsNone(manifest)
            self.assertIn("No study manifest", note)
            path = root / "manifest.json"
            path.write_text(json.dumps({"dataset_fingerprint": "split-b", "leakage": {"status": "pass"}}))
            self.assertIn("different split", load_study_manifest(root, "split-a")[1])
            path.write_text(json.dumps({"dataset_fingerprint": "split-a", "leakage": {"status": "fail"}}))
            self.assertIn("leakage", load_study_manifest(root, "split-a")[1])
            path.write_text(json.dumps({"dataset_fingerprint": "split-a", "leakage": {"status": "pass"}}))
            self.assertIsNotNone(load_study_manifest(root, "split-a")[0])


class ConversionTests(unittest.TestCase):
    def test_fp16_and_int8_variants_run_with_float32_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "best.onnx"
            save_conv_model(source)
            for index in range(3):
                image = np.random.default_rng(index).integers(0, 255, (24, 32, 3), dtype=np.uint8)
                (root / "images").mkdir(exist_ok=True)
                cv2.imwrite(str(root / "images" / f"{index}.jpg"), image)

            fp16 = convert_fp16(source, root / "best_fp16.onnx")
            int8 = quantize_int8(
                source, root / "best_int8.onnx", root, [f"images/{index}.jpg" for index in range(3)],
                {"op_types": ["Conv"], "per_channel": True, "calibrate_method": "MinMax"},
            )
            self.assertEqual(onnx_precision(fp16), "fp16")
            self.assertEqual(onnx_precision(int8), "int8")
            tensor = preprocess(cv2.imread(str(root / "images" / "0.jpg")), 16)
            outputs = []
            for path in (source, fp16, int8):
                session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                outputs.append(session.run(None, {"images": tensor})[0].ravel())
            np.testing.assert_allclose(outputs[1], outputs[0], atol=0.05)
            self.assertGreater(np.corrcoef(outputs[2], outputs[0])[0, 1], 0.95)


if __name__ == "__main__":
    unittest.main()
