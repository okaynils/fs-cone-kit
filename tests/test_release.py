import json
import tempfile
import unittest
from pathlib import Path

import onnxruntime

from core.deployment import OnnxDetector, target_providers
from core.evaluation import sha256_file
from core.release import preprocessing_contract, write_bundle
from tests.test_quantization import save_conv_model


BUNDLE_FILES = {
    "model.onnx", "contract.json", "parity.json", "evaluation.json", "quantization.json",
    "gates.json", "benchmark.json", "experiment.json", "model_card.md", "manifest.json",
}


def check(name, value, limit, passed):
    return {"name": name, "value": value, "limit": limit, "passed": passed}


def records(model_path, calibration=None):
    metrics = {"map50": 0.9, "map50_95": 0.6, "precision": 0.85, "recall": 0.8}
    stage = {"median_batch_latency_ms": 1.0, "p95_batch_latency_ms": 2.0}
    band = {"truths": 4, "predictions": 4, "recall": 0.75, "precision": 1.0}
    row = {
        "model": {"path": str(model_path), "sha256": "f" * 64, "size_bytes": 1000},
        "delta": {"map50": 0.0, "map50_95": -0.01, "precision": 0.0, "recall": -0.02},
        "slices": {"small_cones": {
            "image_count": 2, "metrics": metrics,
            "delta": {"map50_95": -0.03, "recall": -0.04},
        }},
        "calibration": calibration,
    }
    result = {
        "experiment": {
            "experiment": "yolo11n-640", "architecture": "yolo11n", "initial_weights": "yolo11n.pt",
            "epochs": 50, "imgsz": 640, "seed": 42, "git_commit": "abc", "git_dirty": False,
            "dataset_type": "fixture", "dataset_fingerprint": "split-a" * 4, "split_counts": {"test": 2},
        },
        "contract": {
            "input": {"shape": [1, 3, 640, 640], "dtype": "float32", "color_order": "RGB"},
            "letterbox": {"pad_value": 114},
            "class_map": {"0": "blue_cone", "1": "yellow_cone"},
        },
        "parity": {
            "summary": {
                "image_count": 2, "matched_detections": 5, "reference_detections": 5,
                "box_deviation_px": {"max": 0.1, "mean": 0.05},
                "confidence_difference": {"max": 0.001},
                "class_disagreements": 0, "only_in_reference": 0, "only_in_candidate": 0,
            },
            "candidate": {"providers": ["CPUExecutionProvider"]},
            "checks": [check("max_box_deviation_px", 0.1, 1.0, True)],
        },
        "evaluation": {
            "metrics": metrics,
            "dataset_split_counts": {"test": 2},
            "runtime": {"name": "onnxruntime", "version": "1.30.0", "providers": ["CPUExecutionProvider"]},
            "per_class": [{"class_name": "blue_cone", "precision": 0.9, "recall": 0.8, "map50_95": 0.6}],
        },
        "quantization": {"rows": [row]},
        "gates": {
            "metrics": {
                "confidence": 0.25, "truths": 8, "image_count": 2, "recall": 0.75, "precision": 1.0,
                "false_positives_per_image": 0.0,
                "bands": {"far": [0.0, 0.04], "near": [0.04, 1.0]},
                "by_band": {"far": band, "near": band},
                "color_confusion": {
                    "available": True, "classes": ["blue_cone", "yellow_cone"],
                    "swapped": 0, "localized": 6, "rate": 0.0,
                },
            },
            "limits_source": "team-chosen; not derived from FSG rules",
            "checks": [check("min_recall", 0.75, 0.7, True)],
        },
        "benchmark": {
            "note": "One image at a time.",
            "hardware": {"cpu": "Fixture CPU", "accelerator": None},
            "runtime": {"name": "onnxruntime", "version": "1.30.0", "providers": ["CPUExecutionProvider"]},
            "benchmark_context_id": "abcdef123456",
            "stages": {"preprocess": stage, "inference": stage, "postprocess": stage, "total": stage},
            "results": {"throughput_images_per_second": 500.0},
        },
    }
    if calibration:
        result["calibration"] = calibration
    return result


IDENTITY = {"target": "onnxruntime", "precision": "int8", "device": "cpu", "dataset_fingerprint": "split-a"}


class BundleTests(unittest.TestCase):
    def test_layout_hashes_and_passing_card(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "best_int8.onnx"
            model.write_bytes(b"onnx")
            bundle = root / "release" / "onnxruntime-int8"
            bundle.mkdir(parents=True)
            (bundle / "stale.json").write_text("{}")
            calibration = {"split": "train", "image_count": 64, "images_sha256": "c" * 64}
            checks = records(model)["parity"]["checks"] + records(model)["gates"]["checks"]
            manifest = write_bundle(bundle, model, records(model, calibration), [], checks, IDENTITY)

            self.assertEqual({path.name for path in bundle.iterdir()}, BUNDLE_FILES | {"calibration.json"})
            self.assertEqual(manifest["status"], "pass")
            on_disk = json.loads((bundle / "manifest.json").read_text())
            for name, digest in on_disk["files"].items():
                self.assertEqual(sha256_file(bundle / name), digest)
            card = (bundle / "model_card.md").read_text()
            self.assertIn("**Status: PASS**", card)
            self.assertIn("| small_cones | 2 |", card)
            self.assertIn("64 images from the train split", card)
            self.assertIn("not derived from FSG rules", card)

    def test_a_failed_check_marks_the_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "best.onnx"
            model.write_bytes(b"onnx")
            failing = [check("min_recall_far", 0.2, 0.4, False)]
            manifest = write_bundle(root / "bundle", model, records(model), [], failing, IDENTITY)
            self.assertEqual(manifest["status"], "fail")
            self.assertEqual(manifest["failed_checks"], ["min_recall_far"])
            self.assertNotIn("calibration.json", {path.name for path in (root / "bundle").iterdir()})
            card = (root / "bundle" / "model_card.md").read_text()
            self.assertIn("**Status: FAIL** — do not put this on the car.", card)
            self.assertIn("`min_recall_far`", card)


class ContractTests(unittest.TestCase):
    def test_contract_comes_from_the_model_and_class_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "best.onnx"
            save_conv_model(model, size=32)
            detector = OnnxDetector(model)
            contract = preprocessing_contract(
                detector, {"yellow_cone": 1, "blue_cone": 0}, {"blue_cone": [0, 102, 255]},
                {"confidence": 0.25, "nms_iou": 0.7, "max_det": 300},
            )
        self.assertEqual(contract["input"]["shape"], [1, 3, 32, 32])
        self.assertEqual(contract["input"]["color_order"], "RGB")
        self.assertEqual(contract["class_map"], {"0": "blue_cone", "1": "yellow_cone"})
        self.assertEqual(contract["postprocess"]["confidence"], 0.25)
        self.assertEqual(contract["letterbox"]["pad_value"], 114)


class TargetTests(unittest.TestCase):
    def test_unavailable_providers_are_refused_not_replaced(self):
        available = onnxruntime.get_available_providers()
        if "TensorrtExecutionProvider" not in available:
            with self.assertRaisesRegex(RuntimeError, "No TensorRT numbers were produced"):
                target_providers("tensorrt", "cuda", "fp16")
        if "CUDAExecutionProvider" not in available:
            with self.assertRaisesRegex(RuntimeError, "not available"):
                target_providers("onnxruntime", "cuda")
        self.assertEqual(target_providers("onnxruntime", "cpu"), ["CPUExecutionProvider"])
        with self.assertRaises(ValueError):
            target_providers("openvino", "cpu")


if __name__ == "__main__":
    unittest.main()
