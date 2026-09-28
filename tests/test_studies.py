import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.studies import (
    aggregate_study_results,
    assign_small_cone_slice,
    inspect_split_leakage,
    match_failures,
    write_study_report,
)


class SliceAssignmentTests(unittest.TestCase):
    def test_small_cone_threshold_is_inclusive_and_slices_are_disjoint(self):
        small = [{"class_id": 0, "xywh": [0.5, 0.5, 0.05, 0.1]}]
        ordinary = [{"class_id": 0, "xywh": [0.5, 0.5, 0.1, 0.1]}]
        self.assertEqual(assign_small_cone_slice(small, 0.005), "small_cones")
        self.assertEqual(assign_small_cone_slice(ordinary, 0.005), "ordinary")


class LeakageTests(unittest.TestCase):
    def test_exact_duplicate_and_cross_split_source_group_fail_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "test"):
                path = root / "images" / split / "frame.jpg"
                path.parent.mkdir(parents=True)
                cv2.imwrite(str(path), np.full((24, 24, 3), 127, dtype=np.uint8))
            members = {
                "train": [{"image": "images/train/frame.jpg", "source_group": "run-1"}],
                "test": [{"image": "images/test/frame.jpg", "source_group": "run-1"}],
            }
            result = inspect_split_leakage(members, root)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["source_group_check"], "fail")
        self.assertEqual(len(result["exact_duplicates"]), 1)


class ResultAggregationTests(unittest.TestCase):
    def test_aggregation_reports_target_and_ordinary_deltas(self):
        controls = {"model": "same", "seed": 42}
        baseline = {
            "dataset_fingerprint": "same",
            "controls": controls,
            "evaluation_args": {"device": "cpu"},
            "checkpoint": "best",
            "slice_results": {
                name: {"metrics": {"map50_95": value, "recall": value + 0.1}}
                for name, value in (("full", 0.4), ("small_cones", 0.2), ("ordinary", 0.5))
            },
        }
        intervention = {
            "dataset_fingerprint": "same",
            "controls": controls,
            "evaluation_args": {"device": "cpu"},
            "checkpoint": "best",
            "slice_results": {
                name: {"metrics": {"map50_95": value, "recall": value + 0.1}}
                for name, value in (("full", 0.41), ("small_cones", 0.23), ("ordinary", 0.495))
            },
        }
        rows = aggregate_study_results(baseline, intervention)
        deltas = {(row["slice"], row["metric"]): row["delta"] for row in rows}
        self.assertAlmostEqual(deltas[("small_cones", "map50_95")], 0.03)
        self.assertAlmostEqual(deltas[("ordinary", "map50_95")], -0.005)

    def test_failure_matching_is_class_aware(self):
        annotations = [{"class_id": 0, "xywh": [0.5, 0.5, 0.2, 0.2]}]
        predictions = [
            {"class_id": 1, "confidence": 0.9, "xyxy": [0.4, 0.4, 0.6, 0.6]},
        ]
        failures = match_failures(annotations, predictions)
        self.assertEqual(failures["false_negatives"], [0])
        self.assertEqual(failures["false_positives"], [0])

    def test_report_renders_help_hurt_and_unchanged_failure_examples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            study = root / "study"
            output = root / "report"
            items = []
            for name in ("helps", "hurts", "same"):
                image_path = dataset / "images" / "test" / f"{name}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), np.full((40, 40, 3), 100, dtype=np.uint8))
                items.append({
                    "image": str(image_path.relative_to(dataset)),
                    "label": f"labels/test/{name}.txt",
                    "source_group": None,
                    "annotations": [{"class_id": 0, "xywh": [0.5, 0.5, 0.4, 0.4]}],
                })

            slice_summary = {
                "image_count": len(items),
                "cone_count": len(items),
                "per_class_cone_count": {"0": len(items)},
                "images": items,
            }
            manifest = {
                "dataset_root": str(dataset),
                "dataset_fingerprint": "same",
                "class_names": {"0": "blue_cone"},
                "per_class_min_cones": 10,
                "split_counts": {"train": 10, "val": 2, "test": 3},
                "slice_rule": {"caveat": "Every cone in selected images is evaluated."},
                "separation_limit": "Recording IDs are unavailable.",
                "leakage": {
                    "status": "pass",
                    "source_group_overlaps": [],
                    "exact_duplicates": [],
                    "near_duplicates": [],
                },
                "slices": {
                    "full": slice_summary,
                    "small_cones": slice_summary,
                    "ordinary": slice_summary,
                },
            }
            study.mkdir()
            (study / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            correct = [{"class_id": 0, "confidence": 0.9, "xyxy": [0.3, 0.3, 0.7, 0.7]}]
            wrong = [{"class_id": 1, "confidence": 0.9, "xyxy": [0.3, 0.3, 0.7, 0.7]}]
            baseline_boxes = {"helps": [], "hurts": correct, "same": wrong}
            intervention_boxes = {"helps": correct, "hurts": [], "same": wrong}

            controls = {
                "model": {"name": "yolo11n", "weights": "yolo11n.pt"},
                "seed": 42,
                "trainer_args": {"epochs": 50, "imgsz": 640, "batch": 16},
            }
            result_paths = []
            for name, scale, boxes in (
                ("baseline", 0.5, baseline_boxes),
                ("intervention", 0.9, intervention_boxes),
            ):
                prediction_path = study / f"{name}-predictions.json"
                prediction_path.write_text(json.dumps({
                    "predictions": [
                        {"image": item["image"], "boxes": boxes[Path(item["image"]).stem]}
                        for item in items
                    ]
                }), encoding="utf-8")
                per_class = [{
                    "class_id": 0, "class_name": "blue_cone", "map50_95": 0.4, "recall": 0.5,
                }]
                result = {
                    "dataset_fingerprint": "same",
                    "controls": controls,
                    "evaluation_args": {"device": "cpu", "batch": 1},
                    "checkpoint": "best",
                    "intervention": {"scale": scale},
                    "predictions": prediction_path.name,
                    "slice_results": {
                        slice_name: {"metrics": {"map50_95": 0.4, "recall": 0.5}, "per_class": per_class}
                        for slice_name in ("full", "small_cones", "ordinary")
                    },
                }
                result_path = root / f"{name}.json"
                result_path.write_text(json.dumps(result), encoding="utf-8")
                result_paths.append(result_path)

            report, csv_path = write_study_report(study, *result_paths, output)
            text = report.read_text(encoding="utf-8")
            self.assertIn("does not meet the pre-registered", text)
            self.assertTrue(csv_path.exists())
            self.assertTrue((output / "gallery" / "helps-1.jpg").exists())
            self.assertTrue((output / "gallery" / "hurts-1.jpg").exists())
            self.assertTrue((output / "gallery" / "no_clear_difference-1.jpg").exists())


if __name__ == "__main__":
    unittest.main()
