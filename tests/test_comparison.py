import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from core.comparison import collect_comparison_rows, write_comparison


def create_experiment(root: Path, name: str, fingerprint: str, map50: float) -> Path:
    experiment = root / name
    record = experiment / "experiment"
    (record / "evaluations").mkdir(parents=True)
    (record / "benchmarks").mkdir()
    config = {
        "seed": 42,
        "model": {"name": name, "weights": f"{name}.pt"},
        "trainer": {"args": {"imgsz": 640, "batch": 8, "epochs": 10}},
    }
    (record / "config.yaml").write_text(yaml.safe_dump(config))
    (record / "metadata.json").write_text(json.dumps({"git": {"commit": "abc"}}))
    evaluation = {
        "result_type": "final_test", "split": "test", "dataset_fingerprint": fingerprint,
        "metrics": {"map50": map50, "map50_95": 0.4, "precision": 0.7, "recall": 0.6},
        "model": {"parameter_count": 100, "size_bytes": 200},
    }
    (record / "evaluations/test.json").write_text(json.dumps(evaluation))
    benchmark = {
        "benchmark_context_id": "cpu-context",
        "protocol": {"device": "cpu", "precision": "fp32", "batch_size": 1, "input_size": 640},
        "results": {
            "median_latency_ms_per_image": 10.0, "p95_batch_latency_ms": 12.0,
            "throughput_images_per_second": 100.0,
        },
    }
    (record / "benchmarks/cpu.json").write_text(json.dumps(benchmark))
    return experiment


class ComparisonTests(unittest.TestCase):
    def test_same_split_is_marked_comparable_and_exports_both_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = create_experiment(root, "yolo11n", "split-a", 0.8)
            second = create_experiment(root, "yolo11s", "split-a", 0.85)
            rows = collect_comparison_rows([first, second])
            self.assertTrue(all(row["accuracy_comparable"] for row in rows))
            csv_path, markdown_path = write_comparison(rows, root / "report")
            with csv_path.open() as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(len(exported), 2)
            self.assertIn("yolo11s", markdown_path.read_text())

    def test_different_splits_are_marked_non_comparable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = create_experiment(root, "first", "split-a", 0.8)
            second = create_experiment(root, "second", "split-b", 0.9)
            rows = collect_comparison_rows([first, second])
            self.assertFalse(any(row["accuracy_comparable"] for row in rows))

    def test_different_evaluation_settings_are_marked_non_comparable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = create_experiment(root, "first", "split-a", 0.8)
            second = create_experiment(root, "second", "split-a", 0.9)
            first_eval = first / "experiment/evaluations/test.json"
            second_eval = second / "experiment/evaluations/test.json"
            first_payload = json.loads(first_eval.read_text())
            second_payload = json.loads(second_eval.read_text())
            first_payload["evaluation_context_id"] = "img-640"
            second_payload["evaluation_context_id"] = "img-1280"
            first_eval.write_text(json.dumps(first_payload))
            second_eval.write_text(json.dumps(second_payload))
            rows = collect_comparison_rows([first, second])
            self.assertFalse(any(row["accuracy_comparable"] for row in rows))


if __name__ == "__main__":
    unittest.main()
