import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.evaluate import _checkpoint_path
from core.evaluation import serialize_ultralytics_evaluation


class Parameter:
    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


class EvaluationSerializationTests(unittest.TestCase):
    def test_serializes_standard_and_per_class_metrics(self):
        box = SimpleNamespace(
            p=[0.8, 0.6], r=[0.7, 0.5], ap50=[0.9, 0.7], ap=[0.5, 0.3],
            ap_class_index=[0, 1], mp=0.7, mr=0.6, map50=0.8, map=0.4,
        )
        metrics = SimpleNamespace(
            box=box,
            names={0: "blue", 1: "yellow"},
            confusion_matrix=SimpleNamespace(matrix=[[4, 1, 0], [2, 3, 0], [0, 0, 0]]),
            results_dict={"metrics/mAP50(B)": 0.8},
            speed={"inference": 1.25},
        )
        model = SimpleNamespace(model=SimpleNamespace(parameters=lambda: [Parameter(10), Parameter(5)]))
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best.pt"
            checkpoint.write_bytes(b"weights")
            result = serialize_ultralytics_evaluation(
                metrics, model, checkpoint, "test",
                {"fingerprint": "same-split", "split_counts": {"test": 2}},
                {"device": "cpu"},
            )
        self.assertEqual(result["result_type"], "final_test")
        self.assertEqual(result["metrics"]["map50_95"], 0.4)
        self.assertEqual(result["per_class"][1]["class_name"], "yellow")
        self.assertEqual(result["model"]["parameter_count"], 15)
        self.assertEqual(result["confusion_matrix"][0][0], 4)
        self.assertEqual(result["confusion_matrix_labels"], ["blue", "yellow", "background"])
        self.assertTrue(result["evaluation_context_id"])

    def test_interrupted_run_finds_last_checkpoint_without_training_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "ultralytics_files/weights/last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            self.assertEqual(_checkpoint_path(root, "last"), checkpoint)


if __name__ == "__main__":
    unittest.main()
