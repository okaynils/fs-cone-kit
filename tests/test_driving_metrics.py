import unittest

from core.driving_metrics import driving_metrics, evaluate_gates, flatten_metrics


NAMES = {0: "blue_cone", 1: "yellow_cone", 2: "orange_cone"}
BANDS = {"far": [0.0, 0.1], "near": [0.1, 1.0]}


def box(class_id, x, y, height, confidence=None):
    item = {"class_id": class_id, "xyxy": [x, y, x + height * 0.6, y + height]}
    if confidence is not None:
        item["confidence"] = confidence
    return item


def metrics(truth, predictions, colors=("blue_cone", "yellow_cone")):
    return driving_metrics(truth, predictions, NAMES, 0.25, 0.5, BANDS, list(colors))


class DrivingMetricTests(unittest.TestCase):
    def setUp(self):
        # 100 px tall images: a 5 px cone is "far", a 40 px cone is "near".
        self.truth = {
            "a.jpg": {"height": 100, "boxes": [box(0, 10, 10, 40), box(1, 60, 10, 5)]},
            "b.jpg": {"height": 100, "boxes": [box(1, 10, 10, 40), box(2, 60, 60, 5)]},
        }

    def test_band_recall_precision_and_false_positives(self):
        predictions = {
            "a.jpg": [box(0, 10, 10, 40, 0.9), box(1, 60, 10, 5, 0.1)],  # far cone below confidence
            "b.jpg": [box(1, 10, 10, 40, 0.8), box(2, 60, 60, 5, 0.7), box(0, 80, 80, 5, 0.6)],
        }
        result = metrics(self.truth, predictions)
        self.assertEqual(result["recall"], 3 / 4)
        self.assertEqual(result["precision"], 3 / 4)
        self.assertEqual(result["false_positives_per_image"], 0.5)
        self.assertEqual(result["by_band"]["near"]["recall"], 1.0)
        self.assertEqual(result["by_band"]["far"]["recall"], 1 / 2)
        self.assertEqual(result["by_band"]["far"]["precision"], 1 / 2)

    def test_blue_yellow_swap_is_counted_separately_from_misses(self):
        predictions = {
            "a.jpg": [box(1, 10, 10, 40, 0.9), box(1, 60, 10, 5, 0.9)],  # blue called yellow
            "b.jpg": [box(1, 10, 10, 40, 0.9), box(0, 60, 60, 5, 0.9)],  # orange called blue: not a swap
        }
        result = metrics(self.truth, predictions)["color_confusion"]
        self.assertEqual(result["localized"], 3)
        self.assertEqual(result["swapped"], 1)
        self.assertAlmostEqual(result["rate"], 1 / 3)
        self.assertEqual(result["per_class"]["blue_cone"]["rate"], 1.0)
        self.assertEqual(result["per_class"]["yellow_cone"]["rate"], 0.0)

    def test_confusion_is_unavailable_when_the_class_map_lacks_the_colors(self):
        result = metrics(self.truth, {}, colors=("blue", "yellow"))
        self.assertFalse(result["color_confusion"]["available"])
        self.assertIsNone(flatten_metrics(result)["color_confusion_rate"])


class GateTests(unittest.TestCase):
    def test_min_and_max_limits(self):
        checks = evaluate_gates(
            {"recall": 0.8, "recall_far": 0.3, "false_positives_per_image": 0.5},
            {"min_recall": 0.7, "min_recall_far": 0.4, "max_false_positives_per_image": 1.0},
        )
        self.assertEqual({check["name"]: check["passed"] for check in checks}, {
            "min_recall": True,
            "min_recall_far": False,
            "max_false_positives_per_image": True,
        })

    def test_missing_data_fails(self):
        (check,) = evaluate_gates({"recall_far": None}, {"min_recall_far": 0.4})
        self.assertFalse(check["passed"])
        self.assertIn("No data", check["note"])

    def test_unknown_metric_or_prefix_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            evaluate_gates({"recall": 1.0}, {"min_recal": 0.5})
        with self.assertRaisesRegex(ValueError, "min_ or max_"):
            evaluate_gates({"recall": 1.0}, {"recall": 0.5})


if __name__ == "__main__":
    unittest.main()
