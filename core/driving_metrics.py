"""Metrics a driverless stack feels, and pass/fail gates on them.

mAP averages over confidences the car never uses. These metrics are taken at
one deployment confidence: recall and precision by box-size band, blue/yellow
confusion, and false positives per image.

Box height relative to image height is a stand-in for distance. A cone of fixed
size gets shorter in the image as it gets further away, but truncation, camera
pitch, and lens distortion all break that. It is not a distance measurement.
"""

from __future__ import annotations

from typing import Any

from core.deployment import box_iou, match_detections


def _band(height: float, bands: dict[str, list[float]]) -> str | None:
    for name, (low, high) in bands.items():
        if low <= height < high or (high >= 1.0 and height >= low):
            return name
    return None


def _class_aware_matches(
    truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[set[int], set[int]]:
    """Standard detection matching: highest confidence first, same class, best IoU."""
    matched_truths: set[int] = set()
    matched_predictions: set[int] = set()
    order = sorted(range(len(predictions)), key=lambda index: -predictions[index]["confidence"])
    for prediction_index in order:
        prediction = predictions[prediction_index]
        best, best_iou = None, iou_threshold
        for truth_index, truth in enumerate(truths):
            if truth_index in matched_truths or truth["class_id"] != prediction["class_id"]:
                continue
            overlap = box_iou(truth["xyxy"], prediction["xyxy"])
            if overlap >= best_iou:
                best, best_iou = truth_index, overlap
        if best is not None:
            matched_truths.add(best)
            matched_predictions.add(prediction_index)
    return matched_truths, matched_predictions


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def driving_metrics(
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    class_names: dict[int, str],
    confidence: float,
    iou_threshold: float,
    bands: dict[str, list[float]],
    color_classes: list[str],
) -> dict[str, Any]:
    """`ground_truth` maps image -> {"height": px, "boxes": [{class_id, xyxy}]} in pixels."""
    totals = {"truths": 0, "predictions": 0, "true_positives": 0, "false_positives": 0}
    band_counts = {
        name: {"truths": 0, "found": 0, "predictions": 0, "correct": 0} for name in bands
    }
    ids = {name: class_id for class_id, name in class_names.items()}
    color_ids = [ids.get(name) for name in color_classes]
    colors_known = len(color_ids) == 2 and None not in color_ids
    confusion = {"localized": 0, "swapped": 0}
    per_color = {name: {"localized": 0, "swapped": 0} for name in color_classes}

    for image, truth in ground_truth.items():
        height = float(truth["height"])
        truths = truth["boxes"]
        kept = [item for item in predictions.get(image, []) if item["confidence"] >= confidence]
        found, correct = _class_aware_matches(truths, kept, iou_threshold)
        totals["truths"] += len(truths)
        totals["predictions"] += len(kept)
        totals["true_positives"] += len(correct)
        totals["false_positives"] += len(kept) - len(correct)
        for index, item in enumerate(truths):
            band = _band((item["xyxy"][3] - item["xyxy"][1]) / height, bands)
            if band:
                band_counts[band]["truths"] += 1
                band_counts[band]["found"] += index in found
        for index, item in enumerate(kept):
            band = _band((item["xyxy"][3] - item["xyxy"][1]) / height, bands)
            if band:
                band_counts[band]["predictions"] += 1
                band_counts[band]["correct"] += index in correct

        if colors_known:
            # Class-agnostic pairing: the box is in the right place, is the colour right?
            pairs, _, _ = match_detections(truths, kept, iou_threshold)
            for truth_index, prediction_index in pairs:
                truth_class = truths[truth_index]["class_id"]
                if truth_class not in color_ids:
                    continue
                other = color_ids[1 - color_ids.index(truth_class)]
                name = class_names[truth_class]
                swapped = kept[prediction_index]["class_id"] == other
                confusion["localized"] += 1
                confusion["swapped"] += swapped
                per_color[name]["localized"] += 1
                per_color[name]["swapped"] += swapped

    image_count = len(ground_truth)
    return {
        "image_count": image_count,
        "confidence": confidence,
        "match_iou": iou_threshold,
        "band_measure": "box height / image height",
        "bands": bands,
        "truths": totals["truths"],
        "predictions": totals["predictions"],
        "recall": _rate(totals["true_positives"], totals["truths"]),
        "precision": _rate(totals["true_positives"], totals["predictions"]),
        "false_positives_per_image": _rate(totals["false_positives"], image_count),
        "by_band": {
            name: {
                "truths": counts["truths"],
                "predictions": counts["predictions"],
                "recall": _rate(counts["found"], counts["truths"]),
                "precision": _rate(counts["correct"], counts["predictions"]),
            }
            for name, counts in band_counts.items()
        },
        "color_confusion": {
            "classes": color_classes,
            "available": colors_known,
            "localized": confusion["localized"],
            "swapped": confusion["swapped"],
            "rate": _rate(confusion["swapped"], confusion["localized"]) if colors_known else None,
            "per_class": {
                name: {**counts, "rate": _rate(counts["swapped"], counts["localized"])}
                for name, counts in per_color.items()
            } if colors_known else {},
        },
    }


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    """The names gate limits refer to, e.g. recall_far or color_confusion_rate."""
    flat = {
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "false_positives_per_image": metrics["false_positives_per_image"],
        "color_confusion_rate": metrics["color_confusion"]["rate"],
    }
    for name, band in metrics["by_band"].items():
        flat[f"recall_{name}"] = band["recall"]
        flat[f"precision_{name}"] = band["precision"]
    return flat


def evaluate_gates(values: dict[str, float | None], limits: dict[str, float]) -> list[dict[str, Any]]:
    """`min_<metric>` and `max_<metric>` limits. A metric with no data fails; it cannot be certified."""
    checks = []
    for name, limit in sorted(limits.items()):
        kind, _, metric = name.partition("_")
        if kind not in {"min", "max"} or not metric:
            raise ValueError(f"Gate {name!r} must start with min_ or max_")
        if metric not in values:
            raise ValueError(f"Gate {name!r} refers to unknown metric {metric!r}; known: {sorted(values)}")
        value = values[metric]
        if value is None:
            checks.append({
                "name": name, "metric": metric, "limit": float(limit), "value": None, "passed": False,
                "note": "No data for this metric on the test split.",
            })
            continue
        passed = value >= limit if kind == "min" else value <= limit
        checks.append({
            "name": name, "metric": metric, "limit": float(limit), "value": float(value), "passed": passed,
        })
    return checks
