"""Conversion of Ultralytics validation output into a stable JSON schema."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

from core.experiments import SCHEMA_VERSION


def _tolist(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _float(value: Any) -> float | None:
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    return result if math.isfinite(result) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize_ultralytics_evaluation(
    metrics: Any,
    model: Any,
    checkpoint: Path,
    split: str,
    dataset_info: dict[str, Any],
    evaluation_args: dict[str, Any],
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned result without retaining Ultralytics Python objects."""
    result_dict = dict(getattr(metrics, "results_dict", {}) or {})
    box = getattr(metrics, "box", None)
    names = getattr(metrics, "names", None) or getattr(model, "names", {}) or {}
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(key): str(value) for key, value in names.items()}

    precision = _tolist(getattr(box, "p", []))
    recall = _tolist(getattr(box, "r", []))
    ap50 = _tolist(getattr(box, "ap50", []))
    ap = _tolist(getattr(box, "ap", []))
    class_ids = _tolist(getattr(box, "ap_class_index", range(len(ap50))))

    per_class = []
    metric_index = {int(class_id): index for index, class_id in enumerate(class_ids)}
    for class_id in sorted(set(names) | set(metric_index)):
        index = metric_index.get(class_id)
        per_class.append({
            "class_id": class_id,
            "class_name": names.get(class_id, str(class_id)),
            "precision": _float(precision[index]) if index is not None and index < len(precision) else None,
            "recall": _float(recall[index]) if index is not None and index < len(recall) else None,
            "map50": _float(ap50[index]) if index is not None and index < len(ap50) else None,
            "map50_95": _float(ap[index]) if index is not None and index < len(ap) else None,
        })

    confusion = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    torch_model = getattr(model, "model", model)
    # Exported models have no PyTorch parameters to count.
    parameter_count = (
        sum(parameter.numel() for parameter in torch_model.parameters())
        if hasattr(torch_model, "parameters") else None
    )

    standard = {
        "precision": _float(getattr(box, "mp", result_dict.get("metrics/precision(B)", 0.0))),
        "recall": _float(getattr(box, "mr", result_dict.get("metrics/recall(B)", 0.0))),
        "map50": _float(getattr(box, "map50", result_dict.get("metrics/mAP50(B)", 0.0))),
        "map50_95": _float(getattr(box, "map", result_dict.get("metrics/mAP50-95(B)", 0.0))),
    }
    ultralytics_version = importlib.metadata.version("ultralytics")
    evaluation_context = {
        "dataset_fingerprint": dataset_info["fingerprint"],
        "split": split,
        "evaluation_args": evaluation_args,
        "ultralytics_version": ultralytics_version,
    }
    evaluation_context_id = hashlib.sha256(
        json.dumps(evaluation_context, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]

    return {
        "schema_version": SCHEMA_VERSION,
        "result_type": "final_test" if split == "test" else "model_selection_validation",
        "split": split,
        "dataset_fingerprint": dataset_info["fingerprint"],
        "dataset_split_counts": dataset_info["split_counts"],
        "evaluation_context_id": evaluation_context_id,
        "evaluation_context": evaluation_context,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        "model": {
            "parameter_count": parameter_count,
            "size_bytes": checkpoint.stat().st_size,
        },
        "metrics": standard,
        "per_class": per_class,
        "confusion_matrix": _tolist(confusion),
        "confusion_matrix_labels": [names[key] for key in sorted(names)] + ["background"],
        "speed_ms_per_image": {
            str(key): _float(value) for key, value in (getattr(metrics, "speed", {}) or {}).items()
        },
        "evaluation_args": evaluation_args,
        "runtime": runtime,
        "ultralytics_metrics": {
            str(key): _float(value) for key, value in result_dict.items()
            if isinstance(value, (int, float)) or hasattr(value, "item")
        },
    }
