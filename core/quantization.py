"""FP16 and INT8 variants of an exported ONNX model, and the accuracy each one costs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2

from core.deployment import preprocess, select_parity_images
from core.evaluation import sha256_file
from core.experiments import SCHEMA_VERSION


STANDARD_METRICS = ("map50", "map50_95", "precision", "recall")


def quantized_path(fp32_path: Path, precision: str) -> Path:
    if precision == "fp32":
        return fp32_path
    return fp32_path.with_name(f"{fp32_path.stem}_{precision}{fp32_path.suffix}")


def convert_fp16(source: Path, destination: Path) -> Path:
    """Halve weights and activations. Inputs and outputs stay float32, so the contract is unchanged."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = convert_float_to_float16(onnx.load(str(source)), keep_io_types=True)
    onnx.save(model, str(destination))
    return destination


def select_calibration_images(dataset_info: dict[str, Any], count: int) -> list[str]:
    """Calibrate on train images only. Val and test must never shape the quantized model."""
    splits = dataset_info["manifest"].get("splits", {})
    if not splits.get("train"):
        raise ValueError("INT8 calibration needs a non-empty train split")
    selected = select_parity_images(splits["train"], count)
    held_out = {
        member["image"] if isinstance(member, dict) else str(member)
        for split in ("val", "test")
        for member in splits.get(split, [])
    }
    leaked = sorted(set(selected) & held_out)
    if leaked:
        raise RuntimeError(f"Calibration images also appear in val or test: {leaked[:5]}")
    return selected


def calibration_record(
    images: list[str],
    dataset_info: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "train",
        "dataset_fingerprint": dataset_info["fingerprint"],
        "image_count": len(images),
        "images_sha256": hashlib.sha256("\n".join(images).encode()).hexdigest(),
        "images": images,
        "settings": settings,
    }


def quantize_int8(
    source: Path,
    destination: Path,
    dataset_root: Path,
    images: list[str],
    settings: dict[str, Any],
) -> Path:
    """Static QDQ quantization calibrated on the given train images.

    Only the listed op types are quantized. YOLO's decode head concatenates
    pixel box coordinates with 0-1 class scores; one INT8 scale for both wipes
    out the scores, so the head stays float.
    """
    import onnxruntime
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    model_input = onnxruntime.InferenceSession(
        str(source), providers=["CPUExecutionProvider"]
    ).get_inputs()[0]
    input_size = int(model_input.shape[2])

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.remaining = iter(images)

        def get_next(self):
            name = next(self.remaining, None)
            if name is None:
                return None
            image = cv2.imread(str(dataset_root / name))
            if image is None:
                raise FileNotFoundError(f"Could not read calibration image {dataset_root / name}")
            return {model_input.name: preprocess(image, input_size)}

    quantize_static(
        str(source),
        str(destination),
        Reader(),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=list(settings["op_types"]),
        per_channel=bool(settings["per_channel"]),
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=getattr(CalibrationMethod, settings["calibrate_method"]),
    )
    return destination


def _delta(candidate: Any, baseline: Any) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def _per_class(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["class_name"]: item for item in evaluation.get("per_class", [])}


def _row(
    label: str,
    model_path: Path,
    evaluation: dict[str, Any],
    baseline: dict[str, Any],
    slices: dict[str, Any] | None,
    baseline_slices: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    comparable = evaluation["evaluation_context_id"] == baseline["evaluation_context_id"]
    metrics = evaluation["metrics"]
    row = {
        "label": label,
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "size_bytes": model_path.stat().st_size,
        },
        "runtime": evaluation.get("runtime"),
        "dataset_fingerprint": evaluation["dataset_fingerprint"],
        "evaluation_context_id": evaluation["evaluation_context_id"],
        "accuracy_comparable": comparable,
        "metrics": metrics,
        "delta": {
            name: _delta(metrics.get(name), baseline["metrics"].get(name)) if comparable else None
            for name in STANDARD_METRICS
        },
        "per_class_delta": {
            name: {
                metric: _delta(item.get(metric), _per_class(baseline).get(name, {}).get(metric))
                if comparable else None
                for metric in ("map50_95", "recall")
            }
            for name, item in _per_class(evaluation).items()
        },
        "slices": {},
        "calibration": calibration,
    }
    for slice_name, result in (slices or {}).items():
        base = (baseline_slices or {}).get(slice_name, {})
        if result.get("status") == "empty":
            row["slices"][slice_name] = {"status": "empty"}
            continue
        slice_comparable = comparable and result.get("evaluation_context_id") == base.get("evaluation_context_id")
        row["slices"][slice_name] = {
            "image_count": result.get("image_count"),
            "cone_count": result.get("cone_count"),
            "accuracy_comparable": slice_comparable,
            "metrics": result["metrics"],
            "delta": {
                name: _delta(result["metrics"].get(name), base.get("metrics", {}).get(name))
                if slice_comparable else None
                for name in STANDARD_METRICS
            },
        }
    return row


def build_quantization_report(
    experiment_dir: Path,
    rows: list[tuple[str, Path, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]],
    study_note: str | None,
) -> dict[str, Any]:
    """`rows` holds (precision, model path, test evaluation, slice results, calibration); fp32 first."""
    if not rows or rows[0][0] != "fp32":
        raise ValueError("The fp32 export must be the first row; it is the baseline")
    _, _, baseline, baseline_slices, _ = rows[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "result_type": "quantization_accuracy",
        "experiment": experiment_dir.name,
        "baseline": "fp32 ONNX export of the same checkpoint, same evaluation settings",
        "split": "test",
        "dataset_fingerprint": baseline["dataset_fingerprint"],
        "study": study_note,
        "rows": [
            _row(label, path, evaluation, baseline, slices, baseline_slices, calibration)
            for label, path, evaluation, slices, calibration in rows
        ],
    }


def _format(value: Any, signed: bool = False) -> str:
    if value is None:
        return ""
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def quantization_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# quantization: {report['experiment']}",
        "",
        f"Baseline: {report['baseline']}.",
        f"Dataset fingerprint: `{report['dataset_fingerprint']}`.",
        "",
        "| precision | slice | size MB | mAP50-95 | delta | recall | delta | precision | delta | comparable |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        entries = [("test", row["metrics"], row["delta"], row["accuracy_comparable"])]
        entries += [
            (name, item["metrics"], item["delta"], item["accuracy_comparable"])
            for name, item in row["slices"].items() if item.get("status") != "empty"
        ]
        for slice_name, metrics, delta, comparable in entries:
            lines.append(
                f"| {row['label']} | {slice_name} | {row['model']['size_bytes'] / 1e6:.1f} "
                f"| {_format(metrics.get('map50_95'))} | {_format(delta.get('map50_95'), True)} "
                f"| {_format(metrics.get('recall'))} | {_format(delta.get('recall'), True)} "
                f"| {_format(metrics.get('precision'))} | {_format(delta.get('precision'), True)} "
                f"| {'yes' if comparable else 'no'} |"
            )
    if report.get("study"):
        lines += ["", report["study"]]
    for row in report["rows"]:
        if row["calibration"]:
            calibration = row["calibration"]
            lines += [
                "",
                f"{row['label']} calibration: {calibration['image_count']} train images, "
                f"list sha256 `{calibration['images_sha256'][:12]}`.",
            ]
    return "\n".join(lines) + "\n"


def load_study_manifest(study_dir: Path, dataset_fingerprint: str) -> tuple[dict[str, Any] | None, str]:
    """Use the study slices only when they were cut from this exact split."""
    manifest_path = study_dir / "manifest.json"
    if not manifest_path.exists():
        return None, f"No study manifest at {manifest_path}; slices skipped."
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_fingerprint") != dataset_fingerprint:
        return None, f"Study manifest at {manifest_path} uses a different split; slices skipped."
    if manifest.get("leakage", {}).get("status") != "pass":
        return None, f"Study manifest at {manifest_path} failed its leakage audit; slices skipped."
    return manifest, f"Slices from {manifest_path}. Box size is a proxy for distance."
