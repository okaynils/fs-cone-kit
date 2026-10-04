"""Check a model against release gates on the test split. Exits non-zero on failure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2

from core.deployment import OnnxDetector, reference_detections
from core.driving_metrics import driving_metrics, evaluate_gates, flatten_metrics
from core.evaluate import load_recorded_dataset, resolve_model_path
from core.evaluation import sha256_file
from core.experiments import SCHEMA_VERSION, write_json
from core.parity import default_onnx_path, load_release_settings, print_checks
from core.studies import _label_for_member, _read_annotations


def load_ground_truth(dataset_root: Path, members: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read test images and their YOLO labels as pixel boxes."""
    images, truth = {}, {}
    for member in members:
        member = member if isinstance(member, dict) else {"image": str(member)}
        image = cv2.imread(str(dataset_root / member["image"]))
        if image is None:
            raise FileNotFoundError(f"Could not read test image {dataset_root / member['image']}")
        height, width = image.shape[:2]
        boxes = []
        for item in _read_annotations(_label_for_member(member, dataset_root)):
            x, y, w, h = item["xywh"]
            boxes.append({
                "class_id": item["class_id"],
                "xyxy": [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height],
            })
        images[member["image"]] = image
        truth[member["image"]] = {"height": height, "width": width, "boxes": boxes}
    return images, truth


def predict(
    model_path: Path,
    images: dict[str, Any],
    settings: dict[str, Any],
    input_size: int,
    device: str,
    providers: list[Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """ONNX goes through the deployment contract; a checkpoint goes through Ultralytics."""
    if model_path.suffix == ".onnx":
        detector = OnnxDetector(
            model_path,
            providers=providers,
            confidence=settings["confidence"],
            iou=settings["nms_iou"],
            max_det=settings["max_det"],
        )
        return {name: detector.predict(image) for name, image in images.items()}, "core.deployment contract"
    return reference_detections(
        model_path, images, input_size, settings["confidence"], settings["nms_iou"], settings["max_det"], device
    ), "ultralytics-pytorch"


def run_gates(
    experiment_dir: Path,
    model_path: Path,
    settings: dict[str, Any],
    device: str = "cpu",
    extra_values: dict[str, float | None] | None = None,
    providers: list[Any] | None = None,
) -> dict[str, Any]:
    """`extra_values` lets a caller gate on numbers measured elsewhere, such as an INT8 accuracy drop."""
    experiment_dir = experiment_dir.resolve()
    cfg, dataset_yaml, dataset_info = load_recorded_dataset(experiment_dir, "test")
    class_names = {int(value): str(key) for key, value in dict(cfg.dataset.class_map).items()}
    images, truth = load_ground_truth(Path(dataset_yaml).parent, dataset_info["manifest"]["splits"]["test"])
    predictions, pipeline = predict(
        model_path, images, settings, int(cfg.trainer.args.imgsz), device, providers
    )
    metrics = driving_metrics(
        truth, predictions, class_names, settings["confidence"], settings["match_iou"],
        settings["size_bands"], list(settings["color_classes"]),
    )
    values = {**flatten_metrics(metrics), **(extra_values or {})}
    checks = evaluate_gates(values, settings["limits"])
    return {
        "schema_version": SCHEMA_VERSION,
        "result_type": "release_gates",
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "limits_source": "team-chosen; not derived from FSG rules",
        "dataset_fingerprint": dataset_info["fingerprint"],
        "split": "test",
        "model": {"path": str(model_path), "sha256": sha256_file(model_path), "pipeline": pipeline},
        "settings": {key: value for key, value in settings.items() if key != "limits"},
        "limits": settings["limits"],
        "metrics": metrics,
        "values": values,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--model", type=Path, help="model to gate (default: <checkpoint>.onnx)")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--device", default="cpu", help="device when gating a .pt checkpoint")
    parser.add_argument("--config", default="default", help="name in configs/release/")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. gates.limits.min_recall_far=0.6")
    args = parser.parse_intermixed_args()

    experiment_dir = args.experiment_dir.resolve()
    model_path = (
        resolve_model_path(experiment_dir, model=args.model)
        if args.model else default_onnx_path(experiment_dir, args.checkpoint)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}. Export it first.")
    settings = load_release_settings(args.config, args.overrides)["gates"]
    report = run_gates(experiment_dir, model_path, settings, args.device)
    destination = experiment_dir / "experiment" / "gates" / f"{model_path.stem}.json"
    write_json(destination, report)
    print_checks(f"Gates {report['status']}: {model_path.name}", report["checks"])
    print(f"Gate report written to {destination}")
    if report["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
