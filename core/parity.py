"""Check that an exported ONNX model reproduces the checkpoint it came from."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from core.deployment import (
    OnnxDetector,
    check_tolerances,
    compare_detections,
    onnx_precision,
    reference_detections,
    select_parity_images,
)
from core.evaluate import load_recorded_dataset, resolve_model_path
from core.evaluation import sha256_file
from core.experiments import SCHEMA_VERSION, write_json


RELEASE_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "release"


def load_release_settings(name: str = "default", overrides: list[str] | None = None) -> dict[str, Any]:
    """Compose configs/release/<name>.yaml with Hydra command-line overrides."""
    with initialize_config_dir(version_base=None, config_dir=str(RELEASE_CONFIG_DIR)):
        cfg = compose(config_name=name, overrides=list(overrides or []))
    return OmegaConf.to_container(cfg, resolve=True)


def default_onnx_path(experiment_dir: Path, checkpoint: str = "best") -> Path:
    return resolve_model_path(experiment_dir, checkpoint).with_suffix(".onnx")


def run_parity(
    experiment_dir: Path,
    model_path: Path,
    settings: dict[str, Any],
    checkpoint: str = "best",
    device: str = "cpu",
) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    _, dataset_yaml, dataset_info = load_recorded_dataset(experiment_dir, "test")
    dataset_root = Path(dataset_yaml).parent
    checkpoint_path = resolve_model_path(experiment_dir, checkpoint)
    model_path = model_path.resolve()
    precision = onnx_precision(model_path)
    if precision not in settings["tolerances"]:
        raise ValueError(f"No parity tolerances configured for {precision}")

    names = select_parity_images(dataset_info["manifest"]["splits"]["test"], settings["image_count"])
    if not names:
        raise ValueError("The test split is empty; parity needs test images")
    images = {}
    for name in names:
        image = cv2.imread(str(dataset_root / name))
        if image is None:
            raise FileNotFoundError(f"Could not read parity image {dataset_root / name}")
        images[name] = image

    run_confidence = max(0.0, settings["confidence"] - settings["confidence_margin"])
    detector = OnnxDetector(
        model_path, confidence=run_confidence, iou=settings["nms_iou"], max_det=settings["max_det"]
    )
    candidate = {name: detector.predict(image) for name, image in images.items()}
    reference = reference_detections(
        checkpoint_path,
        images,
        input_size=detector.input_size,
        confidence=run_confidence,
        iou=settings["nms_iou"],
        max_det=settings["max_det"],
        device=device,
    )
    summary = compare_detections(
        reference, candidate, settings["confidence"], settings["match_iou"]
    )
    checks = check_tolerances(summary, settings["tolerances"][precision])
    return {
        "schema_version": SCHEMA_VERSION,
        "result_type": "export_parity",
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "dataset_fingerprint": dataset_info["fingerprint"],
        "split": "test",
        "images": names,
        "reference": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "runtime": "ultralytics-pytorch",
            "device": device,
        },
        "candidate": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "runtime": "onnxruntime",
            "providers": detector.providers,
            "precision": precision,
            "input_size": detector.input_size,
            "preprocessing": "core.deployment contract",
        },
        "settings": {key: value for key, value in settings.items() if key != "tolerances"},
        "tolerances": settings["tolerances"][precision],
        "summary": summary,
        "checks": checks,
    }


def write_parity(experiment_dir: Path, model_path: Path, report: dict[str, Any]) -> Path:
    destination = experiment_dir.resolve() / "experiment" / "parity" / f"{model_path.stem}.json"
    write_json(destination, report)
    return destination


def print_checks(title: str, checks: list[dict[str, Any]]) -> None:
    print(title)
    for check in checks:
        mark = "pass" if check["passed"] else "FAIL"
        print(f"  {mark}  {check['name']}: {check['value']:.4g} (limit {check['limit']:.4g})")
        if check.get("note"):
            print(f"        {check['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--model", type=Path, help="exported .onnx file (default: <checkpoint>.onnx)")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--device", default="cpu", help="device for the PyTorch reference")
    parser.add_argument("--config", default="default", help="name in configs/release/")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. parity.image_count=64")
    args = parser.parse_intermixed_args()

    experiment_dir = args.experiment_dir.resolve()
    model_path = (
        resolve_model_path(experiment_dir, model=args.model)
        if args.model else default_onnx_path(experiment_dir, args.checkpoint)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"No exported model at {model_path}. Export it first.")
    settings = load_release_settings(args.config, args.overrides)["parity"]
    report = run_parity(experiment_dir, model_path, settings, args.checkpoint, args.device)
    destination = write_parity(experiment_dir, model_path, report)
    reference_name = Path(report["reference"]["path"]).name
    print_checks(f"Parity {report['status']}: {model_path.name} vs {reference_name}", report["checks"])
    print(f"Parity report written to {destination}")
    if report["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
