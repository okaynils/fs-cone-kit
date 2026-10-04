"""Build a verified release bundle for one target and precision. Exits non-zero if any check fails."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml
from omegaconf import OmegaConf

from core.benchmarking import context_id, hardware_record, time_pipeline
from core.deployment import (
    CLASS_OFFSET_PX,
    PAD_VALUE,
    OnnxDetector,
    select_parity_images,
    target_providers,
)
from core.driving_metrics import evaluate_gates
from core.evaluate import evaluation_filename, load_recorded_dataset
from core.evaluation import sha256_file
from core.experiments import SCHEMA_VERSION, write_json
from core.gates import run_gates
from core.parity import load_release_settings, print_checks, run_parity
from core.quantize import quantize_experiment


def preprocessing_contract(
    detector: OnnxDetector,
    class_map: dict[str, int],
    class_colors: dict[str, list[int]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Everything a team needs to feed the model and read its output, in one file."""
    output = detector.session.get_outputs()[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_implementation": "core/deployment.py (preprocess, postprocess)",
        "input": {
            "name": detector.input_name,
            "shape": [1, 3, detector.input_size, detector.input_size],
            "dtype": "float16" if detector.input_dtype.__name__ == "float16" else "float32",
            "layout": "NCHW",
            "color_order": "RGB",
            "source_color_order": "BGR (OpenCV); reverse the channels",
            "scale": "divide uint8 pixels by 255 to get [0, 1]",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
        "letterbox": {
            "size": detector.input_size,
            "resize": "gain = min(size / height, size / width); new = (round(width * gain), round(height * gain))",
            "interpolation": "bilinear (cv2.INTER_LINEAR); skipped when the size is unchanged",
            "pad_value": PAD_VALUE,
            "pad": "centred; top = round(dh - 0.1), bottom = round(dh + 0.1) with dh = (size - new_height) / 2; "
                   "same for left/right",
            "scale_up": True,
        },
        "output": {
            "name": output.name,
            "shape": output.shape,
            "format": "[1, 4 + classes, anchors]: cx, cy, w, h in input pixels, then one score per class "
                      "(already sigmoid). NMS is not in the graph.",
        },
        "postprocess": {
            "confidence": settings["confidence"],
            "confidence_rule": "keep anchors whose best class score is strictly greater than confidence",
            "class": "argmax over class scores",
            "nms": "class-aware greedy NMS; boxes shifted by class_id * "
                   f"{CLASS_OFFSET_PX} px before NMS so classes never suppress each other",
            "nms_iou": settings["nms_iou"],
            "max_detections": settings["max_det"],
            "to_image_pixels": "x = (x - pad_x) / gain, y = (y - pad_y) / gain with "
                               "pad = round((size - side * gain) / 2 - 0.1); clip to the image",
        },
        "class_map": {str(class_id): name for name, class_id in sorted(class_map.items(), key=lambda item: item[1])},
        "class_colors_rgb": dict(class_colors),
    }


def benchmark_record(
    detector: OnnxDetector,
    model_path: Path,
    images: list[Any],
    target: str,
    device: str,
    precision: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    timing = time_pipeline(detector, images, settings["warmup_runs"], settings["measured_runs"])
    accelerator = None
    if device == "cuda":
        import torch

        accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
    elif device == "coreml":
        accelerator = "Apple CoreML"
    runtime = {
        "name": "onnxruntime",
        "version": importlib.metadata.version("onnxruntime"),
        "format": "onnx",
        "precision": precision,
        "target": target,
        "providers": detector.providers,
    }
    hardware = hardware_record(accelerator)
    protocol = {
        "method": "contract_pipeline_real_images",
        "device": device,
        "precision": precision,
        "batch_size": 1,
        "input_size": detector.input_size,
        "image_count": len(images),
        "warmup_runs": settings["warmup_runs"],
        "measured_runs": settings["measured_runs"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "result_type": "deployment_pipeline_benchmark",
        "deployment_claim": False,
        "note": (
            "Preprocessing, inference, and postprocessing of one real test image at a time on this "
            "machine. Camera capture, transport, and the rest of the stack are not included."
        ),
        "checkpoint": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "size_bytes": model_path.stat().st_size,
        },
        "model": {"parameter_count": None},
        "runtime": runtime,
        "hardware": hardware,
        "protocol": protocol,
        "benchmark_context_id": context_id({"hardware": hardware, "protocol": protocol, "runtime": runtime}),
        # The end-to-end numbers keep the keys core.compare reads.
        "results": timing["total"],
        "stages": timing,
    }


def experiment_record(experiment_dir: Path, dataset_info: dict[str, Any]) -> dict[str, Any]:
    record = experiment_dir / "experiment"
    config = yaml.safe_load((record / "config.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((record / "metadata.json").read_text(encoding="utf-8"))
    return {
        "experiment": experiment_dir.name,
        "run_name": config.get("run_name"),
        "architecture": config.get("model", {}).get("name"),
        "initial_weights": config.get("model", {}).get("weights"),
        "epochs": config.get("trainer", {}).get("args", {}).get("epochs"),
        "imgsz": config.get("trainer", {}).get("args", {}).get("imgsz"),
        "seed": config.get("seed"),
        "git_commit": metadata.get("git", {}).get("commit"),
        "git_dirty": metadata.get("git", {}).get("dirty"),
        "dataset_type": dataset_info["dataset_type"],
        "dataset_fingerprint": dataset_info["fingerprint"],
        "split_counts": dataset_info["split_counts"],
    }


def _number(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _check_rows(checks: list[dict[str, Any]]) -> list[str]:
    return [
        f"| {check['name']} | {_number(check['value'], 4)} | {_number(check['limit'], 4)} "
        f"| {'pass' if check['passed'] else '**FAIL**'} |"
        for check in checks
    ]


def render_model_card(bundle: dict[str, Any]) -> str:
    """Every line comes from a file in the bundle. Nothing here is typed by hand."""
    experiment, manifest = bundle["experiment"], bundle["manifest"]
    evaluation, parity, gates = bundle["evaluation"], bundle["parity"], bundle["gates"]
    benchmark, contract, quantization = bundle["benchmark"], bundle["contract"], bundle["quantization"]
    row = quantization["rows"][-1]
    metrics = evaluation["metrics"]
    driving = gates["metrics"]
    status = manifest["status"].upper()
    lines = [
        f"# {experiment['experiment']}: {manifest['target']} {manifest['precision']}",
        "",
        f"**Status: {status}**" + ("" if status == "PASS" else " — do not put this on the car."),
    ]
    if manifest["failed_checks"]:
        lines += ["", "Failed checks: " + ", ".join(f"`{name}`" for name in manifest["failed_checks"]) + "."]
    lines += [
        "",
        "## model",
        "",
        f"- file: `model.onnx`, {row['model']['size_bytes'] / 1e6:.1f} MB, sha256 `{row['model']['sha256'][:16]}`",
        f"- input: {contract['input']['shape']} {contract['input']['dtype']}, "
        f"{contract['input']['color_order']}, letterboxed with {contract['letterbox']['pad_value']}",
        "- classes: " + ", ".join(f"{key} {value}" for key, value in contract["class_map"].items()),
        f"- trained: {experiment['architecture']} from `{experiment['initial_weights']}`, "
        f"{experiment['epochs']} epochs at {experiment['imgsz']} px, seed {experiment['seed']}",
        f"- code: `{experiment['git_commit']}`" + (" (dirty tree)" if experiment["git_dirty"] else ""),
        f"- dataset: `{experiment['dataset_type']}`, fingerprint `{experiment['dataset_fingerprint'][:16]}`, "
        + ", ".join(f"{name} {count}" for name, count in experiment["split_counts"].items()),
        "",
        f"## accuracy on the test split ({evaluation['dataset_split_counts'].get('test')} images)",
        "",
        f"Evaluated by {evaluation['runtime']['name']} {evaluation['runtime']['version']} "
        f"on {', '.join(evaluation['runtime'].get('providers') or [])}.",
        "",
        "| metric | value | vs fp32 export |",
        "| --- | --- | --- |",
    ]
    for name in ("map50", "map50_95", "precision", "recall"):
        lines.append(f"| {name} | {_number(metrics[name])} | {_number(row['delta'][name], signed=True)} |")
    lines += ["", "| class | precision | recall | mAP50-95 |", "| --- | --- | --- | --- |"]
    for item in evaluation["per_class"]:
        lines.append(
            f"| {item['class_name']} | {_number(item['precision'])} | {_number(item['recall'])} "
            f"| {_number(item['map50_95'])} |"
        )
    if row["slices"]:
        lines += [
            "",
            "| study slice | images | mAP50-95 | vs fp32 | recall | vs fp32 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, item in row["slices"].items():
            if item.get("status") == "empty":
                continue
            lines.append(
                f"| {name} | {item['image_count']} | {_number(item['metrics']['map50_95'])} "
                f"| {_number(item['delta']['map50_95'], signed=True)} | {_number(item['metrics']['recall'])} "
                f"| {_number(item['delta']['recall'], signed=True)} |"
            )
    if quantization.get("study"):
        lines += ["", quantization["study"]]
    if row["calibration"]:
        lines += [
            "",
            f"INT8 calibration used {row['calibration']['image_count']} images from the "
            f"{row['calibration']['split']} split (list sha256 `{row['calibration']['images_sha256'][:16]}`).",
        ]
    lines += [
        "",
        f"## at the deployment confidence ({driving['confidence']})",
        "",
        f"Predictions from `model.onnx` through the contract in `contract.json`. "
        f"{driving['truths']} cones in {driving['image_count']} test images.",
        "",
        "| band (box height / image height) | cones | recall | precision |",
        "| --- | --- | --- | --- |",
        f"| all | {driving['truths']} | {_number(driving['recall'])} | {_number(driving['precision'])} |",
    ]
    for name, item in driving["by_band"].items():
        low, high = driving["bands"][name]
        lines.append(
            f"| {name} ({low}–{high}) | {item['truths']} | {_number(item['recall'])} "
            f"| {_number(item['precision'])} |"
        )
    confusion = driving["color_confusion"]
    lines += [
        "",
        f"- false positives per image: {_number(driving['false_positives_per_image'])}",
        (
            f"- {' / '.join(confusion['classes'])} swapped: {confusion['swapped']} of {confusion['localized']} "
            f"localized cones ({_number(confusion['rate'], 4)})"
            if confusion["available"] else
            f"- colour confusion: not available; the class map has no {' / '.join(confusion['classes'])}"
        ),
        "",
        "## parity against the checkpoint",
        "",
        f"{parity['summary']['image_count']} test images, checkpoint through Ultralytics, `model.onnx` through "
        f"the contract on {', '.join(parity['candidate']['providers'])}.",
        "",
        f"- matched detections: {parity['summary']['matched_detections']} of "
        f"{parity['summary']['reference_detections']}",
        f"- box deviation: max {_number(parity['summary']['box_deviation_px']['max'], 3)} px, "
        f"mean {_number(parity['summary']['box_deviation_px']['mean'], 3)} px",
        f"- confidence difference: max {_number(parity['summary']['confidence_difference']['max'], 4)}",
        f"- class flips: {parity['summary']['class_disagreements']}, only in checkpoint: "
        f"{parity['summary']['only_in_reference']}, only in export: {parity['summary']['only_in_candidate']}",
        "",
        "## latency",
        "",
        f"{benchmark['hardware']['cpu']}"
        + (f", {benchmark['hardware']['accelerator']}" if benchmark["hardware"]["accelerator"] else "")
        + f"; {benchmark['runtime']['name']} {benchmark['runtime']['version']} on "
        f"{', '.join(benchmark['runtime']['providers'])}; context `{benchmark['benchmark_context_id']}`.",
        "",
        "| stage | median ms | p95 ms |",
        "| --- | --- | --- |",
    ]
    for name, item in benchmark["stages"].items():
        lines.append(
            f"| {name} | {_number(item['median_batch_latency_ms'], 2)} "
            f"| {_number(item['p95_batch_latency_ms'], 2)} |"
        )
    lines += [
        "",
        f"Throughput: {_number(benchmark['results']['throughput_images_per_second'], 1)} images/s, "
        "one image at a time.",
        "",
        "## checks",
        "",
        "| check | value | limit | result |",
        "| --- | --- | --- | --- |",
        *_check_rows(parity["checks"]),
        *_check_rows(manifest["quantization_checks"]),
        *_check_rows(gates["checks"]),
        "",
        "## limits of this card",
        "",
        f"- {benchmark['note']}",
        "- Box height is a stand-in for distance, not a measurement.",
        f"- Gate limits are {gates['limits_source']}.",
        "- mAP comes from Ultralytics validation of `model.onnx`; the gates and parity use the contract directly.",
        "",
    ]
    return "\n".join(lines)


def release_experiment(
    experiment_dir: Path,
    target: str,
    precision: str,
    settings: dict[str, Any],
    device: str = "cpu",
    checkpoint: str = "best",
    study_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    experiment_dir = experiment_dir.resolve()
    providers = target_providers(target, device, precision)
    cfg, dataset_yaml, dataset_info = load_recorded_dataset(experiment_dir, "test")
    evaluation_device = "0" if device == "cuda" else "cpu"

    quantization_path, _ = quantize_experiment(
        experiment_dir, [precision] if precision != "fp32" else [], settings["quantization"],
        device=evaluation_device, study_dir=study_dir, checkpoint=checkpoint,
    )
    quantization = json.loads(quantization_path.read_text(encoding="utf-8"))
    row = quantization["rows"][-1]
    model_path = Path(row["model"]["path"])
    evaluation = json.loads(
        (experiment_dir / "experiment" / "evaluations" / evaluation_filename("test", model_path)).read_text(
            encoding="utf-8"
        )
    )
    drops = {
        f"{name}_drop": None if row["delta"][name] is None else 0.0 - row["delta"][name]
        for name in ("map50_95", "recall")
    }
    quantization_checks = evaluate_gates(drops, settings["quantization"]["limits"])

    parity = run_parity(experiment_dir, model_path, settings["parity"], checkpoint, "cpu", providers)
    gates = run_gates(experiment_dir, model_path, settings["gates"], providers=providers)

    detector = OnnxDetector(
        model_path, providers=providers, confidence=settings["gates"]["confidence"],
        iou=settings["gates"]["nms_iou"], max_det=settings["gates"]["max_det"],
    )
    dataset_root = Path(dataset_yaml).parent
    names = select_parity_images(dataset_info["manifest"]["splits"]["test"], settings["benchmark"]["image_count"])
    images = [cv2.imread(str(dataset_root / name)) for name in names]
    benchmark = benchmark_record(detector, model_path, images, target, device, precision, settings["benchmark"])
    write_json(
        experiment_dir / "experiment" / "benchmarks"
        / f"{model_path.stem}_{target}_{device}_{benchmark['benchmark_context_id']}.json",
        benchmark,
    )
    dataset_config = OmegaConf.to_container(cfg.dataset, resolve=True)
    contract = preprocessing_contract(
        detector, dataset_config["class_map"], dataset_config.get("class_colors") or {}, settings["gates"]
    )

    records = {
        "experiment": experiment_record(experiment_dir, dataset_info),
        "contract": contract,
        "parity": parity,
        "evaluation": evaluation,
        "quantization": quantization,
        "gates": gates,
        "benchmark": benchmark,
    }
    if row["calibration"]:
        records["calibration"] = row["calibration"]
    checks = [*parity["checks"], *quantization_checks, *gates["checks"]]
    bundle_dir = experiment_dir / "release" / f"{target}-{precision}"
    manifest = write_bundle(bundle_dir, model_path, records, quantization_checks, checks, {
        "target": target,
        "precision": precision,
        "device": device,
        "dataset_fingerprint": dataset_info["fingerprint"],
    })
    return bundle_dir, {"status": manifest["status"], "checks": checks}


def write_bundle(
    bundle_dir: Path,
    model_path: Path,
    records: dict[str, Any],
    quantization_checks: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Replace the bundle directory with the model, every record, a model card, and a hashed manifest."""
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    shutil.copy2(model_path, bundle_dir / "model.onnx")
    for name, payload in records.items():
        write_json(bundle_dir / f"{name}.json", payload)

    failed = [check["name"] for check in checks if not check["passed"]]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "result_type": "release_bundle",
        "status": "fail" if failed else "pass",
        "failed_checks": failed,
        "quantization_checks": quantization_checks,
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": sha256_file(bundle_dir / "model.onnx"),
    }
    (bundle_dir / "model_card.md").write_text(
        render_model_card({**records, "manifest": manifest}), encoding="utf-8"
    )
    manifest["files"] = {
        path.name: sha256_file(path) for path in sorted(bundle_dir.iterdir()) if path.name != "manifest.json"
    }
    write_json(bundle_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--target", choices=("onnxruntime", "tensorrt"), default="onnxruntime")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--device", choices=("cpu", "cuda", "coreml"), default="cpu")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--study-dir", type=Path, default=Path("study/small-cones"))
    parser.add_argument("--config", default="default", help="name in configs/release/")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. gates.limits.min_recall_far=0.6")
    args = parser.parse_intermixed_args()

    settings = load_release_settings(args.config, args.overrides)
    bundle_dir, result = release_experiment(
        args.experiment_dir, args.target, args.precision, settings,
        device=args.device, checkpoint=args.checkpoint, study_dir=args.study_dir,
    )
    print_checks(f"Release {result['status']}: {args.target} {args.precision}", result["checks"])
    print(f"Release bundle written to {bundle_dir}")
    if result["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
