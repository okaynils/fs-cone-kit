"""Export FP16 and INT8 variants and measure what they cost on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra.utils import instantiate

from core.evaluate import evaluate_experiment, load_recorded_dataset, resolve_model_path
from core.experiments import write_json
from core.parity import load_release_settings
from core.quantization import (
    build_quantization_report,
    calibration_record,
    convert_fp16,
    load_study_manifest,
    quantization_markdown,
    quantize_int8,
    quantized_path,
    select_calibration_images,
)
from core.studies import evaluate_slices


def ensure_fp32_export(experiment_dir: Path, cfg: Any, checkpoint: str = "best") -> Path:
    checkpoint_path = resolve_model_path(experiment_dir, checkpoint)
    onnx_path = checkpoint_path.with_suffix(".onnx")
    if onnx_path.exists():
        return onnx_path
    return instantiate(cfg.trainer).export_checkpoint_to_onnx(checkpoint_path)


def export_variant(
    experiment_dir: Path,
    fp32_path: Path,
    precision: str,
    settings: dict[str, Any],
) -> tuple[Path, dict[str, Any] | None]:
    """Write one precision variant next to the fp32 export. Return its path and calibration record."""
    destination = quantized_path(fp32_path, precision)
    if precision == "fp32":
        return fp32_path, None
    if precision == "fp16":
        return convert_fp16(fp32_path, destination), None
    if precision == "int8":
        _, dataset_yaml, dataset_info = load_recorded_dataset(experiment_dir)
        images = select_calibration_images(dataset_info, settings["calibration_images"])
        quantize_int8(fp32_path, destination, Path(dataset_yaml).parent, images, settings["int8"])
        record = calibration_record(images, dataset_info, settings["int8"])
        write_json(destination.with_suffix(".calibration.json"), record)
        return destination, record
    raise ValueError(f"Unknown precision {precision!r}")


def quantize_experiment(
    experiment_dir: Path,
    precisions: list[str],
    settings: dict[str, Any],
    device: str = "cpu",
    study_dir: Path | None = None,
    checkpoint: str = "best",
) -> tuple[Path, Path]:
    experiment_dir = experiment_dir.resolve()
    cfg, _, dataset_info = load_recorded_dataset(experiment_dir, "test")
    fp32_path = ensure_fp32_export(experiment_dir, cfg, checkpoint)
    manifest, study_note = (
        load_study_manifest(study_dir.resolve(), dataset_info["fingerprint"]) if study_dir else (None, None)
    )
    evaluation_args = {"device": device, "batch": 1, "imgsz": int(cfg.trainer.args.imgsz)}
    trainer = instantiate(cfg.trainer)

    rows = []
    for precision in ["fp32", *[item for item in precisions if item != "fp32"]]:
        model_path, calibration = export_variant(experiment_dir, fp32_path, precision, settings)
        evaluation_path = evaluate_experiment(
            experiment_dir, split="test", model=model_path, **evaluation_args
        )
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        slices = None
        if manifest:
            slices = evaluate_slices(
                trainer, model_path, study_dir.resolve(), manifest, evaluation_args,
                output_root=experiment_dir / "evaluation_files" / "study" / model_path.stem,
            )
        rows.append((precision, model_path, evaluation, slices, calibration))

    report = build_quantization_report(experiment_dir, rows, study_note)
    destination = experiment_dir / "experiment" / "quantization" / f"{fp32_path.stem}.json"
    write_json(destination, report)
    markdown = destination.with_suffix(".md")
    markdown.write_text(quantization_markdown(report), encoding="utf-8")
    return destination, markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--precision", nargs="+", choices=("fp16", "int8"), default=["fp16", "int8"])
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--study-dir", type=Path, default=Path("study/small-cones"))
    parser.add_argument("--config", default="default", help="name in configs/release/")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. quantization.calibration_images=128")
    args = parser.parse_intermixed_args()

    settings = load_release_settings(args.config, args.overrides)["quantization"]
    report, markdown = quantize_experiment(
        args.experiment_dir, args.precision, settings, args.device, args.study_dir, args.checkpoint
    )
    print(markdown.read_text(encoding="utf-8"))
    print(f"Quantization report written to {report}")


if __name__ == "__main__":
    main()
