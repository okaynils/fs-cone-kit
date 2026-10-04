"""Run the real pipeline on a generated three-split dataset without downloads."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from hydra import compose, initialize_config_dir

from core.benchmarking import benchmark_ultralytics_model
from core.comparison import collect_comparison_rows, write_comparison
from core.evaluate import _checkpoint_path, evaluate_experiment
from core.experiments import write_json
from core.parity import load_release_settings, run_parity
from core.quantize import quantize_experiment
from core.studies import evaluate_study, prepare_study
from core.train import run


def _write_dataset(root: Path) -> None:
    names = {
        0: "blue_cone",
        1: "yellow_cone",
        2: "orange_cone",
        3: "large_orange_cone",
        4: "unknown_cone",
    }
    for split_index, (split, count) in enumerate((("train", 2), ("val", 1), ("test", 2))):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(count):
            rng = np.random.default_rng(split_index * 10 + index)
            image = rng.integers(64, 192, size=(64, 64, 3), dtype=np.uint8)
            if split == "test" and index == 0:
                box = (0.5, 0.5, 0.05, 0.08)
                cv2.rectangle(image, (30, 29), (34, 35), (255, 0, 0), -1)
            else:
                box = (0.5, 0.5625, 0.375, 0.5)
                cv2.rectangle(image, (20, 20), (44, 52), (255, 0, 0), -1)
            stem = f"{split}-{index}"
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), image)
            (root / "labels" / split / f"{stem}.txt").write_text(
                f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n", encoding="utf-8"
            )
    (root / "dataset.yaml").write_text(
        yaml.safe_dump({
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names,
        }, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="fs-cone-kit-smoke-") as temporary:
        temporary_root = Path(temporary)
        dataset_root = temporary_root / "dataset"
        experiment = temporary_root / "experiment"
        _write_dataset(dataset_root)

        overrides = [
            "~loggers.wandb",
            "run_name=cpu-smoke",
            "model.name=yolo11n-smoke",
            "model.weights=yolo11n.yaml",
            f"dataset.preprocessed_dir={dataset_root}",
            f"dataset.raw_dir={temporary_root / 'raw'}",
            "trainer.args.epochs=1",
            "trainer.args.imgsz=32",
            "trainer.args.batch=2",
            "+trainer.args.workers=0",
            "+trainer.args.device=cpu",
            "+trainer.args.plots=false",
            "+trainer.args.amp=false",
        ]
        with initialize_config_dir(version_base=None, config_dir=str(project_root / "configs")):
            cfg = compose(config_name="config", overrides=overrides)
        run(cfg, experiment)

        evaluation_path = evaluate_experiment(
            experiment, split="test", device="cpu", batch=1, imgsz=32
        )
        benchmark = benchmark_ultralytics_model(
            checkpoint=_checkpoint_path(experiment, "best"),
            device="cpu",
            precision="fp32",
            batch_size=1,
            input_size=32,
            warmup_runs=1,
            measured_runs=3,
        )
        write_json(experiment / "experiment/benchmarks/cpu-smoke.json", benchmark)
        rows = collect_comparison_rows([experiment])
        csv_path, markdown_path = write_comparison(rows, temporary_root / "comparison")

        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        assert evaluation["result_type"] == "final_test"
        assert len(evaluation["per_class"]) == 5
        assert csv_path.exists() and markdown_path.exists()
        assert (experiment / "experiment/config.yaml").exists()
        assert (experiment / "ultralytics_files/weights/best.pt").exists()

        onnx_path = experiment / "ultralytics_files/weights/best.onnx"
        assert onnx_path.exists()
        # An untrained model scores everything near zero, so compare at a tiny threshold.
        parity_settings = load_release_settings(
            overrides=["parity.confidence=0.001", "parity.confidence_margin=0.0005"]
        )["parity"]
        parity = run_parity(experiment, onnx_path, parity_settings)
        assert parity["status"] == "pass", parity["checks"]
        assert parity["summary"]["reference_detections"] > 0
        onnx_evaluation = json.loads(evaluate_experiment(
            experiment, split="test", model=onnx_path, device="cpu", batch=1, imgsz=32
        ).read_text(encoding="utf-8"))
        assert onnx_evaluation["runtime"]["name"] == "onnxruntime"
        assert onnx_evaluation["dataset_fingerprint"] == evaluation["dataset_fingerprint"]

        study_dir = temporary_root / "study"
        manifest_path = prepare_study(experiment, study_dir)
        study_result, predictions = evaluate_study(
            experiment, study_dir, device="cpu", batch=1, imgsz=32
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["leakage"]["status"] == "pass"
        assert manifest["slices"]["small_cones"]["image_count"] == 1
        assert manifest["slices"]["ordinary"]["image_count"] == 1
        assert study_result.exists() and predictions.exists()

        quantization_path, _ = quantize_experiment(
            experiment, ["fp16", "int8"], load_release_settings()["quantization"], study_dir=study_dir
        )
        quantization = json.loads(quantization_path.read_text(encoding="utf-8"))
        assert [row["label"] for row in quantization["rows"]] == ["fp32", "fp16", "int8"]
        assert all(row["accuracy_comparable"] for row in quantization["rows"])
        assert quantization["rows"][2]["calibration"]["split"] == "train"
        assert set(quantization["rows"][2]["slices"]) == {"full", "small_cones", "ordinary"}

        print("CPU pipeline smoke test passed")


if __name__ == "__main__":
    main()
