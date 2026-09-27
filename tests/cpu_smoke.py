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
from core.train import run


def _write_dataset(root: Path) -> None:
    names = {
        0: "blue_cone",
        1: "yellow_cone",
        2: "orange_cone",
        3: "large_orange_cone",
        4: "unknown_cone",
    }
    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(count):
            image = np.full((64, 64, 3), 127, dtype=np.uint8)
            cv2.rectangle(image, (20, 20), (44, 52), (255, 0, 0), -1)
            stem = f"{split}-{index}"
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), image)
            (root / "labels" / split / f"{stem}.txt").write_text(
                "0 0.5 0.5625 0.375 0.5\n", encoding="utf-8"
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
            "trainer.export_onnx=false",
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
        print("CPU pipeline smoke test passed")


if __name__ == "__main__":
    main()
