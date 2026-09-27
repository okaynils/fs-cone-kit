"""Evaluate a completed experiment on its recorded validation or test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.experiments import write_json


def _checkpoint_path(experiment_dir: Path, checkpoint: str) -> Path:
    training_path = experiment_dir / "experiment" / "training.json"
    if not training_path.exists():
        interrupted_path = experiment_dir / "ultralytics_files" / "weights" / f"{checkpoint}.pt"
        if interrupted_path.exists():
            return interrupted_path
        raise FileNotFoundError(
            f"Missing training record and checkpoint for interrupted run: {interrupted_path}"
        )
    training = json.loads(training_path.read_text(encoding="utf-8"))
    value = training.get(f"{checkpoint}_checkpoint")
    if not value:
        raise ValueError(f"Training record has no {checkpoint!r} checkpoint")
    path = Path(value)
    return path if path.is_absolute() else experiment_dir / path


def evaluate_experiment(
    experiment_dir: Path,
    split: str = "test",
    checkpoint: str = "best",
    **evaluation_args,
) -> Path:
    experiment_dir = experiment_dir.resolve()
    config_path = experiment_dir / "experiment" / "config.yaml"
    dataset_record_path = experiment_dir / "experiment" / "dataset.json"
    if not config_path.exists() or not dataset_record_path.exists():
        raise FileNotFoundError(f"{experiment_dir} is not a recorded experiment")

    cfg = OmegaConf.load(config_path)
    evaluation_args = dict(evaluation_args)
    evaluation_args.setdefault("imgsz", int(cfg.trainer.args.imgsz))
    dataset = instantiate(cfg.dataset)
    dataset_yaml = dataset.prepare()
    current_dataset = dataset.get_dataset_info()
    recorded_dataset = json.loads(dataset_record_path.read_text(encoding="utf-8"))
    if current_dataset["fingerprint"] != recorded_dataset["fingerprint"]:
        raise RuntimeError(
            "The prepared dataset no longer matches this experiment's split manifest. "
            "Restore or regenerate the recorded split before evaluating."
        )
    if split not in current_dataset["manifest"].get("splits", {}):
        raise ValueError(f"Dataset has no {split!r} split")

    trainer = instantiate(cfg.trainer)
    result = trainer.evaluate(
        model_path=_checkpoint_path(experiment_dir, checkpoint),
        data=dataset_yaml,
        split=split,
        output_dir=experiment_dir / "evaluation_files" / split,
        dataset_info=current_dataset,
        evaluation_args=evaluation_args,
    )
    destination = experiment_dir / "experiment" / "evaluations" / f"{split}.json"
    write_json(destination, result)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int)
    args = parser.parse_args()

    evaluation_args = {"device": args.device, "batch": args.batch}
    if args.imgsz is not None:
        evaluation_args["imgsz"] = args.imgsz
    path = evaluate_experiment(
        args.experiment_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        **evaluation_args,
    )
    print(f"Evaluation written to {path}")


if __name__ == "__main__":
    main()
