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


def resolve_model_path(experiment_dir: Path, checkpoint: str = "best", model: str | Path | None = None) -> Path:
    """Use an explicit model file when given, otherwise the recorded checkpoint."""
    if model is None:
        return _checkpoint_path(experiment_dir, checkpoint)
    model = Path(model)
    for candidate in (model, experiment_dir / model):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Model not found: {model} (also tried {experiment_dir / model})")


def load_recorded_dataset(experiment_dir: Path, split: str | None = None):
    """Prepare the experiment's dataset and refuse to continue if the split changed."""
    config_path = experiment_dir / "experiment" / "config.yaml"
    dataset_record_path = experiment_dir / "experiment" / "dataset.json"
    if not config_path.exists() or not dataset_record_path.exists():
        raise FileNotFoundError(f"{experiment_dir} is not a recorded experiment")

    cfg = OmegaConf.load(config_path)
    dataset = instantiate(cfg.dataset)
    dataset_yaml = dataset.prepare()
    current_dataset = dataset.get_dataset_info()
    recorded_dataset = json.loads(dataset_record_path.read_text(encoding="utf-8"))
    if current_dataset["fingerprint"] != recorded_dataset["fingerprint"]:
        raise RuntimeError(
            "The prepared dataset no longer matches this experiment's split manifest. "
            "Restore or regenerate the recorded split before evaluating."
        )
    if split is not None and split not in current_dataset["manifest"].get("splits", {}):
        raise ValueError(f"Dataset has no {split!r} split")
    return cfg, dataset_yaml, current_dataset


def evaluation_filename(split: str, model_path: Path) -> str:
    """Keep `<split>.json` for checkpoints, which is what core.compare reads."""
    if model_path.suffix == ".pt":
        return f"{split}.json"
    return f"{split}_{model_path.stem}_{model_path.suffix.lstrip('.')}.json"


def evaluate_experiment(
    experiment_dir: Path,
    split: str = "test",
    checkpoint: str = "best",
    model: str | Path | None = None,
    **evaluation_args,
) -> Path:
    experiment_dir = experiment_dir.resolve()
    cfg, dataset_yaml, current_dataset = load_recorded_dataset(experiment_dir, split)
    model_path = resolve_model_path(experiment_dir, checkpoint, model)
    evaluation_args = dict(evaluation_args)
    evaluation_args.setdefault("imgsz", int(cfg.trainer.args.imgsz))
    filename = evaluation_filename(split, model_path)

    trainer = instantiate(cfg.trainer)
    result = trainer.evaluate(
        model_path=model_path,
        data=dataset_yaml,
        split=split,
        output_dir=experiment_dir / "evaluation_files" / Path(filename).stem,
        dataset_info=current_dataset,
        evaluation_args=evaluation_args,
    )
    destination = experiment_dir / "experiment" / "evaluations" / filename
    write_json(destination, result)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--model", type=Path, help="exported model to evaluate instead of the checkpoint")
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
        model=args.model,
        **evaluation_args,
    )
    print(f"Evaluation written to {path}")


if __name__ == "__main__":
    main()
