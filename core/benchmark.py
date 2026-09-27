"""Benchmark inference for a completed experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.benchmarking import benchmark_ultralytics_model
from core.evaluate import _checkpoint_path
from core.experiments import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measured-runs", type=int, default=100)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    result = benchmark_ultralytics_model(
        checkpoint=_checkpoint_path(experiment_dir, args.checkpoint),
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        input_size=args.input_size,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )
    filename = (
        f"{args.device.replace(':', '-')}_{args.precision}_b{args.batch_size}_"
        f"{args.input_size}_{result['benchmark_context_id']}.json"
    )
    destination = experiment_dir / "experiment" / "benchmarks" / filename
    write_json(destination, result)
    print(f"Benchmark written to {destination}")


if __name__ == "__main__":
    main()
