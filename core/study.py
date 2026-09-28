"""Prepare, evaluate, and report the small-cone augmentation study."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.studies import evaluate_study, prepare_study, write_study_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build fixed slices and audit leakage")
    prepare.add_argument("experiment_dir", type=Path)
    prepare.add_argument("--output-dir", type=Path, default=Path("study/small-cones"))
    prepare.add_argument("--area-threshold", type=float, default=0.005)
    prepare.add_argument("--near-duplicate-distance", type=int, default=4)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a recorded checkpoint on every slice")
    evaluate.add_argument("experiment_dir", type=Path)
    evaluate.add_argument("--study-dir", type=Path, default=Path("study/small-cones"))
    evaluate.add_argument("--checkpoint", choices=("best", "last"), default="best")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch", type=int, default=1)
    evaluate.add_argument("--imgsz", type=int)

    report = subparsers.add_parser("report", help="compare saved results and render failures")
    report.add_argument("baseline_result", type=Path)
    report.add_argument("intervention_result", type=Path)
    report.add_argument("--study-dir", type=Path, default=Path("study/small-cones"))
    report.add_argument("--output-dir", type=Path, default=Path("study/small-cones/report"))

    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare_study(
            args.experiment_dir,
            args.output_dir,
            area_threshold=args.area_threshold,
            near_duplicate_distance=args.near_duplicate_distance,
        )
        print(f"Study manifest written to {path}")
    elif args.command == "evaluate":
        evaluation_args = {"device": args.device, "batch": args.batch}
        if args.imgsz is not None:
            evaluation_args["imgsz"] = args.imgsz
        result, predictions = evaluate_study(
            args.experiment_dir,
            args.study_dir,
            checkpoint=args.checkpoint,
            **evaluation_args,
        )
        print(f"Study results written to {result}")
        print(f"Predictions written to {predictions}")
    else:
        report_path, csv_path = write_study_report(
            args.study_dir,
            args.baseline_result,
            args.intervention_result,
            args.output_dir,
        )
        print(f"Report written to {report_path}")
        print(f"Result table written to {csv_path}")


if __name__ == "__main__":
    main()
