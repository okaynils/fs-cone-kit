"""Compare completed experiment records without an external tracking service."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.comparison import collect_comparison_rows, write_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("comparison"))
    args = parser.parse_args()
    rows = collect_comparison_rows(args.experiments, split=args.split)
    csv_path, markdown_path = write_comparison(rows, args.output_dir)
    if not all(row["accuracy_comparable"] for row in rows):
        print("Warning: evaluation datasets or settings differ; accuracy rows are marked non-comparable.")
    contexts = {row["benchmark_context_id"] for row in rows if row["benchmark_context_id"]}
    if len(contexts) > 1:
        print("Warning: benchmark contexts differ; do not compare their timing values directly.")
    print(f"Comparison written to {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
