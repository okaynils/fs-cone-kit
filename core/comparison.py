"""Read stable experiment artifacts and produce flat comparison rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


FIELDS = [
    "experiment", "architecture", "weights", "seed", "train_imgsz", "train_batch", "epochs",
    "git_commit", "ultralytics_version", "result_type", "evaluation_split", "dataset_fingerprint",
    "evaluation_context_id", "evaluation_device", "evaluation_batch", "evaluation_imgsz", "accuracy_comparable",
    "map50", "map50_95", "precision", "recall", "parameter_count", "model_size_bytes",
    "benchmark_context_id", "benchmark_system", "benchmark_machine", "benchmark_cpu",
    "benchmark_accelerator", "benchmark_device", "benchmark_precision", "benchmark_batch",
    "benchmark_input_size", "median_latency_ms", "p95_latency_ms", "throughput_images_per_second",
]


def _load(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _row(experiment: Path, evaluation: dict, benchmark: dict | None) -> dict[str, Any]:
    record = experiment / "experiment"
    config = _load(record / "config.yaml")
    metadata = _load(record / "metadata.json")
    metrics = evaluation.get("metrics", {})
    model = evaluation.get("model", {})
    protocol = (benchmark or {}).get("protocol", {})
    hardware = (benchmark or {}).get("hardware", {})
    timings = (benchmark or {}).get("results", {})
    evaluation_args = evaluation.get("evaluation_args", {})
    return {
        "experiment": experiment.name,
        "architecture": config.get("model", {}).get("name"),
        "weights": config.get("model", {}).get("weights"),
        "seed": config.get("seed"),
        "train_imgsz": config.get("trainer", {}).get("args", {}).get("imgsz"),
        "train_batch": config.get("trainer", {}).get("args", {}).get("batch"),
        "epochs": config.get("trainer", {}).get("args", {}).get("epochs"),
        "git_commit": metadata.get("git", {}).get("commit"),
        "ultralytics_version": metadata.get("dependencies", {}).get("ultralytics"),
        "result_type": evaluation.get("result_type"),
        "evaluation_split": evaluation.get("split"),
        "dataset_fingerprint": evaluation.get("dataset_fingerprint"),
        "evaluation_context_id": evaluation.get("evaluation_context_id"),
        "evaluation_device": evaluation_args.get("device"),
        "evaluation_batch": evaluation_args.get("batch"),
        "evaluation_imgsz": evaluation_args.get("imgsz"),
        "map50": metrics.get("map50"),
        "map50_95": metrics.get("map50_95"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "parameter_count": model.get("parameter_count"),
        "model_size_bytes": model.get("size_bytes"),
        "benchmark_context_id": (benchmark or {}).get("benchmark_context_id"),
        "benchmark_system": hardware.get("system"),
        "benchmark_machine": hardware.get("machine"),
        "benchmark_cpu": hardware.get("cpu"),
        "benchmark_accelerator": hardware.get("accelerator"),
        "benchmark_device": protocol.get("device"),
        "benchmark_precision": protocol.get("precision"),
        "benchmark_batch": protocol.get("batch_size"),
        "benchmark_input_size": protocol.get("input_size"),
        "median_latency_ms": timings.get("median_latency_ms_per_image"),
        "p95_latency_ms": (
            timings.get("p95_batch_latency_ms") / protocol["batch_size"]
            if timings.get("p95_batch_latency_ms") is not None and protocol.get("batch_size") else None
        ),
        "throughput_images_per_second": timings.get("throughput_images_per_second"),
    }


def collect_comparison_rows(experiments: list[Path], split: str = "test") -> list[dict[str, Any]]:
    rows = []
    for experiment in experiments:
        experiment = experiment.resolve()
        evaluation_path = experiment / "experiment" / "evaluations" / f"{split}.json"
        if not evaluation_path.exists():
            raise FileNotFoundError(f"Missing {split} evaluation: {evaluation_path}")
        evaluation = _load(evaluation_path)
        benchmarks = sorted((experiment / "experiment" / "benchmarks").glob("*.json"))
        if benchmarks:
            rows.extend(_row(experiment, evaluation, _load(path)) for path in benchmarks)
        else:
            rows.append(_row(experiment, evaluation, None))

    accuracy_keys = {
        row["evaluation_context_id"] or (
            row["dataset_fingerprint"], row["evaluation_split"], row["ultralytics_version"],
            row["evaluation_device"], row["evaluation_batch"], row["evaluation_imgsz"],
        )
        for row in rows
    }
    comparable = len(accuracy_keys) == 1
    for row in rows:
        row["accuracy_comparable"] = comparable
    return rows


def write_comparison(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in FIELDS} for row in rows)

    markdown_path = output_dir / "comparison.md"
    header = "| " + " | ".join(FIELDS) + " |"
    separator = "| " + " | ".join("---" for _ in FIELDS) + " |"
    body = [
        "| " + " | ".join(str(row.get(field, "") if row.get(field) is not None else "") for field in FIELDS) + " |"
        for row in rows
    ]
    markdown_path.write_text("\n".join([header, separator, *body]) + "\n", encoding="utf-8")
    return csv_path, markdown_path
