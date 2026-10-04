"""Inference timing helpers with an explicit, serializable benchmark protocol."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from core.deployment import runtime_record
from core.evaluation import sha256_file
from core.experiments import SCHEMA_VERSION


def summarize_latencies(latencies_seconds: list[float], batch_size: int) -> dict[str, float]:
    if not latencies_seconds:
        raise ValueError("At least one measured inference is required")
    ordered = sorted(latencies_seconds)
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95 + 0.999999) - 1))
    total_seconds = sum(latencies_seconds)
    return {
        "median_batch_latency_ms": statistics.median(ordered) * 1000,
        "p95_batch_latency_ms": ordered[p95_index] * 1000,
        "median_latency_ms_per_image": statistics.median(ordered) * 1000 / batch_size,
        "throughput_images_per_second": len(latencies_seconds) * batch_size / total_seconds,
    }


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def benchmark_ultralytics_model(
    checkpoint: Path,
    device: str,
    precision: str,
    batch_size: int,
    input_size: int,
    warmup_runs: int,
    measured_runs: int,
    model_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    if batch_size < 1 or input_size < 1 or warmup_runs < 0 or measured_runs < 1:
        raise ValueError("batch_size, input_size, and measured_runs must be positive")
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    checkpoint = checkpoint.resolve()
    exported = checkpoint.suffix == ".onnx"
    if exported and precision != "fp32":
        raise ValueError(
            "An ONNX file's precision is fixed at export. Benchmark the fp16 or int8 file with --precision fp32."
        )
    model = (model_factory or YOLO)(str(checkpoint))
    tensor_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if precision == "fp16" and tensor_device.type not in {"cuda", "mps"}:
        raise ValueError("fp16 benchmarking requires a CUDA or MPS device")
    inputs = torch.zeros((batch_size, 3, input_size, input_size), device=tensor_device)
    half = precision == "fp16"
    if half:
        inputs = inputs.half()

    def synchronize() -> None:
        if tensor_device.type == "cuda":
            torch.cuda.synchronize(tensor_device)
        elif tensor_device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    predict_args = {"device": device, "half": half, "verbose": False}
    for _ in range(warmup_runs):
        model.predict(source=inputs, **predict_args)
    synchronize()

    latencies = []
    for _ in range(measured_runs):
        synchronize()
        started = time.perf_counter()
        model.predict(source=inputs, **predict_args)
        synchronize()
        latencies.append(time.perf_counter() - started)

    torch_model = getattr(model, "model", model)
    parameter_count = (
        sum(parameter.numel() for parameter in torch_model.parameters())
        if hasattr(torch_model, "parameters") else None
    )
    runtime = runtime_record(checkpoint, device, half=half)
    hardware = {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "logical_cpu_count": os.cpu_count(),
        "accelerator": (
            torch.cuda.get_device_name(tensor_device)
            if tensor_device.type == "cuda" else "Apple Metal" if tensor_device.type == "mps" else None
        ),
        "torch_version": torch.__version__,
    }
    protocol = {
        "method": "ultralytics_predict_synthetic_tensor",
        "device": device,
        "precision": precision,
        "batch_size": batch_size,
        "input_size": input_size,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
    }
    context = {"hardware": hardware, "protocol": protocol}
    if exported:
        # PyTorch context IDs stay as they were; exported runtimes get their own.
        context["runtime"] = runtime
    context_id = hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]

    return {
        "schema_version": SCHEMA_VERSION,
        "result_type": "synthetic_model_inference_benchmark",
        "deployment_claim": False,
        "note": (
            "This measures repeated model inference on synthetic tensors. It is not an end-to-end "
            "vehicle deployment benchmark."
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        "model": {"parameter_count": parameter_count},
        "runtime": runtime,
        "hardware": hardware,
        "protocol": protocol,
        "benchmark_context_id": context_id,
        "results": summarize_latencies(latencies, batch_size),
        "latencies_ms": [value * 1000 for value in latencies],
    }
