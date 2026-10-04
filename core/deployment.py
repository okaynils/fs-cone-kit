"""Run exported ONNX detectors without Ultralytics and check them against their checkpoint.

The preprocessing and postprocessing here are the contract written into a
release bundle. Parity runs the exported file through this code, not through
Ultralytics. When parity passes, a team that implements the same contract on
the car runs the model that was measured.
"""

from __future__ import annotations

import ast
import importlib.metadata
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PAD_VALUE = 114
# Ultralytics keeps classes apart during NMS by shifting each class this far.
CLASS_OFFSET_PX = 7680
MAX_NMS_CANDIDATES = 30000
QUANTIZED_OPS = {
    "QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul",
    "ConvInteger", "MatMulInteger",
}


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Resize without changing aspect ratio, then pad to a centred square."""
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    new_width, new_height = round(width * gain), round(height * gain)
    if (width, height) != (new_width, new_height):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_width) / 2, (size - new_height) / 2
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )


def preprocess(image_bgr: np.ndarray, size: int, dtype: Any = np.float32) -> np.ndarray:
    """BGR uint8 HWC image to a 1x3xSxS RGB tensor in [0, 1]."""
    padded = letterbox(image_bgr, size)
    tensor = padded[..., ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=dtype) / dtype(255)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        best = order[0]
        keep.append(int(best))
        rest = order[1:]
        x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        union = areas[best] + areas[rest] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        order = rest[overlap <= iou_threshold]
    return keep


def postprocess(
    output: np.ndarray,
    input_size: int,
    image_shape: tuple[int, int],
    confidence: float,
    iou: float,
    max_det: int = 300,
) -> list[dict[str, Any]]:
    """Decode a raw 1x(4+classes)xN YOLO output into boxes in original image pixels."""
    predictions = np.asarray(output, dtype=np.float32)[0].T
    scores = predictions[:, 4:]
    class_ids = scores.argmax(1)
    confidences = scores.max(1)
    keep = confidences > confidence
    predictions, class_ids, confidences = predictions[keep], class_ids[keep], confidences[keep]
    if len(confidences) > MAX_NMS_CANDIDATES:
        top = confidences.argsort()[::-1][:MAX_NMS_CANDIDATES]
        predictions, class_ids, confidences = predictions[top], class_ids[top], confidences[top]

    x, y, width, height = predictions[:, :4].T
    boxes = np.stack([x - width / 2, y - height / 2, x + width / 2, y + height / 2], axis=1)
    shifted = boxes + class_ids[:, None].astype(np.float32) * CLASS_OFFSET_PX
    kept = _nms(shifted, confidences, iou)[:max_det]
    boxes, class_ids, confidences = boxes[kept], class_ids[kept], confidences[kept]

    image_height, image_width = image_shape
    gain = min(input_size / image_height, input_size / image_width)
    pad_x = round((input_size - image_width * gain) / 2 - 0.1)
    pad_y = round((input_size - image_height * gain) / 2 - 0.1)
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / gain).clip(0, image_width)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / gain).clip(0, image_height)
    return [
        {"class_id": int(class_id), "confidence": float(score), "xyxy": [float(v) for v in box]}
        for box, class_id, score in zip(boxes, class_ids, confidences)
    ]


def onnx_precision(model_path: Path) -> str:
    """Read the numeric precision from the graph instead of trusting a file name."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    if any(node.op_type in QUANTIZED_OPS for node in model.graph.node):
        return "int8"
    if any(item.data_type == onnx.TensorProto.FLOAT16 for item in model.graph.initializer):
        return "fp16"
    return "fp32"


def ultralytics_onnx_providers(device: str | int | None) -> list[str]:
    """The providers Ultralytics' AutoBackend picks for an ONNX file on this device."""
    import onnxruntime

    device = str(device if device is not None else "cpu").lower()
    available = onnxruntime.get_available_providers()
    if (device.isdigit() or device.startswith("cuda")) and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "mps" and "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def runtime_record(
    model_path: Path,
    device: str | int | None,
    half: bool = False,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Describe the runtime that produced a result, so .pt and .onnx rows are never confused."""
    model_path = Path(model_path)
    if model_path.suffix == ".onnx":
        return {
            "name": "onnxruntime",
            "version": importlib.metadata.version("onnxruntime"),
            "format": "onnx",
            "precision": onnx_precision(model_path),
            "providers": providers or ultralytics_onnx_providers(device),
        }
    import torch

    return {
        "name": "pytorch",
        "version": torch.__version__,
        "format": model_path.suffix.lstrip(".") or "pt",
        "precision": "fp16" if half else "fp32",
    }


DEVICE_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}


def target_providers(target: str, device: str = "cpu", precision: str = "fp32") -> list[Any]:
    """onnxruntime providers for a release target. Refuse rather than fall back."""
    import onnxruntime

    available = onnxruntime.get_available_providers()
    if target == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(
                "The tensorrt target needs onnxruntime-gpu built with TensorRT, and an NVIDIA GPU. "
                f"This environment only offers {available}. No TensorRT numbers were produced."
            )
        options = {"trt_fp16_enable": precision == "fp16", "trt_int8_enable": precision == "int8"}
        return [("TensorrtExecutionProvider", options), "CUDAExecutionProvider", "CPUExecutionProvider"]
    if target != "onnxruntime":
        raise ValueError(f"Unknown target {target!r}; use onnxruntime or tensorrt")
    if device not in DEVICE_PROVIDERS:
        raise ValueError(f"Unknown device {device!r} for onnxruntime; use one of {sorted(DEVICE_PROVIDERS)}")
    provider = DEVICE_PROVIDERS[device]
    if provider not in available:
        raise RuntimeError(f"{provider} is not available here; this environment offers {available}")
    return [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]


class OnnxDetector:
    """Contract-only inference: onnxruntime, NumPy, and OpenCV. No Ultralytics."""

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
        confidence: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
    ):
        import onnxruntime

        self.model_path = Path(model_path).resolve()
        providers = providers or ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(str(self.model_path), providers=providers)
        requested = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        if self.session.get_providers()[0] != requested:
            # onnxruntime quietly falls back to CPU when a provider fails to load.
            raise RuntimeError(
                f"Asked for {requested}, but onnxruntime is running {self.session.get_providers()[0]}"
            )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_dtype = np.float16 if model_input.type == "tensor(float16)" else np.float32
        metadata = self.session.get_modelmeta().custom_metadata_map
        size = model_input.shape[2]
        if not isinstance(size, int):
            size = ast.literal_eval(metadata.get("imgsz", "None"))
            size = size[0] if isinstance(size, (list, tuple)) else size
        if not isinstance(size, int):
            raise ValueError(f"{self.model_path} has a dynamic input size and no imgsz metadata")
        self.input_size = size
        names = ast.literal_eval(metadata["names"]) if "names" in metadata else {}
        self.names = {int(key): str(value) for key, value in names.items()}
        self.confidence = confidence
        self.iou = iou
        self.max_det = max_det

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        return preprocess(image_bgr, self.input_size, self.input_dtype)

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tensor})[0]

    def postprocess(self, output: np.ndarray, image_shape: tuple[int, int]) -> list[dict[str, Any]]:
        return postprocess(output, self.input_size, image_shape, self.confidence, self.iou, self.max_det)

    def predict(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        return self.postprocess(self.infer(self.preprocess(image_bgr)), image_bgr.shape[:2])


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        + max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def match_detections(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy one-to-one matching by IoU, ignoring class so class flips stay visible."""
    pairs = sorted(
        (
            (box_iou(ref["xyxy"], cand["xyxy"]), ref_index, cand_index)
            for ref_index, ref in enumerate(reference)
            for cand_index, cand in enumerate(candidate)
        ),
        reverse=True,
    )
    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    matches = []
    for overlap, ref_index, cand_index in pairs:
        if overlap < iou_threshold:
            break
        if ref_index in used_reference or cand_index in used_candidate:
            continue
        used_reference.add(ref_index)
        used_candidate.add(cand_index)
        matches.append((ref_index, cand_index))
    return (
        sorted(matches),
        [index for index in range(len(reference)) if index not in used_reference],
        [index for index in range(len(candidate)) if index not in used_candidate],
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    return float(np.percentile(values, fraction * 100)) if values else None


def compare_detections(
    reference: dict[str, list[dict[str, Any]]],
    candidate: dict[str, list[dict[str, Any]]],
    confidence: float,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare two sets of detections per image at a deployment confidence.

    Both sets should be produced below `confidence`. A detection counts as present
    in only one model when it reaches `confidence` and the other model has no box
    for it at all, so scores that straddle the threshold are not reported as misses.
    """
    deviations: list[float] = []
    confidence_differences: list[float] = []
    class_disagreements = 0
    only_reference = 0
    only_candidate = 0
    reference_count = 0
    candidate_count = 0
    worst_images: list[dict[str, Any]] = []
    for image in sorted(reference):
        ref, cand = reference[image], candidate.get(image, [])
        reference_count += sum(item["confidence"] >= confidence for item in ref)
        candidate_count += sum(item["confidence"] >= confidence for item in cand)
        matches, unmatched_ref, unmatched_cand = match_detections(ref, cand, iou_threshold)
        image_issues = 0
        for ref_index, cand_index in matches:
            left, right = ref[ref_index], cand[cand_index]
            if max(left["confidence"], right["confidence"]) < confidence:
                continue
            deviations.append(max(abs(a - b) for a, b in zip(left["xyxy"], right["xyxy"])))
            confidence_differences.append(abs(left["confidence"] - right["confidence"]))
            if left["class_id"] != right["class_id"]:
                class_disagreements += 1
                image_issues += 1
        missing = sum(ref[index]["confidence"] >= confidence for index in unmatched_ref)
        extra = sum(cand[index]["confidence"] >= confidence for index in unmatched_cand)
        only_reference += missing
        only_candidate += extra
        image_issues += missing + extra
        if image_issues:
            worst_images.append({"image": image, "issues": image_issues})

    compared = len(confidence_differences)
    return {
        "image_count": len(reference),
        "confidence_threshold": confidence,
        "match_iou_threshold": iou_threshold,
        "reference_detections": reference_count,
        "candidate_detections": candidate_count,
        "matched_detections": compared,
        "box_deviation_px": {
            "max": max(deviations) if deviations else 0.0,
            "mean": float(np.mean(deviations)) if deviations else 0.0,
            "p95": _percentile(deviations, 0.95) or 0.0,
        },
        "confidence_difference": {
            "max": max(confidence_differences) if confidence_differences else 0.0,
            "mean": float(np.mean(confidence_differences)) if confidence_differences else 0.0,
        },
        "class_disagreements": class_disagreements,
        "class_disagreement_rate": class_disagreements / compared if compared else 0.0,
        "only_in_reference": only_reference,
        "only_in_candidate": only_candidate,
        "unmatched_rate": (only_reference + only_candidate) / max(1, reference_count),
        "images_with_issues": sorted(worst_images, key=lambda item: (-item["issues"], item["image"]))[:20],
    }


TOLERANCE_CHECKS = {
    "max_box_deviation_px": lambda summary: summary["box_deviation_px"]["max"],
    "max_confidence_difference": lambda summary: summary["confidence_difference"]["max"],
    "max_class_disagreement_rate": lambda summary: summary["class_disagreement_rate"],
    "max_unmatched_rate": lambda summary: summary["unmatched_rate"],
}


def check_tolerances(summary: dict[str, Any], tolerances: dict[str, float]) -> list[dict[str, Any]]:
    """Return one record per tolerance. Unknown tolerance names are an error, not a silent pass."""
    unknown = set(tolerances) - set(TOLERANCE_CHECKS)
    if unknown:
        raise ValueError(f"Unknown parity tolerances: {sorted(unknown)}")
    if summary["reference_detections"] == 0:
        return [{
            "name": "reference_detections",
            "limit": 1,
            "value": 0,
            "passed": False,
            "note": "The checkpoint made no detections on the parity images, so nothing was compared.",
        }]
    return [
        {
            "name": name,
            "limit": float(limit),
            "value": float(TOLERANCE_CHECKS[name](summary)),
            "passed": float(TOLERANCE_CHECKS[name](summary)) <= float(limit),
        }
        for name, limit in sorted(tolerances.items())
    ]


def select_parity_images(members: list[Any], count: int) -> list[str]:
    """Evenly spaced, sorted, and therefore the same on every machine."""
    images = sorted(member["image"] if isinstance(member, dict) else str(member) for member in members)
    if count <= 0 or count >= len(images):
        return images
    step = len(images) / count
    return [images[int(index * step)] for index in range(count)]


def reference_detections(
    checkpoint: Path,
    images: dict[str, np.ndarray],
    input_size: int,
    confidence: float,
    iou: float,
    max_det: int,
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    """Run the PyTorch checkpoint through Ultralytics on the same square letterbox."""
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    results = {}
    for name, image in images.items():
        result = model.predict(
            source=image,
            imgsz=input_size,
            conf=confidence,
            iou=iou,
            max_det=max_det,
            rect=False,
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        results[name] = [
            {"class_id": int(class_id), "confidence": float(score), "xyxy": [float(v) for v in xyxy]}
            for xyxy, class_id, score in zip(
                boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()
            )
        ]
    return results
