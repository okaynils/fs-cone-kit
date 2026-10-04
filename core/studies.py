"""Reproducible slice preparation and local analysis for detector studies."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.evaluate import _checkpoint_path
from core.experiments import SCHEMA_VERSION, write_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _difference_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image for leakage check: {path}")
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    differences = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in differences.flat:
        value = (value << 1) | int(bit)
    return value


def _label_for_member(member: dict[str, Any], dataset_root: Path) -> Path:
    if member.get("label"):
        return dataset_root / member["label"]
    image = Path(member["image"])
    parts = list(image.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    return dataset_root.joinpath(*parts).with_suffix(".txt")


def _read_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    annotations = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected five YOLO fields in {path}:{line_number}")
        class_id, x, y, width, height = fields
        annotations.append({
            "class_id": int(class_id),
            "xywh": [float(x), float(y), float(width), float(height)],
        })
    return annotations


def assign_small_cone_slice(
    annotations: list[dict[str, Any]], area_threshold: float = 0.005
) -> str:
    """Assign an image to one of two disjoint, annotation-derived slices."""
    if any(
        item["xywh"][2] * item["xywh"][3] <= area_threshold + 1e-12
        for item in annotations
    ):
        return "small_cones"
    return "ordinary"


def _source_group(member: dict[str, Any]) -> str | None:
    if member.get("source_group"):
        return str(member["source_group"])
    source = member.get("source_annotation")
    return Path(source).parts[0] if source else None


def _normalize_members(dataset_info: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    normalized = {}
    for split, members in dataset_info["manifest"].get("splits", {}).items():
        normalized[split] = [
            dict(member) if isinstance(member, dict) else {"image": str(member)}
            for member in members
        ]
    return normalized


def inspect_split_leakage(
    members: dict[str, list[dict[str, Any]]],
    dataset_root: Path,
    near_duplicate_distance: int = 4,
) -> dict[str, Any]:
    """Check source groups, exact content, and perceptual hashes across splits."""
    source_splits: dict[str, set[str]] = defaultdict(set)
    exact: dict[str, list[tuple[str, str]]] = defaultdict(list)
    hashes: list[tuple[str, str, int, str]] = []

    for split, split_members in sorted(members.items()):
        for member in split_members:
            relative = str(member["image"])
            image_path = dataset_root / relative
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            group = _source_group(member)
            if group:
                source_splits[group].add(split)
            content_hash = _sha256(image_path)
            exact[content_hash].append((split, relative))
            hashes.append((split, relative, _difference_hash(image_path), content_hash))

    source_overlaps = [
        {"source_group": group, "splits": sorted(splits)}
        for group, splits in sorted(source_splits.items())
        if len(splits) > 1
    ]
    exact_pairs = []
    for content_hash, entries in exact.items():
        entry_splits = {entry[0] for entry in entries}
        if len(entry_splits) > 1:
            exact_pairs.append({
                "sha256": content_hash,
                "images": [{"split": split, "image": image} for split, image in sorted(entries)],
            })

    # A pair within four bits must share one of five disjoint hash chunks. This
    # avoids a quadratic all-pairs scan on the full dataset.
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidate_pairs: set[tuple[int, int]] = set()
    offsets = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))
    for index, (split, _, value, _) in enumerate(hashes):
        for chunk_index, (offset, width) in enumerate(offsets):
            key = (chunk_index, (value >> offset) & ((1 << width) - 1))
            for other_index in buckets[key]:
                if hashes[other_index][0] != split:
                    candidate_pairs.add((other_index, index))
            buckets[key].append(index)

    near_pairs = []
    for left_index, right_index in sorted(candidate_pairs):
        left = hashes[left_index]
        right = hashes[right_index]
        if left[3] == right[3]:
            continue
        distance = (left[2] ^ right[2]).bit_count()
        if distance <= near_duplicate_distance:
            near_pairs.append({
                "distance": distance,
                "left": {"split": left[0], "image": left[1]},
                "right": {"split": right[0], "image": right[1]},
            })

    source_status = "pass" if source_splits and not source_overlaps else (
        "unverified" if not source_splits else "fail"
    )
    status = "fail" if source_overlaps or exact_pairs or near_pairs else "pass"
    return {
        "status": status,
        "source_group_check": source_status,
        "near_duplicate_hash": "64-bit difference hash",
        "near_duplicate_max_distance": near_duplicate_distance,
        "source_group_overlaps": source_overlaps,
        "exact_duplicates": exact_pairs,
        "near_duplicates": near_pairs,
    }


def _slice_summary(images: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        annotation["class_id"]
        for image in images
        for annotation in image["annotations"]
    )
    return {
        "image_count": len(images),
        "cone_count": sum(counts.values()),
        "per_class_cone_count": {str(key): counts[key] for key in sorted(counts)},
        "images": images,
    }


def prepare_study(
    experiment_dir: Path,
    output_dir: Path,
    area_threshold: float = 0.005,
    near_duplicate_distance: int = 4,
) -> Path:
    """Build fixed test slices from a recorded experiment's exact dataset."""
    experiment_dir = experiment_dir.resolve()
    config_path = experiment_dir / "experiment" / "config.yaml"
    dataset_record_path = experiment_dir / "experiment" / "dataset.json"
    if not config_path.exists() or not dataset_record_path.exists():
        raise FileNotFoundError(f"{experiment_dir} is not a recorded experiment")

    cfg = OmegaConf.load(config_path)
    dataset = instantiate(cfg.dataset)
    dataset.prepare()
    current = dataset.get_dataset_info()
    recorded = json.loads(dataset_record_path.read_text(encoding="utf-8"))
    if current["fingerprint"] != recorded["fingerprint"]:
        raise RuntimeError("Prepared dataset does not match the experiment's recorded split")

    dataset_root = Path(current["dataset_yaml"]).parent.resolve()
    members = _normalize_members(current)
    if "test" not in members:
        raise ValueError("The recorded dataset has no test split")

    enriched: dict[str, list[dict[str, Any]]] = {}
    for split, split_members in members.items():
        enriched[split] = []
        for member in split_members:
            label = _label_for_member(member, dataset_root)
            enriched[split].append({
                **member,
                "label": str(label.relative_to(dataset_root)),
                "source_group": _source_group(member),
                "annotations": _read_annotations(label),
            })

    test_images = enriched["test"]
    small = [
        image for image in test_images
        if assign_small_cone_slice(image["annotations"], area_threshold) == "small_cones"
    ]
    ordinary = [image for image in test_images if image not in small]
    leakage = inspect_split_leakage(enriched, dataset_root, near_duplicate_distance)
    class_names = {str(value): key for key, value in dict(cfg.dataset.class_map).items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study": "small-cone-scale-augmentation",
        "condition": "small cones inferred from ground-truth normalized box area",
        "dataset_fingerprint": current["fingerprint"],
        "dataset_root": str(dataset_root),
        "dataset_yaml": current["dataset_yaml"],
        "split_counts": current["split_counts"],
        "class_names": class_names,
        "per_class_min_cones": 10,
        "slice_rule": {
            "small_cones": f"at least one box with normalized area <= {area_threshold}",
            "ordinary": f"all boxes have normalized area > {area_threshold}",
            "area_threshold": area_threshold,
            "caveat": "Slice metrics include every annotated cone in each selected image.",
        },
        "slices": {
            "full": _slice_summary(test_images),
            "small_cones": _slice_summary(small),
            "ordinary": _slice_summary(ordinary),
        },
        "leakage": leakage,
        "separation_limit": (
            "FSOCO exposes source-team grouping but not recording-session or track identifiers."
        ),
    }
    destination = output_dir.resolve() / "manifest.json"
    write_json(destination, manifest)
    return destination


def _slice_dataset_yaml(
    study_dir: Path,
    slice_name: str,
    manifest: dict[str, Any],
) -> Path:
    slice_dir = study_dir / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    list_path = slice_dir / f"{slice_name}.txt"
    image_paths = [
        str(Path(manifest["dataset_root"]) / image["image"])
        for image in manifest["slices"][slice_name]["images"]
    ]
    list_path.write_text("\n".join(image_paths) + ("\n" if image_paths else ""), encoding="utf-8")
    yaml_path = slice_dir / f"{slice_name}.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": manifest["dataset_root"],
        "train": str(list_path),
        "val": str(list_path),
        "test": str(list_path),
        "names": {int(key): value for key, value in manifest["class_names"].items()},
    }, sort_keys=False), encoding="utf-8")
    return yaml_path


def _training_controls(cfg: Any) -> dict[str, Any]:
    args = OmegaConf.to_container(cfg.trainer.args, resolve=True)
    trainer = OmegaConf.to_container(cfg.trainer, resolve=True)
    ignored = {"data", "project", "name", "exist_ok", "scale"}
    return {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "seed": int(cfg.seed),
        "trainer_args": {key: value for key, value in args.items() if key not in ignored},
        "trainer_settings": {
            key: value for key, value in trainer.items()
            if key not in {"_target_", "args"}
        },
    }


def evaluate_slices(
    trainer: Any,
    model_path: Path,
    study_dir: Path,
    manifest: dict[str, Any],
    evaluation_args: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Evaluate one model on every fixed slice in a study manifest."""
    slice_results = {}
    for slice_name, summary in manifest["slices"].items():
        if not summary["image_count"]:
            slice_results[slice_name] = {"status": "empty"}
            continue
        slice_yaml = _slice_dataset_yaml(study_dir, slice_name, manifest)
        slice_fingerprint = hashlib.sha256(
            json.dumps({
                "dataset": manifest["dataset_fingerprint"],
                "slice": slice_name,
                "images": [item["image"] for item in summary["images"]],
            }, sort_keys=True).encode()
        ).hexdigest()
        result = trainer.evaluate(
            model_path=model_path,
            data=str(slice_yaml),
            split="test",
            output_dir=output_root / slice_name,
            dataset_info={"fingerprint": slice_fingerprint, "split_counts": {"test": summary["image_count"]}},
            evaluation_args=evaluation_args,
        )
        result["slice"] = slice_name
        result["image_count"] = summary["image_count"]
        result["cone_count"] = summary["cone_count"]
        slice_results[slice_name] = result
    return slice_results


def evaluate_study(
    experiment_dir: Path,
    study_dir: Path,
    checkpoint: str = "best",
    prediction_confidence: float = 0.001,
    **evaluation_args: Any,
) -> tuple[Path, Path]:
    """Evaluate one recorded checkpoint on every fixed slice and save predictions."""
    experiment_dir = experiment_dir.resolve()
    study_dir = study_dir.resolve()
    manifest = json.loads((study_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = OmegaConf.load(experiment_dir / "experiment" / "config.yaml")
    recorded_dataset = json.loads(
        (experiment_dir / "experiment" / "dataset.json").read_text(encoding="utf-8")
    )
    if recorded_dataset["fingerprint"] != manifest["dataset_fingerprint"]:
        raise RuntimeError("Experiment and study use different dataset splits")
    if manifest["leakage"]["status"] != "pass":
        raise RuntimeError("Leakage audit failed; resolve it before evaluating the study")

    args = dict(evaluation_args)
    args.setdefault("imgsz", int(cfg.trainer.args.imgsz))
    trainer = instantiate(cfg.trainer)
    model_path = _checkpoint_path(experiment_dir, checkpoint)
    slice_results = evaluate_slices(
        trainer, model_path, study_dir, manifest, args,
        output_root=experiment_dir / "evaluation_files" / "study",
    )

    full_images = [
        str(Path(manifest["dataset_root"]) / item["image"])
        for item in manifest["slices"]["full"]["images"]
    ]
    predictions = trainer.predict_records(
        model_path=model_path,
        image_paths=full_images,
        dataset_root=Path(manifest["dataset_root"]),
        confidence=prediction_confidence,
        prediction_args=args,
    )

    name = experiment_dir.name
    predictions_path = study_dir / "predictions" / f"{name}.json"
    write_json(predictions_path, {
        "schema_version": SCHEMA_VERSION,
        "experiment": name,
        "checkpoint": str(model_path),
        "minimum_confidence": prediction_confidence,
        "predictions": predictions,
    })
    result_path = study_dir / "results" / f"{name}.json"
    scale = OmegaConf.select(cfg, "trainer.args.scale", default=None)
    write_json(result_path, {
        "schema_version": SCHEMA_VERSION,
        "experiment": name,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "controls": _training_controls(cfg),
        "intervention": {"scale": scale},
        "checkpoint": checkpoint,
        "evaluation_args": args,
        "slice_results": slice_results,
        "predictions": str(predictions_path.relative_to(study_dir)),
    })
    return result_path, predictions_path


def _metric(result: dict[str, Any], slice_name: str, name: str) -> float | None:
    return result["slice_results"].get(slice_name, {}).get("metrics", {}).get(name)


def aggregate_study_results(
    baseline: dict[str, Any], intervention: dict[str, Any]
) -> list[dict[str, Any]]:
    if baseline["dataset_fingerprint"] != intervention["dataset_fingerprint"]:
        raise ValueError("Results use different dataset splits")
    if baseline["controls"] != intervention["controls"]:
        raise ValueError("Training controls differ beyond the intervention")
    if baseline["evaluation_args"] != intervention["evaluation_args"]:
        raise ValueError("Evaluation settings differ")
    if baseline["checkpoint"] != intervention["checkpoint"]:
        raise ValueError("Checkpoint selection differs")
    rows = []
    for slice_name in ("full", "small_cones", "ordinary"):
        for metric_name in ("map50_95", "recall"):
            base_value = _metric(baseline, slice_name, metric_name)
            changed_value = _metric(intervention, slice_name, metric_name)
            rows.append({
                "slice": slice_name,
                "metric": metric_name,
                "baseline": base_value,
                "intervention": changed_value,
                "delta": (
                    changed_value - base_value
                    if base_value is not None and changed_value is not None else None
                ),
            })
    return rows


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [x - width / 2, y - height / 2, x + width / 2, y + height / 2]


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def match_failures(
    annotations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    confidence: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, list[int]]:
    candidates = [
        (index, prediction) for index, prediction in enumerate(predictions)
        if prediction["confidence"] >= confidence
    ]
    candidates.sort(key=lambda item: item[1]["confidence"], reverse=True)
    matched_ground_truth: set[int] = set()
    matched_predictions: set[int] = set()
    for prediction_index, prediction in candidates:
        choices = []
        for gt_index, annotation in enumerate(annotations):
            if gt_index in matched_ground_truth or annotation["class_id"] != prediction["class_id"]:
                continue
            overlap = _iou(_xywh_to_xyxy(annotation["xywh"]), prediction["xyxy"])
            if overlap >= iou_threshold:
                choices.append((overlap, gt_index))
        if choices:
            _, gt_index = max(choices)
            matched_ground_truth.add(gt_index)
            matched_predictions.add(prediction_index)
    return {
        "false_negatives": [index for index in range(len(annotations)) if index not in matched_ground_truth],
        "false_positives": [index for index, _ in candidates if index not in matched_predictions],
        "matched_predictions": sorted(matched_predictions),
    }


def _draw_panel(
    image: Any,
    title: str,
    annotations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    failures: dict[str, list[int]],
) -> Any:
    panel = image.copy()
    height, width = panel.shape[:2]
    false_negatives = set(failures["false_negatives"])
    false_positives = set(failures["false_positives"])
    for index, annotation in enumerate(annotations):
        x1, y1, x2, y2 = _xywh_to_xyxy(annotation["xywh"])
        color = (0, 140, 255) if index in false_negatives else (0, 200, 0)
        cv2.rectangle(panel, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), color, 2)
    for index, prediction in enumerate(predictions):
        if prediction["confidence"] < 0.25:
            continue
        x1, y1, x2, y2 = prediction["xyxy"]
        color = (0, 0, 255) if index in false_positives else (255, 180, 0)
        cv2.rectangle(panel, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), color, 2)
        cv2.putText(panel, f"{prediction['confidence']:.2f}", (int(x1 * width), max(12, int(y1 * height) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    header = cv2.copyMakeBorder(panel, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(header, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return header


def _generate_gallery(
    manifest: dict[str, Any],
    baseline_predictions: dict[str, Any],
    intervention_predictions: dict[str, Any],
    output_dir: Path,
    examples_per_category: int = 2,
) -> dict[str, list[str]]:
    baseline_by_image = {item["image"]: item["boxes"] for item in baseline_predictions["predictions"]}
    intervention_by_image = {item["image"]: item["boxes"] for item in intervention_predictions["predictions"]}
    categories: dict[str, list[tuple[str, dict, dict, dict]]] = defaultdict(list)
    for item in manifest["slices"]["full"]["images"]:
        image_name = item["image"]
        baseline_boxes = baseline_by_image.get(image_name, [])
        intervention_boxes = intervention_by_image.get(image_name, [])
        baseline_failures = match_failures(item["annotations"], baseline_boxes)
        intervention_failures = match_failures(item["annotations"], intervention_boxes)
        baseline_errors = len(baseline_failures["false_negatives"]) + len(baseline_failures["false_positives"])
        intervention_errors = len(intervention_failures["false_negatives"]) + len(intervention_failures["false_positives"])
        if not baseline_errors and not intervention_errors:
            continue
        category = "helps" if intervention_errors < baseline_errors else (
            "hurts" if intervention_errors > baseline_errors else "no_clear_difference"
        )
        categories[category].append((image_name, item, baseline_failures, intervention_failures))

    written: dict[str, list[str]] = {}
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for category in ("helps", "hurts", "no_clear_difference"):
        written[category] = []
        for index, (image_name, item, base_failures, changed_failures) in enumerate(
            sorted(categories[category], key=lambda entry: entry[0])[:examples_per_category], 1
        ):
            image = cv2.imread(str(Path(manifest["dataset_root"]) / image_name))
            if image is None:
                continue
            base_boxes = baseline_by_image.get(image_name, [])
            changed_boxes = intervention_by_image.get(image_name, [])
            base_panel = _draw_panel(image, "baseline", item["annotations"], base_boxes, base_failures)
            changed_panel = _draw_panel(image, "scale augmentation", item["annotations"], changed_boxes, changed_failures)
            canvas = cv2.hconcat([base_panel, changed_panel])
            destination = gallery_dir / f"{category}-{index}.jpg"
            cv2.imwrite(str(destination), canvas)
            written[category].append(str(destination.relative_to(output_dir)))
    return written


def write_study_report(
    study_dir: Path,
    baseline_result_path: Path,
    intervention_result_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Build the final report only from saved study artifacts and local images."""
    study_dir = study_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((study_dir / "manifest.json").read_text(encoding="utf-8"))
    baseline = json.loads(baseline_result_path.read_text(encoding="utf-8"))
    intervention = json.loads(intervention_result_path.read_text(encoding="utf-8"))
    rows = aggregate_study_results(baseline, intervention)
    baseline_predictions_path = Path(baseline["predictions"])
    intervention_predictions_path = Path(intervention["predictions"])
    if not baseline_predictions_path.is_absolute():
        baseline_predictions_path = study_dir / baseline_predictions_path
    if not intervention_predictions_path.is_absolute():
        intervention_predictions_path = study_dir / intervention_predictions_path
    baseline_predictions = json.loads(baseline_predictions_path.read_text(encoding="utf-8"))
    intervention_predictions = json.loads(intervention_predictions_path.read_text(encoding="utf-8"))
    gallery = _generate_gallery(manifest, baseline_predictions, intervention_predictions, output_dir)

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("slice", "metric", "baseline", "intervention", "delta"))
        writer.writeheader()
        writer.writerows(rows)

    row_lookup = {(row["slice"], row["metric"]): row for row in rows}
    small_delta = row_lookup[("small_cones", "map50_95")]["delta"]
    ordinary_delta = row_lookup[("ordinary", "map50_95")]["delta"]
    accepted = small_delta is not None and ordinary_delta is not None and small_delta >= 0.02 and ordinary_delta >= -0.01
    decision = "supports the hypothesis" if accepted else "does not meet the pre-registered acceptance criteria"

    lines = [
        "# small-cone failure study",
        "",
        "## hypothesis",
        "",
        "Stronger scale augmentation improves detection in images containing small, likely distant cones without materially reducing performance on ordinary images.",
        "",
        "## finding",
        "",
        f"This single-seed comparison **{decision}**. It does not establish statistical significance or real-world robustness.",
        "",
        "## setup",
        "",
        "The condition is inferred from annotation size, not measured distance. " + manifest["slice_rule"]["caveat"],
        f"The baseline used `scale={baseline['intervention']['scale']}` and the intervention used `scale={intervention['intervention']['scale']}`. All recorded controls match.",
        (
            f"Both runs used `{baseline['controls']['model']['name']}` from "
            f"`{baseline['controls']['model']['weights']}`, seed `{baseline['controls']['seed']}`, "
            f"`{baseline['controls']['trainer_args'].get('epochs')}` epochs, image size "
            f"`{baseline['controls']['trainer_args'].get('imgsz')}`, and batch size "
            f"`{baseline['controls']['trainer_args'].get('batch')}`."
        ),
        (
            f"The recorded split contains {manifest['split_counts'].get('train', 0)} train, "
            f"{manifest['split_counts'].get('val', 0)} validation, and "
            f"{manifest['split_counts'].get('test', 0)} test images."
        ),
        (
            f"Leakage audit: **{manifest['leakage']['status']}** with "
            f"{len(manifest['leakage']['source_group_overlaps'])} cross-split source groups, "
            f"{len(manifest['leakage']['exact_duplicates'])} exact duplicate sets, and "
            f"{len(manifest['leakage']['near_duplicates'])} near-duplicate pairs. "
            f"{manifest['separation_limit']}"
        ),
        "The pre-registered decision requires at least +0.02 absolute small-slice mAP50-95 and no worse than -0.01 absolute ordinary-slice mAP50-95.",
        "",
        "| slice | images | cones | mAP50-95 baseline | mAP50-95 intervention | delta | recall baseline | recall intervention | delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for slice_name in ("full", "small_cones", "ordinary"):
        summary = manifest["slices"][slice_name]
        map_row = row_lookup[(slice_name, "map50_95")]
        recall_row = row_lookup[(slice_name, "recall")]
        values = [map_row["baseline"], map_row["intervention"], map_row["delta"], recall_row["baseline"], recall_row["intervention"], recall_row["delta"]]
        formatted = ["—" if value is None else f"{value:.4f}" for value in values]
        lines.append(f"| {slice_name} | {summary['image_count']} | {summary['cone_count']} | " + " | ".join(formatted) + " |")

    lines.extend([
        "",
        "## per-class results",
        "",
        f"Classes with fewer than {manifest.get('per_class_min_cones', 10)} annotated cones in a slice are omitted.",
        "",
    ])
    for slice_name in ("full", "small_cones", "ordinary"):
        counts = manifest["slices"][slice_name]["per_class_cone_count"]
        lines.extend([
            f"### {slice_name}",
            "",
            "| class | cones | baseline mAP50-95 | intervention mAP50-95 | baseline recall | intervention recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        base_classes = {str(item["class_id"]): item for item in baseline["slice_results"][slice_name].get("per_class", [])}
        changed_classes = {str(item["class_id"]): item for item in intervention["slice_results"][slice_name].get("per_class", [])}
        for class_id, count in counts.items():
            if count < manifest.get("per_class_min_cones", 10):
                continue
            base = base_classes.get(class_id, {})
            changed = changed_classes.get(class_id, {})
            class_name = manifest["class_names"].get(class_id, class_id)
            metrics = [base.get("map50_95"), changed.get("map50_95"), base.get("recall"), changed.get("recall")]
            values = ["—" if value is None else f"{value:.4f}" for value in metrics]
            lines.append(f"| {class_name} | {count} | " + " | ".join(values) + " |")
        lines.append("")

    lines.extend([
        "## representative failures",
        "",
        "Ground truth is green, false-negative ground truth is orange, matched predictions are blue, and false positives are red. Matching uses confidence 0.25 and class-aware IoU 0.5.",
        "",
    ])
    for category in ("helps", "hurts", "no_clear_difference"):
        lines.append(f"### {category.replace('_', ' ')}")
        lines.append("")
        if gallery[category]:
            lines.extend(f"![{category}]({path})" for path in gallery[category])
        else:
            lines.append("No qualifying example was present in the saved predictions.")
        lines.append("")
    lines.extend([
        "## limitations and next experiment",
        "",
        "There is one run per condition, so run-to-run variation is unknown. Small normalized boxes are only a proxy for distant cones, and image-level slice metrics include larger cones from the same frames. FSOCO does not expose recording-session or track IDs, so separation below source-team level cannot be verified.",
        "",
        "Repeat both conditions with seeds 43 and 44 before treating the observed delta as stable. A follow-up dataset with measured distance, lighting, and recording IDs would test whether the effect transfers beyond this annotation-derived slice.",
    ])
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, csv_path
