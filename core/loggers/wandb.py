import os
from pathlib import Path

import cv2
import numpy as np
import wandb
from ultralytics import YOLO

from core.loggers.base import BaseLogger


class WandBLogger(BaseLogger):
    def __init__(self, project: str | None = None, entity: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.entity = entity
        self.val_image_paths: list[str] = []
        self.reverse_class_map = None
        self.class_colors: dict[int, tuple[int, int, int]] = {}
        self._last_fit_epoch_logged = None

    def setup(
        self,
        val_image_dir: str,
        class_map: dict,
        experiment_name: str,
        run_name: str,
        val_image_paths: list[str] | None = None,
        class_colors: dict[int, tuple[int, int, int]] | None = None,
    ):
        self.reverse_class_map = {v: k for k, v in class_map.items()}
        self.class_colors = class_colors or {}

        selected_image_paths = list(val_image_paths or [])
        if not selected_image_paths:
            selected_image_paths = [str(path) for path in sorted(Path(val_image_dir).glob("*.jpg"))[:4]]
        self.val_image_paths = selected_image_paths

        metric_names = [metric.__class__.__name__ for metric in self.metrics]
        wandb.init(
            project=self.project or experiment_name,
            entity=self.entity,
            name=run_name,
            job_type="training",
            config={"selected_metrics": metric_names},
        )

        print(
            f"[{self.__class__.__name__}] WandB initialized. Run ID: {run_name} | "
            f"Validation plots: {len(self.val_image_paths)}"
        )

    def get_callbacks(self) -> dict:
        return {
            "on_train_epoch_end": self._on_train_epoch_end,
            "on_fit_epoch_end": self._on_fit_epoch_end,
            "on_train_end": self._on_train_end,
        }

    def _on_train_end(self, trainer):
        """Ensures WandB closes gracefully when training finishes."""
        wandb.finish()

    def _get_box_color(self, cls_id: int) -> tuple[int, int, int]:
        color_rgb = self.class_colors.get(cls_id)
        if color_rgb is not None:
            return tuple(int(channel) for channel in reversed(color_rgb))

        palette = [
            (255, 0, 0),
            (0, 255, 255),
            (0, 165, 255),
            (0, 140, 255),
            (160, 160, 160),
        ]
        return palette[cls_id % len(palette)]

    @staticmethod
    def _add_panel_title(image: np.ndarray, title: str) -> np.ndarray:
        banner_height = 42
        titled_image = cv2.copyMakeBorder(
            image,
            banner_height,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        cv2.putText(
            titled_image,
            title,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return titled_image

    def _draw_annotations(self, image: np.ndarray, annotations: list[dict], title: str) -> np.ndarray:
        plotted = image.copy()

        for annotation in annotations:
            cls_id = annotation["cls_id"]
            x1, y1, x2, y2 = annotation["xyxy"]
            label = annotation["label"]
            color = self._get_box_color(cls_id)

            cv2.rectangle(plotted, (x1, y1), (x2, y2), color, 2)

            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_top = max(0, y1 - label_size[1] - baseline - 4)
            label_bottom = label_top + label_size[1] + baseline + 4
            label_right = x1 + label_size[0] + 8
            cv2.rectangle(plotted, (x1, label_top), (label_right, label_bottom), color, -1)
            cv2.putText(
                plotted,
                label,
                (x1 + 4, label_bottom - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return self._add_panel_title(plotted, title)

    def _read_ground_truth_annotations(self, image_path: str) -> list[dict]:
        image = cv2.imread(image_path)
        if image is None:
            return []
        h, w, _ = image.shape

        label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
        if not os.path.exists(label_path):
            return []

        annotations = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                class_name = self.reverse_class_map.get(cls_id, str(cls_id))
                annotations.append({
                    "cls_id": cls_id,
                    "xyxy": (x1, y1, x2, y2),
                    "label": class_name,
                })

        return annotations

    def _read_prediction_annotations(self, result) -> list[dict]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy_values = boxes.xyxy.cpu().tolist()
        cls_values = boxes.cls.cpu().tolist()
        conf_values = boxes.conf.cpu().tolist()
        annotations = []

        for xyxy, cls_id, conf in zip(xyxy_values, cls_values, conf_values):
            class_id = int(cls_id)
            x1, y1, x2, y2 = [int(value) for value in xyxy]
            class_name = self.reverse_class_map.get(class_id, str(class_id))
            annotations.append({
                "cls_id": class_id,
                "xyxy": (x1, y1, x2, y2),
                "label": f"{class_name} {conf:.2f}",
            })

        return annotations

    def _build_epoch_payload(self, trainer, event: str) -> dict[str, object]:
        payload: dict[str, object] = self.collect_metrics(trainer=trainer, event=event)
        payload["epoch"] = trainer.epoch + 1
        return payload

    def _log_epoch_payload(self, trainer, event: str):
        if not wandb.run:
            return

        payload = self._build_epoch_payload(trainer=trainer, event=event)
        if len(payload) == 1 and "epoch" in payload:
            return

        wandb.log(payload, step=trainer.epoch + 1)

    def _on_train_epoch_end(self, trainer):
        self._log_epoch_payload(trainer=trainer, event="on_train_epoch_end")

    def _on_fit_epoch_end(self, trainer):
        if not wandb.run:
            return

        step = trainer.epoch + 1
        if self._last_fit_epoch_logged == step:
            return

        payload = self._build_epoch_payload(trainer=trainer, event="on_fit_epoch_end")

        images = self._build_validation_images(trainer)
        if images:
            payload["val_prediction_comparisons"] = images

        if len(payload) > 1:
            wandb.log(payload, step=step)
            self._last_fit_epoch_logged = step

    def _build_validation_images(self, trainer) -> list[wandb.Image]:
        if not self.val_image_paths:
            return []

        weight_path = getattr(trainer, "last", None)
        if not weight_path or not os.path.exists(weight_path):
            return []

        try:
            inference_model = YOLO(weight_path)
            results = inference_model(self.val_image_paths, conf=0.1, verbose=False)
            comparison_images: list[wandb.Image] = []

            for image_path, result in zip(self.val_image_paths, results):
                source_image = cv2.imread(image_path)
                if source_image is None:
                    continue

                gt_annotations = self._read_ground_truth_annotations(image_path)
                pred_annotations = self._read_prediction_annotations(result)

                gt_panel = self._draw_annotations(source_image, gt_annotations, "Ground Truth")
                pred_panel = self._draw_annotations(source_image, pred_annotations, "Prediction")
                comparison = np.hstack((gt_panel, pred_panel))
                comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)

                comparison_images.append(
                    wandb.Image(
                        comparison_rgb,
                        caption=f"Epoch {trainer.epoch + 1}: {Path(image_path).name}"
                    )
                )

            return comparison_images
        except Exception as e:
            print(f"[{self.__class__.__name__}] WandB image logging failed: {e}")
            return []
