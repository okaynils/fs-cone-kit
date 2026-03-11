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
        self.val_image_path = None
        self.reverse_class_map = None
        self._last_fit_epoch_logged = None

    def setup(self, val_image_dir: str, class_map: dict, experiment_name: str, run_name: str):
        self.reverse_class_map = {v: k for k, v in class_map.items()}

        val_images = list(Path(val_image_dir).glob("*.jpg"))
        if val_images:
            self.val_image_path = str(val_images[0])

        metric_names = [metric.__class__.__name__ for metric in self.metrics]
        wandb.init(
            project=self.project or experiment_name,
            entity=self.entity,
            name=run_name,
            job_type="training",
            config={"selected_metrics": metric_names},
        )

        print(f"[{self.__class__.__name__}] WandB initialized. Run ID: {run_name}")

    def get_callbacks(self) -> dict:
        return {
            "on_train_epoch_end": self._on_train_epoch_end,
            "on_fit_epoch_end": self._on_fit_epoch_end,
            "on_train_end": self._on_train_end,
        }

    def _on_train_end(self, trainer):
        """Ensures WandB closes gracefully when training finishes."""
        wandb.finish()

    def _draw_ground_truth(self, image_path: str):
        """Reads the corresponding label file and draws GT boxes."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w, _ = img.shape

        label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
        if not os.path.exists(label_path):
            return img

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            name = self.reverse_class_map.get(cls_id, str(cls_id))
            cv2.putText(img, f"GT: {name}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img

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

        image = self._build_validation_image(trainer)
        if image is not None:
            payload["val_prediction_comparison"] = image

        if len(payload) > 1:
            wandb.log(payload, step=step)
            self._last_fit_epoch_logged = step

    def _build_validation_image(self, trainer):
        if not self.val_image_path:
            return None

        weight_path = getattr(trainer, "last", None)
        if not weight_path or not os.path.exists(weight_path):
            return None

        try:
            inference_model = YOLO(weight_path)
            results = inference_model(self.val_image_path, conf=0.1, verbose=False)
            res_plotted = results[0].plot(line_width=2)

            gt_plotted = self._draw_ground_truth(self.val_image_path)
            if gt_plotted is not None:
                if gt_plotted.shape != res_plotted.shape:
                    res_plotted = cv2.resize(res_plotted, (gt_plotted.shape[1], gt_plotted.shape[0]))
                combined = np.hstack((gt_plotted, res_plotted))
            else:
                combined = res_plotted

            res_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            return wandb.Image(res_rgb, caption=f"Epoch {trainer.epoch + 1}: Left=GT, Right=Pred")
        except Exception as e:
            print(f"[{self.__class__.__name__}] WandB image logging failed: {e}")
            return None
