import os
import cv2
import numpy as np
import mlflow
from pathlib import Path
from ultralytics import YOLO

from core.loggers.base import BaseLogger

class GitLabMLflowLogger(BaseLogger):
    def __init__(self, **kwargs):
        self.val_image_path = None
        self.reverse_class_map = None

    def setup(self, val_image_dir: str, class_map: dict, experiment_name: str, run_name: str):
        self.reverse_class_map = {v: k for k, v in class_map.items()}

        val_images = list(Path(val_image_dir).glob("*.jpg"))
        if val_images:
            self.val_image_path = str(val_images[0])
            
        print(f"[{self.__class__.__name__}] Callbacks ready for injection.")

    def get_callbacks(self) -> dict:
        return {
            "on_fit_epoch_end": self._on_fit_epoch_end
        }

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
            cv2.putText(img, f"GT: {name}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img

    def _on_fit_epoch_end(self, trainer):
        """Callback triggered by Ultralytics at the end of every epoch."""
        # ensure we have an image and an active MLflow run (started by YOLO)
        if not self.val_image_path or not mlflow.active_run():
            return

        weight_path = getattr(trainer, "last", None)
        if not weight_path or not os.path.exists(weight_path):
            return

        try:
            # inference
            inference_model = YOLO(weight_path)
            results = inference_model(self.val_image_path, conf=0.1, verbose=False)
            res_plotted = results[0].plot(line_width=2)

            # ground truth
            gt_plotted = self._draw_ground_truth(self.val_image_path)

            # combine side-by-side
            if gt_plotted is not None:
                if gt_plotted.shape != res_plotted.shape:
                    res_plotted = cv2.resize(res_plotted, (gt_plotted.shape[1], gt_plotted.shape[0]))
                combined = np.hstack((gt_plotted, res_plotted))
            else:
                combined = res_plotted

            # convert BGR to RGB for MLflow (this is cringe) and then log it
            res_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            mlflow.log_image(res_rgb, f"val_predictions/epoch_{trainer.epoch + 1:03d}.jpg")

            del inference_model
        except Exception as e:
            print(f"[{self.__class__.__name__}] Custom logging failed: {e}")