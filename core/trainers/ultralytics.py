import importlib
import os
import re
from pathlib import Path

from core.trainers.base import BaseTrainer
from ultralytics import YOLO
from ultralytics.utils import SETTINGS


class UltralyticsTrainer(BaseTrainer):
    def __init__(
        self,
        args: dict,
        export_onnx: bool = True,
        onnx_export_args: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_args = args
        self.export_onnx = export_onnx
        self.onnx_export_args = onnx_export_args or {}

    @staticmethod
    def _normalize_mlflow_tracking_uri(tracking_uri: str) -> str:
        if re.match(r"^[a-zA-Z]:[\\/]", tracking_uri):
            return Path(tracking_uri).resolve().as_uri()
        if "://" in tracking_uri:
            return tracking_uri
        return Path(tracking_uri).resolve().as_uri()

    def setup(
        self,
        model_weights: str,
        experiment_name: str,
        run_name: str,
        callbacks: dict = None,
        enable_mlflow: bool = False,
        mlflow_tracking_uri: str | None = None,
    ):
        self.model = YOLO(model_weights)

        dict.__setitem__(SETTINGS, "mlflow", enable_mlflow)
        import ultralytics.utils.callbacks.mlflow as ultralytics_mlflow_callbacks

        importlib.reload(ultralytics_mlflow_callbacks)
        if enable_mlflow:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
            os.environ["MLFLOW_RUN"] = run_name
            if mlflow_tracking_uri:
                os.environ["MLFLOW_TRACKING_URI"] = self._normalize_mlflow_tracking_uri(mlflow_tracking_uri)
        else:
            for key in ("MLFLOW_EXPERIMENT_NAME", "MLFLOW_RUN", "MLFLOW_TRACKING_URI"):
                os.environ.pop(key, None)

        self.train_args["project"] = "."
        self.train_args["name"] = "yolo_run"

        if callbacks:
            for event, func_list in callbacks.items():
                for func in func_list:
                    self.model.add_callback(event, func)

        print(
            f"[{self.__class__.__name__}] MLflow enabled: {enable_mlflow} | "
            f"Experiment: '{experiment_name}' | Run: '{run_name}'"
        )

    def _build_onnx_export_args(self) -> dict:
        export_args = {
            "format": "onnx",
            "simplify": False,
        }

        if "imgsz" in self.train_args:
            export_args["imgsz"] = self.train_args["imgsz"]

        export_args.update(self.onnx_export_args)
        return export_args

    def _export_checkpoint_to_onnx(self, checkpoint_path: Path):
        export_args = self._build_onnx_export_args()
        print(f"[{self.__class__.__name__}] Exporting {checkpoint_path.name} to ONNX...")

        try:
            exported_path = YOLO(str(checkpoint_path)).export(**export_args)
            print(f"[{self.__class__.__name__}] ONNX export complete: {exported_path}")
        except Exception as e:
            print(f"[{self.__class__.__name__}] ONNX export failed for {checkpoint_path.name}: {e}")

    def _export_trained_checkpoints(self):
        trainer = getattr(self.model, "trainer", None)
        if trainer is None:
            print(f"[{self.__class__.__name__}] Skipping ONNX export because no trainer state was found.")
            return

        for checkpoint_name in ("last", "best"):
            checkpoint_path = getattr(trainer, checkpoint_name, None)
            if not checkpoint_path:
                continue

            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                print(
                    f"[{self.__class__.__name__}] Skipping ONNX export for {checkpoint_name}.pt because the checkpoint "
                    f"was not found at {checkpoint_path}."
                )
                continue

            self._export_checkpoint_to_onnx(checkpoint_path)

    def train(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Call setup() first.")

        self.model.train(**self.train_args)

        if self.export_onnx:
            self._export_trained_checkpoints()
