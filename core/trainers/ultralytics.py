import importlib
import os
import re
from pathlib import Path

from core.deployment import runtime_record
from core.experiments import relative_to_experiment
from core.evaluation import serialize_ultralytics_evaluation
from core.trainers.base import BaseTrainer
from ultralytics import YOLO
from ultralytics.utils import SETTINGS


class UltralyticsTrainer(BaseTrainer):
    def __init__(
        self,
        args: dict,
        export_onnx: bool = True,
        resume_from: str | None = None,
        onnx_export_args: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_args = args
        self.export_onnx = export_onnx
        self.resume_from = resume_from
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
        experiment_dir: str | Path = ".",
    ):
        self.experiment_dir = Path(experiment_dir).resolve()
        self.model = YOLO(self.resume_from or model_weights)

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

        self.train_args["project"] = str(self.experiment_dir)
        self.train_args["name"] = "ultralytics_files"
        self.train_args["exist_ok"] = True
        if self.resume_from:
            self.train_args["resume"] = True

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

    def export_checkpoint_to_onnx(self, checkpoint_path: Path) -> Path:
        """Export one checkpoint and fail loudly. A missing ONNX file is not a usable release."""
        export_args = self._build_onnx_export_args()
        print(f"[{self.__class__.__name__}] Exporting {checkpoint_path.name} to ONNX...")

        try:
            exported_path = YOLO(str(checkpoint_path)).export(**export_args)
        except Exception as e:
            raise RuntimeError(f"ONNX export failed for {checkpoint_path}: {e}") from e
        if not exported_path or not Path(exported_path).is_file():
            raise RuntimeError(f"ONNX export for {checkpoint_path} reported no file: {exported_path!r}")
        print(f"[{self.__class__.__name__}] ONNX export complete: {exported_path}")
        return Path(exported_path)

    def export(self) -> list[Path]:
        if not self.export_onnx:
            return []
        trainer = getattr(self.model, "trainer", None)
        if trainer is None:
            print(f"[{self.__class__.__name__}] Skipping ONNX export because no trainer state was found.")
            return []

        exported = []
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

            exported.append(self.export_checkpoint_to_onnx(checkpoint_path))
        return exported

    def train(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Call setup() first.")

        result = self.model.train(**self.train_args)

        trainer = getattr(self.model, "trainer", None)
        if trainer is None:
            return {"status": "complete"}

        results_dict = getattr(getattr(trainer, "validator", None), "metrics", None)
        results_dict = getattr(results_dict, "results_dict", {}) or {}
        return {
            "status": "complete",
            "epochs_completed": int(getattr(trainer, "epoch", -1)) + 1,
            "best_checkpoint": relative_to_experiment(getattr(trainer, "best", None), self.experiment_dir),
            "last_checkpoint": relative_to_experiment(getattr(trainer, "last", None), self.experiment_dir),
            "metrics_csv": relative_to_experiment(
                getattr(trainer, "csv", self.experiment_dir / "ultralytics_files" / "results.csv"),
                self.experiment_dir,
            ),
            "final_validation_metrics": {
                str(key): float(value) for key, value in results_dict.items()
                if isinstance(value, (int, float)) or hasattr(value, "item")
            },
            "resumed_from": self.resume_from,
            "ultralytics_result_type": type(result).__name__,
        }

    def evaluate(
        self,
        model_path: str | Path,
        data: str,
        split: str,
        output_dir: str | Path,
        dataset_info: dict,
        evaluation_args: dict | None = None,
    ) -> dict:
        checkpoint = Path(model_path).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        model = YOLO(str(checkpoint))
        args = dict(evaluation_args or {})
        metrics = model.val(
            data=data,
            split=split,
            project=str(Path(output_dir).resolve().parent),
            name=Path(output_dir).name,
            exist_ok=True,
            plots=True,
            **args,
        )
        return serialize_ultralytics_evaluation(
            metrics=metrics,
            model=model,
            checkpoint=checkpoint,
            split=split,
            dataset_info=dataset_info,
            evaluation_args=args,
            runtime=runtime_record(checkpoint, args.get("device"), half=bool(args.get("half"))),
        )

    def predict_records(
        self,
        model_path: str | Path,
        image_paths: list[str],
        dataset_root: Path,
        confidence: float = 0.001,
        prediction_args: dict | None = None,
    ) -> list[dict]:
        """Save backend-neutral normalized boxes for later local analysis."""
        model = YOLO(str(Path(model_path).resolve()))
        allowed_args = {"imgsz", "device", "batch"}
        args = {
            key: value for key, value in dict(prediction_args or {}).items()
            if key in allowed_args
        }
        results = model.predict(
            source=image_paths,
            conf=confidence,
            stream=True,
            verbose=False,
            **args,
        )
        records = []
        for result in results:
            path = Path(result.path).resolve()
            try:
                image_name = str(path.relative_to(dataset_root.resolve()))
            except ValueError:
                image_name = str(path)
            height, width = result.orig_shape
            boxes = []
            if result.boxes is not None:
                coordinates = result.boxes.xyxy.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                for xyxy, class_id, score in zip(coordinates, classes, confidences):
                    boxes.append({
                        "class_id": int(class_id),
                        "confidence": float(score),
                        "xyxy": [
                            float(xyxy[0]) / width,
                            float(xyxy[1]) / height,
                            float(xyxy[2]) / width,
                            float(xyxy[3]) / height,
                        ],
                    })
            records.append({"image": image_name, "boxes": boxes})
        return records
