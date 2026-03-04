import os
from core.trainers.base import BaseTrainer
from ultralytics import YOLO

class UltralyticsTrainer(BaseTrainer):
    def __init__(self, args: dict, **kwargs):
        super().__init__(**kwargs)
        self.train_args = args

    def setup(self, model_weights: str, experiment_name: str, run_name: str, callbacks: dict = None):
        self.model = YOLO(model_weights)
        
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        os.environ["MLFLOW_RUN"] = run_name
        
        self.train_args["project"] = "."
        self.train_args["name"] = "yolo_run"

        if callbacks:
            for event, func_list in callbacks.items():
                for func in func_list:
                    self.model.add_callback(event, func)
                
        print(f"[{self.__class__.__name__}] MLflow Experiment: '{experiment_name}' | Run: '{run_name}'")

    def train(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Call setup() first.")
        
        self.model.train(**self.train_args)