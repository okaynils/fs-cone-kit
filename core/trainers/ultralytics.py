from core.trainers.base import BaseTrainer
from ultralytics import YOLO
from hydra.core.hydra_config import HydraConfig

class UltralyticsTrainer(BaseTrainer):
    def __init__(self, args: dict, **kwargs):
        super().__init__(**kwargs)
        self.train_args = args

    def setup(self, model_name: str):
        self.model = YOLO(model_name)

        print(f"[{self.__class__.__name__}] Ultralytics model '{model_name}' initialized.")

    def train(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Call setup() first.")
        
        hydra_output_dir = HydraConfig.get().runtime.output_dir
        
        self.train_args["project"] = hydra_output_dir
        self.train_args["name"] = "ultralytics_files"
        
        print(f"[{self.__class__.__name__}] Starting training engine with args: {self.train_args}")
        self.model.train(**self.train_args)