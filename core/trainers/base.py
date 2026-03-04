from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    Abstract base class for all model trainers in the pipeline.
    """
    def __init__(self, **kwargs):
        self.model = None

    @abstractmethod
    def setup(self, model_name: str):
        """
        Initialize the model, load weights, attach callbacks, etc.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Execute the training loop.
        """
        pass