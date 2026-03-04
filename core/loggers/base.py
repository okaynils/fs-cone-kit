from abc import ABC, abstractmethod

class BaseLogger(ABC):
    @abstractmethod
    def setup(self, val_image_dir: str, class_map: dict):
        """Initializes the logger with necessary dataset info."""
        pass

    @abstractmethod
    def get_callbacks(self) -> dict:
        """Returns a dictionary of {event_name: callback_function}"""
        pass