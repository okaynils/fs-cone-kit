from abc import ABC, abstractmethod

from core.metrics.mixins import MetricsMixin


class BaseLogger(MetricsMixin, ABC):
    uses_mlflow = False

    def __init__(self, metrics=None, **kwargs):
        super().__init__(metrics=metrics)

    @abstractmethod
    def setup(
        self,
        val_image_dir: str,
        class_map: dict,
        experiment_name: str,
        run_name: str,
        val_image_paths: list[str] | None = None,
        class_colors: dict[int, tuple[int, int, int]] | None = None,
    ):
        """Initialize the logger with dataset, run, and metric configuration."""
        pass

    @abstractmethod
    def get_callbacks(self) -> dict:
        """Returns a dictionary of {event_name: callback_function}"""
        pass
