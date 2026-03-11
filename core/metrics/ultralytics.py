from typing import Any

from core.metrics.base import BaseMetric


class UltralyticsTrainLossMetric(BaseMetric):
    def __init__(self, prefix: str = "train", events: tuple[str, ...] = ("on_train_epoch_end",)):
        super().__init__(events=events)
        self.prefix = prefix

    def compute(self, trainer: Any) -> dict[str, Any]:
        if getattr(trainer, "tloss", None) is None or not hasattr(trainer, "label_loss_items"):
            return {}
        return dict(trainer.label_loss_items(trainer.tloss, prefix=self.prefix))


class UltralyticsLearningRateMetric(BaseMetric):
    def __init__(self, events: tuple[str, ...] = ("on_train_epoch_end",)):
        super().__init__(events=events)

    def compute(self, trainer: Any) -> dict[str, Any]:
        return dict(getattr(trainer, "lr", {}) or {})


class UltralyticsValidationMetric(BaseMetric):
    def __init__(self, events: tuple[str, ...] = ("on_fit_epoch_end",)):
        super().__init__(events=events)

    def compute(self, trainer: Any) -> dict[str, Any]:
        return dict(getattr(trainer, "metrics", {}) or {})


class UltralyticsFitnessMetric(BaseMetric):
    def __init__(
        self,
        fitness_name: str = "fitness",
        best_fitness_name: str = "best_fitness",
        include_best_fitness: bool = True,
        events: tuple[str, ...] = ("on_fit_epoch_end",),
    ):
        super().__init__(events=events)
        self.fitness_name = fitness_name
        self.best_fitness_name = best_fitness_name
        self.include_best_fitness = include_best_fitness

    def compute(self, trainer: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        if getattr(trainer, "fitness", None) is not None:
            payload[self.fitness_name] = trainer.fitness

        if self.include_best_fitness and getattr(trainer, "best_fitness", None) is not None:
            payload[self.best_fitness_name] = trainer.best_fitness

        return payload
