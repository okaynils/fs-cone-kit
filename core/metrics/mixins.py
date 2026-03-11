from collections.abc import Iterable

from core.metrics.base import BaseMetric


class MetricsMixin:
    """Shared helpers for components that consume configured metrics."""

    def __init__(self, metrics: Iterable[BaseMetric] | None = None):
        self.metrics: list[BaseMetric] = []
        self.set_metrics(metrics)

    def set_metrics(self, metrics: Iterable[BaseMetric] | None = None):
        self.metrics = list(metrics or [])

    def collect_metrics(self, trainer, event: str) -> dict[str, float | int]:
        payload: dict[str, float | int] = {}

        for metric in self.metrics:
            metric_payload = metric.collect(trainer=trainer, event=event)
            overlapping_keys = payload.keys() & metric_payload.keys()
            if overlapping_keys:
                collision_keys = ", ".join(sorted(overlapping_keys))
                raise ValueError(f"Duplicate metric keys configured for '{event}': {collision_keys}")
            payload.update(metric_payload)

        return payload
