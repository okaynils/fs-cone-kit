from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from numbers import Number
from typing import Any


class BaseMetric(ABC):
    """Base class for metrics extracted from trainer callback state."""

    def __init__(self, events: Iterable[str] | None = None):
        self.events = tuple(events or ())

    def supports(self, event: str) -> bool:
        return not self.events or event in self.events

    @abstractmethod
    def compute(self, trainer: Any) -> Mapping[str, Any]:
        """Build a metric payload from the current trainer state."""

    def collect(self, trainer: Any, event: str) -> dict[str, float | int]:
        if not self.supports(event):
            return {}

        payload: dict[str, float | int] = {}
        for name, value in self.compute(trainer).items():
            normalized_value = self._normalize_value(value)
            if normalized_value is not None:
                payload[name] = normalized_value
        return payload

    @staticmethod
    def _normalize_value(value: Any) -> float | int | None:
        if value is None:
            return None

        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, Number):
            return int(value) if isinstance(value, int) else float(value)

        return None
