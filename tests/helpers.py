"""Shared test helpers: a scriptable mock detector."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from duelscope.adapters.base import Detector, DetectorConfig, ImageInput
from duelscope.types import Detections


class MockDetector(Detector):
    """Returns whatever `fn(path)` says; no model needed."""

    framework = "mock"

    def __init__(
        self,
        fn: Callable[[Path], Detections],
        name: str = "mock",
        names: list[str] | None = None,
        cfg: DetectorConfig | None = None,
    ) -> None:
        super().__init__(cfg)
        self._fn, self._name, self._names = fn, name, names
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def class_names(self) -> list[str] | None:
        return self._names

    def load(self) -> None:
        return None

    def predict(self, images: list[ImageInput]) -> list[Detections]:
        self.calls += len(images)
        self.last_timing = {"preprocess": 0.1, "inference": 1.0, "postprocess": 0.1}
        return [self._fn(Path(str(i))) for i in images]
