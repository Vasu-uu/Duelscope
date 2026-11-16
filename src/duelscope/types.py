"""Core in-memory types shared by every module. Boxes are absolute-pixel xyxy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _f32(a: object) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).reshape(-1, 4)


@dataclass(frozen=True)
class GroundTruth:
    """Ground-truth annotations of one image."""

    boxes: np.ndarray  # (N, 4) float32 xyxy
    classes: np.ndarray  # (N,) int64
    iscrowd: np.ndarray  # (N,) bool

    @classmethod
    def make(cls, boxes: object, classes: object, iscrowd: object | None = None) -> GroundTruth:
        """Build from array-likes, normalising dtypes and shapes."""
        b = _f32(boxes)
        c = np.asarray(classes, dtype=np.int64).reshape(-1)
        crowd = (
            np.zeros(len(c), dtype=bool)
            if iscrowd is None
            else np.asarray(iscrowd, dtype=bool).reshape(-1)
        )
        if not (len(b) == len(c) == len(crowd)):
            raise ValueError("boxes, classes and iscrowd must have the same length")
        return cls(b, c, crowd)

    @classmethod
    def empty(cls) -> GroundTruth:
        """A ground truth with no boxes."""
        return cls.make(np.zeros((0, 4)), np.zeros(0))

    @property
    def areas(self) -> np.ndarray:
        """Box areas in pixels."""
        b = self.boxes
        return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    def __len__(self) -> int:
        return len(self.classes)


@dataclass(frozen=True)
class Detections:
    """Predictions of one detector on one image."""

    boxes: np.ndarray  # (N, 4) float32 xyxy
    scores: np.ndarray  # (N,) float32
    classes: np.ndarray  # (N,) int64

    @classmethod
    def make(cls, boxes: object, scores: object, classes: object) -> Detections:
        """Build from array-likes, normalising dtypes and shapes."""
        b = _f32(boxes)
        s = np.asarray(scores, dtype=np.float32).reshape(-1)
        c = np.asarray(classes, dtype=np.int64).reshape(-1)
        if not (len(b) == len(s) == len(c)):
            raise ValueError("boxes, scores and classes must have the same length")
        return cls(b, s, c)

    @classmethod
    def empty(cls) -> Detections:
        """A detection set with no boxes."""
        return cls.make(np.zeros((0, 4)), np.zeros(0), np.zeros(0))

    def filter(self, min_conf: float) -> Detections:
        """Keep detections with score >= min_conf."""
        keep = self.scores >= min_conf
        return Detections(self.boxes[keep], self.scores[keep], self.classes[keep])

    def __len__(self) -> int:
        return len(self.scores)


@dataclass(frozen=True)
class ImageRecord:
    """One dataset image with its ground truth."""

    image_id: int
    path: Path
    width: int
    height: int
    gt: GroundTruth


@dataclass(frozen=True)
class Issue:
    """A dataset validation finding."""

    level: str  # "error" | "warning" | "info"
    kind: str
    path: str
    message: str


@dataclass
class Dataset:
    """Uniform in-memory dataset, whatever the on-disk format was."""

    name: str
    split: str
    names: list[str]
    records: list[ImageRecord]
    root: Path
    issues: list[Issue] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)
