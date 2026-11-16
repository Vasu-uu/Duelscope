"""The Detector interface every adapter implements."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from duelscope.types import Detections

ImageInput = Path | np.ndarray


@dataclass(frozen=True)
class DetectorConfig:
    """Inference settings shared by all adapters."""

    imgsz: int = 640
    conf: float = 0.001
    iou: float = 0.6
    device: str = "auto"  # auto | cpu | cuda
    max_det: int = 300


class AdapterError(Exception):
    """Raised when a detector cannot be created or run. Messages include a suggested fix."""


def resolve_device(device: str) -> str:
    """Map 'auto' to 'cuda' when a GPU is usable, else 'cpu'."""
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if device not in {"cpu", "cuda"}:
        raise AdapterError(f"Unknown device '{device}'. Use cpu, cuda or auto.")
    return device


def free_gpu() -> None:
    """Run garbage collection and release cached CUDA memory when torch is present."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_rgb(image: ImageInput) -> np.ndarray:
    """Load an image path or pass through an RGB uint8 HxWx3 array."""
    if isinstance(image, np.ndarray):
        return image
    with Image.open(image) as im:
        return np.asarray(im.convert("RGB"))


def file_mb(path: Path | None) -> float | None:
    """File size in MiB, or None when there is no local file."""
    return round(path.stat().st_size / 2**20, 2) if path and path.is_file() else None


class Detector(ABC):
    """Base class for detectors. ndarray inputs are RGB uint8; boxes returned are xyxy pixels."""

    framework = "unknown"

    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        self.cfg = cfg or DetectorConfig()
        self.weights_path: Path | None = None
        self.last_timing: dict[str, float] | None = None  # per-image ms: pre/infer/post

    @property
    def name(self) -> str:
        """Short display name used in reports."""
        return type(self).__name__

    @property
    def class_names(self) -> list[str] | None:
        """Model class names indexed by class id, or None when unknown."""
        return None

    @property
    def num_params(self) -> int | None:
        """Parameter count when the framework exposes it."""
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        """Framework, weights size, input size and parameter count."""
        return {
            "framework": self.framework,
            "weights_mb": file_mb(self.weights_path),
            "input_size": self.cfg.imgsz,
            "params": self.num_params,
        }

    @abstractmethod
    def load(self) -> None:
        """Load weights onto the chosen device. Must be idempotent."""

    def close(self) -> None:  # noqa: B027
        """Release the model so memory can be re-measured. Default: nothing to release."""

    @abstractmethod
    def predict(self, images: list[ImageInput]) -> list[Detections]:
        """Detect objects; must return one Detections per input image, in order."""

    def warmup(self, runs: int = 2) -> None:
        """Run a few dummy inferences so kernels and caches are initialised."""
        self.load()
        blank = np.full((self.cfg.imgsz, self.cfg.imgsz, 3), 114, dtype=np.uint8)
        for _ in range(runs):
            self.predict([blank])

    def fingerprint(self) -> str:
        """Identity used in the prediction cache key (weights hash, else the name)."""
        h = hashlib.sha256()
        if self.weights_path and self.weights_path.is_file():
            with self.weights_path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        else:
            h.update(self.name.encode())
        return h.hexdigest()[:16]
