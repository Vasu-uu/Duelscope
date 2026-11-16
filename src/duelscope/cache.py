"""On-disk cache of raw detector predictions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from duelscope.types import Detections

DEFAULT_DIR = Path(".duelscope_cache")


def cache_key(model_fp: str, dataset_hash: str, settings: dict) -> str:
    """Key from model identity, dataset content and inference settings."""
    blob = "|".join([model_fp, dataset_hash, *(f"{k}={settings[k]}" for k in sorted(settings))])
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class PredictionCache:
    """Stores per-image detections of one (model, dataset, settings) run as a single .npz."""

    def __init__(self, directory: Path | str = DEFAULT_DIR, enabled: bool = True) -> None:
        self.dir = Path(directory)
        self.enabled = enabled

    def _path(self, key: str) -> Path:
        return self.dir / f"preds_{key}.npz"

    def load(self, key: str, n_images: int) -> list[Detections] | None:
        """Return cached predictions or None on a miss (or an unreadable file)."""
        p = self._path(key)
        if not self.enabled or not p.is_file():
            return None
        try:
            z = np.load(p)
            off = z["offsets"]
            if len(off) != n_images + 1:
                return None
            return [
                Detections(
                    z["boxes"][off[i] : off[i + 1]],
                    z["scores"][off[i] : off[i + 1]],
                    z["classes"][off[i] : off[i + 1]],
                )
                for i in range(n_images)
            ]
        except (OSError, KeyError, ValueError):
            return None

    def save(self, key: str, preds: list[Detections]) -> None:
        """Write predictions; silently skipped when caching is disabled."""
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        off = np.cumsum([0] + [len(p) for p in preds])
        cat = lambda a, shape, dt: (  # noqa: E731
            np.concatenate(a) if a else np.zeros(shape, dtype=dt)
        )
        np.savez_compressed(
            self._path(key),
            offsets=off,
            boxes=cat([p.boxes for p in preds], (0, 4), np.float32),
            scores=cat([p.scores for p in preds], (0,), np.float32),
            classes=cat([p.classes for p in preds], (0,), np.int64),
        )
