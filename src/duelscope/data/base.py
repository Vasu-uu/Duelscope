"""Dataset helpers shared by all loaders."""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from duelscope.types import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class DatasetError(Exception):
    """Raised for unusable datasets. The message always includes a suggested fix."""


def image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) reading only the image header."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            return im.size
    except Exception as exc:
        raise DatasetError(
            f"Cannot read image '{path}': {exc}. Remove or repair the file."
        ) from exc


def subset(ds: Dataset, max_images: int | None, seed: int = 0) -> Dataset:
    """Deterministic random subset of at most max_images images (order preserved)."""
    if max_images is None or max_images >= len(ds):
        return ds
    if max_images <= 0:
        raise DatasetError("--max-images must be a positive integer.")
    idx = sorted(random.Random(seed).sample(range(len(ds)), max_images))
    return replace(ds, records=[ds.records[i] for i in idx])


def content_hash(ds: Dataset) -> str:
    """Stable hash of class names, image names, sizes and annotations (not pixel data)."""
    h = hashlib.sha256()
    h.update("|".join(ds.names).encode())
    for r in ds.records:
        h.update(f"{r.path.name}:{r.width}x{r.height}".encode())
        h.update(np.ascontiguousarray(r.gt.boxes).tobytes())
        h.update(np.ascontiguousarray(r.gt.classes).tobytes())
    return h.hexdigest()[:16]
