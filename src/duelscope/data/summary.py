"""Dataset summary statistics."""

from __future__ import annotations

import numpy as np

from duelscope.types import Dataset


def summarize(ds: Dataset) -> dict:
    """Return image/box counts, per-class counts, size stats and issue counts."""
    counts = np.zeros(len(ds.names), dtype=int)
    for r in ds.records:
        np.add.at(counts, r.gt.classes, 1)
    widths = np.array([r.width for r in ds.records])
    heights = np.array([r.height for r in ds.records])
    areas = np.concatenate([r.gt.areas for r in ds.records]) if ds.records else np.zeros(0)
    levels: dict[str, int] = {}
    for i in ds.issues:
        levels[i.level] = levels.get(i.level, 0) + 1
    return {
        "name": ds.name,
        "split": ds.split,
        "images": len(ds),
        "boxes": int(counts.sum()),
        "empty_images": sum(1 for r in ds.records if len(r.gt) == 0),
        "classes": {n: int(c) for n, c in zip(ds.names, counts, strict=True)},
        "width": _stats(widths),
        "height": _stats(heights),
        "box_area_px": _stats(areas),
        "issues": levels,
    }


def _stats(a: np.ndarray) -> dict[str, float]:
    if len(a) == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}
