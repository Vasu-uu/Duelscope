"""Dataset loading and validation."""

from __future__ import annotations

from pathlib import Path

from duelscope.data.base import DatasetError, content_hash, subset
from duelscope.data.coco import load_coco
from duelscope.data.summary import summarize
from duelscope.data.yolo import load_yolo
from duelscope.types import Dataset

__all__ = ["DatasetError", "content_hash", "load_dataset", "subset", "summarize"]


def load_dataset(
    path: str | Path,
    split: str = "val",
    fmt: str = "auto",
    images_dir: str | Path | None = None,
) -> Dataset:
    """Load a dataset; fmt is 'yolo', 'coco' or 'auto' (a .json file means COCO)."""
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Dataset path '{p}' does not exist. Check the --data argument.")
    if fmt == "auto":
        fmt = "coco" if p.suffix == ".json" else "yolo"
    if fmt == "coco":
        return load_coco(p, images_dir)
    if fmt == "yolo":
        return load_yolo(p, split)
    raise DatasetError(f"Unknown dataset format '{fmt}'. Use 'yolo', 'coco' or 'auto'.")
