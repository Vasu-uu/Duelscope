"""Box geometry helpers (xyxy pixel boxes)."""

from __future__ import annotations

import numpy as np


def box_area(b: np.ndarray) -> np.ndarray:
    """Areas of (N,4) xyxy boxes."""
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection area between (N,4) and (M,4) boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    inter = intersection(a, b)
    if inter.size == 0:
        return inter
    union = box_area(a)[:, None] + box_area(b)[None] - inter
    return inter / np.maximum(union, 1e-9)


def iou_with_crowd(dt: np.ndarray, gt: np.ndarray, crowd: np.ndarray) -> np.ndarray:
    """IoU matrix where crowd ground-truth columns use intersection over detection area."""
    ious = box_iou(dt, gt)
    if crowd.any() and ious.size:
        inter = intersection(dt, gt[crowd])
        ious[:, crowd] = inter / np.maximum(box_area(dt)[:, None], 1e-9)
    return ious
