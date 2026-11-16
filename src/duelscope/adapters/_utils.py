"""Numpy helpers for adapters: letterbox, NMS and YOLO output decoding."""

from __future__ import annotations

import numpy as np
from PIL import Image


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad to size x size. Returns (image, ratio, (padx, pady))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py : py + nh, px : px + nw] = resized
    return out, r, (float(px), float(py))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by descending score."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float
) -> np.ndarray:
    """Class-aware NMS using the coordinate-offset trick."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    offset = classes.astype(np.float32)[:, None] * (boxes.max() + 1)
    return nms(boxes + offset, scores, iou_thr)


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    """Convert centre/size boxes to corner format."""
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def decode_yolo(
    out: np.ndarray, kind: str, conf: float, iou: float, max_det: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw YOLO output tensor (batch dim removed or size 1) into boxes/scores/classes.

    kind: 'yolov8' (4+nc, N), 'yolov5' (N, 5+nc with objectness), or 'end2end' (N, 6 xyxy).
    Boxes are in network-input pixel space.
    """
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    if kind == "end2end":
        keep = out[:, 4] >= conf
        o = out[keep][:max_det]
        return o[:, :4], o[:, 4], o[:, 5].astype(np.int64)
    if kind == "yolov8":
        p = out.T
        boxes_c, sc = p[:, :4], p[:, 4:]
    elif kind == "yolov5":
        boxes_c, sc = out[:, :4], out[:, 5:] * out[:, 4:5]
    else:
        raise ValueError(f"Unknown YOLO decoder '{kind}'. Use yolov8, yolov5 or end2end.")
    rows, cls = np.nonzero(sc > conf)  # multi-label, as in Ultralytics validation
    scores = sc[rows, cls]
    if len(scores) > 30000:
        top = np.argsort(-scores)[:30000]
        rows, cls, scores = rows[top], cls[top], scores[top]
    boxes = cxcywh_to_xyxy(boxes_c[rows])
    keep = batched_nms(boxes, scores, cls, iou)[:max_det]
    return boxes[keep], scores[keep], cls[keep].astype(np.int64)
