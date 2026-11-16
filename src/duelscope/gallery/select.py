"""Per-image scoring and selection logic for failure galleries (no rendering here)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from duelscope.metrics.errors import IGNORE, TP, classify_image
from duelscope.types import Dataset, Detections, ImageRecord

FAIL_SCORE = 1.0  # roughly "at least one clear error"
OK_SCORE = 0.5  # roughly "no real error"


@dataclass
class ImageAnalysis:
    """What a detector got right and wrong on one image at an operating confidence."""

    tp: np.ndarray  # indices into the confidence-filtered detections
    fp: np.ndarray
    missed: np.ndarray  # indices into the ground truth
    det: Detections  # the confidence-filtered detections
    score: float  # n_missed + sum of confidences of false positives
    class_score: np.ndarray  # same score restricted to each class


def analyze_image(rec: ImageRecord, det: Detections, conf: float, n_classes: int) -> ImageAnalysis:
    """Classify detections >= conf as TP/FP and find missed ground truth (IoU 0.5)."""
    d = det.filter(conf)
    types, _ref, gt_score = classify_image(rec.gt, d)
    fp_mask = (types != TP) & (types != IGNORE)
    missed = np.nonzero((gt_score < 0) & ~rec.gt.iscrowd)[0]
    cs = np.zeros(n_classes)
    np.add.at(cs, d.classes[fp_mask], d.scores[fp_mask])
    np.add.at(cs, rec.gt.classes[missed], 1.0)
    return ImageAnalysis(
        tp=np.nonzero(types == TP)[0],
        fp=np.nonzero(fp_mask)[0],
        missed=missed,
        det=d,
        score=float(cs.sum()),
        class_score=cs,
    )


def analyze_dataset(ds: Dataset, preds: list[Detections], conf: float) -> list[ImageAnalysis]:
    """Analyze every image for one detector."""
    n = len(ds.names)
    return [analyze_image(r, p, conf, n) for r, p in zip(ds.records, preds, strict=True)]


def worst_images(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k highest scores (ties broken by lowest index); zero scores excluded."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [i for i in order[:k] if scores[i] > 0]


def a_ok_b_fails(
    score_a: np.ndarray,
    score_b: np.ndarray,
    k: int,
    ok: float = OK_SCORE,
    fail: float = FAIL_SCORE,
) -> list[int]:
    """Images where A has no real error and B fails, biggest gap first."""
    idx = [i for i in range(len(score_a)) if score_a[i] <= ok and score_b[i] >= fail]
    idx.sort(key=lambda i: (-(score_b[i] - score_a[i]), i))
    return idx[:k]


def all_models_fail(scores: dict[str, np.ndarray], k: int, fail: float = FAIL_SCORE) -> list[int]:
    """Images where every model scores >= fail (hard cases or label errors), worst mean first."""
    mat = np.stack(list(scores.values()))
    idx = [i for i in range(mat.shape[1]) if (mat[:, i] >= fail).all()]
    idx.sort(key=lambda i: (-mat[:, i].mean(), i))
    return idx[:k]


def worst_per_class(
    analyses: list[ImageAnalysis], class_ids: list[int], k: int
) -> dict[int, list[int]]:
    """Top-k worst images for each requested class (classes with no errors are omitted)."""
    out: dict[int, list[int]] = {}
    for c in class_ids:
        s = np.array([a.class_score[c] for a in analyses])
        top = worst_images(s, k)
        if top:
            out[c] = top
    return out
