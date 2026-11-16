"""Simplified TIDE-style error taxonomy with oracle mAP-gain estimates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from duelscope.geometry import iou_with_crowd
from duelscope.metrics.coco import evaluate
from duelscope.types import Dataset, Detections, GroundTruth

TP, CLS, LOC, BOTH, DUPE, BKG, IGNORE = range(7)
ERROR_NAMES = {CLS: "cls", LOC: "loc", BOTH: "both", DUPE: "dupe", BKG: "bkg"}
LABELS = {
    "cls": "Classification",
    "loc": "Localization",
    "both": "Cls + Loc",
    "dupe": "Duplicate",
    "bkg": "Background FP",
    "miss": "Missed GT",
}
FG_IOU, BG_IOU = 0.5, 0.1


def classify_image(gt: GroundTruth, det: Detections) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify each detection (in input order) and find which GTs were never matched.

    Returns (types (D,), ref_gt (D,) index of matched/related GT or -1, matched_score (G,) the
    score of the detection that matched each GT or -1).
    """
    n_d, n_g = len(det), len(gt)
    types = np.full(n_d, BKG, dtype=np.int8)
    ref = np.full(n_d, -1, dtype=np.int64)
    gt_score = np.full(n_g, -1.0)
    if n_d == 0:
        return types, ref, gt_score
    ious = iou_with_crowd(det.boxes, gt.boxes, gt.iscrowd) if n_g else np.zeros((n_d, 0))
    gt_matched = np.zeros(n_g, dtype=bool)
    for d in np.argsort(-det.scores, kind="mergesort"):
        row = ious[d]
        same = (gt.classes == det.classes[d]) & ~gt.iscrowd
        cand = same & ~gt_matched & (row >= FG_IOU)
        if cand.any():
            g = int(np.argmax(np.where(cand, row, -1)))
            gt_matched[g] = True
            gt_score[g] = det.scores[d]
            types[d], ref[d] = TP, g
            continue
        crowd_same = (gt.classes == det.classes[d]) & gt.iscrowd
        if crowd_same.any() and row[crowd_same].max() >= FG_IOU:
            types[d] = IGNORE
            continue
        other = (gt.classes != det.classes[d]) & ~gt.iscrowd
        ms = np.where(same, row, -1.0)
        mo = np.where(other, row, -1.0)
        best_s = ms.max() if n_g else -1.0
        best_o = mo.max() if n_g else -1.0
        if best_s >= FG_IOU:
            types[d], ref[d] = DUPE, int(np.argmax(ms))
        elif best_o >= FG_IOU:
            types[d], ref[d] = CLS, int(np.argmax(mo))
        elif best_s >= BG_IOU:
            types[d], ref[d] = LOC, int(np.argmax(ms))
        elif best_o >= BG_IOU:
            types[d], ref[d] = BOTH, int(np.argmax(mo))
    return types, ref, gt_score


def _oracle(ds: Dataset, preds: list[Detections], info: list, fix: str) -> float:
    """mAP@0.5 after fixing one error type for every image."""
    new_preds, new_records = [], []
    for rec, det, (types, ref, gt_score) in zip(ds.records, preds, info, strict=True):
        boxes, classes, scores = det.boxes.copy(), det.classes.copy(), det.scores
        keep = np.ones(len(det), dtype=bool)
        gt = rec.gt
        if fix == "cls":
            sel = types == CLS
            classes[sel] = gt.classes[ref[sel]]
        elif fix == "loc":
            sel = types == LOC
            boxes[sel] = gt.boxes[ref[sel]]
        elif fix in ("both", "dupe", "bkg"):
            keep = types != {"both": BOTH, "dupe": DUPE, "bkg": BKG}[fix]
        elif fix == "miss":
            drop = (gt_score < 0) & ~gt.iscrowd
            gt = GroundTruth.make(gt.boxes[~drop], gt.classes[~drop], gt.iscrowd[~drop])
        new_preds.append(Detections(boxes[keep], scores[keep], classes[keep]))
        new_records.append(replace(rec, gt=gt))
    return evaluate(replace(ds, records=new_records), new_preds, full=False).map50


def analyze_errors(
    ds: Dataset, preds: list[Detections], conf: float = 0.25, with_oracle: bool = True
) -> dict:
    """Error taxonomy at operating threshold `conf` plus oracle mAP@0.5 gains per error type."""
    info = [classify_image(r.gt, d) for r, d in zip(ds.records, preds, strict=True)]
    names = list(ERROR_NAMES.values())
    counts = dict.fromkeys(["tp", *names, "miss"], 0)
    per_class = {n: dict.fromkeys([*names, "miss"], 0) for n in ds.names}
    for rec, det, (types, _ref, gt_score) in zip(ds.records, preds, info, strict=True):
        hit = det.scores >= conf
        counts["tp"] += int(((types == TP) & hit).sum())
        for t, nm in ERROR_NAMES.items():
            sel = (types == t) & hit
            counts[nm] += int(sel.sum())
            for c in det.classes[sel]:
                per_class[ds.names[c]][nm] += 1
        missed = ~rec.gt.iscrowd & ((gt_score < 0) | (gt_score < conf))
        counts["miss"] += int(missed.sum())
        for c in rec.gt.classes[missed]:
            per_class[ds.names[c]]["miss"] += 1
    err_keys = [*names, "miss"]
    total = sum(counts[k] for k in err_keys)
    share = {k: (counts[k] / total if total else 0.0) for k in err_keys}
    delta = {}
    if with_oracle:
        base = evaluate(ds, preds, full=False).map50
        delta = {k: max(_oracle(ds, preds, info, k) - base, 0.0) for k in err_keys}
    return {
        "conf": conf,
        "counts": counts,
        "total_errors": total,
        "share": share,
        "delta_map50": delta,
        "per_class": per_class,
        "labels": LABELS,
    }
