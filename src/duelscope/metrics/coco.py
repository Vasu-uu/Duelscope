"""COCO-style mAP (101-point interpolation) with per-class AP, PR curves and best-F1 point."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from duelscope.geometry import iou_with_crowd
from duelscope.types import Dataset, Detections

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
CONF_GRID = np.linspace(0.0, 1.0, 1001)
AREAS = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, 1e10),
}


@dataclass
class EvalResult:
    """Accuracy metrics for one detector on one dataset."""

    map50: float
    map5095: float
    map_small: float | None  # None when the dataset has no objects in that size range
    map_medium: float | None
    map_large: float | None
    precision: float  # macro-average at best-F1 confidence, IoU 0.5
    recall: float
    f1: float
    best_conf: float
    per_class: list[dict]
    pr_recall: list[float]
    pr_mean: list[float]  # mean precision over classes at IoU 0.5
    pr_per_class: list[list[float]]
    n_images: int
    n_gt: int
    errors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        return dict(self.__dict__)


def _empty_store() -> dict:
    return {"scores": [], "dtm": [], "dtig": [], "npig": 0}


def _match(
    ious: np.ndarray, gt_ig: np.ndarray, crowd: np.ndarray, thrs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """COCO greedy matching. Dets are already sorted by score; gts sorted non-ignored first.

    Returns (dtm (T,D) bool matched, dtig (T,D) bool ignored-because-matched-to-ignored-gt).
    """
    n_t, n_d, n_g = len(thrs), ious.shape[0], ious.shape[1]
    dtm = np.zeros((n_t, n_d), dtype=bool)
    dtig = np.zeros((n_t, n_d), dtype=bool)
    gtm = np.zeros((n_t, n_g), dtype=bool)
    for ti, t in enumerate(thrs):
        for d in range(n_d):
            best = min(t, 1 - 1e-10)
            m = -1
            for g in range(n_g):
                if gtm[ti, g] and not crowd[g]:
                    continue
                if m > -1 and not gt_ig[m] and gt_ig[g]:
                    break
                if ious[d, g] < best:
                    continue
                best, m = ious[d, g], g
            if m == -1:
                continue
            dtig[ti, d] = gt_ig[m]
            dtm[ti, d] = True
            gtm[ti, m] = True
    return dtm, dtig


def _accumulate(store: dict, n_t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (precision (T,R), final recall (T,), ranked [score,tp,fp] curves for T=0)."""
    npig = store["npig"]
    prec = np.zeros((n_t, len(REC_THRS)))
    if npig == 0:
        return prec, np.zeros(n_t), None
    if not store["scores"]:
        return prec, np.zeros(n_t), np.zeros((0, 3))
    scores = np.concatenate(store["scores"])
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    dtm = np.concatenate(store["dtm"], axis=1)[:, order]
    dtig = np.concatenate(store["dtig"], axis=1)[:, order]
    tps = np.cumsum(dtm & ~dtig, axis=1).astype(float)
    fps = np.cumsum(~dtm & ~dtig, axis=1).astype(float)
    rec_final = np.zeros(n_t)
    for t in range(n_t):
        rc = tps[t] / npig
        pr = tps[t] / np.maximum(tps[t] + fps[t], np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        idx = np.searchsorted(rc, REC_THRS, side="left")
        ok = idx < len(pr)
        prec[t, ok] = pr[idx[ok]]
        rec_final[t] = rc[-1] if len(rc) else 0.0
    return prec, rec_final, np.stack([scores, tps[0], fps[0]], axis=1)


def _mean_valid(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate(
    ds: Dataset,
    preds: list[Detections],
    max_dets: int = 100,
    full: bool = True,
) -> EvalResult:
    """Evaluate predictions (dataset class ids, any confidence) against the dataset.

    full=False only computes IoU 0.5 over all areas (used by the error-oracle reruns).
    """
    if len(preds) != len(ds):
        raise ValueError(f"Got {len(preds)} prediction sets for {len(ds)} images.")
    n_k = len(ds.names)
    thrs = IOU_THRS if full else IOU_THRS[:1]
    areas = AREAS if full else {"all": AREAS["all"]}
    stores = {(k, a): _empty_store() for k in range(n_k) for a in areas}
    for rec, det in zip(ds.records, preds, strict=True):
        gt = rec.gt
        for k in range(n_k):
            gi = np.nonzero(gt.classes == k)[0]
            di = np.nonzero(det.classes == k)[0]
            if len(gi) == 0 and len(di) == 0:
                continue
            di = di[np.argsort(-det.scores[di], kind="mergesort")][:max_dets]
            gboxes, crowd, garea = gt.boxes[gi], gt.iscrowd[gi], gt.areas[gi]
            ious_all = iou_with_crowd(det.boxes[di], gboxes, crowd)
            darea = (det.boxes[di, 2] - det.boxes[di, 0]) * (det.boxes[di, 3] - det.boxes[di, 1])
            for a, (lo, hi) in areas.items():
                st = stores[(k, a)]
                gt_ig = crowd | (garea < lo) | (garea > hi)
                st["npig"] += int((~gt_ig).sum())
                if len(di) == 0:
                    continue
                order = np.argsort(gt_ig, kind="mergesort")
                dtm, dtig = _match(ious_all[:, order], gt_ig[order], crowd[order], thrs)
                out_of_range = (darea < lo) | (darea > hi)
                dtig |= ~dtm & out_of_range[None, :]
                st["scores"].append(det.scores[di])
                st["dtm"].append(dtm)
                st["dtig"].append(dtig)

    n_t = len(thrs)
    ap = {a: np.full((n_k, n_t), -1.0) for a in areas}
    pr_curves = np.zeros((n_k, len(REC_THRS)))
    rank = {}
    valid_k = []
    for k in range(n_k):
        for a in areas:
            prec, _, curve = _accumulate(stores[(k, a)], n_t)
            if stores[(k, a)]["npig"] > 0:
                ap[a][k] = prec.mean(axis=1)
            if a == "all":
                pr_curves[k] = prec[0]
                rank[k] = curve
        if stores[(k, "all")]["npig"] > 0:
            valid_k.append(k)

    def mean_ap(a: str, col: slice | int) -> float | None:
        v = [
            ap[a][k, col].mean() if isinstance(col, slice) else ap[a][k, col]
            for k in range(n_k)
            if ap[a][k, 0] >= 0
        ]
        return _mean_valid(v)

    best_conf, prec_o, rec_o, f1_o, p_k, r_k = _best_f1(stores, rank, valid_k, n_k)
    per_class = []
    for k, name in enumerate(ds.names):
        n_gt = stores[(k, "all")]["npig"]
        per_class.append(
            {
                "name": name,
                "n_gt": int(n_gt),
                "ap50": float(ap["all"][k, 0]) if n_gt else None,
                "ap5095": float(ap["all"][k].mean()) if n_gt and full else None,
                "precision": p_k[k] if n_gt else None,
                "recall": r_k[k] if n_gt else None,
            }
        )
    mean_pr = pr_curves[valid_k].mean(axis=0) if valid_k else np.zeros(len(REC_THRS))
    return EvalResult(
        map50=mean_ap("all", 0) or 0.0,
        map5095=(mean_ap("all", slice(None)) or 0.0) if full else 0.0,
        map_small=mean_ap("small", slice(None)) if full else None,
        map_medium=mean_ap("medium", slice(None)) if full else None,
        map_large=mean_ap("large", slice(None)) if full else None,
        precision=prec_o,
        recall=rec_o,
        f1=f1_o,
        best_conf=best_conf,
        per_class=per_class,
        pr_recall=REC_THRS.tolist(),
        pr_mean=mean_pr.tolist(),
        pr_per_class=[pr_curves[k].tolist() for k in range(n_k)],
        n_images=len(ds),
        n_gt=int(sum(stores[(k, "all")]["npig"] for k in range(n_k))),
    )


def _best_f1(
    stores: dict, rank: dict, valid_k: list[int], n_k: int
) -> tuple[float, float, float, float, list, list]:
    """Pick the confidence maximising macro F1 (IoU 0.5); return macro P/R/F1 and per-class P/R."""
    if not valid_k:
        return 0.0, 0.0, 0.0, 0.0, [None] * n_k, [None] * n_k
    p_curve = np.zeros((n_k, len(CONF_GRID)))
    r_curve = np.zeros((n_k, len(CONF_GRID)))
    for k in valid_k:
        curve = rank[k]
        npig = stores[(k, "all")]["npig"]
        if curve is None or len(curve) == 0:
            continue
        # number of detections with score >= conf
        n_ge = np.searchsorted(-curve[:, 0], -CONF_GRID, side="right")
        idx = np.maximum(n_ge - 1, 0)
        tp = np.where(n_ge > 0, curve[idx, 1], 0.0)
        fp = np.where(n_ge > 0, curve[idx, 2], 0.0)
        p_curve[k] = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 1.0 * (n_ge == 0))
        r_curve[k] = tp / npig
    f1 = 2 * p_curve * r_curve / np.maximum(p_curve + r_curve, 1e-12)
    mean_f1 = f1[valid_k].mean(axis=0)
    i = int(np.argmax(mean_f1))
    p_k = [float(p_curve[k, i]) if k in valid_k else None for k in range(n_k)]
    r_k = [float(r_curve[k, i]) if k in valid_k else None for k in range(n_k)]
    return (
        float(CONF_GRID[i]),
        float(p_curve[valid_k, i].mean()),
        float(r_curve[valid_k, i].mean()),
        float(mean_f1[i]),
        p_k,
        r_k,
    )
