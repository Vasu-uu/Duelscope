"""Assemble the failure galleries for all models and write thumbnails to disk."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np

from duelscope.gallery.render import render_thumbnail
from duelscope.gallery.select import (
    ImageAnalysis,
    a_ok_b_fails,
    all_models_fail,
    analyze_dataset,
    worst_images,
    worst_per_class,
)
from duelscope.types import Dataset, Detections


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


class _Thumbs:
    """Renders each (model, image) thumbnail at most once."""

    def __init__(self, ds: Dataset, out_dir: Path, size: int, quality: int) -> None:
        self.ds, self.dir, self.size, self.quality = ds, out_dir, size, quality
        self.done: dict[tuple[str, int], str] = {}

    def item(self, model: str, i: int, a: ImageAnalysis) -> dict:
        key = (model, i)
        if key not in self.done:
            rel = f"gallery/{_slug(model)}/{i:05d}_{self.ds.records[i].path.stem}.jpg"
            render_thumbnail(
                self.ds.records[i], a, self.ds.names, self.dir / rel, self.size, self.quality
            )
            self.done[key] = rel
        return {
            "model": model,
            "image": self.ds.records[i].path.name,
            "index": i,
            "thumb": self.done[key],
            "score": round(a.score, 3),
            "tp": len(a.tp),
            "fp": len(a.fp),
            "missed": len(a.missed),
        }


def build_galleries(
    ds: Dataset,
    preds: dict[str, list[Detections]],
    out_dir: Path,
    conf: float = 0.25,
    k: int = 6,
    thumb_px: int = 480,
    quality: int = 70,
    per_class_k: int = 2,
    weak_classes: dict[str, list[int]] | None = None,
) -> dict:
    """Return gallery metadata (thumbnails written under out_dir/gallery).

    Keys: per_model (worst-k), a_ok_b_fails ("A>B" views), all_fail, per_class.
    """
    an = {m: analyze_dataset(ds, p, conf) for m, p in preds.items()}
    scores = {m: _scores(a) for m, a in an.items()}
    thumbs = _Thumbs(ds, out_dir, thumb_px, quality)
    result: dict = {"conf": conf, "k": k, "per_model": {}, "a_ok_b_fails": {}, "per_class": {}}
    for m in preds:
        result["per_model"][m] = [thumbs.item(m, i, an[m][i]) for i in worst_images(scores[m], k)]
    if len(preds) > 1:
        for a, b in permutations(preds, 2):
            idx = a_ok_b_fails(scores[a], scores[b], k)
            result["a_ok_b_fails"][f"{a} ok, {b} fails"] = [
                thumbs.item(b, i, an[b][i]) for i in idx
            ]
    first = next(iter(preds))
    result["all_fail"] = [
        {
            **thumbs.item(first, i, an[first][i]),
            "scores": {m: round(float(scores[m][i]), 2) for m in preds},
        }
        for i in all_models_fail(scores, k)
    ]
    for m in preds:
        classes = (weak_classes or {}).get(m) or [
            c for c in range(len(ds.names)) if any(c in r.gt.classes for r in ds.records)
        ][:12]
        picks = worst_per_class(an[m], classes, per_class_k)
        result["per_class"][m] = {
            ds.names[c]: [thumbs.item(m, i, an[m][i]) for i in idxs] for c, idxs in picks.items()
        }
    return result


def _scores(analyses: list[ImageAnalysis]) -> np.ndarray:
    return np.array([a.score for a in analyses])
