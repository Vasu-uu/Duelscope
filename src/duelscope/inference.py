"""Run a detector over a dataset (batched, cached) and map classes to the dataset's."""

from __future__ import annotations

from collections.abc import Callable

from duelscope.adapters import ClassMap, Detector
from duelscope.cache import PredictionCache, cache_key
from duelscope.data import content_hash
from duelscope.types import Dataset, Detections

ProgressFn = Callable[[int, int], None]


def run_inference(
    det: Detector,
    ds: Dataset,
    class_map: ClassMap,
    batch: int = 8,
    cache: PredictionCache | None = None,
    progress: ProgressFn | None = None,
) -> tuple[list[Detections], bool]:
    """Return (predictions in dataset class ids, was_cached). Raw model output is what is cached."""
    det.load()
    cfg = det.cfg
    key = cache_key(
        det.fingerprint(),
        content_hash(ds),
        {
            "imgsz": cfg.imgsz,
            "conf": cfg.conf,
            "iou": cfg.iou,
            "max_det": cfg.max_det,
            "fw": det.framework,
        },
    )
    raw = cache.load(key, len(ds)) if cache else None
    cached = raw is not None
    if raw is None:
        raw = []
        paths = [r.path for r in ds.records]
        for i in range(0, len(paths), max(batch, 1)):
            chunk = paths[i : i + batch]
            out = det.predict(chunk)
            if len(out) != len(chunk):
                raise RuntimeError(
                    f"Detector '{det.name}' returned {len(out)} results for {len(chunk)} images."
                )
            raw.extend(out)
            if progress:
                progress(min(i + batch, len(paths)), len(paths))
        if cache:
            cache.save(key, raw)
    elif progress:
        progress(len(ds), len(ds))
    return [class_map.apply(d) for d in raw], cached
