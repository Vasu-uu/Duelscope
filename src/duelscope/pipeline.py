"""The end-to-end comparison: inference, accuracy, errors, speed, galleries, insights."""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np

from duelscope import __version__
from duelscope.adapters import ClassMap, Detector, DetectorConfig, create_detector
from duelscope.bench import benchmark, hardware_info
from duelscope.cache import PredictionCache
from duelscope.config import RunConfig
from duelscope.data import content_hash, load_dataset, subset, summarize
from duelscope.gallery import build_galleries
from duelscope.inference import run_inference
from duelscope.metrics import analyze_errors, evaluate
from duelscope.report.insights import make_insights
from duelscope.types import Detections

SCHEMA_VERSION = 1
Progress = Callable[[str, int, int], None]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
    return out


def run_comparison(
    cfg: RunConfig,
    progress: Progress | None = None,
    detectors: list[Detector] | None = None,
) -> dict:
    """Run every model on the dataset and return the full results dictionary.

    Thumbnails are written under cfg.out/gallery. `detectors` (already built, same order as
    cfg.models) lets callers inject custom or mock detectors.
    """
    note = progress or (lambda stage, done, total: None)
    _seed_everything(cfg.seed)
    ds = subset(
        load_dataset(cfg.data, cfg.split, cfg.format, cfg.images_dir), cfg.max_images, cfg.seed
    )
    dcfg = DetectorConfig(
        imgsz=cfg.imgsz, conf=cfg.eval_conf, iou=cfg.iou, device=cfg.device, max_det=300
    )
    dets = detectors or [create_detector(spec, dcfg) for spec in cfg.models]
    names = _unique_names([d.name for d in dets])
    cache = PredictionCache(cfg.cache_dir, enabled=cfg.cache)
    speed_paths = [r.path for r in subset(ds, cfg.speed.images, cfg.seed).records]
    all_preds: dict[str, list[Detections]] = {}
    models: list[dict] = []
    for spec, det, name in zip(cfg.models, dets, names, strict=True):
        det.load()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cmap = ClassMap.build(det.class_names, ds.names, cfg.class_map, name)
        warn = [str(w.message) for w in caught]
        if cmap.unmapped:
            shown = ", ".join(cmap.unmapped[:10]) + ("..." if len(cmap.unmapped) > 10 else "")
            warn.append(
                f"{name}: {len(cmap.unmapped)} model classes have no dataset match and their "
                f"detections were dropped ({shown})."
            )
        preds, cached = run_inference(
            det, ds, cmap, cfg.batch, cache, lambda d, t, n=name: note(f"inference {n}", d, t)
        )
        all_preds[name] = preds
        note(f"accuracy {name}", 0, 1)
        ev = evaluate(ds, preds)
        errors = analyze_errors(ds, preds, cfg.conf)
        errors["conf"] = cfg.conf
        speed = None
        if cfg.speed.enabled:
            note(f"speed {name}", 0, 1)
            speed = benchmark(
                det, speed_paths, tuple(cfg.speed.batch_sizes), cfg.speed.warmup, cfg.speed.iters
            ).to_dict()
        models.append(
            {
                "name": name,
                "spec": spec,
                "metadata": det.metadata,
                "cached_predictions": cached,
                "accuracy": ev.to_dict(),
                "errors": errors,
                "speed": speed,
                "warnings": warn,
            }
        )
        det.close()
    gallery = None
    if cfg.gallery.enabled:
        note("gallery", 0, 1)
        weak = {
            m["name"]: [
                i
                for _, i in sorted(
                    (c["ap5095"], i)
                    for i, c in enumerate(m["accuracy"]["per_class"])
                    if c["ap5095"] is not None
                )
            ][:12]
            for m in models
        }
        g = cfg.gallery
        gallery = build_galleries(
            ds, all_preds, cfg.out, cfg.conf, g.k, g.thumb_px, g.quality, g.per_class_k, weak
        )
    results = {
        "schema_version": SCHEMA_VERSION,
        "duelscope_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {**summarize(ds), "hash": content_hash(ds), "class_names": ds.names},
        "settings": {
            "imgsz": cfg.imgsz,
            "eval_conf": cfg.eval_conf,
            "display_conf": cfg.conf,
            "nms_iou": cfg.iou,
            "device": cfg.device,
            "batch": cfg.batch,
            "max_images": cfg.max_images,
            "seed": cfg.seed,
            "speed_images": len(speed_paths) if cfg.speed.enabled else None,
        },
        "hardware": hardware_info(),
        "models": models,
        "gallery": gallery,
    }
    results["insights"] = make_insights(results)
    return results
