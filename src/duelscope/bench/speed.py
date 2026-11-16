"""Latency / throughput / memory benchmark for one detector."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil

from duelscope.adapters.base import Detector, load_rgb, resolve_device


@dataclass
class SpeedResult:
    """Speed and resource numbers for one detector (all latencies in ms)."""

    device: str
    imgsz: int
    n_images: int
    warmup_runs: int
    timed_runs: int
    latency_ms: dict  # batch size 1: mean/std/p50/p90/p99/min/max per image
    fps_bs1: float
    batch: dict  # batch size -> {latency_ms_mean, latency_ms_std, throughput_fps}
    breakdown_ms: dict | None  # preprocess / inference / postprocess at batch size 1
    gpu_peak_mb: float | None
    ram_mb: float
    weights_mb: float | None
    params: int | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        return dict(self.__dict__)


def _stats(ms: np.ndarray) -> dict:
    return {
        "mean": float(ms.mean()),
        "std": float(ms.std(ddof=1)) if len(ms) > 1 else 0.0,
        "p50": float(np.percentile(ms, 50)),
        "p90": float(np.percentile(ms, 90)),
        "p99": float(np.percentile(ms, 99)),
        "min": float(ms.min()),
        "max": float(ms.max()),
    }


def _sync(on_cuda: bool) -> None:
    if on_cuda:
        import torch

        torch.cuda.synchronize()


def _cuda_usable(device: str) -> bool:
    if resolve_device(device) != "cuda":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _timed(det: Detector, batch: list[np.ndarray], on_cuda: bool) -> tuple[float, dict | None]:
    _sync(on_cuda)
    t0 = time.perf_counter()
    det.predict(batch)
    _sync(on_cuda)
    return (time.perf_counter() - t0) * 1e3, det.last_timing


def benchmark(
    det: Detector,
    images: list[Path],
    batch_sizes: tuple[int, ...] = (1, 8),
    warmup: int = 5,
    iters: int = 30,
) -> SpeedResult:
    """Time `det` on a fixed list of images (decoded once, so disk I/O is excluded).

    Per measurement: CUDA synchronised before and after each call, `warmup` untimed calls first,
    then `iters` timed calls per batch size. The detector is unloaded first so GPU memory
    includes the weights.
    """
    if not images:
        raise ValueError("Speed benchmark needs at least one image.")
    det.close()
    on_cuda = _cuda_usable(det.cfg.device)
    notes: list[str] = []
    if resolve_device(det.cfg.device) == "cuda" and not on_cuda:
        notes.append("CUDA requested but unavailable; benchmarked on CPU.")
    base_mem = 0
    if on_cuda:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
    det.load()
    det.warmup()
    arrays = [load_rgb(p) for p in images]

    def pick(n: int, start: int) -> list[np.ndarray]:
        return [arrays[(start + j) % len(arrays)] for j in range(n)]

    for i in range(warmup):
        det.predict(pick(1, i))
    lat, breakdown = [], []
    for i in range(iters):
        ms, tm = _timed(det, pick(1, i), on_cuda)
        lat.append(ms)
        if tm:
            breakdown.append(tm)
    lat_arr = np.asarray(lat)
    batch: dict = {}
    for bs in batch_sizes:
        times = []
        for i in range(max(iters // bs, 3)):
            ms, _ = _timed(det, pick(bs, i * bs), on_cuda)
            times.append(ms)
        t = np.asarray(times)
        batch[str(bs)] = {
            "latency_ms_mean": float(t.mean()),
            "latency_ms_std": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
            "throughput_fps": float(bs / (t.mean() / 1e3)),
        }
    gpu_peak = None
    if on_cuda:
        import torch

        gpu_peak = round((torch.cuda.max_memory_allocated() - base_mem) / 2**20, 1)
    keys = ("preprocess", "inference", "postprocess")
    meta = det.metadata
    return SpeedResult(
        device="cuda" if on_cuda else "cpu",
        imgsz=det.cfg.imgsz,
        n_images=len(images),
        warmup_runs=warmup,
        timed_runs=iters,
        latency_ms=_stats(lat_arr),
        fps_bs1=float(1e3 / lat_arr.mean()),
        batch=batch,
        breakdown_ms={k: float(np.mean([b[k] for b in breakdown])) for k in keys}
        if breakdown
        else None,
        gpu_peak_mb=gpu_peak,
        ram_mb=round(psutil.Process().memory_info().rss / 2**20, 1),
        weights_mb=meta.get("weights_mb"),
        params=meta.get("params"),
        notes=notes,
    )
