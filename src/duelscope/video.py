"""Side-by-side video benchmark: run several detectors on the same video, frame by frame."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from duelscope.adapters.base import AdapterError, Detector
from duelscope.report.charts import PALETTE
from duelscope.types import Detections

TILE_H = 360


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise AdapterError(
            "OpenCV is required for video benchmarks. Run: pip install 'duelscope[video]'"
        ) from exc
    return cv2


def _hex_bgr(h: str) -> tuple[int, int, int]:
    return int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)


def _draw(frame_rgb: np.ndarray, det: Detections, names: list[str] | None, color, label: str):
    cv2 = _cv2()
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for b, s, c in zip(det.boxes, det.scores, det.classes, strict=True):
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        nm = names[c] if names and c < len(names) else str(c)
        cv2.putText(
            img,
            f"{nm} {s:.2f}",
            (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    scale = TILE_H / img.shape[0]
    img = cv2.resize(img, (max(int(img.shape[1] * scale), 1), TILE_H))
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        img, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return img


def _stats(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
    }


def video_benchmark(
    detectors: list[Detector],
    source: Path,
    out_video: Path | None = None,
    conf: float = 0.25,
    max_frames: int | None = None,
    stride: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Run every detector on every `stride`-th frame and return latency, FPS and stability.

    Latency is per-frame inference time for one detector (decode and drawing excluded).
    `flicker` is the mean absolute change in detection count between consecutive frames
    (lower means steadier output). If out_video is given, writes a side-by-side video.
    """
    cv2 = _cv2()
    if not source.is_file():
        raise AdapterError(f"Video '{source}' not found. Check the path.")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise AdapterError(f"OpenCV cannot open '{source}'. Try re-encoding it to H.264 MP4.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    for d in detectors:
        d.warmup()
    names = [d.name for d in detectors]
    lat: dict[str, list[float]] = {n: [] for n in names}
    counts: dict[str, list[int]] = {n: [] for n in names}
    decode_ms: list[float] = []
    writer = None
    idx = used = 0
    wall0 = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        ok, bgr = cap.read()
        if not ok:
            break
        decode_ms.append((time.perf_counter() - t0) * 1e3)
        idx += 1
        if (idx - 1) % stride:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tiles = []
        for i, d in enumerate(detectors):
            t1 = time.perf_counter()
            det = d.predict([rgb])[0].filter(conf)
            ms = (time.perf_counter() - t1) * 1e3
            lat[names[i]].append(ms)
            counts[names[i]].append(len(det))
            if out_video is not None:
                tiles.append(
                    _draw(
                        rgb,
                        det,
                        d.class_names,
                        _hex_bgr(PALETTE[i % len(PALETTE)]),
                        f"{names[i]}  {ms:.0f} ms  {len(det)} det",
                    )
                )
        if out_video is not None:
            cols = 1 if len(tiles) == 1 else 2 if len(tiles) > 3 else len(tiles)
            rows = [tiles[r : r + cols] for r in range(0, len(tiles), cols)]
            w = max(sum(t.shape[1] for t in row) for row in rows)
            canvas = np.vstack(
                [
                    np.pad(np.hstack(row), ((0, 0), (0, w - sum(t.shape[1] for t in row)), (0, 0)))
                    for row in rows
                ]
            )
            if writer is None:
                out_video.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(out_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    src_fps / stride,
                    (canvas.shape[1], canvas.shape[0]),
                )
                if not writer.isOpened():
                    raise AdapterError(f"Cannot write '{out_video}'. Use a .mp4 or .avi path.")
            writer.write(canvas)
        used += 1
        if progress:
            progress(used, max_frames or (total // stride) or used)
        if max_frames and used >= max_frames:
            break
    cap.release()
    if writer is not None:
        writer.release()
    if used == 0:
        raise AdapterError(f"No frames could be read from '{source}'.")
    wall = time.perf_counter() - wall0
    models = []
    for d, n in zip(detectors, names, strict=True):
        c = np.asarray(counts[n])
        s = _stats(lat[n])
        models.append(
            {
                "name": n,
                "metadata": d.metadata,
                "latency_ms": s,
                "fps": 1e3 / s["mean"],
                "mean_detections": float(c.mean()),
                "flicker": float(np.abs(np.diff(c)).mean()) if len(c) > 1 else 0.0,
            }
        )
    return {
        "source": source.name,
        "frames_processed": used,
        "source_fps": float(src_fps),
        "stride": stride,
        "conf": conf,
        "decode_ms_mean": float(np.mean(decode_ms)),
        "wall_seconds": wall,
        "side_by_side": str(out_video) if out_video else None,
        "models": models,
    }
