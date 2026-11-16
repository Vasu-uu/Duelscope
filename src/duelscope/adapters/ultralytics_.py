"""Ultralytics (YOLOv8, YOLO11, RT-DETR, ...) adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from duelscope.adapters.base import (
    AdapterError,
    Detector,
    DetectorConfig,
    ImageInput,
    resolve_device,
)
from duelscope.types import Detections


class UltralyticsDetector(Detector):
    """Any weights file Ultralytics can load (e.g. yolov8n.pt, yolo11s.pt)."""

    framework = "ultralytics"

    def __init__(self, weights: str, cfg: DetectorConfig | None = None) -> None:
        super().__init__(cfg)
        self.weights = weights
        p = Path(weights)
        self.weights_path = p if p.is_file() else None
        self._model = None

    @property
    def name(self) -> str:
        return Path(self.weights).stem

    @property
    def class_names(self) -> list[str] | None:
        if self._model is None:
            return None
        names = self._model.names
        return [str(names[i]) for i in range(len(names))]

    @property
    def num_params(self) -> int | None:
        net = getattr(self._model, "model", None)
        if net is None or not hasattr(net, "parameters"):
            return None
        return int(sum(p.numel() for p in net.parameters()))

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise AdapterError(
                "Ultralytics is not installed. Run: pip install 'duelscope[ultralytics]'"
            ) from exc
        try:
            self._model = YOLO(self.weights)
        except Exception as exc:
            raise AdapterError(
                f"Could not load '{self.weights}': {exc}. Check the path, or that the name is a "
                "known Ultralytics weight and you are online for the first download."
            ) from exc
        if self.weights_path is None:  # downloaded into the working directory by Ultralytics
            cand = Path(Path(self.weights).name)
            self.weights_path = cand if cand.is_file() else None

    def predict(self, images: list[ImageInput]) -> list[Detections]:
        self.load()
        c = self.cfg
        src = [str(i) if not isinstance(i, np.ndarray) else i[..., ::-1].copy() for i in images]
        dev = resolve_device(c.device)
        results = self._model.predict(
            src,
            imgsz=c.imgsz,
            conf=c.conf,
            iou=c.iou,
            max_det=c.max_det,
            device=0 if dev == "cuda" else "cpu",
            verbose=False,
        )
        out = []
        for r in results:
            b = r.boxes
            out.append(
                Detections.make(
                    b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
                )
            )
        sp = [r.speed for r in results]
        self.last_timing = {
            k: float(np.mean([s[k] for s in sp]))
            for k in ("preprocess", "inference", "postprocess")
        }
        return out
