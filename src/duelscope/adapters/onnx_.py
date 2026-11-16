"""ONNX Runtime adapter with a configurable YOLO-style output decoder."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from duelscope.adapters._utils import decode_yolo, letterbox
from duelscope.adapters.base import (
    AdapterError,
    Detector,
    DetectorConfig,
    ImageInput,
    load_rgb,
    resolve_device,
)
from duelscope.types import Detections


class OnnxDetector(Detector):
    """Runs a .onnx detector; `decoder` is auto, yolov8, yolov5 or end2end."""

    framework = "onnxruntime"

    def __init__(
        self,
        path: str,
        cfg: DetectorConfig | None = None,
        decoder: str = "auto",
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.weights_path = Path(path)
        self.decoder = decoder
        self._names = class_names
        self._sess = None
        self._size = self.cfg.imgsz

    @property
    def name(self) -> str:
        return self.weights_path.stem

    @property
    def class_names(self) -> list[str] | None:
        return self._names

    def load(self) -> None:
        if self._sess is not None:
            return
        if not self.weights_path.is_file():
            raise AdapterError(f"ONNX file '{self.weights_path}' not found. Check the path.")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise AdapterError(
                "onnxruntime is not installed. Run: pip install 'duelscope[onnx]'"
            ) from exc
        avail = ort.get_available_providers()
        want = resolve_device(self.cfg.device)
        providers = ["CPUExecutionProvider"]
        if want == "cuda":
            if "CUDAExecutionProvider" in avail:
                providers.insert(0, "CUDAExecutionProvider")
            else:
                print(
                    "[duelscope] WARNING: CUDAExecutionProvider unavailable for "
                    f"'{self.weights_path.name}'; running on CPU. Install onnxruntime-gpu."
                )
        self._sess = ort.InferenceSession(str(self.weights_path), providers=providers)
        inp = self._sess.get_inputs()[0]
        shape = inp.shape
        if isinstance(shape[2], int):
            self._size = shape[2]
        self._fp16 = "float16" in inp.type
        meta = self._sess.get_modelmeta().custom_metadata_map
        if self._names is None and "names" in meta:
            try:
                import ast

                d = ast.literal_eval(meta["names"])
                self._names = [str(d[i]) for i in range(len(d))]
            except (ValueError, SyntaxError, KeyError):
                pass

    def _kind(self, out_shape: tuple) -> str:
        if self.decoder != "auto":
            return self.decoder
        if out_shape[-1] == 6:
            return "end2end"
        return "yolov8" if out_shape[1] < out_shape[2] else "yolov5"

    def predict(self, images: list[ImageInput]) -> list[Detections]:
        self.load()
        name = self._sess.get_inputs()[0].name
        res: list[Detections] = []
        t_pre = t_inf = t_post = 0.0
        for im in images:
            t0 = time.perf_counter()
            rgb = load_rgb(im)
            h, w = rgb.shape[:2]
            lb, ratio, (px, py) = letterbox(rgb, self._size)
            x = lb.transpose(2, 0, 1)[None].astype(np.float16 if self._fp16 else np.float32) / 255
            t1 = time.perf_counter()
            out = self._sess.run(None, {name: x})[0]
            t2 = time.perf_counter()
            b, s, c = decode_yolo(
                out, self._kind(out.shape), self.cfg.conf, self.cfg.iou, self.cfg.max_det
            )
            b = b.copy()
            b[:, [0, 2]] = ((b[:, [0, 2]] - px) / ratio).clip(0, w)
            b[:, [1, 3]] = ((b[:, [1, 3]] - py) / ratio).clip(0, h)
            res.append(Detections.make(b, s, c))
            t3 = time.perf_counter()
            t_pre, t_inf, t_post = t_pre + t1 - t0, t_inf + t2 - t1, t_post + t3 - t2
        n = max(len(images), 1)
        self.last_timing = {
            "preprocess": t_pre * 1e3 / n,
            "inference": t_inf * 1e3 / n,
            "postprocess": t_post * 1e3 / n,
        }
        return res
