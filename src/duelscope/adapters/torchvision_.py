"""torchvision detection models (Faster R-CNN, RetinaNet, SSD, FCOS)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from duelscope.adapters.base import (
    AdapterError,
    Detector,
    DetectorConfig,
    ImageInput,
    free_gpu,
    load_rgb,
    resolve_device,
)
from duelscope.types import Detections

_FAMILIES = {
    "fasterrcnn_resnet50_fpn": "FasterRCNN_ResNet50_FPN_Weights",
    "fasterrcnn_resnet50_fpn_v2": "FasterRCNN_ResNet50_FPN_V2_Weights",
    "fasterrcnn_mobilenet_v3_large_fpn": "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
    "retinanet_resnet50_fpn": "RetinaNet_ResNet50_FPN_Weights",
    "ssd300_vgg16": "SSD300_VGG16_Weights",
    "ssdlite320_mobilenet_v3_large": "SSDLite320_MobileNet_V3_Large_Weights",
    "fcos_resnet50_fpn": "FCOS_ResNet50_FPN_Weights",
}


class TorchvisionDetector(Detector):
    """Pretrained torchvision detector; labels are COCO category ids (index = id)."""

    framework = "torchvision"

    def __init__(
        self, model_name: str, cfg: DetectorConfig | None = None, weights: str | None = "DEFAULT"
    ) -> None:
        super().__init__(cfg)
        if model_name not in _FAMILIES:
            raise AdapterError(
                f"Unknown torchvision model '{model_name}'. Choose one of: "
                + ", ".join(sorted(_FAMILIES))
            )
        self.model_name = model_name
        self.weights = weights
        self._model = None
        self._torch = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def class_names(self) -> list[str] | None:
        import torchvision.models.detection as det

        enum = getattr(det, _FAMILIES[self.model_name])
        return [str(c) for c in enum.DEFAULT.meta["categories"]]

    @property
    def num_params(self) -> int | None:
        if self._model is None:
            return None
        return int(sum(p.numel() for p in self._model.parameters()))

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import torchvision.models.detection as det
        except ImportError as exc:
            raise AdapterError(
                "torch/torchvision not installed. Run: pip install 'duelscope[torchvision]'"
            ) from exc
        self._torch = torch
        kwargs = {"weights": self.weights} if self.weights else {"weights": None}
        if not self.weights:
            kwargs["weights_backbone"] = None
        try:
            model = getattr(det, self.model_name)(**kwargs)
        except Exception as exc:
            raise AdapterError(
                f"Could not build '{self.model_name}': {exc}. Weights download needs internet "
                "the first time."
            ) from exc
        c = self.cfg
        if hasattr(model, "roi_heads"):
            model.roi_heads.score_thresh = c.conf
            model.roi_heads.nms_thresh = c.iou
            model.roi_heads.detections_per_img = c.max_det
        else:
            model.score_thresh, model.nms_thresh = c.conf, c.iou
            model.detections_per_img = c.max_det
        self._device = resolve_device(c.device)
        self._model = model.eval().to(self._device)
        if self.weights:
            fn = Path(getattr(getattr(det, _FAMILIES[self.model_name]).DEFAULT, "url", "")).name
            cand = Path(torch.hub.get_dir()) / "checkpoints" / fn
            self.weights_path = cand if cand.is_file() else None

    def close(self) -> None:
        self._model = None
        free_gpu()

    def predict(self, images: list[ImageInput]) -> list[Detections]:
        self.load()
        torch = self._torch
        t0 = time.perf_counter()
        tensors = [
            torch.from_numpy(np.ascontiguousarray(load_rgb(i))).permute(2, 0, 1).float().div(255)
            for i in images
        ]
        tensors = [t.to(self._device) for t in tensors]
        t1 = time.perf_counter()
        with torch.inference_mode():
            outs = self._model(tensors)
        if self._device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        n = max(len(images), 1)
        self.last_timing = {
            "preprocess": (t1 - t0) * 1e3 / n,
            "inference": (t2 - t1) * 1e3 / n,
            "postprocess": 0.0,  # NMS is inside the model forward
        }
        return [
            Detections.make(
                o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy()
            )
            for o in outs
        ]
