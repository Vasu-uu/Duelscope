"""Minimal custom adapter. Run with:

    duelscope run --data demo_dataset --models custom:examples/custom_adapter.py:EmptyDetector

It predicts nothing, which makes it a handy baseline (mAP 0) and a template to copy.
"""

from duelscope.adapters import Detector
from duelscope.types import Detections


class EmptyDetector(Detector):
    """Replace `predict` with a call into your own model."""

    framework = "example"

    @property
    def name(self) -> str:
        return "empty-baseline"

    def load(self) -> None:
        """Load weights here (must be safe to call twice)."""

    def predict(self, images):
        # images: list of pathlib.Path or RGB uint8 arrays.
        # Return Detections.make(boxes_xyxy_pixels, scores, class_ids) per image.
        return [Detections.empty() for _ in images]
