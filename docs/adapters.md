# Adapters guide

A detector is anything implementing `duelscope.adapters.Detector`.

## Model specs

| Spec | Adapter |
|---|---|
| `yolov8n.pt`, `ultralytics:yolo11s.pt` | Ultralytics |
| `model.onnx` | ONNX Runtime (`decoder`: auto, yolov8, yolov5, end2end) |
| `torchvision:fasterrcnn_resnet50_fpn` | torchvision (also retinanet, ssd300_vgg16, ssdlite320, fcos) |
| `custom:path/to/file.py:ClassName` | your own class |

## Writing a custom adapter

```python
from duelscope.adapters import Detector
from duelscope.types import Detections


class MyDetector(Detector):
    framework = "mine"

    @property
    def name(self):
        return "my-detector"

    @property
    def class_names(self):
        return ["person", "car"]  # indexed by your class id

    def load(self): ...  # idempotent
    def predict(self, images):  # Path or RGB uint8 ndarray
        return [Detections.make(boxes_xyxy, scores, class_ids) for _ in images]
```

Rules: boxes are absolute-pixel `xyxy` in the original image; return one `Detections` per input in order; keep confidences down to `self.cfg.conf` (0.001 for evaluation); set `self.last_timing = {"preprocess": ms, "inference": ms, "postprocess": ms}` per image if you can; set `self.weights_path` so predictions are cached by file hash.

See `examples/custom_adapter.py`.

## Class mapping

Model class names are matched to dataset names case-insensitively (`_`, `-` and spaces are equivalent). Unmatched model classes are dropped with a warning; dataset classes the model can never output get AP 0 with a warning. Override with `class_map: {car: vehicle}` in the config. If `class_names` is `None`, ids are assumed to equal dataset ids.

## Publishing a plugin

Expose a factory `(spec_rest: str, cfg: DetectorConfig) -> Detector` (or a `Detector` subclass taking `(spec_rest, cfg)`) in an entry point:

```toml
[project.entry-points."duelscope.adapters"]
rtdetr = "my_pkg.adapter:make_detector"
```

Then `--models rtdetr:weights.pth` works and `duelscope list-adapters` shows it.
