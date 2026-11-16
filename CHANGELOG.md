# Changelog

## 0.2.0

- `video-bench`: side-by-side video benchmark (latency, FPS, detection count, flicker, tiled output video).
- `quant-compare`: FP32/FP16/INT8 ONNX variants with a quantisation section in the report; INT8 keeps the detection head in float.
- `export`: Ultralytics to ONNX helper.
- Project logo, favicon and logo in the HTML report.
- CI actions updated; ONNX adapter verified on a real export.

## 0.1.0

First release.

- Dataset layer: YOLO and COCO loaders with validation and `check-data`.
- Adapters: Ultralytics, ONNX Runtime, torchvision, custom Python classes, plugin entry points, class-name mapping.
- Metrics: COCO-style mAP (50, 50-95, S/M/L), per-class AP, P/R/F1, PR curves, TIDE-style error taxonomy, prediction cache.
- Speed benchmark with hardware capture and CPU fallback.
- Failure-case galleries (per model, cross-model, per class).
- Reports: `results.json`, `report.md`, self-contained `report.html`, plain-language insights.
- CLI: `run`, `check-data`, `report`, `list-adapters`, `doctor`, `init`, `demo-data`, `compare-reports`, `export` (ONNX).
- ONNX adapter verified on a real yolov8n export: mAP50 0.901 vs 0.901 for the `.pt` on the 8-image coco8 sample (mAP50-95 0.705 vs 0.696).
