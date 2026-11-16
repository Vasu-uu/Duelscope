# Changelog

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
