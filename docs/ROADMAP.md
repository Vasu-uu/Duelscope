# Roadmap

## Good first issues

- [ ] RT-DETR adapter (ONNX decoder `end2end` variant with query outputs)
- [ ] YOLOX adapter
- [ ] EfficientDet adapter
- [ ] COCO-JSON export of predictions (`results` to `predictions.json`)
- [ ] HTML theme switcher (light/dark toggle, no external dependencies)
- [ ] Per-image-size breakdown of AP in the report
- [ ] `--sort-by` option for the terminal summary

## Planned

- [x] `duelscope compare-reports a.json b.json`
- [ ] Side-by-side video benchmark mode
- [ ] ONNX/TensorRT export-and-compare helper
- [ ] FP32 vs FP16 vs INT8 report section
- [ ] Match `model.val()` preprocessing as an opt-in Ultralytics mode

## Known limitations

- Only axis-aligned boxes (no segmentation/pose/OBB metrics).
- Error taxonomy is a simplified TIDE (no separate "dupe vs cls" priority tuning).
- Speed numbers exclude disk I/O and are single-process.
