# Metric definitions

- **Matching**: per image and class, detections in descending score are greedily matched to the best unmatched ground truth with IoU at or above the threshold (COCO rules, `iscrowd` boxes use intersection over detection area and never produce false positives). At most 100 detections per image and class.
- **AP**: 101-point interpolated precision-recall area per class. **mAP50** uses IoU 0.5; **mAP50-95** averages IoU 0.50 to 0.95 in steps of 0.05. Classes with no ground truth are excluded.
- **mAP small/medium/large**: box area below 32^2, 32^2 to 96^2, above 96^2 pixels; `null` when the dataset has no objects in that range.
- **Precision / recall / F1**: macro-averaged over classes at the confidence that maximises mean F1 (IoU 0.5). That confidence is reported as `best_conf`.
- **Evaluation threshold**: predictions are generated at confidence 0.001. A separate `conf` (default 0.25) drives galleries and the error counts.
- **Error taxonomy** (simplified TIDE, IoU thresholds 0.5 foreground / 0.1 background). Each detection above `conf` that is not a true positive is one of: *duplicate* (IoU >= 0.5 with a same-class box already matched), *classification* (IoU >= 0.5 with a different-class box), *localization* (same class, IoU 0.1-0.5), *both* (different class, IoU 0.1-0.5), *background* (IoU < 0.1). *Missed* is a ground truth without a matching detection. Shares are counts over total errors. The **mAP gain** reruns mAP50 after an oracle fix (relabel, snap box, drop duplicates/background/both, drop missed ground truth), always on the full 0.001-confidence predictions.

## Comparison with Ultralytics `model.val()`

1. *Metric math*: fed the exact predictions `model.val()` produced for `coco8` (`yolov8n.pt`, CPU), Duelscope gives mAP50 0.9369 and mAP50-95 0.6445 versus Ultralytics 0.9241 and 0.6376. The small offset is definitional (Ultralytics integrates a trapezoid and caps a perfect class at 0.995). Covered by `tests/test_metrics.py`.
2. *Preprocessing*: `val()` pads rectangular batches by an extra half stride (a 640x488 image becomes 672x512); `predict()` pads to the minimal multiple of 32 (640x512). On the same 4 `coco8` val images that moved end-to-end mAP50 from 0.924 (`val()`) to 0.819 (Duelscope). That set is tiny and memorised by the model, so treat it as a worst case. Duelscope applies the same preprocessing to every model, so relative comparisons stay fair.
