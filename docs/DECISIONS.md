# Design decisions

Decisions taken without asking, recorded so they can be revisited.

1. **Repo URL**: `https://github.com/Vasu-uu/Duelscope.git` (the URL given by the maintainer; GitHub is case-insensitive). Package/CLI name stays lowercase `duelscope`.
2. **Build backend**: hatchling, `src/` layout, version read from `src/duelscope/__init__.py`.
3. **Metrics implemented in-house** (numpy, COCO-style 101-point interpolation) rather than depending on `pycocotools`, to keep the core install small and wheels-free on every platform. Cross-checked against Ultralytics `model.val()` and `pycocotools` where available (see tests).
4. **Box convention**: absolute-pixel `xyxy` float arrays everywhere internally.
5. **Class mapping**: by case-insensitive name; unmapped model classes are dropped with a loud warning.
6. **Cache key**: sha256 over model file hash (or spec string), dataset content hash, and inference settings.
7. **Reports**: self-contained HTML (inline CSS/JS, base64 PNG charts and JPEG thumbnails, no CDN).
8. **Dev environment**: Python 3.14 locally; CI targets 3.10-3.12 as specified. Code avoids 3.11+-only syntax.
9. **Metrics vs `model.val()`**: Duelscope's evaluator, fed the exact detections `model.val()` produced for `coco8` (val split, `yolov8n.pt`, CPU), gave mAP50 0.9369 / mAP50-95 0.6445 versus Ultralytics' 0.9241 / 0.6376 (tested within 0.015 / 0.01). The residual offset is definitional: COCO 101-point sampling versus Ultralytics' trapezoid integral that caps a perfect class at 0.995.
10. **Predict path vs val path**: `model.val()` letterboxes each image to a rectangle with `pad=0.5` extra stride (e.g. 640x488 becomes 672x512) while `model.predict()` uses the minimal stride-aligned rectangle (640x512). On the same 4-image `coco8` val set this made Duelscope's end-to-end mAP50 0.819 / mAP50-95 0.607 versus `val()`'s 0.924 / 0.638 (a tiny, memorised set, so deliberately a worst case). Duelscope uses the same predict-style preprocessing for every model so comparisons between models are consistent; absolute numbers can differ from `val()`. This is documented in `docs/metrics.md`.
