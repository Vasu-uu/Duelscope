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
