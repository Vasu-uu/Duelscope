"""COCO JSON dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

from duelscope.data.base import DatasetError, image_size
from duelscope.types import Dataset, GroundTruth, ImageRecord, Issue


def _image_dir(json_path: Path, images_dir: Path | None) -> Path:
    if images_dir:
        return images_dir
    for cand in (json_path.parent / "images", json_path.parent.parent / "images", json_path.parent):
        if cand.is_dir():
            return cand
    return json_path.parent


def load_coco(path: str | Path, images_dir: str | Path | None = None) -> Dataset:
    """Load a COCO-format annotation file. Category ids are remapped to 0..K-1 by sorted id."""
    jp = Path(path)
    try:
        doc = json.loads(jp.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(
            f"Cannot parse COCO file '{jp}': {exc}. Check it is valid JSON."
        ) from exc
    for key in ("images", "annotations", "categories"):
        if key not in doc:
            raise DatasetError(f"COCO file '{jp}' lacks the '{key}' list.")
    cats = sorted(doc["categories"], key=lambda c: c["id"])
    cat_index = {c["id"]: i for i, c in enumerate(cats)}
    names = [str(c["name"]) for c in cats]
    img_dir = _image_dir(jp, Path(images_dir) if images_dir else None)
    issues: list[Issue] = []
    per_image: dict[int, list[tuple[list[float], int, bool]]] = {}
    for a in doc["annotations"]:
        where = f"{jp}#ann{a.get('id', '?')}"
        if a.get("category_id") not in cat_index:
            issues.append(
                Issue(
                    "error",
                    "class_out_of_range",
                    where,
                    f"category_id {a.get('category_id')} not in categories.",
                )
            )
            continue
        bbox = a.get("bbox")
        if not bbox or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            issues.append(Issue("error", "malformed_box", where, f"Invalid bbox {bbox}."))
            continue
        x, y, w, h = bbox
        per_image.setdefault(a["image_id"], []).append(
            ([x, y, x + w, y + h], cat_index[a["category_id"]], bool(a.get("iscrowd", 0)))
        )
    records: list[ImageRecord] = []
    for i, im in enumerate(sorted(doc["images"], key=lambda d: d["id"])):
        p = img_dir / im["file_name"]
        if not p.is_file():
            issues.append(
                Issue(
                    "error",
                    "missing_image",
                    str(p),
                    "Image listed in JSON not found. Use --images-dir.",
                )
            )
            continue
        w, h = (im.get("width"), im.get("height"))
        if not w or not h:
            w, h = image_size(p)
        anns = per_image.get(im["id"], [])
        if not anns:
            issues.append(Issue("info", "empty_image", str(p), "Image has no boxes."))
        gt = GroundTruth.make([a[0] for a in anns], [a[1] for a in anns], [a[2] for a in anns])
        records.append(ImageRecord(i, p, int(w), int(h), gt))
    if not records:
        raise DatasetError(f"No readable images for '{jp}'. Set --images-dir to the image folder.")
    return Dataset(jp.stem, "coco", names, records, jp.parent, issues)
