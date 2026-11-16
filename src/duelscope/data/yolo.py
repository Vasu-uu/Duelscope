"""YOLO-format dataset loader (data.yaml plus images/labels folders)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from duelscope.data.base import IMG_EXTS, DatasetError, image_size
from duelscope.types import Dataset, GroundTruth, ImageRecord, Issue

_SPLIT_ALIASES = {
    "val": ("val", "valid", "validation", "test"),
    "train": ("train",),
    "test": ("test", "val"),
}


def _find_yaml(path: Path) -> Path:
    if path.is_file() and path.suffix in {".yaml", ".yml"}:
        return path
    if path.is_dir():
        for name in ("data.yaml", "data.yml", "dataset.yaml"):
            if (path / name).is_file():
                return path / name
        found = sorted(path.glob("*.y*ml"))
        if found:
            return found[0]
    raise DatasetError(
        f"No data.yaml found at '{path}'. Point --data to a YOLO data.yaml or its folder, "
        "or to a COCO .json file."
    )


def _names(cfg: dict, yaml_path: Path) -> list[str]:
    names = cfg.get("names")
    if isinstance(names, dict):
        n = max(int(k) for k in names) + 1
        return [str(names.get(i, names.get(str(i), f"class_{i}"))) for i in range(n)]
    if isinstance(names, list) and names:
        return [str(x) for x in names]
    raise DatasetError(f"'{yaml_path}' has no 'names' list or mapping. Add class names to it.")


def _root(cfg: dict, yaml_dir: Path) -> Path:
    p = cfg.get("path")
    if not p:
        return yaml_dir
    root = Path(str(p))
    if not root.is_absolute():
        root = yaml_dir / root
    return root if root.exists() else yaml_dir


def _list_images(entry: str | list, root: Path, split: str) -> list[Path]:
    entries = entry if isinstance(entry, list) else [entry]
    out: list[Path] = []
    for e in entries:
        p = Path(str(e))
        p = p if p.is_absolute() else root / p
        if p.is_dir():
            out += [f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS]
        elif p.suffix == ".txt" and p.is_file():
            for line in p.read_text().splitlines():
                if line.strip():
                    q = Path(line.strip())
                    out.append(q if q.is_absolute() else (root / q))
        elif p.is_file():
            out.append(p)
        else:
            raise DatasetError(
                f"Split '{split}' points to '{p}', which does not exist. "
                "Fix the path in data.yaml or choose another --split."
            )
    return sorted(set(out))


def label_path(image: Path) -> Path:
    """Map .../images/x.jpg to .../labels/x.txt (the Ultralytics convention)."""
    parts = list(image.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _parse_label(path: Path, w: int, h: int, n_classes: int, issues: list[Issue]) -> GroundTruth:
    boxes, classes = [], []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        tok = line.split()
        if not tok:
            continue
        where = f"{path}:{ln}"
        try:
            cls = int(float(tok[0]))
            vals = np.array([float(t) for t in tok[1:]], dtype=np.float64)
        except ValueError:
            issues.append(
                Issue("error", "malformed_box", where, f"Non-numeric label line: '{line}'")
            )
            continue
        if len(vals) == 4:
            cx, cy, bw, bh = vals
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        elif len(vals) >= 6 and len(vals) % 2 == 0:  # segmentation polygon -> enclosing box
            xs, ys = vals[0::2], vals[1::2]
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        else:
            issues.append(
                Issue(
                    "error",
                    "malformed_box",
                    where,
                    f"Expected 5 values (class cx cy w h), got {len(tok)}.",
                )
            )
            continue
        if cls < 0 or cls >= n_classes:
            issues.append(
                Issue(
                    "error",
                    "class_out_of_range",
                    where,
                    f"Class id {cls} not in 0..{n_classes - 1}. Fix the label or 'names'.",
                )
            )
            continue
        if not np.all(np.isfinite([x1, y1, x2, y2])) or x2 <= x1 or y2 <= y1:
            issues.append(Issue("error", "malformed_box", where, "Box has zero or negative size."))
            continue
        if min(x1, y1) < -0.01 or max(x2, y2) > 1.01:
            issues.append(
                Issue(
                    "warning",
                    "box_outside_image",
                    where,
                    "Box extends outside the image; it was clipped.",
                )
            )
        boxes.append([max(x1, 0) * w, max(y1, 0) * h, min(x2, 1) * w, min(y2, 1) * h])
        classes.append(cls)
    return GroundTruth.make(boxes, classes)


def load_yolo(path: str | Path, split: str = "val") -> Dataset:
    """Load a YOLO-format dataset. `path` is a data.yaml or the folder containing it."""
    yaml_path = _find_yaml(Path(path))
    cfg = yaml.safe_load(yaml_path.read_text()) or {}
    names = _names(cfg, yaml_path)
    root = _root(cfg, yaml_path.parent)
    key = next((k for k in _SPLIT_ALIASES.get(split, (split,)) if cfg.get(k)), None)
    if key is None:
        raise DatasetError(
            f"data.yaml '{yaml_path}' has no '{split}' split. Add a '{split}:' entry "
            "or pass --split with one of: "
            + ", ".join(k for k in ("train", "val", "test") if cfg.get(k))
        )
    images = _list_images(cfg[key], root, key)
    if not images:
        raise DatasetError(f"Split '{key}' of '{yaml_path}' contains no images.")
    issues: list[Issue] = []
    if key != split:
        issues.append(
            Issue(
                "info",
                "split_fallback",
                str(yaml_path),
                f"Split '{split}' missing; used '{key}' instead.",
            )
        )
    records: list[ImageRecord] = []
    for i, img in enumerate(images):
        if not img.is_file():
            issues.append(Issue("error", "missing_image", str(img), "Listed image does not exist."))
            continue
        w, h = image_size(img)
        lp = label_path(img)
        if lp.is_file():
            gt = _parse_label(lp, w, h, len(names), issues)
            if len(gt) == 0:
                issues.append(Issue("info", "empty_image", str(img), "Image has no boxes."))
        else:
            gt = GroundTruth.empty()
            issues.append(
                Issue(
                    "warning",
                    "missing_label",
                    str(lp),
                    "No label file; treated as background. Create an empty .txt "
                    "if the image really has no objects.",
                )
            )
        records.append(ImageRecord(i, img, w, h, gt))
    if not records:
        raise DatasetError(f"No readable images found for split '{key}' in '{yaml_path}'.")
    name = str(cfg.get("name") or yaml_path.parent.name or "dataset")
    return Dataset(name, key, names, records, yaml_path.parent, issues)
