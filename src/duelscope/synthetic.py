"""Tiny synthetic YOLO dataset generator for demos, smoke runs and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

CLASS_NAMES = ["square", "wide", "tall", "blob"]
_COLORS = [(220, 60, 60), (60, 180, 75), (60, 90, 220), (230, 190, 40)]


def make_yolo_dataset(
    root: str | Path,
    n_images: int = 12,
    n_classes: int = 3,
    size: tuple[int, int] = (320, 240),
    seed: int = 0,
    split: str = "val",
) -> Path:
    """Write a deterministic YOLO dataset of coloured rectangles; return the data.yaml path."""
    if not 1 <= n_classes <= len(CLASS_NAMES):
        raise ValueError(f"n_classes must be between 1 and {len(CLASS_NAMES)}")
    root = Path(root)
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = size
    for i in range(n_images):
        bg = rng.integers(90, 130, size=(h, w, 3), dtype=np.uint8)
        im = Image.fromarray(bg)
        draw = ImageDraw.Draw(im)
        lines = []
        for _ in range(int(rng.integers(1, 5))):
            c = int(rng.integers(0, n_classes))
            bw = int(rng.integers(w // 8, w // 3))
            bh = int(rng.integers(h // 8, h // 3))
            if c == 1:
                bw = min(bw * 2, w - 2)
            if c == 2:
                bh = min(bh * 2, h - 2)
            x1 = int(rng.integers(0, w - bw))
            y1 = int(rng.integers(0, h - bh))
            draw.rectangle([x1, y1, x1 + bw, y1 + bh], fill=_COLORS[c])
            lines.append(
                f"{c} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
            )
        im.save(img_dir / f"img_{i:03d}.png")
        (lbl_dir / f"img_{i:03d}.txt").write_text("\n".join(lines) + "\n")
    cfg = {"path": ".", split: f"images/{split}", "names": CLASS_NAMES[:n_classes]}
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))
    return yaml_path
