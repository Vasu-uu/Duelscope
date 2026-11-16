"""Draw annotated, downscaled JPEG thumbnails."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from duelscope.gallery.select import ImageAnalysis
from duelscope.types import ImageRecord

GREEN = (46, 160, 67)
BLUE = (41, 121, 255)
RED = (230, 57, 70)
YELLOW = (255, 196, 0)


def _dashed_rect(draw: ImageDraw.ImageDraw, box: list[float], color: tuple, width: int) -> None:
    x1, y1, x2, y2 = box
    dash = 8
    for x in np.arange(x1, x2, dash * 2):
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
    for y in np.arange(y1, y2, dash * 2):
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)


def _label(draw: ImageDraw.ImageDraw, xy: tuple, text: str, color: tuple, font) -> None:
    x, y = xy
    x0, t, r, b = draw.textbbox((x, y), text, font=font)
    top = max(y - (b - t) - 2, 0)
    draw.rectangle([x0, top, r + 2, top + (b - t) + 2], fill=color)
    draw.text((x0 + 1, top), text, fill=(255, 255, 255), font=font)


def render_thumbnail(
    rec: ImageRecord,
    analysis: ImageAnalysis,
    names: list[str],
    out_path: Path,
    max_side: int = 480,
    quality: int = 70,
) -> int:
    """Write an annotated JPEG and return its size in bytes.

    Green: ground truth. Blue: true positive. Red: false positive.
    Yellow dashed: missed ground truth.
    """
    with Image.open(rec.path) as im:
        img = im.convert("RGB")
    scale = min(max_side / max(img.size), 1.0)
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    lw = max(1, round(max_side / 240))
    missed = set(analysis.missed.tolist())
    for i, (b, c) in enumerate(zip(rec.gt.boxes, rec.gt.classes, strict=True)):
        box = (b * scale).tolist()
        if i in missed:
            _dashed_rect(draw, box, YELLOW, lw + 1)
            _label(draw, (box[0], box[1]), f"missed {names[c]}", YELLOW, font)
        else:
            draw.rectangle(box, outline=GREEN, width=lw)
    d = analysis.det
    for idx, color in ((analysis.tp, BLUE), (analysis.fp, RED)):
        for j in idx:
            box = (d.boxes[j] * scale).tolist()
            draw.rectangle(box, outline=color, width=lw + 1)
            _label(draw, (box[0], box[1]), f"{names[d.classes[j]]} {d.scores[j]:.2f}", color, font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=quality, optimize=True)
    return out_path.stat().st_size
