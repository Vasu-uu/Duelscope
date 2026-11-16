from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from duelscope.data import load_dataset
from duelscope.gallery import (
    a_ok_b_fails,
    all_models_fail,
    analyze_dataset,
    analyze_image,
    build_galleries,
    worst_images,
    worst_per_class,
)
from duelscope.gallery.render import BLUE, GREEN, RED, YELLOW, render_thumbnail
from duelscope.synthetic import make_yolo_dataset
from duelscope.types import Dataset, Detections, GroundTruth, ImageRecord


def rec(i: int, boxes, classes) -> ImageRecord:
    return ImageRecord(i, Path(f"im{i}.jpg"), 500, 500, GroundTruth.make(boxes, classes))


def test_analyze_image_counts_and_score() -> None:
    r = rec(0, [[0, 0, 100, 100], [200, 200, 300, 300]], [0, 1])
    det = Detections.make(
        [[0, 0, 100, 100], [400, 400, 450, 450], [10, 10, 20, 20]], [0.9, 0.6, 0.1], [0, 1, 0]
    )
    a = analyze_image(r, det, conf=0.25, n_classes=2)
    assert a.tp.tolist() == [0] and a.fp.tolist() == [1] and a.missed.tolist() == [1]
    assert a.score == pytest.approx(1.0 + 0.6)  # one miss + FP confidence (0.1 det filtered out)
    assert a.class_score.tolist() == pytest.approx([0.0, 1.6])


def test_perfect_image_scores_zero_and_is_never_selected() -> None:
    r = rec(0, [[0, 0, 50, 50]], [0])
    a = analyze_image(r, Detections.make([[0, 0, 50, 50]], [0.9], [0]), 0.25, 1)
    assert a.score == 0.0
    assert worst_images(np.array([0.0, 0.0]), 5) == []


def test_worst_images_orders_and_breaks_ties_by_index() -> None:
    assert worst_images(np.array([1.0, 3.0, 3.0, 0.5]), 3) == [1, 2, 0]
    assert worst_images(np.array([1.0, 3.0]), 1) == [1]


def test_a_ok_b_fails_selection() -> None:
    a = np.array([0.0, 0.4, 2.0, 0.0, 0.0])
    b = np.array([3.0, 1.0, 5.0, 0.9, 2.0])
    # image 2: A fails too; image 3: B only 0.9 (<1.0 so not failing); order by gap then index
    assert a_ok_b_fails(a, b, k=5) == [0, 4, 1]
    assert a_ok_b_fails(b, a, k=5) == []
    assert a_ok_b_fails(a, b, k=1) == [0]


def test_all_models_fail_selection() -> None:
    s = {"x": np.array([2.0, 0.0, 1.0, 3.0]), "y": np.array([1.5, 4.0, 1.0, 0.2])}
    assert all_models_fail(s, k=5) == [0, 2]


def test_worst_per_class() -> None:
    r = rec(0, [[0, 0, 50, 50]], [0])
    miss = analyze_image(r, Detections.empty(), 0.25, 2)
    ok = analyze_image(r, Detections.make([[0, 0, 50, 50]], [0.9], [0]), 0.25, 2)
    out = worst_per_class([ok, miss, ok], [0, 1], k=2)
    assert out == {0: [1]}  # class 1 never errs -> omitted


def test_render_colours_and_size(tmp_path: Path) -> None:
    img = tmp_path / "bg.png"
    Image.fromarray(np.full((400, 400, 3), 100, dtype=np.uint8)).save(img)
    gt = GroundTruth.make([[20, 20, 120, 120], [200, 20, 300, 120]], [0, 0])
    r = ImageRecord(0, img, 400, 400, gt)
    det = Detections.make([[20, 20, 120, 120], [20, 250, 120, 350]], [0.9, 0.8], [0, 0])
    a = analyze_image(r, det, 0.25, 1)
    out = tmp_path / "t" / "x.jpg"
    size = render_thumbnail(r, a, ["thing"], out, max_side=200, quality=60)
    assert out.is_file() and size < 20_000
    im = np.asarray(Image.open(out).convert("RGB")).astype(int)
    assert im.shape[:2] == (200, 200)

    def near(color: tuple, x: int, y: int) -> bool:
        return any(
            np.abs(im[y + dy, x + dx] - color).sum() < 140 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        )

    # boxes scaled by 0.5: the TP box is blue (drawn over green), the FP red, missed GT yellow
    assert near(BLUE, 10, 40)
    assert near(RED, 10, 150)
    assert near(YELLOW, 100, 10) or near(YELLOW, 150, 30) or near(YELLOW, 100, 60)
    assert GREEN != BLUE


def test_build_galleries_end_to_end(tmp_path: Path) -> None:
    ds: Dataset = load_dataset(make_yolo_dataset(tmp_path / "d", n_images=6, n_classes=2, seed=3))
    perfect = [Detections.make(r.gt.boxes, [0.9] * len(r.gt), r.gt.classes) for r in ds.records]
    broken = [Detections.empty() if i % 2 == 0 else p for i, p in enumerate(perfect)]
    out = tmp_path / "out"
    g = build_galleries(ds, {"perfect": perfect, "broken": broken}, out, k=2, thumb_px=160)
    assert g["per_model"]["perfect"] == []
    assert len(g["per_model"]["broken"]) == 2
    assert (
        "perfect ok, broken fails" in g["a_ok_b_fails"]
        and g["a_ok_b_fails"]["broken ok, perfect fails"] == []
    )
    assert len(g["a_ok_b_fails"]["perfect ok, broken fails"]) == 2
    assert g["all_fail"] == []
    for item in g["per_model"]["broken"]:
        assert (out / item["thumb"]).is_file()
    assert set(g["per_class"]["broken"]) <= set(ds.names)
    assert len(analyze_dataset(ds, broken, 0.25)) == 6
