import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from duelscope.cli import app
from duelscope.data import DatasetError, content_hash, load_dataset, subset, summarize
from duelscope.synthetic import make_yolo_dataset

runner = CliRunner()


@pytest.fixture()
def yolo(tmp_path: Path) -> Path:
    return make_yolo_dataset(tmp_path / "ds", n_images=8, n_classes=3, seed=1)


def test_load_yolo_basic(yolo: Path) -> None:
    ds = load_dataset(yolo)
    assert len(ds) == 8
    assert ds.names == ["square", "wide", "tall"]
    r = ds.records[0]
    assert r.width == 320 and r.height == 240
    assert (r.gt.boxes[:, 2] > r.gt.boxes[:, 0]).all()
    assert r.gt.boxes.max() <= 320


def test_box_conversion_exact(tmp_path: Path) -> None:
    p = make_yolo_dataset(tmp_path / "d", n_images=1, seed=0)
    (tmp_path / "d" / "labels" / "val" / "img_000.txt").write_text("1 0.5 0.5 0.5 0.5\n")
    gt = load_dataset(p).records[0].gt
    np.testing.assert_allclose(gt.boxes[0], [80, 60, 240, 180])
    assert gt.classes[0] == 1


def test_yolo_issues(yolo: Path) -> None:
    lbl = yolo.parent / "labels" / "val"
    (lbl / "img_000.txt").write_text("9 0.5 0.5 0.2 0.2\n0 0.5 0.5\nabc 1 2 3 4\n0 0.5 0.5 0 0.2\n")
    (lbl / "img_001.txt").unlink()
    (lbl / "img_002.txt").write_text("")
    ds = load_dataset(yolo)
    kinds = {i.kind for i in ds.issues}
    assert {"class_out_of_range", "malformed_box", "missing_label", "empty_image"} <= kinds
    assert len(ds.records[0].gt) == 0
    assert len(ds.records[1].gt) == 0


def test_polygon_label_becomes_box(yolo: Path) -> None:
    (yolo.parent / "labels" / "val" / "img_000.txt").write_text(
        "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n"
    )
    gt = load_dataset(yolo).records[0].gt
    np.testing.assert_allclose(gt.boxes[0], [32, 24, 160, 120], atol=1e-3)


def test_errors_have_fix_hints(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="does not exist"):
        load_dataset(tmp_path / "nope")
    with pytest.raises(DatasetError, match="No data.yaml"):
        load_dataset(tmp_path)
    p = make_yolo_dataset(tmp_path / "d")
    with pytest.raises(DatasetError, match="no 'train' split"):
        load_dataset(p, split="train")


def test_coco_loader(yolo: Path, tmp_path: Path) -> None:
    ds = load_dataset(yolo)
    images, anns = [], []
    for r in ds.records:
        images.append(
            {"id": r.image_id + 1, "file_name": str(r.path), "width": r.width, "height": r.height}
        )
        for b, c in zip(r.gt.boxes, r.gt.classes, strict=True):
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": r.image_id + 1,
                    "category_id": int(c) + 10,
                    "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                }
            )
    anns.append({"id": 999, "image_id": 1, "category_id": 77, "bbox": [0, 0, 5, 5]})
    cats = [{"id": i + 10, "name": n} for i, n in enumerate(ds.names)]
    jp = tmp_path / "ann.json"
    jp.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    coco = load_dataset(jp)
    assert coco.names == ds.names
    assert sum(len(r.gt) for r in coco.records) == sum(len(r.gt) for r in ds.records)
    assert any(i.kind == "class_out_of_range" for i in coco.issues)
    np.testing.assert_allclose(coco.records[0].gt.boxes, ds.records[0].gt.boxes, atol=1e-3)


def test_subset_and_hash(yolo: Path) -> None:
    ds = load_dataset(yolo)
    a, b = subset(ds, 3, seed=5), subset(ds, 3, seed=5)
    assert [r.path for r in a.records] == [r.path for r in b.records]
    assert len(a) == 3 and subset(ds, None) is ds
    assert content_hash(ds) == content_hash(load_dataset(yolo))
    assert content_hash(ds) != content_hash(a)
    with pytest.raises(DatasetError):
        subset(ds, 0)


def test_summary_counts(yolo: Path) -> None:
    ds = load_dataset(yolo)
    s = summarize(ds)
    assert s["images"] == 8
    assert s["boxes"] == sum(s["classes"].values()) == sum(len(r.gt) for r in ds.records)


def test_check_data_cli(yolo: Path) -> None:
    res = runner.invoke(app, ["check-data", "--data", str(yolo)])
    assert res.exit_code == 0, res.output
    assert "Class counts" in res.output
    bad = runner.invoke(app, ["check-data", "--data", str(yolo.parent / "missing")])
    assert bad.exit_code == 1 and "Error" in bad.output
