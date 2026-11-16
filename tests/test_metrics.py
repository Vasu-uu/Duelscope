from pathlib import Path

import numpy as np
import pytest

from duelscope.adapters import ClassMap
from duelscope.cache import PredictionCache
from duelscope.data import load_dataset
from duelscope.inference import run_inference
from duelscope.metrics import analyze_errors, classify_image, evaluate
from duelscope.metrics.errors import BKG, BOTH, CLS, DUPE, LOC, TP
from duelscope.synthetic import make_yolo_dataset
from duelscope.types import Dataset, Detections, GroundTruth, ImageRecord
from tests.helpers import MockDetector

ROOT = Path(__file__).resolve().parents[1]


def ds_of(*gts: GroundTruth, names: tuple[str, ...] = ("a", "b")) -> Dataset:
    recs = [ImageRecord(i, Path(f"im{i}.jpg"), 1000, 1000, g) for i, g in enumerate(gts)]
    return Dataset("t", "val", list(names), recs, Path("."))


def gt(boxes, classes, crowd=None) -> GroundTruth:
    return GroundTruth.make(boxes, classes, crowd)


def dt(boxes, scores, classes) -> Detections:
    return Detections.make(boxes, scores, classes)


def test_perfect_detections() -> None:
    g = gt([[0, 0, 100, 100], [200, 200, 300, 300]], [0, 1])
    r = evaluate(ds_of(g), [dt(g.boxes, [0.9, 0.8], g.classes)])
    assert r.map50 == pytest.approx(1.0) and r.map5095 == pytest.approx(1.0)
    assert r.f1 == pytest.approx(1.0) and r.n_gt == 2


def test_half_recall_gives_51_of_101() -> None:
    g = gt([[0, 0, 100, 100], [200, 200, 300, 300]], [0, 0])
    r = evaluate(ds_of(g), [dt([[0, 0, 100, 100]], [0.9], [0])])
    assert r.map50 == pytest.approx(51 / 101)
    assert r.per_class[0]["ap50"] == pytest.approx(51 / 101)
    assert r.per_class[1]["ap50"] is None  # class b has no GT


def test_iou_thresholds_averaged() -> None:
    g = gt([[0, 0, 10, 10]], [0])
    r = evaluate(ds_of(g), [dt([[0, 0, 10, 7.4]], [0.9], [0])])  # IoU 0.74 -> 5 of 10 thresholds
    assert r.map50 == pytest.approx(1.0)
    assert r.map5095 == pytest.approx(0.5)


def test_false_positive_ranked_first_halves_precision() -> None:
    g = gt([[0, 0, 100, 100]], [0])
    d = dt([[500, 500, 600, 600], [0, 0, 100, 100]], [0.9, 0.5], [0, 0])
    assert evaluate(ds_of(g), [d]).map50 == pytest.approx(0.5)


def test_wrong_class_scores_zero() -> None:
    g = gt([[0, 0, 100, 100]], [0])
    assert evaluate(ds_of(g), [dt(g.boxes, [0.9], [1])]).map50 == 0.0


def test_size_buckets() -> None:
    g = gt([[0, 0, 10, 10], [0, 0, 200, 200]], [0, 0])
    r = evaluate(ds_of(g), [dt(g.boxes, [0.9, 0.8], [0, 0])])
    assert r.map_small == pytest.approx(1.0) and r.map_large == pytest.approx(1.0)
    assert r.map_medium is None


def test_crowd_gt_does_not_create_false_positives() -> None:
    g = gt([[0, 0, 100, 100], [200, 200, 400, 400]], [0, 0], [False, True])
    d = dt([[0, 0, 100, 100], [250, 250, 300, 300]], [0.9, 0.8], [0, 0])
    assert evaluate(ds_of(g), [d]).map50 == pytest.approx(1.0)


def test_best_f1_threshold() -> None:
    g = gt([[0, 0, 100, 100], [200, 200, 300, 300]], [0, 0])
    d = dt([[0, 0, 100, 100], [200, 200, 300, 300], [600, 600, 700, 700]], [0.9, 0.8, 0.3], [0] * 3)
    r = evaluate(ds_of(g), [d])
    assert r.precision == pytest.approx(1.0) and r.recall == pytest.approx(1.0)
    assert 0.3 < r.best_conf <= 0.8


def test_prediction_count_mismatch_is_loud() -> None:
    with pytest.raises(ValueError, match="prediction sets"):
        evaluate(ds_of(gt([[0, 0, 5, 5]], [0])), [])


ERR_GT = gt([[0, 0, 100, 100], [200, 0, 300, 100]], [0, 1])
ERR_DET = dt(
    [
        [0, 0, 100, 100],  # TP
        [0, 0, 100, 100],  # duplicate
        [200, 0, 300, 100],  # right box, wrong class
        [200, 0, 300, 40],  # right class, IoU 0.4
        [500, 500, 520, 520],  # background
        [200, 0, 230, 100],  # wrong class and poor IoU (0.3)
    ],
    [0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    [0, 0, 0, 1, 1, 0],
)


def test_error_classification() -> None:
    types, ref, gt_score = classify_image(ERR_GT, ERR_DET)
    assert types.tolist() == [TP, DUPE, CLS, LOC, BKG, BOTH]
    assert gt_score.tolist() == pytest.approx([0.95, -1.0])
    assert ref[2] == 1 and ref[3] == 1


def test_error_taxonomy_counts_shares_and_oracle() -> None:
    res = analyze_errors(ds_of(ERR_GT), [ERR_DET], conf=0.25)
    c = res["counts"]
    assert (c["tp"], c["cls"], c["loc"], c["both"], c["dupe"], c["bkg"], c["miss"]) == (
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert res["total_errors"] == 6 and res["share"]["cls"] == pytest.approx(1 / 6)
    assert res["delta_map50"]["cls"] > 0 and res["delta_map50"]["miss"] > 0
    assert res["delta_map50"]["loc"] > 0
    assert res["per_class"]["b"]["miss"] == 1
    # raising conf above every detection turns all detections into misses
    hi = analyze_errors(ds_of(ERR_GT), [ERR_DET], conf=0.99, with_oracle=False)
    assert hi["counts"]["miss"] == 2 and hi["counts"]["tp"] == 0


def test_cache_roundtrip_and_reuse(tmp_path: Path) -> None:
    ds = load_dataset(make_yolo_dataset(tmp_path / "d", n_images=4))
    by_name = {r.path.name: r.gt for r in ds.records}
    det = MockDetector(
        lambda p: dt(by_name[p.name].boxes, [0.9] * len(by_name[p.name]), by_name[p.name].classes),
        names=ds.names,
    )
    cm = ClassMap.build(det.class_names, ds.names)
    cache = PredictionCache(tmp_path / "cache")
    p1, cached1 = run_inference(det, ds, cm, batch=2, cache=cache)
    assert not cached1 and det.calls == 4
    p2, cached2 = run_inference(det, ds, cm, batch=2, cache=cache)
    assert cached2 and det.calls == 4  # nothing re-run
    np.testing.assert_allclose(p1[0].boxes, p2[0].boxes)
    assert evaluate(ds, p2).map50 == pytest.approx(1.0)
    _, cached3 = run_inference(
        det, ds, cm, cache=PredictionCache(tmp_path / "cache", enabled=False)
    )
    assert not cached3 and det.calls == 8


@pytest.mark.ultralytics
def test_metric_math_matches_ultralytics_val(tmp_path: Path) -> None:
    """Feed the exact predictions model.val() produced into Duelscope's evaluator (tol ~0.01).

    This isolates the metric implementation from preprocessing differences between
    Ultralytics' rect-padded validation and plain predict (see docs/metrics.md).
    """
    pytest.importorskip("ultralytics")
    w = ROOT / "weights" / "yolov8n.pt"
    if not w.is_file():
        pytest.skip("weights/yolov8n.pt not available")
    import yaml
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    try:
        info = check_det_dataset("coco8.yaml")
    except Exception as exc:  # offline
        pytest.skip(f"coco8 not downloadable: {exc}")
    cfg = {
        "path": str(info["path"]),
        "train": "images/train",
        "val": "images/val",
        "names": dict(info["names"]),
    }
    ypath = ROOT / "weights" / "coco8_local.yaml"
    ypath.write_text(yaml.safe_dump(cfg))
    r = YOLO(str(w)).val(
        data=str(ypath),
        imgsz=640,
        conf=0.001,
        iou=0.6,
        device="cpu",
        plots=False,
        verbose=False,
        batch=1,
        save_txt=True,
        save_conf=True,
        project=str(tmp_path),
        name="val",
        exist_ok=True,
    )
    ds = load_dataset(ypath)
    preds = []
    for rec in ds.records:
        f = Path(r.save_dir) / "labels" / f"{rec.path.stem}.txt"
        rows = np.array([ln.split() for ln in f.read_text().splitlines()], dtype=np.float64)
        rows = rows.reshape(-1, 6)
        c, cx, cy, bw, bh, conf = rows.T
        boxes = np.stack(
            [
                (cx - bw / 2) * rec.width,
                (cy - bh / 2) * rec.height,
                (cx + bw / 2) * rec.width,
                (cy + bh / 2) * rec.height,
            ],
            axis=1,
        )
        preds.append(Detections.make(boxes, conf, c))
    ours = evaluate(ds, preds)
    print(
        f"\nduelscope map50={ours.map50:.4f} map5095={ours.map5095:.4f} | "
        f"ultralytics map50={r.box.map50:.4f} map5095={r.box.map:.4f}"
    )
    # Ultralytics integrates a trapezoid over the PR envelope and caps a perfect class at
    # 0.995; COCO uses 101-point sampling. Expect a small, systematic offset.
    assert abs(ours.map50 - r.box.map50) < 0.015
    assert abs(ours.map5095 - r.box.map) < 0.01
