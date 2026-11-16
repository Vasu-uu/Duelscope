import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from duelscope.config import RunConfig, SpeedConfig
from duelscope.data import load_dataset
from duelscope.pipeline import run_comparison
from duelscope.report import build_report, load_results, make_insights, pareto_front
from duelscope.synthetic import make_yolo_dataset
from duelscope.types import Detections
from tests.helpers import MockDetector

SNAP = Path(__file__).parent / "snapshots" / "report_structure.json"


def _detectors(ds_path: Path):
    ds = load_dataset(ds_path)
    gts = {r.path.name: r.gt for r in ds.records}

    def good(p: Path) -> Detections:
        g = gts.get(p.name)
        if g is None:  # warm-up / array inputs
            return Detections.empty()
        return Detections.make(g.boxes, np.full(len(g), 0.9), g.classes)

    def weak(p: Path) -> Detections:
        g = gts.get(p.name)
        if g is None:
            return Detections.empty()
        n = len(g) // 2
        jitter = g.boxes[:n] + 6
        extra = np.array([[5, 5, 40, 40]])
        return Detections.make(
            np.concatenate([jitter, extra]),
            np.concatenate([np.full(n, 0.7), [0.6]]),
            np.concatenate([g.classes[:n], [0]]),
        )

    return [MockDetector(good, "good", ds.names), MockDetector(weak, "weak", ds.names)]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("rep")
    yaml_path = make_yolo_dataset(root / "ds", n_images=8, n_classes=3, seed=2)
    cfg = RunConfig(
        data=yaml_path,
        models=["good", "weak"],
        out=root / "out",
        speed=SpeedConfig(images=3, warmup=1, iters=3, batch_sizes=[1, 2]),
        cache_dir=root / "cache",
    )
    results = run_comparison(cfg, detectors=_detectors(yaml_path))
    paths = build_report(results, cfg.out)
    return results, paths, cfg.out


def test_results_structure(run) -> None:
    results, _, out = run
    assert results["schema_version"] == 1 and [m["name"] for m in results["models"]] == [
        "good",
        "weak",
    ]
    good, weak = results["models"]
    assert good["accuracy"]["map50"] == pytest.approx(1.0)
    assert weak["accuracy"]["map5095"] < good["accuracy"]["map5095"]
    assert good["speed"]["latency_ms"]["p50"] > 0 and results["hardware"]["cpu"]
    assert results["gallery"]["per_model"]["weak"]
    assert (out / results["gallery"]["per_model"]["weak"][0]["thumb"]).is_file()
    assert any("mAP50-95 over weak" in s for s in results["insights"])


def test_html_is_self_contained(run) -> None:
    _, paths, _ = run
    html = paths["html"].read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]+src=|<link[^>]+href=|https?://|@import", html)
    assert html.count("data:image/png;base64,") >= 3 and "data:image/jpeg;base64," in html
    assert "good" in html and "weak" in html and "sortable" in html
    assert paths["html"].stat().st_size < 3_000_000


def test_markdown_and_json_outputs(run) -> None:
    results, paths, _ = run
    md = paths["markdown"].read_text(encoding="utf-8")
    assert "| Model | mAP50-95 |" in md and "## Key findings" in md and "charts/pr.png" in md
    assert json.loads(paths["json"].read_text())["models"][0]["name"] == "good"
    assert load_results(paths["json"])["dataset"]["hash"] == results["dataset"]["hash"]


def test_rebuild_from_results_json(run, tmp_path: Path) -> None:
    results, paths, out = run
    again = build_report(load_results(paths["json"]), tmp_path / "again", gallery_root=out)
    assert "data:image/jpeg;base64," in again["html"].read_text(encoding="utf-8")


def test_report_structure_snapshot(run) -> None:
    _, paths, _ = run
    html = paths["html"].read_text(encoding="utf-8")
    md = paths["markdown"].read_text(encoding="utf-8")
    structure = {
        "sections": re.findall(r'<section class="card" id="([a-z]+)"', html),
        "leaderboard_columns": re.findall(
            r"<th>([^<]+)</th>", html.split("</thead>")[0].split("<table")[1]
        ),
        "markdown_headings": re.findall(r"^#+ .*$", md, flags=re.M)[1:],
        "charts": sorted(p.name for p in (paths["html"].parent / "charts").glob("*.png")),
    }
    if os.environ.get("UPDATE_SNAPSHOTS") or not SNAP.exists():
        SNAP.write_text(json.dumps(structure, indent=2) + "\n")
    assert structure == json.loads(SNAP.read_text())


def _model(name: str, ap: float, lat: float | None, classes=None, errors=None) -> dict:
    return {
        "name": name,
        "metadata": {},
        "warnings": [],
        "accuracy": {
            "map5095": ap,
            "map50": ap + 0.1,
            "per_class": classes or [{"name": "a", "n_gt": 5, "ap5095": ap}],
        },
        "speed": {"latency_ms": {"p50": lat}} if lat else None,
        "errors": errors or {"per_class": {}, "delta_map50": {}},
    }


def test_insight_sentences_are_exact() -> None:
    res = {"models": [_model("weak", 0.40, 5.0), _model("good", 0.50, 10.0)]}
    out = make_insights(res)
    assert out[0] == "good is the most accurate model at 50.0 mAP50-95."
    assert "good is +10.0 mAP50-95 over weak at 2.0x the latency." in out
    assert any(
        s.startswith("weak is the fastest (5.0 ms p50) and gives up 10.0 mAP50-95") for s in out
    )
    assert any(s.startswith("Pareto-optimal") and "weak, good" in s for s in out)


def test_weakest_class_names_dominant_error() -> None:
    classes = [
        {"name": "cat", "n_gt": 3, "ap5095": 0.9},
        {"name": "dog", "n_gt": 3, "ap5095": 0.21},
    ]
    errors = {"per_class": {"dog": {"cls": 1, "loc": 7, "miss": 2}}, "delta_map50": {"loc": 0.08}}
    out = make_insights({"models": [_model("m", 0.5, None, classes, errors)]})
    assert "m is weakest on class 'dog' (AP 0.21), mostly due to poor localization." in out
    assert any("fixing poor localization would add the most (+8.0 mAP50)" in s for s in out)


def test_pareto_front_drops_dominated_models() -> None:
    ms = [_model("a", 0.5, 10), _model("b", 0.4, 12), _model("c", 0.6, 20), _model("d", 0.3, 5)]
    assert pareto_front(ms) == ["d", "a", "c"]


def test_load_results_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot read"):
        load_results(tmp_path / "none.json")
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(ValueError, match="not a Duelscope"):
        load_results(tmp_path / "x.json")
