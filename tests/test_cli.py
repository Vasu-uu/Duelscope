import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from duelscope.cli import app, expand_models_args
from duelscope.config import SAMPLE_CONFIG, ConfigError, build_config, load_config
from duelscope.synthetic import make_yolo_dataset

runner = CliRunner()

MOCK = '''
from pathlib import Path

import numpy as np
from PIL import Image

from duelscope.adapters import Detector
from duelscope.types import Detections


class Oracle(Detector):
    """Reads the YOLO label next to each image and returns it (with optional jitter)."""
    JITTER = 0.0

    @property
    def name(self):
        return type(self).__name__.lower()

    def load(self):
        pass

    def predict(self, images):
        out = []
        for p in images:
            if isinstance(p, np.ndarray):
                out.append(Detections.empty())
                continue
            w, h = Image.open(p).size
            lab = Path(str(p).replace("images", "labels")).with_suffix(".txt")
            rows = np.array([l.split() for l in lab.read_text().splitlines() if l.strip()], dtype=float)
            rows = rows.reshape(-1, 5)
            c, cx, cy, bw, bh = rows.T
            boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1) + self.JITTER
            out.append(Detections.make(boxes, np.full(len(c), 0.9), c))
        self.last_timing = {"preprocess": 0.1, "inference": 0.5, "postprocess": 0.1}
        return out


class Sloppy(Oracle):
    JITTER = 9.0
'''


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    make_yolo_dataset(tmp_path / "ds", n_images=6, n_classes=3, seed=4)
    (tmp_path / "mock.py").write_text(MOCK)
    return tmp_path


def _write_cfg(root: Path, **extra) -> Path:
    cfg = {
        "data": "ds",
        "out": "rep",
        "models": ["custom:mock.py:Oracle", "custom:mock.py:Sloppy"],
        "speed": {"images": 2, "warmup": 1, "iters": 3, "batch_sizes": [1, 2]},
        "gallery": {"k": 2, "thumb_px": 160},
        **extra,
    }
    p = root / "duelscope.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_expand_models_args() -> None:
    assert expand_models_args(["run", "--models", "a.pt", "b.pt", "--out", "x"]) == [
        "run",
        "--models",
        "a.pt",
        "--models",
        "b.pt",
        "--out",
        "x",
    ]
    assert expand_models_args(["run", "--models", "a.pt"]) == ["run", "--models", "a.pt"]


def test_sample_config_is_valid_yaml_and_model() -> None:
    raw = yaml.safe_load(SAMPLE_CONFIG)
    cfg = build_config(raw)
    assert cfg.models[:2] == ["yolov8n.pt", "yolov8s.pt"] and cfg.speed.iters == 30


def test_config_errors_are_readable(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="imgsz"):
        build_config({"data": "d", "models": ["a.pt"], "imgsz": 3})
    with pytest.raises(ConfigError, match="duplicate"):
        build_config({"data": "d", "models": ["a.pt", "a.pt"]})
    with pytest.raises(ConfigError, match="bogus"):
        build_config({"data": "d", "models": ["a.pt"], "bogus": 1})
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)
    ok = tmp_path / "c.yaml"
    ok.write_text("data: ds\nmodels: [a.pt]\n")
    assert Path(load_config(ok)["data"]) == (tmp_path / "ds").resolve()


def test_basic_commands(project: Path) -> None:
    assert "duelscope" in runner.invoke(app, ["--version"]).output
    helptxt = runner.invoke(app, ["--help"]).output
    for cmd in ("run", "check-data", "report", "list-adapters", "doctor", "init"):
        assert cmd in helptxt
    assert "ultralytics" in runner.invoke(app, ["list-adapters"]).output
    assert runner.invoke(app, ["doctor"]).exit_code == 0
    assert runner.invoke(app, ["init"]).exit_code == 0
    again = runner.invoke(app, ["init"])
    assert again.exit_code == 1 and "already exists" in again.output
    assert runner.invoke(app, ["demo-data", "demo", "--images", "3"]).exit_code == 0
    assert (project / "demo" / "data.yaml").is_file()


def test_run_end_to_end_with_config(project: Path) -> None:
    cfgp = _write_cfg(project)
    res = runner.invoke(app, ["run", "--config", str(cfgp)])
    assert res.exit_code == 0, res.output
    out = project / "rep"
    for f in ("report.html", "report.md", "results.json"):
        assert (out / f).is_file()
    data = json.loads((out / "results.json").read_text())
    names = [m["name"] for m in data["models"]]
    assert names == ["oracle", "sloppy"]
    assert data["models"][0]["accuracy"]["map50"] > data["models"][1]["accuracy"]["map5095"]
    assert "oracle" in res.output and "Report written" in res.output
    # second run hits the prediction cache
    again = json.loads(
        (
            runner.invoke(app, ["run", "--config", str(cfgp), "--no-gallery"]),
            (out / "results.json").read_text(),
        )[1]
    )
    assert all(m["cached_predictions"] for m in again["models"])
    nocache = runner.invoke(
        app, ["run", "--config", str(cfgp), "--no-cache", "--no-speed", "--no-gallery"]
    )
    assert nocache.exit_code == 0
    d = json.loads((out / "results.json").read_text())
    assert not any(m["cached_predictions"] for m in d["models"]) and d["models"][0]["speed"] is None


def test_cli_overrides_and_report_rebuild(project: Path) -> None:
    cfgp = _write_cfg(project)
    res = runner.invoke(
        app, ["run", "--config", str(cfgp), "--max-images", "3", "--out", "small", "--seed", "1"]
    )
    assert res.exit_code == 0, res.output
    data = json.loads((project / "small" / "results.json").read_text())
    assert data["dataset"]["images"] == 3 and data["settings"]["seed"] == 1
    rebuilt = runner.invoke(
        app,
        [
            "report",
            "--from",
            str(project / "small" / "results.json"),
            "--out",
            str(project / "rebuilt"),
        ],
    )
    assert rebuilt.exit_code == 0 and (project / "rebuilt" / "report.html").is_file()
    bad = runner.invoke(app, ["report", "--from", str(project / "nope.json")])
    assert bad.exit_code == 1 and "Cannot read" in bad.output


def test_run_errors_name_the_problem(project: Path) -> None:
    res = runner.invoke(app, ["run", "--data", "ds"])
    assert res.exit_code == 1 and "Missing 'models'" in res.output
    res = runner.invoke(app, ["run", "--data", "nodir", "--models", "x.pt"])
    assert res.exit_code == 1 and "does not exist" in res.output
    res = runner.invoke(app, ["run", "--data", "ds", "--models", "nonsense:thing"])
    assert res.exit_code == 1 and "Unknown adapter" in res.output
    res = runner.invoke(
        app, ["run", "--data", "ds", "--models", "custom:missing.py:X", "--no-speed"]
    )
    assert res.exit_code == 1 and "not found" in res.output
