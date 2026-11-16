"""Report generation: results.json, report.md and a self-contained report.html."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from duelscope.report import charts
from duelscope.report.html import render_html
from duelscope.report.insights import make_insights, pareto_front
from duelscope.report.markdown import render_markdown

__all__ = ["build_report", "load_results", "make_insights", "pareto_front"]


class _Encoder(json.JSONEncoder):
    def default(self, o: object):  # noqa: ANN201
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def load_results(path: Path) -> dict:
    """Read a results.json, with a helpful error for missing or wrong files."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read results file '{path}': {exc}. Run 'duelscope run' first."
        ) from exc
    if "models" not in data or "dataset" not in data:
        raise ValueError(f"'{path}' is not a Duelscope results.json (no 'models'/'dataset').")
    return data


def build_report(results: dict, out_dir: Path, gallery_root: Path | None = None) -> dict[str, Path]:
    """Write results.json, report.md, report.html and chart PNGs into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(results["models"], key=lambda m: -m["accuracy"]["map5095"])
    pngs = {
        "speed": charts.speed_scatter(models),
        "classes": charts.class_heatmap(models),
        "pr": charts.pr_curves(models),
        "errors": charts.error_bars(models),
    }
    files: dict[str, str] = {}
    (out_dir / "charts").mkdir(exist_ok=True)
    for k, png in pngs.items():
        if png is not None:
            (out_dir / "charts" / f"{k}.png").write_bytes(png)
            files[k] = f"charts/{k}.png"
    paths = {
        "json": out_dir / "results.json",
        "markdown": out_dir / "report.md",
        "html": out_dir / "report.html",
    }
    paths["json"].write_text(json.dumps(results, indent=2, cls=_Encoder))
    paths["markdown"].write_text(render_markdown(results, files), encoding="utf-8")
    paths["html"].write_text(render_html(results, pngs, gallery_root or out_dir), encoding="utf-8")
    return paths
