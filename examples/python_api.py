"""Use Duelscope from Python (no notebook needed)."""

from pathlib import Path

from duelscope.config import RunConfig
from duelscope.pipeline import run_comparison
from duelscope.report import build_report

cfg = RunConfig(
    data=Path("demo_dataset"),  # create with: duelscope demo-data demo_dataset
    models=["custom:examples/custom_adapter.py:EmptyDetector"],
    out=Path("report"),
    max_images=8,
)
results = run_comparison(cfg)
build_report(results, cfg.out)
for m in results["models"]:
    print(m["name"], f"mAP50-95 = {m['accuracy']['map5095']:.3f}")
