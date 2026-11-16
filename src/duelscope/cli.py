"""Duelscope command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from duelscope import __version__
from duelscope.adapters import AdapterError, available_adapters
from duelscope.config import SAMPLE_CONFIG, ConfigError, build_config, load_config
from duelscope.data import DatasetError, load_dataset, summarize
from duelscope.doctor import FAIL, OK, run_checks

app = typer.Typer(
    name="duelscope",
    help="Compare object detectors: mAP, per-class errors, failure gallery and speed.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"duelscope {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """Duelscope entry point."""


def fail(message: str) -> typer.Exit:
    """Print a red error and return an Exit(1) to raise."""
    console.print(f"[bold red]Error:[/bold red] {message}")
    return typer.Exit(1)


def expand_models_args(argv: list[str]) -> list[str]:
    """Turn `--models a b c` into `--models a --models b --models c` (values stop at a flag)."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        out.append(argv[i])
        if argv[i] == "--models":
            i += 1
            first = True
            while i < len(argv) and not argv[i].startswith("--"):
                if not first:
                    out.append("--models")
                out.append(argv[i])
                first = False
                i += 1
            continue
        i += 1
    return out


def cli() -> None:
    """Console-script entry point (adds space-separated --models support)."""
    app(args=expand_models_args(sys.argv[1:]))


@app.command("check-data")
def check_data(
    data: Path = typer.Option(..., "--data", help="YOLO data.yaml (or folder) or COCO .json."),
    split: str = typer.Option("val", help="Split to load for YOLO datasets."),
    fmt: str = typer.Option("auto", "--format", help="yolo | coco | auto."),
    images_dir: Path | None = typer.Option(None, help="Image folder for COCO files."),
    max_issues: int = typer.Option(15, help="Maximum issues to list."),
) -> None:
    """Validate a dataset and print a summary table."""
    try:
        ds = load_dataset(data, split, fmt, images_dir)
    except DatasetError as exc:
        raise fail(str(exc)) from exc
    s = summarize(ds)
    t = Table(title=f"Dataset '{s['name']}' (split: {s['split']})")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_row("Images", str(s["images"]))
    t.add_row("Boxes", str(s["boxes"]))
    t.add_row("Images without boxes", str(s["empty_images"]))
    for key, label in (
        ("width", "Width px"),
        ("height", "Height px"),
        ("box_area_px", "Box area px"),
    ):
        v = s[key]
        t.add_row(label, f"min {v['min']:.0f} / median {v['median']:.0f} / max {v['max']:.0f}")
    console.print(t)
    ct = Table(title="Class counts")
    ct.add_column("Id", justify="right")
    ct.add_column("Class")
    ct.add_column("Boxes", justify="right")
    for i, (n, c) in enumerate(s["classes"].items()):
        ct.add_row(str(i), n, str(c))
    console.print(ct)
    shown = [i for i in ds.issues if i.level != "info"]
    if shown:
        it = Table(title=f"Issues ({len(shown)})")
        for col in ("Level", "Kind", "Where", "Message"):
            it.add_column(col)
        for i in shown[:max_issues]:
            colour = "red" if i.level == "error" else "yellow"
            it.add_row(f"[{colour}]{i.level}[/{colour}]", i.kind, i.path, i.message)
        console.print(it)
        if len(shown) > max_issues:
            console.print(f"... and {len(shown) - max_issues} more.")
    else:
        console.print("[green]No problems found.[/green]")
    if s["issues"].get("error"):
        raise typer.Exit(2)


def _summary_table(results: dict) -> Table:
    t = Table(title="Results (sorted by mAP50-95)")
    for col in ("Model", "mAP50-95", "mAP50", "Precision", "Recall", "p50 ms", "FPS", "Params (M)"):
        t.add_column(col, justify="left" if col == "Model" else "right")
    for m in sorted(results["models"], key=lambda m: -m["accuracy"]["map5095"]):
        a, s, p = m["accuracy"], m.get("speed"), m["metadata"].get("params")
        t.add_row(
            m["name"],
            f"{a['map5095'] * 100:.1f}",
            f"{a['map50'] * 100:.1f}",
            f"{a['precision'] * 100:.1f}",
            f"{a['recall'] * 100:.1f}",
            f"{s['latency_ms']['p50']:.1f}" if s else "-",
            f"{s['fps_bs1']:.1f}" if s else "-",
            f"{p / 1e6:.2f}" if p else "-",
        )
    return t


@app.command()
def run(
    config: Path | None = typer.Option(
        None, "--config", help="YAML config (see 'duelscope init')."
    ),
    data: Path | None = typer.Option(
        None, "--data", help="Dataset path (YOLO yaml/folder or COCO json)."
    ),
    models: list[str] | None = typer.Option(
        None, "--models", help="Model specs: yolov8n.pt a.onnx torchvision:name custom:f.py:Cls"
    ),
    out: Path | None = typer.Option(None, "--out", help="Output folder (default: report)."),
    split: str | None = typer.Option(None, help="Dataset split (YOLO)."),
    fmt: str | None = typer.Option(None, "--format", help="yolo | coco | auto."),
    images_dir: Path | None = typer.Option(None, help="Image folder for COCO files."),
    device: str | None = typer.Option(None, help="cpu | cuda | auto."),
    imgsz: int | None = typer.Option(None, help="Inference image size."),
    conf: float | None = typer.Option(None, help="Threshold for galleries/error analysis."),
    iou: float | None = typer.Option(None, help="NMS IoU threshold."),
    batch: int | None = typer.Option(None, help="Inference batch size."),
    max_images: int | None = typer.Option(None, help="Use at most N images (quick smoke run)."),
    seed: int | None = typer.Option(None, help="Random seed."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore and do not write the cache."),
    no_speed: bool = typer.Option(False, "--no-speed", help="Skip the speed benchmark."),
    no_gallery: bool = typer.Option(False, "--no-gallery", help="Skip failure galleries."),
) -> None:
    """Compare detectors on a dataset and write results.json, report.md and report.html."""
    from duelscope.pipeline import run_comparison
    from duelscope.report import build_report

    try:
        values = load_config(config) if config else {}
        overrides = {
            "data": data,
            "models": list(models) if models else None,
            "out": out,
            "split": split,
            "format": fmt,
            "images_dir": images_dir,
            "device": device,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "batch": batch,
            "max_images": max_images,
            "seed": seed,
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        if no_cache:
            values["cache"] = False
        if no_speed:
            values["speed"] = {**values.get("speed", {}), "enabled": False}
        if no_gallery:
            values["gallery"] = {**values.get("gallery", {}), "enabled": False}
        for need in ("data", "models"):
            if not values.get(need):
                raise ConfigError(
                    f"Missing '{need}'. Pass --{need} on the command line or set it in a config "
                    "file (create one with 'duelscope init')."
                )
        cfg = build_config(values)
    except ConfigError as exc:
        raise fail(str(exc)) from exc

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    task = progress.add_task("starting", total=1)

    def on_progress(stage: str, done: int, total: int) -> None:
        progress.update(task, description=stage, completed=done, total=max(total, 1))

    try:
        with progress:
            results = run_comparison(cfg, on_progress)
    except (DatasetError, AdapterError) as exc:
        raise fail(str(exc)) from exc
    paths = build_report(results, cfg.out)
    for m in results["models"]:
        for w in m["warnings"]:
            console.print(f"[yellow]WARNING:[/yellow] {w}")
    console.print(_summary_table(results))
    for line in results["insights"][:6]:
        console.print(f"  - {line}")
    console.print(f"\n[green]Report written:[/green] {paths['html']}")
    console.print(f"Also: {paths['markdown']} and {paths['json']}")


@app.command()
def report(
    results: Path = typer.Option(..., "--from", help="results.json from a previous run."),
    out: Path | None = typer.Option(
        None, "--out", help="Output folder (default: next to the JSON)."
    ),
) -> None:
    """Rebuild report.html and report.md from results.json without rerunning inference."""
    from duelscope.report import build_report, load_results

    try:
        data = load_results(results)
    except ValueError as exc:
        raise fail(str(exc)) from exc
    paths = build_report(data, out or results.parent, gallery_root=results.parent)
    console.print(f"[green]Report rebuilt:[/green] {paths['html']}")


@app.command("list-adapters")
def list_adapters() -> None:
    """Show available detector adapters and how to use them."""
    usage = {
        "ultralytics": "yolov8n.pt | ultralytics:yolo11s.pt",
        "onnx": "model.onnx | onnx:path/model.onnx",
        "torchvision": "torchvision:fasterrcnn_resnet50_fpn",
        "custom": "custom:path/to/file.py:ClassName",
    }
    t = Table(title="Detector adapters")
    for col in ("Adapter", "Origin", "Example model spec"):
        t.add_column(col)
    for name, origin in available_adapters().items():
        t.add_row(name, origin, usage.get(name, f"{name}:<argument>"))
    console.print(t)
    console.print("Third parties can add adapters via the 'duelscope.adapters' entry-point group.")


@app.command()
def doctor() -> None:
    """Check Python, CUDA and optional extras, and print fixes for anything missing."""
    t = Table(title="Duelscope doctor")
    for col in ("Check", "Status", "Detail"):
        t.add_column(col)
    checks = run_checks()
    for c in checks:
        colour = {"ok": "green", "warn": "yellow", "fail": "red"}[c.status]
        t.add_row(c.name, f"[{colour}]{c.status.upper()}[/{colour}]", c.detail)
    console.print(t)
    fixes = [c for c in checks if c.status != OK and c.fix]
    for c in fixes:
        console.print(f"[bold]{c.name}:[/bold] {c.fix}")
    if not fixes:
        console.print("[green]Everything looks good.[/green]")
    if any(c.status == FAIL for c in checks):
        raise typer.Exit(1)


@app.command()
def init(
    path: Path = typer.Option(Path("duelscope.yaml"), help="Where to write the sample config."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
) -> None:
    """Write a commented sample configuration file."""
    if path.exists() and not force:
        raise fail(
            f"'{path}' already exists. Use --force to overwrite or --path to choose another."
        )
    path.write_text(SAMPLE_CONFIG)
    console.print(
        f"[green]Wrote[/green] {path}. Edit 'data' and 'models', then run: "
        f"duelscope run --config {path}"
    )


@app.command("demo-data")
def demo_data(
    out: Path = typer.Argument(Path("demo_dataset"), help="Folder to create."),
    images: int = typer.Option(12, help="Number of synthetic images."),
    seed: int = typer.Option(0),
) -> None:
    """Create a tiny synthetic YOLO dataset for trying Duelscope without downloads."""
    from duelscope.synthetic import make_yolo_dataset

    yaml_path = make_yolo_dataset(out, n_images=images, seed=seed)
    console.print(f"[green]Created[/green] {yaml_path}  (classes: square, wide, tall)")


@app.command("compare-reports")
def compare_reports(
    a: Path = typer.Argument(..., help="Baseline results.json."),
    b: Path = typer.Argument(..., help="New results.json."),
) -> None:
    """Compare two runs: metric deltas (b minus a) for models present in both."""
    from duelscope.report import load_results

    try:
        ra, rb = load_results(a), load_results(b)
    except ValueError as exc:
        raise fail(str(exc)) from exc
    ma = {m["name"]: m for m in ra["models"]}
    shared = [m for m in rb["models"] if m["name"] in ma]
    if not shared:
        raise fail("The two reports share no model names, so there is nothing to compare.")
    if ra["dataset"]["hash"] != rb["dataset"]["hash"]:
        console.print(
            "[yellow]WARNING:[/yellow] the datasets differ; deltas mix data and model changes."
        )
    t = Table(title=f"{b.name} minus {a.name}")
    for col in ("Model", "mAP50-95", "mAP50", "p50 ms"):
        t.add_column(col, justify="left" if col == "Model" else "right")
    for m in shared:
        o = ma[m["name"]]
        d = [
            (m["accuracy"]["map5095"] - o["accuracy"]["map5095"]) * 100,
            (m["accuracy"]["map50"] - o["accuracy"]["map50"]) * 100,
        ]
        row = [m["name"], f"{d[0]:+.1f}", f"{d[1]:+.1f}"]
        if m.get("speed") and o.get("speed"):
            row.append(f"{m['speed']['latency_ms']['p50'] - o['speed']['latency_ms']['p50']:+.1f}")
        else:
            row.append("-")
        t.add_row(*row)
    console.print(t)


@app.command()
def export(
    weights: Path = typer.Argument(..., help="Ultralytics .pt weights to export."),
    imgsz: int = typer.Option(640, help="Export image size."),
    half: bool = typer.Option(False, "--half", help="Export FP16 (needs a CUDA device)."),
) -> None:
    """Export Ultralytics weights to ONNX and print the command that compares both."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise fail(
            "Ultralytics is not installed. Run: pip install 'duelscope[ultralytics]'"
        ) from exc
    if not weights.is_file():
        raise fail(f"Weights file '{weights}' not found. Check the path.")
    try:
        onnx_path = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, half=half)
    except Exception as exc:
        raise fail(f"Export failed: {exc}. FP16 export requires a CUDA GPU.") from exc
    console.print(f"[green]Exported[/green] {onnx_path}")
    console.print(f"Compare them:  duelscope run --data <data.yaml> --models {weights} {onnx_path}")


@app.command("quant-compare")
def quant_compare(
    data: Path = typer.Option(..., "--data", help="Dataset used for calibration and evaluation."),
    model: Path = typer.Option(
        ..., "--model", help="FP32 .onnx file, or Ultralytics .pt to export."
    ),
    out: Path = typer.Option(Path("quant_report"), "--out", help="Output folder."),
    split: str = typer.Option("val", help="Dataset split (YOLO)."),
    calib: int = typer.Option(32, help="Calibration images for INT8."),
    imgsz: int = typer.Option(640),
    device: str = typer.Option("cpu", help="cpu | cuda | auto (INT8 QDQ is CPU-oriented)."),
    max_images: int | None = typer.Option(None, help="Use at most N evaluation images."),
    seed: int = typer.Option(0),
) -> None:
    """Build FP32/FP16/INT8 ONNX variants of one model and compare accuracy, speed and size."""
    from duelscope.data import subset
    from duelscope.pipeline import run_comparison
    from duelscope.quant import make_variants, quant_insights, quantisation_table
    from duelscope.report import build_report

    try:
        onnx_path = model
        if model.suffix == ".pt":
            from ultralytics import YOLO

            onnx_path = Path(YOLO(str(model)).export(format="onnx", imgsz=imgsz))
        ds = load_dataset(data, split)
        calib_imgs = [r.path for r in subset(ds, calib, seed).records]
        console.print(f"Building variants from {onnx_path} ...")
        variants = make_variants(onnx_path, out / "variants", calib_imgs, imgsz)
        cfg = build_config(
            {
                "data": data,
                "split": split,
                "models": [str(p) for p in variants.values()],
                "out": out,
                "imgsz": imgsz,
                "device": device,
                "max_images": max_images,
                "seed": seed,
                "cache": False,
                "gallery": {"enabled": False},
            }
        )
        with console.status("Evaluating variants ..."):
            results = run_comparison(cfg)
    except (DatasetError, AdapterError, ConfigError, ImportError) as exc:
        raise fail(str(exc)) from exc
    rows = quantisation_table(results)
    results["quantisation"] = rows
    results["insights"] = quant_insights(rows) + results["insights"]
    paths = build_report(results, out)
    console.print(_summary_table(results))
    for line in quant_insights(rows):
        console.print(f"  - {line}")
    console.print(f"\n[green]Report written:[/green] {paths['html']}")


@app.command("video-bench")
def video_bench(
    source: Path = typer.Option(..., "--source", help="Input video file."),
    models: list[str] = typer.Option(..., "--models", help="Model specs to run side by side."),
    out: Path = typer.Option(Path("video_report"), "--out", help="Output folder."),
    conf: float = typer.Option(0.25, help="Confidence threshold for drawing/counting."),
    imgsz: int = typer.Option(640),
    device: str = typer.Option("auto", help="cpu | cuda | auto."),
    max_frames: int | None = typer.Option(None, help="Stop after N processed frames."),
    stride: int = typer.Option(1, help="Process every Nth frame."),
    no_video: bool = typer.Option(False, "--no-video", help="Skip writing the side-by-side video."),
) -> None:
    """Run several models on one video: latency, FPS, stability and a side-by-side clip."""
    import json

    from duelscope.adapters import DetectorConfig, create_detector
    from duelscope.bench import hardware_info
    from duelscope.video import video_benchmark

    try:
        cfg = DetectorConfig(imgsz=imgsz, conf=0.001, device=device)
        dets = [create_detector(m, cfg) for m in models]
        out.mkdir(parents=True, exist_ok=True)
        res = video_benchmark(
            dets, source, None if no_video else out / "side_by_side.mp4", conf, max_frames, stride
        )
    except AdapterError as exc:
        raise fail(str(exc)) from exc
    res["hardware"] = hardware_info()
    (out / "video_results.json").write_text(json.dumps(res, indent=2))
    t = Table(title=f"{res['source']}: {res['frames_processed']} frames")
    for col in ("Model", "p50 ms", "p99 ms", "FPS", "Avg detections", "Flicker"):
        t.add_column(col, justify="left" if col == "Model" else "right")
    for m in res["models"]:
        lat = m["latency_ms"]
        t.add_row(
            m["name"],
            f"{lat['p50']:.1f}",
            f"{lat['p99']:.1f}",
            f"{m['fps']:.1f}",
            f"{m['mean_detections']:.1f}",
            f"{m['flicker']:.2f}",
        )
    console.print(t)
    console.print("Flicker = mean change in detection count between frames (lower is steadier).")
    if res["side_by_side"]:
        console.print(f"[green]Side-by-side video:[/green] {res['side_by_side']}")


if __name__ == "__main__":
    cli()
