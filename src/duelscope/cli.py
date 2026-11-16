"""Duelscope command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from duelscope import __version__
from duelscope.data import DatasetError, load_dataset, summarize

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


if __name__ == "__main__":
    app()
