"""Duelscope command-line interface."""

from __future__ import annotations

import typer

from duelscope import __version__

app = typer.Typer(
    name="duelscope",
    help="Compare object detectors: mAP, per-class errors, failure gallery and speed.",
    no_args_is_help=True,
    add_completion=False,
)


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


if __name__ == "__main__":
    app()
