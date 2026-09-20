"""Command-line entry point: `algo-research report --out dist`."""

from __future__ import annotations

from pathlib import Path

import typer

from .report import write_report

app = typer.Typer(add_completion=False, help="Backtesting research kit.")


@app.callback()
def main() -> None:
    """Backtesting research kit: walk-forward validation and overfitting guards."""


@app.command()
def report(
    out: Path = typer.Option(Path("dist"), help="Output directory for the static report."),
    seed: int = typer.Option(42, help="Seed for the synthetic series."),
    days: int = typer.Option(1500, help="Number of daily closes to simulate."),
) -> None:
    """Generate the walk-forward report as a static site."""
    path = write_report(out, seed=seed, n=days)
    print(f"wrote {path}")


if __name__ == "__main__":
    app()
