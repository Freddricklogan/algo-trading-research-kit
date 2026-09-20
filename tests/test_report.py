from pathlib import Path

from algo_research.backtest import Costs
from algo_research.data import synthetic_series
from algo_research.report import render_html, run_all, write_report


def test_run_all_and_render(tmp_path: Path) -> None:
    s = synthetic_series(n=900, seed=5)
    runs = run_all(s, Costs(), train_days=300, test_days=100)
    assert [r.name for r in runs] == ["ma_crossover", "momentum", "mean_reversion"]
    for r in runs:
        assert 0.0 <= r.deflation.dsr <= 1.0
        assert len(r.walk.folds) == 6
        assert r.no_cost.total_return >= r.in_sample.total_return - 1e-12
    html = render_html(s, runs, Costs(), "https://example.invalid/pages/")
    assert "<!DOCTYPE html>" in html
    assert "Content-Security-Policy" in html
    assert 'id="report-data"' in html
    assert "Deflated Sharpe ratio" in html
    assert "onclick=" not in html and "style=" not in html


def test_write_report_creates_site(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    index = write_report(out, seed=9, n=800)
    assert index.exists()
    for name in (
        "src/exec-shell.css",
        "src/exec-shell.js",
        "src/report.js",
        "src/report.css",
        "report.json",
    ):
        assert (out / name).exists(), name
    text = index.read_text(encoding="utf-8")
    assert "synthetic GBM, seed 9" in text
    assert "800 daily closes" in text
