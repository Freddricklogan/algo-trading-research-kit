"""Static HTML report — the Pages artefact. No JavaScript is required to read it; a small
external script (report.js) drives the Executive Shell and the overlay toggles. Every number in
the page is computed by this module from the run it describes, and embedded as JSON for the
shell's KPI strip.
"""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template

import numpy as np

from .backtest import Costs, Metrics, compute_metrics, run_backtest
from .data import FloatArray, Series, synthetic_series
from .overfitting import Deflation, deflated_sharpe, min_track_record_length
from .strategies import PARAM_GRID, STRATEGIES
from .walkforward import WalkForwardResult, oos_metrics, walk_forward

SHELL_DIR = Path(__file__).parent / "shell"
REPORT_JS = Path(__file__).parent / "report.js"
TEMPLATES = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class StrategyRun:
    name: str
    params: dict[str, float]
    in_sample: Metrics
    no_cost: Metrics
    walk: WalkForwardResult
    oos: dict[str, float]
    deflation: Deflation
    min_track: float
    equity_is: FloatArray
    equity_no_cost: FloatArray


def run_all(
    series: Series, costs: Costs = Costs(), train_days: int = 500, test_days: int = 125
) -> list[StrategyRun]:
    """Full in-sample grid, walk-forward, and deflation for every strategy."""
    runs: list[StrategyRun] = []
    for name, signal in STRATEGIES.items():
        grid = PARAM_GRID[name]
        # In-sample: best of the grid on the whole series — the number a naive report would show.
        best_params, best_sr, sharpes = grid[0], -np.inf, []
        for params in grid:
            sr = compute_metrics(run_backtest(series, signal, params, costs)).sharpe
            sharpes.append(sr / np.sqrt(252))
            if sr > best_sr:
                best_params, best_sr = params, sr
        is_res = run_backtest(series, signal, best_params, costs)
        no_cost = run_backtest(series, signal, best_params, Costs(0.0, 0.0))
        walk = walk_forward(series, signal, grid, train_days, test_days, costs)
        defl = deflated_sharpe(
            is_res.returns, len(grid), float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
        )
        runs.append(
            StrategyRun(
                name=name,
                params=best_params,
                in_sample=compute_metrics(is_res),
                no_cost=compute_metrics(no_cost),
                walk=walk,
                oos=oos_metrics(walk),
                deflation=defl,
                min_track=min_track_record_length(is_res.returns),
                equity_is=is_res.equity,
                equity_no_cost=no_cost.equity,
            )
        )
    return runs


def _svg_lines(
    series_list: list[tuple[str, FloatArray, str]], width: int = 720, height: int = 240
) -> str:
    """Line chart of one or more equity curves on a shared log scale."""
    pad = 36
    all_vals = np.concatenate([s[1] for s in series_list])
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    lo, hi = max(lo, 1e-6), max(hi, lo * 1.0001)
    n = max(len(s[1]) for s in series_list)
    span = float(np.log(hi) - np.log(lo))

    def sx(i: int, length: int) -> float:
        return pad + (i / max(1, length - 1)) * (width - 2 * pad)

    def sy(v: float) -> float:
        frac = float((np.log(max(v, 1e-6)) - np.log(lo)) / span)
        return height - pad - frac * (height - 2 * pad)

    paths = []
    for label, vals, cls in series_list:
        pts = " ".join(f"{sx(i, len(vals)):.1f},{sy(float(v)):.1f}" for i, v in enumerate(vals))
        safe = html.escape(label)
        paths.append(
            f'<polyline class="{cls}" points="{pts}" data-series="{safe}">'
            f"<title>{safe}</title></polyline>"
        )
    axis = (
        f'<path class="atr-axis" d="M{pad},{pad} L{pad},{height - pad} '
        f'L{width - pad},{height - pad}"/>'
    )
    labels = (
        f'<text class="atr-tick" x="{pad - 4}" y="{pad + 4}" text-anchor="end">'
        f"{hi:.2f}&times;</text>"
        f'<text class="atr-tick" x="{pad - 4}" y="{height - pad + 4}" text-anchor="end">'
        f"{lo:.2f}&times;</text>"
        f'<text class="atr-tick" x="{width - pad}" y="{height - pad + 14}" text-anchor="end">'
        f"day {n}</text>"
    )
    return (
        f'<svg class="atr-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Equity curves, log scale">{axis}{"".join(paths)}{labels}</svg>'
    )


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _row(cells: list[str], head: bool = False) -> str:
    tag = "th" if head else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


def _kpi_json(series: Series, runs: list[StrategyRun]) -> str:
    return json.dumps(
        {
            "runs": [
                {
                    "name": r.name,
                    "inSampleSharpe": round(r.in_sample.sharpe, 2),
                    "oosSharpe": round(r.oos["sharpe"], 2),
                    "dsr": round(r.deflation.dsr, 3),
                    "costDrag": round(r.no_cost.total_return - r.in_sample.total_return, 4),
                    "folds": len(r.walk.folds),
                }
                for r in runs
            ],
            "days": len(series.close),
            "source": series.source,
            "trials": sum(len(PARAM_GRID[r.name]) for r in runs),
        }
    )


def _section(r: StrategyRun) -> str:
    tmpl = Template((TEMPLATES / "section.html").read_text(encoding="utf-8"))
    d = r.deflation
    fold_rows = "".join(
        _row(
            [
                str(i + 1),
                f"{f.train_start}&ndash;{f.train_end}",
                f"{f.test_start}&ndash;{f.test_end}",
                html.escape(json.dumps(f.params)),
                f"{f.in_sample_sharpe:.2f}",
                f"{f.out_of_sample_sharpe:.2f}",
            ]
        )
        for i, f in enumerate(r.walk.folds)
    )
    rows_is = "".join(
        [
            _row(["Sharpe (annualised)", f"{r.in_sample.sharpe:.2f}"]),
            _row(["CAGR", _pct(r.in_sample.cagr)]),
            _row(["Max drawdown", _pct(r.in_sample.max_drawdown)]),
            _row(["Turnover (units)", f"{r.in_sample.trades:.0f}"]),
            _row(["Return without costs", _pct(r.no_cost.total_return)]),
            _row(["Return with costs", _pct(r.in_sample.total_return)]),
        ]
    )
    rows_oos = "".join(
        [
            _row(["Out-of-sample Sharpe", f"{r.oos['sharpe']:.2f}"]),
            _row(["Mean in-sample Sharpe across folds", f"{r.walk.in_sample_sharpe_mean:.2f}"]),
            _row(["Out-of-sample CAGR", _pct(r.oos["cagr"])]),
            _row(["Out-of-sample max drawdown", _pct(r.oos["max_drawdown"])]),
        ]
    )
    tone = "ok" if d.dsr >= 0.95 else "warn" if d.dsr >= 0.5 else "danger"
    min_track = "&infin;" if not np.isfinite(r.min_track) else f"{r.min_track:.0f} days"
    return tmpl.substitute(
        name=r.name,
        title=html.escape(r.name.replace("_", " ").title()),
        params=html.escape(json.dumps(r.params)),
        grid_size=len(PARAM_GRID[r.name]),
        chart_is=_svg_lines(
            [
                ("with costs", r.equity_is, "atr-line atr-line--is"),
                ("no costs", r.equity_no_cost, "atr-line atr-line--nocost"),
            ]
        ),
        rows_is=rows_is,
        n_folds=len(r.walk.folds),
        chart_oos=_svg_lines([("out-of-sample", r.walk.oos_equity, "atr-line atr-line--oos")]),
        rows_oos=rows_oos,
        verdict_tone=tone,
        dsr=f"{d.dsr:.3f}",
        n_trials=d.n_trials,
        sr_star=f"{d.expected_max_sharpe_daily * np.sqrt(252):.2f}",
        psr=f"{d.psr_zero:.3f}",
        min_track=min_track,
        verdict=html.escape(d.verdict),
        fold_head=_row(
            ["Fold", "Train days", "Test days", "Chosen params", "IS Sharpe", "OOS Sharpe"],
            head=True,
        ),
        fold_rows=fold_rows,
    )


def render_html(series: Series, runs: list[StrategyRun], costs: Costs, pages: str) -> str:
    """The full report page."""
    tmpl = Template((TEMPLATES / "page.html").read_text(encoding="utf-8"))
    first = runs[0].walk.folds[0]
    return tmpl.substitute(
        pages=html.escape(pages),
        kpi_json=_kpi_json(series, runs),
        source=html.escape(series.source),
        days=len(series.close),
        commission=f"{costs.commission_bps:.0f}",
        slippage=f"{costs.slippage_bps:.0f}",
        sections="".join(_section(r) for r in runs),
        train_days=first.train_end - first.train_start,
        test_days=first.test_end - first.test_start,
    )


def write_report(
    out: Path,
    seed: int = 42,
    n: int = 1500,
    pages: str = "https://freddricklogan.github.io/algo-trading-research-kit/",
) -> Path:
    """Generate the report into `out/` (index.html + src/ assets)."""
    series = synthetic_series(n=n, seed=seed)
    costs = Costs()
    runs = run_all(series, costs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "src").mkdir(exist_ok=True)
    shutil.copy(SHELL_DIR / "exec-shell.css", out / "src" / "exec-shell.css")
    shutil.copy(SHELL_DIR / "exec-shell.js", out / "src" / "exec-shell.js")
    shutil.copy(REPORT_JS, out / "src" / "report.js")
    shutil.copy(TEMPLATES / "report.css", out / "src" / "report.css")
    (out / "index.html").write_text(render_html(series, runs, costs, pages), encoding="utf-8")
    (out / "report.json").write_text(
        json.dumps(
            [
                {**asdict(r.in_sample), "name": r.name, "oos": r.oos, "dsr": r.deflation.dsr}
                for r in runs
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return out / "index.html"
