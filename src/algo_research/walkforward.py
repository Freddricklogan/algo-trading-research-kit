"""Walk-forward validation: choose parameters on an in-sample window, trade them on the next
out-of-sample window, roll forward. The concatenated out-of-sample record is the honest one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backtest import Costs, compute_metrics, run_backtest, sharpe_ratio
from .data import FloatArray, Series
from .strategies import Signal


@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    params: dict[str, float]
    in_sample_sharpe: float
    out_of_sample_sharpe: float


@dataclass(frozen=True)
class WalkForwardResult:
    folds: tuple[Fold, ...]
    oos_returns: FloatArray
    oos_equity: FloatArray
    oos_sharpe: float
    in_sample_sharpe_mean: float
    trials: int


def _slice(series: Series, start: int, end: int) -> Series:
    return Series(
        dates=series.dates[start:end], close=series.close[start:end], source=series.source
    )


def walk_forward(
    series: Series,
    signal: Signal,
    grid: list[dict[str, float]],
    train_days: int = 500,
    test_days: int = 125,
    costs: Costs = Costs(),
) -> WalkForwardResult:
    """Grid-search `grid` on each training window by Sharpe, then trade the winner out of sample."""
    n = len(series.close)
    if train_days < 50 or test_days < 20:
        msg = "train_days must be at least 50 and test_days at least 20"
        raise ValueError(msg)
    if n < train_days + test_days:
        msg = "series too short for one fold"
        raise ValueError(msg)
    if not grid:
        msg = "grid must not be empty"
        raise ValueError(msg)
    folds: list[Fold] = []
    oos: list[FloatArray] = []
    start = 0
    while start + train_days + test_days <= n:
        train = _slice(series, start, start + train_days)
        best: tuple[float, dict[str, float]] | None = None
        for params in grid:
            sr = sharpe_ratio(run_backtest(train, signal, params, costs).returns)
            if best is None or sr > best[0]:
                best = (sr, params)
        if best is None:  # pragma: no cover - grid is non-empty by the guard above
            msg = "no parameters evaluated"
            raise RuntimeError(msg)
        # Out-of-sample window gets enough history before it for the signal to warm up.
        warm = start
        test = _slice(series, warm, start + train_days + test_days)
        res = run_backtest(test, signal, best[1], costs)
        oos_part = res.returns[train_days:]
        folds.append(
            Fold(
                train_start=start,
                train_end=start + train_days,
                test_start=start + train_days,
                test_end=start + train_days + test_days,
                params=best[1],
                in_sample_sharpe=best[0],
                out_of_sample_sharpe=sharpe_ratio(oos_part),
            )
        )
        oos.append(oos_part)
        start += test_days
    oos_returns = np.concatenate(oos)
    return WalkForwardResult(
        folds=tuple(folds),
        oos_returns=oos_returns,
        oos_equity=np.cumprod(1.0 + oos_returns),
        oos_sharpe=sharpe_ratio(oos_returns),
        in_sample_sharpe_mean=float(np.mean([f.in_sample_sharpe for f in folds])),
        trials=len(grid),
    )


def oos_metrics(result: WalkForwardResult) -> dict[str, float]:
    """Standard metrics over the concatenated out-of-sample record."""
    from .backtest import BacktestResult

    fake = BacktestResult(
        positions=np.ones_like(result.oos_returns),
        returns=result.oos_returns,
        equity=result.oos_equity,
        costs_paid=0.0,
        turnover=0.0,
        n_days=len(result.oos_returns),
    )
    m = compute_metrics(fake)
    return {
        "cagr": m.cagr,
        "sharpe": m.sharpe,
        "max_drawdown": m.max_drawdown,
        "volatility": m.volatility,
    }
