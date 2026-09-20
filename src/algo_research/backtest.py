"""Vectorised daily backtest with transaction costs, and the standard performance metrics.

Conventions, stated so they can be checked:
  * the signal on day t is traded at the close of day t+1 (one-day lag, no look-ahead);
  * cost per unit of turnover = (commission_bps + slippage_bps) / 10_000, charged on |Δposition|;
  * daily strategy return = position[t-1] * asset_return[t] - cost[t];
  * Sharpe is annualised with sqrt(252) from daily returns and assumes a zero risk-free rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import FloatArray, Series
from .strategies import Signal

TRADING_DAYS = 252


@dataclass(frozen=True)
class Costs:
    commission_bps: float = 5.0
    slippage_bps: float = 5.0

    @property
    def per_unit(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0


@dataclass(frozen=True)
class BacktestResult:
    positions: FloatArray
    returns: FloatArray
    equity: FloatArray
    costs_paid: float
    turnover: float
    n_days: int


def run_backtest(
    series: Series, signal: Signal, params: dict[str, float], costs: Costs = Costs()
) -> BacktestResult:
    """Trade `signal` on `series`; daily strategy returns and the equity curve (start 1.0)."""
    target = signal(series.close, params)
    if target.shape != series.close.shape:
        msg = "signal must return one position per close"
        raise ValueError(msg)
    # Position held during day t is the target from day t-1 (lagged), so day 0 is flat.
    held = np.concatenate(([0.0], target[:-1]))
    asset = np.concatenate(([0.0], series.returns))
    trades = np.abs(np.diff(np.concatenate(([0.0], held))))
    cost = trades * costs.per_unit
    strat = held * asset - cost
    equity = np.cumprod(1.0 + strat)
    return BacktestResult(
        positions=held,
        returns=strat,
        equity=equity,
        costs_paid=float(cost.sum()),
        turnover=float(trades.sum()),
        n_days=len(strat),
    )


@dataclass(frozen=True)
class Metrics:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    trades: float
    n_days: int


def max_drawdown(equity: FloatArray) -> float:
    """Largest peak-to-trough fall as a positive fraction."""
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = 1.0 - equity / peaks
    return float(dd.max())


def sharpe_ratio(returns: FloatArray) -> float:
    if len(returns) < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(returns) / sd * np.sqrt(TRADING_DAYS))


def compute_metrics(result: BacktestResult) -> Metrics:
    r = result.returns
    n = len(r)
    final = float(result.equity[-1]) if n else 1.0
    years = n / TRADING_DAYS
    cagr = float(final ** (1.0 / years) - 1.0) if years > 0 and final > 0 else 0.0
    active = r[result.positions != 0]
    hit = float(np.mean(active > 0)) if len(active) else 0.0
    return Metrics(
        total_return=final - 1.0,
        cagr=cagr,
        volatility=float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0,
        sharpe=sharpe_ratio(r),
        max_drawdown=max_drawdown(result.equity),
        hit_rate=hit,
        trades=result.turnover,
        n_days=n,
    )
