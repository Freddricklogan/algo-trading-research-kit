"""Signal functions. Each returns a target position in {-1, 0, +1} per day, using only past data.

Signals are shifted by one day inside the backtester, so a signal computed
from today's close is traded at tomorrow's close — no look-ahead.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .data import FloatArray

Signal = Callable[[FloatArray, dict[str, float]], FloatArray]


def _rolling_mean(x: FloatArray, window: int) -> FloatArray:
    """Trailing mean; NaN until the window is full."""
    out = np.full(len(x), np.nan)
    if window <= 0 or window > len(x):
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1 :] = (csum[window:] - csum[:-window]) / window
    return out


def _rolling_std(x: FloatArray, window: int) -> FloatArray:
    out = np.full(len(x), np.nan)
    if window <= 1 or window > len(x):
        return out
    for i in range(window - 1, len(x)):
        out[i] = float(np.std(x[i - window + 1 : i + 1], ddof=1))
    return out


def ma_crossover(close: FloatArray, params: dict[str, float]) -> FloatArray:
    """Long when the fast moving average is above the slow one, flat otherwise."""
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 100))
    if fast >= slow:
        msg = "fast must be shorter than slow"
        raise ValueError(msg)
    f = _rolling_mean(close, fast)
    s = _rolling_mean(close, slow)
    pos = np.where(f > s, 1.0, 0.0)
    pos[np.isnan(s)] = 0.0
    return pos


def momentum(close: FloatArray, params: dict[str, float]) -> FloatArray:
    """Long when the trailing return over `lookback` days is positive, short when negative."""
    lookback = int(params.get("lookback", 60))
    if lookback <= 0:
        msg = "lookback must be positive"
        raise ValueError(msg)
    pos = np.zeros(len(close))
    if lookback < len(close):
        ret = close[lookback:] / close[:-lookback] - 1.0
        pos[lookback:] = np.sign(ret)
    return pos


def mean_reversion(close: FloatArray, params: dict[str, float]) -> FloatArray:
    """Fade z-score extremes: short above +z, long below -z, flat inside the band."""
    window = int(params.get("window", 20))
    z = float(params.get("z", 2.0))
    if window <= 1 or z <= 0:
        msg = "window must exceed 1 and z must be positive"
        raise ValueError(msg)
    m = _rolling_mean(close, window)
    sd = _rolling_std(close, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = (close - m) / sd
    pos = np.zeros(len(close))
    pos[score > z] = -1.0
    pos[score < -z] = 1.0
    return pos


STRATEGIES: dict[str, Signal] = {
    "ma_crossover": ma_crossover,
    "momentum": momentum,
    "mean_reversion": mean_reversion,
}

PARAM_GRID: dict[str, list[dict[str, float]]] = {
    "ma_crossover": [{"fast": f, "slow": s} for f in (10, 20, 50) for s in (50, 100, 200) if f < s],
    "momentum": [{"lookback": lb} for lb in (20, 40, 60, 120, 250)],
    "mean_reversion": [{"window": w, "z": z} for w in (10, 20, 40) for z in (1.5, 2.0, 2.5)],
}
