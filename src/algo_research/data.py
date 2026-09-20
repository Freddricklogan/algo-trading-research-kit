"""Price series: a seeded synthetic generator and a validating CSV loader.

The synthetic series is a geometric Brownian motion with two volatility
regimes, so strategies see calm and turbulent stretches. It is labelled
synthetic everywhere it is shown.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Series:
    """Daily closes and their dates (ISO strings), equal length."""

    dates: tuple[str, ...]
    close: FloatArray
    source: str

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.close):
            msg = "dates and close must have the same length"
            raise ValueError(msg)
        if len(self.close) < 2:
            msg = "a series needs at least two observations"
            raise ValueError(msg)
        if not np.all(np.isfinite(self.close)) or bool(np.any(self.close <= 0)):
            msg = "closes must be finite and positive"
            raise ValueError(msg)

    @property
    def returns(self) -> FloatArray:
        """Simple daily returns, length n-1."""
        return np.diff(self.close) / self.close[:-1]


def synthetic_series(
    n: int = 1500,
    seed: int = 42,
    drift: float = 0.06,
    calm_vol: float = 0.12,
    stormy_vol: float = 0.35,
    regime_switch: float = 0.01,
    start: float = 100.0,
) -> Series:
    """GBM with a two-state volatility regime; annualised drift and vols, 252 days a year."""
    if n < 2:
        msg = "n must be at least 2"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    stormy = False
    closes = np.empty(n, dtype=np.float64)
    closes[0] = start
    for i in range(1, n):
        if rng.random() < regime_switch:
            stormy = not stormy
        vol = stormy_vol if stormy else calm_vol
        shock = rng.standard_normal()
        closes[i] = closes[i - 1] * float(
            np.exp((drift - 0.5 * vol * vol) * dt + vol * np.sqrt(dt) * shock)
        )
    base = np.datetime64("2020-01-01")
    dates = tuple(str(base + np.timedelta64(i, "D")) for i in range(n))
    return Series(dates=dates, close=closes, source=f"synthetic GBM, seed {seed}")


def load_csv(text: str, source: str = "csv") -> tuple[Series | None, list[str]]:
    """Parse a CSV with `date` and `close` columns (case-insensitive); bad rows are reported."""
    warnings: list[str] = []
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if reader.fieldnames is None:
        return None, ["File is empty."]
    fields = {f.strip().lower(): f for f in reader.fieldnames}
    if "date" not in fields or "close" not in fields:
        return None, ["Missing required column(s): date, close."]
    dates: list[str] = []
    closes: list[float] = []
    for i, row in enumerate(reader, start=2):
        raw_date = (row.get(fields["date"]) or "").strip()
        raw_close = (row.get(fields["close"]) or "").strip()
        try:
            close = float(raw_close)
        except ValueError:
            warnings.append(f"Row {i}: close '{raw_close}' is not a number; skipped.")
            continue
        if not raw_date or close <= 0 or not np.isfinite(close):
            warnings.append(f"Row {i}: missing date or non-positive close; skipped.")
            continue
        dates.append(raw_date)
        closes.append(close)
    if len(closes) < 2:
        return None, [*warnings, "Fewer than two valid rows."]
    return Series(
        dates=tuple(dates), close=np.asarray(closes, dtype=np.float64), source=source
    ), warnings
