"""Overfitting guards, after Bailey & López de Prado.

Probabilistic Sharpe ratio (PSR): the probability that the true Sharpe exceeds a benchmark,
given the estimated Sharpe, track length and the skew/kurtosis of returns.

Deflated Sharpe ratio (DSR): PSR evaluated against the expected maximum Sharpe one would find
by chance among N independent trials with a given cross-trial variance — the benchmark that
a strategy selected from a grid must beat.

Reference: Bailey, D. H. & López de Prado, M. (2014), "The Deflated Sharpe Ratio",
Journal of Portfolio Management 40(5). Formulas below use per-period (daily) Sharpe.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .data import FloatArray

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def _moments(returns: FloatArray) -> tuple[float, float]:
    """Sample skewness and kurtosis (non-excess)."""
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n < 4:
        return 0.0, 3.0
    m = r.mean()
    sd = r.std(ddof=0)
    if sd == 0:
        return 0.0, 3.0
    z = (r - m) / sd
    return float(np.mean(z**3)), float(np.mean(z**4))


def probabilistic_sharpe(returns: FloatArray, benchmark_sharpe: float = 0.0) -> float:
    """P[true SR > benchmark], with SR per period (not annualised)."""
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = float(r.mean() / sd)
    skew, kurt = _moments(r)
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    return float(_N.cdf((sr - benchmark_sharpe) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, trial_sharpe_variance: float) -> float:
    """Expected maximum per-period Sharpe across N independent trials with zero true Sharpe."""
    if n_trials <= 1 or trial_sharpe_variance <= 0:
        return 0.0
    sd = float(np.sqrt(trial_sharpe_variance))
    e = np.e
    return sd * (
        (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * e))
    )


@dataclass(frozen=True)
class Deflation:
    sharpe_daily: float
    sharpe_annual: float
    n_trials: int
    expected_max_sharpe_daily: float
    psr_zero: float
    dsr: float
    verdict: str


def deflated_sharpe(returns: FloatArray, n_trials: int, trial_sharpe_variance: float) -> Deflation:
    """DSR for the best of `n_trials`, given the variance of Sharpe across the trials."""
    r = np.asarray(returns, dtype=np.float64)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    sr = float(r.mean() / sd) if sd > 0 else 0.0
    sr_star = expected_max_sharpe(n_trials, trial_sharpe_variance)
    psr0 = probabilistic_sharpe(r, 0.0)
    dsr = probabilistic_sharpe(r, sr_star)
    if dsr >= 0.95:
        verdict = "Survives deflation: unlikely to be a selection artefact at the 95% level."
    elif dsr >= 0.5:
        verdict = "Inconclusive: better than chance but not at a level a committee should fund."
    else:
        verdict = (
            "Probably selection bias: the best of the grid is expected to look this good by luck."
        )
    return Deflation(
        sharpe_daily=sr,
        sharpe_annual=sr * float(np.sqrt(252)),
        n_trials=n_trials,
        expected_max_sharpe_daily=sr_star,
        psr_zero=psr0,
        dsr=dsr,
        verdict=verdict,
    )


def min_track_record_length(
    returns: FloatArray, benchmark_sharpe: float = 0.0, confidence: float = 0.95
) -> float:
    """Days of track record for PSR to reach `confidence` at the observed Sharpe (inf if never)."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return float("inf")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("inf")
    sr = float(r.mean() / sd)
    if sr <= benchmark_sharpe:
        return float("inf")
    skew, kurt = _moments(r)
    z = _N.inv_cdf(confidence)
    return float(
        1 + (1 - skew * sr + (kurt - 1) / 4 * sr * sr) * (z / (sr - benchmark_sharpe)) ** 2
    )
