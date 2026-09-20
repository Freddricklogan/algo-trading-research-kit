import numpy as np
import pytest

from algo_research.overfitting import (
    EULER_GAMMA,
    deflated_sharpe,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe,
)


def test_psr_basic_properties() -> None:
    rng = np.random.default_rng(0)
    good = rng.normal(0.002, 0.01, 1000)
    bad = rng.normal(-0.002, 0.01, 1000)
    assert probabilistic_sharpe(good) > 0.99
    assert probabilistic_sharpe(bad) < 0.01
    assert probabilistic_sharpe(np.array([0.01])) == 0.0
    assert probabilistic_sharpe(np.zeros(50)) == 0.0
    # Higher benchmark -> lower probability
    assert probabilistic_sharpe(good, 0.1) < probabilistic_sharpe(good, 0.0)


def test_psr_matches_closed_form_for_normal_returns() -> None:
    # For symmetric normal returns skew~0, kurt~3 -> PSR ~ Phi(SR·sqrt(n-1)/sqrt(1 + SR²/2))
    from statistics import NormalDist

    rng = np.random.default_rng(7)
    r = rng.normal(0.001, 0.01, 5000)
    sr = r.mean() / r.std(ddof=1)
    skew = float(np.mean(((r - r.mean()) / r.std()) ** 3))
    kurt = float(np.mean(((r - r.mean()) / r.std()) ** 4))
    expected = NormalDist().cdf(
        sr * np.sqrt(len(r) - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr * sr)
    )
    assert probabilistic_sharpe(r) == pytest.approx(expected, abs=1e-9)


def test_expected_max_sharpe_grows_with_trials() -> None:
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(10, 0.0) == 0.0
    e10 = expected_max_sharpe(10, 0.01)
    e100 = expected_max_sharpe(100, 0.01)
    assert 0 < e10 < e100
    # Formula check at N=10, var=1
    from statistics import NormalDist

    nd = NormalDist()
    expect = (1 - EULER_GAMMA) * nd.inv_cdf(1 - 1 / 10) + EULER_GAMMA * nd.inv_cdf(
        1 - 1 / (10 * np.e)
    )
    assert expected_max_sharpe(10, 1.0) == pytest.approx(expect)


def test_deflated_sharpe_verdicts() -> None:
    rng = np.random.default_rng(1)
    strong = rng.normal(0.003, 0.01, 2000)
    d = deflated_sharpe(strong, n_trials=20, trial_sharpe_variance=0.0001)
    assert d.dsr > 0.95
    assert d.verdict.startswith("Survives")
    assert d.sharpe_annual == pytest.approx(d.sharpe_daily * np.sqrt(252))
    weak = rng.normal(0.0002, 0.01, 300)
    d2 = deflated_sharpe(weak, n_trials=200, trial_sharpe_variance=0.01)
    assert d2.dsr < d.dsr
    assert d2.verdict.startswith(("Probably", "Inconclusive"))
    flat = deflated_sharpe(np.zeros(10), 5, 0.01)
    assert flat.sharpe_daily == 0.0


def test_min_track_record_length() -> None:
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 800)
    n = min_track_record_length(r)
    assert np.isfinite(n)
    assert n > 1
    assert min_track_record_length(np.array([0.01])) == float("inf")
    assert min_track_record_length(np.zeros(20)) == float("inf")
    assert min_track_record_length(-r) == float("inf")
