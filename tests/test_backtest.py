import numpy as np
import pytest

from algo_research.backtest import Costs, compute_metrics, max_drawdown, run_backtest, sharpe_ratio
from algo_research.data import Series


def const_signal(value: float):  # type: ignore[no-untyped-def]
    def sig(close, params):  # type: ignore[no-untyped-def]
        return np.full(len(close), value)

    return sig


def series_of(closes: list[float]) -> Series:
    return Series(
        dates=tuple(str(i) for i in range(len(closes))),
        close=np.array(closes, dtype=np.float64),
        source="test",
    )


def test_always_long_equals_buy_and_hold_minus_one_entry_cost() -> None:
    s = series_of([100, 110, 121, 133.1])
    r = run_backtest(s, const_signal(1.0), {}, Costs(10.0, 0.0))
    # Day 0 flat; position 1 from day 1; one unit of turnover on day 1 at 10 bps.
    assert r.positions.tolist() == [0.0, 1.0, 1.0, 1.0]
    assert r.turnover == pytest.approx(1.0)
    assert r.costs_paid == pytest.approx(0.001)
    assert r.equity[-1] == pytest.approx((1 + 0.10 - 0.001) * 1.1 * 1.1)


def test_signal_is_lagged_no_lookahead() -> None:
    s = series_of([100, 200, 100, 200])

    def perfect(close, params):  # type: ignore[no-untyped-def]
        # "Knows" tomorrow: +1 before an up day, -1 before a down day (look-ahead if not lagged)
        return np.array([1.0, -1.0, 1.0, 0.0])

    r = run_backtest(s, perfect, {}, Costs(0.0, 0.0))
    # held = [0, 1, -1, 1]; asset = [0, +1.0, -0.5, +1.0] -> returns 0, +1.0, +0.5, +1.0
    assert r.returns.tolist() == pytest.approx([0.0, 1.0, 0.5, 1.0])


def test_costs_charged_on_position_changes_only() -> None:
    s = series_of([100.0] * 6)

    def flip(close, params):  # type: ignore[no-untyped-def]
        return np.array([1.0, 1.0, -1.0, -1.0, 0.0, 0.0])

    r = run_backtest(s, flip, {}, Costs(5.0, 5.0))
    # held = [0,1,1,-1,-1,0]; trades = 1, 0, 2, 0, 1 -> turnover 4 units at 10 bps
    assert r.turnover == pytest.approx(4.0)
    assert r.costs_paid == pytest.approx(0.004)


def test_shape_mismatch_rejected() -> None:
    s = series_of([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        run_backtest(s, lambda _c, _p: np.zeros(2), {})


def test_metrics_hand_checked() -> None:
    eq = np.array([1.0, 1.2, 0.9, 1.5])
    assert max_drawdown(eq) == pytest.approx(0.25)
    assert max_drawdown(np.array([])) == 0.0
    assert sharpe_ratio(np.array([0.01])) == 0.0
    assert sharpe_ratio(np.array([0.01, 0.01, 0.01])) == 0.0
    r = np.array([0.01, -0.005, 0.02, 0.0])
    assert sharpe_ratio(r) == pytest.approx(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252))
    s = series_of([100, 101, 100, 102, 101])
    res = run_backtest(s, const_signal(1.0), {}, Costs(0.0, 0.0))
    m = compute_metrics(res)
    assert m.total_return == pytest.approx(0.01)
    assert m.n_days == 5
    assert 0.0 <= m.hit_rate <= 1.0
    assert m.max_drawdown >= 0.0
