import numpy as np
import pytest

from algo_research.backtest import Costs
from algo_research.data import synthetic_series
from algo_research.strategies import PARAM_GRID, momentum
from algo_research.walkforward import oos_metrics, walk_forward


def test_walk_forward_folds_and_concatenation() -> None:
    s = synthetic_series(n=900, seed=3)
    wf = walk_forward(
        s, momentum, PARAM_GRID["momentum"], train_days=300, test_days=100, costs=Costs()
    )
    assert len(wf.folds) == 6
    for i, f in enumerate(wf.folds):
        assert f.train_end - f.train_start == 300
        assert f.test_end - f.test_start == 100
        assert f.test_start == f.train_end
        assert f.params in PARAM_GRID["momentum"]
        if i:
            assert f.train_start == wf.folds[i - 1].train_start + 100
    assert len(wf.oos_returns) == 600
    assert wf.oos_equity[-1] == pytest.approx(float(np.prod(1 + wf.oos_returns)))
    assert wf.trials == len(PARAM_GRID["momentum"])
    m = oos_metrics(wf)
    assert set(m) == {"cagr", "sharpe", "max_drawdown", "volatility"}
    assert m["sharpe"] == pytest.approx(wf.oos_sharpe)


def test_walk_forward_validation() -> None:
    s = synthetic_series(n=300, seed=3)
    with pytest.raises(ValueError):
        walk_forward(s, momentum, PARAM_GRID["momentum"], train_days=300, test_days=100)
    with pytest.raises(ValueError):
        walk_forward(s, momentum, [], train_days=100, test_days=50)
    with pytest.raises(ValueError):
        walk_forward(s, momentum, PARAM_GRID["momentum"], train_days=10, test_days=50)
