import numpy as np
import pytest

from algo_research.strategies import (
    PARAM_GRID,
    STRATEGIES,
    _rolling_mean,
    ma_crossover,
    mean_reversion,
    momentum,
)


def test_rolling_mean_matches_numpy() -> None:
    x = np.arange(10, dtype=np.float64)
    m = _rolling_mean(x, 3)
    assert np.isnan(m[:2]).all()
    assert m[2] == pytest.approx(1.0)
    assert m[-1] == pytest.approx(8.0)
    assert np.isnan(_rolling_mean(x, 0)).all()
    assert np.isnan(_rolling_mean(x, 11)).all()


def test_ma_crossover_positions() -> None:
    up = np.linspace(100, 200, 300)
    pos = ma_crossover(up, {"fast": 5, "slow": 20})
    assert set(np.unique(pos)) <= {0.0, 1.0}
    assert pos[:19].sum() == 0  # flat until the slow window fills
    assert pos[-1] == 1.0  # trending up -> long
    with pytest.raises(ValueError):
        ma_crossover(up, {"fast": 50, "slow": 20})


def test_momentum_sign_and_warmup() -> None:
    down = np.linspace(200, 100, 100)
    pos = momentum(down, {"lookback": 10})
    assert pos[:10].sum() == 0
    assert pos[-1] == -1.0
    with pytest.raises(ValueError):
        momentum(down, {"lookback": 0})


def test_mean_reversion_fades_extremes() -> None:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.standard_normal(400))
    close[300] += 40  # spike
    pos = mean_reversion(close, {"window": 20, "z": 2.0})
    assert pos[300] == -1.0
    assert set(np.unique(pos)) <= {-1.0, 0.0, 1.0}
    with pytest.raises(ValueError):
        mean_reversion(close, {"window": 1, "z": 2.0})


def test_registry_and_grid_consistent() -> None:
    assert set(STRATEGIES) == set(PARAM_GRID)
    for name, grid in PARAM_GRID.items():
        assert grid, name
        for params in grid:
            STRATEGIES[name](np.linspace(100, 120, 400), params)
