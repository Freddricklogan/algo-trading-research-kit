import numpy as np
import pytest

from algo_research.data import Series, load_csv, synthetic_series


def test_synthetic_is_seeded_and_positive() -> None:
    a = synthetic_series(n=300, seed=1)
    b = synthetic_series(n=300, seed=1)
    assert np.array_equal(a.close, b.close)
    assert not np.array_equal(a.close, synthetic_series(n=300, seed=2).close)
    assert bool(np.all(a.close > 0))
    assert len(a.dates) == 300
    assert a.returns.shape == (299,)


def test_synthetic_rejects_short() -> None:
    with pytest.raises(ValueError):
        synthetic_series(n=1)


def test_series_validation() -> None:
    with pytest.raises(ValueError):
        Series(dates=("a",), close=np.array([1.0, 2.0]), source="x")
    with pytest.raises(ValueError):
        Series(dates=("a",), close=np.array([1.0]), source="x")
    with pytest.raises(ValueError):
        Series(dates=("a", "b"), close=np.array([1.0, -1.0]), source="x")


def test_load_csv_happy_and_warnings() -> None:
    s, w = load_csv(
        "﻿Date,Close\n2024-01-01,100\n2024-01-02,101.5\n2024-01-03,abc\n,5\n2024-01-04,0\n"
    )
    assert s is not None
    assert w == [
        "Row 4: close 'abc' is not a number; skipped.",
        "Row 5: missing date or non-positive close; skipped.",
        "Row 6: missing date or non-positive close; skipped.",
    ]
    assert s.close.tolist() == [100.0, 101.5]


def test_load_csv_failures() -> None:
    assert load_csv("")[0] is None
    assert load_csv("date,price\n2024-01-01,1\n")[1] == ["Missing required column(s): date, close."]
    s, w = load_csv("date,close\n2024-01-01,1\n")
    assert s is None
    assert w[-1] == "Fewer than two valid rows."
