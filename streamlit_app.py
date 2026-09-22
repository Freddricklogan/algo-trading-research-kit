"""Interactive companion to the static report. Run: `uv run streamlit run streamlit_app.py`.

Everything shown is computed by the algo_research package on a seeded synthetic series or a
CSV you upload (date, close). Nothing here is a forecast or a recommendation.
"""

from __future__ import annotations

# Community Cloud runs this file from a plain checkout; make the src/ package importable there.
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from algo_research.backtest import Costs, compute_metrics, run_backtest
from algo_research.data import load_csv, synthetic_series
from algo_research.overfitting import deflated_sharpe
from algo_research.strategies import PARAM_GRID, STRATEGIES
from algo_research.walkforward import walk_forward

st.set_page_config(page_title="Algo Trading Research Kit", layout="wide")
st.title("Algo Trading Research Kit")
st.caption(
    "Walk-forward validation, transaction costs, deflated Sharpe. Synthetic data unless you upload."
)

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("CSV with date, close", type=["csv"])
    seed = st.number_input("Seed (synthetic)", value=42, min_value=0, step=1)
    days = st.slider("Days (synthetic)", 600, 4000, 1500, 100)
    st.header("Costs")
    commission = st.slider("Commission (bps)", 0.0, 50.0, 5.0, 0.5)
    slippage = st.slider("Slippage (bps)", 0.0, 50.0, 5.0, 0.5)
    st.header("Walk-forward")
    train = st.slider("Train days", 100, 1000, 500, 50)
    test = st.slider("Test days", 25, 500, 125, 25)

if upload is not None:
    series, warnings = load_csv(
        upload.getvalue().decode("utf-8", errors="replace"), source=upload.name
    )
    for w in warnings:
        st.warning(w)
    if series is None:
        st.stop()
else:
    series = synthetic_series(n=int(days), seed=int(seed))

costs = Costs(float(commission), float(slippage))
st.info(
    f"Data: {series.source} - {len(series.close)} closes; "
    f"costs {costs.commission_bps:.1f} + {costs.slippage_bps:.1f} bps per unit of turnover."
)

tabs = st.tabs(list(STRATEGIES))
for tab, name in zip(tabs, STRATEGIES, strict=True):
    with tab:
        signal = STRATEGIES[name]
        grid = PARAM_GRID[name]
        sharpes = []
        best, best_sr = grid[0], -np.inf
        for params in grid:
            m = compute_metrics(run_backtest(series, signal, params, costs))
            sharpes.append(m.sharpe / np.sqrt(252))
            if m.sharpe > best_sr:
                best, best_sr = params, m.sharpe
        is_res = run_backtest(series, signal, best, costs)
        is_m = compute_metrics(is_res)
        try:
            wf = walk_forward(series, signal, grid, int(train), int(test), costs)
        except ValueError as err:
            st.error(str(err))
            continue
        d = deflated_sharpe(
            is_res.returns, len(grid), float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("In-sample Sharpe (best of grid)", f"{is_m.sharpe:.2f}")
        c2.metric("Out-of-sample Sharpe", f"{wf.oos_sharpe:.2f}")
        c3.metric("Deflated Sharpe ratio", f"{d.dsr:.3f}")
        c4.metric("Max drawdown (IS)", f"{is_m.max_drawdown * 100:.1f}%")
        st.write(f"Best in-sample parameters: `{best}` · {d.verdict}")
        st.line_chart({"in-sample equity": is_res.equity})
        st.line_chart({"out-of-sample equity": wf.oos_equity})
        st.dataframe(
            [
                {
                    "fold": i + 1,
                    "train": f"{f.train_start}-{f.train_end}",
                    "test": f"{f.test_start}-{f.test_end}",
                    "params": str(f.params),
                    "IS Sharpe": round(f.in_sample_sharpe, 2),
                    "OOS Sharpe": round(f.out_of_sample_sharpe, 2),
                }
                for i, f in enumerate(wf.folds)
            ],
            use_container_width=True,
        )
