# Case Study — Algo Trading Research Kit

**Repository:** [algo-trading-research-kit](https://github.com/Freddricklogan/algo-trading-research-kit) · **Live demo:** [freddricklogan.github.io/algo-trading-research-kit](https://freddricklogan.github.io/algo-trading-research-kit/) · **Author:** Freddrick Logan

---

## 1. Who has this problem

Anyone who evaluates a trading idea before money is committed: the instructor whose students bring a backtest with a Sharpe of 2.5, an endowment committee reviewing a manager's pitch, a public treasury team being sold a systematic overlay. I sit on the teaching side — the Oxford Saïd programme on algorithmic trading is where I learned to distrust a backtest — and the failure is the same in a classroom and a boardroom: the strategy looked good on the data it was fitted to.

## 2. The problem, as a scenario

A student presents a mean-reversion strategy: a steadily rising equity curve, an annualised Sharpe of 0.72, "backtested over six years". Did it include trading costs? "Roughly." How were the window and threshold chosen? "I tried a few." What did it do on unseen data? No answer; there was none. The committee version is the same conversation with a fee attached. In both rooms the honest questions — costs, out of sample, number of tries — have no numbers.

## 3. What it costs to leave it alone

Capital committed to a strategy that was fitted, not found. For a student the cost is a habit that follows them into a job; for a committee, a first-year drawdown and a fee paid for the privilege. I will not attach a figure — it depends on the capital and the strategy, and the series here is synthetic. What costs is optimism compounded three times: no costs, in-sample parameters, and the best of many tries presented as the only one.

## 4. The approach, and the alternative I rejected

I built a small, typed Python package that applies the three corrections together and a report that shows them side by side. The backtester lags every signal a day and charges commission and slippage on every unit of turnover. Walk-forward validation chooses parameters on a rolling training window and keeps only out-of-sample days. The deflated Sharpe ratio, after Bailey and López de Prado, benchmarks the best-of-grid Sharpe against what selecting among that many parameter sets would produce by chance, and prints a verdict. CI runs the package on a seeded synthetic series and publishes the report, so every number on the live page was computed by the run that published it.

The alternative I rejected was a richer framework — more asset classes, more order types, a live data feed. Those features make a backtest look serious and none addresses why backtests lie. A kit that does three things correctly, states its conventions, and reproduces by one command teaches more than a platform that does fifty and hides its assumptions.

## 5. What the code does today

Real: the seeded synthetic series and validating CSV loader, three strategies with parameter grids, the lagged and costed backtester with standard metrics, walk-forward validation, probabilistic and deflated Sharpe ratios with skew and kurtosis terms, minimum track-record length, the static report generator, a command-line interface and a Streamlit app that accepts a CSV. All strict-mode typed Python with tests; the report ships with a strict content-security policy and no inline script.

Simulated: the market. The series is a seeded geometric Brownian motion with two volatility regimes and no exploitable structure by construction — which is why the correct result for every strategy is "does not survive deflation". The page says so, and that nothing on it is a forecast or a recommendation.

Worth knowing: deflation uses the cross-trial variance of Sharpe from the same grid, a standard but simplified estimate; costs are a flat basis-point charge with no market-impact model; Sharpe assumes a zero risk-free rate. Each convention is written on the report so a reader can disagree with it.

## 6. Evidence

Measured in continuous integration and locally with the same commands: 24 tests passing across six files; 99% coverage of statements and branches; ruff, ruff format and `mypy --strict` clean across 14 source files; bandit clean; pip-audit with no known vulnerabilities; CodeQL for Python and JavaScript. The tests pin the lag with a look-ahead signal that earns nothing, the cost model on hand-counted turnover, the probabilistic Sharpe ratio against its closed form, the expected-maximum-Sharpe formula, and fold geometry. The shipped report (seed 42, 1,500 days, 22 parameter sets) shows in-sample against out-of-sample Sharpe: moving-average crossover 0.08 to −0.13, momentum 0.37 to 0.46, mean reversion 0.72 to −0.09, with deflated Sharpe ratios of 0.242, 0.440 and 0.488 — none at the 0.95 a committee should require. Headless Chrome on the built report: zero console errors.

## 7. What it would take to run this in production

As a teaching instrument and a review checklist it is production now. As a research platform it would need survivorship-bias-free price histories, a market-impact model instead of flat basis points, portfolio-level backtests with sizing and risk limits, and a results database recording every trial — deflation is only honest if the number of tries is. Months of work, mostly data engineering, with a compliance function beside it; the overfitting guards carry over unchanged.

## 8. Limits and next steps

One instrument, unit positions, daily bars, flat costs, no borrowing or execution model, synthetic data by default. Next: purged cross-validation alongside walk-forward, a combinatorial backtest for the probability of overfitting, and a bootstrap confidence interval on the out-of-sample Sharpe so the single numbers carry their uncertainty.

## 9. Who should look at this

**Hiring manager:** evidence that I implement quantitative-finance methods from the primary literature, test them against closed forms, and ship a reproducible artefact.
**Consulting client:** a checklist, in running code, for reviewing any systematic strategy a manager brings — costs, out-of-sample, number of tries.
**Engineer:** read `src/algo_research/overfitting.py` and `tests/test_overfitting.py` for the deflation and its closed-form checks, and `report.py` for a CSP-safe static site from templates.
