# Factor-Optimizer

Walk-forward multi-factor portfolio optimizer validated across three independent out-of-sample regimes (2010–2014, 2015–2019, 2020–2024). Long-only Fama-French 5 + Carhart UMD with mean-variance and CVaR objectives, statistical inference via deflated Sharpe and paired block-bootstrap, on a survivorship-bias-free S&P 500 constituent universe.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Headline result

Three non-overlapping five-year OOS windows, independent training data. Sharpe-maximizing objective:

| Period | Strategy Sharpe | SPY Sharpe | Strategy Return | SPY Return | Strategy Max DD |
|---|---|---|---|---|---|
| 2010–2014 | **1.31** | 0.98 | 15.2% | 13.4% | -12.0% |
| 2015–2019 | **0.93** | 0.81 | 11.8% | 11.5% | -15.3% |
| 2020–2024 | 0.27 | 0.64 | 6.0% | 13.7% | -36.9% |
| **Aggregate** | mean 0.84, σ 0.43 | — | — | — | — |

Paired block-bootstrap 95% CIs for Sharpe difference vs SPY contain zero in every period. The 2020–2024 underperformance is consistent with weakening of the low-volatility factor in a regime where index returns concentrated in mega-cap growth (Blitz & van Vliet, 2023). Cross-period Sharpe standard deviation of 0.43 quantifies regime sensitivity explicitly. No parameter tuning was performed between periods.

See `robustness_periods.png` for cumulative return visualization.

---

## What this demonstrates

Rigorous walk-forward portfolio construction with:

- **Survivorship-bias-free universe.** Point-in-time S&P 500 membership at each rebalance. Delistings handled explicitly.
- **No factor lookahead.** Fama-French factors filtered to those published as of each rebalance date.
- **CVaR as a linear program.** Rockafellar-Uryasev formulation solved globally via `scipy.linprog` HiGHS.
- **Deflated Sharpe.** Multiple-testing correction applied (Bailey & López de Prado, 2014).
- **Paired bootstrap vs benchmark.** 95% CIs for Sharpe-difference vs SPY in each window.
- **Multi-period validation.** Three independent OOS windows, with aggregate cross-period statistics.

---

## Methodology

**Universe.** S&P 500 historical constituents, top 50 by training-window liquidity proxy at each quarterly rebalance.

**Expected returns.** Fama-French 5 + Carhart UMD. HAC-robust OLS on rolling 24-month windows. Long-run factor premia from Ken French data. In-sample alpha excluded (does not persist OOS; McLean & Pontiff, 2016).

**Covariance.** Ledoit-Wolf shrinkage on 36-month daily returns.

**Optimization.** Two objectives run independently:
- Mean-variance: SLSQP on negative Sharpe with sum-to-one, 0 ≤ w_i ≤ 0.20.
- CVaR: `scipy.linprog` HiGHS, 5% tail, 8% target return, sum-to-one, box and turnover constraints.

**Rebalancing.** Quarterly. 10 bps transaction cost on drifted-weight turnover.

**Inference.** Deflated Sharpe (n_trials = 12). Block-bootstrap (block size 21, 1000 samples) for standalone and paired-vs-SPY Sharpe CIs.

---

## Repository structure

```
Factor-Optimizer/
├── src/
│   ├── data_fetcher.py        # yfinance with delisting-aware return handling
│   ├── universe.py            # Point-in-time S&P 500 constituent loader
│   ├── factor_model.py        # FF5 + UMD regression, expected returns, attribution
│   ├── factor_optimizer.py    # Sharpe (SLSQP) and CVaR (HiGHS LP) optimizers
│   └── statistics.py          # Deflated Sharpe, paired bootstrap
├── tests/
│   ├── test_invariants.py     # Weight caps, sum-to-one, universe rotation
│   └── test_attribution.py    # Factor decomposition balance check
├── data/
│   └── sp500_history.csv      # Historical constituent membership
├── main.py
└── requirements.txt
```

---

## Quick start

```bash
git clone https://github.com/abdulmoiz23278-ctrl/Factor-Optimizer.git
cd Factor-Optimizer
pip install -r requirements.txt
python main.py
pytest tests/ -v
```

Runtime ~15–20 minutes (six walk-forward backtests, FF + UMD factor downloads cached after first run). Outputs: robustness table to stdout, plot to `robustness_periods.png`, state pickled to `tests/_robustness_state.pkl`.

---

## Limitations

- **Price data.** yfinance is not point-in-time; CRSP would be the production replacement.
- **Factor exposures.** Static within each 24-month rolling window.
- **Universe.** Top 50 by liquidity within S&P 500. Production strategies use 500–3000 names.
- **Execution.** Linear 10 bps cost; production would require square-root market impact with ADV-based sizing.
- **Long-only.** Market-neutral and 130/30 variants are natural extensions.

---

## References

1. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22.
2. Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57–82.
3. Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2, 21–42.
4. Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94–107.
5. Blitz, D., & van Vliet, P. (2023). The volatility effect revisited. *Journal of Portfolio Management*, 50(2).

---

*Python, scipy, statsmodels, pandas, yfinance.*
