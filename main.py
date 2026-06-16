import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.ticker as mticker  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import yfinance as yf  # type: ignore[import-untyped]

from src.factor_optimizer import optimize_portfolio
from src.data_fetcher import fetch_stock_returns
from src.factor_model import (
    estimate_expected_returns,
    estimate_factor_exposures,
    attribute_oos_returns,
    get_rf_daily_series,
)
from src.universe import load_sp500_history, get_universe_union, get_universe_as_of
from src.statistics import deflated_sharpe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Backtest
# ─────────────────────────────────────────────────────────────────────────────

def _annualized_covariance(train_daily: pd.DataFrame, use_ledoit_wolf: bool) -> np.ndarray:
    """Annualized covariance matrix; Ledoit-Wolf shrinkage when enabled."""
    if use_ledoit_wolf:
        try:
            from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]
            return LedoitWolf().fit(train_daily.values).covariance_ * 252
        except ImportError:
            logger.warning("sklearn unavailable; falling back to sample covariance.")
    return train_daily.cov().values * 252


def _apply_min_weight_threshold(weights, min_weight, max_weight):
    """Zero sub-threshold weights; renormalize without violating max_weight."""
    if min_weight <= 0:
        return weights
    w = np.asarray(weights, dtype=float).copy()
    w[w < min_weight] = 0.0

    for _ in range(20):
        total = w.sum()
        if total < 1e-12:
            return np.ones(len(w)) / len(w)
        w = w / total
        excess_mask = w > max_weight
        if not excess_mask.any():
            break
        excess = (w[excess_mask] - max_weight).sum()
        w[excess_mask] = max_weight
        active_mask = (w > 0) & ~excess_mask
        if active_mask.any():
            w[active_mask] += excess * (w[active_mask] / w[active_mask].sum())

    assert np.all(w <= max_weight + 1e-6), f"max_weight violated: {w.max():.4f}"
    return w


def _tickers_in_panel(requested: list[str], panel_columns) -> list[str]:
    """Map SP500 symbols to downloaded column names (e.g. BRK.B → BRK-B)."""
    colset = set(panel_columns)
    out: list[str] = []
    for t in requested:
        if t in colset:
            out.append(t)
        else:
            alt = t.replace(".", "-")
            if alt in colset:
                out.append(alt)
    return out


def _select_period_universe(
    train_daily: pd.DataFrame,
    universe_tickers: list[str],
    max_names: int = 50,
    min_coverage: float = 0.95,
) -> list[str]:
    """Filter by data coverage; keep top names by highest training-window coverage."""
    available = _tickers_in_panel(universe_tickers, train_daily.columns)
    sub = train_daily[available]
    coverage = sub.notna().mean()
    valid = coverage[coverage >= min_coverage].index.tolist()
    if len(valid) <= max_names:
        return valid
    return coverage.loc[valid].nlargest(max_names).index.tolist()


def _align_weights_to_tickers(
    weights_by_ticker: dict[str, float],
    tickers: list[str],
) -> np.ndarray:
    return np.array([weights_by_ticker.get(t, 0.0) for t in tickers], dtype=float)


def _compute_drifted_weights(
    prev_weights: np.ndarray,
    prev_tickers: list[str],
    prev_oos_daily: pd.DataFrame,
) -> np.ndarray:
    """Drift prior target weights over the completed OOS window."""
    cum = (1 + prev_oos_daily.fillna(0)).prod()
    aligned_cum = cum.reindex(prev_tickers).fillna(1.0).values
    drifted = prev_weights * aligned_cum
    if drifted.sum() > 1e-12:
        return drifted / drifted.sum()
    return prev_weights.copy()


def _regime_adjusted_max_weight(
    train_daily: pd.DataFrame,
    base_max_weight: float,
    use_regime: bool,
    high_vol_scale: float = 0.70,
) -> float:
    """2-state HMM on equal-weight market returns; scale max_weight in high-vol state."""
    if not use_regime:
        return base_max_weight
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("hmmlearn unavailable; regime scaling disabled.")
        return base_max_weight

    try:
        market = train_daily.mean(axis=1).values.reshape(-1, 1)
        if len(market) < 60:
            return base_max_weight
        model = GaussianHMM(n_components=2, covariance_type='full', n_iter=200, random_state=42)
        model.fit(market)
        states = model.predict(market)
        vol_by_state = {
            s: train_daily.mean(axis=1).iloc[states == s].std()
            for s in range(2)
        }
        high_vol_state = max(vol_by_state, key=vol_by_state.get)
        if states[-1] == high_vol_state:
            adjusted = base_max_weight * high_vol_scale
            logger.info(
                f"Regime: high-vol state → max_weight {base_max_weight:.0%} → {adjusted:.0%}"
            )
            return adjusted
    except Exception as e:
        logger.warning(f"HMM regime detection failed ({e}); using base max_weight.")
    return base_max_weight


def walk_forward_backtest(
    daily_returns: pd.DataFrame,
    universe_history: pd.DataFrame,
    train_months: int = 36,
    rebal_months: int = 3,
    risk_free_rate: float = 0.02,
    max_weight: float = 0.20,
    transaction_cost_bps: float = 10.0,
    shrinkage: float = 0.50,
    use_ledoit_wolf: bool = True,
    use_momentum: bool = False,
    optimization_objective: str = 'sharpe',
    max_turnover: float | None = None,
    use_regime: bool = False,
    regime_high_vol_scale: float = 0.70,
    store_factor_exposures: bool = False,
    min_weight_threshold: float = 0.0,
    use_alpha: bool = False,
    factor_mean_window: str = 'long_run',
    beta_window: int | None = None,
    max_universe_size: int = 50,
    cvar_target_return: float | None = None,
) -> tuple:
    """
    Walk-forward portfolio optimization — eliminates lookahead bias entirely.

    Universe: point-in-time S&P 500 constituents; top `max_universe_size` names
    by training-window volatility after coverage filter. OOS delistings: NaN
    returns are treated as cash (fillna(0) before portfolio return).
    """
    # Month-end dates — used as the windowing anchor
    monthly_ends = daily_returns.resample('ME').last().index
    n_months = len(monthly_ends)

    # ~21 trading days per month: used to slice the training window from daily data
    approx_train_days = train_months * 21

    all_oos_returns: list[pd.Series] = []
    weight_history: list[dict] = []
    prev_weights: np.ndarray | None = None
    prev_tickers: list[str] | None = None
    prev_oos_daily: pd.DataFrame | None = None
    prev_universe_set: set[str] | None = None

    rebal_points = list(range(train_months, n_months, rebal_months))
    logger.info(
        f"Walk-forward: {len(rebal_points)} rebalancing periods | "
        f"train={train_months}m, rebal_freq={rebal_months}m"
    )

    for i in rebal_points:
        # ── Window boundaries ─────────────────────────────────────────────
        train_end = monthly_ends[i - 1]                                  # last in-sample day
        oos_end   = monthly_ends[min(i + rebal_months - 1, n_months - 1)]  # last OOS day

        # ── Slice training data (rolling window, daily) ───────────────────
        train_daily = daily_returns[daily_returns.index <= train_end]
        if len(train_daily) > approx_train_days:
            train_daily = train_daily.iloc[-approx_train_days:]  # keep last N trading days

        # ── Slice out-of-sample data ──────────────────────────────────────
        oos_daily = daily_returns[
            (daily_returns.index > train_end) &
            (daily_returns.index <= oos_end)
        ]

        if len(train_daily) < 60 or oos_daily.empty:
            logger.warning(f"Skipping {train_end.date()}: insufficient data.")
            continue

        universe_t = _tickers_in_panel(
            get_universe_as_of(train_end, universe_history),
            daily_returns.columns,
        )
        universe_set = set(universe_t)
        if prev_universe_set is not None:
            entered = sorted(universe_set - prev_universe_set)
            exited = sorted(prev_universe_set - universe_set)
            logger.info(
                f"  Universe @ {train_end.date()}: {len(universe_t)} names | "
                f"+{len(entered)} / -{len(exited)} vs prior rebalance"
            )
        else:
            logger.info(f"  Universe @ {train_end.date()}: {len(universe_t)} names")

        period_tickers = _select_period_universe(
            train_daily, universe_t, max_names=max_universe_size,
        )
        if len(period_tickers) < 5:
            logger.warning(f"Skipping {train_end.date()}: <5 eligible names.")
            continue

        train_sub = train_daily[period_tickers].dropna(how='all')
        oos_sub = oos_daily[[c for c in period_tickers if c in oos_daily.columns]]

        # ── Step 1: Estimate expected returns (FF model, in-sample only) ──
        try:
            exp_returns_monthly = estimate_expected_returns(
                train_sub,
                end_date=train_end,
                shrinkage=shrinkage,
                use_momentum=use_momentum,
                use_alpha=use_alpha,
                factor_mean_window=factor_mean_window,
                beta_window=beta_window,
            )
            exp_returns_annual = (
                exp_returns_monthly.reindex(period_tickers).values * 12
            )
        except Exception as e:
            logger.warning(f"FF model failed at {train_end.date()}: {e}")
            continue

        factor_exposures = None
        if store_factor_exposures:
            try:
                factor_exposures = estimate_factor_exposures(
                    train_sub,
                    end_date=train_end,
                    use_momentum=use_momentum,
                    factor_mean_window=factor_mean_window,
                    beta_window=beta_window,
                )
            except Exception as e:
                logger.warning(f"Factor exposures unavailable at {train_end.date()}: {e}")

        # ── Step 2: Annualized covariance (in-sample only) ────────────────
        train_fit = train_sub[period_tickers].fillna(0)
        cov_matrix = _annualized_covariance(train_fit, use_ledoit_wolf)

        period_max_weight = _regime_adjusted_max_weight(
            train_fit, max_weight, use_regime, regime_high_vol_scale
        )

        # ── Step 3: Optimize weights ──────────────────────────────────────
        opt_kwargs = {
            'risk_free_rate': risk_free_rate,
            'max_weight': period_max_weight,
            'optimization_objective': optimization_objective,
            'returns_matrix': train_fit.values,
        }
        if optimization_objective == 'cvar' and cvar_target_return is not None:
            opt_kwargs['cvar_target_return'] = cvar_target_return
        n = len(period_tickers)
        opt_prev = None
        if prev_weights is not None and prev_tickers is not None and max_turnover is not None:
            opt_prev = _align_weights_to_tickers(
                dict(zip(prev_tickers, prev_weights)), period_tickers,
            )
            opt_kwargs['prev_weights'] = opt_prev
            opt_kwargs['max_turnover'] = max_turnover

        objective_used = "equal_weight"
        try:
            weights, objective_used = optimize_portfolio(
                exp_returns_annual, cov_matrix,
                **opt_kwargs,
            )
        except Exception as e:
            logger.warning(
                f"Optimization failed at {train_end.date()}: {e}. "
                f"Falling back to equal weight."
            )
            weights = np.ones(n) / n
            objective_used = "equal_weight"

        weights = _apply_min_weight_threshold(
            weights, min_weight_threshold, period_max_weight,
        )

        # ── Step 4: Record OOS portfolio returns ──────────────────────────
        weight_dict = dict(zip(period_tickers, weights))
        wh_entry = {
            'date': train_end,
            'weights': weight_dict,
            'period_tickers': period_tickers,
            'objective_used': objective_used,
        }
        if factor_exposures is not None:
            wh_entry['factor_exposures'] = factor_exposures
        weight_history.append(wh_entry)

        oos_fit = oos_sub.reindex(columns=period_tickers).fillna(0)
        oos_portfolio = oos_fit @ weights

        if (
            prev_weights is not None
            and prev_tickers is not None
            and prev_oos_daily is not None
        ):
            drifted = _compute_drifted_weights(prev_weights, prev_tickers, prev_oos_daily)
            drifted_aligned = _align_weights_to_tickers(
                dict(zip(prev_tickers, drifted)), period_tickers,
            )
        else:
            drifted_aligned = np.zeros(n)

        turnover = 0.5 * np.sum(np.abs(weights - drifted_aligned))
        txn_cost = turnover * (transaction_cost_bps / 10_000)
        if txn_cost > 0 and len(oos_portfolio) > 0:
            oos_portfolio = oos_portfolio.copy()
            oos_portfolio.iloc[0] -= txn_cost

        prev_weights = weights.copy()
        prev_tickers = period_tickers.copy()
        prev_oos_daily = oos_fit.copy()
        prev_universe_set = universe_set
        all_oos_returns.append(oos_portfolio)

        top = sorted(weight_dict.items(), key=lambda x: -x[1])[:3]
        top_str = ', '.join(f"{t}={w:.0%}" for t, w in top)
        logger.info(f"  [{train_end.date()} → {oos_end.date()}]  Top 3: {top_str}")

    if not all_oos_returns:
        raise ValueError(
            "No out-of-sample periods were generated. "
            "Increase your date range or reduce train_months."
        )

    portfolio_returns = pd.concat(all_oos_returns).sort_index()
    # Safety: drop any accidental duplicate dates at period boundaries
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]

    return portfolio_returns, weight_history


# ─────────────────────────────────────────────────────────────────────────────
# Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _annualized_sharpe_from_excess(excess: np.ndarray, ann_factor: int = 252) -> float:
    if excess.std() < 1e-12:
        return np.nan
    return float((excess.mean() / excess.std()) * np.sqrt(ann_factor))


def _block_bootstrap_sharpe_ci(
    daily_returns: pd.Series,
    risk_free_rate: float | pd.Series = 0.02,
    block_size: int = 21,
    n_bootstrap: int = 1000,
    ci_alpha: float = 0.05,
    benchmark_returns: pd.Series | None = None,
) -> tuple[float, float, float | None, float | None]:
    """Block bootstrap 95% CI for Sharpe; optional paired diff vs benchmark."""
    r = daily_returns.values
    n = len(r)
    if n < block_size * 2:
        return np.nan, np.nan, np.nan, np.nan

    if isinstance(risk_free_rate, pd.Series):
        rf = risk_free_rate.reindex(daily_returns.index, method='ffill').fillna(0).values
    else:
        rf = np.full(n, risk_free_rate / 252)

    bench = None
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(daily_returns.index).fillna(0).values

    sharpes: list[float] = []
    diffs: list[float] = []
    ann_factor = 252

    for _ in range(n_bootstrap):
        idx: list[int] = []
        while len(idx) < n:
            start = np.random.randint(0, max(1, n - block_size + 1))
            idx.extend(range(start, min(start + block_size, n)))
        idx_arr = np.array(idx[:n])
        sample = r[idx_arr]
        excess = sample - rf[idx_arr]
        s = _annualized_sharpe_from_excess(excess, ann_factor)
        if np.isfinite(s):
            sharpes.append(s)
        if bench is not None:
            bench_excess = bench[idx_arr] - rf[idx_arr]
            sb = _annualized_sharpe_from_excess(bench_excess, ann_factor)
            if np.isfinite(s) and np.isfinite(sb):
                diffs.append(s - sb)

    if len(sharpes) < 10:
        return np.nan, np.nan, np.nan, np.nan

    lo = (ci_alpha / 2) * 100
    hi = (1 - ci_alpha / 2) * 100
    diff_lo = diff_hi = None
    if len(diffs) >= 10:
        diff_lo = float(np.percentile(diffs, lo))
        diff_hi = float(np.percentile(diffs, hi))
    return (
        float(np.percentile(sharpes, lo)),
        float(np.percentile(sharpes, hi)),
        diff_lo,
        diff_hi,
    )


def compute_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float | pd.Series = 0.02,
    sharpe_bootstrap: bool = False,
    block_size: int = 21,
    n_bootstrap: int = 1000,
    ci_alpha: float = 0.05,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """
    Compute standard performance metrics from a daily returns series.

    risk_free_rate may be a scalar (annual) or a daily pd.Series (FF RF).
    """
    cumulative = (1 + daily_returns).cumprod()
    n_years = len(daily_returns) / 252

    ann_return = cumulative.iloc[-1] ** (1 / n_years) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)

    if isinstance(risk_free_rate, pd.Series):
        rf_daily = risk_free_rate.reindex(daily_returns.index, method='ffill').fillna(0)
        excess = daily_returns - rf_daily
        sharpe = _annualized_sharpe_from_excess(excess.values)
        ann_rf = float(rf_daily.mean() * 252)
    else:
        excess = daily_returns - risk_free_rate / 252
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-12 else np.nan
        ann_rf = risk_free_rate

    downside_vol = excess[excess < 0].std() * np.sqrt(252)
    sortino = (ann_return - ann_rf) / downside_vol if downside_vol > 1e-12 else np.nan

    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    metrics = {
        'annual_return': ann_return,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'total_return': cumulative.iloc[-1] - 1,
    }

    if sharpe_bootstrap:
        lo, hi, dlo, dhi = _block_bootstrap_sharpe_ci(
            daily_returns,
            risk_free_rate,
            block_size,
            n_bootstrap,
            ci_alpha,
            benchmark_returns=benchmark_returns,
        )
        metrics['sharpe_ci_lower'] = lo
        metrics['sharpe_ci_upper'] = hi
        if dlo is not None:
            metrics['sharpe_diff_ci_lower'] = dlo
            metrics['sharpe_diff_ci_upper'] = dhi

    return metrics


def _compute_dynamic_ew_returns(
    daily_returns: pd.DataFrame,
    weight_history: list,
) -> pd.Series:
    """Equal-weight OOS returns using each period's tradable universe."""
    parts: list[pd.Series] = []
    for i, wh in enumerate(weight_history):
        tickers = wh['period_tickers']
        train_end = wh['date']
        if i + 1 < len(weight_history):
            oos_end = weight_history[i + 1]['date']
            mask = (daily_returns.index > train_end) & (daily_returns.index <= oos_end)
        else:
            mask = daily_returns.index > train_end
        oos = daily_returns.loc[mask, tickers].fillna(0)
        if oos.empty:
            continue
        w = np.ones(len(tickers)) / len(tickers)
        parts.append(oos @ w)
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep='first')]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    portfolio_returns: pd.Series,
    ew_returns: pd.Series,
    spy_returns: pd.Series,
    qqq_returns: pd.Series,
    weight_history: list,
):
    """
    Three-panel figure:
      1. Cumulative returns (optimized, EW, SPY, QQQ) with rebalancing markers.
      2. Drawdown over time.
      3. Weight evolution across rebalancing periods (stacked bar).
    """
    port_cumulative = (1 + portfolio_returns).cumprod()
    ew_cumulative   = (1 + ew_returns).cumprod()
    spy_cumulative  = (1 + spy_returns).cumprod()
    qqq_cumulative  = (1 + qqq_returns).cumprod()

    port_rolling_max = port_cumulative.cummax()
    port_drawdown    = (port_cumulative - port_rolling_max) / port_rolling_max

    weight_df = pd.DataFrame(
        [wh['weights'] for wh in weight_history],
        index=[wh['date'] for wh in weight_history],
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.suptitle(
        'Walk-Forward Backtest — Out-of-Sample Results',
        fontsize=15, fontweight='bold', y=0.98
    )

    # ── Panel 1: Cumulative returns ───────────────────────────────────────
    ax = axes[0]
    ax.plot(port_cumulative, label='Optimized Portfolio (OOS)', linewidth=2, color='steelblue')
    ax.plot(ew_cumulative,   label='Equal-Weight Same Universe', linewidth=2, color='seagreen',
            linestyle='-.')
    ax.plot(spy_cumulative,  label='S&P 500 (SPY)', linewidth=2, color='darkorange',
            linestyle='--')
    ax.plot(qqq_cumulative,  label='QQQ', linewidth=2, color='purple', linestyle=':')

    # Grey vertical lines at each rebalancing date
    for wh in weight_history:
        d = wh['date']
        if d in port_cumulative.index:
            ax.axvline(x=d, color='grey', alpha=0.25, linewidth=0.8)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel('Growth of $1')
    ax.set_xlabel('Date')
    ax.set_title('Cumulative Performance  (grey lines = rebalancing dates)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(port_drawdown.index, port_drawdown, 0,
                    color='steelblue', alpha=0.4, label='Portfolio Drawdown')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Weight evolution ─────────────────────────────────────────
    ax = axes[2]
    n_cols = max(len(weight_df.columns), 1)
    colors = plt.cm.tab20(np.linspace(0, 1, n_cols))
    weight_df.plot(kind='bar', stacked=True, ax=ax, color=colors, legend=True, width=0.8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xticklabels(
        [d.strftime('%Y-%m') for d in weight_df.index],
        rotation=45, ha='right', fontsize=8
    )
    ax.set_ylabel('Weight')
    ax.set_title('Portfolio Weights at Each Rebalancing Date')
    ax.legend(loc='upper right', fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('walk_forward_results.png', dpi=150, bbox_inches='tight')
    logger.info("✓ Saved: walk_forward_results.png")
    plt.close(fig)


def plot_robustness_periods(
    period_results: dict[str, dict],
    save_path: str = 'robustness_periods.png',
):
    """Cumulative optimized (Sharpe) vs SPY per test period."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['steelblue', 'seagreen', 'darkorange']

    for i, (label, runs) in enumerate(period_results.items()):
        run = runs.get('sharpe')
        if run is None or run.get('failed'):
            continue
        port = run['portfolio_returns']
        spy = run['spy_returns']
        common = port.index.intersection(spy.index)
        if common.empty:
            continue
        port_c = (1 + port.loc[common]).cumprod()
        spy_c = (1 + spy.loc[common]).cumprod()
        c = colors[i % len(colors)]
        ax.plot(port_c, label=f'{label} Optimized', color=c, linewidth=2)
        ax.plot(spy_c, label=f'{label} SPY', color=c, linewidth=1.5, linestyle='--', alpha=0.8)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel('Growth of $1')
    ax.set_xlabel('Date')
    ax.set_title('Robustness Across Periods — Optimized (Sharpe) vs SPY')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close(fig)


def _midpoint_top_holdings(weight_history: list, n: int = 3) -> list[tuple[str, float]]:
    """Top holdings at the midpoint rebalance of a walk-forward run."""
    if not weight_history:
        return []
    mid = weight_history[len(weight_history) // 2]
    return sorted(mid['weights'].items(), key=lambda x: -x[1])[:n]


def _print_robustness_table(table_rows: list[dict], aggregates: dict):
    print("\n" + "=" * 100)
    print("ROBUSTNESS ACROSS PERIODS")
    print("=" * 100)
    print(
        f"{'Period':<12} {'Objective':<9} {'Sharpe':>7} {'DSR':>7} {'AnnRet':>8} {'MaxDD':>8} "
        f"{'Calmar':>7} {'Δ vs SPY 95% CI':>18} {'SPY Sharpe':>11}"
    )
    print("-" * 100)
    for row in table_rows:
        ci = row['paired_ci']
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else "—"
        dsr = row.get('dsr')
        dsr_str = f"{dsr:.3f}" if dsr is not None and dsr == dsr else "—"
        print(
            f"{row['period']:<12} {row['objective']:<9} "
            f"{row['sharpe']:>7.2f} {dsr_str:>7} {row['ann_ret']:>7.1%} {row['max_dd']:>7.1%} "
            f"{row['calmar']:>7.2f} {ci_str:>18} {row['spy_sharpe']:>11.2f}"
        )
    print("-" * 100)
    for obj in ('sharpe', 'cvar'):
        agg = aggregates[obj]
        print(
            f"{'Aggregate':<12} {obj.capitalize():<9} "
            f"mean={agg['mean']:.2f}  std={agg['std']:.2f}  min={agg['min']:.2f}  "
            f"n_periods_positive={agg['n_positive']}/3"
        )
    print("=" * 100)


TEST_PERIODS = [
    ('2010-2014', '2007-01-01', '2015-01-01'),
    ('2015-2019', '2012-01-01', '2020-01-01'),
    ('2020-2024', '2017-01-01', '2025-01-01'),
]
DATA_START = '2007-01-01'
DATA_END = '2025-01-01'


def run_one_period(
    period_label: str,
    start: str,
    end: str,
    all_returns: pd.DataFrame,
    universe_history: pd.DataFrame,
    config: dict,
    objective: str,
    spy_returns_full: pd.Series,
) -> dict:
    """
    Run a single walk-forward backtest on a specified date window.
    Returns dict with metrics, portfolio_returns, weight_history, label.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    period_returns = all_returns.loc[start_ts:end_ts]
    if period_returns.empty or len(period_returns) < 252:
        raise ValueError(f"Insufficient return data for {period_label}")

    rf_daily = get_rf_daily_series(start, end)
    period_config = {**config, 'optimization_objective': objective}

    portfolio_returns, weight_history = walk_forward_backtest(
        period_returns,
        universe_history,
        train_months=period_config['train_months'],
        rebal_months=period_config['rebal_months'],
        risk_free_rate=period_config['risk_free'],
        max_weight=period_config['max_weight'],
        transaction_cost_bps=period_config['transaction_cost_bps'],
        shrinkage=period_config['shrinkage'],
        use_ledoit_wolf=period_config['use_ledoit_wolf'],
        use_momentum=period_config['use_momentum'],
        optimization_objective=objective,
        max_turnover=period_config['max_turnover'],
        use_regime=period_config['use_regime'],
        regime_high_vol_scale=period_config['regime_high_vol_scale'],
        store_factor_exposures=False,
        min_weight_threshold=period_config['min_weight_threshold'],
        use_alpha=period_config['use_alpha'],
        factor_mean_window=period_config['factor_mean_window'],
        beta_window=period_config['beta_window'],
        max_universe_size=period_config['max_universe_size'],
        cvar_target_return=period_config.get('cvar_target_return'),
    )

    common_idx = portfolio_returns.index.intersection(spy_returns_full.index)
    portfolio_returns = portfolio_returns.loc[common_idx]
    spy_returns = spy_returns_full.loc[common_idx]

    boot_kw = {
        'sharpe_bootstrap': period_config['sharpe_bootstrap'],
        'block_size': period_config['bootstrap_block_size'],
        'n_bootstrap': period_config['bootstrap_samples'],
    }
    metrics = compute_metrics(
        portfolio_returns,
        rf_daily,
        benchmark_returns=spy_returns,
        **boot_kw,
    )
    spy_metrics = compute_metrics(spy_returns, rf_daily, **boot_kw)

    dsr, sr_star = deflated_sharpe(portfolio_returns, period_config["n_trials"])
    metrics["dsr"] = dsr
    metrics["sr_star"] = sr_star

    paired_ci = (
        metrics.get('sharpe_diff_ci_lower'),
        metrics.get('sharpe_diff_ci_upper'),
    )

    return {
        'period_label': period_label,
        'objective': objective,
        'portfolio_returns': portfolio_returns,
        'weight_history': weight_history,
        'metrics': metrics,
        'spy_returns': spy_returns,
        'spy_metrics': spy_metrics,
        'paired_ci': paired_ci,
        'failed': False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("MULTI-FACTOR PORTFOLIO OPTIMIZER  —  ROBUSTNESS EXPERIMENT")
    print("=" * 80)

    CONFIG = {
        'start': DATA_START,
        'end': DATA_END,
        'train_months': 36,
        'rebal_months': 3,
        'risk_free': 0.02,
        'max_weight': 0.20,
        'transaction_cost_bps': 10,
        'shrinkage': 0.75,
        'use_ledoit_wolf': True,
        'use_momentum': True,
        'optimization_objective': 'cvar',
        'cvar_target_return': 0.08,
        'min_weight_threshold': 0.03,
        'max_turnover': None,
        'use_regime': False,
        'regime_high_vol_scale': 0.70,
        'sharpe_bootstrap': True,
        'bootstrap_block_size': 21,
        'bootstrap_samples': 1000,
        'run_attribution': False,
        'use_alpha': False,
        'factor_mean_window': 'long_run',
        'beta_window': 24,
        'max_universe_size': 50,
        'n_trials': 12,
    }

    data_start_ts = pd.Timestamp(DATA_START)
    data_end_ts = pd.Timestamp(DATA_END)

    print(f"\n[Step 1] Loading S&P 500 history & downloading returns "
          f"({DATA_START} → {DATA_END})...")
    try:
        universe_history = load_sp500_history()
        all_tickers = get_universe_union(data_start_ts, data_end_ts, universe_history)
        u_start = get_universe_as_of(data_start_ts, universe_history)
        u_end = get_universe_as_of(data_end_ts - pd.Timedelta(days=1), universe_history)
        print(f"  ✓ Union universe: {len(all_tickers)} tickers "
              f"(snapshots: {len(u_start)} @ start → {len(u_end)} @ end)")
        all_returns = fetch_stock_returns(all_tickers, start=DATA_START, end=DATA_END)
        print(f"  ✓ Price panel: {len(all_returns.columns)} tickers, "
              f"{len(all_returns)} trading days")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return

    print("\n[Step 2] Downloading SPY benchmark (full window)...")
    spy_raw = yf.download("SPY", start=DATA_START, end=DATA_END, progress=False)["Close"]
    if isinstance(spy_raw, pd.DataFrame):
        spy_raw = spy_raw.squeeze()
    spy_returns_full = spy_raw.pct_change().dropna()

    print(f"\n[Step 3] Running {len(TEST_PERIODS)} periods × 2 objectives = "
          f"{len(TEST_PERIODS) * 2} backtests...")
    robustness_state: dict = {'periods': {}, 'config': CONFIG}
    table_rows: list[dict] = []
    period_plot_data: dict[str, dict] = {}
    failures: list[str] = []

    for period_label, start, end in TEST_PERIODS:
        robustness_state['periods'][period_label] = {}
        period_plot_data[period_label] = {}

        for objective in ('sharpe', 'cvar'):
            tag = f"{period_label} / {objective}"
            print(f"\n  → {tag}  ({start} → {end})")
            try:
                result = run_one_period(
                    period_label, start, end,
                    all_returns, universe_history, CONFIG, objective,
                    spy_returns_full,
                )
                m = result['metrics']
                spy_m = result['spy_metrics']
                table_rows.append({
                    'period': period_label,
                    'objective': objective.capitalize(),
                    'sharpe': m['sharpe'],
                    'dsr': m.get('dsr'),
                    'ann_ret': m['annual_return'],
                    'max_dd': m['max_drawdown'],
                    'calmar': m['calmar'],
                    'paired_ci': result['paired_ci'],
                    'spy_sharpe': spy_m['sharpe'],
                })
                robustness_state['periods'][period_label][objective] = {
                    'portfolio_returns': result['portfolio_returns'],
                    'weight_history': result['weight_history'],
                    'metrics': result['metrics'],
                    'spy_returns': result['spy_returns'],
                    'paired_ci': result['paired_ci'],
                }
                period_plot_data[period_label][objective] = result
                oos_s = result['portfolio_returns'].index[0].date()
                oos_e = result['portfolio_returns'].index[-1].date()
                print(f"     ✓ OOS {oos_s} → {oos_e} | Sharpe={m['sharpe']:.3f} | "
                      f"DSR={m.get('dsr', float('nan')):.3f} | "
                      f"AnnRet={m['annual_return']:.1%}")
            except Exception as e:
                msg = f"{tag}: {e}"
                logger.error(f"FAILED — {msg}")
                failures.append(msg)
                robustness_state['periods'][period_label][objective] = {'failed': True, 'error': str(e)}

    aggregates: dict[str, dict] = {}
    for obj_key, obj_label in (('sharpe', 'Sharpe'), ('cvar', 'CVaR')):
        sharpes = [
            r['sharpe'] for r in table_rows
            if r['objective'].lower() == obj_key and np.isfinite(r['sharpe'])
        ]
        aggregates[obj_key] = {
            'mean': float(np.mean(sharpes)) if sharpes else float('nan'),
            'std': float(np.std(sharpes)) if sharpes else float('nan'),
            'min': float(np.min(sharpes)) if sharpes else float('nan'),
            'n_positive': sum(1 for s in sharpes if s > 0),
        }

    if table_rows:
        _print_robustness_table(table_rows, aggregates)

    print("\n[Step 4] Midpoint holdings (Sharpe objective):")
    for period_label, _, _ in TEST_PERIODS:
        run = period_plot_data.get(period_label, {}).get('sharpe')
        if run is None or run.get('failed'):
            print(f"  {period_label}: unavailable")
            continue
        top = _midpoint_top_holdings(run['weight_history'])
        top_str = ', '.join(f"{t}={w:.0%}" for t, w in top)
        mid_date = run['weight_history'][len(run['weight_history']) // 2]['date'].date()
        print(f"  {period_label} @ {mid_date}: {top_str}")

    print("\n[Step 5] Generating robustness plot...")
    plot_ok = False
    try:
        plot_robustness_periods(period_plot_data)
        plot_ok = True
        print("  ✓ robustness_periods.png")
    except Exception as e:
        logger.error(f"Robustness plot failed: {e}")

    state_path = Path(__file__).resolve().parent / "tests" / "_robustness_state.pkl"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("wb") as f:
        pickle.dump(robustness_state, f)
    logger.info(f"✓ Saved robustness state: {state_path}")

    if failures:
        print("\nFailed runs:")
        for f in failures:
            print(f"  ✗ {f}")

    print("\n✓ Done!")
    print("=" * 65)
    return robustness_state, table_rows, aggregates, failures, plot_ok


if __name__ == "__main__":
    main()