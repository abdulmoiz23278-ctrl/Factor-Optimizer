import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.ticker as mticker  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import yfinance as yf  # type: ignore[import-untyped]
import logging

from src.factor_optimizer import optimize_portfolio
from src.data_fetcher import fetch_stock_returns
from src.factor_model import (
    estimate_expected_returns,
    estimate_factor_exposures,
    attribute_oos_returns,
)

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


def _apply_min_weight_threshold(
    weights: np.ndarray,
    min_weight: float = 0.03,
) -> np.ndarray:
    """Zero weights below min_weight, then renormalize to sum to 1."""
    if min_weight <= 0:
        return weights
    w = np.asarray(weights, dtype=float).copy()
    w[w < min_weight] = 0.0
    total = w.sum()
    if total < 1e-12:
        return np.ones(len(w)) / len(w)
    return w / total


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
    tickers: list,
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
) -> tuple:
    """
    Walk-forward portfolio optimization — eliminates lookahead bias entirely.

    At each rebalancing date t:
      1. Estimate FF expected returns using ONLY data before t  (in-sample).
      2. Optimize weights                                        (in-sample).
      3. Apply those weights to the NEXT rebal_months           (out-of-sample).
      4. Record only the out-of-sample returns.

    This means the portfolio NEVER sees future data when choosing weights.

    Args:
        daily_returns:  Full daily returns DataFrame (fetch_stock_returns output).
        tickers:        List of ticker symbols (same order as daily_returns columns).
        train_months:   Rolling training window length in months (default: 36).
        rebal_months:   Rebalancing frequency in months (default: 3 = quarterly).
        risk_free_rate: Annual risk-free rate used in Sharpe optimization.
        max_weight:     Maximum weight per asset (position size cap).
        transaction_cost_bps: One-way cost in bps applied to rebalance turnover.
        shrinkage:      Cross-sectional shrinkage for expected returns (0–1).
        use_ledoit_wolf: Use Ledoit-Wolf covariance if sklearn is available.
        use_momentum:      Add 12-1 cross-sectional momentum as 6th regression factor.
        optimization_objective: 'sharpe' or 'cvar'.
        max_turnover:    Max one-way turnover when prev_weights supplied to optimizer.
        use_regime:      HMM regime filter; reduce max_weight in high-vol state.
        regime_high_vol_scale: Multiplier on max_weight in high-vol state (default 0.70).
        store_factor_exposures: Store per-rebalance factor betas for attribution.
        min_weight_threshold: Zero weights below this level, then renormalize.

    Returns:
        portfolio_returns (pd.Series):  Daily OOS portfolio returns.
        weight_history   (list[dict]):  Weights + date at each rebalancing point.
    """
    # Month-end dates — used as the windowing anchor
    monthly_ends = daily_returns.resample('ME').last().index
    n_months = len(monthly_ends)

    # ~21 trading days per month: used to slice the training window from daily data
    approx_train_days = train_months * 21

    all_oos_returns: list[pd.Series] = []
    weight_history: list[dict] = []
    prev_weights: np.ndarray | None = None

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

        # ── Step 1: Estimate expected returns (FF model, in-sample only) ──
        try:
            exp_returns_monthly = estimate_expected_returns(
                train_daily,
                end_date=train_end,
                shrinkage=shrinkage,
                use_momentum=use_momentum,
            ).values
            exp_returns_annual  = exp_returns_monthly * 12
        except Exception as e:
            logger.warning(f"FF model failed at {train_end.date()}: {e}")
            continue

        factor_exposures = None
        if store_factor_exposures:
            try:
                factor_exposures = estimate_factor_exposures(
                    train_daily, end_date=train_end, use_momentum=use_momentum
                )
            except Exception as e:
                logger.warning(f"Factor exposures unavailable at {train_end.date()}: {e}")

        # ── Step 2: Annualized covariance (in-sample only) ────────────────
        cov_matrix = _annualized_covariance(train_daily, use_ledoit_wolf)

        period_max_weight = _regime_adjusted_max_weight(
            train_daily, max_weight, use_regime, regime_high_vol_scale
        )

        # ── Step 3: Optimize weights ──────────────────────────────────────
        opt_kwargs = {
            'risk_free_rate': risk_free_rate,
            'max_weight': period_max_weight,
            'optimization_objective': optimization_objective,
            'returns_matrix': train_daily.values,
        }
        if prev_weights is not None and max_turnover is not None:
            opt_kwargs['prev_weights'] = prev_weights
            opt_kwargs['max_turnover'] = max_turnover

        try:
            weights = optimize_portfolio(
                exp_returns_annual, cov_matrix,
                **opt_kwargs,
            )
        except Exception as e:
            logger.warning(
                f"Optimization failed at {train_end.date()}: {e}. "
                f"Falling back to equal weight."
            )
            weights = np.ones(len(tickers)) / len(tickers)

        weights = _apply_min_weight_threshold(weights, min_weight_threshold)

        # ── Step 4: Record OOS portfolio returns ──────────────────────────
        wh_entry = {
            'date':    train_end,
            'weights': dict(zip(tickers, weights)),
        }
        if factor_exposures is not None:
            wh_entry['factor_exposures'] = factor_exposures
        weight_history.append(wh_entry)

        oos_portfolio = oos_daily @ weights  # daily portfolio return = Σ wᵢ rᵢ

        # Transaction cost on first OOS day: cost_bps × one-way turnover
        old_w = prev_weights if prev_weights is not None else np.zeros(len(tickers))
        turnover = 0.5 * np.sum(np.abs(weights - old_w))
        txn_cost = turnover * (transaction_cost_bps / 10_000)
        if txn_cost > 0 and len(oos_portfolio) > 0:
            oos_portfolio = oos_portfolio.copy()
            oos_portfolio.iloc[0] -= txn_cost

        prev_weights = weights.copy()
        all_oos_returns.append(oos_portfolio)

        top = sorted(zip(tickers, weights), key=lambda x: -x[1])[:3]
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

def _block_bootstrap_sharpe_ci(
    daily_returns: pd.Series,
    risk_free_rate: float,
    block_size: int = 21,
    n_bootstrap: int = 1000,
    ci_alpha: float = 0.05,
) -> tuple[float, float]:
    """Block bootstrap 95% CI for annualized Sharpe ratio."""
    r = daily_returns.values
    n = len(r)
    if n < block_size * 2:
        return np.nan, np.nan

    sharpes = []
    ann_factor = 252
    for _ in range(n_bootstrap):
        idx: list[int] = []
        while len(idx) < n:
            start = np.random.randint(0, max(1, n - block_size + 1))
            idx.extend(range(start, min(start + block_size, n)))
        sample = r[np.array(idx[:n])]
        mu = sample.mean() * ann_factor
        vol = sample.std() * np.sqrt(ann_factor)
        if vol > 1e-12:
            sharpes.append((mu - risk_free_rate) / vol)

    if len(sharpes) < 10:
        return np.nan, np.nan

    lo = (ci_alpha / 2) * 100
    hi = (1 - ci_alpha / 2) * 100
    return float(np.percentile(sharpes, lo)), float(np.percentile(sharpes, hi))


def compute_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.02,
    sharpe_bootstrap: bool = False,
    block_size: int = 21,
    n_bootstrap: int = 1000,
    ci_alpha: float = 0.05,
) -> dict:
    """
    Compute standard performance metrics from a daily returns series.

    Returns dict with: annual_return, annual_vol, sharpe, sortino, max_drawdown,
                       calmar, total_return.
                       Optionally sharpe_ci_lower, sharpe_ci_upper (block bootstrap).
    """
    cumulative = (1 + daily_returns).cumprod()
    n_years    = len(daily_returns) / 252

    ann_return = cumulative.iloc[-1] ** (1 / n_years) - 1
    ann_vol    = daily_returns.std() * np.sqrt(252)
    sharpe     = (ann_return - risk_free_rate) / ann_vol

    downside_vol = daily_returns[daily_returns < 0].std() * np.sqrt(252)
    sortino      = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else np.nan

    rolling_max  = cumulative.cummax()
    drawdown     = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    calmar       = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    metrics = {
        'annual_return': ann_return,
        'annual_vol':    ann_vol,
        'sharpe':        sharpe,
        'sortino':       sortino,
        'max_drawdown':  max_drawdown,
        'calmar':        calmar,
        'total_return':  cumulative.iloc[-1] - 1,
    }

    if sharpe_bootstrap:
        lo, hi = _block_bootstrap_sharpe_ci(
            daily_returns, risk_free_rate, block_size, n_bootstrap, ci_alpha
        )
        metrics['sharpe_ci_lower'] = lo
        metrics['sharpe_ci_upper'] = hi

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    portfolio_returns: pd.Series,
    ew_returns: pd.Series,
    spy_returns: pd.Series,
    qqq_returns: pd.Series,
    weight_history: list,
    tickers: list,
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
    colors = plt.cm.tab20(np.linspace(0, 1, len(tickers)))
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("MULTI-FACTOR PORTFOLIO OPTIMIZER  —  WALK-FORWARD BACKTEST")
    print("=" * 80)

    TICKERS = [
        # Original 15
        "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA",
        "JPM",  "V",    "UNH",  "HD",   "PG",   "DIS",  "MA", "PYPL", "INTC",
        # +15 for 11 GICS sector coverage
        "JNJ",  "LLY",                    # Health Care
        "XOM",  "CVX",                    # Energy
        "CAT",  "HON",                    # Industrials
        "KO",   "PEP",                    # Consumer Staples
        "NEE",  "DUK",                    # Utilities
        "BAC",  "GS",                     # Financials
        "LIN",                            # Materials
        "AMT",                            # Real Estate
        "CMCSA",                          # Communication Services
    ]

    # Config — centralise all parameters here
    CONFIG = {
        'start':         '2015-01-01',
        'end':           '2025-01-01',
        'train_months':  36,    # 3-year rolling training window
        'rebal_months':  3,     # Quarterly rebalancing
        'risk_free':     0.02,
        'max_weight':    0.20,
        'transaction_cost_bps': 10,
        'shrinkage':     0.75,
        'use_ledoit_wolf': True,
        # Extensions (defaults preserve prior behaviour)
        'use_momentum': False,
        'optimization_objective': 'cvar',
        'min_weight_threshold': 0.03,
        'max_turnover': None,
        'use_regime': False,
        'regime_high_vol_scale': 0.70,
        'sharpe_bootstrap': True,
        'bootstrap_block_size': 21,
        'bootstrap_samples': 1000,
        'run_attribution': False,
    }

    # ── Step 1: Download data ─────────────────────────────────────────────
    print(f"\n[Step 1] Downloading {len(TICKERS)} stocks "
          f"({CONFIG['start']} → {CONFIG['end']})...")
    try:
        returns = fetch_stock_returns(
            TICKERS, start=CONFIG['start'], end=CONFIG['end']
        )
        print(f"  ✓ {len(returns.columns)} tickers, {len(returns)} trading days")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return

    # ── Step 2: Walk-forward backtest ─────────────────────────────────────
    # Each set of weights is estimated using ONLY past data (no lookahead).
    # The reported returns are purely out-of-sample.
    print(
        f"\n[Step 2] Walk-forward backtest  "
        f"(train={CONFIG['train_months']}m, rebal={CONFIG['rebal_months']}m)..."
    )
    try:
        portfolio_returns, weight_history = walk_forward_backtest(
            returns, TICKERS,
            train_months=CONFIG['train_months'],
            rebal_months=CONFIG['rebal_months'],
            risk_free_rate=CONFIG['risk_free'],
            max_weight=CONFIG['max_weight'],
            transaction_cost_bps=CONFIG['transaction_cost_bps'],
            shrinkage=CONFIG['shrinkage'],
            use_ledoit_wolf=CONFIG['use_ledoit_wolf'],
            use_momentum=CONFIG['use_momentum'],
            optimization_objective=CONFIG['optimization_objective'],
            max_turnover=CONFIG['max_turnover'],
            use_regime=CONFIG['use_regime'],
            regime_high_vol_scale=CONFIG['regime_high_vol_scale'],
            store_factor_exposures=CONFIG['run_attribution'],
            min_weight_threshold=CONFIG['min_weight_threshold'],
        )
        oos_start = portfolio_returns.index[0]
        oos_end   = portfolio_returns.index[-1]
        print(f"  ✓ {len(weight_history)} rebalancing periods | "
              f"OOS window: {oos_start.date()} → {oos_end.date()}")
    except Exception as e:
        logger.error(f"Walk-forward failed: {e}")
        return

    # ── Step 3: Benchmarks ────────────────────────────────────────────────
    print("\n[Step 3] Downloading benchmarks (SPY, QQQ)...")

    def _download_returns(symbol: str) -> pd.Series:
        raw = yf.download(symbol, start=CONFIG['start'], end=CONFIG['end'], progress=False)["Close"]
        if isinstance(raw, pd.DataFrame):
            raw = raw.squeeze()
        return raw.pct_change().dropna()

    spy_returns_full = _download_returns("SPY")
    qqq_returns_full = _download_returns("QQQ")

    # Align all series to the OOS window
    common_idx = portfolio_returns.index
    portfolio_returns = portfolio_returns.loc[common_idx]
    ew_returns = (returns.loc[common_idx] @ (np.ones(len(TICKERS)) / len(TICKERS)))
    spy_returns_oos = spy_returns_full.reindex(common_idx).dropna()
    qqq_returns_oos = qqq_returns_full.reindex(common_idx).dropna()
    common_idx = (
        portfolio_returns.index
        .intersection(ew_returns.index)
        .intersection(spy_returns_oos.index)
        .intersection(qqq_returns_oos.index)
    )
    portfolio_returns = portfolio_returns.loc[common_idx]
    ew_returns        = ew_returns.loc[common_idx]
    spy_returns_oos   = spy_returns_oos.loc[common_idx]
    qqq_returns_oos   = qqq_returns_oos.loc[common_idx]
    oos_start = common_idx[0]
    oos_end   = common_idx[-1]

    # ── Step 4: Performance metrics ───────────────────────────────────────
    print("\n[Step 4] Computing metrics...")
    boot_kw = {
        'sharpe_bootstrap': CONFIG['sharpe_bootstrap'],
        'block_size': CONFIG['bootstrap_block_size'],
        'n_bootstrap': CONFIG['bootstrap_samples'],
    }
    port_m = compute_metrics(portfolio_returns, CONFIG['risk_free'], **boot_kw)
    ew_m   = compute_metrics(ew_returns,        CONFIG['risk_free'], **boot_kw)
    spy_m  = compute_metrics(spy_returns_oos,   CONFIG['risk_free'], **boot_kw)
    qqq_m  = compute_metrics(qqq_returns_oos,   CONFIG['risk_free'], **boot_kw)

    benchmarks = [
        ('Optimized Portfolio', port_m),
        ('Equal-Weight Same Universe', ew_m),
        ('SPY', spy_m),
        ('QQQ', qqq_m),
    ]

    rows = [
        ('Annual Return',    'annual_return', '.2%'),
        ('Annual Volatility','annual_vol',    '.2%'),
        ('Sharpe Ratio',     'sharpe',        '.3f'),
        ('Sortino Ratio',    'sortino',       '.3f'),
        ('Max Drawdown',     'max_drawdown',  '.2%'),
        ('Calmar Ratio',     'calmar',        '.3f'),
        ('Total Return',     'total_return',  '.2%'),
    ]

    col_w = 18
    bench_labels = ['Optimized', 'Eq-Weight Univ.', 'SPY', 'QQQ']
    print("\n" + "=" * 88)
    print("WALK-FORWARD RESULTS  (OUT-OF-SAMPLE ONLY — no lookahead bias)")
    print("=" * 88)
    print(f"\n  OOS period:  {oos_start.date()} → {oos_end.date()}")
    print(f"  Rebalanced:  {len(weight_history)}x (quarterly)")
    print(f"  Txn cost:    {CONFIG['transaction_cost_bps']} bps per one-way turnover")
    print(f"  Objective:   {CONFIG['optimization_objective']} | "
          f"Shrinkage: {CONFIG['shrinkage']:.0%} | "
          f"Min weight: {CONFIG['min_weight_threshold']:.0%}\n")
    header = f"  {'Metric':<22}" + "".join(f"{name:>{col_w}}" for name in bench_labels)
    print(header)
    print(f"  {'-' * (22 + col_w * len(benchmarks))}")
    for label, key, fmt in rows:
        vals = []
        for _, m in benchmarks:
            v = m[key]
            vals.append(format(v, fmt) if v is not None and not (isinstance(v, float) and np.isnan(v)) else '—')
        print(f"  {label:<22}" + "".join(f"{v:>{col_w}}" for v in vals))

    if CONFIG['sharpe_bootstrap']:
        print(f"  {'Sharpe 95% CI (lo)':<22}", end="")
        for _, m in benchmarks:
            v = m.get('sharpe_ci_lower', np.nan)
            vals = format(v, '.3f') if v is not None and not (isinstance(v, float) and np.isnan(v)) else '—'
            print(f"{vals:>{col_w}}", end="")
        print()
        print(f"  {'Sharpe 95% CI (hi)':<22}", end="")
        for _, m in benchmarks:
            v = m.get('sharpe_ci_upper', np.nan)
            vals = format(v, '.3f') if v is not None and not (isinstance(v, float) and np.isnan(v)) else '—'
            print(f"{vals:>{col_w}}", end="")
        print()

    print("=" * 88)

    if CONFIG['run_attribution']:
        print("\n" + "=" * 88)
        print("OOS RETURN ATTRIBUTION  (factor contributions + residual alpha)")
        print("=" * 88)
        attr = attribute_oos_returns(
            portfolio_returns, weight_history, returns,
            use_momentum=CONFIG['use_momentum'],
        )
        if attr:
            print(f"\n  Total realized (sum of daily):  {attr['total_realized']:.4f}")
            print(f"  Alpha contribution:             {attr['total_alpha']:.4f}")
            print(f"  Risk-free contribution:         {attr['total_rf']:.4f}")
            for fac, val in attr['factor_contributions'].items():
                print(f"  {fac:12s} contribution:       {val:.4f}")
            print(f"  Residual:                       {attr['total_residual']:.4f}")
        else:
            print("  Attribution unavailable.")

    # ── Last rebalancing allocation ───────────────────────────────────────
    print(f"\nMost Recent Allocation  ({weight_history[-1]['date'].date()}):")
    print("-" * 42)
    for ticker, w in sorted(weight_history[-1]['weights'].items(), key=lambda x: -x[1]):
        if w > 0.001:
            bar = '█' * int(w * 60)
            print(f"  {ticker:6s}  {w:5.1%}  {bar}")
    print("-" * 42)

    # ── Step 5: Plots ─────────────────────────────────────────────────────
    print("\n[Step 5] Generating plots...")
    try:
        plot_results(
            portfolio_returns, ew_returns, spy_returns_oos, qqq_returns_oos,
            weight_history, TICKERS,
        )
        print("  ✓ walk_forward_results.png")
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

    print("\n✓ Done!")
    print("=" * 65)


if __name__ == "__main__":
    main()