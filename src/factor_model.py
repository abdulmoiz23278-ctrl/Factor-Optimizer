import io
import logging
import urllib.request
import zipfile

import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import statsmodels.api as sm  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_FF_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
_FF_CACHE: pd.DataFrame | None = None
_UMD_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)
_UMD_CACHE: pd.Series | None = None
BASE_FACTOR_COLS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
UMD_COL = 'UMD'


def _load_fama_french_from_zip() -> pd.DataFrame:
    """Download and parse the monthly Fama-French 5-factor series from Ken French."""
    with urllib.request.urlopen(_FF_URL) as resp:
        with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith('.csv'))
            text = zf.read(csv_name).decode('utf-8')

    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    rows: list[tuple] = []

    for line in text.splitlines():
        parts = [p.strip() for p in line.split(',')]
        date_str = parts[0]

        if len(date_str) == 6 and date_str.isdigit():
            rows.append((date_str, *parts[1:7]))
        elif rows and (len(date_str) == 8 and date_str.isdigit()):
            break

    if not rows:
        raise ValueError("No monthly Fama-French rows found in downloaded file.")

    ff = pd.DataFrame(rows, columns=['Date'] + factor_cols)
    ff['Date'] = pd.to_datetime(ff['Date'], format='%Y%m')
    ff = ff.set_index('Date')
    ff = ff.astype(float) / 100

    return ff


def get_fama_french_factors() -> pd.DataFrame:
    """
    Fetch Fama-French 5 factors from Ken French's data library.

    Returns:
        pd.DataFrame: Monthly factor returns as decimals (not percentages).
                      Columns: Mkt-RF, SMB, HML, RMW, CMA, RF
    """
    global _FF_CACHE
    try:
        if _FF_CACHE is None:
            _FF_CACHE = _load_fama_french_from_zip()
        logger.info(f"✓ Loaded Fama-French 5 factors: {_FF_CACHE.shape[0]} months")
        return _FF_CACHE
    except Exception as e:
        logger.error(f"Failed to load Fama-French factors: {e}")
        raise


def _load_umd_from_zip() -> pd.Series:
    """Download Ken French monthly momentum factor (Mom). Returns Series in decimals."""
    with urllib.request.urlopen(_UMD_URL) as resp:
        with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith('.csv'))
            text = zf.read(csv_name).decode('utf-8', errors='replace')
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(',')]
        date_str = parts[0]
        if len(date_str) == 6 and date_str.isdigit() and len(parts) >= 2:
            rows.append((date_str, parts[1]))
        elif rows and len(date_str) == 8 and date_str.isdigit():
            break
    if not rows:
        raise ValueError("No monthly UMD rows found in download.")
    df = pd.DataFrame(rows, columns=['Date', 'Mom'])
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m')
    series = df.set_index('Date')['Mom'].astype(float) / 100.0
    series.name = UMD_COL
    return series


def get_umd_factor() -> pd.Series:
    global _UMD_CACHE
    if _UMD_CACHE is None:
        _UMD_CACHE = _load_umd_from_zip()
    return _UMD_CACHE


def get_rf_daily_series(start, end) -> pd.Series:
    """Daily RF series from FF monthly RF, forward-filled (monthly / 21)."""
    ff = get_fama_french_factors()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rf_monthly = ff['RF'].loc[:end_ts]
    daily_idx = pd.bdate_range(start_ts, end_ts)
    return rf_monthly.reindex(daily_idx, method='ffill') / 21.0


def _factor_means_and_rf(
    factors: pd.DataFrame,
    factor_cols: list[str],
    in_sample_data: pd.DataFrame,
    factor_mean_window: str,
) -> tuple[pd.Series, float]:
    """Expected factor means and RF for forward projection."""
    if factor_mean_window == 'in_sample':
        return in_sample_data[factor_cols].mean(), float(in_sample_data['RF'].mean())
    if factor_mean_window == 'rolling_120m':
        tail = factors[factor_cols].iloc[-120:]
        rf_tail = factors['RF'].iloc[-120:]
        if len(tail) < 12:
            tail = factors[factor_cols]
            rf_tail = factors['RF']
        return tail.mean(), float(rf_tail.mean())
    return factors[factor_cols].mean(), float(factors['RF'].mean())


def _factors_through_date(
    end_date: pd.Timestamp | None,
    use_momentum: bool,
) -> pd.DataFrame:
    """FF factors (+ optional UMD) through publication-lag cutoff."""
    factors = get_fama_french_factors()
    cutoff_month = None
    if end_date is not None:
        cutoff_month = (
            pd.Timestamp(end_date) - pd.DateOffset(months=1)
        ).to_period('M').to_timestamp()
        factors = factors[factors.index <= cutoff_month]
    if use_momentum:
        umd = get_umd_factor()
        if cutoff_month is not None:
            umd = umd[umd.index <= cutoff_month]
        factors = factors.join(umd, how='left')
    return factors


def _prepare_regression_data(
    stock_returns: pd.DataFrame,
    end_date: pd.Timestamp | None,
    min_months: int,
    use_momentum: bool,
    factor_mean_window: str = 'long_run',
    beta_window: int | None = None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Build aligned monthly panel for FF (+ optional Carhart UMD) regressions."""
    factors = _factors_through_date(end_date, use_momentum)
    factor_cols = list(BASE_FACTOR_COLS)
    if use_momentum:
        factor_cols.append(UMD_COL)

    stock_returns_monthly = (1 + stock_returns).resample('ME').prod() - 1
    stock_returns_monthly.index = (
        stock_returns_monthly.index.to_period('M').to_timestamp()
    )

    data = stock_returns_monthly.join(factors, how='inner').dropna()

    if beta_window is not None and len(data) > beta_window:
        data = data.iloc[-beta_window:]

    if data.empty:
        raise ValueError(
            "No overlapping months between stock returns and Fama-French factors. "
            "Check your date range."
        )

    if len(data) < min_months:
        raise ValueError(
            f"Insufficient overlapping months for FF regression: "
            f"{len(data)} < {min_months} required."
        )

    return data, factor_cols, stock_returns_monthly


def _fit_stock_expected_return(
    data: pd.DataFrame,
    factor_cols: list[str],
    ticker: str,
    factors: pd.DataFrame,
    factor_mean_window: str,
    use_alpha: bool,
) -> float:
    y = data[ticker] - data['RF']
    X = sm.add_constant(data[factor_cols])
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

    factor_means, rf_mean = _factor_means_and_rf(
        factors, factor_cols, data, factor_mean_window
    )
    betas = model.params[factor_cols]
    if use_alpha:
        alpha = model.params['const']
        expected_excess = alpha + (betas * factor_means).sum()
    else:
        expected_excess = (betas * factor_means).sum()
    return float(expected_excess + rf_mean)


def estimate_factor_exposures(
    stock_returns: pd.DataFrame,
    end_date: pd.Timestamp | None = None,
    min_months: int = 24,
    use_momentum: bool = False,
    factor_mean_window: str = 'long_run',
    beta_window: int | None = None,
) -> pd.DataFrame:
    """Per-ticker OLS factor loadings (const + FF (+ UMD) betas)."""
    data, factor_cols, stock_returns_monthly = _prepare_regression_data(
        stock_returns, end_date, min_months, use_momentum,
        factor_mean_window, beta_window,
    )
    factors = _factors_through_date(end_date, use_momentum)

    rows = {}
    for ticker in stock_returns_monthly.columns:
        if ticker not in data.columns:
            continue
        y = data[ticker] - data['RF']
        X = sm.add_constant(data[factor_cols])
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        rows[ticker] = model.params

    return pd.DataFrame(rows).T


def estimate_expected_returns(
    stock_returns: pd.DataFrame,
    end_date: pd.Timestamp | None = None,
    min_months: int = 24,
    shrinkage: float = 0.50,
    use_momentum: bool = False,
    use_alpha: bool = False,
    factor_mean_window: str = 'long_run',
    beta_window: int | None = None,
) -> pd.Series:
    """
    Estimate expected monthly returns using the Fama-French 5-factor model.

    use_alpha=False (default): omit stock-specific alpha from the projection;
    individual alphas on 24–36 months are noisy. factor_mean_window controls
    E[factor]: 'long_run' (default), 'in_sample', or 'rolling_120m'.
    """
    data, factor_cols, stock_returns_monthly = _prepare_regression_data(
        stock_returns, end_date, min_months, use_momentum,
        factor_mean_window, beta_window,
    )
    factors = _factors_through_date(end_date, use_momentum)

    expected_returns = []
    for ticker in stock_returns_monthly.columns:
        if ticker not in data.columns:
            continue
        expected_returns.append(
            _fit_stock_expected_return(
                data, factor_cols, ticker, factors,
                factor_mean_window, use_alpha,
            )
        )

    result = pd.Series(
        expected_returns,
        index=[t for t in stock_returns_monthly.columns if t in data.columns],
    )

    if shrinkage > 0 and len(result) > 0:
        cs_mean = result.mean()
        result = shrinkage * cs_mean + (1 - shrinkage) * result

    logger.info(f"✓ Estimated monthly returns for {len(result)} stocks")
    return result


def attribute_oos_returns(
    portfolio_returns: pd.Series,
    weight_history: list,
    daily_returns: pd.DataFrame,
    use_momentum: bool = False,
) -> dict:
    """
    Decompose realized OOS portfolio return into factor contributions and residual alpha.
    """
    if not weight_history:
        return {}

    try:
        factors = _factors_through_date(None, use_momentum)
    except Exception as e:
        logger.warning(f"Attribution skipped: cannot load FF factors ({e}).")
        return {}

    factor_cols = list(BASE_FACTOR_COLS)
    if use_momentum:
        factor_cols.append(UMD_COL)

    total_factor = {f: 0.0 for f in factor_cols}
    total_alpha = 0.0
    total_rf = 0.0
    total_realized = 0.0
    period_rows = []

    for i, wh in enumerate(weight_history):
        train_end = wh['date']
        weights = wh['weights']
        exposures = wh.get('factor_exposures')
        if exposures is None:
            continue

        period_tickers = wh.get('period_tickers', list(weights.keys()))

        if i + 1 < len(weight_history):
            oos_end = weight_history[i + 1]['date']
            oos_mask = (daily_returns.index > train_end) & (daily_returns.index <= oos_end)
        else:
            oos_mask = daily_returns.index > train_end

        oos_daily = daily_returns.loc[oos_mask, period_tickers].fillna(0)
        if oos_daily.empty:
            continue

        w_vec = np.array([weights.get(t, 0.0) for t in period_tickers])
        period_realized = float((oos_daily @ w_vec).sum())
        total_realized += period_realized

        oos_months = oos_daily.index.to_period('M').to_timestamp().unique()

        period_factor = {f: 0.0 for f in factor_cols}
        period_alpha = 0.0
        period_rf = 0.0

        for month in oos_months:
            if month not in factors.index:
                continue
            f_row = factors.loc[month, factor_cols]
            rf = factors.loc[month, 'RF']
            days_in_month = oos_daily.index.to_period('M').to_timestamp() == month
            n_days = int(days_in_month.sum())
            if n_days == 0:
                continue
            scale = n_days / 21.0

            for ticker in period_tickers:
                w = weights.get(ticker, 0.0)
                if w < 1e-12 or ticker not in exposures.index:
                    continue
                exp = exposures.loc[ticker]
                alpha = exp.get('const', 0.0)
                period_alpha += w * alpha * scale
                period_rf += w * rf * scale
                for f in factor_cols:
                    beta = exp.get(f, 0.0)
                    contrib = w * beta * f_row[f] * scale
                    period_factor[f] += contrib
                    total_factor[f] += contrib

        total_alpha += period_alpha
        total_rf += period_rf
        period_residual = period_realized - period_alpha - period_rf - sum(period_factor.values())

        period_rows.append({
            'period_start': train_end,
            'realized': period_realized,
            'alpha': period_alpha,
            'rf': period_rf,
            **{f: period_factor[f] for f in factor_cols},
            'residual': period_residual,
        })

    total_explained = total_alpha + total_rf + sum(total_factor.values())
    total_residual = total_realized - total_explained

    return {
        'total_realized': total_realized,
        'total_alpha': total_alpha,
        'total_rf': total_rf,
        'factor_contributions': total_factor,
        'total_residual': total_residual,
        'periods': period_rows,
    }
