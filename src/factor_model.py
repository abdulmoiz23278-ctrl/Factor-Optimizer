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
BASE_FACTOR_COLS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']


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


def _compute_cross_sectional_momentum_factor(stock_returns_monthly: pd.DataFrame) -> pd.Series:
    """
    12-1 cross-sectional momentum factor: long top-half / short bottom-half by past 12-1 return.
    """
    mom_returns = []
    mom_index = []
    idx = stock_returns_monthly.index

    for i in range(len(idx)):
        if i < 13:
            continue
        past = stock_returns_monthly.iloc[i - 12:i - 1]
        scores = (1 + past).prod() - 1
        if scores.isna().all():
            continue
        median = scores.median()
        high = scores >= median
        low = scores < median
        r_t = stock_returns_monthly.iloc[i]
        if high.sum() == 0 or low.sum() == 0:
            continue
        mom_ret = r_t[high].mean() - r_t[low].mean()
        mom_returns.append(mom_ret)
        mom_index.append(idx[i])

    if not mom_returns:
        return pd.Series(dtype=float)

    return pd.Series(mom_returns, index=mom_index, name='MOM')


def _prepare_regression_data(
    stock_returns: pd.DataFrame,
    end_date: pd.Timestamp | None,
    min_months: int,
    use_momentum: bool,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Build aligned monthly panel for FF (+ optional MOM) regressions."""
    factors = get_fama_french_factors()
    factor_cols = list(BASE_FACTOR_COLS)

    if end_date is not None:
        cutoff = pd.Timestamp(end_date).to_period('M').to_timestamp()
        factors = factors[factors.index <= cutoff]

    stock_returns_monthly = (1 + stock_returns).resample('ME').prod() - 1
    stock_returns_monthly.index = (
        stock_returns_monthly.index.to_period('M').to_timestamp()
    )

    if use_momentum:
        mom = _compute_cross_sectional_momentum_factor(stock_returns_monthly)
        if mom.empty:
            logger.warning(
                "Momentum factor unavailable; continuing with FF 5 factors only."
            )
        else:
            factors = factors.join(mom, how='left')
            factor_cols.append('MOM')

    data = stock_returns_monthly.join(factors, how='inner').dropna()

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


def estimate_factor_exposures(
    stock_returns: pd.DataFrame,
    end_date: pd.Timestamp | None = None,
    min_months: int = 24,
    use_momentum: bool = False,
) -> pd.DataFrame:
    """
    Per-ticker OLS factor loadings (const + FF (+ MOM) betas).

    Returns:
        DataFrame indexed by ticker; columns: const, factor names.
    """
    data, factor_cols, stock_returns_monthly = _prepare_regression_data(
        stock_returns, end_date, min_months, use_momentum
    )
    rows = {}
    for ticker in stock_returns_monthly.columns:
        y = data[ticker] - data['RF']
        X = sm.add_constant(data[factor_cols])
        model = sm.OLS(y, X).fit()
        rows[ticker] = model.params

    return pd.DataFrame(rows).T


def estimate_expected_returns(
    stock_returns: pd.DataFrame,
    end_date: pd.Timestamp | None = None,
    min_months: int = 24,
    shrinkage: float = 0.50,
    use_momentum: bool = False,
) -> pd.Series:
    """
    Estimate expected monthly returns using the Fama-French 5-factor model.

    Methodology
    -----------
    1. Compound daily returns to monthly:
           R_month = Π(1 + R_day) - 1
    2. Align with FF factors (inner join on month).
    3. OLS regression per stock (excess returns on excess market + 4 style factors):
           R_i - RF = α + β₁(Mkt-RF) + β₂SMB + β₃HML + β₄RMW + β₅CMA + ε
    4. Expected monthly return = α + β @ E[factors] + E[RF]
       — NOT model.predict(X).mean(), which is just y.mean() (OLS identity).

    Args:
        stock_returns: Daily returns DataFrame (one column per ticker).
        end_date:      Last allowable month for factor data (walk-forward cutoff).
        min_months:    Minimum overlapping months required for regression.
        shrinkage:     Cross-sectional shrinkage toward the mean (0–1).
        use_momentum:  Include 12-1 cross-sectional momentum as 6th factor.

    Returns:
        pd.Series: Monthly expected returns per ticker.
                   Multiply by 12 to annualize.

    Raises:
        ValueError: If stock and factor data share no overlapping months, or if
                    overlap is below min_months.
    """
    data, factor_cols, stock_returns_monthly = _prepare_regression_data(
        stock_returns, end_date, min_months, use_momentum
    )

    expected_returns = []
    factor_means = data[factor_cols].mean()
    rf_mean = data['RF'].mean()

    for ticker in stock_returns_monthly.columns:
        y = data[ticker] - data['RF']
        X = sm.add_constant(data[factor_cols])
        model = sm.OLS(y, X).fit()

        alpha = model.params['const']
        betas = model.params[factor_cols]
        expected_excess = alpha + (betas * factor_means).sum()
        expected_return = expected_excess + rf_mean

        expected_returns.append(expected_return)

        logger.debug(
            f"{ticker}: α={alpha:.5f} (t={model.tvalues['const']:.2f}), "
            f"R²={model.rsquared:.3f}, adj-R²={model.rsquared_adj:.3f}"
        )

    result = pd.Series(expected_returns, index=stock_returns_monthly.columns)

    if shrinkage > 0:
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

    Uses in-sample betas stored at each rebalance and realized factor returns over OOS months.
    """
    if not weight_history:
        return {}

    try:
        factors = get_fama_french_factors()
    except Exception as e:
        logger.warning(f"Attribution skipped: cannot load FF factors ({e}).")
        return {}

    factor_cols = list(BASE_FACTOR_COLS)
    stock_monthly = (1 + daily_returns).resample('ME').prod() - 1
    stock_monthly.index = stock_monthly.index.to_period('M').to_timestamp()

    if use_momentum:
        mom = _compute_cross_sectional_momentum_factor(stock_monthly)
        if not mom.empty:
            factors = factors.join(mom, how='left')
            factor_cols.append('MOM')

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

        if i + 1 < len(weight_history):
            oos_end = weight_history[i + 1]['date']
            oos_mask = (daily_returns.index > train_end) & (daily_returns.index <= oos_end)
        else:
            oos_mask = daily_returns.index > train_end

        oos_daily = daily_returns.loc[oos_mask]
        if oos_daily.empty:
            continue

        w_vec = np.array([weights.get(t, 0.0) for t in daily_returns.columns])
        period_realized = float((oos_daily @ w_vec).sum())
        total_realized += period_realized

        oos_months = (
            stock_monthly.loc[oos_mask]
            .index.to_period('M')
            .to_timestamp()
            .unique()
        )

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

            for ticker in daily_returns.columns:
                w = weights.get(ticker, 0.0)
                if w < 1e-12 or ticker not in exposures.index:
                    continue
                exp = exposures.loc[ticker]
                alpha = exp.get('const', 0.0)
                period_alpha += w * alpha * (n_days / 21.0)
                period_rf += w * rf * n_days
                for f in factor_cols:
                    beta = exp.get(f, 0.0)
                    contrib = w * beta * f_row[f] * n_days
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
