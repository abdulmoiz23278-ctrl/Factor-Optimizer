"""Statistical helpers for strategy evaluation."""

from scipy.stats import norm, skew, kurtosis  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

EULER_MASCHERONI = 0.5772156649


def deflated_sharpe(daily_returns, n_trials, sr_trials=None):
    """
    DSR per Bailey & López de Prado (2014).

    All Sharpe quantities are per-period (daily), not annualized. N counts
    daily observations; SR = mean(r) / std(r, ddof=1).

    Args:
        daily_returns: Daily return series.
        n_trials: Number of strategy variants tested (deflation factor).
        sr_trials: Optional array of per-period Sharpes from those trials.
            When omitted, Var(SR) is estimated from the return moments of
            the observed series (single-path Bailey estimator).

    Returns:
        (DSR probability, expected max Sharpe threshold SR0 in daily units).
    """
    r = np.asarray(daily_returns, dtype=float)
    n = len(r)
    if n < 30 or r.std(ddof=1) < 1e-12:
        return float("nan"), float("nan")

    sr = r.mean() / r.std(ddof=1)
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)

    den = np.sqrt(max(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2, 1e-12))

    if sr_trials is not None:
        trials = np.asarray(sr_trials, dtype=float)
        if len(trials) < 2:
            return float("nan"), float("nan")
        var_sr = float(np.var(trials, ddof=1))
    else:
        var_sr = (den**2) / max(n - 1, 1)

    if not np.isfinite(var_sr) or var_sr < 1e-18:
        return float("nan"), float("nan")

    nt = max(int(n_trials), 2)
    z1 = norm.ppf(1 - 1.0 / nt)
    z2 = norm.ppf(1 - 1.0 / (nt * np.e))
    sr0 = np.sqrt(var_sr) * (
        (1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    )

    num = (sr - sr0) * np.sqrt(n - 1)
    dsr = float(norm.cdf(num / den))
    return dsr, float(sr0)
