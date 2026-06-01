"""Statistical helpers for strategy evaluation."""

from scipy.stats import norm, skew, kurtosis  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

EULER_MASCHERONI = 0.5772156649


def deflated_sharpe(daily_returns, n_trials, annualization=252):
    """
    DSR per Bailey & López de Prado (2014).
    Returns (DSR probability, expected max Sharpe under null).
    """
    r = np.asarray(daily_returns, dtype=float)
    if r.std() < 1e-12 or len(r) < 30:
        return float("nan"), float("nan")

    sr_obs = (r.mean() / r.std()) * np.sqrt(annualization)
    n = len(r)
    skw = skew(r)
    krt = kurtosis(r, fisher=False)

    sr_var = (1 - skw * sr_obs + (krt - 1) / 4 * sr_obs ** 2) / max(n - 1, 1)
    sr_sd = np.sqrt(max(sr_var, 1e-12))

    z1 = norm.ppf(1 - 1.0 / max(n_trials, 2))
    z2 = norm.ppf(1 - 1.0 / (max(n_trials, 2) * np.e))
    sr_star = sr_sd * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)

    dsr = norm.cdf((sr_obs - sr_star) / sr_sd)
    return float(dsr), float(sr_star)
