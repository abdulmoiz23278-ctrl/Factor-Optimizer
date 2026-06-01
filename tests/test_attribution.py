"""Attribution scaling sanity checks."""

import numpy as np
import pandas as pd

from src.factor_model import (
    BASE_FACTOR_COLS,
    attribute_oos_returns,
    get_fama_french_factors,
)


def test_attribution_balance_synthetic():
    """Realized OOS return should match FF decomposition within 10% (scaling fix)."""
    ff = get_fama_french_factors()
    month = pd.Timestamp("2020-03-01")
    if month not in ff.index:
        month = ff.index[ff.index <= month][-1]

    f_row = ff.loc[month, BASE_FACTOR_COLS]
    rf = float(ff.loc[month, "RF"])

    tickers = ["AAA", "BBB"]
    exposures = pd.DataFrame(
        {
            "const": [0.0, 0.0],
            "Mkt-RF": [1.0, 0.5],
            "SMB": [0.2, -0.1],
            "HML": [0.0, 0.0],
            "RMW": [0.0, 0.0],
            "CMA": [0.0, 0.0],
        },
        index=tickers,
    )

    n_days = 15
    dates = pd.bdate_range("2020-03-02", periods=n_days)
    weights = {t: 0.5 for t in tickers}

    scale = n_days / 21.0
    daily = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in tickers:
        exp = exposures.loc[t]
        monthly = float(exp["const"] + rf + (exp[BASE_FACTOR_COLS] * f_row).sum())
        # Partial-month OOS: only scale days/21 of the monthly factor-model return
        daily[t] = monthly * scale / n_days

    train_end = dates[0] - pd.Timedelta(days=1)
    port = (daily.fillna(0) @ np.array([0.5, 0.5])).rename("port")
    wh = [
        {
            "date": train_end,
            "weights": weights,
            "period_tickers": tickers,
            "factor_exposures": exposures,
        },
    ]

    attr = attribute_oos_returns(port, wh, daily, use_momentum=False)
    assert attr

    explained = (
        attr["total_alpha"]
        + attr["total_rf"]
        + sum(attr["factor_contributions"].values())
    )
    residual_pct = abs(attr["total_residual"]) / max(abs(attr["total_realized"]), 1e-8)
    assert residual_pct < 0.10, (
        f"residual {residual_pct:.1%} of realized; "
        f"realized={attr['total_realized']:.6f} explained={explained:.6f}"
    )
