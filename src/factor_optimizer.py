import logging

import numpy as np  # type: ignore[import-untyped]
from scipy.optimize import linprog, minimize  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def optimize_cvar_lp(
    returns_matrix,
    expected_returns=None,
    max_weight=0.2,
    alpha=0.05,
    target_return=None,
    prev_weights=None,
    max_turnover=None,
):
    """
    Min-CVaR portfolio via the Rockafellar-Uryasev LP, solved with scipy/HiGHS.

    Variables: [w (n), zeta (1), u (T)]

        min  zeta + (1/(alpha*T)) * sum(u)
        s.t. u_t >= -w'r_t - zeta   for all t
             u_t >= 0
             sum(w) = 1
             0 <= w_i <= max_weight
             (optional) w' @ expected_returns >= target_return

    Turnover (optional): auxiliary s_i with |w_i - prev_i| <= s_i and sum(s) <= 2*max_turnover.
    """
    R = np.asarray(returns_matrix, dtype=float)
    T, n = R.shape

    nvar = n + 1 + T

    c = np.zeros(nvar)
    c[n] = 1.0
    c[n + 1:] = 1.0 / (alpha * T)

    A_ub = np.zeros((T, nvar))
    A_ub[:, :n] = -R
    A_ub[:, n] = -1.0
    A_ub[np.arange(T), n + 1 + np.arange(T)] = -1.0
    b_ub = np.zeros(T)

    if target_return is not None and expected_returns is not None:
        mu = np.asarray(expected_returns, dtype=float)
        row = np.zeros((1, nvar))
        row[0, :n] = -mu
        A_ub = np.vstack([A_ub, row])
        b_ub = np.append(b_ub, -target_return)

    A_eq = np.zeros((1, nvar))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, max_weight)] * n + [(None, None)] + [(0.0, None)] * T

    if prev_weights is not None and max_turnover is not None:
        prev = np.asarray(prev_weights, dtype=float)
        c = np.concatenate([c, np.zeros(n)])
        bounds = bounds + [(0.0, None)] * n

        A_ub = np.hstack([A_ub, np.zeros((A_ub.shape[0], n))])
        A_eq = np.hstack([A_eq, np.zeros((A_eq.shape[0], n))])

        ncol = nvar + n
        block1 = np.zeros((n, ncol))
        for i in range(n):
            block1[i, i] = 1.0
            block1[i, nvar + i] = -1.0
        A_ub = np.vstack([A_ub, block1])
        b_ub = np.concatenate([b_ub, prev])

        block2 = np.zeros((n, ncol))
        for i in range(n):
            block2[i, i] = -1.0
            block2[i, nvar + i] = -1.0
        A_ub = np.vstack([A_ub, block2])
        b_ub = np.concatenate([b_ub, -prev])

        row = np.zeros((1, ncol))
        row[0, nvar:] = 1.0
        A_ub = np.vstack([A_ub, row])
        b_ub = np.append(b_ub, 2.0 * max_turnover)

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs',
    )

    if not result.success:
        raise ValueError(f"CVaR LP failed: {result.message}")

    w = result.x[:n]
    if not np.all(np.isfinite(w)):
        raise ValueError("CVaR LP returned non-finite weights.")

    logger.info(f"✓ CVaR LP converged | obj={result.fun:.5f}")
    return w


def optimize_portfolio(
    expected_returns,
    cov_matrix,
    risk_free_rate=0.02,
    max_weight=0.2,
    allow_short_selling=False,
    optimization_objective='sharpe',
    prev_weights=None,
    max_turnover=None,
    returns_matrix=None,
    cvar_alpha=0.05,
    cvar_target_return=None,
):
    """
    Maximize Sharpe ratio or minimize CVaR for optimal portfolio allocation.

    Returns:
        (weights, objective_used): objective_used is 'cvar', 'sharpe', or 'sharpe_fallback'.
    """
    n = len(expected_returns)
    expected_returns = np.asarray(expected_returns, dtype=float)
    prev_arr = None
    if prev_weights is not None:
        prev_arr = np.asarray(prev_weights, dtype=float)
        if prev_arr.shape != (n,):
            raise ValueError("prev_weights must have length n_assets.")

    use_cvar_lp = (
        optimization_objective == 'cvar'
        and returns_matrix is not None
        and returns_matrix.shape[1] == n
        and len(returns_matrix) >= 20
        and not allow_short_selling
    )

    if optimization_objective == 'cvar' and not use_cvar_lp:
        logger.warning(
            "CVaR requested but returns_matrix unavailable or invalid; "
            "falling back to Sharpe."
        )

    if use_cvar_lp:
        try:
            weights = optimize_cvar_lp(
                returns_matrix,
                expected_returns=expected_returns,
                max_weight=max_weight,
                alpha=cvar_alpha,
                target_return=cvar_target_return,
                prev_weights=prev_arr,
                max_turnover=max_turnover,
            )
            return weights, "cvar"
        except Exception as e:
            logger.warning(f"CVaR LP failed ({e}); falling back to Sharpe.")

    used_sharpe_fallback = optimization_objective == "cvar"

    def negative_sharpe(weights):
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        if portfolio_vol < 1e-10:
            return 1e10
        return -(portfolio_return - risk_free_rate) / portfolio_vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    if prev_arr is not None and max_turnover is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, pw=prev_arr, mt=max_turnover: mt - 0.5 * np.sum(np.abs(w - pw)),
        })

    if allow_short_selling:
        bounds = [(-1.0, 1.0)] * n
    else:
        bounds = [(0, max_weight)] * n

    if prev_arr is not None and np.all(prev_arr >= 0) and np.isclose(prev_arr.sum(), 1.0):
        initial = prev_arr.copy()
    else:
        initial = np.ones(n) / n

    result = minimize(
        negative_sharpe,
        initial,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9},
    )

    if not result.success:
        logger.warning(f"Optimization did not converge: {result.message}")
        raise ValueError(f"Portfolio optimization failed: {result.message}")

    if not np.all(np.isfinite(result.x)):
        raise ValueError("Portfolio optimization returned non-finite weights.")

    logger.info(f"✓ Optimization converged | Sharpe: {-result.fun:.3f}")
    objective_used = "sharpe_fallback" if used_sharpe_fallback else "sharpe"
    return result.x, objective_used
