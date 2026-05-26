import numpy as np  # type: ignore[import-untyped]
from scipy.optimize import minimize  # type: ignore[import-untyped]
import logging

logger = logging.getLogger(__name__)


def _cvar_loss(weights: np.ndarray, returns_matrix: np.ndarray, alpha: float = 0.05) -> float:
    """Mean return of the worst alpha-fraction of scenarios (minimize for CVaR of losses)."""
    port = returns_matrix @ weights
    q = np.quantile(port, alpha)
    tail = port[port <= q]
    if len(tail) == 0:
        return 1e10
    return -tail.mean()


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
):
    """
    Maximize Sharpe ratio for optimal portfolio allocation.

    Args:
        expected_returns (np.ndarray): Expected return for each asset (annualized)
        cov_matrix (np.ndarray): Covariance matrix of returns (annualized)
        risk_free_rate (float): Risk-free rate for Sharpe calculation (default: 2%)
        max_weight (float): Maximum weight per asset (default: 20%, no short selling)
        allow_short_selling (bool): If True, weights can be negative
        optimization_objective (str): 'sharpe' (default) or 'cvar'
        prev_weights (np.ndarray): Previous portfolio weights for turnover constraint
        max_turnover (float): Max one-way turnover 0.5*sum(|dw|); active only with prev_weights
        returns_matrix (np.ndarray): T x n scenario returns for CVaR (in-sample daily)
        cvar_alpha (float): Tail probability for CVaR (default 5%)

    Returns:
        np.ndarray: Optimal portfolio weights

    Raises:
        ValueError: If optimization fails to converge

    Notes:
        - Minimizes negative Sharpe ratio (equivalent to maximizing Sharpe)
        - Constraint: sum(weights) = 1
        - Default bounds: [0, 0.2] (no short selling, max 20% per position)
    """
    n = len(expected_returns)
    expected_returns = np.asarray(expected_returns, dtype=float)
    prev_arr = None
    if prev_weights is not None:
        prev_arr = np.asarray(prev_weights, dtype=float)
        if prev_arr.shape != (n,):
            raise ValueError("prev_weights must have length n_assets.")

    use_cvar = (
        optimization_objective == 'cvar'
        and returns_matrix is not None
        and returns_matrix.shape[1] == n
        and len(returns_matrix) >= 20
    )
    if optimization_objective == 'cvar' and not use_cvar:
        logger.warning(
            "CVaR objective unavailable (missing returns_matrix); falling back to Sharpe."
        )

    def negative_sharpe(weights):
        """Objective function: negative Sharpe ratio"""
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

        if portfolio_vol < 1e-10:
            return 1e10

        return -(portfolio_return - risk_free_rate) / portfolio_vol

    if use_cvar:
        def objective(weights):
            return _cvar_loss(weights, returns_matrix, cvar_alpha)
    else:
        objective = negative_sharpe

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
        objective,
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

    if use_cvar:
        logger.info(f"✓ Optimization converged | CVaR obj: {result.fun:.5f}")
    else:
        logger.info(f"✓ Optimization converged | Sharpe: {-result.fun:.3f}")
    return result.x
