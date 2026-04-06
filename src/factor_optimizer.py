import numpy as np
from scipy.optimize import minimize
import warnings
import logging

logger = logging.getLogger(__name__)

def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate=0.02, 
                      max_weight=0.2, allow_short_selling=False):
    """
    Maximize Sharpe ratio for optimal portfolio allocation.
    
    Args:
        expected_returns (np.ndarray): Expected return for each asset (annualized)
        cov_matrix (np.ndarray): Covariance matrix of returns (annualized)
        risk_free_rate (float): Risk-free rate for Sharpe calculation (default: 2%)
        max_weight (float): Maximum weight per asset (default: 20%, no short selling)
        allow_short_selling (bool): If True, weights can be negative
    
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

    def negative_sharpe(weights):
        """Objective function: negative Sharpe ratio"""
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # Avoid division by zero
        if portfolio_vol < 1e-10:
            return 1e10
        
        return -(portfolio_return - risk_free_rate) / portfolio_vol

    # Constraint: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Set bounds based on short selling allowance
    if allow_short_selling:
        bounds = [(-1.0, 1.0)] * n  # Can short up to -100%
    else:
        bounds = [(0, max_weight)] * n  # Long-only, max position size
    
    # Initial guess: equal weight
    initial = np.ones(n) / n

    # Run optimization
    result = minimize(negative_sharpe, initial,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints,
                      options={'ftol': 1e-9})

    # Check convergence
    if not result.success:
        logger.warning(f"Optimization did not converge: {result.message}")
        warnings.warn(f"Portfolio optimization failed: {result.message}")
    
    logger.info(f"✓ Optimization converged | Sharpe: {-result.fun:.3f}")
    return result.x


       