import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate=0.01):
    n = len(expected_returns)

    def negative_sharpe(weights):
        portfolio_return = weights @ expected_returns
        portfolio_vol = (weights.T @ cov_matrix @ weights) ** 0.5
        return -(portfolio_return - risk_free_rate) / portfolio_vol

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 0.2)] * n
    initial = np.ones(n) / n

    result = minimize(negative_sharpe, initial,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    return result.x


       