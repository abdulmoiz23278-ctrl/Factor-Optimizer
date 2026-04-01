import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(expected_returns, cov_matrix):
    n = len(expected_returns)

    def volatility(w):
        return np.sqrt(w.T @ cov_matrix @ w)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0,1)] * n
    initial = np.ones(n) / n

    result = minimize(volatility, initial,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    return result.x