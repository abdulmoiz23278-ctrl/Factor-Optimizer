import numpy as np
from src.factor_optimizer import optimize_portfolio

def main():
    print("Portfolio Optimizer Running")

    returns = np.random.randn(100, 5)
    expected_returns = returns.mean(axis=0)
    cov_matrix = np.cov(returns.T)

    weights = optimize_portfolio(expected_returns, cov_matrix)

    print("Weights:", weights)

if __name__ == "__main__":
    main()