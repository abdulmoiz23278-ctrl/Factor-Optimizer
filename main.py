import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from src.factor_optimizer import optimize_portfolio
from src.data_fetcher import fetch_stock_returns
from src.factor_model import estimate_expected_returns

def plot_efficient_frontier(returns, cov_matrix, optimal_return, optimal_vol, num_portfolios=1000):
    results = []
    weights_record = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(returns))
        weights /= np.sum(weights)

        portfolio_return = weights @ returns
        portfolio_vol = (weights.T @ cov_matrix @ weights) ** 0.5

        results.append((portfolio_vol, portfolio_return))
        weights_record.append(weights)

    results = np.array(results)

    plt.figure(figsize=(10,6))
    plt.scatter(results[:,0], results[:,1], c=results[:,1]/results[:,0], cmap='viridis')
    plt.scatter(portfolio_vol, portfolio_return,
            color='red', s=100, label='Optimal Portfolio')
    plt.legend()
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    

def backtest_portfolio(returns, weights):
    portfolio_returns = returns @ weights
    cumulative = (1 + portfolio_returns).cumprod()
    return cumulative

def main():
    print("="*80)
    print("MULTI-FACTOR PORTFOLIO OPTIMIZER")
    print("="*80)

    # Step 1: Download data
    print("\n[Step 1] Downloading data...")

    tickers = ["AAPL","MSFT","GOOG","AMZN","META","NVDA","JPM","V","UNH","HD","PG","DIS","MA","PYPL","INTC"]
    returns = fetch_stock_returns(tickers)

    print(f"  ✓ Downloaded returns for {len(tickers)} stocks")

    # Compute stats
    expected_returns = estimate_expected_returns(returns).values * 252
    cov_matrix = returns.cov().values * 252

    # Step 5: Optimize
    print("\n[Step 5] Optimizing portfolio weights...")

    weights = optimize_portfolio(expected_returns, cov_matrix)
    portfolio_cumulative = backtest_portfolio(returns, weights)

    spy = yf.download("SPY", start="2015-01-01")["Close"]
    spy_returns = spy.pct_change().dropna()
    spy_cumulative = (1 + spy_returns).cumprod()

    # 🔥 FIX alignment (IMPORTANT)
    common_index = portfolio_cumulative.index.intersection(spy_cumulative.index)

    portfolio_cumulative = portfolio_cumulative.loc[common_index]
    spy_cumulative = spy_cumulative.loc[common_index]

    print("Backtest lengths:", len(portfolio_cumulative), len(spy_cumulative))
    portfolio_return = weights @ expected_returns
    portfolio_vol = (weights.T @ cov_matrix @ weights) ** 0.5
    risk_free_rate = 0.01
    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

    print("\n  Maximum Sharpe Ratio Portfolio:")
    print("-"*50)
    for i, w in enumerate(weights):
       print(f"{tickers[i]}: {w:.2%}")
    print("-"*50)
    print(f"    Expected Return: {portfolio_return:.2%}")
    print(f"    Volatility: {portfolio_vol:.2%}")
    print(f"    Sharpe Ratio: {sharpe:.3f}")

    print("\n✓ Optimization complete!")
    print("="*80)

    plot_efficient_frontier(expected_returns, cov_matrix, portfolio_return, portfolio_vol)
    
    plt.figure(figsize=(10,6))
    plt.plot(portfolio_cumulative, label="Optimized Portfolio")
    plt.plot(spy_cumulative, label="S&P 500 (SPY)")
    plt.title("Portfolio vs Benchmark")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()