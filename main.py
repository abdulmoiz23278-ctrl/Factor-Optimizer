import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from src.factor_optimizer import optimize_portfolio
from src.data_fetcher import fetch_stock_returns
from src.factor_model import estimate_expected_returns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_efficient_frontier(returns, cov_matrix, optimal_return, optimal_vol, 
                           optimal_weights=None, num_portfolios=1000):
    """
    Plot efficient frontier with random portfolios and optimal portfolio.
    
    Args:
        returns (np.ndarray): Expected returns
        cov_matrix (np.ndarray): Covariance matrix (annualized)
        optimal_return (float): Optimal portfolio expected return
        optimal_vol (float): Optimal portfolio volatility
        optimal_weights (np.ndarray): Optimal weights (for display)
        num_portfolios (int): Number of random portfolios to generate
    """
    results = []

    for _ in range(num_portfolios):
        # Generate random weights (normalized)
        weights = np.random.random(len(returns))
        weights /= np.sum(weights)

        # Calculate return and volatility
        portfolio_return = weights @ returns
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

        results.append((portfolio_vol, portfolio_return))

    results = np.array(results)
    
    # Calculate Sharpe ratio for coloring (using 2% risk-free rate)
    sharpe_ratios = results[:, 1] / results[:, 0]

    # Create plot
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(results[:, 0], results[:, 1], 
                         c=sharpe_ratios, cmap='viridis', 
                         alpha=0.5, s=20, label='Random Portfolios')
    
    # Mark optimal portfolio
    plt.scatter(optimal_vol, optimal_return,
                color='red', s=300, marker='*', 
                label='Optimal Portfolio (Max Sharpe)', zorder=5)
    
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.xlabel('Volatility (Annualized)', fontsize=12)
    plt.ylabel('Expected Return (Annualized)', fontsize=12)
    plt.title('Efficient Frontier', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def backtest_portfolio(returns, weights):
    """
    Calculate cumulative portfolio returns.
    
    Args:
        returns (pd.DataFrame): Daily returns
        weights (np.ndarray): Portfolio weights
    
    Returns:
        pd.Series: Cumulative portfolio value
    """
    # Calculate daily portfolio returns
    portfolio_returns = returns @ weights
    
    # Calculate cumulative returns: (1 + r1) * (1 + r2) * ...
    cumulative = (1 + portfolio_returns).cumprod()
    
    return cumulative

def main():
    print("="*80)
    print("MULTI-FACTOR PORTFOLIO OPTIMIZER")
    print("="*80)

    # Step 1: Download data
    print("\n[Step 1] Downloading data...")
    
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", 
               "JPM", "V", "UNH", "HD", "PG", "DIS", "MA", "PYPL", "INTC"]
    
    try:
        returns = fetch_stock_returns(tickers, start="2015-01-01", end="2023-01-01")
        print(f"  ✓ Downloaded {len(tickers)} stocks, {len(returns)} trading days")
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return

    # Step 2: Estimate expected returns using Fama-French model
    print("\n[Step 2] Estimating expected returns (Fama-French 5-factor)...")
    
    try:
        expected_returns_monthly = estimate_expected_returns(returns).values
    except Exception as e:
        logger.error(f"Failed to estimate returns: {e}")
        return
    
    # ⚠️ FIX: Convert monthly returns to annualized
    # estimate_expected_returns() returns MONTHLY returns
    expected_returns = expected_returns_monthly * 12
    
    print(f"  ✓ Expected annual returns: {expected_returns.mean():.2%} (mean)")

    # Step 3: Calculate covariance matrix
    print("\n[Step 3] Calculating covariance matrix...")
    
    # ⚠️ FIX: Convert daily to annualized covariance
    # Daily returns → multiply by 252 (trading days per year)
    cov_matrix = returns.cov().values * 252
    
    print(f"  ✓ Covariance matrix: {cov_matrix.shape}")

    # Step 4: Optimize portfolio
    print("\n[Step 4] Optimizing portfolio weights...")
    
    try:
        weights = optimize_portfolio(expected_returns, cov_matrix, 
                                    risk_free_rate=0.02, max_weight=0.2)
        print(f"  ✓ Optimization successful")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return

    # Step 5: Backtest portfolio
    print("\n[Step 5] Backtesting portfolio...")
    
    portfolio_cumulative = backtest_portfolio(returns, weights)
    
    # Download SPY benchmark
    spy = yf.download("SPY", start="2015-01-01", end="2023-01-01", progress=False)["Close"]
    spy_returns = spy.pct_change().dropna()
    spy_cumulative = (1 + spy_returns).cumprod()

    # Align indices (only use overlapping dates)
    common_index = portfolio_cumulative.index.intersection(spy_cumulative.index)
    portfolio_cumulative = portfolio_cumulative.loc[common_index]
    spy_cumulative = spy_cumulative.loc[common_index]
    
    print(f"  ✓ Backtest period: {len(portfolio_cumulative)} trading days")

    # Step 6: Calculate final statistics
    print("\n[Step 6] Calculating portfolio statistics...")
    
    portfolio_return = weights @ expected_returns
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    risk_free_rate = 0.02
    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

    # Display results
    print("\n" + "="*80)
    print("MAXIMUM SHARPE RATIO PORTFOLIO")
    print("="*80)
    print("\nAllocations:")
    print("-" * 50)
    
    for ticker, weight in zip(tickers, weights):
        if weight > 0.001:  # Only show positions > 0.1%
            print(f"  {ticker:6s}: {weight:7.2%}")
    
    print("-" * 50)
    print(f"\nMetrics:")
    print(f"  Expected Annual Return: {portfolio_return:7.2%}")
    print(f"  Annual Volatility:      {portfolio_vol:7.2%}")
    print(f"  Sharpe Ratio:           {sharpe:7.3f}")
    print(f"  Risk-Free Rate:         {risk_free_rate:7.2%}")
    print("="*80)

    # Calculate benchmark performance
    spy_return = float((spy_cumulative.iloc[-1] - spy_cumulative.iloc[0]) / spy_cumulative.iloc[0])
    portfolio_final_return = float((portfolio_cumulative.iloc[-1] - portfolio_cumulative.iloc[0]) / portfolio_cumulative.iloc[0])
    
    print(f"\nBacktest Performance (2015-2023):")
    print(f"  Optimized Portfolio: {portfolio_final_return:7.2%}")
    print(f"  S&P 500 (SPY):       {spy_return:7.2%}")
    print(f"  Outperformance:      {portfolio_final_return - spy_return:7.2%}")
    print("="*80)

    # Generate visualizations
    print("\n[Step 7] Generating plots...")
    
    try:
        # Plot 1: Efficient Frontier
        plot_efficient_frontier(expected_returns, cov_matrix, 
                               portfolio_return, portfolio_vol, 
                               weights, num_portfolios=1000)
        plt.savefig('efficient_frontier.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: efficient_frontier.png")
        
        # Plot 2: Cumulative returns
        plt.figure(figsize=(12, 7))
        plt.plot(portfolio_cumulative, label="Optimized Portfolio", linewidth=2)
        plt.plot(spy_cumulative, label="S&P 500 (SPY)", linewidth=2)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.title('Portfolio vs Benchmark Performance', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('backtest_performance.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: backtest_performance.png")
        
        # Plot 3: Allocation pie chart
        plt.figure(figsize=(10, 8))
        # Only show allocations > 0.1%
        allocation_dict = {t: w for t, w in zip(tickers, weights) if w > 0.001}
        colors = plt.cm.Set3(np.linspace(0, 1, len(allocation_dict)))
        plt.pie(allocation_dict.values(), labels=allocation_dict.keys(), 
               autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Portfolio Allocation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('portfolio_allocation.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: portfolio_allocation.png")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")

    print("\n✓ Optimization complete!")
    print("="*80)

if __name__ == "__main__":
    main()