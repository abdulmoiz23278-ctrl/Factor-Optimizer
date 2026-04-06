import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def fetch_stock_returns(tickers, start="2015-01-01", end="2023-01-01"):
    """
    Fetch daily stock returns for given tickers.
    
    Args:
        tickers (list or str): Stock ticker(s)
        start (str): Start date (YYYY-MM-DD)
        end (str): End date (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: Daily returns (percent change)
    
    Raises:
        ValueError: If download fails or no data retrieved
    """
    try:
        # Suppress yfinance progress bar
        data = yf.download(tickers, start=start, end=end, progress=False)
        
        # Validate data was retrieved
        if data.empty:
            raise ValueError(f"No data retrieved for tickers: {tickers}")
        
        # Extract close prices
        data = data["Close"]
        
        # Handle single ticker (returns Series instead of DataFrame)
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # Calculate returns and remove NaN
        returns = data.pct_change().dropna()
        
        logger.info(f"✓ Downloaded {len(tickers)} tickers, {len(returns)} trading days")
        return returns
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise ValueError(f"Data fetch failed for {tickers}: {str(e)}")