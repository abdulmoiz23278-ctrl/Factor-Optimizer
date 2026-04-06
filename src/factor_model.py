import pandas as pd
import statsmodels.api as sm
import pandas_datareader.data as web
import logging

logger = logging.getLogger(__name__)

def get_fama_french_factors():
    """
    Fetch Fama-French 5 factors from Ken French's data library.
    
    Returns:
        pd.DataFrame: Monthly factor returns (as decimals, not percentages)
    """
    try:
        ff = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench')[0]
        ff = ff / 100  # Convert from percentages to decimals
        
        # Convert index to datetime (handles YearMonth format)
        ff.index = pd.to_datetime(ff.index.astype(str))
        
        logger.info(f"✓ Loaded Fama-French factors: {ff.shape[0]} months")
        return ff
    except Exception as e:
        logger.error(f"Failed to load Fama-French factors: {e}")
        raise

def estimate_expected_returns(stock_returns):
    """
    Estimate expected returns using Fama-French 5-factor model.
    
    Args:
        stock_returns (pd.DataFrame): Daily stock returns
    
    Returns:
        pd.Series: MONTHLY expected returns by stock
        
    Note:
        - Input: daily returns
        - Output: monthly returns (multiply by 12 to annualize)
        - Uses OLS regression: Stock_Return = α + β(Mkt-RF) + ...
    """
    factors = get_fama_french_factors()

    # Convert daily returns to monthly (end of month average)
    stock_returns_monthly = stock_returns.resample('M').mean()

    # Align stock index to monthly periods
    stock_returns_monthly.index = stock_returns_monthly.index.to_period('M').to_timestamp()

    # Align both datasets (inner join = only overlapping dates)
    data = stock_returns_monthly.join(factors, how='inner')

    # Remove any rows with missing values
    data = data.dropna()
    
    if data.empty:
        raise ValueError("No overlapping data between stock returns and Fama-French factors")

    expected_returns = []
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    for col in stock_returns_monthly.columns:
        y = data[col]  # Dependent variable
        X = data[factor_cols]  # Independent variables
        X = sm.add_constant(X)  # Add intercept

        # Fit OLS model
        model = sm.OLS(y, X).fit()

        # Predicted return = mean of all fitted values
        expected_returns.append(model.predict(X).mean())

    result = pd.Series(expected_returns, index=stock_returns_monthly.columns)
    logger.info(f"✓ Estimated returns for {len(result)} stocks")
    return result