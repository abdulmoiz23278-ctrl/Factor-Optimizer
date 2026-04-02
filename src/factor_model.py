import pandas as pd
import statsmodels.api as sm
import pandas_datareader.data as web

def get_fama_french_factors():
    ff = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench')[0]
    ff = ff / 100

    # 🔥 Convert index to datetime (CRITICAL FIX)
    ff.index = pd.to_datetime(ff.index.astype(str))

    return ff

def estimate_expected_returns(stock_returns):
    factors = get_fama_french_factors()

    # 🔥 Convert daily → monthly (end of month)
    stock_returns = stock_returns.resample('M').mean()

    # 🔥 Convert stock index to match format
    stock_returns.index = stock_returns.index.to_period('M').to_timestamp()

    # 🔥 Align datasets
    data = stock_returns.join(factors, how='inner')

    # 🔥 Drop any missing values
    data = data.dropna()

    expected_returns = []

    for col in stock_returns.columns:
        y = data[col]
        X = data[['Mkt-RF','SMB','HML','RMW','CMA']]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        expected_returns.append(model.predict(X).mean())

    return pd.Series(expected_returns, index=stock_returns.columns)