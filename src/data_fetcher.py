import yfinance as yf

def fetch_stock_returns(tickers, start="2015-01-01", end="2023-01-01"):
    data = yf.download(tickers, start=start, end=end)
    data = data["Close"]
    returns = data.pct_change().dropna()
    return returns