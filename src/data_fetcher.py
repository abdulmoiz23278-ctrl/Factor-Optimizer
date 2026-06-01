import logging

import pandas as pd  # type: ignore[import-untyped]
import yfinance as yf  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

CHUNK_SIZE = 50


def _normalize_ticker(ticker: str) -> str:
    """yfinance: BRK.B → BRK-B."""
    return ticker.replace(".", "-")


def _download_chunk(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download Close prices for one chunk; columns are yfinance symbols."""
    yf_tickers = [_normalize_ticker(t) for t in tickers]
    data = yf.download(
        yf_tickers,
        start=start,
        end=end,
        progress=False,
        group_by="column",
        auto_adjust=True,
    )
    if data.empty:
        return pd.DataFrame()

    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    # Map yfinance columns back to original ticker labels where possible
    rename = {}
    for orig, yf_sym in zip(tickers, yf_tickers):
        if yf_sym in close.columns:
            rename[yf_sym] = orig
        elif orig in close.columns:
            rename[orig] = orig
    close = close.rename(columns=rename)
    return close


def fetch_stock_returns(tickers, start="2015-01-01", end="2023-01-01"):
    """
    Fetch daily stock returns for given tickers.

    Preserves NaN tails for delisted names (no global dropna).
    Downloads in chunks of 50 to reduce rate-limit issues.

    Returns:
        pd.DataFrame: Daily returns; NaN where price data missing (e.g. post-delisting).

    Raises:
        ValueError: If no data retrieved for any ticker.
    """
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("No tickers provided.")

    try:
        frames: list[pd.DataFrame] = []
        skipped: list[str] = []

        for i in range(0, len(tickers), CHUNK_SIZE):
            chunk = tickers[i : i + CHUNK_SIZE]
            close = _download_chunk(chunk, start, end)
            for t in chunk:
                if t not in close.columns or close[t].dropna().empty:
                    skipped.append(t)
            if not close.empty:
                frames.append(close)

        if skipped:
            logger.warning(
                f"Skipped {len(skipped)} tickers with no yfinance data "
                f"(first 10: {skipped[:10]})"
            )

        if not frames:
            raise ValueError(f"No data retrieved for tickers: {tickers[:20]}...")

        prices = pd.concat(frames, axis=1)
        prices = prices.loc[:, ~prices.columns.duplicated()]
        prices = prices.sort_index()

        returns = prices.pct_change().iloc[1:]

        valid_cols = [c for c in returns.columns if returns[c].notna().any()]
        returns = returns[valid_cols]

        logger.info(
            f"✓ Downloaded {len(valid_cols)}/{len(tickers)} tickers, "
            f"{len(returns)} trading days"
        )
        return returns

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise ValueError(f"Data fetch failed: {e}") from e
