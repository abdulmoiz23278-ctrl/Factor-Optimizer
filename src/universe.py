"""Point-in-time S&P 500 constituent universe (fja05680/sp500)."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "sp500_history.csv"
SP500_RAW_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%2801-17-2026%29.csv"
)


def ensure_sp500_csv(csv_path: str | Path = DEFAULT_CSV) -> Path:
    """Download constituent history if missing."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        logger.info(f"Downloading S&P 500 history → {path}")
        urllib.request.urlretrieve(SP500_RAW_URL, path)
    return path


def load_sp500_history(csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Returns DataFrame indexed by date with a 'tickers' set column."""
    path = ensure_sp500_csv(csv_path)
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    if "date" not in raw.columns or "tickers" not in raw.columns:
        raise ValueError(f"Unexpected SP500 CSV columns: {list(raw.columns)}")

    raw["date"] = pd.to_datetime(raw["date"])
    raw["tickers"] = raw["tickers"].apply(
        lambda s: {t.strip().upper() for t in str(s).split(",") if t.strip()}
    )
    history = raw.set_index("date").sort_index()
    logger.info(f"✓ Loaded S&P 500 history: {len(history)} snapshots")
    return history


def get_universe_as_of(date: pd.Timestamp, history: pd.DataFrame) -> list[str]:
    """Returns S&P 500 constituents as of date (largest date <= input)."""
    date = pd.Timestamp(date)
    eligible = history.index[history.index <= date]
    if len(eligible) == 0:
        return []
    tickers = history.loc[eligible[-1], "tickers"]
    return sorted(tickers)


def get_universe_union(
    start: pd.Timestamp,
    end: pd.Timestamp,
    history: pd.DataFrame,
) -> list[str]:
    """Union of tickers that were S&P 500 members at any point in [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    subset = history[(history.index >= start) & (history.index <= end)]
    if subset.empty:
        subset = history[history.index <= end]
    universe: set[str] = set()
    for tickers in subset["tickers"]:
        universe |= tickers
    return sorted(universe)
