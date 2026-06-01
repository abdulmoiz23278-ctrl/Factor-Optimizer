"""Post-backtest invariants loaded from tests/_backtest_state.pkl (written by main.py)."""

import pickle
from pathlib import Path

import numpy as np
import pytest

STATE_PATH = Path(__file__).resolve().parent / "_backtest_state.pkl"


@pytest.fixture(scope="module")
def backtest_state():
    if not STATE_PATH.exists():
        pytest.skip(
            f"Missing {STATE_PATH}; run `python main.py` from repo root first."
        )
    with STATE_PATH.open("rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def weight_history(backtest_state):
    return backtest_state["weight_history"]


@pytest.fixture(scope="module")
def config(backtest_state):
    return backtest_state["config"]


def test_max_weight_respected(weight_history, config):
    """No weight in any period exceeds max_weight + 1e-6."""
    max_weight = config["max_weight"]
    for wh in weight_history:
        max_w = max(wh["weights"].values())
        assert max_w <= max_weight + 1e-6, (
            f"max_weight violated at {wh['date']}: {max_w:.4f}"
        )


def test_min_weight_respected(weight_history, config):
    """No nonzero weight is below min_weight - 1e-6."""
    min_weight = config["min_weight_threshold"]
    for wh in weight_history:
        nonzero = [w for w in wh["weights"].values() if w > 1e-9]
        if nonzero:
            min_w = min(nonzero)
            assert min_w >= min_weight - 1e-6, (
                f"min_weight violated at {wh['date']}: {min_w:.4f}"
            )


def test_weights_sum_to_one(weight_history):
    """Each rebalance's weights sum to ~1."""
    for wh in weight_history:
        total = sum(wh["weights"].values())
        assert abs(total - 1.0) < 1e-4, (
            f"Weights don't sum to 1 at {wh['date']}: {total}"
        )


def test_universe_changes_over_time(weight_history):
    """Holdings should change across rebalances (selected 50-name subset)."""
    first_tickers = set(weight_history[0]["weights"].keys())
    last_tickers = set(weight_history[-1]["weights"].keys())
    diff = first_tickers.symmetric_difference(last_tickers)
    assert len(diff) > 0, (
        "Universe identical at first and last rebalance — survivorship bias still present"
    )
