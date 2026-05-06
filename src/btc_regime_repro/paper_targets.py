from __future__ import annotations

from typing import Iterable

import pandas as pd


PAPER_TABLE1 = {
    "Close Price (USD)": {"mean": 8732.41, "std_dev": 10485.23, "min": 65.53, "max": 67617.00},
    "Log Return": {"mean": 0.00021, "std_dev": 0.00294, "min": -0.0813, "max": 0.0759},
    "Volatility (σₜ)": {"mean": 0.00287, "std_dev": 0.00311, "min": 0.0001, "max": 0.0392},
    "Volume (BTC)": {"mean": 134.27, "std_dev": 289.15, "min": 0.0, "max": 15309.0},
}

PAPER_TABLE2 = {
    "bullish": {"mean_log_return": 0.0024, "mean_volatility": 0.0011},
    "bearish": {"mean_log_return": -0.0018, "mean_volatility": 0.0029},
    "sideways": {"mean_log_return": 0.0003, "mean_volatility": 0.0007},
}

PAPER_TABLE3 = {
    "bullish": {"bullish": 0.76, "bearish": 0.18, "sideways": 0.06},
    "bearish": {"bullish": 0.21, "bearish": 0.70, "sideways": 0.09},
    "sideways": {"bullish": 0.12, "bearish": 0.15, "sideways": 0.73},
}


def relative_closeness(actual: float, target: float) -> float:
    denom = abs(target) if abs(target) > 1e-12 else max(abs(actual), 1e-12)
    rel_error = abs(actual - target) / denom
    return max(0.0, 1.0 - rel_error)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def score_table1(table1: pd.DataFrame) -> float:
    indexed = table1.set_index("feature")
    scores: list[float] = []
    for feature_name, target_row in PAPER_TABLE1.items():
        for col, target in target_row.items():
            scores.append(relative_closeness(float(indexed.loc[feature_name, col]), float(target)))
    return _mean(scores)


def score_table2(table2: pd.DataFrame) -> float:
    indexed = table2.set_index("mapped_regime")
    scores: list[float] = []
    for regime_name, target_row in PAPER_TABLE2.items():
        for col, target in target_row.items():
            scores.append(relative_closeness(float(indexed.loc[regime_name, col]), float(target)))
    return _mean(scores)


def score_table3(table3: pd.DataFrame) -> float:
    indexed = table3.copy()
    if "from_regime" in indexed.columns:
        indexed = indexed.set_index("from_regime")
    scores: list[float] = []
    for from_regime, target_row in PAPER_TABLE3.items():
        for to_regime, target in target_row.items():
            scores.append(relative_closeness(float(indexed.loc[from_regime, to_regime]), float(target)))
    return _mean(scores)


def score_tables(table1: pd.DataFrame, table2: pd.DataFrame, table3: pd.DataFrame) -> dict[str, float]:
    table1_score = score_table1(table1)
    table2_score = score_table2(table2)
    table3_score = score_table3(table3)
    overall_score = (table1_score + table2_score + table3_score) / 3.0
    return {
        "table1_score": table1_score,
        "table2_score": table2_score,
        "table3_score": table3_score,
        "overall_score": overall_score,
    }
