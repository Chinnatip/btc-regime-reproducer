from __future__ import annotations

import numpy as np
import pandas as pd

from btc_regime_repro.clustering import elbow_diagnostics, fit_kmeans, map_clusters_to_regimes
from btc_regime_repro.hmm_model import build_transition_matrix, fit_hmm_variant


def feature_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=24, freq="5min")
    log_return = np.concatenate([np.full(8, -0.02), np.full(8, 0.0), np.full(8, 0.02)])
    rolling_volatility = np.concatenate([np.full(8, 0.03), np.full(8, 0.01), np.full(8, 0.02)])
    frame = pd.DataFrame(
        {
            "log_return": log_return,
            "rolling_volatility": rolling_volatility,
        },
        index=index,
    )
    for col in ("log_return", "rolling_volatility"):
        frame[f"{col}_z"] = (frame[col] - frame[col].mean()) / frame[col].std(ddof=0)
    return frame


def test_elbow_runs_all_requested_k() -> None:
    out = elbow_diagnostics(feature_frame(), (2, 3, 4), random_seed=42)
    assert out["k"].tolist() == [2, 3, 4]


def test_cluster_mapping_produces_named_regimes() -> None:
    frame = feature_frame()
    labels = pd.Series(([0] * 8) + ([1] * 8) + ([2] * 8), index=frame.index, name="cluster")
    regimes, stats = map_clusters_to_regimes(frame, labels)
    assert len(regimes) == len(frame)
    assert set(stats["mapped_regime"]).issubset({"bullish", "bearish", "sideways"})
    assert stats.loc[0, "mapped_regime"] == "bearish"
    assert stats.loc[1, "mapped_regime"] == "sideways"
    assert stats.loc[2, "mapped_regime"] == "bullish"


def test_hmm_variants_and_transition_rows_sum_to_one() -> None:
    frame = feature_frame()
    seed_labels = fit_kmeans(frame, n_clusters=3, random_seed=42).labels.to_numpy(dtype=int)
    result = fit_hmm_variant(
        frame,
        model_name="seeded",
        n_states=3,
        covariance_type="diag",
        max_iterations=50,
        random_seed=42,
        initial_labels=seed_labels,
    )
    row_sums = result.model_transition_matrix.sum(axis=1).round(8)
    assert all(value in (0.0, 1.0) for value in row_sums.tolist())
    random_result = fit_hmm_variant(
        frame,
        model_name="random",
        n_states=3,
        covariance_type="diag",
        max_iterations=50,
        random_seed=42,
        initial_labels=None,
    )
    assert random_result.metrics["n_states"] == 3


def test_transition_matrix_normalizes_rows() -> None:
    series = pd.Series(["bullish", "bullish", "sideways", "bearish"])
    matrix = build_transition_matrix(series)
    assert matrix.loc["bullish"].sum() == 1.0
