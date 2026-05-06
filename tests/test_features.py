from __future__ import annotations

import numpy as np
import pandas as pd

from btc_regime_repro.features import build_feature_frame, clip_feature_tails, zscore_features


def test_feature_build_matches_expected_columns() -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="5min")
    frame = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5, 6],
            "high": [2, 3, 4, 5, 6, 7],
            "low": [1, 2, 3, 4, 5, 6],
            "close": [2, 3, 4, 5, 6, 7],
            "volume": [10, 10, 10, 10, 10, 10],
        },
        index=index,
    )
    out = build_feature_frame(frame, volatility_window=3)
    assert "log_return" in out.columns
    assert "rolling_volatility" in out.columns
    assert len(out) > 0


def test_zscore_features_builds_normalized_columns() -> None:
    frame = pd.DataFrame(
        {
            "log_return": [-0.1, 0.0, 0.2],
            "rolling_volatility": [0.5, 0.6, 0.8],
        },
        index=pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="5min"),
    )
    out = zscore_features(frame)
    assert "log_return_z" in out.columns
    assert "rolling_volatility_z" in out.columns
    assert abs(float(out["log_return_z"].mean())) < 1e-9


def test_clip_feature_tails_limits_extremes() -> None:
    frame = pd.DataFrame(
        {
            "log_return": [-0.5, 0.0, 0.1, 0.2, 1.5],
            "rolling_volatility": [0.001, 0.002, 0.003, 0.004, 0.5],
        }
    )
    out = clip_feature_tails(frame, 0.8)
    assert out["log_return"].max() < frame["log_return"].max()
    assert out["rolling_volatility"].max() < frame["rolling_volatility"].max()
    assert np.isclose(out["log_return"].iloc[1], frame["log_return"].iloc[1])


def test_clip_feature_tails_supports_per_feature_quantiles() -> None:
    frame = pd.DataFrame(
        {
            "log_return": [-0.5, 0.0, 0.1, 0.2, 1.5],
            "rolling_volatility": [0.001, 0.002, 0.003, 0.004, 0.5],
        }
    )
    out = clip_feature_tails(frame, None, log_return_clip_quantile=0.95, volatility_clip_quantile=0.8)
    assert out["rolling_volatility"].max() < out["log_return"].max()
