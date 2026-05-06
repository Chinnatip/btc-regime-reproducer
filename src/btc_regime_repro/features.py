from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_frame(frame: pd.DataFrame, volatility_window: int) -> pd.DataFrame:
    out = frame.copy()
    out["log_return"] = np.log(out["close"]).diff()
    out["rolling_volatility"] = out["log_return"].rolling(volatility_window, min_periods=volatility_window).std(ddof=0)
    out = out.dropna(subset=["log_return", "rolling_volatility"]).copy()
    return out


def _clip_series(series: pd.Series, clip_quantile: float | None) -> pd.Series:
    if clip_quantile is None:
        return series.copy()
    if not 0.5 < float(clip_quantile) < 1.0:
        raise ValueError(f"feature clip quantile must be in (0.5, 1.0), got {clip_quantile}")
    low = series.quantile(1.0 - float(clip_quantile))
    high = series.quantile(float(clip_quantile))
    return series.clip(low, high)


def clip_feature_tails(
    frame: pd.DataFrame,
    clip_quantile: float | None = None,
    *,
    log_return_clip_quantile: float | None = None,
    volatility_clip_quantile: float | None = None,
) -> pd.DataFrame:
    if clip_quantile is None and log_return_clip_quantile is None and volatility_clip_quantile is None:
        return frame.copy()
    out = frame.copy()
    return_q = log_return_clip_quantile if log_return_clip_quantile is not None else clip_quantile
    vol_q = volatility_clip_quantile if volatility_clip_quantile is not None else clip_quantile
    out["log_return"] = _clip_series(out["log_return"], return_q)
    out["rolling_volatility"] = _clip_series(out["rolling_volatility"], vol_q)
    return out


def zscore_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cols = ["log_return", "rolling_volatility"]
    for col in cols:
        mean = out[col].mean()
        std = out[col].std(ddof=0)
        if not np.isfinite(std) or std == 0.0:
            raise ValueError(f"Cannot z-score feature {col}: std={std}")
        out[f"{col}_z"] = (out[col] - mean) / std
    return out
