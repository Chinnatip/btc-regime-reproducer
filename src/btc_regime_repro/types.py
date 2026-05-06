from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    source_path: Path
    source_format: str
    timestamp_column: str
    timestamp_unit: str | None
    open_column: str
    high_column: str
    low_column: str
    close_column: str
    volume_column: str
    symbol: str
    market_type: str
    source_name: str
    symbol_alias: str | None
    resample_rule: str
    start_utc: str | None
    end_utc: str | None
    short_gap_fill_limit_bars: int
    gap_fill_method: str
    alternate_gap_fill_method: str | None
    outlier_zscore_threshold: float | None
    outlier_reference: str
    paper_dataset_match: bool


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    output_root: Path
    volatility_window: int
    volatility_window_candidates: tuple[int, ...]
    feature_clip_quantile: float | None
    log_return_clip_quantile: float | None
    volatility_clip_quantile: float | None
    elbow_train_stride: int
    kmeans_train_stride: int
    hmm_train_stride: int
    k_values: tuple[int, ...]
    main_k: int
    n_states: int
    covariance_type: str
    max_iterations: int
    random_seed: int
    plot_start_utc: str | None
    plot_end_utc: str | None
    plot_max_points: int


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value
