from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .types import DatasetConfig, ExperimentConfig


ENV_VAR_PATTERN = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _find_dotenv(config_path: Path) -> Path | None:
    candidates = [
        Path.cwd() / ".env",
        config_path.parent / ".env",
        config_path.parent.parent / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _expand_env_in_value(value: Any) -> Any:
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            name = match.group("braced") or match.group("plain")
            if name not in os.environ:
                raise KeyError(f"Missing required environment variable: {name}")
            return os.environ[name]

        return ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, list):
        return [_expand_env_in_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_in_value(item) for key, item in value.items()}
    return value


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path).expanduser().resolve()
    dotenv_path = _find_dotenv(path)
    if dotenv_path is not None:
        _load_dotenv(dotenv_path)

    data = _expand_env_in_value(yaml.safe_load(path.read_text()))
    dataset = data["dataset"]
    experiment = data["experiment"]

    return ExperimentConfig(
        dataset=DatasetConfig(
            source_path=Path(dataset["source_path"]).expanduser().resolve(),
            source_format=str(dataset.get("source_format", "parquet")),
            timestamp_column=str(dataset.get("timestamp_column", "timestamp")),
            timestamp_unit=dataset.get("timestamp_unit"),
            open_column=str(dataset.get("open_column", "open")),
            high_column=str(dataset.get("high_column", "high")),
            low_column=str(dataset.get("low_column", "low")),
            close_column=str(dataset.get("close_column", "close")),
            volume_column=str(dataset.get("volume_column", "volume")),
            symbol=str(dataset["symbol"]),
            market_type=str(dataset["market_type"]),
            source_name=str(dataset["source_name"]),
            symbol_alias=dataset.get("symbol_alias"),
            resample_rule=str(dataset.get("resample_rule", "5min")),
            start_utc=dataset.get("start_utc"),
            end_utc=dataset.get("end_utc"),
            short_gap_fill_limit_bars=int(dataset.get("short_gap_fill_limit_bars", 3)),
            gap_fill_method=str(dataset.get("gap_fill_method", "forward_fill")),
            alternate_gap_fill_method=dataset.get("alternate_gap_fill_method", "time_interpolation"),
            outlier_zscore_threshold=(
                float(dataset["outlier_zscore_threshold"])
                if dataset.get("outlier_zscore_threshold") is not None
                else None
            ),
            outlier_reference=str(dataset.get("outlier_reference", "log_return")),
            paper_dataset_match=bool(dataset.get("paper_dataset_match", False)),
        ),
        output_root=Path(experiment.get("output_root", "runs")).expanduser().resolve(),
        volatility_window=int(experiment.get("volatility_window", 20)),
        volatility_window_candidates=tuple(
            int(v) for v in experiment.get("volatility_window_candidates", [experiment.get("volatility_window", 20)])
        ),
        feature_clip_quantile=(
            float(experiment["feature_clip_quantile"])
            if experiment.get("feature_clip_quantile") is not None
            else None
        ),
        log_return_clip_quantile=(
            float(experiment["log_return_clip_quantile"])
            if experiment.get("log_return_clip_quantile") is not None
            else (
                float(experiment["feature_clip_quantile"])
                if experiment.get("feature_clip_quantile") is not None
                else None
            )
        ),
        volatility_clip_quantile=(
            float(experiment["volatility_clip_quantile"])
            if experiment.get("volatility_clip_quantile") is not None
            else (
                float(experiment["feature_clip_quantile"])
                if experiment.get("feature_clip_quantile") is not None
                else None
            )
        ),
        elbow_train_stride=int(experiment.get("elbow_train_stride", experiment.get("kmeans_train_stride", 1))),
        kmeans_train_stride=int(experiment.get("kmeans_train_stride", 1)),
        hmm_train_stride=int(experiment.get("hmm_train_stride", 1)),
        k_values=tuple(int(v) for v in experiment.get("k_values", [2, 3, 4, 5, 6])),
        main_k=int(experiment.get("main_k", 3)),
        n_states=int(experiment.get("n_states", 3)),
        covariance_type=str(experiment.get("covariance_type", "diag")),
        max_iterations=int(experiment.get("max_iterations", 300)),
        random_seed=int(experiment.get("random_seed", 42)),
        plot_start_utc=experiment.get("plot_start_utc"),
        plot_end_utc=experiment.get("plot_end_utc"),
        plot_max_points=int(experiment.get("plot_max_points", 25000)),
    )
