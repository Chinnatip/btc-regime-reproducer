from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import write_json
from .types import DatasetConfig

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


def _utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _read_source(config: DatasetConfig) -> pd.DataFrame:
    if config.source_format == "parquet":
        return pd.read_parquet(config.source_path)
    if config.source_format == "csv":
        return pd.read_csv(config.source_path)
    raise ValueError(f"Unsupported source_format: {config.source_format}")


def _parse_timestamp_series(series: pd.Series, unit: str | None) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, utc=True, unit=unit or "s", errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _canonicalize_columns(frame: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    report = {"rows_with_invalid_timestamp": 0}
    renamed = frame.rename(
        columns={
            config.timestamp_column: "timestamp",
            config.open_column: "open",
            config.high_column: "high",
            config.low_column: "low",
            config.close_column: "close",
            config.volume_column: "volume",
        }
    ).copy()

    if "timestamp" in renamed.columns:
        renamed["timestamp"] = _parse_timestamp_series(renamed["timestamp"], config.timestamp_unit)
        report["rows_with_invalid_timestamp"] = int(renamed["timestamp"].isna().sum())
        renamed = renamed.dropna(subset=["timestamp"]).set_index("timestamp")
    else:
        renamed.index = _parse_timestamp_series(pd.Series(renamed.index), config.timestamp_unit)
        report["rows_with_invalid_timestamp"] = int(pd.Series(renamed.index).isna().sum())
        renamed = renamed[~pd.Series(renamed.index).isna().to_numpy()]

    renamed.index = pd.to_datetime(renamed.index, utc=True)
    renamed.index.name = "timestamp"

    missing = [col for col in REQUIRED_COLUMNS if col not in renamed.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    out = renamed.loc[:, list(REQUIRED_COLUMNS)].copy()
    for col in REQUIRED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["close"])
    return out, report


def _minute_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="1min", tz="UTC")


def _summarize_missing_runs(mask: pd.Series) -> list[dict[str, object]]:
    if mask.empty:
        return []
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    runs: list[dict[str, object]] = []
    for _, run_mask in mask.groupby(groups):
        if not bool(run_mask.iloc[0]):
            continue
        runs.append(
            {
                "start_utc": str(run_mask.index[0]),
                "end_utc": str(run_mask.index[-1]),
                "bars": int(len(run_mask)),
            }
        )
    return runs


def _remove_outliers(frame: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    threshold = config.outlier_zscore_threshold
    if threshold is None:
        return frame.copy(), {"enabled": False, "removed_rows": 0, "reference": config.outlier_reference}

    out = frame.copy()
    if config.outlier_reference == "close":
        series = out["close"]
        z = (series - series.mean()) / series.std(ddof=0)
        outlier_mask = z.abs() > threshold
    elif config.outlier_reference == "log_return":
        log_return = np.log(out["close"]).diff()
        z = (log_return - log_return.mean()) / log_return.std(ddof=0)
        outlier_mask = z.abs() > threshold
        outlier_mask = outlier_mask.fillna(False)
    else:
        raise ValueError(f"Unsupported outlier_reference: {config.outlier_reference}")

    cleaned = out.loc[~outlier_mask].copy()
    return cleaned, {
        "enabled": True,
        "reference": config.outlier_reference,
        "zscore_threshold": float(threshold),
        "removed_rows": int(outlier_mask.sum()),
    }


def _repair_gaps(frame: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    if frame.empty:
        raise ValueError("Cannot repair gaps on an empty frame")

    full = frame.reindex(_minute_index(frame.index.min(), frame.index.max()))
    missing_mask = full["close"].isna()
    runs = _summarize_missing_runs(missing_mask)
    short_limit = max(int(config.short_gap_fill_limit_bars), 0)

    if short_limit > 0:
        for run in runs:
            bars = int(run["bars"])
            if bars > short_limit:
                continue
            start = pd.Timestamp(run["start_utc"])
            end = pd.Timestamp(run["end_utc"])
            run_index = full.loc[start:end].index
            prev_index = start - pd.Timedelta(minutes=1)
            next_index = end + pd.Timedelta(minutes=1)

            if config.gap_fill_method == "forward_fill":
                if prev_index in full.index and full.loc[prev_index, "close"] == full.loc[prev_index, "close"]:
                    prev_values = full.loc[prev_index, list(PRICE_COLUMNS)]
                    for col in PRICE_COLUMNS:
                        full.loc[run_index, col] = float(prev_values[col])
                    full.loc[run_index, "volume"] = 0.0
            elif config.gap_fill_method == "time_interpolation":
                neighbor_index = [idx for idx in (prev_index, next_index) if idx in full.index]
                if len(neighbor_index) == 2:
                    segment = full.loc[prev_index:next_index, list(PRICE_COLUMNS)].interpolate(method="time")
                    full.loc[run_index, list(PRICE_COLUMNS)] = segment.loc[run_index, list(PRICE_COLUMNS)]
                    volume_segment = full.loc[prev_index:next_index, ["volume"]].interpolate(method="time")
                    full.loc[run_index, "volume"] = volume_segment.loc[run_index, "volume"].fillna(0.0)
                elif prev_index in full.index and full.loc[prev_index, "close"] == full.loc[prev_index, "close"]:
                    prev_values = full.loc[prev_index, list(PRICE_COLUMNS)]
                    for col in PRICE_COLUMNS:
                        full.loc[run_index, col] = float(prev_values[col])
                    full.loc[run_index, "volume"] = 0.0
            else:
                raise ValueError(f"Unsupported gap_fill_method: {config.gap_fill_method}")

    cleaned = full.dropna(subset=["close"]).copy()
    cleaned.index.name = "timestamp"

    short_runs = [run for run in runs if int(run["bars"]) <= short_limit]
    long_runs = [run for run in runs if int(run["bars"]) > short_limit]
    return cleaned, {
        "gap_fill_method": config.gap_fill_method,
        "short_gap_fill_limit_bars": short_limit,
        "total_missing_bars_before_repair": int(sum(int(run["bars"]) for run in runs)),
        "short_gap_count": int(len(short_runs)),
        "short_gap_filled_bars": int(sum(int(run["bars"]) for run in short_runs)),
        "long_gap_count": int(len(long_runs)),
        "long_gap_removed_bars": int(sum(int(run["bars"]) for run in long_runs)),
        "long_gap_examples": long_runs[:5],
    }


def load_raw_ohlcv(config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = _read_source(config)
    canonical, canonical_report = _canonicalize_columns(frame, config)
    canonical = canonical.sort_index()
    duplicate_count = int(canonical.index.duplicated(keep="last").sum())
    canonical = canonical[~canonical.index.duplicated(keep="last")]

    if config.start_utc:
        canonical = canonical.loc[_utc_timestamp(config.start_utc) :]
    if config.end_utc:
        canonical = canonical.loc[: _utc_timestamp(config.end_utc)]

    if canonical.empty:
        raise ValueError("Dataset is empty after applying time filters")

    return canonical, {
        "source_format": config.source_format,
        "rows_after_ingest": int(len(canonical)),
        "duplicate_timestamps_removed": duplicate_count,
        **canonical_report,
    }


def build_plotting_frame(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        frame.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


def build_modeling_frame(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        frame.resample(rule)
        .agg(
            {
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


def coverage_summary(frame: pd.DataFrame, expected_freq: str, duplicate_timestamps_removed: int = 0) -> dict[str, object]:
    diffs = frame.index.to_series().diff().dropna()
    expected_delta = pd.Timedelta(expected_freq)
    gap_series = diffs[diffs > expected_delta]
    return {
        "start_utc": str(frame.index.min()),
        "end_utc": str(frame.index.max()),
        "row_count": int(len(frame)),
        "duplicate_timestamps_removed": int(duplicate_timestamps_removed),
        "expected_frequency": expected_freq,
        "missing_bar_gaps": int(len(gap_series)),
        "largest_gap_seconds": float(gap_series.max().total_seconds()) if not gap_series.empty else 0.0,
    }


def prepare_dataset(
    config: DatasetConfig,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    raw, ingest_report = load_raw_ohlcv(config)
    outlier_cleaned, outlier_report = _remove_outliers(raw, config)
    repaired, gap_report = _repair_gaps(outlier_cleaned, config)
    plotting_5m = build_plotting_frame(repaired, config.resample_rule)
    modeling_5m = build_modeling_frame(repaired, config.resample_rule)

    raw_summary = coverage_summary(raw, "1min", duplicate_timestamps_removed=ingest_report["duplicate_timestamps_removed"])
    preprocessed_summary = coverage_summary(repaired, "1min")
    plotting_summary = coverage_summary(plotting_5m, config.resample_rule)
    modeling_summary = coverage_summary(modeling_5m, config.resample_rule)

    dataset_manifest = {
        "dataset": asdict(config),
        "ingest": ingest_report,
        "raw_1m": raw_summary,
        "preprocessed_1m": preprocessed_summary,
        "plotting_5m": plotting_summary,
        "modeling_5m": modeling_summary,
        "paper_alignment": {
            "methods_primary": True,
            "figure2_is_treated_as_illustrative": True,
            "paper_dataset_match": bool(config.paper_dataset_match),
        },
        "fidelity_caveat": (
            "This run is not paper-faithful until the exact BTC/USD 2012-2021 dataset is used "
            "and all unresolved preprocessing ambiguities are reviewed."
            if not config.paper_dataset_match
            else "This run uses a paper-native dataset contract, but unresolved paper ambiguities may remain."
        ),
    }
    coverage_report = {
        "raw_1m": raw_summary,
        "preprocessed_1m": preprocessed_summary,
        "plotting_5m": plotting_summary,
        "modeling_5m": modeling_summary,
        "outlier_filter": outlier_report,
        "gap_repair": gap_report,
    }
    write_json(out_dir / "dataset_manifest.json", dataset_manifest)
    write_json(out_dir / "coverage_report.json", coverage_report)
    return repaired, plotting_5m, modeling_5m, dataset_manifest, coverage_report
