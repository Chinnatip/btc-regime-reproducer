from __future__ import annotations

import pandas as pd

from btc_regime_repro.dataset import (
    _remove_outliers,
    _repair_gaps,
    build_modeling_frame,
    build_plotting_frame,
    coverage_summary,
    load_raw_ohlcv,
)
from btc_regime_repro.types import DatasetConfig


def dataset_config(**overrides: object) -> DatasetConfig:
    base = DatasetConfig(
        source_path=overrides.pop("source_path", None),
        source_format=str(overrides.pop("source_format", "csv")),
        timestamp_column=str(overrides.pop("timestamp_column", "Timestamp")),
        timestamp_unit=overrides.pop("timestamp_unit", "s"),
        open_column=str(overrides.pop("open_column", "Open")),
        high_column=str(overrides.pop("high_column", "High")),
        low_column=str(overrides.pop("low_column", "Low")),
        close_column=str(overrides.pop("close_column", "Close")),
        volume_column=str(overrides.pop("volume_column", "Volume")),
        symbol=str(overrides.pop("symbol", "BTCUSD")),
        market_type=str(overrides.pop("market_type", "paper_native_csv")),
        source_name=str(overrides.pop("source_name", "test_source")),
        symbol_alias=overrides.pop("symbol_alias", "BTC/USD"),
        resample_rule=str(overrides.pop("resample_rule", "5min")),
        start_utc=overrides.pop("start_utc", None),
        end_utc=overrides.pop("end_utc", None),
        short_gap_fill_limit_bars=int(overrides.pop("short_gap_fill_limit_bars", 1)),
        gap_fill_method=str(overrides.pop("gap_fill_method", "forward_fill")),
        alternate_gap_fill_method=overrides.pop("alternate_gap_fill_method", "time_interpolation"),
        outlier_zscore_threshold=overrides.pop("outlier_zscore_threshold", 1.5),
        outlier_reference=str(overrides.pop("outlier_reference", "log_return")),
        paper_dataset_match=bool(overrides.pop("paper_dataset_match", False)),
    )
    if overrides:
        raise AssertionError(f"Unexpected overrides: {sorted(overrides)}")
    return base


def test_build_plotting_and_modeling_frames_use_last_trade_close() -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="1min")
    frame = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0, 1, 2, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [10, 20, 30, 40, 50],
        },
        index=index,
    )
    plotting = build_plotting_frame(frame, "5min")
    modeling = build_modeling_frame(frame, "5min")
    assert plotting.iloc[0]["close"] == 5.5
    assert modeling.iloc[0]["close"] == 5.5
    assert modeling.iloc[0]["volume"] == 150


def test_coverage_summary_detects_gap() -> None:
    index = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
            "2026-01-01T00:03:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1], "volume": [1, 1, 1]},
        index=index,
    )
    summary = coverage_summary(frame, "1min")
    assert summary["missing_bar_gaps"] == 1


def test_repair_gaps_fills_short_and_drops_long() -> None:
    index = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
            "2026-01-01T00:03:00Z",
            "2026-01-01T00:07:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "open": [100, 101, 103, 107],
            "high": [100, 101, 103, 107],
            "low": [100, 101, 103, 107],
            "close": [100, 101, 103, 107],
            "volume": [1, 1, 1, 1],
        },
        index=index,
    )
    repaired, report = _repair_gaps(frame, dataset_config(short_gap_fill_limit_bars=1))
    assert pd.Timestamp("2026-01-01T00:02:00Z") in repaired.index
    assert pd.Timestamp("2026-01-01T00:04:00Z") not in repaired.index
    assert report["short_gap_filled_bars"] == 1
    assert report["long_gap_removed_bars"] == 3


def test_remove_outliers_drops_spike_from_log_return() -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="1min")
    frame = pd.DataFrame(
        {
            "open": [100, 101, 500, 102, 103],
            "high": [100, 101, 500, 102, 103],
            "low": [100, 101, 500, 102, 103],
            "close": [100, 101, 500, 102, 103],
            "volume": [1, 1, 1, 1, 1],
        },
        index=index,
    )
    cleaned, report = _remove_outliers(frame, dataset_config(outlier_zscore_threshold=1.0))
    assert len(cleaned) < len(frame)
    assert report["removed_rows"] >= 1


def test_load_raw_ohlcv_supports_csv_input(tmp_path) -> None:
    csv_path = tmp_path / "btcusd_1-min_data.csv"
    csv_path.write_text(
        "Timestamp,Open,High,Low,Close,Volume\n"
        "1640995200,100,101,99,100.5,2\n"
        "1640995260,100.5,102,100,101.0,3\n"
    )
    frame, report = load_raw_ohlcv(dataset_config(source_path=csv_path))
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert str(frame.index[0]) == "2022-01-01 00:00:00+00:00"
    assert report["source_format"] == "csv"
