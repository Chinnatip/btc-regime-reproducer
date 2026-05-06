from __future__ import annotations

from pathlib import Path

from btc_regime_repro.config import load_config


def test_load_config_expands_env_from_dotenv(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "data.csv"
    output_root = tmp_path / "runs"
    source_path.write_text("Timestamp,Open,High,Low,Close,Volume\n1640995200,1,1,1,1,1\n")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                f"BTC_TEST_SOURCE_PATH={source_path}",
                f"BTC_TEST_OUTPUT_ROOT={output_root}",
            ]
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  source_path: ${BTC_TEST_SOURCE_PATH}",
                "  source_format: csv",
                "  timestamp_column: Timestamp",
                "  timestamp_unit: s",
                "  open_column: Open",
                "  high_column: High",
                "  low_column: Low",
                "  close_column: Close",
                "  volume_column: Volume",
                "  symbol: BTCUSD",
                "  symbol_alias: BTC/USD",
                "  market_type: paper_native_csv",
                "  source_name: test_csv",
                "  resample_rule: 5min",
                "  start_utc:",
                "  end_utc:",
                "  short_gap_fill_limit_bars: 3",
                "  gap_fill_method: forward_fill",
                "  alternate_gap_fill_method: time_interpolation",
                "  outlier_zscore_threshold: 4.0",
                "  outlier_reference: log_return",
                "  paper_dataset_match: true",
                "experiment:",
                "  output_root: ${BTC_TEST_OUTPUT_ROOT}",
                "  volatility_window: 10",
                "  volatility_window_candidates: [10, 20]",
                "  feature_clip_quantile:",
                "  log_return_clip_quantile:",
                "  volatility_clip_quantile:",
                "  elbow_train_stride: 1",
                "  kmeans_train_stride: 1",
                "  hmm_train_stride: 1",
                "  k_values: [2, 3]",
                "  main_k: 3",
                "  n_states: 3",
                "  covariance_type: diag",
                "  max_iterations: 20",
                "  random_seed: 42",
                "  plot_start_utc:",
                "  plot_end_utc:",
                "  plot_max_points: 1000",
            ]
        )
    )

    monkeypatch.chdir(tmp_path)
    config = load_config(config_path)
    assert config.dataset.source_path == source_path.resolve()
    assert config.output_root == output_root.resolve()
