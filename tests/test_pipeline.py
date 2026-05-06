from __future__ import annotations

from pathlib import Path

from btc_regime_repro.pipeline import run_reproduction
from btc_regime_repro.types import DatasetConfig, ExperimentConfig


def test_run_reproduction_emits_paper_outputs(tmp_path) -> None:
    csv_path = tmp_path / "btcusd_1-min_data.csv"
    lines = ["Timestamp,Open,High,Low,Close,Volume"]
    price = 100.0
    ts = 1640995200
    for i in range(180):
        if i < 60:
            price += 0.4
        elif i < 120:
            price -= 0.35
        else:
            price += 0.05
        open_price = price - 0.1
        high_price = price + 0.2
        low_price = price - 0.2
        volume = 10 + (i % 5)
        lines.append(f"{ts},{open_price:.4f},{high_price:.4f},{low_price:.4f},{price:.4f},{volume}")
        ts += 60
    csv_path.write_text("\n".join(lines))

    config = ExperimentConfig(
        dataset=DatasetConfig(
            source_path=csv_path,
            source_format="csv",
            timestamp_column="Timestamp",
            timestamp_unit="s",
            open_column="Open",
            high_column="High",
            low_column="Low",
            close_column="Close",
            volume_column="Volume",
            symbol="BTCUSD",
            market_type="paper_native_csv",
            source_name="btcusd_1-min_data",
            symbol_alias="BTC/USD",
            resample_rule="5min",
            start_utc=None,
            end_utc=None,
            short_gap_fill_limit_bars=3,
            gap_fill_method="forward_fill",
            alternate_gap_fill_method="time_interpolation",
            outlier_zscore_threshold=4.0,
            outlier_reference="log_return",
            paper_dataset_match=True,
        ),
        output_root=tmp_path / "runs",
        volatility_window=3,
        volatility_window_candidates=(2, 3),
        feature_clip_quantile=None,
        log_return_clip_quantile=None,
        volatility_clip_quantile=None,
        elbow_train_stride=1,
        kmeans_train_stride=1,
        hmm_train_stride=1,
        k_values=(2, 3),
        main_k=3,
        n_states=3,
        covariance_type="diag",
        max_iterations=50,
        random_seed=42,
        plot_start_utc="2022-01-01T00:00:00Z",
        plot_end_utc="2022-01-01T02:59:00Z",
        plot_max_points=2000,
    )

    run_dir = run_reproduction(config)
    expected = [
        "preprocessed_1m.parquet",
        "prepared_5m.parquet",
        "modeling_5m.parquet",
        "table1_descriptive_stats.csv",
        "table2_cluster_centroids.csv",
        "table3_transition_matrix.csv",
        "table4_model_comparison.csv",
        "paper_fidelity_report.md",
        "kmeans_only_regime_plot.png",
        "hybrid_hmm_regime_plot.png",
    ]
    for name in expected:
        assert (run_dir / name).exists(), name
