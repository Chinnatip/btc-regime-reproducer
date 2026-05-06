from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_regime_repro.clustering import fit_kmeans_with_stride, map_clusters_to_regimes
from btc_regime_repro.config import load_config
from btc_regime_repro.dataset import prepare_dataset
from btc_regime_repro.features import build_feature_frame, zscore_features
from btc_regime_repro.hmm_model import fit_hmm_variant
from btc_regime_repro.paper_targets import score_table1, score_table2, score_tables
from btc_regime_repro.pipeline import _table1_descriptive_stats, _table2_cluster_centroids, make_run_dir


def _table3_from_hmm(result) -> pd.DataFrame:
    table3 = result.model_transition_matrix.copy()
    table3.index.name = "from_regime"
    return table3.reset_index()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sweep paper-facing parameters to improve Table 1-3 fidelity.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    base = load_config(args.config)
    run_dir = make_run_dir(base.output_root, prefix="paper_tune")

    volatility_windows = [10, 20, 30]
    outlier_thresholds = [2.5, 3.0, 3.5]
    gap_fill_methods = ["forward_fill"]
    hmm_strides = [5, 10, 20]

    shortlist_rows: list[dict[str, object]] = []
    for gap_fill_method in gap_fill_methods:
        for outlier_threshold in outlier_thresholds:
            dataset_cfg = replace(
                base.dataset,
                gap_fill_method=gap_fill_method,
                outlier_zscore_threshold=outlier_threshold,
            )
            combo_dir = run_dir / f"prep_{gap_fill_method}_z{str(outlier_threshold).replace('.', '_')}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            _, _, modeling_5m, _, _ = prepare_dataset(dataset_cfg, combo_dir)

            for volatility_window in volatility_windows:
                feature_frame = zscore_features(build_feature_frame(modeling_5m, volatility_window=volatility_window))
                kmeans_run = fit_kmeans_with_stride(
                    feature_frame,
                    n_clusters=base.main_k,
                    random_seed=base.random_seed,
                    train_stride=base.kmeans_train_stride,
                )
                _, cluster_stats = map_clusters_to_regimes(feature_frame, kmeans_run.labels)
                table1 = _table1_descriptive_stats(modeling_5m, feature_frame)
                table2 = _table2_cluster_centroids(cluster_stats)
                table1_score = score_table1(table1)
                table2_score = score_table2(table2)
                shortlist_rows.append(
                    {
                        "gap_fill_method": gap_fill_method,
                        "outlier_zscore_threshold": outlier_threshold,
                        "volatility_window": volatility_window,
                        "table1_score": table1_score,
                        "table2_score": table2_score,
                        "shortlist_score": (table1_score + table2_score) / 2.0,
                    }
                )

    shortlist = (
        pd.DataFrame(shortlist_rows)
        .sort_values(["shortlist_score", "table2_score", "table1_score"], ascending=[False, False, False])
        .head(6)
        .reset_index(drop=True)
    )
    shortlist.to_csv(run_dir / "paper_table_shortlist.csv", index=False)

    rows: list[dict[str, object]] = []
    for _, candidate in shortlist.iterrows():
        gap_fill_method = str(candidate["gap_fill_method"])
        outlier_threshold = float(candidate["outlier_zscore_threshold"])
        volatility_window = int(candidate["volatility_window"])
        dataset_cfg = replace(
            base.dataset,
            gap_fill_method=gap_fill_method,
            outlier_zscore_threshold=outlier_threshold,
        )
        combo_dir = run_dir / f"hmm_{gap_fill_method}_z{str(outlier_threshold).replace('.', '_')}_vw{volatility_window}"
        combo_dir.mkdir(parents=True, exist_ok=True)
        _, _, modeling_5m, _, _ = prepare_dataset(dataset_cfg, combo_dir)
        feature_frame = zscore_features(build_feature_frame(modeling_5m, volatility_window=volatility_window))
        kmeans_run = fit_kmeans_with_stride(
            feature_frame,
            n_clusters=base.main_k,
            random_seed=base.random_seed,
            train_stride=base.kmeans_train_stride,
        )
        _, cluster_stats = map_clusters_to_regimes(feature_frame, kmeans_run.labels)
        table1 = _table1_descriptive_stats(modeling_5m, feature_frame)
        table2 = _table2_cluster_centroids(cluster_stats)

        for hmm_stride in hmm_strides:
            hmm_result = fit_hmm_variant(
                feature_frame,
                model_name="kmeans_initialized_hmm",
                n_states=base.n_states,
                covariance_type=base.covariance_type,
                max_iterations=base.max_iterations,
                random_seed=base.random_seed,
                train_stride=hmm_stride,
                initial_labels=kmeans_run.labels.to_numpy(dtype=int),
            )
            table3 = _table3_from_hmm(hmm_result)
            scores = score_tables(table1, table2, table3)
            rows.append(
                {
                    "gap_fill_method": gap_fill_method,
                    "outlier_zscore_threshold": outlier_threshold,
                    "volatility_window": volatility_window,
                    "hmm_train_stride": hmm_stride,
                    **scores,
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["overall_score", "table2_score", "table3_score", "table1_score"],
        ascending=[False, False, False, False],
    )
    results.to_csv(run_dir / "paper_table_tuning_results.csv", index=False)
    print(run_dir)
    print(results.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
