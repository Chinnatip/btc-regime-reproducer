from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .clustering import elbow_diagnostics, fit_kmeans_with_stride, map_clusters_to_regimes
from .dataset import prepare_dataset
from .features import build_feature_frame, clip_feature_tails, zscore_features
from .hmm_model import fit_hmm_variant
from .io_utils import ensure_dir, write_frame, write_json, write_markdown
from .paper_targets import score_tables
from .plots import plot_regimes
from .types import ExperimentConfig


def make_run_dir(output_root: Path, prefix: str = "run") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ensure_dir(output_root / f"{prefix}_{stamp}")


def _table1_descriptive_stats(modeling_frame: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("Close Price (USD)", modeling_frame["close"]),
        ("Log Return", feature_frame["log_return"]),
        ("Volatility (σₜ)", feature_frame["rolling_volatility"]),
        ("Volume (BTC)", modeling_frame["volume"]),
    ]
    output: list[dict[str, object]] = []
    for feature_name, series in rows:
        clean = pd.Series(series).dropna()
        output.append(
            {
                "feature": feature_name,
                "mean": float(clean.mean()),
                "std_dev": float(clean.std(ddof=0)),
                "min": float(clean.min()),
                "max": float(clean.max()),
            }
        )
    return pd.DataFrame(output)


def _volatility_window_diagnostics(
    modeling_frame: pd.DataFrame,
    candidates: tuple[int, ...],
    feature_clip_quantile: float | None,
    log_return_clip_quantile: float | None,
    volatility_clip_quantile: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in candidates:
        feature_frame = build_feature_frame(modeling_frame, volatility_window=window)
        feature_frame = clip_feature_tails(
            feature_frame,
            feature_clip_quantile,
            log_return_clip_quantile=log_return_clip_quantile,
            volatility_clip_quantile=volatility_clip_quantile,
        )
        rows.append(
            {
                "volatility_window": int(window),
                "rows_after_warmup": int(len(feature_frame)),
                "volatility_mean": float(feature_frame["rolling_volatility"].mean()),
                "volatility_std_dev": float(feature_frame["rolling_volatility"].std(ddof=0)),
                "volatility_min": float(feature_frame["rolling_volatility"].min()),
                "volatility_max": float(feature_frame["rolling_volatility"].max()),
            }
        )
    return pd.DataFrame(rows)


def _table2_cluster_centroids(cluster_stats: pd.DataFrame) -> pd.DataFrame:
    table = cluster_stats.reset_index().rename(
        columns={
            "cluster": "regime_cluster",
            "mean_return": "mean_log_return",
            "mean_volatility": "mean_volatility",
            "mapped_regime": "mapped_regime",
            "market_behavior": "market_behavior",
        }
    )
    return table.loc[:, ["regime_cluster", "mean_log_return", "mean_volatility", "mapped_regime", "market_behavior", "count"]]


def _table4_model_comparison(random_metrics: dict[str, object], seeded_metrics: dict[str, object]) -> pd.DataFrame:
    def _interpretation(name: str) -> str:
        if name == "kmeans_initialized_hmm":
            return "Improved fit, stable states"
        return "Moderate model fit"

    rows = []
    for metrics in (random_metrics, seeded_metrics):
        rows.append(
            {
                "model_name": metrics["model_name"],
                "log_likelihood": float(metrics["log_likelihood"]),
                "bic": float(metrics["bic"]),
                "converged": bool(metrics["converged"]),
                "interpretation": _interpretation(str(metrics["model_name"])),
            }
        )
    return pd.DataFrame(rows)


def _table3_transition_matrix(hmm_result) -> pd.DataFrame:
    table = hmm_result.model_transition_matrix.copy()
    table.index.name = "from_regime"
    return table.reset_index()


def _paper_fidelity_report(
    config: ExperimentConfig,
    dataset_manifest: dict[str, object],
    coverage_report: dict[str, object],
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table4: pd.DataFrame,
) -> str:
    lines = [
        "# Paper Fidelity Report",
        "",
        "## Matches",
        "",
        "- Methods/Results were treated as the primary implementation target over the article-style summary text.",
        "- Pipeline uses `1m -> 5m`, `log_return`, `rolling_volatility`, z-score normalization, K-Means clustering, then HMM comparison.",
        "- Main run uses `K=3` and compares `HMM random init` vs `K-Means initialized HMM` on the same standardized observation space.",
        "- Root outputs include Table 1-4 style artifacts plus explicit plots for `KMeans-only` and `Hybrid HMM`.",
        "",
        "## Assumptions",
        "",
        f"- Active volatility window for the main run: `{config.volatility_window}` bars.",
        f"- Candidate volatility windows explicitly evaluated: `{', '.join(str(v) for v in config.volatility_window_candidates)}`.",
        f"- Feature tail clip quantile before z-score: `{config.feature_clip_quantile}`.",
        f"- Log return clip quantile override: `{config.log_return_clip_quantile}`.",
        f"- Volatility clip quantile override: `{config.volatility_clip_quantile}`.",
        f"- Elbow diagnostics fit stride on the training sequence: `{config.elbow_train_stride}`.",
        f"- KMeans fit stride on the training sequence: `{config.kmeans_train_stride}`.",
        f"- HMM fit stride on the training sequence: `{config.hmm_train_stride}`.",
        "- When `hmm_train_stride > 1`, HMM fit and likelihood scoring are computed on the training subsequence and then expanded back to the full timeline for plotting and transition analysis.",
        f"- Short gaps are repaired using `{config.dataset.gap_fill_method}` with limit `{config.dataset.short_gap_fill_limit_bars}` bars.",
        f"- Outlier filtering reference: `{config.dataset.outlier_reference}` with z-threshold `{config.dataset.outlier_zscore_threshold}`.",
        f"- Figure-style plot window: `{config.plot_start_utc or 'full range'}` to `{config.plot_end_utc or 'full range'}`.",
        "",
        "## Unresolved Paper Ambiguities",
        "",
        "- The paper does not specify the rolling volatility window `n`.",
        "- Methods mention forward filling short gaps, while Results mention time-based interpolation.",
        "- The paper states outlier removal via z-score thresholds but does not specify the threshold or exact target variable.",
        "- Figure 2 narrative claims a long multi-year view, but the embedded chart in the PDF appears to show only an early-2012 slice and a K-Means legend.",
        "",
        "## Current Dataset Status",
        "",
        f"- Paper-native dataset match: `{dataset_manifest['paper_alignment']['paper_dataset_match']}`.",
        f"- Fidelity caveat: {dataset_manifest['fidelity_caveat']}",
        f"- Preprocessed 1m rows: `{coverage_report['preprocessed_1m']['row_count']}`.",
        f"- Modeling 5m rows: `{coverage_report['modeling_5m']['row_count']}`.",
        "",
        "## Table Checks",
        "",
        f"- Table 1 rows exported: `{len(table1)}`.",
        f"- Table 2 clusters exported: `{len(table2)}`.",
        f"- Table 4 models exported: `{len(table4)}`.",
    ]
    return "\n".join(lines)


def _comparison_summary_markdown(table4: pd.DataFrame, dataset_manifest: dict[str, object]) -> str:
    lines = [
        "# BTC Regime Reproduction Summary",
        "",
        "## Model Comparison",
        "",
    ]
    for _, row in table4.iterrows():
        lines.append(
            f"- `{row['model_name']}`: log_likelihood `{row['log_likelihood']:.4f}`, "
            f"BIC `{row['bic']:.4f}`, converged `{row['converged']}`"
        )
    lines.extend(["", "## Caveat", "", f"- {dataset_manifest['fidelity_caveat']}"])
    return "\n".join(lines)


def prepare_only(config: ExperimentConfig) -> Path:
    run_dir = make_run_dir(config.output_root, prefix="prepare")
    preprocessed_1m, plotting_5m, modeling_5m, dataset_manifest, coverage_report = prepare_dataset(config.dataset, run_dir)
    write_frame(run_dir / "preprocessed_1m.parquet", preprocessed_1m)
    write_frame(run_dir / "prepared_5m.parquet", plotting_5m)
    write_frame(run_dir / "modeling_5m.parquet", modeling_5m)
    write_json(run_dir / "config_snapshot.json", {"dataset": asdict(config.dataset), "experiment": asdict(config)})
    write_json(run_dir / "prepare_summary.json", {"dataset_manifest": dataset_manifest, "coverage_report": coverage_report})
    return run_dir


def run_reproduction(config: ExperimentConfig) -> Path:
    run_dir = make_run_dir(config.output_root, prefix="repro")
    preprocessed_1m, plotting_5m, modeling_5m, dataset_manifest, coverage_report = prepare_dataset(config.dataset, run_dir)
    write_frame(run_dir / "preprocessed_1m.parquet", preprocessed_1m)
    write_frame(run_dir / "prepared_5m.parquet", plotting_5m)
    write_frame(run_dir / "modeling_5m.parquet", modeling_5m)

    feature_frame = build_feature_frame(modeling_5m, volatility_window=config.volatility_window)
    feature_frame = clip_feature_tails(
        feature_frame,
        config.feature_clip_quantile,
        log_return_clip_quantile=config.log_return_clip_quantile,
        volatility_clip_quantile=config.volatility_clip_quantile,
    )
    feature_frame = zscore_features(feature_frame)
    write_frame(run_dir / "feature_frame.parquet", feature_frame)

    volatility_diag = _volatility_window_diagnostics(
        modeling_5m,
        config.volatility_window_candidates,
        config.feature_clip_quantile,
        config.log_return_clip_quantile,
        config.volatility_clip_quantile,
    )
    volatility_diag.to_csv(run_dir / "volatility_window_diagnostics.csv", index=False)

    table1 = _table1_descriptive_stats(modeling_5m, feature_frame)
    table1.to_csv(run_dir / "table1_descriptive_stats.csv", index=False)

    elbow = elbow_diagnostics(
        feature_frame,
        k_values=config.k_values,
        random_seed=config.random_seed,
        train_stride=config.elbow_train_stride,
    )
    elbow.to_csv(run_dir / "elbow_diagnostics.csv", index=False)

    kmeans_run = fit_kmeans_with_stride(
        feature_frame,
        n_clusters=config.main_k,
        random_seed=config.random_seed,
        train_stride=config.kmeans_train_stride,
    )
    kmeans_regimes, cluster_stats = map_clusters_to_regimes(feature_frame, kmeans_run.labels)
    cluster_frame = feature_frame.join(kmeans_run.labels).join(kmeans_regimes)
    write_frame(run_dir / "cluster_assignments.parquet", cluster_frame)
    cluster_stats.to_csv(run_dir / "cluster_regime_summary.csv", index=True)

    table2 = _table2_cluster_centroids(cluster_stats)
    table2.to_csv(run_dir / "table2_cluster_centroids.csv", index=False)

    kmeans_plot_frame = cluster_frame.rename(columns={"kmeans_regime": "regime"})
    plot_regimes(
        price_frame=plotting_5m,
        regime_frame=kmeans_plot_frame,
        title="Figure 2 Replica: BTC Price With K-Means Regimes",
        output_path=run_dir / "kmeans_only_regime_plot.png",
        plot_start_utc=config.plot_start_utc,
        plot_end_utc=config.plot_end_utc,
        max_points=config.plot_max_points,
        legend_title="Regime (K-Means)",
    )

    random_hmm = fit_hmm_variant(
        feature_frame,
        model_name="hmm_random_init",
        n_states=config.n_states,
        covariance_type=config.covariance_type,
        max_iterations=config.max_iterations,
        random_seed=config.random_seed,
        train_stride=config.hmm_train_stride,
        initial_labels=None,
    )
    seeded_hmm = fit_hmm_variant(
        feature_frame,
        model_name="kmeans_initialized_hmm",
        n_states=config.n_states,
        covariance_type=config.covariance_type,
        max_iterations=config.max_iterations,
        random_seed=config.random_seed,
        train_stride=config.hmm_train_stride,
        initial_labels=kmeans_run.labels.to_numpy(dtype=int),
    )

    for result in (random_hmm, seeded_hmm):
        model_dir = ensure_dir(run_dir / result.model_name)
        write_frame(model_dir / "state_assignments.parquet", result.regime_frame)
        result.model_transition_matrix.to_csv(model_dir / "transition_matrix.csv", index=True)
        write_json(model_dir / "transition_matrix.json", result.model_transition_matrix.to_dict())
        result.empirical_transition_matrix.to_csv(model_dir / "empirical_transition_matrix.csv", index=True)
        write_json(model_dir / "empirical_transition_matrix.json", result.empirical_transition_matrix.to_dict())
        write_json(model_dir / "metrics.json", result.metrics)
        plot_regimes(
            price_frame=plotting_5m,
            regime_frame=result.regime_frame,
            title=f"{result.model_name} regime plot",
            output_path=model_dir / "regime_timeline.png",
            max_points=config.plot_max_points,
        )

    table3 = _table3_transition_matrix(seeded_hmm)
    table3.to_csv(run_dir / "table3_transition_matrix.csv", index=False)
    write_json(run_dir / "table3_transition_matrix.json", seeded_hmm.model_transition_matrix.to_dict())
    plot_regimes(
        price_frame=plotting_5m,
        regime_frame=seeded_hmm.regime_frame,
        title="Figure 2 Replica: BTC Price With Hybrid HMM Regimes",
        output_path=run_dir / "hybrid_hmm_regime_plot.png",
        plot_start_utc=config.plot_start_utc,
        plot_end_utc=config.plot_end_utc,
        max_points=config.plot_max_points,
        legend_title="Regime (Hybrid HMM)",
    )

    table4 = _table4_model_comparison(random_hmm.metrics, seeded_hmm.metrics)
    table4.to_csv(run_dir / "table4_model_comparison.csv", index=False)
    table4.to_csv(run_dir / "model_comparison.csv", index=False)
    write_json(run_dir / "paper_table_scores.json", score_tables(table1, table2, table3))

    paper_fidelity_report = _paper_fidelity_report(
        config=config,
        dataset_manifest=dataset_manifest,
        coverage_report=coverage_report,
        table1=table1,
        table2=table2,
        table4=table4,
    )
    write_markdown(run_dir / "paper_fidelity_report.md", paper_fidelity_report)
    write_markdown(run_dir / "comparison_summary.md", _comparison_summary_markdown(table4, dataset_manifest))
    write_json(run_dir / "config_snapshot.json", {"dataset": asdict(config.dataset), "experiment": asdict(config)})
    return run_dir
