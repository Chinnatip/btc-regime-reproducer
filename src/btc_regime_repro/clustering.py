from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class KMeansRun:
    labels: pd.Series
    centers: pd.DataFrame
    inertia: float


def fit_kmeans(feature_frame: pd.DataFrame, n_clusters: int, random_seed: int) -> KMeansRun:
    x = feature_frame[["log_return_z", "rolling_volatility_z"]].to_numpy(dtype=float)
    return fit_kmeans_with_stride(feature_frame, n_clusters=n_clusters, random_seed=random_seed, train_stride=1)


def fit_kmeans_with_stride(
    feature_frame: pd.DataFrame,
    n_clusters: int,
    random_seed: int,
    train_stride: int = 1,
) -> KMeansRun:
    x = feature_frame[["log_return_z", "rolling_volatility_z"]].to_numpy(dtype=float)
    stride = max(int(train_stride), 1)
    x_train = x[::stride]
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_seed)
    model.fit(x_train)
    labels = model.predict(x)
    centers = pd.DataFrame(
        model.cluster_centers_,
        columns=["log_return_z", "rolling_volatility_z"],
    )
    return KMeansRun(
        labels=pd.Series(labels, index=feature_frame.index, name="cluster"),
        centers=centers,
        inertia=float(model.inertia_),
    )


def elbow_diagnostics(
    feature_frame: pd.DataFrame,
    k_values: tuple[int, ...],
    random_seed: int,
    train_stride: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for k in k_values:
        run = fit_kmeans_with_stride(feature_frame, n_clusters=k, random_seed=random_seed, train_stride=train_stride)
        rows.append({"k": int(k), "inertia": run.inertia})
    return pd.DataFrame(rows)


def map_clusters_to_regimes(feature_frame: pd.DataFrame, cluster_labels: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    joined = feature_frame.join(cluster_labels)
    stats = (
        joined.groupby("cluster")
        .agg(
            mean_return=("log_return", "mean"),
            mean_volatility=("rolling_volatility", "mean"),
            count=("cluster", "size"),
        )
        .sort_index()
    )
    mapping: dict[int, str] = {}

    bearish_cluster = int(
        stats.sort_values(["mean_return", "mean_volatility"], ascending=[True, False]).index[0]
    )
    bullish_cluster = int(
        stats.sort_values(["mean_return", "mean_volatility"], ascending=[False, True]).index[0]
    )
    mapping[bearish_cluster] = "bearish"
    mapping[bullish_cluster] = "bullish"

    remaining = [int(cluster_id) for cluster_id in stats.index.tolist() if int(cluster_id) not in mapping]
    for cluster_id in remaining:
        mapping[cluster_id] = "sideways"

    regimes = cluster_labels.map(mapping).fillna("sideways").rename("kmeans_regime")
    stats["mapped_regime"] = stats.index.map(mapping)
    stats["market_behavior"] = stats["mapped_regime"].map(
        {
            "bullish": "Bullish Regime (price increase, low volatility)",
            "bearish": "Bearish Regime (price drop, high volatility)",
            "sideways": "Sideways Regime (neutral, stable price movement)",
        }
    )
    return regimes, stats
