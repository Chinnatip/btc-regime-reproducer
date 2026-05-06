from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass
class HMMResult:
    model_name: str
    regime_frame: pd.DataFrame
    model_transition_matrix: pd.DataFrame
    empirical_transition_matrix: pd.DataFrame
    metrics: dict[str, object]
    model: GaussianHMM


def _state_name_mapping(feature_frame: pd.DataFrame, states: np.ndarray) -> tuple[dict[int, str], pd.DataFrame]:
    tmp = feature_frame[["log_return", "rolling_volatility"]].copy()
    tmp["state"] = states
    state_stats = (
        tmp.groupby("state")
        .agg(
            mean_return=("log_return", "mean"),
            mean_volatility=("rolling_volatility", "mean"),
            count=("state", "size"),
        )
        .sort_values(["mean_return", "mean_volatility"])
    )
    order = state_stats.index.tolist()
    mapping: dict[int, str] = {}
    if order:
        mapping[int(order[0])] = "bearish"
    if len(order) >= 2:
        mapping[int(order[-1])] = "bullish"
    for state in order[1:-1]:
        mapping[int(state)] = "sideways"
    if len(order) == 2:
        unmapped = [int(state) for state in order if int(state) not in mapping]
        for state in unmapped:
            mapping[state] = "sideways"
    state_stats["mapped_regime"] = state_stats.index.map(lambda value: mapping[int(value)])
    return mapping, state_stats


def build_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    labels = list(dict.fromkeys(regimes.astype(str).tolist()))
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)
    pairs = pd.DataFrame({"current": regimes.astype(str), "next": regimes.astype(str).shift(-1)}).dropna()
    if pairs.empty:
        return matrix
    counts = pairs.groupby(["current", "next"]).size()
    for (current, nxt), value in counts.items():
        matrix.loc[current, nxt] = float(value)
    row_sums = matrix.sum(axis=1).replace(0.0, np.nan)
    matrix = matrix.div(row_sums, axis=0).fillna(0.0)
    return matrix


def build_named_model_transition_matrix(model: GaussianHMM, mapping: dict[int, str]) -> pd.DataFrame:
    labels = ["bullish", "bearish", "sideways"]
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)
    raw = pd.DataFrame(model.transmat_)
    for from_state in raw.index:
        from_regime = mapping.get(int(from_state), "sideways")
        for to_state in raw.columns:
            to_regime = mapping.get(int(to_state), "sideways")
            matrix.loc[from_regime, to_regime] += float(raw.loc[from_state, to_state])
    row_sums = matrix.sum(axis=1).replace(0.0, np.nan)
    return matrix.div(row_sums, axis=0).fillna(0.0)


def _seeded_hmm(
    x: np.ndarray,
    initial_labels: np.ndarray | None,
    n_states: int,
    covariance_type: str,
    max_iterations: int,
    random_seed: int,
) -> GaussianHMM:
    if initial_labels is None:
        return GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=max_iterations,
            random_state=random_seed,
        )

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=max_iterations,
        random_state=random_seed,
        init_params="",
        params="stmc",
    )

    label_series = pd.Series(initial_labels)
    startprob = np.full(n_states, 1e-6, dtype=float)
    first_state = int(label_series.iloc[0])
    startprob[first_state] += 1.0
    startprob = startprob / startprob.sum()

    trans_counts = np.full((n_states, n_states), 1e-3, dtype=float)
    for current, nxt in zip(initial_labels[:-1], initial_labels[1:]):
        trans_counts[int(current), int(nxt)] += 1.0
    transmat = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    means = []
    covars = []
    for state in range(n_states):
        mask = initial_labels == state
        subset = x[mask]
        if len(subset) == 0:
            means.append(np.zeros(x.shape[1], dtype=float))
            covars.append(np.ones(x.shape[1], dtype=float))
            continue
        means.append(subset.mean(axis=0))
        var = subset.var(axis=0)
        var[var <= 1e-6] = 1e-6
        covars.append(var)

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = np.vstack(means)
    model.covars_ = np.vstack(covars)
    return model


def fit_hmm_variant(
    feature_frame: pd.DataFrame,
    model_name: str,
    n_states: int,
    covariance_type: str,
    max_iterations: int,
    random_seed: int,
    train_stride: int = 1,
    initial_labels: np.ndarray | None = None,
) -> HMMResult:
    x = feature_frame[["log_return_z", "rolling_volatility_z"]].to_numpy(dtype=float)
    stride = max(int(train_stride), 1)
    feature_frame_train = feature_frame.iloc[::stride].copy()
    x_train = x[::stride]
    initial_labels_train = initial_labels[::stride] if initial_labels is not None else None
    model = _seeded_hmm(
        x=x_train,
        initial_labels=initial_labels_train,
        n_states=n_states,
        covariance_type=covariance_type,
        max_iterations=max_iterations,
        random_seed=random_seed,
    )
    model.fit(x_train)

    if stride == 1:
        states = model.predict(x)
        mapping, state_stats = _state_name_mapping(feature_frame, states)
        scoring_x = x
        score_on_train_only = False
    else:
        train_states = model.predict(x_train)
        mapping, state_stats = _state_name_mapping(feature_frame_train, train_states)
        train_state_series = pd.Series(train_states, index=feature_frame_train.index, name="hmm_state")
        expanded_states = train_state_series.reindex(feature_frame.index, method="ffill")
        if expanded_states.isna().any():
            expanded_states = expanded_states.bfill()
        states = expanded_states.astype(int).to_numpy()
        scoring_x = x_train
        score_on_train_only = True

    regime_frame = feature_frame.copy()
    regime_frame["hmm_state"] = states
    regime_frame["regime"] = pd.Series(states, index=feature_frame.index).map(mapping).fillna("sideways")
    empirical_transitions = build_transition_matrix(regime_frame["regime"])
    model_transitions = build_named_model_transition_matrix(model, mapping)

    n_samples, n_features = x.shape
    free_params = (n_states - 1) + n_states * (n_states - 1) + (n_states * n_features) + (n_states * n_features)
    scoring_samples = len(scoring_x)
    log_likelihood = float(model.score(scoring_x))
    bic = float(-2.0 * log_likelihood + free_params * np.log(max(scoring_samples, 1)))
    monitor = getattr(model, "monitor_", None)
    persistence = {
        label: float(model_transitions.loc[label, label]) if label in model_transitions.index and label in model_transitions.columns else 0.0
        for label in model_transitions.index
    }
    metrics = {
        "model_name": model_name,
        "n_samples": int(n_samples),
        "train_samples": int(len(x_train)),
        "train_stride": int(stride),
        "likelihood_scored_on_train_sequence_only": bool(score_on_train_only),
        "n_features": int(n_features),
        "n_states": int(n_states),
        "covariance_type": covariance_type,
        "log_likelihood": log_likelihood,
        "bic": bic,
        "iterations": int(len(getattr(monitor, "history", []))) if monitor is not None else 0,
        "converged": bool(getattr(monitor, "converged", False)) if monitor is not None else False,
        "persistence": persistence,
        "state_summary": state_stats.reset_index().to_dict(orient="records"),
    }
    return HMMResult(
        model_name=model_name,
        regime_frame=regime_frame,
        model_transition_matrix=model_transitions,
        empirical_transition_matrix=empirical_transitions,
        metrics=metrics,
        model=model,
    )
