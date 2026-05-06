from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


REGIME_COLORS = {
    "bullish": "#6baed6",
    "bearish": "#f2b134",
    "sideways": "#74c476",
}


def _utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _line_legend_items() -> list[Line2D]:
    return [
        Line2D([0], [0], color=REGIME_COLORS["bullish"], lw=1.8, label="Bullish"),
        Line2D([0], [0], color=REGIME_COLORS["bearish"], lw=1.8, label="Bearish"),
        Line2D([0], [0], color=REGIME_COLORS["sideways"], lw=1.8, label="Sideways"),
    ]


def _downsample_for_plot(frame: pd.DataFrame, max_points: int = 50000) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    stride = max(len(frame) // max_points, 1)
    sampled = frame.iloc[::stride].copy()
    if sampled.index[-1] != frame.index[-1]:
        sampled = pd.concat([sampled, frame.iloc[[-1]]]).sort_index()
    return sampled[~sampled.index.duplicated(keep="last")]


def _plot_regime_segments(ax: plt.Axes, frame: pd.DataFrame, max_points: int) -> None:
    frame = _downsample_for_plot(frame, max_points=max_points)
    segments = frame.loc[:, ["close", "regime"]].copy()
    segments["next_close"] = segments["close"].shift(-1)
    segments["next_time"] = segments.index.to_series().shift(-1)
    segments = segments.dropna(subset=["next_close", "next_time"])
    for ts, row in segments.iterrows():
        ax.plot(
            [ts, row["next_time"]],
            [float(row["close"]), float(row["next_close"])],
            color=REGIME_COLORS.get(str(row["regime"]).lower(), "#6b7280"),
            lw=0.9,
            alpha=0.95,
        )


def plot_regimes(
    price_frame: pd.DataFrame,
    regime_frame: pd.DataFrame,
    title: str,
    output_path: Path,
    plot_start_utc: str | None = None,
    plot_end_utc: str | None = None,
    max_points: int = 50000,
    legend_title: str = "Regime",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    aligned = price_frame.loc[regime_frame.index.min() : regime_frame.index.max()]
    joined = aligned.join(regime_frame[["regime"]], how="inner")
    if plot_start_utc is not None:
        joined = joined.loc[_utc_timestamp(plot_start_utc) :]
    if plot_end_utc is not None:
        joined = joined.loc[: _utc_timestamp(plot_end_utc)]
    _plot_regime_segments(ax, joined, max_points=max_points)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time")
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.legend(handles=_line_legend_items(), loc="upper right", ncol=3, framealpha=0.95, title=legend_title, fontsize=8, title_fontsize=8)

    span_days = max((joined.index.max() - joined.index.min()).days, 1)
    if span_days > 365 * 2:
        locator = mdates.YearLocator()
    elif span_days > 120:
        locator = mdates.MonthLocator(interval=3)
    else:
        locator = mdates.AutoDateLocator(minticks=6, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
