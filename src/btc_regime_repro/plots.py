from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


REGIME_COLORS = {
    "bullish": "#9ed89e",
    "bearish": "#f3a6a6",
    "sideways": "#e5e7eb",
}
PRICE_LINE_COLOR = "#1f2937"


def _utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _background_legend_items() -> list[Patch]:
    return [
        Patch(facecolor=REGIME_COLORS["bullish"], edgecolor="none", alpha=0.35, label="Bullish"),
        Patch(facecolor=REGIME_COLORS["bearish"], edgecolor="none", alpha=0.35, label="Bearish"),
        Patch(facecolor=REGIME_COLORS["sideways"], edgecolor="none", alpha=0.6, label="Sideways"),
    ]


def _downsample_for_plot(frame: pd.DataFrame, max_points: int = 50000) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    stride = max(len(frame) // max_points, 1)
    sampled = frame.iloc[::stride].copy()
    if sampled.index[-1] != frame.index[-1]:
        sampled = pd.concat([sampled, frame.iloc[[-1]]]).sort_index()
    return sampled[~sampled.index.duplicated(keep="last")]


def _plot_price_line(ax: plt.Axes, frame: pd.DataFrame, max_points: int) -> None:
    sampled = _downsample_for_plot(frame.loc[:, ["close"]], max_points=max_points)
    ax.plot(sampled.index, sampled["close"], color=PRICE_LINE_COLOR, lw=1.0, alpha=0.95, zorder=3)


def _plot_regime_background(ax: plt.Axes, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    spans = frame.loc[:, ["regime"]].copy()
    spans["group"] = spans["regime"].astype(str).ne(spans["regime"].astype(str).shift()).cumsum()
    default_delta = frame.index.to_series().diff().median()
    if pd.isna(default_delta) or default_delta <= pd.Timedelta(0):
        default_delta = pd.Timedelta(minutes=5)
    for _, group in spans.groupby("group"):
        start = group.index[0]
        end = group.index[-1] + default_delta
        regime = str(group["regime"].iloc[0]).lower()
        ax.axvspan(
            start,
            end,
            facecolor=REGIME_COLORS.get(regime, "#e5e7eb"),
            alpha=0.35 if regime != "sideways" else 0.6,
            lw=0,
            zorder=1,
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
    _plot_regime_background(ax, joined)
    _plot_price_line(ax, joined, max_points=max_points)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time")
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.legend(handles=_background_legend_items(), loc="upper right", ncol=3, framealpha=0.95, title=legend_title, fontsize=8, title_fontsize=8)

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
