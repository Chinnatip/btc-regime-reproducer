from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_regime_repro.plots import plot_regimes


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
    else:
        frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render BTC price overlays for both HMM variants in a completed run.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    price_frame = _load_frame(run_dir / "prepared_5m.parquet")
    baseline = _load_frame(run_dir / "hmm_random_init" / "state_assignments.parquet")
    hybrid = _load_frame(run_dir / "kmeans_initialized_hmm" / "state_assignments.parquet")

    hybrid_path = run_dir / "btc_price_overlay_hybrid.png"
    baseline_path = run_dir / "btc_price_overlay_baseline.png"

    plot_regimes(
        price_frame=price_frame,
        regime_frame=hybrid,
        title="BTC 5m Price With Hybrid HMM Regime Overlay",
        output_path=hybrid_path,
    )
    plot_regimes(
        price_frame=price_frame,
        regime_frame=baseline,
        title="BTC 5m Price With Baseline Random-Init HMM Overlay",
        output_path=baseline_path,
    )

    print(hybrid_path)
    print(baseline_path)


if __name__ == "__main__":
    main()
