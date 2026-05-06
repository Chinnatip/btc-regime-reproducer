from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_regime_repro.config import load_config
from btc_regime_repro.pipeline import prepare_only


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare BTC dataset for regime reproduction.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_dir = prepare_only(load_config(args.config))
    print(run_dir)


if __name__ == "__main__":
    main()
