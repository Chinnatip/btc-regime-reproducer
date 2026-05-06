from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_regime_repro.reporting import rebuild_comparison_summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild model comparison markdown for a run directory.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    out_path = rebuild_comparison_summary(Path(args.run_dir).expanduser().resolve())
    print(out_path)


if __name__ == "__main__":
    main()
