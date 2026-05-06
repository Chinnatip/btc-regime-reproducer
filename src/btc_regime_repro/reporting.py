from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import write_markdown


def rebuild_comparison_summary(run_dir: Path) -> Path:
    comparison = pd.read_csv(run_dir / "model_comparison.csv")
    summary_lines = [
        "# Model Comparison",
        "",
        "## Ranking",
        "",
    ]
    best = comparison.sort_values(["bic", "log_likelihood"], ascending=[True, False]).reset_index(drop=True)
    for _, row in best.iterrows():
        summary_lines.append(
            f"- `{row['model_name']}`: log_likelihood `{row['log_likelihood']:.4f}`, "
            f"BIC `{row['bic']:.4f}`, converged `{row['converged']}`"
        )
    summary_lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- This run remains a partial reproduction until the exact BTC/USD 1m 2012-2021 source is swapped in.",
        ]
    )
    out_path = run_dir / "model_comparison_summary.md"
    write_markdown(out_path, "\n".join(summary_lines))
    return out_path
