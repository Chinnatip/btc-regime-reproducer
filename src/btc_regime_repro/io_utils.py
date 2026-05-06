from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .types import json_ready


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    if path.suffix == ".csv":
        frame.to_csv(path, index=True)
        return
    frame.to_parquet(path, index=True)
