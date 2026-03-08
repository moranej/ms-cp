from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from .metrics import format_summary_text
from .utils import dump_json

def save_per_sample_csv(df: pd.DataFrame, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "per_sample.csv"
    df.to_csv(path, index=False)
    return path


def save_summary(summary: Dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    dump_json(summary, output_dir / "summary.json")
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write(format_summary_text(summary))


def save_config(config: Dict[str, Any], output_dir: str | Path) -> None:
    dump_json(config, Path(output_dir) / "config.json")


def save_calibration(calibration: Dict[str, Any], output_dir: str | Path) -> None:
    dump_json(calibration, Path(output_dir) / "calibration.json")
