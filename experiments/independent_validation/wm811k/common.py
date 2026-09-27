"""Small shared helpers for the WM-811K independent validation scripts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]
RANDOM_STATE = 42


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "예" if value else "아니오"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame without adding a tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    rows = [[_cell(value) for value in record] for record in frame.itertuples(index=False)]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        if rows
        else len(columns[index])
        for index in range(len(columns))
    ]
    header = "| " + " | ".join(
        column.ljust(width) for column, width in zip(columns, widths)
    ) + " |"
    divider = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    body = [
        "| " + " | ".join(
            cell.ljust(width) for cell, width in zip(row, widths)
        ) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


