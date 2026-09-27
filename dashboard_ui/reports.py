"""Downloadable dashboard report bundles."""

from __future__ import annotations

from io import BytesIO
import json
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def build_diagnosis_bundle(
    *,
    results: pd.DataFrame,
    review_queue: pd.DataFrame | None,
    ood_results: pd.DataFrame | None,
    metadata: dict[str, Any],
    html_report: bytes | str | None = None,
) -> bytes:
    """Create one audit-friendly ZIP with tables, metadata, and an optional report."""
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("diagnosis_results.csv", _csv_bytes(results))
        if review_queue is not None:
            archive.writestr("review_queue.csv", _csv_bytes(review_queue))
        if ood_results is not None:
            archive.writestr("ood_results.csv", _csv_bytes(ood_results))
        archive.writestr(
            "run_metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        )
        if html_report is not None:
            payload = html_report.encode("utf-8") if isinstance(html_report, str) else html_report
            archive.writestr("diagnosis_report.html", payload)
    return buffer.getvalue()

