"""Compare recorded text-file hashes without hiding newline-only differences.

Evidence bundles store the SHA-256 of the exact bytes they read. A CSV that was
read on Windows with CRLF line endings therefore has a different raw hash from
the same CSV checked out with LF endings. This module keeps the two questions
apart: whether the bytes are identical, and whether the CSV rows and fields are
identical once only the newline convention is canonicalized.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path


CANONICALIZATION = "UTF-8 bytes with CRLF and CR converted to LF"
NEWLINE_ONLY_WARNING = (
    "줄바꿈 차이는 있으나 논리적 내용은 일치합니다. 기록된 원시 SHA-256과 "
    "현재 원시 SHA-256은 다릅니다."
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_newline_bytes(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def csv_rows(raw: bytes) -> list[list[str]]:
    """Parse CSV rows exactly, keeping any newline inside quoted fields."""
    return list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))


def text_hashes(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    return {
        "raw_sha256": sha256_bytes(raw),
        "canonical_sha256": sha256_bytes(canonical_newline_bytes(raw)),
        "canonicalization": CANONICALIZATION,
    }


def compare_recorded_text_hash(
    path: Path,
    recorded_raw_sha256: str | None,
    recorded_canonical_sha256: str | None = None,
) -> dict[str, object]:
    """Compare a current CSV with the hashes an earlier run recorded.

    ``exact_byte_hash_match`` is True only when the current bytes are the
    recorded bytes. ``canonical_content_match`` is True when the recorded hash
    belongs to the current file rewritten with LF or CRLF endings and parsing
    both byte strings yields identical CSV rows and fields. Anything else is a
    mismatch and must fail validation.
    """
    raw = path.read_bytes()
    canonical = canonical_newline_bytes(raw)
    result: dict[str, object] = {
        "file_name": path.name,
        "current_raw_sha256": sha256_bytes(raw),
        "current_canonical_sha256": sha256_bytes(canonical),
        "canonicalization": CANONICALIZATION,
    }
    if recorded_raw_sha256 is None and recorded_canonical_sha256 is None:
        return result

    recorded_raw = None if recorded_raw_sha256 is None else recorded_raw_sha256.lower()
    recorded_canonical = (
        None if recorded_canonical_sha256 is None else recorded_canonical_sha256.lower()
    )
    exact = recorded_raw is not None and recorded_raw == result["current_raw_sha256"]
    recorded_canonical_consistent = (
        recorded_canonical is None
        or recorded_canonical == result["current_canonical_sha256"]
    )
    variant_name = None
    logical_rows_identical = None
    if exact:
        logical_rows_identical = True
    elif recorded_raw is not None:
        variants = {"lf": canonical, "crlf": canonical.replace(b"\n", b"\r\n")}
        for name, candidate in variants.items():
            if candidate != raw and sha256_bytes(candidate) == recorded_raw:
                variant_name = name
                logical_rows_identical = csv_rows(raw) == csv_rows(candidate)
                break
        else:
            logical_rows_identical = False
    canonical_match = bool(
        recorded_canonical_consistent
        and (
            exact
            or logical_rows_identical is True
            or (recorded_raw is None and recorded_canonical is not None)
        )
    )
    status = (
        "exact_match"
        if exact
        else "newline_only_difference"
        if canonical_match
        else "mismatch"
    )
    result.update(
        {
            "recorded_raw_sha256": recorded_raw,
            "recorded_canonical_sha256": recorded_canonical,
            "exact_byte_hash_match": exact,
            "canonical_content_match": canonical_match,
            "recorded_newline_variant": variant_name,
            "logical_rows_identical": logical_rows_identical,
            "status": status,
        }
    )
    if status == "newline_only_difference":
        result["warning"] = NEWLINE_ONLY_WARNING
    return result
