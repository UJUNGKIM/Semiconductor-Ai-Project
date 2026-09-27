"""Hash-based, raw-value-free data lineage helpers for SECOM artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


CANONICAL_TEXT_SUFFIXES = {".data", ".csv", ".json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_text_bytes(path: Path) -> bytes:
    text_value = path.read_text(encoding="utf-8-sig")
    canonical = text_value.replace("\r\n", "\n").replace("\r", "\n")
    return canonical.encode("utf-8")


def hash_mode_for_path(path: Path) -> str:
    return (
        "UTF-8-with-canonical-LF"
        if path.suffix.lower() in CANONICAL_TEXT_SUFFIXES
        else "raw-bytes"
    )


def artifact_bytes(path: Path, hash_mode: str | None = None) -> bytes:
    mode = hash_mode or hash_mode_for_path(path)
    if mode == "UTF-8-with-canonical-LF":
        return canonical_text_bytes(path)
    if mode == "raw-bytes":
        return path.read_bytes()
    raise ValueError(f"지원하지 않는 계보 hash_mode입니다: {mode}")


def artifact_sha256(path: Path, hash_mode: str | None = None) -> str:
    return hashlib.sha256(artifact_bytes(path, hash_mode)).hexdigest().upper()


def canonical_hash(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def safe_project_path(project: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("계보 artifact_path는 프로젝트 내부 상대경로여야 합니다.")
    project = project.resolve()
    artifact = (project / relative_path).resolve()
    if artifact != project and project not in artifact.parents:
        raise ValueError("계보 artifact_path가 프로젝트 범위를 벗어납니다.")
    return artifact


def build_file_inventory(project: Path, paths: Iterable[tuple[str, str]]) -> list[dict]:
    inventory = []
    seen = set()
    for stage, relative_path in paths:
        normalized = Path(relative_path).as_posix()
        if normalized in seen:
            raise ValueError(f"계보 파일 경로가 중복됐습니다: {normalized}")
        seen.add(normalized)
        path = safe_project_path(project, normalized)
        if not path.is_file():
            raise FileNotFoundError(path)
        hash_mode = hash_mode_for_path(path)
        payload = artifact_bytes(path, hash_mode)
        inventory.append(
            {
                "stage": str(stage),
                "artifact_path": normalized,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest().upper(),
                "hash_mode": hash_mode,
            }
        )
    return inventory


def validate_lineage_manifest(manifest: Mapping, project: Path) -> dict:
    if int(manifest.get("manifest_version", 0)) != 2:
        raise ValueError("지원하지 않는 데이터 계보 manifest 버전입니다.")
    if manifest.get("raw_sensor_values_embedded") is not False:
        raise ValueError("데이터 계보 manifest에는 센서 원본값을 포함할 수 없습니다.")
    inventory = manifest.get("files")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("데이터 계보 파일 목록이 비어 있습니다.")
    seen = set()
    stages = set()
    for item in inventory:
        relative = str(item.get("artifact_path", ""))
        if relative in seen:
            raise ValueError(f"계보 파일 경로가 중복됐습니다: {relative}")
        seen.add(relative)
        stages.add(str(item.get("stage", "")))
        path = safe_project_path(project, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        hash_mode = str(item.get("hash_mode", ""))
        payload = artifact_bytes(path, hash_mode)
        if len(payload) != int(item.get("size_bytes", -1)):
            raise ValueError(f"계보 파일 크기가 바뀌었습니다: {relative}")
        if hashlib.sha256(payload).hexdigest().upper() != str(
            item.get("sha256", "")
        ).upper():
            raise ValueError(f"계보 파일 SHA-256이 일치하지 않습니다: {relative}")
    required_fingerprints = {
        "raw_label_vector_sha256",
        "retained_feature_schema_sha256",
        "train_row_ids_sha256",
        "test_row_ids_sha256",
        "train_median_statistics_sha256",
    }
    fingerprints = manifest.get("fingerprints", {})
    missing = sorted(required_fingerprints - set(fingerprints))
    if missing:
        raise ValueError(f"데이터 계보 fingerprint 누락: {missing}")
    for key in required_fingerprints:
        digest = str(fingerprints[key])
        if len(digest) != 64 or any(char not in "0123456789ABCDEF" for char in digest):
            raise ValueError(f"올바르지 않은 SHA-256 fingerprint: {key}")
    return {
        "valid": True,
        "file_count": len(inventory),
        "stages": sorted(stages),
        "raw_sensor_values_embedded": False,
    }


def simulated_tamper_is_detected(path: Path, expected_hash: str) -> bool:
    original = artifact_bytes(path)
    changed_hash = hashlib.sha256(
        original + b"SECOM_LINEAGE_TAMPER_DRILL"
    ).hexdigest().upper()
    return (
        changed_hash != expected_hash.upper()
        and artifact_sha256(path) == expected_hash.upper()
    )
