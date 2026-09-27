"""Integrity-checked model registry helpers for the SECOM finalists."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import joblib


MODEL_ROLES = {"ACTIVE", "VERIFIED_STANDBY"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text_file(path: Path) -> str:
    """Hash UTF-8 text with canonical LF line endings for cross-platform use."""
    text_value = path.read_text(encoding="utf-8-sig")
    canonical = text_value.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def canonical_hash(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _safe_artifact_path(project: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("모델 artifact_path는 프로젝트 내부 상대경로여야 합니다.")
    project = project.resolve()
    artifact = (project / relative_path).resolve()
    if artifact != project and project not in artifact.parents:
        raise ValueError("모델 artifact_path가 프로젝트 범위를 벗어납니다.")
    return artifact


def registry_entry(
    project: Path,
    *,
    model_name: str,
    role: str,
    artifact_path: str,
) -> dict:
    if role not in MODEL_ROLES:
        raise ValueError(f"허용되지 않은 모델 역할입니다: {role}")
    path = _safe_artifact_path(project, artifact_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    bundle = joblib.load(path)
    required = {
        "model",
        "imputer",
        "selector",
        "config_id",
        "primary_threshold",
        "input_features",
        "selected_features",
        "label_mapping",
        "random_state",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"{model_name} bundle 필수 항목 누락: {missing}")
    threshold = float(bundle["primary_threshold"])
    if not 0 < threshold < 1:
        raise ValueError(f"{model_name} 임계값이 0~1 범위가 아닙니다.")
    return {
        "model_name": model_name,
        "role": role,
        "artifact_path": Path(artifact_path).as_posix(),
        "artifact_sha256": sha256_file(path),
        "artifact_size_bytes": path.stat().st_size,
        "config_id": str(bundle["config_id"]),
        "primary_threshold": threshold,
        "input_feature_count": len(bundle["input_features"]),
        "selected_feature_count": len(bundle["selected_features"]),
        "input_schema_sha256": canonical_hash(list(bundle["input_features"])),
        "label_mapping_sha256": canonical_hash(bundle["label_mapping"]),
        "random_state": int(bundle["random_state"]),
    }


def validate_registry(registry: Mapping, project: Path) -> dict:
    if int(registry.get("registry_version", 0)) != 1:
        raise ValueError("지원하지 않는 모델 레지스트리 버전입니다.")
    if registry.get("automatic_failover_allowed") is not False:
        raise ValueError("SECOM 대기 모델 자동 전환은 허용되지 않습니다.")
    entries = registry.get("models")
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("레지스트리에는 운영·대기 모델 두 개가 필요합니다.")
    names = [str(entry.get("model_name")) for entry in entries]
    roles = [str(entry.get("role")) for entry in entries]
    if len(set(names)) != 2 or set(roles) != MODEL_ROLES:
        raise ValueError("모델 이름 또는 ACTIVE/VERIFIED_STANDBY 역할이 올바르지 않습니다.")
    verified = []
    for entry in entries:
        path = _safe_artifact_path(project, str(entry.get("artifact_path", "")))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != str(entry.get("artifact_sha256", "")).upper():
            raise ValueError(f"{entry.get('model_name')} 모델 SHA-256이 일치하지 않습니다.")
        actual = registry_entry(
            project,
            model_name=str(entry["model_name"]),
            role=str(entry["role"]),
            artifact_path=str(entry["artifact_path"]),
        )
        for key in (
            "artifact_size_bytes",
            "config_id",
            "primary_threshold",
            "input_feature_count",
            "selected_feature_count",
            "input_schema_sha256",
            "label_mapping_sha256",
            "random_state",
        ):
            if actual[key] != entry.get(key):
                raise ValueError(
                    f"{entry.get('model_name')} 레지스트리 메타데이터 불일치: {key}"
                )
        verified.append(actual)
    if len({entry["input_schema_sha256"] for entry in verified}) != 1:
        raise ValueError("운영 모델과 대기 모델의 입력 스키마가 다릅니다.")
    if len({entry["label_mapping_sha256"] for entry in verified}) != 1:
        raise ValueError("운영 모델과 대기 모델의 라벨 매핑이 다릅니다.")
    return {
        "valid": True,
        "model_count": len(verified),
        "active_model": next(
            entry["model_name"] for entry in verified if entry["role"] == "ACTIVE"
        ),
        "standby_model": next(
            entry["model_name"]
            for entry in verified
            if entry["role"] == "VERIFIED_STANDBY"
        ),
        "input_schema_compatible": True,
        "label_mapping_compatible": True,
    }


def simulated_corruption_is_detected(path: Path, expected_hash: str) -> bool:
    """Test digest failure in memory without modifying the real model artifact."""
    original = path.read_bytes()
    corrupted_digest = hashlib.sha256(original + b"SECOM_CORRUPTION_DRILL").hexdigest().upper()
    return corrupted_digest != expected_hash.upper() and sha256_file(path) == expected_hash.upper()


def load_verified_bundles(registry_path: Path, project: Path) -> dict[str, dict]:
    """Verify every artifact hash and contract before deserializing for inference."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry, project)
    bundles = {}
    for entry in registry["models"]:
        path = _safe_artifact_path(project, str(entry["artifact_path"]))
        # validate_registry checked the digest before any load of this artifact.
        bundles[str(entry["model_name"])] = joblib.load(path)
    return bundles
