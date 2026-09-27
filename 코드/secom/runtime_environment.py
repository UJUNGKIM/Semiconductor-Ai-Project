"""Compare the running dashboard environment with the committed Streamlit lock."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Mapping

from dependency_lock import inspect_pylock
from environment_provenance import (
    dependency_drift,
    installed_distributions,
    parse_pinned_requirements,
)


def build_runtime_environment_report(
    project: Path,
    *,
    installed: Mapping[str, Mapping] | None = None,
    python_version: str | None = None,
    platform_system: str | None = None,
    platform_machine: str | None = None,
) -> dict:
    """Build a secrets-free, deterministic runtime-to-lock comparison."""
    requirements_path = project / "requirements.txt"
    lock_path = project / "pylock.streamlit.toml"
    lock = inspect_pylock(
        lock_path,
        requirements_path,
        expected_lock_name="pylock.streamlit.toml",
        allowed_hosts=("files.pythonhosted.org",),
    )
    current = dict(installed) if installed is not None else installed_distributions()
    direct = parse_pinned_requirements(requirements_path)
    direct_drift = dependency_drift(direct, current)
    lock_drift = dependency_drift(lock["locked_packages"], current)
    current_python = python_version or platform.python_version()
    current_system = platform_system or platform.system()
    current_machine = platform_machine or platform.machine()
    expected_python = "3.11.9"
    expected_system = "Linux"
    expected_machine = "x86_64"
    checks = {
        "python_version_matches": current_python == expected_python,
        "platform_system_matches": current_system == expected_system,
        "platform_machine_matches": current_machine == expected_machine,
        "direct_dependencies_match": direct_drift["valid"],
        "locked_dependencies_match": lock_drift["valid"],
    }
    fingerprint_payload = {
        "python_version": current_python,
        "platform_system": current_system,
        "platform_machine": current_machine,
        "locked_versions": {
            name: (
                str(current[name].get("version"))
                if name in current
                else "<missing>"
            )
            for name in sorted(lock["locked_packages"])
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest().upper()
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "WARN",
        "scope": "current_process_package_metadata_only",
        "expected": {
            "python_version": expected_python,
            "platform_system": expected_system,
            "platform_machine": expected_machine,
            "direct_dependency_count": len(direct),
            "locked_dependency_count": lock["package_count"],
        },
        "runtime": {
            "python_version": current_python,
            "platform_system": current_system,
            "platform_machine": current_machine,
        },
        "checks": checks,
        "direct_drift": direct_drift,
        "lock_drift": lock_drift,
        "runtime_fingerprint_sha256": fingerprint,
        "limitations": [
            "설치 metadata와 버전만 비교하며 wheel 파일 자체의 설치 출처는 증명하지 않습니다.",
            "호스트명, 사용자명, 환경변수, API 키와 Streamlit secrets는 수집하지 않습니다.",
            "PASS는 현재 프로세스 재현성 판정이며 모델 성능이나 생산 배포 승인이 아닙니다.",
        ],
    }


def report_json_bytes(report: Mapping) -> bytes:
    return json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
