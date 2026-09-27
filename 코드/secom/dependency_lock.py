"""Validate the Windows PEP 751 dependency lock used by the SECOM prototype."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from packaging.utils import canonicalize_name

from environment_provenance import (
    dependency_drift,
    parse_pinned_requirements,
    sha256_text_file,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_LOCK_NAME = "pylock.windows.toml"


def inspect_pylock(
    lock_path: Path,
    requirements_path: Path,
    *,
    expected_lock_name: str = EXPECTED_LOCK_NAME,
    allowed_hosts: tuple[str, ...] = ("files.pythonhosted.org",),
) -> dict:
    """Parse and strictly validate a platform-specific PEP 751 lock file."""
    if lock_path.name != expected_lock_name:
        raise ValueError(f"잠금 파일명은 {expected_lock_name}이어야 합니다.")
    data = tomllib.loads(lock_path.read_text(encoding="utf-8-sig"))
    if data.get("lock-version") != "1.0":
        raise ValueError("지원하지 않는 pylock.toml 버전입니다.")
    if data.get("created-by") != "pip":
        raise ValueError("잠금 생성기는 pip이어야 합니다.")
    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ValueError("잠금 파일에 package가 없습니다.")

    locked = {}
    wheel_count = 0
    for package in packages:
        name = canonicalize_name(str(package.get("name", "")))
        version = str(package.get("version", "")).strip()
        if not name or not version:
            raise ValueError("package 이름 또는 버전이 비어 있습니다.")
        if name in locked:
            raise ValueError(f"잠금 파일에 중복 package가 있습니다: {name}")
        if "sdist" in package:
            raise ValueError(f"소스 배포본은 허용하지 않습니다: {name}")
        wheels = package.get("wheels")
        if not isinstance(wheels, list) or not wheels:
            raise ValueError(f"wheel이 없는 package입니다: {name}")
        for wheel in wheels:
            wheel_name = str(wheel.get("name", ""))
            if not wheel_name.endswith(".whl"):
                raise ValueError(f"wheel 파일명이 올바르지 않습니다: {wheel_name}")
            parsed = urlsplit(str(wheel.get("url", "")))
            if (
                parsed.scheme != "https"
                or parsed.hostname not in allowed_hosts
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(f"허용되지 않은 wheel URL입니다: {name}")
            sha256 = str(wheel.get("hashes", {}).get("sha256", ""))
            if not SHA256_PATTERN.fullmatch(sha256):
                raise ValueError(f"유효한 wheel SHA-256이 없습니다: {name}")
            wheel_count += 1
        locked[name] = {"name": name, "version": version}

    requirements = parse_pinned_requirements(requirements_path)
    direct_drift = dependency_drift(requirements, locked)
    if not direct_drift["valid"]:
        raise ValueError(f"직접 의존성 잠금이 requirements.txt와 다릅니다: {direct_drift}")
    return {
        "valid": True,
        "lock_version": data["lock-version"],
        "created_by": data["created-by"],
        "package_count": len(locked),
        "wheel_count": wheel_count,
        "sdist_count": 0,
        "direct_dependency_count": len(requirements),
        "all_direct_versions_match": True,
        "all_artifacts_hashed": True,
        "only_binary": True,
        "urls_sanitized": True,
        "locked_packages": locked,
    }


def validate_lock_against_installed(
    lock_details: Mapping, installed: Mapping[str, Mapping]
) -> dict:
    """Assert that the generated lock describes the locally tested environment."""
    drift = dependency_drift(lock_details["locked_packages"], installed)
    if not drift["valid"]:
        raise ValueError(f"잠금 파일과 현재 테스트 환경 버전이 다릅니다: {drift}")
    return {
        "valid": True,
        "all_locked_versions_match_tested_environment": True,
        "locked_distribution_count": len(lock_details["locked_packages"]),
    }


def validate_dependency_lock_manifest(manifest: Mapping, project: Path) -> dict:
    """Validate committed lock evidence without assuming the current OS is Windows."""
    if int(manifest.get("manifest_version", 0)) != 1:
        raise ValueError("지원하지 않는 dependency lock manifest 버전입니다.")
    lock_path = project / str(manifest.get("lock_path", ""))
    requirements_path = project / "requirements.txt"
    if lock_path.resolve().parent != project.resolve():
        raise ValueError("잠금 파일은 프로젝트 루트에 있어야 합니다.")
    details = inspect_pylock(lock_path, requirements_path)
    if manifest.get("format") != "PEP 751 pylock.toml":
        raise ValueError("잠금 형식 표기가 올바르지 않습니다.")
    if manifest.get("generator") != "pip":
        raise ValueError("잠금 생성기 표기가 올바르지 않습니다.")
    if manifest.get("platform_system") != "Windows":
        raise ValueError("이 잠금 증거는 Windows 전용이어야 합니다.")
    if manifest.get("python_version") != "3.11.9":
        raise ValueError("이 잠금 증거는 Python 3.11.9 전용이어야 합니다.")
    if manifest.get("lock_sha256") != sha256_text_file(lock_path):
        raise ValueError("잠금 파일 SHA-256이 manifest와 다릅니다.")
    if manifest.get("requirements_sha256") != sha256_text_file(requirements_path):
        raise ValueError("requirements.txt SHA-256이 dependency lock manifest와 다릅니다.")
    for field in (
        "package_count",
        "wheel_count",
        "sdist_count",
        "direct_dependency_count",
    ):
        if int(manifest.get(field, -1)) != int(details[field]):
            raise ValueError(f"잠금 통계가 manifest와 다릅니다: {field}")
    required_true = (
        "all_direct_versions_match",
        "all_locked_versions_match_tested_environment",
        "all_artifacts_hashed",
        "only_binary",
        "urls_sanitized",
        "experimental_tooling",
        "network_registry_resolution_performed",
    )
    if any(manifest.get(field) is not True for field in required_true):
        raise ValueError("잠금 보증 또는 적용 범위 표기가 불완전합니다.")
    if manifest.get("vulnerability_scan_performed") is not False:
        raise ValueError("잠금 생성 단계가 취약점 검사를 수행했다고 과장할 수 없습니다.")
    return {**details, "manifest_valid": True}


def validate_linux_dependency_lock_manifest(manifest: Mapping, project: Path) -> dict:
    """Validate Ubuntu CI and Streamlit resolver locks on any host OS."""
    if int(manifest.get("manifest_version", 0)) != 1:
        raise ValueError("지원하지 않는 Linux dependency lock manifest 버전입니다.")
    if manifest.get("status") != "PASS":
        raise ValueError("Linux dependency lock manifest가 PASS가 아닙니다.")
    if manifest.get("format") != "PEP 751 pylock.toml":
        raise ValueError("Linux 잠금 형식 표기가 올바르지 않습니다.")
    if manifest.get("generator") != "pip" or manifest.get("generator_version") != "26.2.1":
        raise ValueError("Linux 잠금 생성기 버전이 고정 계약과 다릅니다.")
    if manifest.get("platform_system") != "Linux" or manifest.get("platform_machine") != "x86_64":
        raise ValueError("Linux 잠금 플랫폼 표기가 올바르지 않습니다.")
    if manifest.get("python_version") != "3.11.9":
        raise ValueError("Linux 잠금 Python 버전이 올바르지 않습니다.")
    expected = {
        "github_actions": {
            "lock": "pylock.github-actions.toml",
            "requirements": "requirements-linux-ci.txt",
            "hosts": (
                "files.pythonhosted.org",
                "download.pytorch.org",
                "download-r2.pytorch.org",
            ),
        },
        "streamlit": {
            "lock": "pylock.streamlit.toml",
            "requirements": "requirements.txt",
            "hosts": ("files.pythonhosted.org",),
        },
    }
    validations = {}
    for target, contract in expected.items():
        item = manifest.get(target, {})
        if item.get("lock_path") != contract["lock"]:
            raise ValueError(f"{target} 잠금 경로가 올바르지 않습니다.")
        if item.get("source_requirements") != contract["requirements"]:
            raise ValueError(f"{target} requirements 경로가 올바르지 않습니다.")
        if item.get("allowed_artifact_hosts") != list(contract["hosts"]):
            raise ValueError(f"{target} 허용 artifact host가 고정 계약과 다릅니다.")
        lock_path = project / contract["lock"]
        requirements_path = project / contract["requirements"]
        details = inspect_pylock(
            lock_path,
            requirements_path,
            expected_lock_name=contract["lock"],
            allowed_hosts=contract["hosts"],
        )
        if item.get("lock_sha256") != sha256_text_file(lock_path):
            raise ValueError(f"{target} 잠금 SHA-256이 manifest와 다릅니다.")
        if item.get("source_requirements_sha256") != sha256_text_file(requirements_path):
            raise ValueError(f"{target} requirements SHA-256이 manifest와 다릅니다.")
        for field in (
            "package_count",
            "wheel_count",
            "sdist_count",
            "direct_dependency_count",
        ):
            if int(item.get(field, -1)) != int(details[field]):
                raise ValueError(f"{target} 잠금 통계가 manifest와 다릅니다: {field}")
        validations[target] = details
    if manifest["github_actions"].get("all_locked_versions_match_tested_environment") is not True:
        raise ValueError("GitHub Actions 잠금의 실제 설치 버전 대조 증거가 없습니다.")
    streamlit = manifest["streamlit"]
    if (
        streamlit.get("resolution_validated_on_target_platform") is not True
        or streamlit.get("installation_validated_by_this_step") is not False
        or streamlit.get("all_locked_versions_match_tested_environment") is not False
    ):
        raise ValueError("Streamlit 잠금의 검증 범위가 정직하게 표시되지 않았습니다.")
    required_true = (
        "all_artifacts_hashed",
        "only_binary",
        "urls_sanitized",
        "experimental_tooling",
        "network_registry_resolution_performed",
    )
    if any(manifest.get(field) is not True for field in required_true):
        raise ValueError("Linux 잠금 보증 표기가 불완전합니다.")
    if manifest.get("vulnerability_scan_performed") is not False:
        raise ValueError("Linux 잠금 생성이 취약점 검사를 수행했다고 과장할 수 없습니다.")
    return {"valid": True, **validations}
