"""Dependency and runtime provenance helpers for the SECOM application."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
from typing import Mapping

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text_file(path: Path) -> str:
    """Hash UTF-8 text after canonical LF normalization across operating systems."""
    text_value = path.read_text(encoding="utf-8-sig")
    canonical = text_value.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def parse_pinned_requirements(path: Path) -> dict[str, dict]:
    requirements = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        specs = list(requirement.specifier)
        if requirement.url or requirement.marker or requirement.extras:
            raise ValueError(f"requirements.txt {line_number}행은 단순 exact pin이어야 합니다.")
        if len(specs) != 1 or specs[0].operator != "==" or "*" in specs[0].version:
            raise ValueError(f"requirements.txt {line_number}행이 == 버전으로 고정되지 않았습니다.")
        name = canonicalize_name(requirement.name)
        if name in requirements:
            raise ValueError(f"requirements.txt에 중복 패키지가 있습니다: {name}")
        requirements[name] = {
            "name": requirement.name,
            "version": specs[0].version,
            "source_line": line_number,
        }
    if not requirements:
        raise ValueError("requirements.txt가 비어 있습니다.")
    return requirements


def installed_distributions() -> dict[str, dict]:
    installed = {}
    for distribution in metadata.distributions():
        raw_name = distribution.metadata.get("Name") or distribution.name
        if not raw_name:
            continue
        name = canonicalize_name(raw_name)
        item = {
            "name": str(raw_name),
            "version": str(distribution.version),
            "purl": f"pkg:pypi/{name}@{distribution.version}",
        }
        current = installed.get(name)
        if current is not None and current["version"] != item["version"]:
            raise ValueError(f"같은 패키지의 여러 버전이 설치됐습니다: {name}")
        installed[name] = item
    return installed


def dependency_drift(
    requirements: Mapping[str, Mapping], installed: Mapping[str, Mapping]
) -> dict:
    missing = []
    mismatched = []
    for name, requirement in requirements.items():
        current = installed.get(name)
        if current is None:
            missing.append(name)
        elif str(current.get("version")) != str(requirement.get("version")):
            mismatched.append(
                {
                    "package": name,
                    "required": str(requirement.get("version")),
                    "installed": str(current.get("version")),
                }
            )
    return {
        "valid": not missing and not mismatched,
        "missing": sorted(missing),
        "mismatched": sorted(mismatched, key=lambda item: item["package"]),
    }


def validate_environment_manifest(manifest: Mapping, project: Path) -> dict:
    if int(manifest.get("manifest_version", 0)) != 1:
        raise ValueError("지원하지 않는 실행환경 manifest 버전입니다.")
    if manifest.get("vulnerability_scan_performed") is not False:
        raise ValueError("취약점 스캔 여부를 과장해서는 안 됩니다.")
    requirements_path = project / "requirements.txt"
    if manifest.get("text_hash_normalization") != "UTF-8-with-canonical-LF":
        raise ValueError("텍스트 해시 정규화 계약이 없거나 지원되지 않습니다.")
    if sha256_text_file(requirements_path) != str(manifest.get("requirements_sha256", "")):
        raise ValueError("requirements.txt SHA-256이 manifest와 다릅니다.")
    requirements = parse_pinned_requirements(requirements_path)
    direct = manifest.get("direct_dependencies", [])
    manifest_direct = {
        canonicalize_name(str(item["name"])): {
            "version": str(item["version"]),
        }
        for item in direct
    }
    drift = dependency_drift(requirements, manifest_direct)
    if not drift["valid"]:
        raise ValueError(f"직접 의존성 manifest가 requirements와 다릅니다: {drift}")
    project_files = manifest.get("project_files", [])
    for item in project_files:
        path = (project / str(item.get("path", ""))).resolve()
        if project.resolve() not in path.parents:
            raise ValueError("실행환경 파일 경로가 프로젝트 범위를 벗어납니다.")
        if not path.is_file() or sha256_text_file(path) != str(item.get("sha256", "")):
            raise ValueError(
                f"실행환경 파일 무결성 불일치: {item.get('path')}. "
                "의존성 변경이 없는 코드 수정이라면 "
                "build_environment_provenance.py --refresh-project-hashes-only를 실행하세요."
            )
    return {
        "valid": True,
        "direct_dependency_count": len(direct),
        "project_file_count": len(project_files),
        "vulnerability_scan_performed": False,
    }


def environment_fingerprint(python_version: str, direct_dependencies: list[dict]) -> str:
    payload = {
        "python_version": python_version,
        "direct_dependencies": sorted(
            [
                {"name": canonicalize_name(item["name"]), "version": item["version"]}
                for item in direct_dependencies
            ],
            key=lambda item: item["name"],
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
