"""Normalize and validate point-in-time Python dependency vulnerability evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

from packaging.utils import canonicalize_name
from packaging.version import Version


SCANNER_NAME = "pip-audit"
SCANNER_VERSION = "2.10.1"
MINIMUM_SAFE_PIP_VERSION = Version("26.2.1")


def summarize_pip_audit(
    raw_report: Mapping[str, object],
    *,
    scanner_version: str,
    pip_version: str,
    python_version: str,
    requirements_sha256: str,
    audited_at: str,
) -> dict:
    dependencies = raw_report.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("pip-audit JSON에 dependencies 배열이 없습니다.")
    normalized: dict[tuple[str, str, str], dict] = {}
    for dependency in dependencies:
        if not isinstance(dependency, Mapping):
            raise ValueError("pip-audit dependency 항목 형식이 올바르지 않습니다.")
        name = canonicalize_name(str(dependency.get("name", "")))
        version = str(dependency.get("version", ""))
        if not name or not version:
            raise ValueError("pip-audit dependency 이름 또는 버전이 비어 있습니다.")
        vulnerabilities = dependency.get("vulns", [])
        if not isinstance(vulnerabilities, list):
            raise ValueError("pip-audit vulns 항목은 배열이어야 합니다.")
        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, Mapping):
                raise ValueError("pip-audit vulnerability 항목 형식이 올바르지 않습니다.")
            vulnerability_id = str(vulnerability.get("id", "")).strip()
            if not vulnerability_id:
                raise ValueError("취약점 ID가 비어 있습니다.")
            key = (name, version, vulnerability_id)
            item = normalized.setdefault(
                key,
                {
                    "package": name,
                    "version": version,
                    "vulnerability_id": vulnerability_id,
                    "aliases": set(),
                    "fix_versions": set(),
                },
            )
            item["aliases"].update(
                str(value) for value in vulnerability.get("aliases", []) if str(value)
            )
            item["fix_versions"].update(
                str(value) for value in vulnerability.get("fix_versions", []) if str(value)
            )
    vulnerability_records = []
    for key in sorted(normalized):
        item = normalized[key]
        vulnerability_records.append(
            {
                "package": item["package"],
                "version": item["version"],
                "vulnerability_id": item["vulnerability_id"],
                "aliases": sorted(item["aliases"]),
                "fix_versions": sorted(item["fix_versions"], key=Version),
            }
        )
    vulnerable_packages = sorted({item["package"] for item in vulnerability_records})
    return {
        "report_version": 1,
        "status": "PASS" if not vulnerability_records else "BLOCK",
        "scanner": SCANNER_NAME,
        "scanner_version": scanner_version,
        "advisory_service": "pypi",
        "audited_at": audited_at,
        "scope": "resolved_requirements_graph",
        "input_manifest": "requirements.txt",
        "pip_version": pip_version,
        "python_version": python_version,
        "requirements_sha256": requirements_sha256,
        "audited_distribution_count": len(dependencies),
        "vulnerable_package_count": len(vulnerable_packages),
        "known_vulnerability_count": len(vulnerability_records),
        "vulnerable_packages": vulnerable_packages,
        "vulnerabilities": vulnerability_records,
        "network_registry_resolution_performed": True,
        "full_package_inventory_persisted": False,
        "raw_advisory_descriptions_persisted": False,
        "limitations": [
            "검사 시점에 PyPI advisory service가 알고 있는 Python distribution만 확인합니다.",
            "운영체제·드라이버·CUDA·네이티브 라이브러리 취약점은 이 보고서 범위가 아닙니다.",
            "미공개 취약점과 검사 이후 새로 등록된 advisory가 없음을 보증하지 않습니다.",
        ],
    }


def validate_dependency_security_report(
    report: Mapping[str, object], *, minimum_distributions: int = 17
) -> dict:
    if int(report.get("report_version", 0)) != 1:
        raise ValueError("지원하지 않는 의존성 보안 보고서 버전입니다.")
    if report.get("scanner") != SCANNER_NAME:
        raise ValueError("승인된 취약점 스캐너가 아닙니다.")
    if report.get("scanner_version") != SCANNER_VERSION:
        raise ValueError("취약점 스캐너 버전이 고정 계약과 다릅니다.")
    if report.get("advisory_service") != "pypi":
        raise ValueError("취약점 advisory service가 고정 계약과 다릅니다.")
    if report.get("scope") != "resolved_requirements_graph":
        raise ValueError("취약점 검사 범위가 올바르지 않습니다.")
    if report.get("input_manifest") != "requirements.txt":
        raise ValueError("취약점 검사 입력 manifest가 올바르지 않습니다.")
    requirements_hash = str(report.get("requirements_sha256", ""))
    if len(requirements_hash) != 64 or any(
        character not in "0123456789ABCDEF" for character in requirements_hash
    ):
        raise ValueError("requirements.txt 연결 해시가 올바르지 않습니다.")
    if not str(report.get("python_version", "")):
        raise ValueError("취약점 검사 Python 버전이 없습니다.")
    if report.get("network_registry_resolution_performed") is not True:
        raise ValueError("온라인 advisory 조회 증거가 없습니다.")
    if report.get("full_package_inventory_persisted") is not False:
        raise ValueError("전체 패키지 목록 저장 여부를 과장해서는 안 됩니다.")
    if report.get("raw_advisory_descriptions_persisted") is not False:
        raise ValueError("원문 advisory 설명 저장 계약이 올바르지 않습니다.")
    if int(report.get("audited_distribution_count", 0)) < minimum_distributions:
        raise ValueError("취약점 검사 패키지 수가 너무 적습니다.")
    if Version(str(report.get("pip_version", "0"))) < MINIMUM_SAFE_PIP_VERSION:
        raise ValueError("pip 자체가 승인된 최소 보안 버전보다 낮습니다.")
    audited_at = datetime.fromisoformat(str(report.get("audited_at", "")).replace("Z", "+00:00"))
    if audited_at.tzinfo is None:
        raise ValueError("취약점 검사 시각에는 시간대가 필요합니다.")
    vulnerabilities = report.get("vulnerabilities")
    vulnerable_packages = report.get("vulnerable_packages")
    if not isinstance(vulnerabilities, list) or not isinstance(vulnerable_packages, list):
        raise ValueError("취약점 집계 배열이 없습니다.")
    count = int(report.get("known_vulnerability_count", -1))
    package_count = int(report.get("vulnerable_package_count", -1))
    if count != len(vulnerabilities) or package_count != len(vulnerable_packages):
        raise ValueError("취약점 집계 수가 상세 기록과 다릅니다.")
    expected_status = "PASS" if count == 0 else "BLOCK"
    if report.get("status") != expected_status:
        raise ValueError("취약점 수와 보고서 판정이 일치하지 않습니다.")
    if count == 0 and vulnerable_packages:
        raise ValueError("취약점 0건 보고서에 취약 패키지가 기록됐습니다.")
    limitations = report.get("limitations")
    if not isinstance(limitations, list) or len(limitations) < 3:
        raise ValueError("취약점 검사의 한계가 충분히 기록되지 않았습니다.")
    return {
        "valid": True,
        "status": expected_status,
        "known_vulnerability_count": count,
        "audited_distribution_count": int(report["audited_distribution_count"]),
    }
