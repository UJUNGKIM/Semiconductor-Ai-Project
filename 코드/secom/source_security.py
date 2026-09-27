"""Normalize and validate Bandit source-security evidence for SECOM."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


SCANNER_NAME = "bandit"
SCANNER_VERSION = "1.9.4"
MINIMUM_SEVERITY = "MEDIUM"
MINIMUM_CONFIDENCE = "MEDIUM"
SCAN_TARGETS = ("app.py", "dashboard_ui", "코드/secom")
_LEVELS = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def target_source_files(project: Path) -> list[Path]:
    files = [project / "app.py"]
    files.extend(sorted((project / "dashboard_ui").rglob("*.py")))
    files.extend(sorted((project / "코드" / "secom").rglob("*.py")))
    return [path for path in files if path.is_file()]


def source_fingerprint(project: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    files = target_source_files(project)
    for path in files:
        relative = path.relative_to(project).as_posix()
        normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
    return len(files), digest.hexdigest().upper()


def _relative_filename(value: object, project: Path) -> str:
    text = str(value).replace("\\", "/")
    candidate = Path(str(value))
    if candidate.is_absolute():
        try:
            text = candidate.resolve().relative_to(project.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("검사 결과에 프로젝트 외부 경로가 포함됐습니다.") from exc
    text = text.lstrip("./")
    if (
        text != "app.py"
        and not text.startswith("dashboard_ui/")
        and not text.startswith("코드/secom/")
    ):
        raise ValueError(f"승인되지 않은 정적분석 대상입니다: {text}")
    return text


def summarize_bandit(
    raw_report: Mapping[str, object],
    *,
    project: Path,
    scanner_version: str,
    scanned_at: str,
) -> dict:
    raw_results = raw_report.get("results")
    raw_errors = raw_report.get("errors")
    metrics = raw_report.get("metrics")
    if not isinstance(raw_results, list) or not isinstance(raw_errors, list):
        raise ValueError("Bandit JSON 결과 또는 오류 배열이 없습니다.")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get("_totals"), Mapping):
        raise ValueError("Bandit JSON 전체 지표가 없습니다.")

    issues: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    for item in raw_results:
        if not isinstance(item, Mapping):
            raise ValueError("Bandit 이슈 형식이 올바르지 않습니다.")
        filename = _relative_filename(item.get("filename", ""), project)
        line_number = int(item.get("line_number", 0))
        test_id = str(item.get("test_id", "")).strip()
        severity = str(item.get("issue_severity", "")).upper()
        confidence = str(item.get("issue_confidence", "")).upper()
        if line_number < 1 or not test_id or severity not in _LEVELS or confidence not in _LEVELS:
            raise ValueError("Bandit 이슈 필수 필드가 올바르지 않습니다.")
        if _LEVELS[severity] < _LEVELS[MINIMUM_SEVERITY]:
            raise ValueError("설정한 심각도보다 낮은 이슈가 결과에 포함됐습니다.")
        if _LEVELS[confidence] < _LEVELS[MINIMUM_CONFIDENCE]:
            raise ValueError("설정한 신뢰도보다 낮은 이슈가 결과에 포함됐습니다.")
        key = (filename, line_number, test_id)
        if key in seen:
            continue
        seen.add(key)
        cwe = item.get("issue_cwe")
        cwe_id = int(cwe.get("id", 0)) if isinstance(cwe, Mapping) else 0
        issues.append(
            {
                "file": filename,
                "line": line_number,
                "test_id": test_id,
                "test_name": str(item.get("test_name", "")),
                "severity": severity,
                "confidence": confidence,
                "cwe_id": cwe_id,
                "more_info": str(item.get("more_info", "")),
            }
        )
    issues.sort(key=lambda row: (row["file"], row["line"], row["test_id"]))

    totals = metrics["_totals"]
    nosec_count = int(totals.get("nosec", 0))
    error_count = len(raw_errors)
    scanned_file_count, fingerprint = source_fingerprint(project)
    status = "PASS" if not issues and error_count == 0 and nosec_count == 0 else "BLOCK"
    return {
        "report_version": 1,
        "status": status,
        "scanner": SCANNER_NAME,
        "scanner_version": scanner_version,
        "scanned_at": scanned_at,
        "scope": "secom_and_integrated_dashboard_python_source",
        "targets": list(SCAN_TARGETS),
        "minimum_severity": MINIMUM_SEVERITY,
        "minimum_confidence": MINIMUM_CONFIDENCE,
        "scanned_file_count": scanned_file_count,
        "source_sha256": fingerprint,
        "lines_of_code": int(totals.get("loc", 0)),
        "suppression_count": nosec_count,
        "scanner_error_count": error_count,
        "issue_count": len(issues),
        "issues": issues,
        "source_code_snippets_persisted": False,
        "limitations": [
            "Python AST 기반 규칙 검사이며 실행 시점의 모든 취약 동작을 증명하지 않습니다.",
            "SECOM 코드와 통합 app.py·dashboard_ui만 대상으로 하며 WM-811K 코드는 별도 작업 범위입니다.",
            "중간 이상 심각도·신뢰도만 릴리스 차단 대상으로 사용합니다.",
        ],
    }


def validate_source_security_report(report: Mapping[str, object], project: Path) -> dict:
    if int(report.get("report_version", 0)) != 1:
        raise ValueError("지원하지 않는 소스 보안 보고서 버전입니다.")
    if report.get("scanner") != SCANNER_NAME or report.get("scanner_version") != SCANNER_VERSION:
        raise ValueError("승인된 Bandit 스캐너와 버전이 아닙니다.")
    if report.get("scope") != "secom_and_integrated_dashboard_python_source":
        raise ValueError("정적분석 범위가 올바르지 않습니다.")
    if report.get("targets") != list(SCAN_TARGETS):
        raise ValueError("정적분석 대상이 고정 계약과 다릅니다.")
    if report.get("minimum_severity") != MINIMUM_SEVERITY:
        raise ValueError("정적분석 최소 심각도가 다릅니다.")
    if report.get("minimum_confidence") != MINIMUM_CONFIDENCE:
        raise ValueError("정적분석 최소 신뢰도가 다릅니다.")
    file_count, fingerprint = source_fingerprint(project)
    if int(report.get("scanned_file_count", 0)) != file_count:
        raise ValueError("정적분석 파일 수가 현재 소스와 다릅니다.")
    if report.get("source_sha256") != fingerprint:
        raise ValueError("정적분석 소스 해시가 현재 소스와 다릅니다.")
    issues = report.get("issues")
    if not isinstance(issues, list) or int(report.get("issue_count", -1)) != len(issues):
        raise ValueError("정적분석 이슈 집계가 상세 기록과 다릅니다.")
    suppression_count = int(report.get("suppression_count", -1))
    error_count = int(report.get("scanner_error_count", -1))
    expected = "PASS" if not issues and suppression_count == 0 and error_count == 0 else "BLOCK"
    if report.get("status") != expected:
        raise ValueError("정적분석 결과와 판정이 일치하지 않습니다.")
    if report.get("source_code_snippets_persisted") is not False:
        raise ValueError("소스 코드 조각 저장 계약이 올바르지 않습니다.")
    limitations = report.get("limitations")
    if not isinstance(limitations, list) or len(limitations) < 3:
        raise ValueError("정적분석 한계가 충분히 기록되지 않았습니다.")
    return {
        "valid": True,
        "status": expected,
        "issue_count": len(issues),
        "scanned_file_count": file_count,
    }
