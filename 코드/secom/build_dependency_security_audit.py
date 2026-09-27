"""Audit the resolved deployment requirements graph and write limited evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from dependency_security import (
    SCANNER_VERSION,
    summarize_pip_audit,
    validate_dependency_security_report,
)
from environment_provenance import sha256_text_file


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT / "결과물" / "secom" / "dependency_security"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_scanner_version = metadata.version("pip-audit")
    if actual_scanner_version != SCANNER_VERSION:
        raise RuntimeError(
            f"pip-audit 버전 불일치: expected={SCANNER_VERSION}, actual={actual_scanner_version}"
        )
    with tempfile.TemporaryDirectory(prefix="secom-pip-audit-") as temp_dir:
        raw_path = Path(temp_dir) / "pip-audit-raw.json"
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--requirement",
                str(PROJECT / "requirements.txt"),
                "--format=json",
                f"--output={raw_path}",
            ],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if not raw_path.is_file():
            raise RuntimeError(
                "pip-audit가 JSON 증거를 생성하지 못했습니다: "
                f"exit={process.returncode}, stderr={process.stderr.strip()}"
            )
        raw_report = json.loads(raw_path.read_text(encoding="utf-8"))
    report = summarize_pip_audit(
        raw_report,
        scanner_version=actual_scanner_version,
        pip_version=metadata.version("pip"),
        python_version=platform.python_version(),
        requirements_sha256=sha256_text_file(PROJECT / "requirements.txt"),
        audited_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    validation = validate_dependency_security_report(report)
    if process.returncode not in {0, 1}:
        raise RuntimeError(
            f"pip-audit 실행 오류: exit={process.returncode}, stderr={process.stderr.strip()}"
        )
    if (process.returncode == 0) != (report["status"] == "PASS"):
        raise RuntimeError("pip-audit 종료 코드와 정규화 보고서 판정이 다릅니다.")
    report_path = output_dir / "dependency_security_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        "# SECOM Python 의존성 취약점 감사\n\n"
        f"- 판정: **{report['status']}**\n"
        f"- 스캐너: pip-audit {report['scanner_version']}\n"
        f"- advisory service: {report['advisory_service']}\n"
        f"- 검사 범위: requirements.txt 해석 의존성 그래프\n"
        f"- 검사 distribution: {report['audited_distribution_count']}개\n"
        f"- 알려진 취약점: {report['known_vulnerability_count']}건\n"
        f"- pip: {report['pip_version']}\n"
        f"- Python: {report['python_version']}\n"
        f"- requirements SHA-256: {report['requirements_sha256']}\n"
        f"- 검사 시각: {report['audited_at']}\n\n"
        "이 결과는 검사 시점의 Python 패키지 advisory 조회이며 운영체제·드라이버·CUDA·"
        "네이티브 라이브러리 및 미공개 취약점까지 안전하다는 보증이 아닙니다.\n",
        encoding="utf-8",
    )
    print(
        f"dependency audit: status={validation['status']} "
        f"packages={validation['audited_distribution_count']} "
        f"vulnerabilities={validation['known_vulnerability_count']}"
    )
    if report["status"] != "PASS":
        raise RuntimeError("알려진 Python 의존성 취약점이 발견되어 릴리스를 차단합니다.")


if __name__ == "__main__":
    main()
