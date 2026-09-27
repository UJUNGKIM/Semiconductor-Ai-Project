"""Run pinned Bandit and write normalized SECOM source-security evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from source_security import (
    SCANNER_VERSION,
    summarize_bandit,
    validate_source_security_report,
)


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT / "결과물" / "secom" / "source_security"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_version = metadata.version("bandit")
    if actual_version != SCANNER_VERSION:
        raise RuntimeError(
            f"Bandit 버전 불일치: expected={SCANNER_VERSION}, actual={actual_version}"
        )
    with tempfile.TemporaryDirectory(prefix="secom-bandit-") as temp_dir:
        raw_path = Path(temp_dir) / "bandit-raw.json"
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                "app.py",
                "dashboard_ui",
                "코드/secom",
                "--format=json",
                f"--output={raw_path}",
                "--severity-level=medium",
                "--confidence-level=medium",
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
                "Bandit이 JSON 증거를 생성하지 못했습니다: "
                f"exit={process.returncode}, stderr={process.stderr.strip()}"
            )
        raw_report = json.loads(raw_path.read_text(encoding="utf-8"))

    report = summarize_bandit(
        raw_report,
        project=PROJECT,
        scanner_version=actual_version,
        scanned_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    validation = validate_source_security_report(report, PROJECT)
    report_path = output_dir / "source_security_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        "# SECOM Python 소스 보안 정적분석\n\n"
        f"- 판정: **{report['status']}**\n"
        f"- 스캐너: Bandit {report['scanner_version']}\n"
        f"- 범위: app.py, dashboard_ui, 코드/secom\n"
        f"- 검사 파일: {report['scanned_file_count']}개\n"
        f"- Python LOC: {report['lines_of_code']}\n"
        f"- 중간 이상 이슈: {report['issue_count']}건\n"
        f"- 스캐너 오류: {report['scanner_error_count']}건\n"
        f"- nosec 억제: {report['suppression_count']}건\n"
        f"- 검사 시각: {report['scanned_at']}\n\n"
        "AST 규칙 기반 검사이며 동적 취약점·WM-811K 코드·낮은 심각도 이슈까지 "
        "안전하다는 보증은 아닙니다.\n",
        encoding="utf-8",
    )
    print(
        f"source audit: status={validation['status']} "
        f"files={validation['scanned_file_count']} issues={validation['issue_count']}"
    )
    if process.returncode not in {0, 1}:
        raise RuntimeError(
            f"Bandit 실행 오류: exit={process.returncode}, stderr={process.stderr.strip()}"
        )
    if report["status"] != "PASS":
        raise RuntimeError("SECOM 소스 보안 정적분석이 릴리스를 차단합니다.")


if __name__ == "__main__":
    main()
