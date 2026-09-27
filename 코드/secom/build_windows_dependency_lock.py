"""Build a hash-pinned Windows/Python 3.11 PEP 751 lock from the tested venv."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from dependency_lock import (
    inspect_pylock,
    validate_dependency_lock_manifest,
    validate_lock_against_installed,
)
from environment_provenance import installed_distributions, sha256_text_file


PROJECT = Path(__file__).resolve().parents[2]
LOCK_PATH = PROJECT / "pylock.windows.toml"
OUTPUT_DIR = PROJECT / "결과물" / "secom" / "dependency_lock"


def main() -> None:
    python_version = platform.python_version()
    if platform.system() != "Windows" or python_version != "3.11.9":
        raise RuntimeError("이 builder는 Windows + Python 3.11.9 환경에서만 실행합니다.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    installed = installed_distributions()
    constraints = "".join(
        f"{item['name']}=={item['version']}\n"
        for _, item in sorted(installed.items())
    )

    print("[1/4] 현재 검증 환경 버전으로 resolver를 제한합니다...")
    with tempfile.TemporaryDirectory(prefix="secom-lock-") as temp_dir:
        constraints_path = Path(temp_dir) / "constraints.txt"
        constraints_path.write_text(constraints, encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "pip",
            "lock",
            "--requirement",
            str(PROJECT / "requirements.txt"),
            "--constraint",
            str(constraints_path),
            "--only-binary=:all:",
            "--output",
            str(LOCK_PATH),
            "--disable-pip-version-check",
        ]
        completed = subprocess.run(command, cwd=PROJECT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"pip lock 실패 (exit code: {completed.returncode})")

    print("[2/4] PEP 751 구조, wheel URL과 SHA-256을 검증합니다...")
    details = inspect_pylock(LOCK_PATH, PROJECT / "requirements.txt")
    local_check = validate_lock_against_installed(details, installed)
    pip_version = subprocess.check_output(
        [sys.executable, "-m", "pip", "--version"], text=True
    ).split()[1]
    manifest = {
        "manifest_version": 1,
        "status": "PASS",
        "format": "PEP 751 pylock.toml",
        "lock_version": details["lock_version"],
        "generator": details["created_by"],
        "generator_version": pip_version,
        "experimental_tooling": True,
        "lock_path": LOCK_PATH.name,
        "python_version": python_version,
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "requirements_sha256": sha256_text_file(PROJECT / "requirements.txt"),
        "lock_sha256": sha256_text_file(LOCK_PATH),
        "package_count": details["package_count"],
        "wheel_count": details["wheel_count"],
        "sdist_count": details["sdist_count"],
        "direct_dependency_count": details["direct_dependency_count"],
        "all_direct_versions_match": details["all_direct_versions_match"],
        "all_locked_versions_match_tested_environment": local_check[
            "all_locked_versions_match_tested_environment"
        ],
        "all_artifacts_hashed": details["all_artifacts_hashed"],
        "only_binary": details["only_binary"],
        "urls_sanitized": details["urls_sanitized"],
        "vulnerability_scan_performed": False,
        "network_registry_resolution_performed": True,
        "limitations": [
            "이 lock은 Windows x86-64와 Python 3.11.9 조합에만 적용됩니다.",
            "pip lock은 현재 experimental 명령이므로 설치 전 pip 지원 상태를 확인해야 합니다.",
            "Linux Streamlit Cloud와 GitHub Actions에는 별도 플랫폼 lock이 필요합니다.",
            "wheel hash는 무결성을 검증하지만 패키지 취약점 부재를 보증하지 않습니다.",
            "lock과 manifest는 아직 외부 서명 또는 투명성 로그에 고정되지 않았습니다.",
        ],
    }

    print("[3/4] 잠금 manifest의 재검증 가능성을 확인합니다...")
    validate_dependency_lock_manifest(manifest, PROJECT)
    (OUTPUT_DIR / "windows_lock_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "summary.md").write_text(
        "# SECOM Windows 전이 의존성 잠금\n\n"
        f"- 판정: **{manifest['status']}**\n"
        f"- 대상: {manifest['platform_system']} {manifest['platform_machine']} / "
        f"Python {manifest['python_version']}\n"
        f"- 잠금 형식: {manifest['format']} (pip {pip_version}, experimental)\n"
        f"- 전체 package/wheel: {manifest['package_count']}/{manifest['wheel_count']}\n"
        f"- 직접 의존성: {manifest['direct_dependency_count']}개\n"
        "- sdist: 0개, 모든 wheel SHA-256 기록\n"
        "- 현재 테스트 환경과 모든 잠금 버전 일치: True\n\n"
        "이 파일은 Windows/Python 3.11.9용입니다. Linux 배포에는 사용하지 않으며, "
        "취약점 감사와 외부 artifact 서명은 별도 통제입니다.\n",
        encoding="utf-8",
    )
    print("[4/4] Windows 의존성 잠금 증거를 저장했습니다.")
    print(
        f"잠금 완료: packages={manifest['package_count']}, "
        f"wheels={manifest['wheel_count']}, sha256={manifest['lock_sha256'][:16]}..."
    )


if __name__ == "__main__":
    main()
