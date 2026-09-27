"""Build Linux/Python 3.11 locks on an actual Ubuntu runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib

from dependency_lock import inspect_pylock, validate_lock_against_installed
from environment_provenance import installed_distributions, sha256_text_file


PROJECT = Path(__file__).resolve().parents[2]
PYPI_HOSTS = ("files.pythonhosted.org",)
CI_HOSTS = (
    "files.pythonhosted.org",
    "download.pytorch.org",
    "download-r2.pytorch.org",
)


def _seed_constraints(seed_lock: Path, destination: Path) -> None:
    """Convert a committed PEP 751 lock into resolver constraints.

    A lock verification run must not silently adopt a transitive release that
    appeared after the committed environment was tested.  Explicit refreshes
    bypass this helper and intentionally resolve the newest compatible graph.
    """
    data = tomllib.loads(seed_lock.read_text(encoding="utf-8-sig"))
    lines = []
    for package in data.get("packages", []):
        name = str(package.get("name", "")).strip()
        version = str(package.get("version", "")).strip()
        if not name or not version or any(char.isspace() for char in name + version):
            raise ValueError(f"seed lock package가 올바르지 않습니다: {name!r}")
        lines.append(f"{name}=={version}")
    if not lines:
        raise ValueError(f"seed lock에 package가 없습니다: {seed_lock}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_lock(
    requirements: Path,
    output: Path,
    find_links: str | None = None,
    seed_lock: Path | None = None,
) -> None:
    with TemporaryDirectory(prefix="secom-lock-") as temporary_dir:
        command = [
            sys.executable,
            "-m",
            "pip",
            "lock",
            "--requirement",
            str(requirements),
            "--only-binary=:all:",
            "--output",
            str(output),
            "--disable-pip-version-check",
        ]
        if seed_lock is not None:
            constraints = Path(temporary_dir) / "committed-lock-constraints.txt"
            _seed_constraints(seed_lock, constraints)
            command.extend(["--constraint", str(constraints)])
        if find_links:
            command.extend(["--find-links", find_links])
        completed = subprocess.run(command, cwd=PROJECT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"pip lock 실패: {requirements.name} (exit code: {completed.returncode})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "결과물" / "secom" / "linux_dependency_lock",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="기존 lock 제약을 사용하지 않고 최신 호환 전이 의존성을 다시 해석합니다.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if platform.system() != "Linux" or platform.python_version() != "3.11.9":
        raise RuntimeError("이 builder는 Linux + Python 3.11.9에서만 실행합니다.")
    output_dir.mkdir(parents=True, exist_ok=True)

    ci_requirements = PROJECT / "requirements-linux-ci.txt"
    streamlit_requirements = PROJECT / "requirements.txt"
    ci_lock = output_dir / "pylock.github-actions.toml"
    streamlit_lock = output_dir / "pylock.streamlit.toml"
    ci_seed = None if args.refresh else PROJECT / ci_lock.name
    streamlit_seed = None if args.refresh else PROJECT / streamlit_lock.name
    if not args.refresh and (not ci_seed.is_file() or not streamlit_seed.is_file()):
        raise FileNotFoundError(
            "커밋된 Linux lock이 없습니다. 최초 생성에는 --refresh를 사용하세요."
        )
    print("[1/4] GitHub Actions CPU 의존성 그래프를 잠급니다...")
    _run_lock(
        ci_requirements,
        ci_lock,
        find_links="https://download.pytorch.org/whl/cpu/torch/",
        seed_lock=ci_seed,
    )
    print("[2/4] Streamlit Linux 의존성 그래프를 잠급니다...")
    _run_lock(streamlit_requirements, streamlit_lock, seed_lock=streamlit_seed)

    print("[3/4] Linux wheel URL·SHA-256과 실제 CI 설치 버전을 검증합니다...")
    ci_details = inspect_pylock(
        ci_lock,
        ci_requirements,
        expected_lock_name=ci_lock.name,
        allowed_hosts=CI_HOSTS,
    )
    streamlit_details = inspect_pylock(
        streamlit_lock,
        streamlit_requirements,
        expected_lock_name=streamlit_lock.name,
        allowed_hosts=PYPI_HOSTS,
    )
    installed_check = validate_lock_against_installed(
        ci_details, installed_distributions()
    )
    pip_version = subprocess.check_output(
        [sys.executable, "-m", "pip", "--version"], text=True
    ).split()[1]
    manifest = {
        "manifest_version": 1,
        "status": "PASS",
        "format": "PEP 751 pylock.toml",
        "generator": "pip",
        "generator_version": pip_version,
        "experimental_tooling": True,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "github_actions": {
            "lock_path": ci_lock.name,
            "source_requirements": ci_requirements.name,
            "source_requirements_sha256": sha256_text_file(ci_requirements),
            "lock_sha256": sha256_text_file(ci_lock),
            "package_count": ci_details["package_count"],
            "wheel_count": ci_details["wheel_count"],
            "sdist_count": ci_details["sdist_count"],
            "direct_dependency_count": ci_details["direct_dependency_count"],
            "all_locked_versions_match_tested_environment": installed_check["valid"],
            "allowed_artifact_hosts": list(CI_HOSTS),
        },
        "streamlit": {
            "lock_path": streamlit_lock.name,
            "source_requirements": streamlit_requirements.name,
            "source_requirements_sha256": sha256_text_file(streamlit_requirements),
            "lock_sha256": sha256_text_file(streamlit_lock),
            "package_count": streamlit_details["package_count"],
            "wheel_count": streamlit_details["wheel_count"],
            "sdist_count": streamlit_details["sdist_count"],
            "direct_dependency_count": streamlit_details["direct_dependency_count"],
            "all_locked_versions_match_tested_environment": False,
            "resolution_validated_on_target_platform": True,
            "installation_validated_by_this_step": False,
            "allowed_artifact_hosts": list(PYPI_HOSTS),
        },
        "all_artifacts_hashed": True,
        "only_binary": True,
        "urls_sanitized": True,
        "vulnerability_scan_performed": False,
        "network_registry_resolution_performed": True,
        "limitations": [
            "pip lock은 experimental 기능입니다.",
            "GitHub Actions lock은 Ubuntu CPU 실행환경에서 설치 버전까지 대조했습니다.",
            "Streamlit lock은 Ubuntu에서 해석했지만 이 단계에서 별도 설치하지 않았습니다.",
            "Streamlit Community Cloud가 named pylock을 자동 사용한다는 가정은 하지 않습니다.",
            "wheel hash는 무결성을 검증하지만 취약점 부재를 보증하지 않습니다.",
        ],
    }
    (output_dir / "linux_lock_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        "# SECOM Linux 전이 의존성 잠금\n\n"
        f"- 판정: **{manifest['status']}**\n"
        f"- 대상: Linux {manifest['platform_machine']} / Python {manifest['python_version']}\n"
        f"- GitHub Actions: {ci_details['package_count']} package, 실제 설치 버전 일치\n"
        f"- Streamlit: {streamlit_details['package_count']} package, resolver 검증만 완료\n"
        "- sdist: 0개, 모든 wheel SHA-256 기록\n\n"
        "CI는 커밋된 lock으로 설치합니다. Streamlit lock은 Community Cloud 자동 적용을 "
        "가정하지 않으며 실제 설치 재현 검증 전까지 경고 상태입니다.\n",
        encoding="utf-8",
    )
    print("[4/4] Linux 잠금 bundle을 저장했습니다.")
    print(
        f"Linux lock 완료: CI={ci_details['package_count']} packages, "
        f"Streamlit={streamlit_details['package_count']} packages"
    )


if __name__ == "__main__":
    main()
