"""Build a local runtime manifest and package SBOM for the SECOM dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import pandas as pd

from environment_provenance import (
    dependency_drift,
    environment_fingerprint,
    installed_distributions,
    parse_pinned_requirements,
    sha256_text_file,
    validate_environment_manifest,
)


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "결과물" / "secom" / "environment_provenance"
# Runtime and security-decision code only; tests, notebooks, and generated
# JSON/CSV evidence are never listed. `코드/secom` modules are integrity-checked
# through the source-security fingerprint instead, so every `코드/wm811k` module
# the dashboard imports at runtime must be listed here. The site_pages files
# are executed by st.navigation rather than imported, so they are listed by hand.
PROJECT_FILE_NAMES = (
    ".github/workflows/project-ci.yml",
    "requirements.txt",
    "app.py",
    "dashboard_ui/__init__.py",
    "dashboard_ui/access.py",
    "dashboard_ui/brand.py",
    "dashboard_ui/components.py",
    "dashboard_ui/independent_validation.py",
    "dashboard_ui/navigation.py",
    "dashboard_ui/reports.py",
    "dashboard_ui/secom.py",
    "dashboard_ui/site.py",
    "dashboard_ui/wafer_view.py",
    "dashboard_ui/wm_gallery.py",
    "dashboard_ui/site_pages/admin.py",
    "dashboard_ui/site_pages/home.py",
    "dashboard_ui/site_pages/intro.py",
    "dashboard_ui/site_pages/project.py",
    "dashboard_ui/site_pages/secom.py",
    "dashboard_ui/site_pages/wm811k.py",
    "코드/verify_ci.py",
    "코드/secom/build_environment_provenance.py",
    "코드/secom/environment_provenance.py",
    "코드/wm811k/diagnose_wm811k.py",
    "코드/wm811k/hash_evidence.py",
    "코드/wm811k/train_wm811k_cnn.py",
    "코드/wm811k/validate_wm811k_gradient_shap_results.py",
    "코드/wm811k/wafer_shape.py",
    "코드/wm811k/wafer_cause_guidance.py",
    "코드/wm811k/wm811k_monitoring.py",
    "run_tests.ps1",
    "run_pipeline.ps1",
)


def build_project_file_manifest(project: Path = PROJECT) -> list[dict]:
    """Hash release-critical text files with cross-platform line normalization."""
    return [
        {
            "path": name,
            "size_bytes": (project / name).stat().st_size,
            "sha256": sha256_text_file(project / name),
        }
        for name in PROJECT_FILE_NAMES
    ]


def pre_deserialization_hash_gate_declared(project: Path = PROJECT) -> bool:
    app_text = (project / "app.py").read_text(encoding="utf-8")
    return all(
        token in app_text
        for token in (
            "load_verified_bundles",
            "expected_registry_hash",
            "고급 진단 참조 SHA-256이 릴리스 manifest와 다릅니다.",
        )
    )


def refresh_project_hashes() -> None:
    """Refresh source hashes without claiming a new installed-environment snapshot."""
    manifest_path = OUTPUT_DIR / "environment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "기존 environment_manifest.json이 없습니다. 먼저 전체 실행환경 증거를 생성하세요."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["project_files"] = build_project_file_manifest()
    manifest["pre_deserialization_hash_gate_declared"] = (
        pre_deserialization_hash_gate_declared()
    )
    validate_environment_manifest(manifest, PROJECT)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "프로젝트 파일 해시 갱신 완료: "
        f"files={len(manifest['project_files'])}, environment_snapshot=preserved"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-project-hashes-only",
        action="store_true",
        help=(
            "설치 환경·SBOM 스냅샷은 유지하고 코드·워크플로 파일 SHA-256만 갱신합니다. "
            "requirements.txt 또는 설치 환경을 바꾼 경우에는 사용하지 마세요."
        ),
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh_project_hashes_only:
        refresh_project_hashes()
        return
    requirements_path = PROJECT / "requirements.txt"
    print("[1/4] exact-pinned 직접 의존성과 설치 버전을 대조합니다...")
    requirements = parse_pinned_requirements(requirements_path)
    installed = installed_distributions()
    drift = dependency_drift(requirements, installed)
    direct = []
    for canonical_name, requirement in sorted(requirements.items()):
        current = installed.get(canonical_name)
        direct.append(
            {
                "name": requirement["name"],
                "canonical_name": canonical_name,
                "version": requirement["version"],
                "installed_version": None if current is None else current["version"],
                "exact_match": current is not None
                and current["version"] == requirement["version"],
                "purl": f"pkg:pypi/{canonical_name}@{requirement['version']}",
            }
        )

    print("[2/4] pip 의존성 일관성과 버전 이탈 탐지를 검사합니다...")
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    simulated = {name: dict(item) for name, item in installed.items()}
    first_name = sorted(requirements)[0]
    simulated[first_name] = {**simulated[first_name], "version": "0.0.0-drift"}
    simulated_drift = dependency_drift(requirements, simulated)
    drift_detection_passed = (
        not simulated_drift["valid"]
        and any(item["package"] == first_name for item in simulated_drift["mismatched"])
    )

    print("[3/4] 실행 스크립트와 사전 역직렬화 해시 게이트를 기록합니다...")
    project_files = build_project_file_manifest()
    pre_deserialization_gate = pre_deserialization_hash_gate_declared()
    python_version = platform.python_version()
    manifest = {
        "manifest_version": 1,
        "python_version": python_version,
        "python_implementation": platform.python_implementation(),
        "platform_snapshot": platform.platform(),
        "requirements_sha256": sha256_text_file(requirements_path),
        "text_hash_normalization": "UTF-8-with-canonical-LF",
        "environment_fingerprint_sha256": environment_fingerprint(
            python_version, direct
        ),
        "direct_dependencies": direct,
        "project_files": project_files,
        "serialization_formats": ["joblib/pickle", "PyTorch checkpoint"],
        "pre_deserialization_hash_gate_declared": pre_deserialization_gate,
        "vulnerability_scan_performed": False,
        "network_registry_resolution_performed": False,
    }
    manifest_check = validate_environment_manifest(manifest, PROJECT)
    checks = {
        "all_direct_dependencies_exactly_pinned": True,
        "all_direct_versions_match_environment": drift["valid"],
        "pip_dependency_check_passed": pip_check.returncode == 0,
        "simulated_version_drift_detected": drift_detection_passed,
        "project_file_hashes_valid": manifest_check["valid"],
        "pre_deserialization_hash_gate_declared": pre_deserialization_gate,
    }
    status = "PASS" if all(checks.values()) else "BLOCK"
    validation = {
        "version": 1,
        "status": status,
        "validation_scope": "local_environment_reproducibility_and_integrity",
        "python_version": python_version,
        "direct_dependency_count": len(direct),
        "installed_distribution_count": len(installed),
        "checks": checks,
        "pip_check_output": (pip_check.stdout + pip_check.stderr).strip(),
        "vulnerability_scan_performed": False,
        "known_vulnerability_count": None,
        "limitations": [
            "이 SBOM은 로컬 설치 스냅샷이며 패키지 취약점 검사를 수행하지 않았습니다.",
            "Windows와 Linux CI는 별도 pylock으로 잠갔지만 Streamlit lock의 실제 설치 적용은 미검증입니다.",
            "해시 기준 manifest 자체는 외부 서명·투명성 로그에 고정되지 않았습니다.",
            "joblib/pickle은 신뢰된 해시의 프로젝트 artifact에만 사용해야 합니다.",
        ],
    }

    print("[4/4] SBOM과 실행환경 검증 보고서를 저장합니다...")
    sbom = {
        "bom_format": "AI-project lightweight Python SBOM",
        "spec_version": "1.0",
        "component_count": len(installed),
        "components": [
            installed[name] for name in sorted(installed)
        ],
        "vulnerability_information_included": False,
    }
    (OUTPUT_DIR / "environment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "environment_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "software_bom.json").write_text(
        json.dumps(sbom, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(direct).to_csv(
        OUTPUT_DIR / "direct_dependencies.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([installed[name] for name in sorted(installed)]).to_csv(
        OUTPUT_DIR / "installed_environment.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "summary.md").write_text(
        "# SECOM 실행환경 재현성·소프트웨어 공급망 검증\n\n"
        f"- 판정: **{status}**\n"
        f"- Python: {python_version}\n"
        f"- exact-pinned 직접 의존성: {len(direct)}개\n"
        f"- 설치 distribution: {len(installed)}개\n"
        f"- pip check: {'통과' if pip_check.returncode == 0 else '실패'}\n"
        f"- 모의 버전 이탈 탐지: {drift_detection_passed}\n"
        f"- 모델 역직렬화 전 해시 게이트: {pre_deserialization_gate}\n"
        "- 취약점 스캔: 수행하지 않음\n\n"
        "이 결과는 현재 실행환경 재현성과 무결성 확인 자료이며 보안 취약점이 없다는 "
        "증명이 아닙니다. Windows와 Linux CI lock은 별도 생성되며 Streamlit 설치 적용, "
        "서명된 artifact와 정기 취약점 스캔은 별도 통제로 유지해야 합니다.\n",
        encoding="utf-8",
    )
    if status != "PASS":
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"실행환경 검증 실패: {failed}; pip={validation['pip_check_output']}")
    print(
        f"실행환경 검증 완료: status={status}, direct={len(direct)}, "
        f"installed={len(installed)}"
    )


if __name__ == "__main__":
    main()
