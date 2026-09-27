"""Operating-system-independent verification entry point for GitHub Actions."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SECOM_CODE = PROJECT / "코드" / "secom"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(SECOM_CODE) not in sys.path:
    sys.path.insert(0, str(SECOM_CODE))

from environment_provenance import validate_environment_manifest
from release_readiness import build_release_readiness


REQUIRED_IMPORTS = (
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "catboost",
    "lightgbm",
    "torch",
    "shap",
    "dice_ml",
    "joblib",
    "matplotlib",
    "scipy",
    "imblearn",
    "streamlit",
    "authlib",
    "plotly",
    "ctgan",
    "xlsxwriter",
)


LOCAL_ONLY_TESTS = {
    (
        "test_wm811k_robust_flat_bias.Wm811kRobustFlatBiasTests."
        "test_received_bundle_selects_registered_bias"
    ): "Git에서 제외된 robust-flat Colab prediction ZIP 필요",
    (
        "test_wm811k_multiseed_preparation.Wm811kMultiseedPreparationTests."
        "test_received_bundle_and_imported_evidence"
    ): "Git에서 제외된 원본 multi-seed Colab ZIP 필요",
    (
        "test_wm811k_weak_class_analysis.Wm811kWeakClassTests."
        "test_result_contract_and_source_provenance"
    ): "Git에서 제외된 strategy comparison 원본 폴더 필요",
}


def run_checked(command: list[str], label: str) -> None:
    print(f"[CI] {label}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} 실패 (exit code: {completed.returncode})")


def verify_imports() -> None:
    print("[CI] 필수 라이브러리 import", flush=True)
    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)
    print(f"[CI] import 완료: {len(REQUIRED_IMPORTS)}개", flush=True)


def verify_secom_raw_contract() -> None:
    raw_dir = PROJECT / "데이터" / "SECOM 데이터셋" / "raw"
    sensor_path = raw_dir / "secom.data"
    label_path = raw_dir / "secom_labels.data"
    sensors = pd.read_csv(sensor_path, sep=r"\s+", header=None)
    labels = pd.read_csv(label_path, sep=r"\s+", header=None)
    if sensors.shape != (1567, 590):
        raise ValueError(f"SECOM 센서 계약 불일치: {sensors.shape}")
    if labels.shape != (1567, 2):
        raise ValueError(f"SECOM 라벨 계약 불일치: {labels.shape}")
    mapped = labels.iloc[:, 0].map({-1: 0, 1: 1})
    if mapped.isna().any() or mapped.value_counts().to_dict() != {0: 1463, 1: 104}:
        raise ValueError("SECOM 라벨 분포 또는 -1/1 매핑 계약 불일치")
    print("[CI] SECOM 원본 계약: 1567행 × 590변수, 정상 1463, 불량 104", flush=True)


def verify_release_evidence() -> None:
    provenance_dir = PROJECT / "결과물" / "secom" / "environment_provenance"
    environment_manifest = json.loads(
        (provenance_dir / "environment_manifest.json").read_text(encoding="utf-8")
    )
    validate_environment_manifest(environment_manifest, PROJECT)
    report, manifest = build_release_readiness(PROJECT)
    if report["demo_readiness"] == "BLOCKED":
        raise ValueError("SECOM 대회 데모 릴리스 증거가 BLOCKED입니다.")
    expected_production_blockers = {
        "SENSOR_SEMANTICS",
        "PROSPECTIVE_EXTERNAL_VALIDATION",
        "PERSISTENT_AUDIT_STORE",
    }
    actual_production_blockers = {
        item["check_id"] for item in report["checks"] if item["status"] == "BLOCK"
    }
    if actual_production_blockers != expected_production_blockers:
        raise ValueError(
            "생산 배포 차단 계약 불일치: "
            f"expected={sorted(expected_production_blockers)}, "
            f"actual={sorted(actual_production_blockers)}"
        )
    print(
        f"[CI] 릴리스 증거: {manifest['release_id']} · "
        f"demo={report['demo_readiness']} · production={report['production_readiness']}",
        flush=True,
    )


def _flatten_tests(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten_tests(item)
        else:
            yield item


def run_clean_clone_tests() -> None:
    print("[CI] clean-clone 회귀 테스트", flush=True)
    discovered = unittest.defaultTestLoader.discover(
        str(PROJECT / "테스트"), pattern="test_*.py"
    )
    selected = unittest.TestSuite()
    excluded = {}
    for test in _flatten_tests(discovered):
        test_id = test.id()
        if test_id in LOCAL_ONLY_TESTS:
            excluded[test_id] = LOCAL_ONLY_TESTS[test_id]
        else:
            selected.addTest(test)
    missing_exclusions = set(LOCAL_ONLY_TESTS) - set(excluded)
    if missing_exclusions:
        raise RuntimeError(f"CI 로컬 전용 테스트 ID가 바뀌었습니다: {missing_exclusions}")
    for test_id, reason in sorted(excluded.items()):
        print(f"[CI][LOCAL-ONLY 제외] {test_id}: {reason}", flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(selected)
    if not result.wasSuccessful():
        raise RuntimeError(
            f"clean-clone 회귀 테스트 실패: failures={len(result.failures)}, "
            f"errors={len(result.errors)}"
        )
    print(
        f"[CI] clean-clone 테스트 통과: {result.testsRun}개 · "
        f"로컬 전용 제외: {len(excluded)}개",
        flush=True,
    )


def main() -> None:
    print(f"[CI] Python {sys.version.split()[0]}", flush=True)
    if sys.version_info[:3] != (3, 11, 9):
        raise RuntimeError(f"Python 버전 계약 불일치: {sys.version.split()[0]}")
    verify_imports()
    run_checked([sys.executable, "-m", "pip", "check"], "pip dependency check")
    verify_secom_raw_contract()
    verify_release_evidence()
    run_clean_clone_tests()
    print("[CI] 모든 검증을 통과했습니다.", flush=True)


if __name__ == "__main__":
    main()
