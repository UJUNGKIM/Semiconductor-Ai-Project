"""Evidence-based SECOM demo/production release readiness gate."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from external_validation import validated_summary_is_release_evidence
from dependency_lock import (
    validate_dependency_lock_manifest,
    validate_linux_dependency_lock_manifest,
)
from sensor_semantics import dictionary_is_release_evidence
from sensor_failure_containment import validate_sensor_failure_containment
from source_security import source_fingerprint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text_file(path: Path) -> str:
    text_value = path.read_text(encoding="utf-8-sig")
    canonical = text_value.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def sha256_release_component(path: Path) -> str:
    if path.suffix.lower() in {".json", ".csv", ".html", ".md", ".txt", ".toml"}:
        return sha256_text_file(path)
    return sha256_file(path)


def _load_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size < 2:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _check(
    check_id: str,
    title: str,
    status: str,
    evidence: str,
    required_action: str = "",
) -> dict:
    return {
        "check_id": check_id,
        "title": title,
        "status": status,
        "evidence": evidence,
        "required_action": required_action,
    }


def build_release_readiness(project: Path) -> tuple[dict, dict]:
    artifacts = project / "결과물" / "secom"
    processed = project / "데이터" / "SECOM 데이터셋" / "processed"
    model_dir = artifacts / "dual_model_results"
    advanced_dir = artifacts / "advanced_diagnostics"
    paths = {
        "preprocessing_metadata": processed / "preprocessing_metadata.json",
        "median_imputer": processed / "median_imputer.joblib",
        "catboost_model": model_dir / "catboost_model.joblib",
        "xgboost_model": model_dir / "xgboost_model.joblib",
        "advanced_metadata": advanced_dir / "metadata.json",
        "advanced_reference": advanced_dir / "advanced_reference.joblib",
        "safety_policy": artifacts / "safety_policy_results" / "safety_policy_summary.json",
        "robustness": artifacts / "robustness_results" / "robustness_summary.json",
        "batch_gate": artifacts / "batch_safety_gate" / "batch_gate_summary.json",
        "sensor_failure_containment": artifacts / "sensor_failure_containment" / "containment_report.json",
        "feedback_reference": artifacts / "feedback_monitoring" / "feedback_reference.json",
        "promotion_policy": artifacts / "champion_challenger" / "promotion_policy.json",
        "audit_policy": artifacts / "audit_ledger" / "audit_policy.json",
        "audit_verification": artifacts / "audit_ledger" / "audit_ledger_verification.json",
        "tamper_detection": artifacts / "audit_ledger" / "tamper_detection_demo.json",
        "shap_reliability": artifacts / "shap_reliability" / "shap_reliability_summary.json",
        "model_registry": artifacts / "model_failover" / "model_registry.json",
        "failover_drill": artifacts / "model_failover" / "failover_drill.json",
        "data_lineage_manifest": artifacts / "data_lineage" / "data_lineage_manifest.json",
        "data_lineage_validation": artifacts / "data_lineage" / "lineage_validation.json",
        "environment_manifest": artifacts / "environment_provenance" / "environment_manifest.json",
        "environment_validation": artifacts / "environment_provenance" / "environment_validation.json",
        "software_bom": artifacts / "environment_provenance" / "software_bom.json",
        "windows_dependency_lock": project / "pylock.windows.toml",
        "dependency_lock_manifest": artifacts / "dependency_lock" / "windows_lock_manifest.json",
        "linux_ci_dependency_lock": project / "pylock.github-actions.toml",
        "streamlit_dependency_lock": project / "pylock.streamlit.toml",
        "linux_dependency_lock_manifest": artifacts / "linux_dependency_lock" / "linux_lock_manifest.json",
        "dependency_security_audit": artifacts / "dependency_security" / "dependency_security_audit.json",
        "source_security_audit": artifacts / "source_security" / "source_security_audit.json",
        "audit_store_config": artifacts / "production_audit_store" / "store_config.json",
        "audit_store_verification": artifacts / "production_audit_store" / "store_verification.json",
        "external_validation_protocol": artifacts / "external_validation" / "protocol.json",
        "external_validation_status": artifacts / "external_validation" / "status.json",
        "sensor_semantics_schema": artifacts / "sensor_semantics" / "input_schema.json",
        "sensor_semantics_status": artifacts / "sensor_semantics" / "status.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"릴리스 증거 파일 누락: {missing}")
    component_hashes = {
        name: sha256_release_component(path) for name, path in paths.items()
    }
    preprocessing = _load_json(paths["preprocessing_metadata"])
    advanced = _load_json(paths["advanced_metadata"])
    safety = _load_json(paths["safety_policy"])
    robustness = _load_json(paths["robustness"])
    batch_gate = _load_json(paths["batch_gate"])
    sensor_failure_containment = _load_json(paths["sensor_failure_containment"])
    feedback = _load_json(paths["feedback_reference"])
    promotion = _load_json(paths["promotion_policy"])
    audit_policy = _load_json(paths["audit_policy"])
    audit_verification = _load_json(paths["audit_verification"])
    tamper_detection = _load_json(paths["tamper_detection"])
    shap_reliability = _load_json(paths["shap_reliability"])
    model_registry = _load_json(paths["model_registry"])
    failover_drill = _load_json(paths["failover_drill"])
    data_lineage_manifest = _load_json(paths["data_lineage_manifest"])
    data_lineage_validation = _load_json(paths["data_lineage_validation"])
    environment_manifest = _load_json(paths["environment_manifest"])
    environment_validation = _load_json(paths["environment_validation"])
    software_bom = _load_json(paths["software_bom"])
    dependency_lock_manifest = _load_json(paths["dependency_lock_manifest"])
    linux_dependency_lock_manifest = _load_json(paths["linux_dependency_lock_manifest"])
    dependency_security_audit = _load_json(paths["dependency_security_audit"])
    source_security_audit = _load_json(paths["source_security_audit"])
    audit_store_config = _load_json(paths["audit_store_config"])
    audit_store_verification = _load_json(paths["audit_store_verification"])
    external_validation_protocol = _load_json(paths["external_validation_protocol"])
    external_validation_status = _load_json(paths["external_validation_status"])
    sensor_semantics_schema = _load_json(paths["sensor_semantics_schema"])
    sensor_semantics_status = _load_json(paths["sensor_semantics_status"])

    checks = []
    data_ok = (
        preprocessing.get("raw_feature_count") == 590
        and preprocessing.get("retained_feature_count") == 446
        and preprocessing.get("random_state") == 42
        and preprocessing.get("test_size") == 0.2
    )
    checks.append(
        _check(
            "DATA_CONTRACT",
            "데이터·전처리 계약",
            "PASS" if data_ok else "BLOCK",
            "590개 원변수 → 결측/상수 제거 후 446개, 고정 stratified split"
            if data_ok
            else "전처리 메타데이터가 고정 계약과 다름",
            "전처리 파이프라인과 데이터 버전을 재검증" if not data_ok else "",
        )
    )
    dependency_security_ok = (
        dependency_security_audit.get("status") == "PASS"
        and dependency_security_audit.get("scanner") == "pip-audit"
        and dependency_security_audit.get("scanner_version") == "2.10.1"
        and dependency_security_audit.get("advisory_service") == "pypi"
        and dependency_security_audit.get("scope")
        == "resolved_requirements_graph"
        and dependency_security_audit.get("input_manifest") == "requirements.txt"
        and dependency_security_audit.get("python_version")
        == environment_validation.get("python_version")
        and dependency_security_audit.get("requirements_sha256")
        == environment_manifest.get("requirements_sha256")
        and dependency_security_audit.get("network_registry_resolution_performed") is True
        and int(dependency_security_audit.get("audited_distribution_count", 0))
        >= int(environment_validation.get("direct_dependency_count", 0))
        and int(dependency_security_audit.get("known_vulnerability_count", -1)) == 0
        and dependency_security_audit.get("vulnerabilities") == []
    )
    checks.append(
        _check(
            "DEPENDENCY_SECURITY_AUDIT",
            "Python 의존성 취약점 감사",
            "PASS" if dependency_security_ok else "BLOCK",
            (
                f"pip-audit 2.10.1·PyPI advisory로 "
                f"{dependency_security_audit.get('audited_distribution_count', 0)}개 distribution 검사, "
                "알려진 취약점 0건"
            )
            if dependency_security_ok
            else "의존성 취약점 감사 실패 또는 알려진 취약점 발견",
            "pip와 취약 패키지를 수정 버전으로 갱신하고 온라인 감사를 재실행"
            if not dependency_security_ok
            else "",
        )
    )
    source_security_ok = (
        source_security_audit.get("status") == "PASS"
        and source_security_audit.get("scanner") == "bandit"
        and source_security_audit.get("scanner_version") == "1.9.4"
        and source_security_audit.get("scope")
        == "secom_and_integrated_dashboard_python_source"
        and source_security_audit.get("targets")
        == ["app.py", "dashboard_ui", "코드/secom"]
        and source_security_audit.get("minimum_severity") == "MEDIUM"
        and source_security_audit.get("minimum_confidence") == "MEDIUM"
        and int(source_security_audit.get("issue_count", -1)) == 0
        and int(source_security_audit.get("scanner_error_count", -1)) == 0
        and int(source_security_audit.get("suppression_count", -1)) == 0
        and int(source_security_audit.get("scanned_file_count", 0))
        == source_fingerprint(project)[0]
        and source_security_audit.get("source_sha256")
        == source_fingerprint(project)[1]
    )
    checks.append(
        _check(
            "SOURCE_SECURITY_AUDIT",
            "Python 소스 보안 정적분석",
            "PASS" if source_security_ok else "BLOCK",
            (
                f"Bandit 1.9.4로 app.py·dashboard_ui·코드/secom "
                f"{source_security_audit.get('scanned_file_count', 0)}개 파일 검사, "
                "중간 이상 이슈·오류·nosec 억제 0건"
            )
            if source_security_ok
            else "SECOM 소스 정적분석 실패, 이슈·오류 또는 nosec 억제 발견",
            "탐지 이슈를 수정하고 Bandit 증거를 재생성"
            if not source_security_ok
            else "",
        )
    )
    lineage_checks = data_lineage_validation.get("checks", {})
    lineage_ok = (
        data_lineage_manifest.get("raw_sensor_values_embedded") is False
        and data_lineage_validation.get("status") == "PASS"
        and data_lineage_validation.get("validation_scope")
        == "raw_to_saved_split_reconstruction"
        and data_lineage_validation.get("real_files_modified") is False
        and bool(lineage_checks)
        and all(value is True for value in lineage_checks.values())
    )
    checks.append(
        _check(
            "DATA_LINEAGE",
            "원본→분할 데이터 계보",
            "PASS" if lineage_ok else "BLOCK",
            "원본·라벨·전처리 SHA-256, 행·열·값·median 재현, 모의 오염 탐지"
            if lineage_ok
            else "데이터 계보 manifest 또는 저장 분할 재현 검증 실패",
            "원본·전처리 파일 버전을 맞추고 계보 검증 재실행" if not lineage_ok else "",
        )
    )
    environment_checks = environment_validation.get("checks", {})
    environment_ok = (
        environment_validation.get("status") == "PASS"
        and environment_validation.get("validation_scope")
        == "local_environment_reproducibility_and_integrity"
        and environment_validation.get("vulnerability_scan_performed") is False
        and environment_validation.get("known_vulnerability_count") is None
        and bool(environment_checks)
        and all(value is True for value in environment_checks.values())
        and environment_manifest.get("vulnerability_scan_performed") is False
        and environment_manifest.get("pre_deserialization_hash_gate_declared") is True
        and environment_manifest.get("text_hash_normalization")
        == "UTF-8-with-canonical-LF"
        and bool(environment_manifest.get("requirements_sha256"))
        and software_bom.get("vulnerability_information_included") is False
        and int(software_bom.get("component_count", 0))
        >= int(environment_validation.get("direct_dependency_count", 0))
    )
    checks.append(
        _check(
            "ENVIRONMENT_REPRODUCIBILITY",
            "실행환경 재현성·소프트웨어 명세",
            "PASS" if environment_ok else "BLOCK",
            "exact pin·설치 버전·pip 의존성·실행 파일 해시·역직렬화 전 해시 게이트 검증"
            if environment_ok
            else "실행환경 manifest, SBOM 또는 무결성 검증 증거가 불완전함",
            "동일 가상환경에서 환경 증거를 다시 생성하고 불일치를 해소"
            if not environment_ok
            else "",
        )
    )
    dependency_lock_validation = validate_dependency_lock_manifest(
        dependency_lock_manifest, project
    )
    dependency_lock_ok = (
        dependency_lock_manifest.get("status") == "PASS"
        and dependency_lock_manifest.get("format") == "PEP 751 pylock.toml"
        and dependency_lock_manifest.get("generator") == "pip"
        and dependency_lock_manifest.get("generator_version") == "26.2.1"
        and dependency_lock_manifest.get("platform_system") == "Windows"
        and dependency_lock_manifest.get("python_version") == "3.11.9"
        and dependency_lock_manifest.get("experimental_tooling") is True
        and dependency_lock_manifest.get("vulnerability_scan_performed") is False
        and dependency_lock_validation["manifest_valid"] is True
        and dependency_lock_validation["all_artifacts_hashed"] is True
        and dependency_lock_validation["only_binary"] is True
        and dependency_lock_validation["urls_sanitized"] is True
        and dependency_lock_validation["sdist_count"] == 0
        and dependency_lock_validation["direct_dependency_count"]
        == environment_validation.get("direct_dependency_count")
        and dependency_lock_manifest.get("requirements_sha256")
        == environment_manifest.get("requirements_sha256")
    )
    checks.append(
        _check(
            "WINDOWS_DEPENDENCY_LOCK",
            "Windows 전이 의존성·wheel 무결성 잠금",
            "PASS" if dependency_lock_ok else "BLOCK",
            (
                f"Python 3.11.9용 {dependency_lock_validation['package_count']}개 package와 "
                f"{dependency_lock_validation['wheel_count']}개 wheel 버전·SHA-256 고정"
            )
            if dependency_lock_ok
            else "Windows pylock 구조, 버전, wheel hash 또는 환경 일치 증거가 불완전함",
            "Windows 테스트 환경에서 dependency lock builder를 다시 실행"
            if not dependency_lock_ok
            else "",
        )
    )
    linux_lock_validation = validate_linux_dependency_lock_manifest(
        linux_dependency_lock_manifest, project
    )
    linux_ci_lock = linux_dependency_lock_manifest["github_actions"]
    linux_ci_ok = (
        linux_lock_validation["valid"] is True
        and linux_ci_lock.get("all_locked_versions_match_tested_environment") is True
        and linux_lock_validation["github_actions"]["sdist_count"] == 0
        and linux_lock_validation["github_actions"]["package_count"] >= 19
    )
    checks.append(
        _check(
            "LINUX_CI_DEPENDENCY_LOCK",
            "Linux CI 전이 의존성·wheel 무결성 잠금",
            "PASS" if linux_ci_ok else "BLOCK",
            (
                f"Ubuntu/Python 3.11.9에서 {linux_ci_lock['package_count']}개 package의 "
                "wheel·SHA-256·실제 설치 버전 일치 검증"
            )
            if linux_ci_ok
            else "Linux CI 잠금 구조, hash 또는 실제 설치 버전 대조 증거가 불완전함",
            "Ubuntu Actions에서 Linux 잠금을 재생성하고 차이를 검토"
            if not linux_ci_ok
            else "",
        )
    )
    streamlit_lock = linux_dependency_lock_manifest["streamlit"]
    streamlit_resolution_ok = (
        linux_lock_validation["valid"] is True
        and streamlit_lock.get("resolution_validated_on_target_platform") is True
        and streamlit_lock.get("installation_validated_by_this_step") is False
        and linux_lock_validation["streamlit"]["sdist_count"] == 0
        and linux_lock_validation["streamlit"]["direct_dependency_count"]
        == environment_validation.get("direct_dependency_count")
    )
    checks.append(
        _check(
            "STREAMLIT_DEPENDENCY_LOCK",
            "Streamlit Linux 의존성 해석 잠금",
            "WARN" if streamlit_resolution_ok else "BLOCK",
            (
                f"Ubuntu에서 {streamlit_lock['package_count']}개 package·wheel SHA-256 해석 완료; "
                "Community Cloud 설치 적용·재현 검증은 미완료"
            )
            if streamlit_resolution_ok
            else "Streamlit Linux 잠금의 구조·hash·requirements 연결이 불완전함",
            "배포 서비스의 pylock 지원 확인 후 잠금 설치를 적용하고 smoke test 재검증",
        )
    )
    expected_hashes = advanced["model_hashes"]
    model_hash_ok = (
        component_hashes["catboost_model"] == expected_hashes["CatBoost"]
        and component_hashes["xgboost_model"] == expected_hashes["XGBoost"]
    )
    checks.append(
        _check(
            "MODEL_INTEGRITY",
            "배포 모델 무결성",
            "PASS" if model_hash_ok else "BLOCK",
            "CatBoost·XGBoost SHA-256이 안전 기준 파일과 일치"
            if model_hash_ok
            else "모델 SHA-256 불일치",
            "모델과 모든 파생 기준을 같은 버전으로 다시 생성" if not model_hash_ok else "",
        )
    )
    registry_hashes = {
        item.get("model_name"): item.get("artifact_sha256")
        for item in model_registry.get("models", [])
    }
    failover_ok = (
        registry_hashes == expected_hashes
        and model_registry.get("automatic_failover_allowed") is False
        and model_registry.get("activation_requires_human_approval") is True
        and failover_drill.get("status") == "PASS"
        and failover_drill.get("simulated_active_corruption_detected") is True
        and failover_drill.get("real_model_file_modified") is False
        and failover_drill.get("standby_integrity_verified") is True
        and failover_drill.get("input_schema_compatible") is True
        and failover_drill.get("label_mapping_compatible") is True
        and failover_drill.get("automatic_failover_allowed") is False
        and failover_drill.get("recommended_action")
        == "HALT_AND_REQUIRE_HUMAN_APPROVAL_FOR_STANDBY"
    )
    checks.append(
        _check(
            "MODEL_FAILOVER_DRILL",
            "모델 무결성·장애 복구 훈련",
            "PASS" if failover_ok else "BLOCK",
            "운영 모델 손상 탐지, 대기 모델·입력 계약 검증, 자동 전환 금지"
            if failover_ok
            else "모델 레지스트리 또는 장애 복구 통제 증거가 불완전함",
            "모델 해시·스키마를 복구하고 사람 승인형 복구 훈련 재실행"
            if not failover_ok
            else "",
        )
    )
    selection_safe = (
        safety.get("policy_selection_uses_test_labels") is False
        and safety.get("model_threshold_source") == "repeated_5x3_oof_train_only"
    )
    checks.append(
        _check(
            "TEST_LOCK",
            "Test 라벨 잠금",
            "PASS" if selection_safe else "BLOCK",
            "모델 임계값·검토정책은 반복 train OOF에서 고정",
            "test 라벨과 분리된 선택 절차 재실행" if not selection_safe else "",
        )
    )
    ood_safe = advanced.get("fit_scope") == "fixed_train_only"
    checks.append(
        _check(
            "OOD_AND_XAI_SCOPE",
            "OOD·XAI 기준 범위",
            "PASS" if ood_safe and not advanced.get("physical_causality_claimed") else "BLOCK",
            "OOD는 fixed train에만 적합, XAI 물리 인과관계 미주장",
        )
    )
    shap_scope_ok = (
        shap_reliability.get("evaluation_scope")
        == "fixed_test_post_selection_audit_only"
        and shap_reliability.get("model_selection_uses_results") is False
        and shap_reliability.get("threshold_selection_uses_results") is False
        and shap_reliability.get("physical_causality_claimed") is False
        and shap_reliability.get("model_hashes") == expected_hashes
    )
    shap_status = (
        "BLOCK"
        if not shap_scope_ok
        else ("PASS" if shap_reliability.get("status") == "PASS" else "WARN")
    )
    checks.append(
        _check(
            "SHAP_RELIABILITY",
            "SHAP 설명 안정성·충실도",
            shap_status,
            "고정 test 사후 감사에서 두 모델 모두 안정성·마스킹 guardrail 통과"
            if shap_status == "PASS"
            else "설명 감사 범위·모델 해시가 불일치하거나 일부 guardrail 미통과",
            "설명 변동이 큰 입력은 전문가 검토하고 실제 센서 교란으로 외부 검증"
            if shap_status != "PASS"
            else "",
        )
    )
    clean_pass = batch_gate.get("clean_test_decision", {}).get("status") == "PASS"
    stress_detected = int(batch_gate.get("stress_stop_count", 0)) > 0
    checks.append(
        _check(
            "BATCH_CIRCUIT_BREAKER",
            "배치 안전 차단기",
            "PASS" if clean_pass and stress_detected else "BLOCK",
            f"clean={batch_gate.get('clean_test_decision', {}).get('status')}, "
            f"stress STOP={batch_gate.get('stress_stop_count', 0)}/{batch_gate.get('stress_batches', 0)}",
        )
    )
    robustness_failures = int(robustness.get("guardrail_fail_count", 0))
    checks.append(
        _check(
            "SENSOR_ROBUSTNESS",
            "센서 오류 강건성",
            "PASS" if robustness_failures == 0 else "WARN",
            f"사전 정의 10개 조건 중 {robustness.get('guardrail_pass_count', 0)}개 통과; "
            f"실패={', '.join(robustness.get('failed_scenarios', [])) or '없음'}",
            "강한 노이즈 입력은 자동판정 차단 및 실제 장비 교란 전향 검증"
            if robustness_failures
            else "",
        )
    )
    containment_validation = validate_sensor_failure_containment(
        sensor_failure_containment, project
    )
    containment_ok = (
        containment_validation["status"] == "PASS"
        and containment_validation["all_failed_scenarios_contained"] is True
        and sensor_failure_containment.get("model_robustness_improved_by_this_audit")
        is False
        and sensor_failure_containment.get("actual_hardware_faults_validated") is False
    )
    checks.append(
        _check(
            "SENSOR_FAILURE_CONTAINMENT",
            "센서 성능 실패 안전격리",
            "PASS" if containment_ok else "BLOCK",
            (
                f"강건성 실패 {containment_validation['failed_scenario_count']}개 조건의 "
                "모든 반복에서 배치 STOP·자동판정 금지 확인"
            )
            if containment_ok
            else "강건성 성능 실패 조건 중 자동판정이 허용된 반복이 존재",
            "차단 신호·임계값·자동판정 정책을 수정하고 안전격리 교차검증 재실행"
            if not containment_ok
            else "",
        )
    )
    governance_ok = (
        feedback.get("test_labels_used_for_guardrails") is False
        and feedback.get("automatic_retraining_allowed") is False
        and promotion.get("automatic_model_promotion_allowed") is False
    )
    checks.append(
        _check(
            "HUMAN_GOVERNANCE",
            "검수·재학습·승격 통제",
            "PASS" if governance_ok else "BLOCK",
            "행 단위 전체 검수, train OOF guardrail, 자동 재학습·승격 금지",
        )
    )
    audit_chain_ok = (
        audit_policy.get("algorithm") == "SHA-256"
        and audit_policy.get("raw_sensor_values_persisted") is False
        and audit_verification.get("valid") is True
        and audit_verification.get("contains_raw_sensor_values") is False
        and tamper_detection.get("tamper_detected") is True
    )
    checks.append(
        _check(
            "TAMPER_EVIDENT_AUDIT_CHAIN",
            "변조 감지형 감사 원장",
            "PASS" if audit_chain_ok else "BLOCK",
            "SHA-256 연결, JSONL 재검증, 센서 원본 미저장, 위변조 데모 탐지 성공"
            if audit_chain_ok
            else "감사 원장의 체인·민감정보·위변조 탐지 증거가 불완전함",
            "감사 원장 데모와 정책 증거를 다시 생성" if not audit_chain_ok else "",
        )
    )

    temporal_path = artifacts / "temporal_results" / "split_summary.json"
    temporal_valid = temporal_path.is_file() and temporal_path.stat().st_size > 20
    checks.append(
        _check(
            "TEMPORAL_VALIDATION",
            "시간 외삽 검증",
            "PASS" if temporal_valid else "BLOCK",
            "유효한 시간순 검증 결과 존재" if temporal_valid else "현재 시간순 결과가 비어 있거나 placeholder임",
            "실제 timestamp가 있는 신규 기간 데이터를 확보해 시간 외삽 평가",
        )
    )
    dictionary_path = project / "데이터" / "SECOM 데이터셋" / "sensor_dictionary.csv"
    dictionary_report_path = artifacts / "sensor_semantics" / "validation_report.json"
    dictionary_report = _load_json(dictionary_report_path) if dictionary_report_path.is_file() else {}
    dictionary_valid = dictionary_is_release_evidence(dictionary_path, dictionary_report)
    intake_ready = (
        sensor_semantics_schema.get("expected_rows") == 590
        and sensor_semantics_status.get("status") == "AWAITING_OWNER_MAPPING"
        and sensor_semantics_status.get("verified_count") == 0
        and sensor_semantics_status.get("production_claim_allowed") is False
    )
    checks.append(
        _check(
            "SENSOR_SEMANTICS",
            "센서 의미·단위 사전",
            "PASS" if dictionary_valid else "BLOCK",
            "590개 센서의 실제 의미·단위·공정·범위와 책임자 검증 완료"
            if dictionary_valid
            else (
                "590개 입력 템플릿과 검증 gate는 준비됐으나 실제 장비 담당자 매핑은 0/590"
                if intake_ready
                else "공개 SECOM 변수는 익명이며 검증된 물리 의미·단위 사전이 없음"
            ),
            "실제 장비의 센서명·단위·허용범위·담당 공정 매핑 확보",
        )
    )
    external_path = artifacts / "external_validation" / "validated_summary.json"
    external_summary = _load_json(external_path) if external_path.is_file() else {}
    external_valid = validated_summary_is_release_evidence(
        external_summary, external_validation_protocol
    )
    checks.append(
        _check(
            "PROSPECTIVE_EXTERNAL_VALIDATION",
            "외부 전향 검증",
            "PASS" if external_valid else "BLOCK",
            "사전 등록 프로토콜과 일치하는 독립 신규 lot 검증의 모든 gate 통과"
            if external_valid
            else (
                "모델 해시·임계값·최소 표본·합격 기준은 사전 고정했으나 독립 신규 데이터 대기 중"
                if external_validation_status.get("status") == "AWAITING_INDEPENDENT_DATA"
                else "외부 검증 결과가 없거나 사전 등록 gate를 통과하지 못함"
            ),
            "모델·임계값 고정 후 신규 lot에서 재현율·오탐률·검토부하 평가",
        )
    )
    audit_store_prototype_ok = (
        audit_store_config.get("backend") == "SQLite"
        and audit_store_config.get("raw_sensor_values_persisted") is False
        and audit_store_verification.get("persistence_reopen_verified") is True
        and audit_store_verification.get("sqlite_integrity_verified") is True
        and audit_store_verification.get("hash_chain_verified") is True
        and audit_store_verification.get("append_only_update_blocked") is True
        and audit_store_verification.get("append_only_delete_blocked") is True
        and audit_store_verification.get("control_removal_then_tamper_detected") is True
    )
    audit_store_production_ready = (
        audit_store_prototype_ok
        and audit_store_config.get("production_ready") is True
        and audit_store_config.get("access_control", {}).get("configured") is True
        and audit_store_config.get("retention", {}).get("configured") is True
        and audit_store_config.get("backup", {}).get("configured") is True
        and audit_store_config.get("external_immutable_anchor", {}).get("configured") is True
    )
    checks.append(
        _check(
            "PERSISTENT_AUDIT_STORE",
            "영구 감사 로그 저장소",
            "PASS" if audit_store_production_ready else "BLOCK",
            "접근통제·보존·백업·외부 불변 앵커가 연결된 영구 저장소"
            if audit_store_production_ready
            else (
                "SQLite 재연결·트랜잭션·UPDATE/DELETE 차단·해시 검증 프로토타입 통과, "
                "운영 접근통제·보존·외부 백업은 미연결"
                if audit_store_prototype_ok
                else "영구 감사 저장소 프로토타입 증거가 불완전함"
            ),
            "SSO/RBAC, 승인된 보존기간, 암호화 외부 백업·복원 훈련, 불변 체인 앵커 연결",
        )
    )

    production_only = {
        "SENSOR_SEMANTICS",
        "PROSPECTIVE_EXTERNAL_VALIDATION",
        "PERSISTENT_AUDIT_STORE",
    }
    demo_checks = [item for item in checks if item["check_id"] not in production_only]
    demo_blockers = [item for item in demo_checks if item["status"] == "BLOCK"]
    demo_warnings = [item for item in demo_checks if item["status"] == "WARN"]
    production_blockers = [item for item in checks if item["status"] == "BLOCK"]
    demo_status = "BLOCKED" if demo_blockers else (
        "READY_WITH_WARNINGS" if demo_warnings else "READY"
    )
    production_status = "BLOCKED" if production_blockers else "READY"
    manifest_core = {
        "manifest_version": 1,
        "system": "SECOM explainable defect diagnosis prototype",
        "component_hashes": component_hashes,
        "model_hashes": expected_hashes,
        "profile": feedback["profile"],
        "automatic_retraining": False,
        "automatic_model_promotion": False,
    }
    release_id = hashlib.sha256(
        json.dumps(manifest_core, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16].upper()
    manifest = {"release_id": release_id, **manifest_core}
    report = {
        "version": 1,
        "release_id": release_id,
        "demo_readiness": demo_status,
        "production_readiness": production_status,
        "demo_blocker_count": len(demo_blockers),
        "demo_warning_count": len(demo_warnings),
        "production_blocker_count": len(production_blockers),
        "checks": checks,
        "interpretation": (
            "대학생 대회 시연 준비도와 실제 반도체 생산 배포 승인은 서로 다른 판정입니다. "
            "READY는 성능 보증이나 규제 승인을 의미하지 않습니다."
        ),
    }
    return report, manifest


def report_frame(report: Mapping) -> pd.DataFrame:
    return pd.DataFrame(report["checks"])


def render_readiness_html(report: Mapping, manifest: Mapping) -> str:
    rows = []
    for item in report["checks"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['check_id'])}</td>"
            f"<td>{html.escape(item['title'])}</td>"
            f"<td>{html.escape(item['status'])}</td>"
            f"<td>{html.escape(item['evidence'])}</td>"
            f"<td>{html.escape(item['required_action'])}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><meta charset='utf-8'><title>SECOM release readiness</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;line-height:1.5}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:8px;text-align:left}"
        "th{background:#eee}.blocked{color:#b00020}</style>"
        f"<h1>SECOM 릴리스 준비도</h1><p>Release ID: {html.escape(manifest['release_id'])}</p>"
        f"<p>대회 데모: <b>{html.escape(report['demo_readiness'])}</b> · 실제 생산: "
        f"<b class='blocked'>{html.escape(report['production_readiness'])}</b></p>"
        f"<p>{html.escape(report['interpretation'])}</p>"
        "<table><thead><tr><th>ID</th><th>항목</th><th>상태</th><th>증거</th><th>필요 조치</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
