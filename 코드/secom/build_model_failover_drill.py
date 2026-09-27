"""Build a human-controlled SECOM model integrity and failover drill."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from model_registry import (
    registry_entry,
    simulated_corruption_is_detected,
    validate_registry,
)
from train_compare_models import RANDOM_STATE, TEST_SIZE, load_data, structural_filter


PROJECT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT / "결과물" / "secom" / "dual_model_results"
OUTPUT_DIR = PROJECT / "결과물" / "secom" / "model_failover"


def _metrics(labels, prediction) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f2": float(fbeta_score(labels, prediction, beta=2, zero_division=0)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    relative_dir = "결과물/secom/dual_model_results"
    registry = {
        "registry_version": 1,
        "system": "SECOM explainable defect diagnosis prototype",
        "automatic_failover_allowed": False,
        "activation_requires_human_approval": True,
        "failure_action": "HALT_AUTOMATED_DIAGNOSIS",
        "models": [
            registry_entry(
                PROJECT,
                model_name="CatBoost",
                role="ACTIVE",
                artifact_path=f"{relative_dir}/catboost_model.joblib",
            ),
            registry_entry(
                PROJECT,
                model_name="XGBoost",
                role="VERIFIED_STANDBY",
                artifact_path=f"{relative_dir}/xgboost_model.joblib",
            ),
        ],
        "activation_checklist": [
            "운영 모델 SHA-256 불일치 또는 로드 실패 확인",
            "자동 진단 중단 및 영향 배치 격리",
            "대기 모델 SHA-256·입력 스키마·라벨 매핑 검증",
            "대기 모델 임계값과 검토 업무량 확인",
            "책임자 승인 후 제한된 배치에서 수동 활성화",
            "감사 원장에 장애·승인·복구 이벤트 기록",
        ],
    }
    registry_validation = validate_registry(registry, PROJECT)

    print("[1/3] 원본을 수정하지 않고 운영 모델 손상을 모의합니다...")
    active = next(item for item in registry["models"] if item["role"] == "ACTIVE")
    active_path = PROJECT / active["artifact_path"]
    corruption_detected = simulated_corruption_is_detected(
        active_path, active["artifact_sha256"]
    )

    print("[2/3] 고정 test에서 운영·대기 모델 행동 차이를 기록합니다...")
    X_raw, y, _ = load_data(PROJECT)
    X, _, _ = structural_filter(X_raw)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    predictions = {}
    metrics = {}
    for entry in registry["models"]:
        bundle = joblib.load(PROJECT / entry["artifact_path"])
        imputed = bundle["imputer"].transform(X_test[bundle["input_features"]])
        ready = bundle["selector"].transform(imputed)
        probability = bundle["model"].predict_proba(ready)[:, 1]
        prediction = (probability >= float(bundle["primary_threshold"])).astype(int)
        predictions[entry["model_name"]] = prediction
        metrics[entry["model_name"]] = _metrics(y_test, prediction)
    disagreement_rate = float(
        (predictions["CatBoost"] != predictions["XGBoost"]).mean()
    )

    print("[3/3] 모델 레지스트리와 장애 복구 훈련 증거를 저장합니다...")
    standby = next(
        item for item in registry["models"] if item["role"] == "VERIFIED_STANDBY"
    )
    drill_passed = (
        registry_validation["valid"]
        and corruption_detected
        and registry_validation["input_schema_compatible"]
        and registry_validation["label_mapping_compatible"]
        and registry["automatic_failover_allowed"] is False
    )
    drill = {
        "version": 1,
        "status": "PASS" if drill_passed else "BLOCK",
        "drill_scope": "offline_integrity_and_human_controlled_failover_simulation",
        "active_model": active["model_name"],
        "standby_model": standby["model_name"],
        "active_integrity_before_drill": True,
        "simulated_active_corruption_detected": corruption_detected,
        "real_model_file_modified": False,
        "standby_integrity_verified": registry_validation["valid"],
        "input_schema_compatible": registry_validation["input_schema_compatible"],
        "label_mapping_compatible": registry_validation["label_mapping_compatible"],
        "automatic_failover_allowed": False,
        "recommended_action": "HALT_AND_REQUIRE_HUMAN_APPROVAL_FOR_STANDBY",
        "fixed_test_post_selection_audit_only": True,
        "test_rows": len(X_test),
        "model_metrics": metrics,
        "model_prediction_disagreement_rate": disagreement_rate,
        "limitations": [
            "실제 서비스 장애가 아니라 모델 바이트 손상을 메모리에서 모의한 훈련입니다.",
            "XGBoost 대기 모델 활성화는 자동으로 수행하지 않으며 책임자 승인이 필요합니다.",
            "고정 test 지표는 대기 모델 재선택이나 임계값 변경에 사용하지 않습니다.",
            "실제 배포에서는 영구 레지스트리, 접근통제, 서명, 백업과 복구 훈련이 필요합니다.",
        ],
    }
    (OUTPUT_DIR / "model_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "failover_drill.json").write_text(
        json.dumps(drill, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(registry["models"]).to_csv(
        OUTPUT_DIR / "artifact_inventory.csv", index=False, encoding="utf-8-sig"
    )
    comparison = pd.DataFrame(metrics).T.reset_index(names="model")
    comparison.to_csv(
        OUTPUT_DIR / "fixed_test_behavior.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "summary.md").write_text(
        "# SECOM 모델 무결성·장애 복구 훈련\n\n"
        f"- 판정: **{drill['status']}**\n"
        f"- 운영 모델: {active['model_name']}\n"
        f"- 검증된 대기 모델: {standby['model_name']}\n"
        f"- 원본 파일 수정 없는 모의 손상 탐지: {corruption_detected}\n"
        f"- 입력 스키마·라벨 매핑 호환: {registry_validation['input_schema_compatible'] and registry_validation['label_mapping_compatible']}\n"
        f"- 자동 전환: 금지\n"
        f"- 고정 test 모델 판정 불일치율: {disagreement_rate:.2%}\n\n"
        "손상이 탐지되면 자동 진단을 중단하고 영향 배치를 격리합니다. 대기 모델의 해시, "
        "입력 계약, 임계값과 검토 업무량을 확인한 뒤 책임자가 제한된 범위의 활성화를 "
        "승인해야 합니다. 이 오프라인 훈련은 실제 서비스 복구 훈련을 대체하지 않습니다.\n",
        encoding="utf-8",
    )
    if not drill_passed:
        raise RuntimeError("모델 장애 복구 훈련이 실패했습니다.")
    print(
        f"모델 장애 복구 훈련 완료: status={drill['status']}, "
        f"disagreement={disagreement_rate:.2%}"
    )


if __name__ == "__main__":
    main()
