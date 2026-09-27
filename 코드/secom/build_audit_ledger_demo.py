"""Build deterministic SECOM tamper-evident audit ledger evidence."""

from __future__ import annotations

import json
from pathlib import Path

from audit_ledger import append_event, empty_ledger, ledger_jsonl_bytes, verify_ledger


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "결과물" / "secom" / "audit_ledger"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger = empty_ledger()
    ledger = append_event(
        ledger,
        event_type="BATCH_DIAGNOSIS",
        actor="demo-reviewer",
        batch_id="demo-batch-001",
        recorded_at="2026-01-01T09:00:00+09:00",
        payload={
            "profile": "balanced_f2",
            "row_count": 314,
            "defect_alert_rate": 0.070064,
            "ood_any_rate": 0.012739,
            "gate_status": "PASS",
            "input_digest": "A" * 64,
            "demo_data": True,
        },
    )
    ledger = append_event(
        ledger,
        event_type="LABEL_FEEDBACK",
        actor="demo-reviewer",
        batch_id="demo-batch-001",
        recorded_at="2026-01-01T10:00:00+09:00",
        payload={
            "label_coverage": 1.0,
            "confirmed_defects": 20,
            "true_positive": 12,
            "false_positive": 10,
            "false_negative": 8,
            "true_negative": 284,
            "feedback_evidence": "row_level_complete",
            "demo_data": True,
        },
    )
    ledger = append_event(
        ledger,
        event_type="RELEASE_READINESS",
        actor="system",
        batch_id="",
        recorded_at="2026-01-01T11:00:00+09:00",
        payload={
            "demo_readiness": "READY_WITH_WARNINGS",
            "production_readiness": "BLOCKED",
            "automatic_release_allowed": False,
            "demo_data": True,
        },
    )
    verification = verify_ledger(ledger)
    (OUTPUT_DIR / "audit_ledger_demo.jsonl").write_bytes(ledger_jsonl_bytes(ledger))
    (OUTPUT_DIR / "audit_ledger_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tampered = ledger.copy()
    tampered.loc[1, "payload_json"] = tampered.loc[1, "payload_json"].replace(
        '"confirmed_defects":20', '"confirmed_defects":21'
    )
    detected = False
    error_message = ""
    try:
        verify_ledger(tampered)
    except ValueError as error:
        detected = True
        error_message = str(error)
    tamper_result = {
        "tamper_detected": detected,
        "tamper_type": "payload_value_changed_without_rehash",
        "error": error_message,
    }
    (OUTPUT_DIR / "tamper_detection_demo.json").write_text(
        json.dumps(tamper_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    policy = {
        "policy_version": 1,
        "algorithm": "SHA-256",
        "chain_structure": "previous_hash + canonical_event_content",
        "raw_sensor_values_persisted": False,
        "import_export_format": "UTF-8 JSON Lines",
        "max_import_bytes": 5 * 1024 * 1024,
        "persistent_external_store_connected": False,
        "storage_scope": "browser_session_and_user_export",
        "tamper_detection_demo_passed": detected,
        "limitations": [
            "해시 체인은 변경을 탐지하지만 JSONL 파일의 삭제 자체를 막지는 않습니다.",
            "Streamlit 세션은 영구 저장소가 아니므로 사용자가 JSONL을 내려받아 보관해야 합니다.",
            "운영 배포에는 접근통제·백업·보존기간·서명 또는 외부 원장 고정이 필요합니다.",
        ],
    }
    (OUTPUT_DIR / "audit_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "summary.md").write_text(
        "# SECOM 변조 감지형 감사 원장\n\n"
        "실제 운영 기록이 아니라 기능 검증용 결정적 데모입니다. 센서 원본값은 저장하지 않고 "
        "집계 지표·입력 결과 해시·모델 해시만 연결합니다.\n\n"
        f"- 알고리즘: {verification['algorithm']}\n"
        f"- 연결 이벤트: {verification['entry_count']}개\n"
        f"- payload 위변조 탐지: {'성공' if detected else '실패'}\n"
        "- JSONL 가져오기·내보내기: 지원\n"
        "- 외부 영구 저장소: 미연결\n\n"
        "해시 체인은 중간 이벤트 수정·삭제·재정렬을 탐지하지만 파일 자체의 삭제 방지, 접근통제, "
        "백업을 제공하지 않습니다. 실제 생산 배포 전 영구 감사 저장소가 필요합니다.\n",
        encoding="utf-8",
    )
    if not detected:
        raise RuntimeError("감사 원장 위변조 탐지 데모가 실패했습니다.")
    print(
        f"감사 원장 데모 생성 완료: entries={verification['entry_count']}, "
        f"tamper_detected={detected}"
    )


if __name__ == "__main__":
    main()
