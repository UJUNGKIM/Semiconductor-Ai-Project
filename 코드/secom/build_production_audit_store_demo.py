"""Build deterministic evidence for the local persistent audit-store prototype."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from production_audit_store import append_store_event, initialize_store, verify_store


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "결과물" / "secom" / "production_audit_store"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="secom-audit-") as directory:
        database = initialize_store(Path(directory) / "audit.sqlite3")
        append_store_event(
            database,
            event_type="BATCH_DIAGNOSIS",
            actor="demo-reviewer",
            batch_id="persistent-demo-001",
            recorded_at="2026-01-01T09:00:00+09:00",
            payload={
                "row_count": 314,
                "gate_status": "PASS",
                "input_digest": "A" * 64,
                "demo_data": True,
            },
        )
        append_store_event(
            database,
            event_type="LABEL_FEEDBACK",
            actor="demo-reviewer",
            batch_id="persistent-demo-001",
            recorded_at="2026-01-01T10:00:00+09:00",
            payload={
                "label_coverage": 1.0,
                "confirmed_defects": 20,
                "demo_data": True,
            },
        )
        verification = verify_store(database)

        update_blocked = False
        delete_blocked = False
        with closing(sqlite3.connect(database)) as connection:
            try:
                connection.execute("UPDATE audit_events SET actor = 'tampered' WHERE sequence = 1")
            except sqlite3.IntegrityError:
                update_blocked = True
            try:
                connection.execute("DELETE FROM audit_events WHERE sequence = 1")
            except sqlite3.IntegrityError:
                delete_blocked = True

        tamper_detected = False
        tamper_error = ""
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("DROP TRIGGER audit_events_no_update")
            connection.execute(
                "UPDATE audit_events SET actor = 'tampered' WHERE sequence = 1"
            )
            connection.commit()
        try:
            verify_store(database)
        except ValueError as error:
            tamper_detected = True
            tamper_error = str(error)

    evidence = {
        "evidence_version": 1,
        "demo_only": True,
        "database_committed": False,
        "persistence_reopen_verified": verification["entry_count"] == 2,
        "sqlite_integrity_verified": verification["sqlite_integrity_check"] == "ok",
        "hash_chain_verified": verification["valid"],
        "append_only_update_blocked": update_blocked,
        "append_only_delete_blocked": delete_blocked,
        "control_removal_then_tamper_detected": tamper_detected,
        "tamper_error": tamper_error,
        "contains_raw_sensor_values": verification["contains_raw_sensor_values"],
        "entry_count": verification["entry_count"],
    }
    config = {
        "config_version": 1,
        "backend": "SQLite",
        "deployment_mode": "local_persistence_prototype",
        "database_path_environment_variable": "SECOM_AUDIT_DB_PATH",
        "database_committed_to_git": False,
        "production_ready": False,
        "raw_sensor_values_persisted": False,
        "transaction_mode": "BEGIN IMMEDIATE",
        "sqlite_synchronous": "FULL",
        "append_only_controls": ["UPDATE trigger block", "DELETE trigger block", "SHA-256 chain"],
        "access_control": {"configured": False, "required": "SSO/RBAC and named service identity"},
        "retention": {"configured": False, "required": "approved retention and legal-hold policy"},
        "backup": {"configured": False, "required": "encrypted off-host backup with restore drill"},
        "external_immutable_anchor": {"configured": False, "required": "signed off-host chain-head anchor"},
        "limitations": [
            "로컬 SQLite 파일은 호스트 관리자에 의한 파일 삭제를 막지 못합니다.",
            "DB 접근 주체 인증과 역할 기반 권한은 배포 환경에서 별도로 연결해야 합니다.",
            "외부 백업·복원 훈련과 불변 보존은 아직 구현되지 않았습니다.",
        ],
    }
    _write_json(OUTPUT_DIR / "store_config.json", config)
    _write_json(OUTPUT_DIR / "store_verification.json", evidence)
    (OUTPUT_DIR / "summary.md").write_text(
        "# SECOM 영구 감사 저장소 프로토타입\n\n"
        "브라우저 세션을 넘어 재연결 후에도 감사 이벤트가 유지되는 SQLite 어댑터를 "
        "검증했습니다. 센서 원본값은 저장하지 않습니다.\n\n"
        f"- 재연결 후 이벤트 복원: {'통과' if evidence['persistence_reopen_verified'] else '실패'}\n"
        f"- SQLite 무결성 검사: {'통과' if evidence['sqlite_integrity_verified'] else '실패'}\n"
        f"- UPDATE/DELETE 차단: {'통과' if update_blocked and delete_blocked else '실패'}\n"
        f"- 보호장치 제거 후 해시 변조 탐지: {'통과' if tamper_detected else '실패'}\n"
        "- 운영 준비 상태: 미완료(SSO/RBAC, 보존정책, 외부 백업·복원, 불변 앵커 필요)\n\n"
        "이 결과는 로컬 내구성 프로토타입의 기능 증거이며 실제 생산 배포 승인을 의미하지 않습니다.\n",
        encoding="utf-8",
    )
    required = (
        evidence["persistence_reopen_verified"],
        evidence["sqlite_integrity_verified"],
        evidence["hash_chain_verified"],
        evidence["append_only_update_blocked"],
        evidence["append_only_delete_blocked"],
        evidence["control_removal_then_tamper_detected"],
        not evidence["contains_raw_sensor_values"],
    )
    if not all(required):
        raise RuntimeError("영구 감사 저장소 프로토타입 검증이 실패했습니다.")
    print("영구 감사 저장소 프로토타입 검증 완료: production_ready=False")


if __name__ == "__main__":
    main()
