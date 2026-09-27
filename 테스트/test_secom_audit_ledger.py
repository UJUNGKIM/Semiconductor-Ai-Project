import json
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from audit_ledger import (
    append_event,
    empty_ledger,
    ledger_jsonl_bytes,
    parse_ledger_jsonl,
    validate_ledger,
    verify_ledger,
)


class AuditLedgerTests(unittest.TestCase):
    def build_ledger(self):
        ledger = append_event(
            empty_ledger(),
            event_type="BATCH_DIAGNOSIS",
            actor="reviewer",
            batch_id="batch-1",
            recorded_at="2026-01-01T09:00:00+09:00",
            payload={"row_count": 12, "input_digest": "a" * 64},
        )
        return append_event(
            ledger,
            event_type="LABEL_FEEDBACK",
            actor="reviewer",
            batch_id="batch-1",
            recorded_at="2026-01-01T10:00:00+09:00",
            payload={"label_coverage": 1.0, "confirmed_defects": 2},
        )

    def test_append_and_verify_chain(self):
        ledger = self.build_ledger()
        summary = verify_ledger(ledger)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["entry_count"], 2)
        self.assertEqual(len(summary["latest_hash"]), 64)
        self.assertFalse(summary["contains_raw_sensor_values"])

    def test_payload_tampering_is_rejected(self):
        ledger = self.build_ledger()
        tampered = ledger.copy()
        tampered.loc[0, "payload_json"] = tampered.loc[0, "payload_json"].replace(
            '"row_count":12', '"row_count":13'
        )
        with self.assertRaisesRegex(ValueError, "이벤트 해시"):
            validate_ledger(tampered)

    def test_middle_deletion_and_reordering_are_rejected(self):
        ledger = self.build_ledger()
        ledger = append_event(
            ledger,
            event_type="RELEASE_READINESS",
            actor="system",
            recorded_at="2026-01-01T11:00:00+09:00",
            payload={"production_readiness": "BLOCKED"},
        )
        with self.assertRaises(ValueError):
            validate_ledger(ledger.drop(index=1).reset_index(drop=True))
        with self.assertRaises(ValueError):
            validate_ledger(ledger.iloc[[1, 0, 2]].reset_index(drop=True))

    def test_raw_sensor_payload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "센서 원본"):
            append_event(
                empty_ledger(),
                event_type="BATCH_DIAGNOSIS",
                recorded_at="2026-01-01T09:00:00+09:00",
                payload={"sensor_values": [[1.0, 2.0]]},
            )
        with self.assertRaisesRegex(ValueError, "센서 원본"):
            append_event(
                empty_ledger(),
                event_type="BATCH_DIAGNOSIS",
                recorded_at="2026-01-01T09:00:00+09:00",
                payload={"nested": {"feature_values": [1.0]}},
            )

    def test_jsonl_round_trip_and_duplicate_key_rejection(self):
        ledger = self.build_ledger()
        restored = parse_ledger_jsonl(ledger_jsonl_bytes(ledger))
        pd.testing.assert_frame_equal(ledger, restored)
        bad = b'{"sequence":1,"sequence":1,"payload":{}}\n'
        with self.assertRaisesRegex(ValueError, "중복 JSON 키"):
            parse_ledger_jsonl(bad)

    def test_demo_artifacts_and_dashboard_contract(self):
        output = PROJECT / "결과물" / "secom" / "audit_ledger"
        for name in (
            "audit_ledger_demo.jsonl",
            "audit_ledger_verification.json",
            "tamper_detection_demo.json",
            "audit_policy.json",
            "summary.md",
        ):
            self.assertTrue((output / name).is_file(), name)
        verification = json.loads(
            (output / "audit_ledger_verification.json").read_text(encoding="utf-8")
        )
        tamper = json.loads(
            (output / "tamper_detection_demo.json").read_text(encoding="utf-8")
        )
        policy = json.loads((output / "audit_policy.json").read_text(encoding="utf-8"))
        self.assertTrue(verification["valid"])
        self.assertTrue(tamper["tamper_detected"])
        self.assertFalse(policy["raw_sensor_values_persisted"])
        self.assertFalse(policy["persistent_external_store_connected"])
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("변조 감지형 감사 원장", app)
        self.assertIn("parse_ledger_jsonl", app)
        self.assertIn("secom_audit_ledger.jsonl", app)


if __name__ == "__main__":
    unittest.main()
