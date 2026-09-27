import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from production_audit_store import append_store_event, read_store, verify_store
from release_readiness import build_release_readiness


class ProductionAuditStoreTests(unittest.TestCase):
    def _database(self, directory: str) -> Path:
        return Path(directory) / "audit.sqlite3"

    def test_events_persist_across_connections(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            append_store_event(
                database,
                event_type="BATCH_DIAGNOSIS",
                recorded_at="2026-01-01T09:00:00+09:00",
                payload={"row_count": 12, "input_digest": "A" * 64},
            )
            restored = read_store(database)
            self.assertEqual(len(restored), 1)
            self.assertEqual(restored.iloc[0]["event_type"], "BATCH_DIAGNOSIS")
            result = verify_store(database)
            self.assertTrue(result["valid"])
            self.assertTrue(result["append_only_triggers_present"])

    def test_raw_sensor_values_are_rejected_before_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with self.assertRaisesRegex(ValueError, "센서 원본"):
                append_store_event(
                    database,
                    event_type="BATCH_DIAGNOSIS",
                    recorded_at="2026-01-01T09:00:00+09:00",
                    payload={"raw_sensor_values": [[1.0, 2.0]]},
                )
            self.assertEqual(len(read_store(database)), 0)

    def test_update_and_delete_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            append_store_event(
                database,
                event_type="MANUAL_NOTE",
                recorded_at="2026-01-01T09:00:00+09:00",
                payload={"note_code": "reviewed"},
            )
            with closing(sqlite3.connect(database)) as connection:
                with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                    connection.execute("UPDATE audit_events SET actor = 'x'")
                with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                    connection.execute("DELETE FROM audit_events")

    def test_hash_chain_detects_tamper_after_trigger_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            append_store_event(
                database,
                event_type="MANUAL_NOTE",
                recorded_at="2026-01-01T09:00:00+09:00",
                payload={"note_code": "reviewed"},
            )
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("DROP TRIGGER audit_events_no_update")
                connection.execute("UPDATE audit_events SET actor = 'tampered'")
                connection.commit()
            with self.assertRaisesRegex(ValueError, "trigger|해시"):
                verify_store(database)

    def test_evidence_is_honest_about_production_limitations(self):
        output = PROJECT / "결과물" / "secom" / "production_audit_store"
        config = json.loads((output / "store_config.json").read_text(encoding="utf-8"))
        evidence = json.loads(
            (output / "store_verification.json").read_text(encoding="utf-8")
        )
        self.assertFalse(config["production_ready"])
        self.assertFalse(config["access_control"]["configured"])
        self.assertFalse(config["backup"]["configured"])
        self.assertTrue(evidence["persistence_reopen_verified"])
        self.assertTrue(evidence["append_only_update_blocked"])
        self.assertTrue(evidence["control_removal_then_tamper_detected"])
        report, _ = build_release_readiness(PROJECT)
        audit_check = next(
            item for item in report["checks"] if item["check_id"] == "PERSISTENT_AUDIT_STORE"
        )
        self.assertEqual(audit_check["status"], "BLOCK")
        self.assertIn("SQLite", audit_check["evidence"])


if __name__ == "__main__":
    unittest.main()
