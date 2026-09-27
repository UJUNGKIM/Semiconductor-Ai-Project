import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from external_validation import (  # noqa: E402
    FEATURE_COLUMNS,
    build_protocol,
    evaluate_external_frame,
    parse_declarations_bytes,
    parse_external_csv_bytes,
    per_lot_csv_bytes,
    summary_json_bytes,
    validate_declarations,
    validate_external_frame,
    validated_summary_is_release_evidence,
    wilson_interval,
)


DECLARATIONS = {
    "prospective_collection": True,
    "independent_source": True,
    "labels_finalized_before_scoring": True,
    "model_outputs_hidden_from_labelers": True,
    "data_collected_after_protocol_freeze": True,
}


class ExternalValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = build_protocol(PROJECT)

    def sample_frame(self, rows: int = 3) -> pd.DataFrame:
        sensors = pd.read_csv(
            PROJECT / "데이터" / "SECOM 데이터셋" / "raw" / "secom.data",
            sep=r"\s+",
            header=None,
            names=FEATURE_COLUMNS,
            nrows=rows,
        )
        labels = pd.read_csv(
            PROJECT / "데이터" / "SECOM 데이터셋" / "raw" / "secom_labels.data",
            sep=r"\s+",
            header=None,
            nrows=rows,
        ).iloc[:, 0].map({-1: 0, 1: 1})
        frame = sensors.copy()
        frame.insert(0, "label", labels.to_numpy())
        frame.insert(0, "captured_at", ["2026-10-01T09:00:00+09:00"] * rows)
        frame.insert(0, "lot_id", [f"lot-{index % 3}" for index in range(rows)])
        frame.insert(0, "sample_id", [f"external-{index}" for index in range(rows)])
        return frame

    def test_protocol_freezes_models_thresholds_and_minimums(self):
        self.assertEqual(len(self.protocol["model_artifact_sha256"]), 2)
        self.assertEqual(set(self.protocol["operating_thresholds"]), {"CatBoost", "XGBoost"})
        self.assertEqual(len(self.protocol["expected_sensor_columns"]), 590)
        self.assertGreaterEqual(self.protocol["minimum_evidence"]["distinct_lots"], 3)
        self.assertIn("threshold_retuning", self.protocol["selection_prohibitions"])

    def test_input_contract_rejects_duplicate_ids_bad_time_and_bad_labels(self):
        frame = self.sample_frame()
        frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
        with self.assertRaisesRegex(ValueError, "중복"):
            validate_external_frame(frame, self.protocol)
        frame = self.sample_frame()
        frame.loc[0, "captured_at"] = "2026-10-01 09:00:00"
        with self.assertRaisesRegex(ValueError, "시간대"):
            validate_external_frame(frame, self.protocol)
        frame = self.sample_frame()
        frame.loc[0, "label"] = -1
        with self.assertRaisesRegex(ValueError, "0\(정상\)/1"):
            validate_external_frame(frame, self.protocol)

    def test_incomplete_independence_declaration_is_rejected(self):
        incomplete = dict(DECLARATIONS)
        incomplete["model_outputs_hidden_from_labelers"] = False
        with self.assertRaisesRegex(ValueError, "선언"):
            validate_declarations(incomplete)

    def test_uploaded_declarations_are_strict_and_truthful(self):
        valid = parse_declarations_bytes(json.dumps(DECLARATIONS).encode("utf-8"))
        self.assertEqual(valid, DECLARATIONS)
        false_claim = dict(DECLARATIONS)
        false_claim["independent_source"] = False
        with self.assertRaisesRegex(ValueError, "참이 아닙니다"):
            parse_declarations_bytes(json.dumps(false_claim).encode("utf-8"))
        extra = {**DECLARATIONS, "approved": True}
        with self.assertRaisesRegex(ValueError, "알 수 없음"):
            parse_declarations_bytes(json.dumps(extra).encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "형식"):
            parse_declarations_bytes(b"{not-json")

    def test_uploaded_csv_limits_and_aggregate_exports(self):
        frame = self.sample_frame(3)
        parsed = parse_external_csv_bytes(frame.to_csv(index=False).encode("utf-8"))
        self.assertEqual(parsed.shape, frame.shape)
        with self.assertRaisesRegex(ValueError, "데이터 행"):
            parse_external_csv_bytes(",".join(frame.columns).encode("utf-8"))
        summary, per_lot = evaluate_external_frame(
            frame, DECLARATIONS, PROJECT, self.protocol
        )
        summary_text = summary_json_bytes(summary).decode("utf-8")
        self.assertNotIn("feature_0", summary_text)
        self.assertNotIn("external-0", summary_text)
        per_lot.loc[0, "lot_id"] = "=FORMULA()"
        exported = per_lot_csv_bytes(per_lot).decode("utf-8-sig")
        self.assertIn("'=FORMULA()", exported)
        self.assertNotIn("feature_0", exported)

    def test_known_rows_are_blocked_not_misrepresented_as_external(self):
        summary, per_lot = evaluate_external_frame(
            self.sample_frame(9), DECLARATIONS, PROJECT, self.protocol
        )
        self.assertEqual(summary["validation_status"], "BLOCKED")
        self.assertEqual(summary["known_secom_exact_row_overlap_count"], 9)
        self.assertFalse(summary["thresholds_retuned"])
        self.assertFalse(summary["model_or_threshold_selection_used"])
        self.assertFalse(summary["raw_rows_persisted"])
        self.assertEqual(len(per_lot), 3)
        self.assertFalse(validated_summary_is_release_evidence(summary, self.protocol))

    def test_release_evidence_rejects_forged_or_incomplete_summary(self):
        forged = {
            "validation_status": "PASS",
            "protocol_id": self.protocol["protocol_id"],
            "gates": {"minimum_rows": True},
        }
        self.assertFalse(validated_summary_is_release_evidence(forged, self.protocol))
        sophisticated_forgery = {
            "validation_status": "PASS",
            "protocol_id": self.protocol["protocol_id"],
            **DECLARATIONS,
            "model_or_threshold_selection_used": False,
            "thresholds_retuned": False,
            "raw_rows_persisted": False,
            "known_secom_exact_row_overlap_count": 0,
            "model_artifact_sha256": self.protocol["model_artifact_sha256"],
            "operating_thresholds": self.protocol["operating_thresholds"],
            "dataset_digest": "A" * 64,
            "distinct_lots": 3,
            "overall": {
                "rows": 10,
                "defects": 1,
                "recall_wilson_95": {"lower": 1.0},
                "false_positive_rate_wilson_95": {"upper": 0.0},
                "review_rate_wilson_95": {"upper": 0.0},
            },
            "gates": {
                "minimum_rows": True,
                "minimum_distinct_lots": True,
                "minimum_defects": True,
                "no_known_secom_row_overlap": True,
                "recall_lower_bound": True,
                "false_positive_upper_bound": True,
                "review_rate_upper_bound": True,
            },
        }
        self.assertFalse(
            validated_summary_is_release_evidence(sophisticated_forgery, self.protocol)
        )
        lower, upper = wilson_interval(8, 10)
        self.assertLess(lower, 0.8)
        self.assertGreater(upper, 0.8)

    def test_committed_protocol_is_waiting_for_real_data(self):
        output = PROJECT / "결과물" / "secom" / "external_validation"
        protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol, self.protocol)
        self.assertEqual(status["status"], "AWAITING_INDEPENDENT_DATA")
        self.assertFalse(status["production_claim_allowed"])
        self.assertFalse((output / "validated_summary.json").exists())
        template = pd.read_csv(output / "external_lot_template.csv")
        self.assertEqual(len(template), 0)
        self.assertEqual(len(template.columns), 596)
        declarations = json.loads(
            (output / "declarations_template.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(declarations), set(DECLARATIONS))
        self.assertTrue(all(value is False for value in declarations.values()))


if __name__ == "__main__":
    unittest.main()
