from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard_ui.secom import (
    audit_events_frame,
    build_profile_comparison,
    diagnosis_error_guidance,
    filter_review_queue,
    prepare_review_queue,
    review_filter_options,
)

PROJECT = Path(__file__).resolve().parents[1]
SECOM_CODE = PROJECT / "코드" / "secom"
if str(SECOM_CODE) not in sys.path:
    sys.path.insert(0, str(SECOM_CODE))

from production_audit_store import append_store_event, read_store, verify_store


class DashboardSecomUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = prepare_review_queue(
            pd.DataFrame(
                {
                    "행 번호": [1, 2, 3],
                    "처리 권고": ["불량 긴급 검토", "자동 처리 후보", "OOD 검토"],
                    "종합 판정": ["두 모델 모두 불량", "두 모델 모두 정상", "XGBoost만 불량"],
                    "입력 신뢰도": ["학습 분포 내부", "학습 분포 내부", "분포 이탈"],
                }
            )
        )

    def test_filter_labels_include_counts_and_search_is_literal(self) -> None:
        options = review_filter_options(self.queue)
        self.assertEqual(options["전체 3"], "all")
        self.assertEqual(options["불량 긴급 1"], "urgent")
        self.assertEqual(len(filter_review_queue(self.queue, "review")), 2)
        self.assertEqual(
            filter_review_queue(self.queue, "all", "OOD").iloc[0]["행 번호"], 3
        )

    def test_priority_has_non_color_signal(self) -> None:
        self.assertEqual(self.queue["우선도"].tolist(), ["● 긴급", "○ 일반", "◆ 확인"])

    def test_collapsed_sidebar_keeps_reopen_control_visible(self) -> None:
        source = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn('[data-testid="stExpandSidebarButton"]', source)
        self.assertIn('[data-testid="stToolbarActions"]', source)
        self.assertNotIn(
            '[data-testid="stToolbar"],\n        [data-testid="stStatusWidget"]',
            source,
        )

    def test_profile_comparison_reuses_scores(self) -> None:
        bundles = {
            "CatBoost": {"operating_thresholds": {"balanced": 0.5, "recall": 0.2}},
            "XGBoost": {"operating_thresholds": {"balanced": 0.5, "recall": 0.2}},
        }
        frame = build_profile_comparison(
            np.array([0.1, 0.6]),
            np.array([0.3, 0.7]),
            bundles,
            {"균형": "balanced", "재현": "recall"},
        )
        self.assertEqual(frame.loc[0, "두 모델 불량"], 1)
        self.assertEqual(frame.loc[1, "두 모델 불량"], 1)
        self.assertEqual(frame.loc[1, "모델 불일치"], 1)

    def test_error_guidance_and_audit_decode(self) -> None:
        guidance = diagnosis_error_guidance("센서 열 590개가 필요하지만 10개")
        self.assertTrue(any("590" in item for item in guidance))
        ledger = pd.DataFrame(
            [
                {
                    "sequence": 1,
                    "recorded_at": "2026-01-01T00:00:00+09:00",
                    "event_type": "BATCH_DIAGNOSIS",
                    "actor": "system",
                    "batch_id": "run-1",
                    "payload_json": json.dumps({"row_count": 3, "both_defect": 1}),
                }
            ]
        )
        decoded = audit_events_frame(ledger, "BATCH_DIAGNOSIS")
        self.assertEqual(decoded.iloc[0]["row_count"], 3)

    def test_dashboard_events_survive_sqlite_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "dashboard.sqlite3"
            append_store_event(
                database,
                event_type="BATCH_DIAGNOSIS",
                actor="dashboard",
                batch_id="run-1",
                payload={"row_count": 3, "result_digest": "a" * 64},
            )
            append_store_event(
                database,
                event_type="MANUAL_NOTE",
                actor="engineer_01",
                batch_id="run-1",
                payload={"row_number": 2, "decision": "불량 확인"},
            )
            restored = audit_events_frame(read_store(database))
            self.assertEqual(restored["event_type"].tolist(), ["BATCH_DIAGNOSIS", "MANUAL_NOTE"])
            self.assertTrue(verify_store(database)["append_only_triggers_present"])


if __name__ == "__main__":
    unittest.main()
