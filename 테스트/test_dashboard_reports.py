from __future__ import annotations

from io import BytesIO
import json
import unittest
from zipfile import ZipFile

import pandas as pd

from dashboard_ui.reports import build_diagnosis_bundle


class DashboardReportTests(unittest.TestCase):
    def test_diagnosis_bundle_contains_audit_files(self) -> None:
        results = pd.DataFrame({"행 번호": [1], "종합 판정": ["두 모델 모두 정상"]})
        review = pd.DataFrame({"행 번호": [1], "처리 권고": ["자동 처리 후보"]})
        ood = pd.DataFrame({"행 번호": [1], "OOD 상태": ["in_distribution"]})

        payload = build_diagnosis_bundle(
            results=results,
            review_queue=review,
            ood_results=ood,
            metadata={"source_name": "sample", "rows": 1},
            html_report=b"<html><body>report</body></html>",
        )

        with ZipFile(BytesIO(payload)) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {
                    "diagnosis_results.csv",
                    "review_queue.csv",
                    "ood_results.csv",
                    "run_metadata.json",
                    "diagnosis_report.html",
                },
            )
            metadata = json.loads(archive.read("run_metadata.json"))
            self.assertEqual(metadata["rows"], 1)


if __name__ == "__main__":
    unittest.main()
