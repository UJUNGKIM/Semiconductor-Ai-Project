"""Regression tests for SECOM safety and decision-support features."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "코드" / "secom"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from advanced_diagnosis import (  # noqa: E402
    assess_ood,
    assign_failure_signature,
    build_html_report,
    build_review_queue,
    constrained_counterfactual,
)
from train_compare_models import load_data  # noqa: E402


class SecomAdvancedDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_dir = PROJECT_DIR / "결과물" / "secom" / "advanced_diagnostics"
        cls.reference = joblib.load(cls.result_dir / "advanced_reference.joblib")
        cls.metadata = json.loads(
            (cls.result_dir / "metadata.json").read_text(encoding="utf-8")
        )
        cls.X_raw, _, _ = load_data(PROJECT_DIR)
        model_dir = PROJECT_DIR / "결과물" / "secom" / "dual_model_results"
        cls.bundles = {
            model_name: joblib.load(
                model_dir / f"{model_name.lower()}_model.joblib"
            )
            for model_name in ("CatBoost", "XGBoost")
        }

    def test_train_only_reference_and_signature_contract(self) -> None:
        self.assertEqual(self.metadata["fit_scope"], "fixed_train_only")
        self.assertEqual(self.metadata["train_rows"], 1253)
        self.assertEqual(self.metadata["train_defect_rows"], 83)
        self.assertEqual(self.metadata["retained_features"], 446)
        self.assertFalse(self.metadata["physical_causality_claimed"])
        self.assertEqual(self.metadata["signature_clusters"], 4)
        self.assertGreater(self.metadata["signature_silhouette"], 0.3)
        self.assertGreater(
            self.metadata["signature_bootstrap_stability_ari"], 0.75
        )
        signatures = pd.read_csv(self.result_dir / "failure_signatures.csv")
        self.assertEqual(len(signatures), 4)
        self.assertEqual(int(signatures["training_defect_rows"].sum()), 83)

    def test_ood_detects_extreme_rows(self) -> None:
        ordinary = assess_ood(self.X_raw.iloc[:30], self.reference)
        self.assertEqual(len(ordinary), 30)
        self.assertTrue(np.isfinite(ordinary["OOD 점수"]).all())
        self.assertGreater(
            int((ordinary["OOD 상태"] == "in_distribution").sum()), 20
        )

        extreme = self.X_raw.iloc[[0]].copy()
        extreme.loc[:, self.reference["input_features"]] = (
            np.asarray(self.reference["q99"])
            + 100 * np.asarray(self.reference["iqr"])
        )
        assessed = assess_ood(extreme, self.reference).iloc[0]
        self.assertEqual(assessed["OOD 상태"], "out_of_distribution")
        self.assertGreater(float(assessed["범위 이탈률"]), 0.9)

    def test_failure_signature_assignment(self) -> None:
        signature = assign_failure_signature(
            self.X_raw.iloc[[0]], self.bundles["CatBoost"], self.reference
        )
        self.assertIn(signature["cluster"], range(4))
        self.assertTrue(signature["pattern"].startswith("패턴 "))
        self.assertGreater(signature["training_defect_rows"], 0)
        self.assertGreater(len(signature["top_risk_features"]), 0)

    def test_counterfactual_respects_observed_constraints(self) -> None:
        result = constrained_counterfactual(
            self.X_raw.iloc[[243]],
            self.bundles,
            self.reference,
            "balanced_f2",
        )
        self.assertLessEqual(len(result["changes"]), 4)
        self.assertLessEqual(result["objective_after"], result["objective_before"])
        for change in result["changes"]:
            index = self.reference["feature_index"][change["feature"]]
            self.assertGreaterEqual(change["to"], self.reference["q01"][index])
            self.assertLessEqual(change["to"], self.reference["q99"][index])
            self.assertLessEqual(change["normalized_change"], 3.0)
        self.assertIn("물리적 공정 처방", result["warning"])

    def test_review_queue_and_escaped_html_report(self) -> None:
        predictions = pd.DataFrame(
            {
                "행 번호": [0, 1, 2],
                "CatBoost 점수": [0.2, 0.01, 0.08],
                "CatBoost 임계값": [0.07, 0.07, 0.07],
                "CatBoost 판정": ["불량", "정상", "불량"],
                "XGBoost 점수": [0.2, 0.01, 0.01],
                "XGBoost 임계값": [0.02, 0.02, 0.02],
                "XGBoost 판정": ["불량", "정상", "정상"],
                "종합 판정": [
                    "두 모델 모두 불량",
                    "두 모델 모두 정상",
                    "CatBoost만 불량",
                ],
                "우선 확인 점수": [10.0, 0.5, 1.2],
            }
        )
        ood = assess_ood(self.X_raw.iloc[:3], self.reference)
        queue = build_review_queue(predictions, ood)
        self.assertEqual(len(queue), 3)
        self.assertIn("불량 긴급 검토", set(queue["처리 권고"]))
        report = build_html_report(
            source_name="<script>alert(1)</script>",
            profile_label="균형형",
            predictions=predictions,
            ood=ood,
            review_queue=queue,
            model_hashes=self.reference["model_hashes"],
        ).decode("utf-8")
        self.assertIn("SECOM 설명가능 AI 진단 감사 보고서", report)
        self.assertNotIn("<script>alert(1)</script>", report)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", report)
        self.assertIn("SHA-256", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
