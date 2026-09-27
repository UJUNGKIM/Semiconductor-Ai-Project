import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from shap_reliability import (
    build_masked_matrices,
    explanation_similarity,
    perturb_within_reference,
)


class ShapReliabilityTests(unittest.TestCase):
    def test_similarity_has_expected_overlap_and_sign(self):
        reference = np.array([4.0, -3.0, 0.2, 0.1])
        candidate = np.array([0.1, -2.0, 5.0, 0.2])
        result = explanation_similarity(reference, candidate, top_k=2)
        self.assertAlmostEqual(result["top_k_jaccard"], 1 / 3)
        self.assertEqual(result["top_k_sign_agreement"], 1.0)
        self.assertTrue(-1 <= result["absolute_rank_spearman"] <= 1)

    def test_perturbation_is_deterministic_bounded_and_non_mutating(self):
        values = np.array([[0.0, 1.0], [0.5, -1.0]])
        original = values.copy()
        args = (values, np.ones(2), np.array([-0.2, -0.5]), np.array([0.8, 1.2]))
        first = perturb_within_reference(
            *args, noise_fraction=0.1, rng=np.random.default_rng(42)
        )
        second = perturb_within_reference(
            *args, noise_fraction=0.1, rng=np.random.default_rng(42)
        )
        np.testing.assert_allclose(first, second)
        np.testing.assert_array_equal(values, original)
        self.assertTrue((first >= args[2]).all())
        self.assertTrue((first <= args[3]).all())

    def test_masking_replaces_top_features_without_mutating_input(self):
        values = np.array([[10.0, 20.0, 30.0, 40.0]])
        shap_values = np.array([[0.1, -9.0, 0.2, 8.0]])
        replacement = np.zeros(4)
        original = values.copy()
        top, random_matrices = build_masked_matrices(
            values,
            shap_values,
            replacement,
            top_k=2,
            random_repeats=3,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(top, [[10.0, 0.0, 30.0, 0.0]])
        self.assertEqual(len(random_matrices), 3)
        for matrix in random_matrices:
            self.assertEqual(int((matrix == 0).sum()), 2)

    def test_committed_artifact_and_dashboard_contract(self):
        result_dir = PROJECT / "결과물" / "secom" / "shap_reliability"
        summary = json.loads(
            (result_dir / "shap_reliability_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["evaluation_scope"], "fixed_test_post_selection_audit_only")
        self.assertFalse(summary["model_selection_uses_results"])
        self.assertFalse(summary["threshold_selection_uses_results"])
        self.assertFalse(summary["physical_causality_claimed"])
        self.assertEqual(summary["test_rows"], 314)
        self.assertEqual(summary["status"], "PASS")
        rows = pd.read_csv(result_dir / "per_sample_reliability.csv")
        self.assertEqual(len(rows), 314 * 2 * 3)
        self.assertTrue(rows["top_k_jaccard"].between(0, 1).all())
        for name in (
            "model_summary.csv",
            "shap_reliability_dashboard.png",
            "summary.md",
        ):
            self.assertTrue((result_dir / name).is_file(), name)
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("SHAP 설명 신뢰성 감사", app)
        self.assertIn("상위 SHAP 마스킹 승률", app)
        self.assertIn("물리적 원인", app)


if __name__ == "__main__":
    unittest.main()
