import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from champion_challenger import evaluate_champion_challenger


class ChampionChallengerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result_dir = PROJECT / "결과물" / "secom" / "champion_challenger"
        cls.policy = json.loads(
            (cls.result_dir / "promotion_policy.json").read_text(encoding="utf-8")
        )

    def frame(self, cat, xgb, labels):
        rows = len(labels)
        return pd.DataFrame(
            {
                "batch_id": [f"batch-{index % 3}" for index in range(rows)],
                "actual_label": labels,
                "catboost_alert": cat,
                "xgboost_alert": xgb,
            }
        )

    def test_committed_oof_demo_is_non_promotional(self):
        report = json.loads(
            (self.result_dir / "oof_comparison_demo.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["decision"], "HISTORICAL_DEMO_ONLY")
        self.assertFalse(report["automatic_promotion_performed"])
        self.assertFalse(report["false_positive_guardrail_pass"])
        self.assertTrue((self.result_dir / "oof_comparison_demo.png").is_file())

    def test_one_batch_is_insufficient(self):
        labels = np.array([1] * 40 + [0] * 260)
        frame = self.frame(labels, labels, labels)
        frame["batch_id"] = "one-batch"
        report = evaluate_champion_challenger(
            frame, self.policy, allow_promotion_review=True, bootstrap_draws=100
        )
        self.assertEqual(report["decision"], "INSUFFICIENT_EVIDENCE")

    def test_clear_safe_improvement_reaches_human_review_only(self):
        labels = np.array([1] * 60 + [0] * 240)
        champion = labels.astype(bool).copy()
        champion[:25] = False
        challenger = labels.astype(bool)
        frame = self.frame(champion, challenger, labels)
        report = evaluate_champion_challenger(
            frame, self.policy, allow_promotion_review=True, bootstrap_draws=300
        )
        self.assertEqual(report["decision"], "REVIEW_PROMOTION")
        self.assertFalse(report["automatic_promotion_performed"])

    def test_false_positive_increase_keeps_champion(self):
        labels = np.array([1] * 60 + [0] * 240)
        champion = labels.astype(bool)
        challenger = labels.astype(bool).copy()
        challenger[60:140] = True
        frame = self.frame(champion, challenger, labels)
        report = evaluate_champion_challenger(
            frame, self.policy, allow_promotion_review=True, bootstrap_draws=300
        )
        self.assertEqual(report["decision"], "KEEP_CHAMPION")
        self.assertFalse(report["false_positive_guardrail_pass"])

    def test_bootstrap_is_deterministic(self):
        labels = np.array([1] * 40 + [0] * 260)
        champion = labels.astype(bool)
        challenger = champion.copy(); challenger[:10] = False; challenger[40:70] = True
        frame = self.frame(champion, challenger, labels)
        first = evaluate_champion_challenger(
            frame, self.policy, allow_promotion_review=True, bootstrap_draws=100
        )
        second = evaluate_champion_challenger(
            frame, self.policy, allow_promotion_review=True, bootstrap_draws=100
        )
        self.assertEqual(
            first["paired_bootstrap_delta_challenger_minus_champion"],
            second["paired_bootstrap_delta_challenger_minus_champion"],
        )

    def test_dashboard_contract(self):
        text = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("CatBoost–XGBoost 동일 행 비교", text)
        self.assertIn("automatic_promotion_performed", (PROJECT / "코드" / "secom" / "champion_challenger.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
