"""Tests for the WM-811K deployment model registry."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

from build_wm811k_model_registry import (  # noqa: E402
    build_registry,
    sha256_artifact,
    validate_registry,
)


class Wm811kModelRegistryTests(unittest.TestCase):
    def test_text_artifact_hash_is_cross_platform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lf_path = Path(directory) / "lf.json"
            crlf_path = Path(directory) / "crlf.json"
            lf_path.write_bytes(b'{\n  "value": 1\n}\n')
            crlf_path.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
            self.assertEqual(sha256_artifact(lf_path), sha256_artifact(crlf_path))

    def test_champion_is_hash_linked_and_challenger_is_not_promoted(self) -> None:
        manifest, decisions = build_registry(PROJECT_DIR)
        saved = json.loads(
            (
                PROJECT_DIR / "결과물" / "wm811k" / "model_registry"
                / "champion_manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest, saved)
        validation = validate_registry(manifest, PROJECT_DIR)
        self.assertEqual(validation["status"], "validated")
        self.assertTrue(all(validation["checks"].values()))
        self.assertEqual(manifest["status"], "champion_frozen")
        self.assertEqual(
            manifest["deployment"]["model_id"],
            "wm811k_ce_sqrt_balanced_seed42",
        )
        self.assertFalse(manifest["deployment"]["selection_uses_test"])
        self.assertFalse(manifest["latest_challenger_gate"]["eligible"])
        self.assertFalse(manifest["automatic_promotion_performed"])
        self.assertEqual(len(decisions), 4)
        self.assertEqual(decisions[0]["status"], "deployed")
        self.assertTrue(all(item["status"] == "rejected" for item in decisions[1:]))
        self.assertEqual(
            manifest["latest_challenger_gate"]["candidate"],
            "baseline_robust_ensemble_50_50_bias_010",
        )
        self.assertTrue(
            manifest["latest_challenger_gate"]["validation_gate_passed"]
        )
        self.assertTrue(manifest["latest_challenger_gate"]["test_evaluated"])
        self.assertEqual(
            manifest["latest_challenger_gate"]["failed_guardrails"],
            ["none_recall_guardrail"],
        )

    def test_dashboard_distinguishes_champion_from_research_baseline(self) -> None:
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("WM_MODEL_REGISTRY_DIR", app)
        self.assertIn("운영 모델 상태", app)
        self.assertIn("Champion·Challenger 결정 기록", app)
        self.assertIn("운영 체크포인트가 아닙니다", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
