import json
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from release_readiness import (
    build_release_readiness,
    render_readiness_html,
    sha256_file,
    sha256_text_file,
)


class ReleaseReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result_dir = PROJECT / "결과물" / "secom" / "release_readiness"
        cls.report, cls.manifest = build_release_readiness(PROJECT)

    def test_expected_demo_and_production_decisions(self):
        self.assertEqual(self.report["demo_readiness"], "READY_WITH_WARNINGS")
        self.assertEqual(self.report["production_readiness"], "BLOCKED")
        blockers = {
            item["check_id"]
            for item in self.report["checks"]
            if item["status"] == "BLOCK"
        }
        self.assertEqual(
            blockers,
            {
                "SENSOR_SEMANTICS",
                "PROSPECTIVE_EXTERNAL_VALIDATION",
                "PERSISTENT_AUDIT_STORE",
            },
        )

    def test_release_id_is_deterministic(self):
        second_report, second_manifest = build_release_readiness(PROJECT)
        self.assertEqual(self.report["release_id"], second_report["release_id"])
        self.assertEqual(self.manifest, second_manifest)

    def test_deployed_model_hashes_are_verified(self):
        model_dir = PROJECT / "결과물" / "secom" / "dual_model_results"
        self.assertEqual(
            self.manifest["component_hashes"]["catboost_model"],
            sha256_file(model_dir / "catboost_model.joblib"),
        )
        self.assertFalse(self.manifest["automatic_retraining"])
        self.assertFalse(self.manifest["automatic_model_promotion"])

    def test_tamper_evident_audit_chain_is_release_evidence(self):
        checks = {item["check_id"]: item for item in self.report["checks"]}
        self.assertEqual(checks["TAMPER_EVIDENT_AUDIT_CHAIN"]["status"], "PASS")
        self.assertIn("audit_policy", self.manifest["component_hashes"])
        self.assertEqual(checks["PERSISTENT_AUDIT_STORE"]["status"], "BLOCK")

    def test_shap_reliability_is_post_selection_release_evidence(self):
        checks = {item["check_id"]: item for item in self.report["checks"]}
        self.assertEqual(checks["SHAP_RELIABILITY"]["status"], "PASS")
        self.assertIn("shap_reliability", self.manifest["component_hashes"])

    def test_model_failover_is_human_controlled_release_evidence(self):
        checks = {item["check_id"]: item for item in self.report["checks"]}
        self.assertEqual(checks["MODEL_FAILOVER_DRILL"]["status"], "PASS")
        self.assertIn("model_registry", self.manifest["component_hashes"])
        self.assertIn("failover_drill", self.manifest["component_hashes"])

    def test_data_lineage_is_reconstructed_release_evidence(self):
        checks = {item["check_id"]: item for item in self.report["checks"]}
        self.assertEqual(checks["DATA_LINEAGE"]["status"], "PASS")
        self.assertIn("data_lineage_manifest", self.manifest["component_hashes"])
        self.assertIn("data_lineage_validation", self.manifest["component_hashes"])

    def test_environment_reproducibility_is_release_evidence(self):
        checks = {item["check_id"]: item for item in self.report["checks"]}
        self.assertEqual(checks["ENVIRONMENT_REPRODUCIBILITY"]["status"], "PASS")
        self.assertIn("environment_manifest", self.manifest["component_hashes"])
        self.assertIn("environment_validation", self.manifest["component_hashes"])
        self.assertIn("software_bom", self.manifest["component_hashes"])
        self.assertEqual(checks["DEPENDENCY_SECURITY_AUDIT"]["status"], "PASS")
        self.assertIn("dependency_security_audit", self.manifest["component_hashes"])
        self.assertEqual(checks["WINDOWS_DEPENDENCY_LOCK"]["status"], "PASS")
        self.assertIn("windows_dependency_lock", self.manifest["component_hashes"])
        self.assertIn("dependency_lock_manifest", self.manifest["component_hashes"])
        self.assertEqual(checks["LINUX_CI_DEPENDENCY_LOCK"]["status"], "PASS")
        self.assertEqual(checks["STREAMLIT_DEPENDENCY_LOCK"]["status"], "WARN")
        self.assertIn("linux_ci_dependency_lock", self.manifest["component_hashes"])
        self.assertIn("streamlit_dependency_lock", self.manifest["component_hashes"])
        self.assertIn("linux_dependency_lock_manifest", self.manifest["component_hashes"])
        self.assertEqual(checks["SOURCE_SECURITY_AUDIT"]["status"], "PASS")
        self.assertIn("source_security_audit", self.manifest["component_hashes"])
        self.assertEqual(checks["SENSOR_FAILURE_CONTAINMENT"]["status"], "PASS")
        self.assertIn(
            "sensor_failure_containment", self.manifest["component_hashes"]
        )

    def test_text_component_hash_is_cross_platform(self):
        registry_path = (
            PROJECT / "결과물" / "secom" / "model_failover" / "model_registry.json"
        )
        self.assertEqual(
            self.manifest["component_hashes"]["model_registry"],
            sha256_text_file(registry_path),
        )

    def test_html_escapes_evidence(self):
        altered = json.loads(json.dumps(self.report))
        altered["checks"][0]["evidence"] = "<script>alert(1)</script>"
        rendered = render_readiness_html(altered, self.manifest)
        self.assertNotIn("<script>alert(1)</script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_artifacts_and_dashboard_contract(self):
        for name in (
            "release_readiness.json",
            "release_manifest.json",
            "readiness_matrix.csv",
            "readiness_matrix.png",
            "release_readiness.html",
            "summary.md",
        ):
            self.assertTrue((self.result_dir / name).is_file(), name)
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn('"릴리스 준비도"', app)
        self.assertIn("실제 반도체 생산 배포는 승인되지 않았습니다.", app)


if __name__ == "__main__":
    unittest.main()
