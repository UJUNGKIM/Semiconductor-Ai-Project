import json
import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from dependency_security import (  # noqa: E402
    SCANNER_VERSION,
    summarize_pip_audit,
    validate_dependency_security_report,
)


class DependencySecurityTests(unittest.TestCase):
    def test_duplicate_advisories_are_merged_without_descriptions(self):
        raw = {
            "dependencies": [
                {
                    "name": "Example_Package",
                    "version": "1.0",
                    "vulns": [
                        {
                            "id": "PYSEC-1",
                            "aliases": ["CVE-1"],
                            "fix_versions": ["2.0"],
                            "description": "must not persist",
                        },
                        {
                            "id": "PYSEC-1",
                            "aliases": ["GHSA-1"],
                            "fix_versions": ["2.1"],
                            "description": "must not persist either",
                        },
                    ],
                }
            ]
        }
        report = summarize_pip_audit(
            raw,
            scanner_version=SCANNER_VERSION,
            pip_version="26.2.1",
            python_version="3.11.9",
            requirements_sha256="A" * 64,
            audited_at="2026-09-18T00:00:00Z",
        )
        self.assertEqual(report["status"], "BLOCK")
        self.assertEqual(report["known_vulnerability_count"], 1)
        self.assertEqual(report["vulnerabilities"][0]["package"], "example-package")
        self.assertEqual(report["vulnerabilities"][0]["fix_versions"], ["2.0", "2.1"])
        self.assertNotIn("description", json.dumps(report["vulnerabilities"]))
        self.assertFalse(report["raw_advisory_descriptions_persisted"])
        self.assertEqual(report["scope"], "resolved_requirements_graph")
        self.assertEqual(report["input_manifest"], "requirements.txt")
        self.assertEqual(
            validate_dependency_security_report(report, minimum_distributions=1)["status"],
            "BLOCK",
        )

    def test_forged_pass_and_old_pip_are_rejected(self):
        report = summarize_pip_audit(
            {"dependencies": [{"name": "safe", "version": "1", "vulns": []}]},
            scanner_version=SCANNER_VERSION,
            pip_version="26.2.1",
            python_version="3.11.9",
            requirements_sha256="A" * 64,
            audited_at="2026-09-18T00:00:00Z",
        )
        report["known_vulnerability_count"] = 1
        with self.assertRaisesRegex(ValueError, "집계 수"):
            validate_dependency_security_report(report, minimum_distributions=1)
        report["known_vulnerability_count"] = 0
        report["pip_version"] = "24.0"
        with self.assertRaisesRegex(ValueError, "pip"):
            validate_dependency_security_report(report, minimum_distributions=1)

    def test_committed_audit_and_release_evidence(self):
        path = (
            PROJECT
            / "결과물"
            / "secom"
            / "dependency_security"
            / "dependency_security_audit.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        validation = validate_dependency_security_report(report)
        self.assertEqual(validation["status"], "PASS")
        self.assertEqual(validation["known_vulnerability_count"], 0)
        self.assertGreaterEqual(validation["audited_distribution_count"], 17)
        self.assertGreaterEqual(len(report["limitations"]), 3)
        self.assertFalse(report["full_package_inventory_persisted"])
        manifest = json.loads(
            (
                PROJECT
                / "결과물"
                / "secom"
                / "environment_provenance"
                / "environment_manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(report["requirements_sha256"], manifest["requirements_sha256"])

    def test_ci_and_dashboard_contract(self):
        workflow = (
            PROJECT / ".github" / "workflows" / "project-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn('"pip==26.2.1"', workflow)
        self.assertIn('"pip-audit==2.10.1"', workflow)
        self.assertIn("build_dependency_security_audit.py", workflow)
        builder = (PROJECT / "코드" / "secom" / "build_dependency_security_audit.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--requirement"', builder)
        self.assertNotIn('"--local"', builder)
        self.assertIn(
            "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
            workflow,
        )
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("Python 의존성 취약점 감사", app)
        self.assertIn("secom_dependency_security_audit.json", app)


if __name__ == "__main__":
    unittest.main()
