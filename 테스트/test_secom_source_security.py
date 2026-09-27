import json
import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from source_security import (  # noqa: E402
    SCANNER_VERSION,
    source_fingerprint,
    summarize_bandit,
    validate_source_security_report,
)


class SourceSecurityTests(unittest.TestCase):
    def _raw(self, results=None, *, nosec=0, errors=None):
        return {
            "results": results or [],
            "errors": errors or [],
            "metrics": {"_totals": {"loc": 123, "nosec": nosec}},
        }

    def test_issue_is_normalized_without_source_snippet(self):
        raw = self._raw(
            [
                {
                    "filename": "코드/secom/example.py",
                    "line_number": 7,
                    "test_id": "B301",
                    "test_name": "blacklist_calls",
                    "issue_severity": "MEDIUM",
                    "issue_confidence": "HIGH",
                    "issue_cwe": {"id": 502},
                    "issue_text": "unsafe",
                    "code": "secret source must not persist",
                    "more_info": "https://example.invalid/B301",
                }
            ]
        )
        report = summarize_bandit(
            raw,
            project=PROJECT,
            scanner_version=SCANNER_VERSION,
            scanned_at="2026-09-18T00:00:00Z",
        )
        self.assertEqual(report["status"], "BLOCK")
        self.assertEqual(report["issue_count"], 1)
        self.assertNotIn("secret source", json.dumps(report))
        self.assertFalse(report["source_code_snippets_persisted"])

    def test_forged_pass_and_suppression_are_rejected(self):
        report = summarize_bandit(
            self._raw(nosec=1),
            project=PROJECT,
            scanner_version=SCANNER_VERSION,
            scanned_at="2026-09-18T00:00:00Z",
        )
        self.assertEqual(report["status"], "BLOCK")
        report["status"] = "PASS"
        with self.assertRaisesRegex(ValueError, "판정"):
            validate_source_security_report(report, PROJECT)

    def test_committed_report_matches_current_source(self):
        path = (
            PROJECT / "결과물" / "secom" / "source_security" / "source_security_audit.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        validation = validate_source_security_report(report, PROJECT)
        self.assertEqual(validation["status"], "PASS")
        self.assertEqual(validation["issue_count"], 0)
        count, digest = source_fingerprint(PROJECT)
        self.assertEqual(report["scanned_file_count"], count)
        self.assertEqual(report["source_sha256"], digest)
        self.assertEqual(report["suppression_count"], 0)
        self.assertIn("dashboard_ui", report["targets"])

    def test_ci_and_dashboard_contract(self):
        workflow = (
            PROJECT / ".github" / "workflows" / "project-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn('"bandit==1.9.4"', workflow)
        self.assertIn("build_source_security_audit.py", workflow)
        self.assertIn("secom-source-security-audit", workflow)
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("Python 소스 보안 정적분석", app)
        self.assertIn("secom_source_security_audit.json", app)


if __name__ == "__main__":
    unittest.main()
