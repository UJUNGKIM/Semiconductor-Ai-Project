import json
import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from dependency_lock import inspect_pylock
from runtime_environment import build_runtime_environment_report, report_json_bytes


class RuntimeEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lock = inspect_pylock(
            PROJECT / "pylock.streamlit.toml",
            PROJECT / "requirements.txt",
            expected_lock_name="pylock.streamlit.toml",
            allowed_hosts=("files.pythonhosted.org",),
        )
        cls.installed = {
            name: {"name": name, "version": item["version"]}
            for name, item in lock["locked_packages"].items()
        }

    def test_exact_streamlit_runtime_passes(self):
        report = build_runtime_environment_report(
            PROJECT,
            installed=self.installed,
            python_version="3.11.9",
            platform_system="Linux",
            platform_machine="x86_64",
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["expected"]["direct_dependency_count"], 18)
        self.assertEqual(report["expected"]["locked_dependency_count"], 99)
        self.assertEqual(len(report["runtime_fingerprint_sha256"]), 64)

    def test_version_drift_and_python_drift_warn(self):
        installed = dict(self.installed)
        installed["streamlit"] = {"name": "streamlit", "version": "0.0.1"}
        report = build_runtime_environment_report(
            PROJECT,
            installed=installed,
            python_version="3.12.0",
            platform_system="Linux",
            platform_machine="x86_64",
        )
        self.assertEqual(report["status"], "WARN")
        self.assertFalse(report["checks"]["python_version_matches"])
        self.assertFalse(report["checks"]["direct_dependencies_match"])
        self.assertFalse(report["checks"]["locked_dependencies_match"])
        self.assertEqual(
            report["direct_drift"]["mismatched"][0]["package"], "streamlit"
        )

    def test_report_excludes_unlocked_packages_and_secrets(self):
        installed = dict(self.installed)
        installed["private-token-provider"] = {
            "name": "private-token-provider",
            "version": "secret-value",
        }
        payload = report_json_bytes(
            build_runtime_environment_report(
                PROJECT,
                installed=installed,
                python_version="3.11.9",
                platform_system="Linux",
                platform_machine="x86_64",
            )
        )
        decoded = json.loads(payload)
        self.assertEqual(decoded["status"], "PASS")
        self.assertNotIn(b"private-token-provider", payload)
        self.assertNotIn(b"secret-value", payload)

    def test_dashboard_exposes_runtime_attestation(self):
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("현재 실행 프로세스 대조", app)
        self.assertIn("secom_streamlit_runtime_report.json", app)
        self.assertIn("build_runtime_environment_report", app)


if __name__ == "__main__":
    unittest.main()
