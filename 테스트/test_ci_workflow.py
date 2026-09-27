import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드"))

import verify_ci


class ContinuousIntegrationTests(unittest.TestCase):
    def test_workflow_uses_read_only_pinned_official_actions(self):
        workflow = (
            PROJECT / ".github" / "workflows" / "project-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("contents: read", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("python-version: \"3.11.9\"", workflow)
        self.assertIn(
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
            workflow,
        )
        self.assertIn(
            "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
            workflow,
        )
        self.assertIn(
            "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
            workflow,
        )
        self.assertNotIn("secrets.", workflow)
        self.assertNotIn("pull_request_target", workflow)

    def test_workflow_runs_cross_platform_verifier(self):
        workflow = (
            PROJECT / ".github" / "workflows" / "project-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("python 코드/verify_ci.py", workflow)
        self.assertIn(
            "python -m pip install --requirement pylock.github-actions.toml",
            workflow,
        )
        self.assertIn("verify_committed_linux_locks.py", workflow)
        self.assertIn('python -m pip install --upgrade "pip==26.2.1"', workflow)
        self.assertIn('python -m pip install "pip-audit==2.10.1"', workflow)
        self.assertIn("build_dependency_security_audit.py", workflow)
        self.assertIn('python -m pip install "bandit==1.9.4"', workflow)
        self.assertIn("build_source_security_audit.py", workflow)
        self.assertIn('cron: "17 3 * * 1"', workflow)
        self.assertNotIn("pip install --requirement requirements.txt", workflow)
        self.assertNotIn("pip install --requirement requirements-ci.txt", workflow)
        self.assertNotIn('pip install "torch==2.14.0+cpu"', workflow)
        self.assertNotIn("run_tests.ps1", workflow)
        self.assertNotIn("run_pipeline.ps1", workflow)

    def test_ci_cpu_variants_match_primary_dependency_versions(self):
        def exact_versions(path: Path) -> dict[str, str]:
            versions = {}
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                requirement = Requirement(line)
                specifiers = list(requirement.specifier)
                self.assertEqual(len(specifiers), 1)
                self.assertEqual(specifiers[0].operator, "==")
                versions[canonicalize_name(requirement.name)] = specifiers[0].version
            return versions

        primary = exact_versions(PROJECT / "requirements.txt")
        ci = exact_versions(PROJECT / "requirements-ci.txt")
        self.assertNotIn("torch", ci)
        self.assertEqual(ci.pop("xgboost-cpu"), primary.pop("xgboost"))
        self.assertEqual(ci.pop("filelock"), "4.0.0")
        self.assertEqual(ci.pop("setuptools"), "84.0.0")
        primary.pop("torch")
        self.assertEqual(ci, primary)

    def test_linux_lock_inputs_match_ci_cpu_variants(self):
        def exact_versions(path: Path) -> dict[str, str]:
            versions = {}
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                requirement = Requirement(line)
                specifiers = list(requirement.specifier)
                self.assertEqual(len(specifiers), 1)
                self.assertEqual(specifiers[0].operator, "==")
                versions[canonicalize_name(requirement.name)] = specifiers[0].version
            return versions

        primary = exact_versions(PROJECT / "requirements.txt")
        linux = exact_versions(PROJECT / "requirements-linux-ci.txt")
        self.assertEqual(len(linux), 20)
        self.assertEqual(linux.pop("xgboost-cpu"), primary.pop("xgboost"))
        self.assertEqual(linux.pop("torch"), primary.pop("torch") + "+cpu")
        self.assertEqual(linux.pop("filelock"), "4.0.0")
        self.assertEqual(linux.pop("setuptools"), "84.0.0")
        self.assertEqual(linux, primary)

    def test_workflow_exports_linux_lock_evidence(self):
        workflow = (
            PROJECT / ".github" / "workflows" / "project-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("build_linux_dependency_locks.py", workflow)
        self.assertIn("secom-linux-dependency-lock", workflow)
        builder = (
            PROJECT / "코드" / "secom" / "build_linux_dependency_locks.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--find-links"', builder)
        self.assertIn('"--constraint"', builder)
        self.assertIn('"--refresh"', builder)
        self.assertNotIn('"--extra-index-url"', builder)

    def test_raw_contract_rejects_unexpected_shape(self):
        wrong_sensors = pd.DataFrame([[0.0]])
        labels = pd.DataFrame([[-1, "timestamp"]])
        with patch("verify_ci.pd.read_csv", side_effect=[wrong_sensors, labels]):
            with self.assertRaisesRegex(ValueError, "센서 계약"):
                verify_ci.verify_secom_raw_contract()

    def test_committed_raw_contract_and_release_evidence(self):
        verify_ci.verify_secom_raw_contract()
        verify_ci.verify_release_evidence()

    def test_local_only_exclusions_are_explicit_and_limited(self):
        self.assertEqual(len(verify_ci.LOCAL_ONLY_TESTS), 3)
        self.assertTrue(
            all(test_id.startswith("test_wm811k_") for test_id in verify_ci.LOCAL_ONLY_TESTS)
        )
        self.assertTrue(all(verify_ci.LOCAL_ONLY_TESTS.values()))

    def test_audit_artifact_uploads_do_not_mask_an_earlier_failure(self):
        workflow = (
            PROJECT / ".github" / "workflows" / "project-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "always() && hashFiles('ci-security-audit/**') != ''",
            workflow,
        )
        self.assertIn(
            "always() && hashFiles('ci-source-security-audit/**') != ''",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
