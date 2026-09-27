import copy
import json
import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from dependency_lock import validate_linux_dependency_lock_manifest


class LinuxDependencyLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result_dir = PROJECT / "결과물" / "secom" / "linux_dependency_lock"
        cls.manifest = json.loads(
            (cls.result_dir / "linux_lock_manifest.json").read_text(encoding="utf-8")
        )

    def test_committed_linux_locks_are_complete(self):
        result = validate_linux_dependency_lock_manifest(self.manifest, PROJECT)
        self.assertTrue(result["valid"])
        self.assertEqual(result["github_actions"]["direct_dependency_count"], 20)
        self.assertEqual(result["streamlit"]["direct_dependency_count"], 18)
        self.assertEqual(result["github_actions"]["sdist_count"], 0)
        self.assertEqual(result["streamlit"]["sdist_count"], 0)

    def test_ci_and_streamlit_evidence_scopes_are_distinct(self):
        self.assertTrue(
            self.manifest["github_actions"][
                "all_locked_versions_match_tested_environment"
            ]
        )
        self.assertTrue(
            self.manifest["streamlit"]["resolution_validated_on_target_platform"]
        )
        self.assertFalse(
            self.manifest["streamlit"]["installation_validated_by_this_step"]
        )

    def test_forged_streamlit_installation_claim_is_rejected(self):
        altered = copy.deepcopy(self.manifest)
        altered["streamlit"]["installation_validated_by_this_step"] = True
        with self.assertRaisesRegex(ValueError, "검증 범위"):
            validate_linux_dependency_lock_manifest(altered, PROJECT)

    def test_manifest_hash_tampering_is_rejected(self):
        altered = copy.deepcopy(self.manifest)
        altered["github_actions"]["lock_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_linux_dependency_lock_manifest(altered, PROJECT)

    def test_dashboard_release_and_ci_contract(self):
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        release = (PROJECT / "코드" / "secom" / "release_readiness.py").read_text(
            encoding="utf-8"
        )
        workflow = (PROJECT / ".github" / "workflows" / "project-ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("Linux CI·Streamlit 의존성 잠금", app)
        self.assertIn("LINUX_CI_DEPENDENCY_LOCK", release)
        self.assertIn("STREAMLIT_DEPENDENCY_LOCK", release)
        self.assertIn("pylock.github-actions.toml", workflow)
        self.assertIn("verify_committed_linux_locks.py", workflow)


if __name__ == "__main__":
    unittest.main()
