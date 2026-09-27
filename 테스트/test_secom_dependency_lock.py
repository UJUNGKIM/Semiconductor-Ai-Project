import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from dependency_lock import inspect_pylock, validate_dependency_lock_manifest


class DependencyLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock_path = PROJECT / "pylock.windows.toml"
        cls.result_dir = PROJECT / "결과물" / "secom" / "dependency_lock"
        cls.manifest = json.loads(
            (cls.result_dir / "windows_lock_manifest.json").read_text(encoding="utf-8")
        )

    def test_committed_lock_is_complete_and_matches_requirements(self):
        details = inspect_pylock(self.lock_path, PROJECT / "requirements.txt")
        self.assertTrue(details["valid"])
        self.assertEqual(details["direct_dependency_count"], 18)
        self.assertGreaterEqual(details["package_count"], 17)
        self.assertGreaterEqual(details["wheel_count"], details["package_count"])
        self.assertEqual(details["sdist_count"], 0)
        self.assertTrue(details["all_artifacts_hashed"])

    def test_saved_manifest_is_truthful_and_reproducible(self):
        result = validate_dependency_lock_manifest(self.manifest, PROJECT)
        self.assertTrue(result["manifest_valid"])
        self.assertEqual(self.manifest["status"], "PASS")
        self.assertTrue(self.manifest["experimental_tooling"])
        self.assertEqual(self.manifest["platform_system"], "Windows")
        self.assertEqual(self.manifest["python_version"], "3.11.9")
        self.assertFalse(self.manifest["vulnerability_scan_performed"])

    def test_tampered_direct_version_is_rejected(self):
        content = self.lock_path.read_text(encoding="utf-8")
        tampered = content.replace('version = "2.3.3"', 'version = "0.0.0"', 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pylock.windows.toml"
            path.write_text(tampered, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "직접 의존성"):
                inspect_pylock(path, PROJECT / "requirements.txt")

    def test_missing_hash_and_query_string_are_rejected(self):
        content = self.lock_path.read_text(encoding="utf-8")
        no_hash = content.replace('sha256 = "', 'sha256 = "missing-', 1)
        with_query = content.replace(
            '.whl"\n\n[packages.wheels.hashes]',
            '.whl?token=secret"\n\n[packages.wheels.hashes]',
            1,
        )
        for altered, message in ((no_hash, "SHA-256"), (with_query, "URL")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "pylock.windows.toml"
                path.write_text(altered, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    inspect_pylock(path, PROJECT / "requirements.txt")

    def test_manifest_hash_tampering_is_rejected(self):
        altered = copy.deepcopy(self.manifest)
        altered["lock_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_dependency_lock_manifest(altered, PROJECT)

    def test_dashboard_and_release_contract(self):
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        release = (PROJECT / "코드" / "secom" / "release_readiness.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Windows 전이 의존성 잠금", app)
        self.assertIn("WINDOWS_DEPENDENCY_LOCK", release)
        self.assertTrue((self.result_dir / "summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
