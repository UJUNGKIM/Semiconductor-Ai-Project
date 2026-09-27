import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from environment_provenance import (
    dependency_drift,
    environment_fingerprint,
    parse_pinned_requirements,
    sha256_text_file,
    validate_environment_manifest,
)
from build_environment_provenance import (
    PROJECT_FILE_NAMES,
    build_project_file_manifest,
)


class EnvironmentProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result_dir = PROJECT / "결과물" / "secom" / "environment_provenance"
        cls.manifest = json.loads(
            (cls.result_dir / "environment_manifest.json").read_text(encoding="utf-8")
        )
        cls.validation = json.loads(
            (cls.result_dir / "environment_validation.json").read_text(encoding="utf-8")
        )
        cls.sbom = json.loads(
            (cls.result_dir / "software_bom.json").read_text(encoding="utf-8")
        )

    def test_all_direct_dependencies_are_exactly_pinned(self):
        requirements = parse_pinned_requirements(PROJECT / "requirements.txt")
        self.assertEqual(len(requirements), 18)
        self.assertEqual(
            {item["canonical_name"] for item in self.manifest["direct_dependencies"]},
            set(requirements),
        )
        self.assertTrue(
            all(item["exact_match"] for item in self.manifest["direct_dependencies"])
        )

    def test_version_drift_is_detected(self):
        requirements = {"pandas": {"version": "2.3.3"}}
        installed = {"pandas": {"version": "0.0.0-drift"}}
        result = dependency_drift(requirements, installed)
        self.assertFalse(result["valid"])
        self.assertEqual(result["mismatched"][0]["package"], "pandas")

    def test_environment_fingerprint_is_order_independent(self):
        first = [
            {"name": "NumPy", "version": "1"},
            {"name": "pandas", "version": "2"},
        ]
        self.assertEqual(
            environment_fingerprint("3.11", first),
            environment_fingerprint("3.11", list(reversed(first))),
        )

    def test_text_hash_is_line_ending_independent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            left = Path(temp_dir) / "left.txt"
            right = Path(temp_dir) / "right.txt"
            left.write_bytes(b"alpha\nbeta\n")
            right.write_bytes(b"alpha\r\nbeta\r\n")
            self.assertEqual(sha256_text_file(left), sha256_text_file(right))

    def test_saved_manifest_matches_project_files(self):
        result = validate_environment_manifest(self.manifest, PROJECT)
        self.assertTrue(result["valid"])
        self.assertFalse(result["vulnerability_scan_performed"])

    def test_manifest_covers_dashboard_and_ci_sources(self):
        required = {
            ".github/workflows/project-ci.yml",
            "app.py",
            "dashboard_ui/access.py",
            "dashboard_ui/components.py",
            "dashboard_ui/independent_validation.py",
            "dashboard_ui/navigation.py",
            "dashboard_ui/reports.py",
            "dashboard_ui/secom.py",
            "코드/verify_ci.py",
            "코드/secom/environment_provenance.py",
            "코드/wm811k/diagnose_wm811k.py",
            "코드/wm811k/hash_evidence.py",
            "코드/wm811k/validate_wm811k_gradient_shap_results.py",
            "코드/wm811k/wafer_shape.py",
            "코드/wm811k/wafer_cause_guidance.py",
        }
        self.assertTrue(required.issubset(PROJECT_FILE_NAMES))
        records = build_project_file_manifest(PROJECT)
        self.assertEqual(
            {item["path"] for item in records},
            set(PROJECT_FILE_NAMES),
        )

    def test_runtime_imports_outside_source_scan_are_hashed(self):
        """Dashboard and WM-811K modules the app imports must all be listed.

        `코드/secom` is covered by the source-security fingerprint, so only
        `dashboard_ui` and `코드/wm811k` imports are required here.
        """
        search = (PROJECT, PROJECT / "코드" / "secom", PROJECT / "코드" / "wm811k")

        def resolve(name):
            for base in search:
                for candidate in (
                    base.joinpath(*name.split(".")).with_suffix(".py"),
                    base.joinpath(*name.split("."), "__init__.py"),
                ):
                    if candidate.is_file():
                        return candidate
            return None

        seen, stack = set(), [PROJECT / "app.py"]
        while stack:
            path = stack.pop()
            if path in seen:
                continue
            seen.add(path)
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
                else:
                    continue
                stack.extend(target for target in map(resolve, names) if target)
        runtime = {
            path.relative_to(PROJECT).as_posix()
            for path in seen
            if path.relative_to(PROJECT).parts[0] == "dashboard_ui"
            or path.relative_to(PROJECT).parts[:2] == ("코드", "wm811k")
        }
        self.assertIn("dashboard_ui/access.py", runtime)
        self.assertEqual(runtime - set(PROJECT_FILE_NAMES), set())
        for name in PROJECT_FILE_NAMES:
            self.assertNotIn(Path(name).parts[0], {"테스트", "노트북", "결과물"}, name)
            self.assertIn(Path(name).suffix, {".py", ".yml", ".txt", ".ps1"}, name)

    def test_validation_does_not_claim_vulnerability_scan(self):
        self.assertEqual(self.validation["status"], "PASS")
        self.assertTrue(all(self.validation["checks"].values()))
        self.assertFalse(self.validation["vulnerability_scan_performed"])
        self.assertIsNone(self.validation["known_vulnerability_count"])
        self.assertFalse(self.sbom["vulnerability_information_included"])
        self.assertGreaterEqual(
            self.sbom["component_count"], self.validation["direct_dependency_count"]
        )

    def test_dashboard_verifies_hash_before_deserialization(self):
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        bundle_function = app[
            app.index("def load_bundles(") : app.index("def load_advanced_reference(")
        ]
        self.assertLess(
            bundle_function.index("secom_sha256_text_file(path)"),
            bundle_function.index("load_verified_bundles(path, PROJECT_DIR)"),
        )
        advanced_function = app[
            app.index("def load_advanced_reference(") : app.index("def load_sample_rows(")
        ]
        self.assertLess(
            advanced_function.index("secom_sha256_file(Path(path))"),
            advanced_function.index("joblib.load(path)"),
        )

    def test_dashboard_and_artifact_contract(self):
        for name in (
            "environment_manifest.json",
            "environment_validation.json",
            "software_bom.json",
            "direct_dependencies.csv",
            "installed_environment.csv",
            "summary.md",
        ):
            self.assertTrue((self.result_dir / name).is_file(), name)
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("실행환경 재현성·소프트웨어 명세", app)
        self.assertIn("취약점 스캔은 수행하지 않았으므로", app)


if __name__ == "__main__":
    unittest.main()
