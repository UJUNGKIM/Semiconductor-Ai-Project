import copy
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from data_lineage import (
    artifact_sha256,
    simulated_tamper_is_detected,
    validate_lineage_manifest,
)


class DataLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = PROJECT / "결과물" / "secom" / "data_lineage"
        cls.manifest = json.loads(
            (cls.output / "data_lineage_manifest.json").read_text(encoding="utf-8")
        )
        cls.validation = json.loads(
            (cls.output / "lineage_validation.json").read_text(encoding="utf-8")
        )

    def test_manifest_hashes_and_no_raw_values(self):
        result = validate_lineage_manifest(self.manifest, PROJECT)
        self.assertTrue(result["valid"])
        self.assertEqual(result["file_count"], 9)
        self.assertFalse(result["raw_sensor_values_embedded"])
        self.assertFalse(self.manifest["raw_sensor_values_embedded"])
        self.assertEqual(self.manifest["manifest_version"], 2)
        self.assertEqual(
            self.manifest["text_hash_normalization"], "UTF-8-with-canonical-LF"
        )

    def test_path_traversal_and_hash_changes_are_rejected(self):
        unsafe = copy.deepcopy(self.manifest)
        unsafe["files"][0]["artifact_path"] = "../outside.data"
        with self.assertRaisesRegex(ValueError, "프로젝트 범위"):
            validate_lineage_manifest(unsafe, PROJECT)
        altered = copy.deepcopy(self.manifest)
        altered["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_lineage_manifest(altered, PROJECT)

    def test_tamper_drill_does_not_modify_source(self):
        source = next(
            item
            for item in self.manifest["files"]
            if item["artifact_path"].endswith("secom.data")
        )
        path = PROJECT / source["artifact_path"]
        before = path.read_bytes()
        self.assertTrue(simulated_tamper_is_detected(path, source["sha256"]))
        self.assertEqual(before, path.read_bytes())
        self.assertFalse(self.validation["real_files_modified"])

    def test_text_artifact_hash_is_line_ending_independent(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            left = Path(temp_dir) / "left.data"
            right = Path(temp_dir) / "right.data"
            left.write_bytes(b"1 2\n3 4\n")
            right.write_bytes(b"1 2\r\n3 4\r\n")
            self.assertEqual(artifact_sha256(left), artifact_sha256(right))

    def test_reconstruction_contract(self):
        self.assertEqual(self.validation["status"], "PASS")
        self.assertEqual(self.validation["raw_rows"], 1567)
        self.assertEqual(self.validation["raw_features"], 590)
        self.assertEqual(self.validation["retained_features"], 446)
        self.assertEqual(self.validation["train_rows"], 1253)
        self.assertEqual(self.validation["test_rows"], 314)
        self.assertTrue(all(self.validation["checks"].values()))

    def test_artifacts_and_dashboard_contract(self):
        inventory = pd.read_csv(self.output / "file_inventory.csv")
        self.assertEqual(len(inventory), 9)
        self.assertTrue((self.output / "reconstruction_checks.csv").is_file())
        self.assertTrue((self.output / "summary.md").is_file())
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("데이터 계보·오염 탐지 검증", app)
        self.assertIn("secom_data_lineage_manifest.json", app)
        self.assertIn("측정값의 진실성", app)


if __name__ == "__main__":
    unittest.main()
