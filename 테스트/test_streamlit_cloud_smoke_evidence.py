"""Validate the persisted Streamlit Cloud smoke-test evidence."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class StreamlitCloudSmokeEvidenceTests(unittest.TestCase):
    def test_public_deployment_evidence_is_complete_and_hash_linked(self) -> None:
        result_dir = PROJECT_DIR / "결과물" / "deployment_smoke"
        report = json.loads(
            (result_dir / "streamlit_cloud_smoke.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            report["url"],
            "https://shapgpt-semiconductor-diagnosis.streamlit.app/",
        )
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["diagnosis_observation"]["user_file_uploaded"])
        self.assertEqual(
            report["champion_observation"]["checkpoint_sha256"].upper(),
            sha256_file(
                PROJECT_DIR / "결과물" / "wm811k"
                / "selected_model_results" / "best_model.pt"
            ),
        )
        for filename, expected_hash in report["evidence"].items():
            self.assertEqual(sha256_file(result_dir / filename), expected_hash)
        self.assertTrue((result_dir / "검증결과.md").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
