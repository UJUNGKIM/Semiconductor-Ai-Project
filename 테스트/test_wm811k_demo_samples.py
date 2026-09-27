"""Contract tests for the WM-811K real demo sample extraction handoff."""

from __future__ import annotations

import json
import hashlib
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_DIR = PROJECT_DIR / "결과물" / "wm811k"
SELECTED_MANIFEST = WM_DIR / "selected_model_results" / "demo_sample_manifest.json"
WM_CLASS_NAMES = {
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
}
LEGACY_CASES = {
    "high_confidence_correct",
    "low_confidence_correct",
    "representative_error",
}
ALL_CASES = LEGACY_CASES | {"typical_correct", "boundary_correct"}
# The 27 samples deployed before the 4~5 per-class gallery; they must survive
# every regeneration unchanged (same wafer, same case, same prediction).
LEGACY_SAMPLE_WAFERS = {
    "center_high_confidence_correct_27941": "99a475bfb007f9743319c13fa8bda5705543ba8e941c993e2caa626773441d89",
    "center_low_confidence_correct_127737": "cd07548eb146a76a9f27fe3b9eff264401e6e4426f1d29de6658ad967d76a67e",
    "center_representative_error_61725": "4e19771333cfcc2d4ae2565d89fcf409cdec7f8ec169213823d5052a94416d95",
    "donut_high_confidence_correct_6723": "076af922c2a734975015966f3207ba1c2a6d5d234e2bc364e09d30d338532278",
    "donut_low_confidence_correct_51806": "2cf8c94ff07077d36bf9b1bc96de1b8673f4ddb10e226a543ea6e7bf92113fe4",
    "donut_representative_error_53876": "176371daba89635850f2e3dbafbfe6a04f01c8b310a18a6e80b11bac7ef3561c",
    "edge_loc_high_confidence_correct_83707": "fcec46475d00554da62a1757463060e3a1263c7dd4d29d88f945351f5f4762f5",
    "edge_loc_low_confidence_correct_133011": "92d042a608e492ef266575764ff28118601a235fd747e0b3dd2300c4ce641a3e",
    "edge_loc_representative_error_79147": "6ac6107d01bdbd59ca549b66fb7886448b180b0229ca67111be9a8d3a778e173",
    "edge_ring_high_confidence_correct_10372": "8446ea0f1514c765e21749d7dc0d776577edc06559c927eb0a2c491c3ea83eaa",
    "edge_ring_low_confidence_correct_25224": "8e2bb3ecaae1dca218650b91d58823aaa51af7b71a81f92713db7df55c5d3f3d",
    "edge_ring_representative_error_25588": "bd5df9ddfc99419031cc1f90e53fceb2e64b6da20d1ad51da2698842985a9871",
    "loc_high_confidence_correct_9393": "b410ac5e801c5ac16b408f69fbefb3990f28d030098e023e05dc6a6477362fff",
    "loc_low_confidence_correct_125517": "9e1f7af1f93f15be0f08c55edfd65c94178e851846ba3a39eebc8086aaf0dce7",
    "loc_representative_error_63604": "3643a698915e36e2daf4e2e2346c6ecae57f144df964fab82f8b5fb078de48a7",
    "near_full_high_confidence_correct_136999": "d90bf8b0d5b688b0ee7b2cb9f4b050586aa8556f46aa9d1473aec4068e338937",
    "near_full_low_confidence_correct_122743": "0180ddc915078806c35e402db848b57007bff54ea505fdc90185b0f4daaf4d19",
    "near_full_representative_error_126184": "3db87fae99098ada33ab3814f7b878df5fc83fa2f013985ce62b4e28c2d982f4",
    "random_high_confidence_correct_27614": "76b00c44e0c0b05a4e8289edfde46a50d713a7a9810aba1a96323e024fd68ba1",
    "random_low_confidence_correct_1805": "2f1f9e5cf95ce4a40ca309e5c527ea78034f05642e6ece197b509cf22a554a8f",
    "random_representative_error_127754": "caecdcd0b1a097bfca03adbf7ff507cba999a0458eb74b45cccbb6d711d2b0ee",
    "scratch_high_confidence_correct_19100": "4ace5079737842a1704aafe3cef6f2d63fda48d1d1d279b8afa974e9b072b089",
    "scratch_low_confidence_correct_166947": "1e0c15f90e142a76b09b6e2cd54b46ecae795b8495eac72234e97102f08f7486",
    "scratch_representative_error_122677": "bee5366c1121b3e5b3bde8da5ac87de6daffdb0aab3078cced3d1ae246895a88",
    "none_high_confidence_correct_116323": "5297edc8e34b0afd0aa54c14ebb1288b52d8dfae0112ac2ed2cc819becf117f9",
    "none_low_confidence_correct_91342": "2c509b5307f1ccb32addc8125bcb43701c03ed0baee8aba67bc9f8d79640f446",
    "none_representative_error_138722": "bd5733c60ef98ffe11d93d50df6a2645a1da6ef81a7bb2b159146f0d3f975676",
}


class Wm811kDemoSampleTests(unittest.TestCase):
    def test_manifest_contract(self) -> None:
        manifest = json.loads(SELECTED_MANIFEST.read_text(encoding="utf-8"))
        samples = manifest["samples"]
        self.assertEqual(manifest["source_split"], "test")
        self.assertFalse(manifest["used_for_model_selection"])
        self.assertEqual(manifest["sample_count"], len(samples))
        self.assertEqual(len({item["array_index"] for item in samples}), len(samples))
        self.assertEqual(len({item["sample_id"] for item in samples}), len(samples))
        counts = Counter(item["true_class"] for item in samples)
        self.assertEqual(set(counts), WM_CLASS_NAMES)
        self.assertTrue(all(4 <= count <= 5 for count in counts.values()), counts)
        self.assertEqual(manifest["class_sample_counts"], dict(counts))
        self.assertEqual(manifest["samples_per_class_range"], [4, 5])
        self.assertEqual(set(manifest["selection_rules"]), ALL_CASES)
        cases = pd.DataFrame(samples).groupby("true_class")["selection_case"]
        for class_name, class_cases in cases:
            self.assertEqual(len(set(class_cases)), len(class_cases), class_name)
            self.assertTrue(set(class_cases).issubset(ALL_CASES))
            missing = ALL_CASES - set(class_cases)
            self.assertEqual(missing, set(manifest["missing_cases"].get(class_name, {})))
        for item in samples:
            self.assertEqual(item["correct"], item["predicted_label_id"] == item["true_label_id"])
            self.assertEqual(
                item["review_required"],
                item["calibrated_confidence"] < manifest["review_threshold"],
            )
            if item["selection_case"] == "representative_error":
                self.assertFalse(item["correct"])
            else:
                self.assertTrue(item["correct"])

    def test_selection_uses_reproduced_deployed_predictions(self) -> None:
        manifest = json.loads(SELECTED_MANIFEST.read_text(encoding="utf-8"))
        reproduction = json.loads(
            (WM_DIR / "selected_model_results" / "test_prediction_reproduction.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoint = WM_DIR / "selected_model_results" / "best_model.pt"
        self.assertEqual(
            reproduction["checkpoint_sha256"],
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )
        verification = reproduction["verification"]
        self.assertTrue(verification["all_checks_passed"])
        self.assertTrue(all(verification["checks"].values()))
        # The references are independent of this reproduction: committed test
        # metrics, the Colab XAI run's 421 wafers, and the pre-expansion manifest.
        self.assertEqual(verification["xai_bundle_reference"]["samples_compared"], 421)
        self.assertEqual(verification["xai_bundle_reference"]["prediction_mismatches"], 0)
        self.assertLessEqual(verification["xai_bundle_reference"]["max_calibrated_confidence_difference"], 1e-5)
        self.assertEqual(verification["legacy_demo_reference"]["samples_compared"], 27)
        self.assertEqual(verification["legacy_demo_reference"]["prediction_mismatches"], 0)
        self.assertEqual(verification["review_policy_counts"], verification["committed_review_policy_counts"])
        self.assertEqual(reproduction["rows"], 24705)
        self.assertFalse(reproduction["used_for_model_selection"])
        self.assertEqual(
            manifest["source_predictions"]["sha256"], reproduction["predictions_sha256"]
        )
        self.assertEqual(manifest["source_predictions"]["rows"], reproduction["rows"])

    def test_colab_notebook_contract(self) -> None:
        notebook_path = (
            PROJECT_DIR / "노트북" / "WM811K_04_실제샘플_추출_Colab.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        code = "\n".join("".join(cell["source"]) for cell in code_cells)
        self.assertEqual(notebook["nbformat"], 4)
        self.assertNotEqual(notebook["metadata"].get("accelerator"), "GPU")
        self.assertIn("demo_sample_manifest.json", code)
        self.assertIn("wafer_maps_64.npy", code)
        self.assertIn("mmap_mode='r'", code)
        self.assertIn("int(labels[index]) == int(item['true_label_id'])", code)
        self.assertIn("set(np.unique(wafer)).issubset({0, 1, 2})", code)
        self.assertIn("wm811k_demo_samples", code)
        self.assertIn("files.download", code)
        self.assertNotIn("== 27", code)
        self.assertIn("4 <= count <= 5", code)
        for case in ALL_CASES:
            self.assertIn(f"'{case}'", code)
        for number, cell in enumerate(code_cells, 1):
            compile("".join(cell["source"]), f"demo_cell_{number}", "exec")

    def test_deployment_samples_contract(self) -> None:
        selected_manifest = json.loads(SELECTED_MANIFEST.read_text(encoding="utf-8"))
        demo_dir = WM_DIR / "demo_samples"
        exported_manifest = json.loads(
            (demo_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(exported_manifest["sample_count"], selected_manifest["sample_count"])
        self.assertEqual(exported_manifest["exported_map_shape"], [64, 64])
        self.assertEqual(exported_manifest["allowed_values"], [0, 1, 2])
        expected_by_id = {
            item["sample_id"]: item for item in selected_manifest["samples"]
        }
        actual_by_id = {
            item["sample_id"]: item for item in exported_manifest["samples"]
        }
        self.assertEqual(set(expected_by_id), set(actual_by_id))
        npy_files = set((demo_dir / "samples").glob("*.npy"))
        self.assertEqual(len(npy_files), selected_manifest["sample_count"])
        wafer_hashes = set()
        for sample_id, item in actual_by_id.items():
            for key, value in expected_by_id[sample_id].items():
                self.assertEqual(item[key], value, f"{sample_id}: {key}")
            sample_path = demo_dir / item["npy_file"]
            self.assertIn(sample_path, npy_files)
            wafer = np.load(sample_path, allow_pickle=False)
            self.assertEqual(wafer.shape, (64, 64))
            self.assertEqual(wafer.dtype, np.uint8)
            self.assertTrue(set(np.unique(wafer)).issubset({0, 1, 2}))
            digest = hashlib.sha256(wafer.tobytes()).hexdigest()
            self.assertEqual(digest, item["wafer_sha256"])
            wafer_hashes.add(digest)
        # Every exported wafer is a distinct real array, not a copy of another sample.
        self.assertEqual(len(wafer_hashes), len(actual_by_id))
        export = exported_manifest["export"]
        self.assertEqual(
            export["previous_samples_preserved"] + export["new_samples_added"],
            exported_manifest["sample_count"],
        )
        self.assertEqual(export["deployed_formats"], ["npy"])

    def test_legacy_samples_are_preserved(self) -> None:
        manifest = json.loads(SELECTED_MANIFEST.read_text(encoding="utf-8"))
        legacy = [item for item in manifest["samples"] if item["selection_case"] in LEGACY_CASES]
        self.assertEqual(len(legacy), 27)
        self.assertEqual(Counter(item["true_class"] for item in legacy), Counter({name: 3 for name in WM_CLASS_NAMES}))
        exported = {
            item["sample_id"]: item
            for item in json.loads((WM_DIR / "demo_samples" / "manifest.json").read_text(encoding="utf-8"))["samples"]
        }
        self.assertEqual({item["sample_id"] for item in legacy}, set(LEGACY_SAMPLE_WAFERS))
        for sample_id, wafer_sha256 in LEGACY_SAMPLE_WAFERS.items():
            self.assertEqual(exported[sample_id]["wafer_sha256"], wafer_sha256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
