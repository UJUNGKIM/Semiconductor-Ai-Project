"""Tests for the fixed WM-811K XAI method comparison."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

import diagnose_wm811k  # noqa: E402
from evaluate_wm811k_xai_methods import (  # noqa: E402
    WaferCNN,
    fixed_class_sample,
    flip_random_defects,
    flip_ranked_defects,
    gradient_shap_attribution,
    integrated_gradients_heatmap,
    verify_checkpoint_hash,
)
from hash_evidence import compare_recorded_text_hash  # noqa: E402
from validate_wm811k_gradient_shap_results import validate_gradient_shap_results  # noqa: E402
from validate_wm811k_xai_bundle import sha256_file, validate_bundle  # noqa: E402
from diagnose_wm811k import assess_additivity, predict_with_gradient_shap  # noqa: E402


CLASS_IDS = [str(index) for index in range(9)]
WM_DIR = PROJECT_DIR / "결과물" / "wm811k"
ASSIGNMENTS = WM_DIR / "split_results" / "split_assignments.csv"
CHECKPOINT = WM_DIR / "selected_model_results" / "best_model.pt"
GRADIENT_SHAP_RESULTS = WM_DIR / "xai_gradient_shap_validation"


def square_defect_wafer() -> np.ndarray:
    wafer = np.zeros((64, 64), dtype=np.uint8)
    wafer[8:56, 8:56] = 1
    wafer[24:32, 24:32] = 2
    return wafer


def wafer_and_baseline(wafer: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.from_numpy(np.stack((wafer > 0, wafer == 2)).astype(np.float32))[None]
    baseline = inputs.clone()
    baseline[:, 1] = 0.0
    return inputs, baseline


def small_nonlinear_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(2, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(4),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 16 * 16, 9),
    ).eval()


class Wm811kXaiMethodTests(unittest.TestCase):
    def test_fixed_class_sample_is_deterministic_and_test_only(self) -> None:
        rows = []
        for class_id in range(9):
            for offset in range(5):
                rows.append({"array_index": class_id * 10 + offset, "label_id": class_id, "failure_type": str(class_id), "split": "test"})
            rows.append({"array_index": class_id * 10 + 9, "label_id": class_id, "failure_type": str(class_id), "split": "train"})
        frame = pd.DataFrame(rows)
        first = fixed_class_sample(frame, 3, 42)
        second = fixed_class_sample(frame, 3, 42)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 27)
        self.assertTrue(first.groupby("label_id").size().eq(3).all())

    def test_integrated_gradients_heatmap_contract(self) -> None:
        torch.manual_seed(42)
        model = WaferCNN().eval()
        wafer = np.zeros((64, 64), dtype=np.uint8)
        wafer[8:56, 8:56] = 1
        wafer[24:32, 24:32] = 2
        inputs = torch.from_numpy(np.stack((wafer > 0, wafer == 2)).astype(np.float32))[None]
        heatmap = integrated_gradients_heatmap(model, inputs, target=0, steps=3)
        self.assertEqual(heatmap.shape, (64, 64))
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_gradient_shap_dashboard_contract(self) -> None:
        torch.manual_seed(42)
        model = WaferCNN().eval()
        wafer = square_defect_wafer()
        first = predict_with_gradient_shap(model, CLASS_IDS, wafer, samples=4, max_samples=4, seed=7)
        second = predict_with_gradient_shap(model, CLASS_IDS, wafer, samples=4, max_samples=4, seed=7)
        self.assertEqual(first["method"], "expected_gradients_gradient_shap_approximation")
        self.assertEqual(first["baseline"], "same_geometry_all_active_dies_normal")
        self.assertFalse(first["rescaled_to_logit_delta"])
        self.assertNotIn("local_accuracy_normalized", first)
        np.testing.assert_array_equal(first["signed_attribution_map"], second["signed_attribution_map"])
        signed = first["signed_attribution_map"]
        self.assertEqual(signed.shape, (64, 64))
        # The heatmap only rescales colours; the raw attribution keeps its logit units.
        np.testing.assert_allclose(first["heatmap"], np.abs(signed) / np.abs(signed).max(), atol=1e-6)
        self.assertAlmostEqual(first["raw_attribution_sum"], float(signed.sum()), places=9)
        self.assertTrue(np.all(signed[wafer != 2] == 0))
        # The logit change is recomputed here, outside the function under test.
        inputs, baseline = wafer_and_baseline(wafer)
        with torch.no_grad():
            target = first["selected_class"]
            expected_delta = float(model(inputs)[0, target] - model(baseline)[0, target])
        self.assertAlmostEqual(first["selected_logit_delta"], expected_delta, places=5)
        self.assertAlmostEqual(
            first["raw_additivity_residual"], first["raw_attribution_sum"] - expected_delta, places=5
        )
        with self.assertRaisesRegex(ValueError, "온도는"):
            predict_with_gradient_shap(model, CLASS_IDS, wafer, temperature=0.0)

    def test_gradient_shap_is_exact_for_a_linear_model(self) -> None:
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2 * 64 * 64, 9)).eval()
        result = predict_with_gradient_shap(model, CLASS_IDS, square_defect_wafer(), samples=4, max_samples=4)
        # Constant gradients make the path integral exact, so completeness must hold
        # without any rescaling.
        self.assertLess(abs(result["raw_additivity_residual"]), 1e-4 * max(1.0, abs(result["selected_logit_delta"])))
        self.assertTrue(result["additivity_check_passed"])
        self.assertEqual([item["samples"] for item in result["sample_attempts"]], [4])

    def test_gradient_shap_converges_to_an_independent_dense_path_integral(self) -> None:
        torch.manual_seed(3)
        model = small_nonlinear_model()
        wafer = square_defect_wafer()
        coarse = predict_with_gradient_shap(model, CLASS_IDS, wafer, samples=4, max_samples=4, seed=1)
        fine = predict_with_gradient_shap(model, CLASS_IDS, wafer, samples=256, max_samples=256, seed=1)
        target = fine["selected_class"]
        inputs, baseline = wafer_and_baseline(wafer)
        steps = 4096
        alphas = ((torch.arange(steps) + 0.5) / steps).view(-1, 1, 1, 1)
        path = (baseline + alphas * (inputs - baseline)).requires_grad_(True)
        gradients = torch.autograd.grad(model(path)[:, target].sum(), path)[0]
        reference = ((inputs - baseline) * gradients.mean(dim=0, keepdim=True)).sum(dim=1)[0].detach().numpy()
        delta = fine["selected_logit_delta"]
        self.assertLess(abs(float(reference.sum()) - delta), 0.01 * max(1.0, abs(delta)))
        coarse_error = np.abs(coarse["signed_attribution_map"] - reference).sum()
        fine_error = np.abs(fine["signed_attribution_map"] - reference).sum()
        self.assertLess(fine_error, coarse_error)
        self.assertLess(fine_error, 0.05 * np.abs(reference).sum())
        self.assertLess(abs(fine["raw_additivity_residual"]), 0.05 * max(1.0, abs(delta)))

    def test_additivity_failure_is_reported_not_hidden(self) -> None:
        passed = assess_additivity(10.5, 10.0)
        failed = assess_additivity(11.0, 10.0)
        self.assertTrue(passed["additivity_check_passed"])
        self.assertFalse(failed["additivity_check_passed"])
        self.assertAlmostEqual(failed["raw_additivity_residual"], 1.0)
        self.assertAlmostEqual(failed["relative_additivity_residual"], 0.1)
        self.assertTrue(assess_additivity(0.0, 0.0)["additivity_check_passed"])
        self.assertFalse(assess_additivity(0.01, 0.0)["additivity_check_passed"])
        torch.manual_seed(4)
        model = small_nonlinear_model()

        def broken_estimate(model, inputs, baseline, target_class, samples, seed, local_smoothing=0.0):
            return torch.zeros_like(inputs)

        with patch.object(diagnose_wm811k, "expected_gradients_attribution", broken_estimate):
            result = predict_with_gradient_shap(model, CLASS_IDS, square_defect_wafer(), samples=4, max_samples=16)
        self.assertNotEqual(result["selected_logit_delta"], 0.0)
        self.assertFalse(result["additivity_check_passed"])
        self.assertEqual(result["raw_attribution_sum"], 0.0)
        self.assertAlmostEqual(result["raw_additivity_residual"], -result["selected_logit_delta"])
        self.assertEqual([item["samples"] for item in result["sample_attempts"]], [4, 8, 16])

    def test_evaluation_script_matches_dashboard_gradient_shap(self) -> None:
        torch.manual_seed(5)
        model = WaferCNN().eval()
        wafer = square_defect_wafer()
        dashboard = predict_with_gradient_shap(model, CLASS_IDS, wafer, samples=8, max_samples=32, seed=42)
        inputs, _ = wafer_and_baseline(wafer)
        signed, record = gradient_shap_attribution(model, inputs, dashboard["selected_class"], samples=8, max_samples=32, seed=42)
        np.testing.assert_allclose(signed, dashboard["signed_attribution_map"], rtol=1e-6, atol=1e-8)
        self.assertEqual(record["gradient_shap_samples"], dashboard["samples"])
        self.assertEqual(record["additivity_check_passed"], dashboard["additivity_check_passed"])
        self.assertAlmostEqual(record["raw_additivity_residual"], dashboard["raw_additivity_residual"], places=6)

    def test_defect_flip_only_changes_defect_dies(self) -> None:
        wafer = square_defect_wafer()
        scores = np.arange(64 * 64, dtype=float).reshape(64, 64)
        top = flip_ranked_defects(wafer, scores, 0.1, largest=True)
        bottom = flip_ranked_defects(wafer, scores, 0.1, largest=False)
        random_flip = flip_random_defects(wafer, 0.1, np.random.default_rng(0))
        for flipped in (top, bottom, random_flip):
            changed = flipped != wafer
            self.assertTrue(np.all(wafer[changed] == 2))
            self.assertTrue(np.all(flipped[changed] == 1))
            self.assertEqual(int(changed.sum()), round(int((wafer == 2).sum()) * 0.1))
        self.assertFalse(np.array_equal(top, bottom))
        self.assertIsNone(flip_ranked_defects(np.ones((64, 64), dtype=np.uint8), scores, 0.1, largest=True))

    def test_split_hash_comparison_separates_bytes_from_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.csv"
            lf = b"array_index,label_id,split\n0,8,test\n1,3,train\n"
            path.write_bytes(lf)
            exact = compare_recorded_text_hash(path, hashlib.sha256(lf).hexdigest())
            self.assertEqual(exact["status"], "exact_match")
            self.assertTrue(exact["exact_byte_hash_match"])
            crlf_hash = hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()
            newline = compare_recorded_text_hash(path, crlf_hash)
            self.assertEqual(newline["status"], "newline_only_difference")
            self.assertFalse(newline["exact_byte_hash_match"])
            self.assertTrue(newline["canonical_content_match"])
            self.assertEqual(newline["recorded_newline_variant"], "crlf")
            self.assertTrue(newline["logical_rows_identical"])
            self.assertIn("줄바꿈 차이는 있으나 논리적 내용은 일치", newline["warning"])
            self.assertNotEqual(newline["recorded_raw_sha256"], newline["current_raw_sha256"])
            canonical = hashlib.sha256(lf).hexdigest()
            self.assertEqual(compare_recorded_text_hash(path, crlf_hash, canonical)["status"], "newline_only_difference")
            self.assertEqual(compare_recorded_text_hash(path, crlf_hash, "0" * 64)["status"], "mismatch")
            changed = lf.replace(b"0,8", b"0,7")
            for recorded in (changed, changed.replace(b"\n", b"\r\n")):
                result = compare_recorded_text_hash(path, hashlib.sha256(recorded).hexdigest())
                self.assertEqual(result["status"], "mismatch")
                self.assertFalse(result["canonical_content_match"])
            quoted = b'id,note\n1,"a\nb"\n'
            path.write_bytes(quoted)
            embedded = compare_recorded_text_hash(path, hashlib.sha256(quoted.replace(b"\n", b"\r\n")).hexdigest())
            self.assertFalse(embedded["logical_rows_identical"])
            self.assertEqual(embedded["status"], "mismatch")

    def test_colab_notebook_contract(self) -> None:
        notebook_path = PROJECT_DIR / "노트북" / "WM811K_07_XAI_방법비교_Colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn("evaluate_wm811k_xai_methods.py", code)
        self.assertIn("wafer_maps_64.npy", code)
        self.assertIn("xai_gradient_shap_validation_results", code)
        self.assertIn("'--gradient-shap-samples', '64'", code)
        self.assertIn("'--gradient-shap-max-samples', '256'", code)
        self.assertIn("assignments_canonical_sha256", code)
        self.assertIn("EXPECTED_CHECKPOINT_SHA256", code)
        self.assertNotIn("WM811K_training/focal_baseline/best_model.pt", code)
        self.assertIn("--per-class", code)
        self.assertIn("'50'", code)
        self.assertIn("--ig-steps", code)
        self.assertIn("'24'", code)
        self.assertIn("'1.047171711723721'", code)

    def test_received_xai_bundle_and_dashboard_contract(self) -> None:
        bundle = PROJECT_DIR / "결과물" / "wm811k" / "colab_가져오기" / "xai_method_comparison_results.zip"
        assignments = PROJECT_DIR / "결과물" / "wm811k" / "split_results" / "split_assignments.csv"
        report = validate_bundle(bundle, assignments)
        self.assertTrue(report["validation_passed"])
        self.assertFalse(report["deployment_evidence_verified"])
        self.assertEqual(report["sample_count"], 421)
        self.assertGreater(report["integrated_gradients_macro_top_vs_random"], 0)
        self.assertLess(report["gradcam_macro_top_vs_random"], 0)
        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "xai_comparison_results"
        self.assertTrue((result_dir / "xai_method_comparison_dashboard.png").is_file())
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("Grad-CAM·Integrated Gradients 비교", app)
        self.assertIn("paired_ig_minus_gradcam", app)
        self.assertIn("현재 모델의 설명 성능 근거로 사용하지 않습니다", app)

    def test_deployed_xai_bundle_matches_current_checkpoint(self) -> None:
        bundle = PROJECT_DIR / "결과물" / "wm811k" / "colab_가져오기" / "xai_deployed_model_results.zip"
        assignments = PROJECT_DIR / "결과물" / "wm811k" / "split_results" / "split_assignments.csv"
        checkpoint = PROJECT_DIR / "결과물" / "wm811k" / "selected_model_results" / "best_model.pt"
        report = validate_bundle(bundle, assignments, checkpoint)
        self.assertTrue(report["deployment_evidence_verified"])
        self.assertEqual(report["checkpoint_sha256"], "77ea8394f593ccd0e1ffa85ce1b8e59c52fcd44b109d7e3587c66f7290d4557b")
        self.assertGreater(report["integrated_gradients_macro_top_vs_random"], report["gradcam_macro_top_vs_random"])
        # The historical bundle recorded the CRLF bytes of the split table. The raw
        # mismatch stays visible; only the newline-canonical content is accepted.
        hash_check = report["assignments_hash_check"]
        self.assertEqual(hash_check["recorded_raw_sha256"], "6c6994db76fbeebcc8e3b64d9791ca5a9a0cf2dfb861f68316f3a924867139ce")
        self.assertFalse(hash_check["exact_byte_hash_match"])
        self.assertTrue(hash_check["canonical_content_match"])
        self.assertEqual(hash_check["recorded_newline_variant"], "crlf")
        self.assertEqual(report["warnings"], [hash_check["warning"]])
        self.assertFalse(report["validates_gradient_shap"])
        extracted = json.loads((PROJECT_DIR / "결과물" / "wm811k" / "xai_comparison_results" / "xai_method_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(extracted["checkpoint_sha256"], report["checkpoint_sha256"])

    def test_gradient_shap_validation_evidence_is_recomputable(self) -> None:
        report = validate_gradient_shap_results(GRADIENT_SHAP_RESULTS, ASSIGNMENTS, CHECKPOINT)
        self.assertTrue(report["validation_passed"])
        self.assertTrue(report["validates_gradient_shap"])
        self.assertTrue(report["deployment_evidence_verified"])
        self.assertEqual(report["sample_count"], 421)
        self.assertEqual(report["assignments_hash_check"]["status"], "exact_match")
        records = pd.read_csv(GRADIENT_SHAP_RESULTS / "xai_method_records.csv")
        shap_rows = records.loc[records["method"].eq("Gradient SHAP")]
        self.assertEqual(report["gradient_shap"]["additivity_pass_count"], int(shap_rows["additivity_check_passed"].sum()))
        self.assertLess(report["gradient_shap"]["additivity_pass_count"], 421 + 1)
        self.assertTrue(shap_rows["gradient_shap_samples"].isin([64, 128, 256]).all())
        summary = json.loads((GRADIENT_SHAP_RESULTS / "xai_method_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["gradient_shap"]["rescaled_to_logit_delta"])
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("과거 IG·Grad-CAM 방법 비교 증거 · Gradient SHAP 검증 아님", app)
        self.assertIn("Gradient SHAP 배포 검증: 현재 증거 없음(검증 예정).", app)

        with tempfile.TemporaryDirectory() as temporary:
            copy_dir = Path(temporary) / "results"
            shutil.copytree(GRADIENT_SHAP_RESULTS, copy_dir)
            tampered = records.copy()
            failing = tampered.index[
                tampered["method"].eq("Gradient SHAP") & ~tampered["additivity_check_passed"].astype(bool)
            ]
            target = failing[0] if len(failing) else tampered.index[tampered["method"].eq("Gradient SHAP")][0]
            tampered.loc[target, "raw_additivity_residual"] = 0.0
            tampered.to_csv(copy_dir / "xai_method_records.csv", index=False)
            with self.assertRaisesRegex(ValueError, "잔차"):
                validate_gradient_shap_results(copy_dir, ASSIGNMENTS, CHECKPOINT)
            records.to_csv(copy_dir / "xai_method_records.csv", index=False)
            forged = dict(summary)
            forged["gradient_shap"] = {**summary["gradient_shap"], "additivity_pass_rate": 1.0}
            (copy_dir / "xai_method_summary.json").write_text(json.dumps(forged), encoding="utf-8")
            if summary["gradient_shap"]["additivity_pass_rate"] < 1.0:
                with self.assertRaisesRegex(ValueError, "통과율"):
                    validate_gradient_shap_results(copy_dir, ASSIGNMENTS, CHECKPOINT)

    def test_checkpoint_hash_rejects_wrong_model_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            path.write_bytes(b"test checkpoint")
            actual = verify_checkpoint_hash(path, None)
            self.assertEqual(verify_checkpoint_hash(path, actual), actual)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_checkpoint_hash(path, "0" * 64)

    def test_version_two_bundle_requires_matching_provenance(self) -> None:
        source = PROJECT_DIR / "결과물" / "wm811k" / "colab_가져오기" / "xai_method_comparison_results.zip"
        assignments = PROJECT_DIR / "결과물" / "wm811k" / "split_results" / "split_assignments.csv"
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            checkpoint.write_bytes(b"provenance validation fixture")
            with zipfile.ZipFile(source) as archive:
                files = {name: archive.read(name) for name in archive.namelist()}
            summary = json.loads(files["xai_method_summary.json"])
            summary.update(schema_version=2, checkpoint_sha256=sha256_file(checkpoint),
                           assignments_sha256=sha256_file(assignments), temperature=1.047171711723721)
            bundle = Path(temporary) / "results.zip"
            files["xai_method_summary.json"] = json.dumps(summary).encode("utf-8")
            with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in files.items():
                    archive.writestr(name, data)
            self.assertFalse(validate_bundle(bundle, assignments)["deployment_evidence_verified"])
            self.assertTrue(validate_bundle(bundle, assignments, checkpoint)["deployment_evidence_verified"])
            checkpoint.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "체크포인트"):
                validate_bundle(bundle, assignments, checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
