"""Contract tests for the WM-811K multi-seed reproducibility experiment."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from validate_wm811k_multiseed_bundle import (  # noqa: E402
    EXPECTED_LOSS,
    EXPECTED_SAMPLING,
    EXPECTED_SEEDS,
    EXPECTED_STRATEGY,
    REQUIRED_FILES,
    sha256_file,
    validate_bundle,
)
from import_wm811k_multiseed import inspect_class_reports  # noqa: E402


class Wm811kMultiseedPreparationTests(unittest.TestCase):
    def test_protocol_constants(self) -> None:
        self.assertEqual(EXPECTED_SEEDS, (17, 42, 2026))
        self.assertEqual(EXPECTED_STRATEGY, "ce_sqrt_balanced")
        self.assertEqual(EXPECTED_LOSS, "cross_entropy")
        self.assertEqual(EXPECTED_SAMPLING, "sqrt_balanced")
        self.assertEqual(len(REQUIRED_FILES), 14)

    def test_colab_notebook_contract(self) -> None:
        notebook_path = (
            PROJECT_DIR / "노트북" / "WM811K_06_다중시드_재현성_Colab.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("torch.cuda.is_available()", code)
        self.assertIn("SEEDS = [17, 42, 2026]", code)
        self.assertIn("EXPECTED_SPLIT_SHA256", code)
        self.assertIn("cross_entropy", code)
        self.assertIn("sqrt_balanced", code)
        self.assertIn("validation_macro_f1", code)
        self.assertIn("test_used_for_selection", code)
        self.assertIn("deployment_model_changed", code)
        self.assertIn("ddof=1", code)
        self.assertIn("wm811k_multiseed_reproducibility_results", code)
        self.assertIn("files.download", code)
        self.assertIn("두 번", markdown)
        self.assertIn("test는 모델 선택에 사용하지", markdown)

    def test_notebook_keeps_large_checkpoints_out_of_download_zip(self) -> None:
        notebook_path = (
            PROJECT_DIR / "노트북" / "WM811K_06_다중시드_재현성_Colab.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("checkpoint_sha256", code)
        self.assertNotIn("copy2(seed_dir / 'best_model.pt'", code)
        self.assertNotIn("copy2(seed_dir / \"best_model.pt\"", code)

    def test_compact_bundle_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            assignments = temp_dir / "split_assignments.csv"
            assignments.write_text("array_index,split\n0,train\n", encoding="utf-8")
            assignments_hash = sha256_file(assignments)
            protocol = {
                "schema_version": 1,
                "experiment_type": "wm811k_multiseed_reproducibility",
                "purpose": "stability_evidence_not_model_selection",
                "strategy_id": EXPECTED_STRATEGY,
                "loss": EXPECTED_LOSS,
                "sampling": EXPECTED_SAMPLING,
                "sampling_max_multiplier": 8.0,
                "seeds": list(EXPECTED_SEEDS),
                "split_assignments_sha256": assignments_hash,
                "selection_split": "validation",
                "primary_metric": "validation_macro_f1",
                "test_used_for_selection": False,
                "deployment_model_changed": False,
                "existing_deployment_seed": 42,
            }
            rows = []
            run_summaries = {}
            for position, seed in enumerate(EXPECTED_SEEDS):
                offset = position * 0.01
                validation = {
                    "accuracy": 0.95 + offset,
                    "balanced_accuracy": 0.84 + offset,
                    "macro_f1": 0.85 + offset,
                }
                test = {
                    "accuracy": 0.94 + offset,
                    "balanced_accuracy": 0.82 + offset,
                    "macro_f1": 0.83 + offset,
                }
                run_summaries[seed] = {
                    "seed": seed,
                    "loss": EXPECTED_LOSS,
                    "sampling": EXPECTED_SAMPLING,
                    "sampling_max_multiplier": 8.0,
                    "best_epoch": 8 + position,
                    "epochs_ran": 12 + position,
                    "validation_metrics": validation,
                    "test_metrics": test,
                }
                rows.append(
                    {
                        "seed": seed,
                        "best_epoch": 8 + position,
                        "epochs_ran": 12 + position,
                        "elapsed_seconds": 400.0 + position,
                        **{f"validation_{key}": value for key, value in validation.items()},
                        **{f"test_{key}": value for key, value in test.items()},
                        "checkpoint_sha256": f"{seed:064x}",
                    }
                )
            seed_metrics = pd.DataFrame(rows)
            metric_names = [
                "validation_accuracy",
                "validation_balanced_accuracy",
                "validation_macro_f1",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
            ]
            metrics = {}
            metric_rows = []
            for metric in metric_names:
                values = seed_metrics[metric].to_numpy()
                stats = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                metrics[metric] = stats
                metric_rows.append({"metric": metric, **stats})
            summary = {**protocol, "std_ddof": 1, "metrics": metrics}

            bundle = temp_dir / "results.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("protocol.json", json.dumps(protocol))
                archive.writestr("reproducibility_summary.json", json.dumps(summary))
                archive.writestr("seed_metrics.csv", seed_metrics.to_csv(index=False))
                archive.writestr(
                    "metric_summary.csv", pd.DataFrame(metric_rows).to_csv(index=False)
                )
                archive.writestr("reproducibility_metrics.png", b"\x89PNG\r\n\x1a\n")
                for seed, run in run_summaries.items():
                    prefix = f"seeds/seed_{seed}/"
                    archive.writestr(prefix + "run_summary.json", json.dumps(run))
                    archive.writestr(
                        prefix + "validation_classification_report.csv", "class,f1\n"
                    )
                    archive.writestr(
                        prefix + "test_classification_report.csv", "class,f1\n"
                    )

            report = validate_bundle(bundle, assignments)
            self.assertTrue(report["validation_passed"])
            self.assertFalse(report["deployment_model_changed"])
            self.assertEqual(report["entries"], 14)

    def test_received_bundle_and_imported_evidence(self) -> None:
        bundle = (
            PROJECT_DIR
            / "결과물"
            / "wm811k"
            / "colab_가져오기"
            / "wm811k_multiseed_reproducibility_results.zip"
        )
        assignments = (
            PROJECT_DIR / "결과물" / "wm811k" / "split_results" / "split_assignments.csv"
        )
        report = validate_bundle(bundle, assignments)
        self.assertTrue(report["validation_passed"])
        class_rows = inspect_class_reports(bundle, assignments)
        self.assertEqual(len(class_rows), 18)
        test_rows = {row["class_name"]: row for row in class_rows if row["split"] == "test"}
        self.assertEqual(test_rows["Near-full"]["support_per_seed"], 21)
        self.assertLess(test_rows["Scratch"]["recall_min"], 0.72)
        output_dir = PROJECT_DIR / "결과물" / "wm811k" / "multiseed_results"
        imported = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
        self.assertTrue(imported["class_reports_reconciled"])
        self.assertTrue(imported["seed42_checkpoint_matches_deployment"])
        self.assertFalse(imported["deployment_model_changed"])
        self.assertTrue((output_dir / "검증결과.md").is_file())

    def test_dashboard_exposes_multiseed_evidence_and_limitations(self) -> None:
        app_code = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("다중 시드 재현성 검증", app_code)
        self.assertIn("WM_MULTISEED_DIR", app_code)
        self.assertIn("표준편차는 신뢰구간이 아니며", app_code)
        self.assertIn("각 시드의 보정·OOD 정책 재검증을 포함하지 않습니다", app_code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
