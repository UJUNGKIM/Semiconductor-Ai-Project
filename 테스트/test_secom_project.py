"""Automated regression tests for the SECOM project."""

from __future__ import annotations

import csv
import gc
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "코드" / "secom"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from diagnose_secom import consensus_label, load_rows, parse_sensor_bytes  # noqa: E402
from train_compare_models import load_data, structural_filter  # noqa: E402
from preprocess_wm811k import convert_dataframe  # noqa: E402
from create_lot_aware_splits import create_splits  # noqa: E402
from diagnose_wm811k import (  # noqa: E402
    artificial_demo_map,
    load_checkpoint,
    parse_wafer_bytes,
    predict_with_gradcam,
)
from build_uncertainty_policy import temperature_scale  # noqa: E402
from train_wm811k_cnn import (  # noqa: E402
    WaferCNN,
    effective_class_weights,
    make_loaders,
    sqrt_sampling_multipliers,
    train as train_wm811k_cnn,
    validate_inputs,
)


UNIFIED_FRAME_STEPS = ("1. 입력 신뢰도", "2. 모델 판정", "3. SHAP 기여", "4. 검토 행동")


def assert_unified_xai_frame(
    test: unittest.TestCase, app: AppTest, explanation_keyword: str
) -> None:
    """The shared 입력 신뢰도 → 모델 판정 → SHAP 기여 → 검토 행동 frame renders in order."""
    markdown = [element.value for element in app.markdown]
    test.assertIn("#### 공통 SHAP 의사결정 프레임", markdown)
    positions = [markdown.index(f"**{step}**") for step in UNIFIED_FRAME_STEPS]
    test.assertEqual(positions, sorted(positions))
    test.assertTrue(
        any(explanation_keyword in element.value for element in app.caption),
        f"SHAP 기여 단계 설명에 {explanation_keyword}가 없습니다.",
    )


class SecomProjectTests(unittest.TestCase):
    """Check data contracts, saved models, and the dashboard smoke path."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_dir = PROJECT_DIR / "데이터" / "SECOM 데이터셋" / "raw"
        cls.processed_dir = (
            PROJECT_DIR / "데이터" / "SECOM 데이터셋" / "processed"
        )
        cls.model_dir = PROJECT_DIR / "결과물" / "secom" / "dual_model_results"
        cls.X_raw, cls.y, cls.timestamps = load_data(PROJECT_DIR)

    def test_raw_data_contract(self) -> None:
        self.assertEqual(self.X_raw.shape, (1567, 590))
        self.assertEqual(self.y.value_counts().sort_index().to_dict(), {0: 1463, 1: 104})
        self.assertEqual(len(self.timestamps), 1567)
        self.assertFalse(self.timestamps.isna().any())

    def test_structural_filter_contract(self) -> None:
        reduced, high_missing, constants = structural_filter(self.X_raw)
        self.assertEqual(reduced.shape, (1567, 446))
        self.assertEqual(len(high_missing), 28)
        self.assertEqual(len(constants), 116)

    def test_saved_preprocessing_contract(self) -> None:
        metadata = json.loads(
            (self.processed_dir / "preprocessing_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(metadata["random_state"], 42)
        self.assertEqual(metadata["test_size"], 0.2)
        self.assertEqual(metadata["retained_feature_count"], 446)

        X_train = pd.read_csv(self.processed_dir / "X_train.csv")
        X_test = pd.read_csv(self.processed_dir / "X_test.csv")
        y_train = pd.read_csv(self.processed_dir / "y_train.csv")
        y_test = pd.read_csv(self.processed_dir / "y_test.csv")
        self.assertEqual(X_train.shape, (1253, 447))
        self.assertEqual(X_test.shape, (314, 447))
        self.assertFalse(X_train.drop(columns="row_id").isna().any().any())
        self.assertFalse(X_test.drop(columns="row_id").isna().any().any())
        self.assertEqual(y_train["label"].value_counts().sort_index().to_dict(), {0: 1170, 1: 83})
        self.assertEqual(y_test["label"].value_counts().sort_index().to_dict(), {0: 293, 1: 21})

    def test_saved_models_predict_probabilities(self) -> None:
        required_keys = {
            "imputer",
            "input_features",
            "model",
            "model_name",
            "operating_thresholds",
            "selected_features",
            "selector",
        }
        for filename in ("catboost_model.joblib", "xgboost_model.joblib"):
            with self.subTest(model=filename):
                bundle = joblib.load(self.model_dir / filename)
                self.assertTrue(required_keys.issubset(bundle))
                X_model = self.X_raw.iloc[:3][bundle["input_features"]]
                X_ready = bundle["selector"].transform(
                    bundle["imputer"].transform(X_model)
                )
                probabilities = bundle["model"].predict_proba(X_ready)[:, 1]
                self.assertEqual(probabilities.shape, (3,))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
                self.assertIn("balanced_f2", bundle["operating_thresholds"])

    def test_enhanced_synthetic_validation_contract(self) -> None:
        result_dir = (
            PROJECT_DIR / "결과물" / "secom" / "enhanced_synthetic_results"
        )
        recommendation = json.loads(
            (result_dir / "recommendation.json").read_text(encoding="utf-8")
        )
        quality = pd.read_csv(result_dir / "synthetic_quality.csv")
        summary = pd.read_csv(result_dir / "cv_summary.csv")

        self.assertFalse(recommendation["selection_uses_test"])
        self.assertIn(
            recommendation["decision"],
            {"adopt_ddpm", "retain_scale_pos_weight"},
        )
        self.assertEqual(len(quality), 6)
        self.assertTrue((quality["exact_copy_rate"] == 0).all())
        self.assertTrue((quality["memorization_risk_rate"] <= 0.05).all())
        self.assertTrue(summary["strategy"].str.startswith("DDPM").any())
        self.assertEqual(set(summary["model"]), {"CatBoost", "XGBoost"})

    def test_diagnosis_input_and_consensus(self) -> None:
        sample = load_rows(self.raw_dir / "secom.data", row_limit=3)
        self.assertEqual(sample.shape, (3, 590))
        expected = {
            (0, 0): "both_models_normal",
            (1, 0): "catboost_only_defect",
            (0, 1): "xgboost_only_defect",
            (1, 1): "both_models_defect",
        }
        for predictions, label in expected.items():
            with self.subTest(predictions=predictions):
                self.assertEqual(consensus_label(*predictions), label)

    def test_csv_upload_with_headers_and_metadata(self) -> None:
        sample = self.X_raw.iloc[:2].copy()
        sample.insert(0, "timestamp", ["2008-01-01", "2008-01-02"])
        sample.insert(0, "row_id", ["sample_a", "sample_b"])
        parsed, info = parse_sensor_bytes(
            sample.to_csv(index=False).encode("utf-8-sig")
        )

        self.assertEqual(parsed.shape, (2, 590))
        self.assertEqual(info["format"], "CSV · 헤더 있음")
        self.assertEqual(info["ignored_columns"], ["row_id", "timestamp"])
        pd.testing.assert_frame_equal(
            parsed,
            sample.drop(columns=["row_id", "timestamp"]),
            check_dtype=False,
        )

        invalid = sample.rename(columns={"feature_589": "unknown_sensor"})
        with self.assertRaisesRegex(ValueError, "누락 센서"):
            parse_sensor_bytes(invalid.to_csv(index=False).encode("utf-8"))

    def test_dashboard_smoke(self) -> None:
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30)
        app.run()
        app.switch_page("dashboard_ui/site_pages/secom.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn(
            "SECOM 공정 센서 SHAP 진단",
            [element.value for element in app.title],
        )
        assert_unified_xai_frame(self, app, "TreeSHAP")
        self.assertTrue(
            any(
                "데이터 한계" in element.value and "익명화" in element.value
                for element in app.warning
            )
        )
        self.assertIn("진단 실행", [element.label for element in app.button])
        self.assertIn(
            "한 모델만 불량",
            [element.label for element in app.metric],
        )

    def test_colab_tabddpm_notebook_contract(self) -> None:
        notebook_path = PROJECT_DIR / "노트북" / "SECOM_TabDDPM_Colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertEqual(notebook["nbformat"], 4)
        self.assertIn("yandex-research/tab-ddpm", code)
        self.assertIn("torch.cuda.is_available", code)
        self.assertIn("secom.data", code)
        self.assertIn("secom_labels.data", code)
        self.assertIn("archive.ics.uci.edu/static/public/179/secom.zip", code)
        self.assertIn("threshold_f2", code)
        self.assertIn("official_commit", code)
        self.assertIn("DRIVE_OUTPUT_DIR", code)
        self.assertIn("selection_uses_test", code)
        self.assertIn("secom_tabddpm_results", code)
        self.assertIn("files.download", code)

    def test_received_tabddpm_results_contract(self) -> None:
        result_dir = (
            PROJECT_DIR
            / "결과물"
            / "secom"
            / "colab_가져오기"
            / "secom_tabddpm_results"
        )
        metadata = json.loads(
            (result_dir / "run_metadata.json").read_text(encoding="utf-8")
        )
        validation = pd.read_csv(result_dir / "validation_results.csv")
        test = pd.read_csv(result_dir / "test_results.csv")
        quality = pd.read_csv(result_dir / "synthetic_quality.csv").iloc[0]
        synthetic = pd.read_csv(result_dir / "tabddpm_synthetic_defects.csv")

        self.assertEqual(metadata["run_mode"], "paper")
        self.assertEqual(metadata["train_steps"], 30_000)
        self.assertEqual(metadata["retained_features"], 446)
        self.assertFalse(metadata["selection_uses_test"])
        self.assertEqual(metadata["threshold_metric"], "F2_on_validation")
        self.assertEqual(set(validation["model"]), {"CatBoost", "XGBoost"})
        self.assertEqual(len(validation), 10)
        self.assertEqual(len(test), 3)
        self.assertEqual(synthetic.shape, (int(quality["synthetic_defect_rows"]), 446))
        self.assertTrue(np.isfinite(synthetic.to_numpy()).all())
        self.assertEqual(float(quality["exact_copy_rate"]), 0.0)
        self.assertGreaterEqual(float(quality["discriminator_auc"]), 0.5)
        self.assertLessEqual(float(quality["discriminator_auc"]), 1.0)
        self.assertGreater((result_dir / "tabddpm_model.pt").stat().st_size, 0)

    def test_wm811k_compact_preprocessing_contract(self) -> None:
        frame = pd.DataFrame(
            {
                "waferMap": [
                    np.array([[0, 1], [2, 1]], dtype=np.uint8),
                    np.array([[0, 0, 1], [2, 2, 1]], dtype=np.uint8),
                    np.array([[1, 2], [0, 1]], dtype=np.uint8),
                    np.array([[2, 1], [1, 0]], dtype=np.uint8),
                ],
                "dieSize": [4, 6, 4, 4],
                "lotName": ["lot1", "lot1", "lot2", "lot2"],
                "waferIndex": [1, 2, 1, 2],
                "trianTestLabel": [
                    np.array([["Training"]]),
                    np.array([["Training"]]),
                    np.array([], dtype=object),
                    np.array([["Test"]]),
                ],
                "failureType": [
                    np.array([["Center"]]),
                    np.array([["none"]]),
                    np.array([], dtype=object),
                    np.array([["Scratch"]]),
                ],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = convert_dataframe(frame, output_dir, image_size=16)
            maps = np.load(output_dir / "wafer_maps_16.npy")
            labels = np.load(output_dir / "labels.npy")
            metadata = pd.read_csv(output_dir / "labeled_metadata.csv")

            self.assertEqual(maps.shape, (3, 16, 16))
            self.assertTrue(set(np.unique(maps)).issubset({0, 1, 2}))
            self.assertEqual(labels.shape, (3,))
            self.assertEqual(len(metadata), 3)
            self.assertEqual(summary["invalid_labeled_rows"], 0)
            self.assertEqual(
                set(summary["label_mapping"]), {"Center", "Scratch", "none"}
            )

    def test_wm811k_colab_preprocessing_notebook_contract(self) -> None:
        notebook_path = PROJECT_DIR / "노트북" / "WM811K_01_전처리_Colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertEqual(notebook["nbformat"], 4)
        self.assertIn("drive.mount", code)
        self.assertIn("LSWMD.pkl", code)
        self.assertIn("qingyi/wm811k-wafer-map", code)
        self.assertIn("KAGGLE_API_TOKEN", code)
        self.assertIn("kagglehub.dataset_download", code)
        self.assertIn("MemAvailable:", code)
        self.assertIn("available_gb >= 8.0", code)
        self.assertIn("preprocess_wm811k.py", code)
        self.assertIn("wafer_maps_64.npy", code)
        self.assertIn("wm811k_preprocessing_receipt", code)
        self.assertIn("files.download", code)

    def test_wm811k_preprocessing_receipt_contract(self) -> None:
        receipt_path = (
            PROJECT_DIR
            / "결과물"
            / "wm811k"
            / "colab_가져오기"
            / "wm811k_preprocessing_receipt.zip"
        )
        with zipfile.ZipFile(receipt_path) as archive:
            names = set(archive.namelist())
            self.assertEqual(
                names,
                {
                    "dataset_summary.json",
                    "label_mapping.json",
                    "class_distribution.csv",
                    "labeled_metadata.csv",
                },
            )
            summary = json.loads(archive.read("dataset_summary.json"))
            distribution = list(
                csv.DictReader(
                    io.StringIO(
                        archive.read("class_distribution.csv").decode("utf-8")
                    )
                )
            )

        self.assertEqual(summary["source_rows"], 811_457)
        self.assertEqual(summary["labeled_rows_detected"], 172_950)
        self.assertEqual(summary["valid_labeled_rows"], 172_950)
        self.assertEqual(summary["invalid_labeled_rows"], 0)
        self.assertEqual(summary["image_size"], 64)
        self.assertEqual(summary["map_dtype"], "uint8")
        self.assertEqual(summary["map_values"], [0, 1, 2])
        self.assertEqual(summary["class_count"], 9)
        self.assertEqual(
            sum(int(row["size"]) for row in distribution),
            summary["valid_labeled_rows"],
        )

    def test_wm811k_lot_aware_split_contract(self) -> None:
        rows = []
        array_index = 0
        for label_id in range(9):
            for lot_id in range(14):
                for wafer_id in range(2):
                    rows.append(
                        {
                            "array_index": array_index,
                            "source_row": array_index,
                            "failure_type": f"class_{label_id}",
                            "label_id": label_id,
                            "source_split": "Training",
                            "lot_name": f"lot_{label_id}_{lot_id}",
                            "wafer_index": wafer_id,
                        }
                    )
                    array_index += 1
        metadata = pd.DataFrame(rows)

        first, summary = create_splits(metadata, random_state=42)
        second, _ = create_splits(metadata, random_state=42)
        self.assertTrue(first["split"].equals(second["split"]))
        self.assertEqual(summary["lot_overlap"], {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        })
        for split_name in ("train", "validation", "test"):
            subset = first.loc[first["split"] == split_name]
            self.assertEqual(subset["label_id"].nunique(), 9)

        split_dir = PROJECT_DIR / "결과물" / "wm811k" / "split_results"
        saved_summary = json.loads(
            (split_dir / "split_summary.json").read_text(encoding="utf-8")
        )
        assignments = pd.read_csv(split_dir / "split_assignments.csv")
        self.assertEqual(saved_summary["total_rows"], 172_950)
        self.assertEqual(len(assignments), 172_950)
        self.assertEqual(assignments["array_index"].tolist(), list(range(172_950)))
        self.assertEqual(saved_summary["lot_overlap"], {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        })

    def test_wm811k_cnn_training_contract(self) -> None:
        model = WaferCNN(num_classes=9)
        logits = model(torch.zeros((2, 2, 64, 64), dtype=torch.float32))
        self.assertEqual(tuple(logits.shape), (2, 9))

        counts, weights = effective_class_weights(
            np.repeat(np.arange(9), np.arange(1, 10)),
            num_classes=9,
            beta=0.9999,
        )
        self.assertEqual(counts.tolist(), list(range(1, 10)))
        self.assertEqual(weights.shape, (9,))
        self.assertTrue(np.isfinite(weights).all())
        _, sampling_multipliers = sqrt_sampling_multipliers(
            np.repeat(np.arange(9), np.arange(1, 10)),
            num_classes=9,
            maximum=3.0,
        )
        self.assertEqual(sampling_multipliers.shape, (9,))
        self.assertLessEqual(float(sampling_multipliers.max()), 3.0)
        self.assertGreater(
            float(sampling_multipliers[0]), float(sampling_multipliers[-1])
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            maps = np.zeros((27, 64, 64), dtype=np.uint8)
            labels = np.tile(np.arange(9, dtype=np.int64), 3)
            assignments = pd.DataFrame(
                {
                    "array_index": np.arange(27),
                    "label_id": labels,
                    "failure_type": [f"class_{label}" for label in labels],
                    "lot_name": [f"lot_{index}" for index in range(27)],
                    "split": np.repeat(("train", "validation", "test"), 9),
                }
            )
            maps_path = root / "maps.npy"
            labels_path = root / "labels.npy"
            assignments_path = root / "assignments.csv"
            np.save(maps_path, maps)
            np.save(labels_path, labels)
            assignments.to_csv(assignments_path, index=False)
            loaded_maps, loaded_labels, loaded_assignments = validate_inputs(
                maps_path, labels_path, assignments_path
            )
            self.assertEqual(loaded_maps.shape, (27, 64, 64))
            self.assertEqual(loaded_labels.shape, (27,))
            self.assertEqual(len(loaded_assignments), 27)
            del loaded_maps, loaded_labels, loaded_assignments
            gc.collect()

            sampled_loaders, sampled_multipliers = make_loaders(
                maps_path,
                assignments,
                batch_size=9,
                num_workers=0,
                seed=42,
                device=torch.device("cpu"),
                sampling="sqrt_balanced",
                sampling_max_multiplier=3.0,
            )
            sampled_inputs, sampled_targets = next(iter(sampled_loaders["train"]))
            self.assertEqual(tuple(sampled_inputs.shape), (9, 2, 64, 64))
            self.assertEqual(tuple(sampled_targets.shape), (9,))
            self.assertEqual(sampled_multipliers.shape, (9,))
            del sampled_loaders, sampled_inputs, sampled_targets
            gc.collect()

            output_dir = root / "output"
            summary = train_wm811k_cnn(
                SimpleNamespace(
                    maps=maps_path,
                    labels=labels_path,
                    assignments=assignments_path,
                    output_dir=output_dir,
                    epochs=1,
                    batch_size=9,
                    num_workers=0,
                    learning_rate=1e-3,
                    weight_decay=1e-4,
                    loss="focal",
                    focal_gamma=2.0,
                    class_weight_beta=0.9999,
                    patience=1,
                    minimum_delta=1e-4,
                    seed=42,
                    cpu=True,
                )
            )
            self.assertEqual(summary["epochs_ran"], 1)
            for name in (
                "best_model.pt",
                "run_summary.json",
                "training_history.csv",
                "test_classification_report.csv",
                "test_confusion_matrix.csv",
                "test_predictions.csv",
                "validation_classification_report.csv",
                "validation_confusion_matrix.csv",
                "validation_predictions.csv",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

    def test_wm811k_received_cnn_results_contract(self) -> None:
        result_dir = (
            PROJECT_DIR
            / "결과물"
            / "wm811k"
            / "colab_가져오기"
            / "wm811k_cnn_results"
        )
        summary = json.loads(
            (result_dir / "run_summary.json").read_text(encoding="utf-8")
        )
        history = pd.read_csv(result_dir / "training_history.csv")
        predictions = pd.read_csv(result_dir / "test_predictions.csv")
        matrix = pd.read_csv(result_dir / "test_confusion_matrix.csv", index_col=0)

        self.assertEqual(summary["device"], "cuda")
        self.assertEqual(summary["gpu"], "Tesla T4")
        self.assertEqual(summary["best_epoch"], 9)
        self.assertAlmostEqual(summary["test_metrics"]["macro_f1"], 0.8280245679)
        self.assertEqual(
            int(history.loc[history["validation_macro_f1"].idxmax(), "epoch"]),
            summary["best_epoch"],
        )
        self.assertEqual(len(predictions), 24_705)
        self.assertTrue(predictions["array_index"].is_unique)
        self.assertEqual(int(matrix.to_numpy().sum()), len(predictions))
        probability_columns = [
            column for column in predictions if column.startswith("probability_")
        ]
        probabilities = predictions[probability_columns].to_numpy()
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=3e-7)

        model, class_names, checkpoint = load_checkpoint(
            result_dir / "best_model.pt"
        )
        self.assertEqual(checkpoint["best_epoch"], summary["best_epoch"])
        self.assertEqual(class_names, summary["class_names"])
        self.assertTrue(
            all(
                torch.isfinite(tensor).all()
                for tensor in checkpoint["model_state_dict"].values()
            )
        )

    def test_wm811k_selected_strategy_contract(self) -> None:
        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "selected_model_results"
        summary = json.loads(
            (result_dir / "run_summary.json").read_text(encoding="utf-8")
        )
        selection = json.loads(
            (result_dir / "selection_protocol.json").read_text(encoding="utf-8")
        )
        validation = pd.read_csv(result_dir / "validation_strategy_comparison.csv")
        exploratory_test = pd.read_csv(
            result_dir / "test_strategy_comparison_exploratory.csv"
        )
        matrix = pd.read_csv(result_dir / "test_confusion_matrix.csv", index_col=0)
        policy = json.loads(
            (result_dir / "uncertainty_policy.json").read_text(encoding="utf-8")
        )
        error_analysis = json.loads(
            (result_dir / "error_analysis.json").read_text(encoding="utf-8")
        )

        self.assertEqual(selection["selected_strategy"], "ce_sqrt_balanced")
        self.assertEqual(selection["primary_metric"], "validation_macro_f1")
        self.assertEqual(
            selection["tie_breaker"], "validation_balanced_accuracy"
        )
        self.assertFalse(selection["selection_uses_test"])
        selected_by_rule = validation.sort_values(
            ["validation_macro_f1", "validation_balanced_accuracy"],
            ascending=False,
        ).iloc[0]
        self.assertEqual(selected_by_rule["strategy"], selection["selected_strategy"])

        self.assertEqual(summary["loss"], "cross_entropy")
        self.assertEqual(summary["sampling"], "sqrt_balanced")
        self.assertEqual(summary["best_epoch"], 10)
        self.assertEqual(summary["epochs_ran"], 14)
        self.assertAlmostEqual(
            summary["validation_metrics"]["macro_f1"], 0.8965338122291533
        )
        self.assertAlmostEqual(
            summary["test_metrics"]["macro_f1"], 0.8828173704348624
        )
        self.assertEqual(int(matrix.to_numpy().sum()), 24_705)
        self.assertEqual(int(exploratory_test["selected_on_validation"].sum()), 1)
        self.assertEqual(policy["fitted_on"], "validation")
        self.assertFalse(policy["selection_uses_test"])
        self.assertAlmostEqual(policy["temperature"], 1.047171711723721)
        self.assertAlmostEqual(policy["review_threshold"], 0.974998285716378)
        self.assertEqual(policy["test_evaluation"]["automatic_count"], 22_335)
        self.assertEqual(policy["test_evaluation"]["captured_errors"], 525)
        self.assertAlmostEqual(
            policy["test_evaluation"]["automatic_accuracy"], 0.9983881799865681
        )
        self.assertAlmostEqual(
            policy["test_evaluation"]["error_capture_rate"], 0.9358288770053476
        )
        self.assertEqual(error_analysis["source_split"], "test")
        self.assertFalse(error_analysis["used_for_model_selection"])
        self.assertEqual(error_analysis["total_rows"], 24_705)
        self.assertEqual(error_analysis["total_errors"], 561)
        self.assertEqual(
            sum(item["support"] for item in error_analysis["class_analysis"]),
            24_705,
        )
        self.assertEqual(
            sum(item["error_count"] for item in error_analysis["class_analysis"]),
            561,
        )
        self.assertEqual(
            sum(item["count"] for item in error_analysis["confusion_pairs"]),
            561,
        )
        top_pair = error_analysis["confusion_pairs"][0]
        self.assertEqual(
            (top_pair["true_class"], top_pair["predicted_class"], top_pair["count"]),
            ("Edge-Loc", "none", 92),
        )
        scratch = next(
            item
            for item in error_analysis["class_analysis"]
            if item["class_name"] == "Scratch"
        )
        self.assertAlmostEqual(scratch["recall"], 0.7176470588235294)

        model, class_names, checkpoint = load_checkpoint(result_dir / "best_model.pt")
        self.assertEqual(checkpoint["best_epoch"], summary["best_epoch"])
        self.assertEqual(checkpoint["loss"], summary["loss"])
        self.assertEqual(checkpoint["sampling"], summary["sampling"])
        self.assertEqual(class_names, summary["class_names"])
        self.assertTrue(
            all(
                torch.isfinite(tensor).all()
                for tensor in checkpoint["model_state_dict"].values()
            )
        )
        del model

    def test_wm811k_upload_and_gradcam_contract(self) -> None:
        wafer = artificial_demo_map(size=32)
        buffer = io.BytesIO()
        np.save(buffer, wafer, allow_pickle=False)
        parsed = parse_wafer_bytes(buffer.getvalue(), "sample.npy")
        np.testing.assert_array_equal(parsed, wafer)

        csv_bytes = pd.DataFrame(wafer).to_csv(
            index=False, header=False
        ).encode("utf-8")
        parsed_csv = parse_wafer_bytes(csv_bytes, "sample.csv")
        np.testing.assert_array_equal(parsed_csv, wafer)

        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "selected_model_results"
        model, class_names, _ = load_checkpoint(result_dir / "best_model.pt")
        result = predict_with_gradcam(model, class_names, wafer)
        self.assertEqual(result["resized_map"].shape, (64, 64))
        self.assertEqual(result["heatmap"].shape, (64, 64))
        self.assertTrue(np.isfinite(result["heatmap"]).all())
        self.assertGreaterEqual(float(result["heatmap"].min()), 0.0)
        self.assertLessEqual(float(result["heatmap"].max()), 1.0)
        self.assertAlmostEqual(float(result["probabilities"].sum()), 1.0, places=5)
        calibrated = predict_with_gradcam(
            model, class_names, wafer, temperature=1.047171711723721
        )
        self.assertEqual(calibrated["predicted_class"], result["predicted_class"])
        self.assertAlmostEqual(
            float(calibrated["probabilities"].sum()), 1.0, places=5
        )
        scaled = temperature_scale(
            np.array([[0.8, 0.1, 0.1]], dtype=float), temperature=2.0
        )
        self.assertAlmostEqual(float(scaled.sum()), 1.0)
        self.assertLess(float(scaled.max()), 0.8)
        with self.assertRaisesRegex(ValueError, "temperature"):
            predict_with_gradcam(model, class_names, wafer, temperature=0.0)

        with self.assertRaisesRegex(ValueError, "0, 1, 2"):
            parse_wafer_bytes(
                b"0,1,1,0\n1,2,3,1\n1,1,2,1\n0,1,1,0\n", "bad.csv"
            )

    def test_wm811k_dashboard_smoke(self) -> None:
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30)
        app.run()
        app.switch_page("dashboard_ui/site_pages/wm811k.py")
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn(
            "WM-811K 웨이퍼 맵 SHAP 진단",
            [element.value for element in app.title],
        )
        assert_unified_xai_frame(self, app, "Gradient SHAP")

    def test_wm811k_cnn_colab_notebook_contract(self) -> None:
        notebook_path = PROJECT_DIR / "노트북" / "WM811K_02_CNN_학습_Colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("torch.cuda.is_available()", code)
        self.assertIn("wafer_maps_64.npy", code)
        self.assertIn("split_assignments.csv", code)
        self.assertIn("train_wm811k_cnn.py", code)
        self.assertIn("'--loss', 'focal'", code)
        self.assertIn("wm811k_cnn_results", code)
        self.assertIn("files.download", code)

    def test_wm811k_strategy_colab_notebook_contract(self) -> None:
        notebook_path = (
            PROJECT_DIR / "노트북" / "WM811K_03_취약클래스_개선_Colab.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("focal_shuffle", code)
        self.assertIn("weighted_ce_shuffle", code)
        self.assertIn("ce_sqrt_balanced", code)
        self.assertIn("sqrt_balanced", code)
        self.assertIn("validation_macro_f1", code)
        self.assertIn("'selection_uses_test': False", code)
        self.assertIn("validation_classification_report.csv", code)
        self.assertIn("wm811k_strategy_comparison_results", code)
        self.assertIn("files.download", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
