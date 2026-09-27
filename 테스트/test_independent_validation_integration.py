"""Contracts for compact independent-validation evidence and dashboard wiring."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import pandas as pd

try:
    from streamlit.testing.v1 import AppTest
except ModuleNotFoundError:  # Local static-check runtimes may omit Streamlit.
    AppTest = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_DIR = PROJECT_DIR / "결과물" / "wm811k" / "independent_validation"
WM_FINAL_DIR = WM_DIR / "final_comparison"
SECOM_DIR = PROJECT_DIR / "결과물" / "secom" / "independent_validation"


class IndependentValidationArtifactTests(unittest.TestCase):
    def test_compact_artifacts_exist_without_serialized_models(self) -> None:
        required = (
            WM_DIR / "shape_model" / "metrics.json",
            WM_FINAL_DIR / "paired_bootstrap.json",
            WM_FINAL_DIR / "model_comparison.csv",
            WM_FINAL_DIR / "class_disagreement_summary.csv",
            WM_DIR / "eda" / "eda_summary.json",
            WM_DIR / "provenance.json",
            SECOM_DIR / "candidate_oof_summary.csv",
            SECOM_DIR / "baseline_comparison.csv",
            SECOM_DIR / "selected_model.json",
            SECOM_DIR / "provenance.json",
        )
        for path in required:
            self.assertTrue(path.is_file(), path)
        self.assertFalse(any(WM_DIR.rglob("*.joblib")))
        self.assertFalse(any(SECOM_DIR.rglob("*.joblib")))
        self.assertFalse(any(WM_DIR.rglob("*predictions.csv")))
        self.assertFalse((SECOM_DIR / "test_predictions.csv").exists())

        shape_source = (
            PROJECT_DIR
            / "experiments"
            / "independent_validation"
            / "wm811k"
            / "wm811k_train_shape.py"
        ).read_text(encoding="utf-8")
        control_source = (
            PROJECT_DIR
            / "experiments"
            / "independent_validation"
            / "wm811k"
            / "wm811k_geometry_control.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"model_sha256": file_sha256(model_path)', shape_source)
        self.assertIn('"model_sha256": file_sha256(model_path)', control_source)

    def test_wm_shape_result_matches_the_deployed_cnn_report(self) -> None:
        paired = json.loads(
            (WM_FINAL_DIR / "paired_bootstrap.json").read_text(encoding="utf-8")
        )
        report = pd.read_csv(
            PROJECT_DIR
            / "결과물"
            / "wm811k"
            / "selected_model_results"
            / "test_classification_report.csv",
            index_col=0,
        )
        self.assertEqual(paired["n_test"], 24_705)
        self.assertAlmostEqual(
            paired["observed"]["cnn"]["macro_f1"],
            float(report.loc["macro avg", "f1-score"]),
            places=9,
        )
        for name, recall in paired["observed"]["cnn"]["per_class_recall"].items():
            self.assertAlmostEqual(recall, float(report.loc[name, "recall"]), places=9)

    def test_paired_bootstrap_conclusion_is_not_overstated(self) -> None:
        payload = json.loads(
            (WM_FINAL_DIR / "paired_bootstrap.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(payload["resamples"], 10_000)
        self.assertEqual(payload["rng"], "numpy.random.default_rng")
        for metric in ("macro_f1", "balanced_accuracy"):
            result = payload["overall_difference"][metric]
            self.assertLessEqual(result["ci95_low"], 0.0)
            self.assertGreaterEqual(result["ci95_high"], 0.0)
            self.assertFalse(result["excludes_zero"])

    def test_disagreement_summary_matches_mcnemar_totals(self) -> None:
        payload = json.loads(
            (WM_FINAL_DIR / "paired_bootstrap.json").read_text(encoding="utf-8")
        )
        frame = pd.read_csv(WM_FINAL_DIR / "class_disagreement_summary.csv")
        mcnemar = payload["mcnemar"]
        self.assertEqual(int(frame["support"].sum()), payload["n_test"])
        self.assertEqual(int(frame["both_correct"].sum()), mcnemar["both_correct"])
        self.assertEqual(
            int(frame["shape_only_correct"].sum()), mcnemar["shape_only_correct"]
        )
        self.assertEqual(
            int(frame["cnn_only_correct"].sum()), mcnemar["cnn_only_correct"]
        )
        self.assertEqual(int(frame["both_wrong"].sum()), mcnemar["both_wrong"])

    def test_wm_eda_counts_are_consistent(self) -> None:
        payload = json.loads(
            (WM_DIR / "eda" / "eda_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["n_labeled"] + payload["n_unlabeled"], payload["n_total"])
        self.assertEqual(sum(payload["failure_type_counts"].values()), payload["n_labeled"])
        self.assertEqual(payload["n_duplicate_lot_waferindex"], 0)

    def test_secom_model_was_selected_from_oof_not_test(self) -> None:
        selected = json.loads(
            (SECOM_DIR / "selected_model.json").read_text(encoding="utf-8")
        )
        candidates = pd.read_csv(SECOM_DIR / "candidate_oof_summary.csv")
        self.assertEqual(selected["candidate"], candidates.iloc[0]["candidate"])
        self.assertIn("OOF", selected["selection_rule"])
        self.assertFalse(selected["artifact_committed"])
        self.assertIn("expected_local_model_file", selected)
        self.assertNotIn("model_file", selected)

        source_path = (
            PROJECT_DIR
            / "experiments"
            / "independent_validation"
            / "secom"
            / "train_validate.py"
        )
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_candidates"
        )
        candidate_source = ast.get_source_segment(source, function) or ""
        self.assertNotIn("X_test", candidate_source)
        self.assertNotIn("y_test", candidate_source)


class IndependentValidationDashboardTests(unittest.TestCase):
    def test_dashboard_module_and_app_parse(self) -> None:
        module_path = PROJECT_DIR / "dashboard_ui" / "independent_validation.py"
        app_path = PROJECT_DIR / "app.py"
        module_source = module_path.read_text(encoding="utf-8")
        app_source = app_path.read_text(encoding="utf-8")
        ast.parse(module_source)
        ast.parse(app_source)
        self.assertIn("render_wm_independent_validation", app_source)
        self.assertIn("render_secom_independent_validation", app_source)
        for section in (
            "evaluation_safety",
            "evaluation_xai",
            "independent_validation",
            "research_log",
        ):
            self.assertIn(f'"{section}"', app_source)

    def test_dashboard_preserves_main_model_roles_and_limitations(self) -> None:
        source = (
            PROJECT_DIR / "dashboard_ui" / "independent_validation.py"
        ).read_text(encoding="utf-8")
        for phrase in (
            "배포 모델은 CNN입니다",
            "모델 교체 근거가 아닙니다",
            "물리적 불량 원인",
            "배포 후보가 아니라",
            "모델 바이너리는 저장소에 포함하지 않았습니다",
        ):
            self.assertIn(phrase, source)


@unittest.skipIf(AppTest is None, "Streamlit testing runtime is unavailable")
class IndependentValidationRenderTests(unittest.TestCase):
    """Exercise the real app routes so source-only checks cannot hide UI crashes."""

    @staticmethod
    def _rendered_text(app) -> str:
        chunks = []
        for name in (
            "title",
            "header",
            "subheader",
            "markdown",
            "caption",
            "info",
            "warning",
            "error",
        ):
            for element in getattr(app, name, []):
                chunks.append(str(getattr(element, "value", "")))
        return "\n".join(chunks)

    def test_wm_independent_validation_route_renders(self) -> None:
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=120)
        app.query_params["module"] = "wm811k"
        app.query_params["section"] = "independent_validation"
        app.run()
        self.assertEqual(len(app.exception), 0)
        rendered = self._rendered_text(app)
        self.assertIn("독립 검증 · 형상 대조와 데이터 구성", rendered)
        self.assertIn("배포 모델은 CNN입니다", rendered)
        metrics = {element.label: element.value for element in app.metric}
        self.assertEqual(metrics["형태 모델 Macro-F1"], "0.8794")
        self.assertEqual(metrics["CNN Macro-F1"], "0.8828")
        self.assertEqual(metrics["형태 모델만 정답"], "258개")
        self.assertEqual(metrics["CNN만 정답"], "361개")

    def test_wm_compact_performance_routes_render(self) -> None:
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=120)
        app.query_params["module"] = "wm811k"
        for section, expected_text in (
            ("evaluation", "운영 성능 요약"),
            ("evaluation_safety", "안전성·OOD"),
            ("evaluation_xai", "설명 검증"),
        ):
            with self.subTest(section=section):
                app.query_params["section"] = section
                app.run()
                self.assertEqual(len(app.exception), 0)
                self.assertIn(expected_text, self._rendered_text(app))

    def test_secom_independent_validation_route_renders(self) -> None:
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=120)
        app.query_params["module"] = "secom"
        app.query_params["section"] = "validation"
        app.run()
        self.assertEqual(len(app.exception), 0)
        labels = [element.label for element in app.expander]
        self.assertIn("독립 검증 · ExtraTrees 대조 모델", labels)
        rendered = self._rendered_text(app)
        self.assertIn("배포 후보가 아니라 독립 재현 참고값", rendered)
        metrics = {element.label: element.value for element in app.metric}
        self.assertEqual(metrics["ExtraTrees PR-AUC"], "0.2376")
        self.assertEqual(metrics["CatBoost PR-AUC"], "0.2505")


if __name__ == "__main__":
    unittest.main()
