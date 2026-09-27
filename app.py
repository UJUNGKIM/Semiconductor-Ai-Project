"""Interactive SECOM defect diagnosis dashboard."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_ui.navigation import (
    navigate_to_section,
    query_value,
    render_section_navigation,
    set_query_state,
)
from dashboard_ui.components import (
    render_dashboard_header,
    render_run_context,
)
from dashboard_ui import access as dashboard_access
from dashboard_ui import site, wafer_view, wm_gallery
from dashboard_ui.reports import build_diagnosis_bundle
from dashboard_ui.independent_validation import (
    render_secom_independent_validation,
    render_wm_independent_validation,
)
from dashboard_ui.secom import (
    audit_events_frame,
    build_profile_comparison,
    diagnosis_error_guidance,
    filter_review_queue,
    prepare_review_queue,
    review_filter_options,
)

PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_DIR / "코드" / "secom"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
WM_SCRIPTS_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPTS_DIR))

from diagnose_secom import consensus_label, parse_sensor_bytes, shap_matrix
from advanced_diagnosis import (
    assess_ood,
    assign_failure_signature,
    build_html_report,
    build_review_queue,
    constrained_counterfactual,
)
from batch_safety_gate import evaluate_batch_gate
from monitoring_log import (
    analyze_monitoring_history,
    append_monitoring_record,
    build_monitoring_record,
    monitoring_csv_bytes,
    validate_monitoring_log,
)
from feedback_monitoring import analyze_feedback_history, manifest_json_bytes
from row_label_feedback import (
    build_label_template,
    manifest_json_bytes as row_manifest_json_bytes,
    template_csv_bytes,
    validate_completed_feedback,
)
from champion_challenger import (
    comparison_json_bytes,
    evaluate_champion_challenger,
)
from audit_ledger import (
    append_event,
    ledger_jsonl_bytes,
    monitoring_record_payload,
    parse_ledger_jsonl,
    verify_ledger,
)
from production_audit_store import append_store_event, read_store, verify_store
from model_registry import (
    load_verified_bundles,
    sha256_file as secom_sha256_file,
    sha256_text_file as secom_sha256_text_file,
)
from external_validation import (
    evaluate_external_frame,
    parse_declarations_bytes,
    parse_external_csv_bytes,
    per_lot_csv_bytes,
    summary_json_bytes as external_summary_json_bytes,
)
from sensor_semantics import (
    normalized_dictionary_csv_bytes,
    parse_sensor_dictionary_bytes,
    sensor_report_json_bytes,
)
from runtime_environment import (
    build_runtime_environment_report,
    report_json_bytes as runtime_report_json_bytes,
)
from diagnose_wm811k import (
    assess_wafer_ood_batch,
    artificial_demo_map,
    build_wafer_audit_report,
    build_wafer_excel_report,
    canonical_wafer_shape_features,
    canonical_wafer_sha256,
    class_report_table,
    load_checkpoint,
    load_ood_reference,
    parse_wafer_bytes,
    predict_wafer_batch,
    predict_with_gradcam,
    predict_with_gradient_shap,
    sha256_file,
)
from wm811k_monitoring import assess_batch_drift, load_monitoring_reference
from validate_wm811k_gradient_shap_results import validate_gradient_shap_results
from wafer_cause_guidance import build_wafer_cause_guidance


ARTIFACTS_DIR = PROJECT_DIR / "결과물" / "secom"
MODEL_DIR = ARTIFACTS_DIR / "dual_model_results"
TABDDPM_RESULTS_DIR = (
    ARTIFACTS_DIR / "colab_가져오기" / "secom_tabddpm_results"
)
ADVANCED_DIAGNOSTICS_DIR = ARTIFACTS_DIR / "advanced_diagnostics"
SAFETY_POLICY_DIR = ARTIFACTS_DIR / "safety_policy_results"
UNCERTAINTY_RESCUE_DIR = ARTIFACTS_DIR / "uncertainty_rescue_results"
ROBUSTNESS_DIR = ARTIFACTS_DIR / "robustness_results"
BATCH_GATE_DIR = ARTIFACTS_DIR / "batch_safety_gate"
SENSOR_FAILURE_CONTAINMENT_DIR = ARTIFACTS_DIR / "sensor_failure_containment"
FEEDBACK_MONITORING_DIR = ARTIFACTS_DIR / "feedback_monitoring"
CHAMPION_CHALLENGER_DIR = ARTIFACTS_DIR / "champion_challenger"
AUDIT_LEDGER_DIR = ARTIFACTS_DIR / "audit_ledger"
SHAP_RELIABILITY_DIR = ARTIFACTS_DIR / "shap_reliability"
MODEL_FAILOVER_DIR = ARTIFACTS_DIR / "model_failover"
DATA_LINEAGE_DIR = ARTIFACTS_DIR / "data_lineage"
ENVIRONMENT_PROVENANCE_DIR = ARTIFACTS_DIR / "environment_provenance"
DEPENDENCY_LOCK_DIR = ARTIFACTS_DIR / "dependency_lock"
LINUX_DEPENDENCY_LOCK_DIR = ARTIFACTS_DIR / "linux_dependency_lock"
DEPENDENCY_SECURITY_DIR = ARTIFACTS_DIR / "dependency_security"
SOURCE_SECURITY_DIR = ARTIFACTS_DIR / "source_security"
RELEASE_READINESS_DIR = ARTIFACTS_DIR / "release_readiness"
RUNTIME_AUDIT_DATABASE = ARTIFACTS_DIR / "production_audit_store" / "dashboard_runtime.sqlite3"
EXTERNAL_VALIDATION_DIR = ARTIFACTS_DIR / "external_validation"
SENSOR_SEMANTICS_DIR = ARTIFACTS_DIR / "sensor_semantics"
SECOM_INDEPENDENT_VALIDATION_DIR = ARTIFACTS_DIR / "independent_validation"
SENSOR_DICTIONARY_TEMPLATE_PATH = (
    PROJECT_DIR / "데이터" / "SECOM 데이터셋" / "sensor_dictionary_template.csv"
)
WM_RESULTS_DIR = (
    PROJECT_DIR
    / "결과물"
    / "wm811k"
    / "selected_model_results"
)
WM_CHECKPOINT_PATH = WM_RESULTS_DIR / "best_model.pt"
WM_DEMO_DIR = PROJECT_DIR / "결과물" / "wm811k" / "demo_samples"
WM_OOD_DIR = PROJECT_DIR / "결과물" / "wm811k" / "ood_results"
WM_ROBUSTNESS_DIR = PROJECT_DIR / "결과물" / "wm811k" / "robustness_results"
WM_MULTISEED_DIR = PROJECT_DIR / "결과물" / "wm811k" / "multiseed_results"
WM_LOT_AUDIT_DIR = PROJECT_DIR / "결과물" / "wm811k" / "lot_generalization_results"
WM_GEOMETRY_DIR = PROJECT_DIR / "결과물" / "wm811k" / "geometry_adjusted_results"
WM_GRADCAM_AUDIT_DIR = PROJECT_DIR / "결과물" / "wm811k" / "gradcam_faithfulness_results"
WM_XAI_COMPARISON_DIR = PROJECT_DIR / "결과물" / "wm811k" / "xai_comparison_results"
WM_GRADIENT_SHAP_VALIDATION_DIR = (
    PROJECT_DIR / "결과물" / "wm811k" / "xai_gradient_shap_validation"
)
WM_SPLIT_ASSIGNMENTS_PATH = (
    PROJECT_DIR / "결과물" / "wm811k" / "split_results" / "split_assignments.csv"
)
WM_NEAR_FULL_DIR = PROJECT_DIR / "결과물" / "wm811k" / "near_full_diagnostic_results"
WM_NEAR_FULL_SHAPE_DIR = (
    PROJECT_DIR / "결과물" / "wm811k" / "near_full_global_shape_results"
)
WM_WEAK_CLASS_DIR = PROJECT_DIR / "결과물" / "wm811k" / "weak_class_results"
WM_WEAK_VALIDATION_DIR = (
    PROJECT_DIR / "결과물" / "wm811k" / "weak_class_validation_results"
)
WM_WEAK_MARGIN_DIR = PROJECT_DIR / "결과물" / "wm811k" / "weak_margin_results"
WM_WEAK_MARGIN_MULTISEED_DIR = (
    PROJECT_DIR / "결과물" / "wm811k" / "weak_margin_multiseed_results"
)
WM_MODEL_REGISTRY_DIR = PROJECT_DIR / "결과물" / "wm811k" / "model_registry"
WM_MONITORING_DIR = PROJECT_DIR / "결과물" / "wm811k" / "monitoring_reference"
WM_SSL_VALIDATION_DIR = (
    PROJECT_DIR / "결과물" / "wm811k" / "ssl_validation_results"
)
WM_SSL_OOF_CALIBRATION_DIR = (
    PROJECT_DIR / "결과물" / "wm811k" / "ssl_oof_calibration_results"
)
WM_INDEPENDENT_VALIDATION_DIR = (
    PROJECT_DIR / "결과물" / "wm811k" / "independent_validation"
)
WM_OOD_LABELS = {
    "in_distribution": "분포 내 입력",
    "review": "분포 경계 · 검토",
    "out_of_distribution": "분포 이탈 · 판정 보류",
}
MODEL_NAMES = ("CatBoost", "XGBoost")
PROFILE_LABELS = {
    "균형형 · F2 최대": "balanced_f2",
    "목표 재현율 50%": "recall_50",
    "목표 재현율 60%": "recall_60",
    "목표 재현율 70%": "recall_70",
    "목표 재현율 80%": "recall_80",
    "목표 재현율 90%": "recall_90",
}
CONSENSUS_KOREAN = {
    "both_models_defect": "두 모델 모두 불량",
    "catboost_only_defect": "CatBoost만 불량",
    "xgboost_only_defect": "XGBoost만 불량",
    "both_models_normal": "두 모델 모두 정상",
}


st.set_page_config(
    page_title="SHAPGPT · 반도체 불량 진단",
    page_icon=":material/memory:",
    layout="wide",
)

# SHAPGPT_REQUIRE_LOGIN=1 turns on OIDC login. Administrator screens follow the
# signed-in user's verified email or issuer|subject allowlist; see
# dashboard_ui/access.py. SHAPGPT_ADMIN_MODE only works without login on a
# loopback server address.
ACCESS = dashboard_access.resolve_access(
    env=os.environ,
    claims=dashboard_access.current_user_claims(),
    secrets=dashboard_access.streamlit_secrets(),
    server_address=st.get_option("server.address"),
)


def format_model_score(value: float) -> str:
    """Format a model score without presenting rounded 100% as certainty."""
    score = float(value)
    if score >= 0.9995:
        return ">99.9%"
    return f"{score:.1%}"


def render_unified_xai_frame(*, explanation: str, input_limit: str) -> None:
    """Show the common decision frame shared by both data modalities."""
    st.markdown("#### 공통 SHAP 의사결정 프레임")
    columns = st.columns(4)
    steps = (
        ("1. 입력 신뢰도", input_limit),
        ("2. 모델 판정", "점수·임계값과 예측 결과"),
        ("3. SHAP 기여", explanation),
        ("4. 검토 행동", "자동 확정·전문가 검토·판정 보류"),
    )
    for column, (title, body) in zip(columns, steps):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(body)


def wm_xai_matches_deployment() -> bool:
    summary_path = WM_XAI_COMPARISON_DIR / "xai_method_summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    deployed_hash = load_file_sha256(
        str(WM_CHECKPOINT_PATH), WM_CHECKPOINT_PATH.stat().st_mtime_ns
    ).lower()
    return summary.get("checkpoint_sha256") == deployed_hash


def inject_dashboard_style() -> None:
    """Apply a shared, accessible visual system to the Streamlit dashboard."""
    st.markdown(
        """
        <style>
        :root {
            --dashboard-navy: #0b2545;
            --dashboard-blue: #13315c;
            --dashboard-cyan: #0f766e;
            --dashboard-amber: #7a7a7a;
            --dashboard-ink: #102a43;
            --dashboard-muted: #52606d;
            --dashboard-line: #e2e8f0;
            --dashboard-surface: #ffffff;
            --dashboard-bg: #ffffff;
            --dashboard-soft: #f5f8fa;
        }

        .stApp {
            background: var(--dashboard-bg);
            color: var(--dashboard-ink);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1480px;
            padding: 2.6rem 2.25rem 4rem;
        }

        [data-testid="stToolbarActions"],
        [data-testid="stMainMenu"],
        [data-testid="stStatusWidget"],
        [data-testid="stDecoration"] {
            display: none !important;
        }

        [data-testid="stToolbar"] {
            display: flex !important;
            background: transparent !important;
        }

        [data-testid="stExpandSidebarButton"] {
            display: flex !important;
        }

        [data-testid="stHeader"] {
            background: #ffffff;
            border-bottom: 1px solid var(--dashboard-line);
        }

        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid var(--dashboard-line);
        }

        [data-testid="stSidebarContent"] {
            padding-top: 1.25rem;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--dashboard-navy);
            letter-spacing: -0.02em;
        }

        .st-key-dashboard_hero {
            position: relative;
            overflow: hidden;
            padding: 1.05rem 1.3rem 0.95rem;
            margin-bottom: 0.9rem;
            color: var(--dashboard-ink);
            border: 1px solid #e5e7eb;
            border-radius: 0.9rem;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(31, 41, 51, 0.045);
        }

        .st-key-dashboard_hero::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 0.22rem;
            background: var(--dashboard-cyan);
        }

        .st-key-dashboard_hero h1 {
            margin: 0.2rem 0 0 !important;
            padding: 0 !important;
            color: var(--dashboard-navy) !important;
            font-size: clamp(1.55rem, 2.35vw, 2.05rem) !important;
            line-height: 1.18 !important;
            letter-spacing: -0.035em !important;
        }

        .st-key-dashboard_hero [data-testid="stCaptionContainer"] {
            max-width: 68rem;
            margin-top: 0.42rem;
            color: #6b7280 !important;
            font-size: 0.86rem;
            line-height: 1.52;
            opacity: 1 !important;
        }

        .st-key-dashboard_hero [data-testid="stCaptionContainer"] p {
            color: #6b7280 !important;
            opacity: 1 !important;
        }

        .run-context {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem 1rem;
            margin: 0 0 0.8rem;
            padding: 0.58rem 0.1rem;
            color: #6b7280;
            border-bottom: 1px solid #eceef1;
            font-size: 0.76rem;
            line-height: 1.4;
        }

        .run-context span {
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .run-context b {
            margin-right: 0.18rem;
            color: #374151;
            font-weight: 750;
        }

        .review-detail-card {
            padding: 0.95rem 1rem;
            margin: 0.85rem 0 0.65rem;
            border: 1px solid #e5e7eb;
            border-radius: 0.85rem;
            background: #fafafa;
        }

        .review-detail-card h4 {
            margin: 0 0 0.4rem;
            font-size: 1rem;
        }

        .review-detail-card p {
            margin: 0.18rem 0;
            color: #5f6670;
            font-size: 0.79rem;
        }

        .review-detail-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.55rem 1rem;
            color: #5f6670;
            font-size: 0.79rem;
        }

        .review-detail-grid span {
            min-width: 0;
            overflow-wrap: anywhere;
        }

        .review-detail-grid b {
            display: block;
            margin-bottom: 0.12rem;
            color: #374151;
            font-size: 0.72rem;
        }

        [data-testid="stMetric"] {
            min-height: 6rem;
            padding: 0.82rem 0.92rem;
            border: 1px solid var(--dashboard-line);
            border-radius: 0.85rem;
            background: #ffffff;
            box-shadow:
                inset 0 2px 0 #cfe6e2,
                0 5px 16px rgba(11, 37, 69, 0.045);
        }

        [data-testid="stMetricLabel"] {
            color: var(--dashboard-muted);
            font-weight: 650;
            font-size: 0.8rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--dashboard-navy);
            font-size: clamp(1.25rem, 1.8vw, 1.72rem);
            font-weight: 780;
            line-height: 1.15;
            letter-spacing: -0.035em;
        }

        [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"],
        [data-testid="stMetricValue"] p {
            overflow: visible;
            overflow-wrap: anywhere;
            text-overflow: clip;
            white-space: normal;
        }

        .st-key-secom_metric_primary [data-testid="stMetric"] {
            border-color: #9fb3c8;
            background: #ffffff;
            box-shadow: inset 0 3px 0 var(--dashboard-navy), 0 7px 18px rgba(11, 37, 69, 0.07);
        }

        .st-key-secom_metric_primary [data-testid="stMetricLabel"] {
            color: var(--dashboard-blue);
        }

        .st-key-secom_metric_primary [data-testid="stMetricValue"] {
            color: var(--dashboard-navy);
        }

        .st-key-secom_metric_attention [data-testid="stMetric"] {
            border-color: #d1d5db;
            background: #ffffff;
            box-shadow: inset 0 2px 0 #d1d5db;
        }

        .st-key-secom_metric_normal [data-testid="stMetric"],
        .st-key-secom_metric_agreement [data-testid="stMetric"] {
            border-color: #e5e7eb;
            background: #ffffff;
            box-shadow: none;
        }

        .st-key-secom_metric_normal [data-testid="stMetricValue"],
        .st-key-secom_metric_agreement [data-testid="stMetricValue"] {
            color: #56514b;
        }

        /* Keep lazy rendering while presenting the controls like real tabs. */
        [role="radiogroup"][aria-label="업무 영역"],
        [role="radiogroup"][aria-label="세부 화면"] {
            display: flex;
            justify-content: flex-start;
            gap: 1.35rem;
            min-height: 2.6rem;
            padding: 0;
            overflow-x: auto;
            border: 0;
            border-bottom: 1px solid #e5e7eb;
            border-radius: 0;
            background: transparent;
            scrollbar-width: thin;
        }

        [role="radiogroup"][aria-label="업무 영역"] {
            margin-top: 0.2rem;
        }

        [role="radiogroup"][aria-label="세부 화면"] {
            min-height: 2.45rem;
            margin: 0.05rem 0 0.75rem;
        }

        [role="radiogroup"][aria-label="업무 영역"] > button,
        [role="radiogroup"][aria-label="세부 화면"] > button {
            flex: 0 0 auto !important;
            min-height: 2.45rem;
            padding: 0.42rem 0.1rem 0.52rem !important;
            border: 0 !important;
            border-radius: 0 !important;
            color: #737983 !important;
            background: transparent !important;
            box-shadow: none !important;
            font-size: 0.88rem;
            font-weight: 650;
        }

        [role="radiogroup"][aria-label="업무 영역"] > button:hover,
        [role="radiogroup"][aria-label="세부 화면"] > button:hover {
            color: #20242a !important;
            background: transparent !important;
        }

        [role="radiogroup"][aria-label="업무 영역"] > button[data-selected="true"],
        [role="radiogroup"][aria-label="세부 화면"] > button[data-selected="true"] {
            color: var(--dashboard-navy) !important;
            background: transparent !important;
            box-shadow: inset 0 -2px 0 var(--dashboard-cyan) !important;
            font-weight: 780;
        }

        [role="radiogroup"][aria-label="표시 범위"] {
            display: flex;
            justify-content: flex-start;
            gap: 0.42rem;
            padding: 0.08rem 0 0.28rem;
            overflow-x: auto;
            border: 0;
            border-radius: 0;
            background: transparent;
            scrollbar-width: thin;
        }

        [role="radiogroup"][aria-label="표시 범위"] > button {
            flex: 0 0 auto !important;
            min-height: 2rem;
            padding: 0.34rem 0.72rem !important;
            border: 1px solid #e1e4e8 !important;
            border-radius: 999px !important;
            color: #626974 !important;
            background: #ffffff !important;
            box-shadow: none !important;
            font-size: 0.8rem;
            font-weight: 650;
        }

        [role="radiogroup"][aria-label="표시 범위"] > button:hover {
            border-color: #b7bdc6 !important;
            color: #20242a !important;
            background: #f7f7f8 !important;
        }

        [role="radiogroup"][aria-label="표시 범위"] > button[data-selected="true"] {
            border-color: var(--dashboard-cyan) !important;
            color: var(--dashboard-navy) !important;
            background: #e6f4f2 !important;
            box-shadow: inset 0 0 0 1px var(--dashboard-cyan) !important;
            font-weight: 750;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            overflow: hidden;
            border: 1px solid var(--dashboard-line);
            border-radius: 0.9rem;
            background: var(--dashboard-surface);
            box-shadow: 0 5px 18px rgba(40, 62, 94, 0.045);
        }

        [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
            color: #374151 !important;
            border: 1px solid #e5e7eb;
            background: #ffffff !important;
        }

        [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
            color: var(--dashboard-blue) !important;
            border: 1px solid #d6e2ee;
            background: #f1f6fb !important;
        }

        [data-testid="stAlertContentSuccess"] p,
        [data-testid="stAlertContentInfo"] p {
            color: inherit !important;
        }

        [data-testid="stAlert"],
        [data-testid="stExpander"],
        [data-testid="stFileUploader"] {
            border-radius: 0.9rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 0.9rem;
            border-color: var(--dashboard-line) !important;
            background: rgba(255, 255, 255, 0.7);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 0.72rem;
            font-weight: 700;
            transition: transform 120ms ease, box-shadow 120ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 7px 16px rgba(31, 41, 51, 0.12);
        }

        h1, h2, h3, h4 {
            color: var(--dashboard-navy);
            letter-spacing: -0.025em;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--dashboard-line);
            background: rgba(255, 255, 255, 0.54);
        }

        hr {
            border-color: var(--dashboard-line) !important;
        }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] {
                padding: 2.4rem 0.9rem 2.75rem;
            }

            .st-key-dashboard_hero {
                padding: 1.05rem 0.95rem 0.9rem;
                border-radius: 0.85rem;
            }

            [data-testid="stMetric"] {
                min-height: 5.55rem;
            }
        }

        @media (max-width: 700px) {
            [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.65rem;
            }

            [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"])
            > [data-testid="stColumn"] {
                width: auto !important;
                min-width: 0 !important;
                flex: none !important;
            }

            [data-testid="stMetric"] {
                padding: 0.75rem 0.78rem;
            }

            [data-testid="stMetricValue"] {
                font-size: 1.25rem;
            }

            .review-detail-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

        }

        </style>
        """,
        unsafe_allow_html=True,
    )
    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = [
        "#0b2545",
        "#0f766e",
        "#5bb5a9",
        "#13315c",
        "#8da9c4",
        "#94a3b8",
    ]


inject_dashboard_style()
site.inject_site_style()


@st.cache_resource
def load_bundles(
    registry_path: str,
    registry_modified_ns: int,
    expected_registry_hash: str,
) -> dict[str, dict]:
    del registry_modified_ns
    path = Path(registry_path)
    if secom_sha256_text_file(path) != expected_registry_hash.upper():
        raise ValueError("모델 레지스트리 SHA-256이 릴리스 manifest와 다릅니다.")
    return load_verified_bundles(path, PROJECT_DIR)


@st.cache_resource
def load_advanced_reference(
    path: str, modified_ns: int, expected_hash: str
) -> dict:
    del modified_ns
    if secom_sha256_file(Path(path)) != expected_hash.upper():
        raise ValueError("고급 진단 참조 SHA-256이 릴리스 manifest와 다릅니다.")
    return joblib.load(path)


@st.cache_data
def load_sample_rows(limit: int) -> pd.DataFrame:
    path = PROJECT_DIR / "데이터" / "SECOM 데이터셋" / "raw" / "secom.data"
    return read_sensor_data(path.read_bytes(), limit)


def read_sensor_data(raw_bytes: bytes, limit: int | None = None) -> pd.DataFrame:
    frame, _ = parse_sensor_bytes(raw_bytes, limit)
    return frame


@st.cache_data
def sample_csv_template() -> bytes:
    sample = load_sample_rows(1).copy()
    sample.insert(0, "timestamp", "2008-07-19 11:55:00")
    sample.insert(0, "row_id", "sample_001")
    return sample.to_csv(index=False).encode("utf-8-sig")


def run_predictions(
    X_raw: pd.DataFrame, bundles: dict[str, dict], profile: str
) -> tuple[pd.DataFrame, dict[str, dict]]:
    results = pd.DataFrame({"행 번호": np.arange(len(X_raw), dtype=int)})
    prepared = {}
    predictions = {}
    for model_name in MODEL_NAMES:
        bundle = bundles[model_name]
        X_model = X_raw[bundle["input_features"]]
        X_imputed = bundle["imputer"].transform(X_model)
        X_ready = bundle["selector"].transform(X_imputed)
        probability = bundle["model"].predict_proba(X_ready)[:, 1]
        threshold = float(bundle["operating_thresholds"][profile])
        prediction = (probability >= threshold).astype(int)
        key = model_name.lower()
        results[f"{model_name} 점수"] = probability
        results[f"{model_name} 임계값"] = threshold
        results[f"{model_name} 판정"] = np.where(prediction == 1, "불량", "정상")
        prepared[model_name] = {
            "X_ready": X_ready,
            "probability": probability,
            "prediction": prediction,
            "threshold": threshold,
            "selected_features": bundle["selected_features"],
            "model": bundle["model"],
        }
        predictions[model_name] = prediction
    consensus = [
        consensus_label(int(cat), int(xgb))
        for cat, xgb in zip(predictions["CatBoost"], predictions["XGBoost"])
    ]
    results["종합 판정"] = [CONSENSUS_KOREAN[value] for value in consensus]
    results["우선 확인 점수"] = np.maximum(
        results["CatBoost 점수"] / results["CatBoost 임계값"],
        results["XGBoost 점수"] / results["XGBoost 임계값"],
    )
    return results, prepared


def explain_selected_row(
    model_name: str, info: dict, row_number: int, top_n: int
) -> pd.DataFrame:
    row_matrix = info["X_ready"][row_number : row_number + 1]
    values = shap_matrix(info["model"], row_matrix)[0]
    order = np.argsort(np.abs(values))[::-1][:top_n]
    explanation = pd.DataFrame(
        {
            "변수": [info["selected_features"][index] for index in order],
            "입력값(결측 대체 후)": row_matrix[0, order],
            "SHAP 기여도": values[order],
        }
    )
    explanation["방향"] = np.where(
        explanation["SHAP 기여도"] > 0, "불량 쪽", "정상 쪽"
    )
    return explanation.sort_values("SHAP 기여도")


def display_score(model_name: str, info: dict, row_number: int) -> None:
    probability = float(info["probability"][row_number])
    threshold = float(info["threshold"])
    prediction = "불량" if probability >= threshold else "정상"
    st.metric(
        model_name,
        prediction,
        delta=f"점수 {probability:.4f} / 임계값 {threshold:.4f}",
        delta_color="inverse" if prediction == "불량" else "normal",
    )


@st.cache_resource
def load_wafer_model(checkpoint_path: str, modified_ns: int):
    del modified_ns
    return load_checkpoint(Path(checkpoint_path))


@st.cache_resource
def load_wafer_ood_reference(
    metadata_path: str,
    arrays_path: str,
    metadata_modified_ns: int,
    arrays_modified_ns: int,
):
    del metadata_modified_ns, arrays_modified_ns
    return load_ood_reference(Path(metadata_path), Path(arrays_path))


@st.cache_data
def load_file_sha256(path: str, modified_ns: int) -> str:
    del modified_ns
    return sha256_file(Path(path))


@st.cache_data(show_spinner=False, max_entries=64)
def explain_wafer_gradient_shap(
    checkpoint_path: str, modified_ns: int, wafer: np.ndarray, temperature: float
) -> dict:
    """Cache Gradient SHAP per wafer so widget reruns do not redo the path integral."""
    model, class_names, _ = load_wafer_model(checkpoint_path, modified_ns)
    return predict_with_gradient_shap(model, class_names, wafer, temperature=temperature)


@st.cache_data(show_spinner=False)
def load_gradient_shap_evidence(
    result_dir: str,
    summary_modified_ns: int,
    checkpoint_path: str,
    checkpoint_modified_ns: int,
    assignments_path: str,
    assignments_modified_ns: int,
) -> dict:
    """Validate the Gradient SHAP evidence against the deployed checkpoint."""
    del summary_modified_ns, checkpoint_modified_ns, assignments_modified_ns
    if not (Path(result_dir) / "xai_method_summary.json").is_file():
        return {"status": "missing"}
    try:
        report = validate_gradient_shap_results(
            Path(result_dir), Path(assignments_path), Path(checkpoint_path)
        )
    except (ValueError, KeyError, FileNotFoundError) as error:
        return {"status": "invalid", "error": str(error)}
    return {"status": "verified", "report": report}


def wm_gradient_shap_evidence() -> dict:
    summary_path = WM_GRADIENT_SHAP_VALIDATION_DIR / "xai_method_summary.json"
    return load_gradient_shap_evidence(
        str(WM_GRADIENT_SHAP_VALIDATION_DIR),
        summary_path.stat().st_mtime_ns if summary_path.is_file() else 0,
        str(WM_CHECKPOINT_PATH),
        WM_CHECKPOINT_PATH.stat().st_mtime_ns,
        str(WM_SPLIT_ASSIGNMENTS_PATH),
        WM_SPLIT_ASSIGNMENTS_PATH.stat().st_mtime_ns,
    )


def gradient_shap_evidence_statement(evidence: dict) -> str:
    if evidence["status"] == "verified":
        shap_summary = evidence["report"]["gradient_shap"]
        return (
            "Gradient SHAP 배포 검증: 현재 배포 체크포인트로 고정 test "
            f"{evidence['report']['sample_count']}개에서 원시 가산성 통과 "
            f"{shap_summary['additivity_pass_rate']:.1%}와 결함→정상 교란 점검을 "
            "수행했습니다(모델 선택 후 사후 점검)."
        )
    if evidence["status"] == "invalid":
        return (
            "Gradient SHAP 검증 결과가 현재 배포 체크포인트·분할표와 일치하지 않거나 "
            "손상되어 근거로 사용하지 않습니다."
        )
    return "Gradient SHAP 배포 검증: 현재 증거 없음(검증 예정)."


@st.cache_data
def load_wafer_demo_manifest(manifest_path: str, modified_ns: int) -> dict:
    del modified_ns
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


@st.cache_data
def load_wafer_evaluation(
    summary_path: str, matrix_path: str, report_path: str
) -> tuple[dict, dict, pd.DataFrame]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    matrix = pd.read_csv(matrix_path, index_col=0)
    if "none" not in matrix.index or "none" not in matrix.columns:
        raise ValueError("혼동행렬에 none 클래스가 없습니다.")
    defect_rows = matrix.index != "none"
    defect_columns = matrix.columns != "none"
    true_positive = int(matrix.loc[defect_rows, defect_columns].to_numpy().sum())
    false_positive = int(matrix.loc["none", defect_columns].sum())
    false_negative = int(matrix.loc[defect_rows, "none"].sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    binary_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    binary = {
        "precision": precision,
        "recall": recall,
        "f1": binary_f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "test_rows": int(matrix.to_numpy().sum()),
    }
    return summary, binary, class_report_table(Path(report_path))


@st.cache_data(show_spinner=False, max_entries=128)
def demo_wafer_png(npy_path: str, modified_ns: int, min_side: int) -> bytes:
    """Render a stored demo wafer for the gallery without running the model."""
    del modified_ns
    path = Path(npy_path)
    wafer = parse_wafer_bytes(path.read_bytes(), path.name)
    return wafer_view.categorical_png(wafer, min_side=min_side)[0]


@st.cache_data(show_spinner=False)
def load_json_file(path: str, modified_ns: int) -> dict:
    del modified_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv_file(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return pd.read_csv(path)


def collect_site_evidence() -> dict:
    """Headline numbers for the public pages, read from saved result files only."""
    evidence: dict = {"wm": None, "shap": None, "secom": None, "readiness": None}
    try:
        summary, binary, class_report = load_wafer_evaluation(
            str(WM_RESULTS_DIR / "run_summary.json"),
            str(WM_RESULTS_DIR / "test_confusion_matrix.csv"),
            str(WM_RESULTS_DIR / "test_classification_report.csv"),
        )
        policy_path = WM_RESULTS_DIR / "uncertainty_policy.json"
        policy_test = load_json_file(
            str(policy_path), policy_path.stat().st_mtime_ns
        )["test_evaluation"]
        evidence["wm"] = {
            "macro_f1": float(summary["test_metrics"]["macro_f1"]),
            "accuracy": float(summary["test_metrics"]["accuracy"]),
            "balanced_accuracy": float(summary["test_metrics"]["balanced_accuracy"]),
            "recall": float(binary["recall"]),
            "test_rows": int(binary["test_rows"]),
            "class_count": int(len(class_report)),
            "automatic_coverage": float(policy_test["automatic_coverage"]),
            "error_capture_rate": float(policy_test["error_capture_rate"]),
        }
    except (OSError, KeyError, ValueError):
        evidence["wm"] = None
    try:
        shap_evidence = wm_gradient_shap_evidence()
    except OSError:
        shap_evidence = {"status": "missing"}
    if shap_evidence["status"] == "verified":
        report = shap_evidence["report"]
        evidence["shap"] = {
            "status": "verified",
            "pass_count": int(report["gradient_shap"]["additivity_pass_count"]),
            "sample_count": int(report["sample_count"]),
            "relative_tolerance": float(report["gradient_shap"]["relative_tolerance"]),
            "top_vs_random": float(
                report["defect_flip"]["Gradient SHAP"]["macro_top_vs_random_advantage"]
            ),
        }
    else:
        evidence["shap"] = {"status": shap_evidence["status"]}
    operating_path = MODEL_DIR / "oof_operating_points.csv"
    folds_path = MODEL_DIR / "repeated_cv_fold_auc.csv"
    if operating_path.is_file() and folds_path.is_file():
        operating = load_csv_file(
            str(operating_path), operating_path.stat().st_mtime_ns
        )
        folds = load_csv_file(str(folds_path), folds_path.stat().st_mtime_ns)
        balanced = operating.loc[operating["operating_point"] == "balanced_f2"]
        evidence["secom"] = [
            {
                "model": str(row["model"]),
                "pr_auc": float(row["pr_auc"]),
                "recall": float(row["recall"]),
                "precision": float(row["precision"]),
                "folds": int(folds["fold"].nunique()),
                "repeats": int(folds["repeat"].nunique()),
                "normal_per_defect": float(
                    (row["fp"] + row["tn"]) / max(row["tp"] + row["fn"], 1)
                ),
            }
            for _, row in balanced.iterrows()
        ]
    readiness_path = RELEASE_READINESS_DIR / "release_readiness.json"
    if readiness_path.is_file():
        evidence["readiness"] = load_json_file(
            str(readiness_path), readiness_path.stat().st_mtime_ns
        )
    return evidence


def render_gradient_shap_validation() -> None:
    """Deployed-checkpoint Gradient SHAP evidence, read from saved results."""
    st.markdown("#### 현재 배포 설명법 · Gradient SHAP 검증")
    evidence = wm_gradient_shap_evidence()
    if evidence["status"] == "verified":
        report = evidence["report"]
        shap_summary = report["gradient_shap"]
        shap_flip = report["defect_flip"]["Gradient SHAP"]
        shap_columns = st.columns(4)
        shap_columns[0].metric(
            "원시 가산성 통과",
            f"{shap_summary['additivity_pass_count']}/{report['sample_count']}",
            f"{shap_summary['additivity_pass_rate']:.1%}",
            delta_color="off",
        )
        shap_columns[1].metric(
            "상대 잔차 중앙값",
            f"{shap_summary['relative_residual_median']:.2%}",
            f"90% {shap_summary['relative_residual_p90']:.2%}",
            delta_color="off",
        )
        shap_columns[2].metric(
            "결함→정상 상위−무작위",
            f"{shap_flip['macro_top_vs_random_advantage']:+.4f}",
            f"상위 > 무작위 {shap_flip['top_exceeds_random_rate']:.1%}",
            delta_color="off",
        )
        paired = report["paired_gradient_shap_minus_integrated_gradients_defect_flip"]
        paired_interval = report[
            "paired_gradient_shap_minus_integrated_gradients_defect_flip_95ci"
        ]
        shap_columns[3].metric(
            "Gradient SHAP−IG",
            f"{paired:+.4f}",
            f"95% {paired_interval[0]:+.4f}~{paired_interval[1]:+.4f}",
            delta_color="off",
        )
        st.caption(
            f"현재 배포 체크포인트({str(report['checkpoint_sha256'])[:12]}…)로 고정 test "
            f"클래스 균형 표본 {report['sample_count']}개를 {report['device']}에서 계산했습니다. "
            "가산성은 사후 보정 전 원시 attribution 합계와 로짓 변화의 차이이며, "
            f"허용오차(로짓 변화의 {shap_summary['relative_tolerance']:.0%})를 넘은 웨이퍼는 "
            "통과로 세지 않았습니다. 결함→정상 교란은 상위 10% 결함 die를 정상으로 바꿨을 때의 "
            "예측 점수 하락을 같은 수의 무작위 결함 die와 비교합니다."
        )
        for warning in report["warnings"]:
            st.caption("분할표 해시: " + warning)
        with st.expander("Gradient SHAP 검증 그래프"):
            st.image(
                str(WM_GRADIENT_SHAP_VALIDATION_DIR / "xai_method_comparison_dashboard.png"),
                caption="고정 Test 균형 표본의 die 제거·결함 교란 충실도와 원시 가산성 잔차",
                width="stretch",
            )
        st.warning(
            "모델 선택이 끝난 뒤의 사후 점검입니다. 클래스별 최대 50개 균형 표본이라 실제 "
            "test 분포 평균이 아니며, 교란은 실제 공정 개입이 아닙니다."
        )
    elif evidence["status"] == "invalid":
        st.error(
            gradient_shap_evidence_statement(evidence)
            + f" ({evidence.get('error', '')})"
        )
    else:
        st.warning(gradient_shap_evidence_statement(evidence))


def render_secom_shap_reliability() -> None:
    """TreeSHAP stability and faithfulness audit, read from saved results."""
    reliability_path = SHAP_RELIABILITY_DIR / "shap_reliability_summary.json"
    if reliability_path.is_file():
        st.divider()
        st.subheader("SHAP 설명 신뢰성 감사")
        reliability = json.loads(reliability_path.read_text(encoding="utf-8"))
        st.caption(
            f"모델 선택 후 고정 test {reliability['test_rows']}행에서 train IQR의 "
            f"{reliability['noise_fraction_of_train_iqr']:.0%} 노이즈를 "
            f"{reliability['stability_repeats']}회 적용하고, 상위 SHAP 변수와 동일 개수의 "
            "무작위 변수를 중앙값으로 가리는 삭제 충실도 실험입니다."
        )
        for model_name in MODEL_NAMES:
            item = reliability["models"][model_name]
            st.markdown(f"#### {model_name} · {item['status']}")
            reliability_cols = st.columns(4)
            reliability_cols[0].metric(
                "Top-10 Jaccard", f"{item['median_top10_jaccard']:.3f}"
            )
            reliability_cols[1].metric(
                "부호 일치율", f"{item['median_top10_sign_agreement']:.1%}"
            )
            reliability_cols[2].metric(
                "노이즈 판정 반전", f"{item['prediction_flip_rate_under_noise']:.1%}"
            )
            reliability_cols[3].metric(
                "상위 SHAP 마스킹 승률", f"{item['top_shap_mask_win_rate']:.1%}"
            )
        reliability_figure = SHAP_RELIABILITY_DIR / "shap_reliability_dashboard.png"
        if reliability_figure.is_file():
            st.image(
                str(reliability_figure),
                caption="작은 입력 교란 안정성과 동일 개수 변수 마스킹 충실도",
                width="stretch",
            )
        st.warning(
            "이 검증은 모델 설명의 수치적 안정성·충실도 근거입니다. 익명 센서의 물리적 "
            "원인, 실제 장비 고장 분포 또는 공정 개입의 인과 효과를 증명하지 않습니다."
        )
    else:
        st.info("SHAP 신뢰성 결과가 없습니다. evaluate_shap_reliability.py를 실행하세요.")


def render_secom_augmentation() -> None:
    st.subheader("생성형 AI 증강 검증")
    enhanced_dir = ARTIFACTS_DIR / "enhanced_synthetic_results"
    if enhanced_dir.exists():
        recommendation = json.loads(
            (enhanced_dir / "recommendation.json").read_text(encoding="utf-8")
        )
        quality = pd.read_csv(enhanced_dir / "synthetic_quality.csv")
        cv_summary = pd.read_csv(enhanced_dir / "cv_summary.csv")
        fold_quality = quality[quality["context"].str.startswith("fold_")]

        metrics = st.columns(4)
        metrics[0].metric(
            "DDPM 품질 점수", f"{fold_quality['quality_score'].mean():.4f}"
        )
        metrics[1].metric(
            "실제/합성 판별 AUC",
            f"{fold_quality['discriminator_auc'].mean():.4f}",
            help="0.5에 가까울수록 실제와 합성을 구별하기 어렵습니다.",
        )
        metrics[2].metric(
            "암기 위험률", f"{fold_quality['memorization_risk_rate'].mean():.2%}"
        )
        metrics[3].metric(
            "기준 대비 CV PR-AUC",
            f"{recommendation['cv_pr_auc_gain']:+.4f}",
        )

        display_cv = cv_summary[
            ["model", "strategy", "pr_auc_mean", "pr_auc_std", "recall_mean", "f1_mean"]
        ].rename(
            columns={
                "model": "모델",
                "strategy": "학습 전략",
                "pr_auc_mean": "CV PR-AUC",
                "pr_auc_std": "표준편차",
                "recall_mean": "재현율",
                "f1_mean": "F1",
            }
        )
        st.dataframe(display_cv, hide_index=True, width="stretch")
        st.image(
            str(enhanced_dir / "cv_pr_auc_comparison.png"),
            caption="모든 validation 행은 실제 데이터만 사용한 5-fold 비교",
            width="stretch",
        )
        st.image(
            str(enhanced_dir / "synthetic_quality_dashboard.png"),
            caption="fold 내부 DDPM 합성 데이터 품질",
            width="stretch",
        )
        if recommendation["adopt_for_primary_model"]:
            st.success("사전 정의한 교차검증 기준을 통과해 DDPM 증강을 채택했습니다.")
        else:
            st.warning(
                "DDPM은 다양성과 암기 위험 검사를 통과했지만 실제 validation PR-AUC를 "
                "개선하지 못했습니다. 주 모델은 scale_pos_weight CatBoost를 유지합니다."
            )
        st.caption(
            "증강 선택에는 test 결과를 사용하지 않았습니다. 합성 행 수는 새로운 실측 "
            "공정 데이터의 정보량과 같지 않습니다."
        )
    else:
        st.caption(
            "강화 증강 결과가 없습니다. .\\run_pipeline.ps1 -Mode augment를 실행하세요."
        )

    st.divider()
    st.subheader("공식 TabDDPM Colab 실험")
    tabddpm_required = {
        "실행 메타데이터": TABDDPM_RESULTS_DIR / "run_metadata.json",
        "validation 결과": TABDDPM_RESULTS_DIR / "validation_results.csv",
        "test 결과": TABDDPM_RESULTS_DIR / "test_results.csv",
        "합성 품질": TABDDPM_RESULTS_DIR / "synthetic_quality.csv",
    }
    tabddpm_missing = [
        name for name, path in tabddpm_required.items() if not path.is_file()
    ]
    if tabddpm_missing:
        st.caption("TabDDPM 결과 파일이 없습니다: " + ", ".join(tabddpm_missing))
    else:
        tabddpm_metadata = json.loads(
            tabddpm_required["실행 메타데이터"].read_text(encoding="utf-8")
        )
        tabddpm_validation = pd.read_csv(tabddpm_required["validation 결과"])
        tabddpm_test = pd.read_csv(tabddpm_required["test 결과"])
        tabddpm_quality = pd.read_csv(tabddpm_required["합성 품질"]).iloc[0]

        selected_mask = (
            (tabddpm_test["model"] == tabddpm_metadata["selected_model"])
            & (tabddpm_test["strategy"] == tabddpm_metadata["selected_strategy"])
        )
        baseline_mask = (
            (tabddpm_test["model"] == "CatBoost")
            & (tabddpm_test["strategy"] == "scale_pos_weight")
        )
        selected_test = tabddpm_test.loc[selected_mask].iloc[0]
        baseline_test = tabddpm_test.loc[baseline_mask].iloc[0]

        tabddpm_metrics = st.columns(5)
        tabddpm_metrics[0].metric(
            "Test PR-AUC",
            f"{selected_test['pr_auc']:.4f}",
            delta=f"{selected_test['pr_auc'] - baseline_test['pr_auc']:+.4f}",
            help="delta는 CatBoost scale_pos_weight 기준선과의 절대 차이입니다.",
        )
        tabddpm_metrics[1].metric(
            "Test F2",
            f"{selected_test['f2']:.4f}",
            delta=f"{selected_test['f2'] - baseline_test['f2']:+.4f}",
        )
        tabddpm_metrics[2].metric(
            "불량 재현율", f"{selected_test['recall']:.2%}"
        )
        tabddpm_metrics[3].metric(
            "합성 불량", f"{int(tabddpm_quality['synthetic_defect_rows']):,}행"
        )
        tabddpm_metrics[4].metric(
            "합성 판별 AUC",
            f"{tabddpm_quality['discriminator_auc']:.4f}",
            help="0.5에 가까울수록 실제와 합성을 구별하기 어렵습니다.",
        )

        st.caption(
            f"{tabddpm_metadata['gpu']} · {tabddpm_metadata['train_steps']:,} step · "
            f"{tabddpm_metadata['training_seconds'] / 60:.1f}분 · "
            f"선택: {tabddpm_metadata['selected_model']} / "
            f"{tabddpm_metadata['selected_strategy']} · "
            "임계값은 validation F2로 선택"
        )

        validation_display = tabddpm_validation.copy()
        validation_display["전략"] = validation_display["strategy"].replace(
            {
                "scale_pos_weight": "가중치 기준선",
                "TabDDPM_0.25": "TabDDPM 0.25",
                "TabDDPM_0.50": "TabDDPM 0.50",
                "TabDDPM_0.75": "TabDDPM 0.75",
                "TabDDPM_1.00": "TabDDPM 1.00",
            }
        )
        validation_chart = px.bar(
            validation_display,
            x="전략",
            y="pr_auc",
            color="model",
            barmode="group",
            labels={"pr_auc": "Validation PR-AUC", "model": "모델"},
            title="증강 비율별 실제 validation PR-AUC",
        )
        validation_chart.update_layout(yaxis_range=[0, None])
        st.plotly_chart(validation_chart, width="stretch")

        test_display = tabddpm_test[
            ["model", "strategy", "pr_auc", "precision", "recall", "f1", "f2", "threshold"]
        ].rename(
            columns={
                "model": "모델",
                "strategy": "전략",
                "pr_auc": "PR-AUC",
                "precision": "정밀도",
                "recall": "재현율",
                "f1": "F1",
                "f2": "F2",
                "threshold": "임계값",
            }
        )
        st.markdown("**선택 고정 후 실제 test 비교**")
        st.dataframe(
            test_display,
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="%.4f")
                for column in ("PR-AUC", "정밀도", "재현율", "F1", "F2", "임계값")
            },
        )

        if float(tabddpm_quality["discriminator_auc"]) >= 0.8:
            st.error(
                "합성 판별기 AUC가 "
                f"{tabddpm_quality['discriminator_auc']:.4f}로 실제 불량과 합성 불량이 "
                "쉽게 구분됩니다. 성능 개선 비교에는 사용할 수 있지만 합성 데이터를 "
                "실제 데이터처럼 해석하거나 실제 표본 수 증가로 표현하면 안 됩니다."
            )
        else:
            st.success(
                "합성 판별기 기준으로 실제·합성 데이터의 구분 가능성이 낮습니다."
            )
        st.caption(
            f"평균 KS 유사도 {tabddpm_quality['ks_similarity_mean']:.4f} · "
            f"상관구조 유사도 {tabddpm_quality['correlation_similarity']:.4f} · "
            f"완전 복제율 {tabddpm_quality['exact_copy_rate']:.2%}. "
            "이번 결과는 한 번의 고정 내부 분할이며 외부 공정 일반화를 보장하지 않습니다."
        )

def render_secom_validation() -> None:
    st.warning(
        "성능 해석 전제: 센서 익명화, 약 14:1의 정상:불량 클래스 불균형, 다수 결측치, "
        "오래된 공개 데이터라는 한계가 모델 간 성능 편차와 재현율 제약에 큰 "
        "영향을 줍니다. 이 화면은 후보 모델 우열보다 한계 아래의 안정성을 검토합니다."
    )
    st.subheader("반복 교차검증 결과")
    operating = pd.read_csv(MODEL_DIR / "oof_operating_points.csv")
    balanced = operating.loc[
        operating["operating_point"] == "balanced_f2",
        [
            "model",
            "pr_auc",
            "precision",
            "recall",
            "f2",
            "false_alarms_per_100_normal",
        ],
    ].rename(
        columns={
            "model": "모델",
            "pr_auc": "PR-AUC",
            "precision": "정밀도",
            "recall": "재현율",
            "f2": "F2",
            "false_alarms_per_100_normal": "정상 100개당 오탐",
        }
    )
    st.dataframe(balanced, hide_index=True, width="stretch")
    st.image(
        str(MODEL_DIR / "oof_precision_recall_comparison.png"),
        caption="Repeated 5-fold × 3회 OOF precision-recall",
        width="stretch",
    )
    st.image(
        str(MODEL_DIR / "recall_false_alarm_tradeoff.png"),
        caption="목표 재현율에 따른 오탐 부담",
        width="stretch",
    )
    st.info(
        "고정 test는 앞선 실험에서 이미 관찰했습니다. 최종 성능 평가는 신규 시간대나 "
        "신규 lot의 외부 데이터로 다시 수행해야 합니다."
    )

    st.divider()
    st.subheader("센서 오류 강건성 스트레스 테스트")
    robustness_summary_path = ROBUSTNESS_DIR / "robustness_summary.json"
    robustness_table_path = ROBUSTNESS_DIR / "stress_test_summary.csv"
    if robustness_summary_path.is_file() and robustness_table_path.is_file():
        robustness = json.loads(
            robustness_summary_path.read_text(encoding="utf-8")
        )
        robustness_table = pd.read_csv(robustness_table_path)
        robustness_metrics = st.columns(4)
        robustness_metrics[0].metric(
            "Guardrail 통과",
            f"{robustness['guardrail_pass_count']}/{robustness['scenario_count']}",
        )
        robustness_metrics[1].metric(
            "원본 재현율", f"{robustness['baseline']['recall']:.1%}"
        )
        robustness_metrics[2].metric(
            "최저 재현율", f"{robustness['worst_recall_mean']:.1%}"
        )
        robustness_metrics[3].metric(
            "원본 OOD 감지",
            f"{robustness['baseline']['ood_review_rate']:.1%}",
        )
        dashboard_path = ROBUSTNESS_DIR / "robustness_dashboard.png"
        if dashboard_path.is_file():
            st.image(
                str(dashboard_path),
                caption="파란색은 사전 guardrail 통과, 빨간색은 실패 조건",
                width="stretch",
            )
        robustness_display = robustness_table[
            [
                "scenario",
                "recall_mean",
                "prediction_flip_rate_mean",
                "ood_review_rate_mean",
                "safety_false_negative_capture_mean",
                "safety_auto_accuracy_mean",
                "guardrail_pass",
            ]
        ].rename(
            columns={
                "scenario": "교란 조건",
                "recall_mean": "재현율",
                "prediction_flip_rate_mean": "판정 뒤집힘",
                "ood_review_rate_mean": "OOD 검토·보류",
                "safety_false_negative_capture_mean": "안전우선 미탐 포착",
                "safety_auto_accuracy_mean": "안전우선 자동 정확도",
                "guardrail_pass": "Guardrail 통과",
            }
        )
        st.dataframe(
            robustness_display,
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="percent")
                for column in (
                    "재현율",
                    "판정 뒤집힘",
                    "OOD 검토·보류",
                    "안전우선 미탐 포착",
                    "안전우선 자동 정확도",
                )
            },
        )
        if robustness["guardrail_fail_count"]:
            st.error(
                "취약 조건: "
                + ", ".join(robustness["failed_scenarios"])
                + ". 강한 연속형 센서 노이즈에서는 자동판정을 중단하고 측정계·센서 "
                "상태를 우선 확인해야 합니다."
            )
        containment_path = (
            SENSOR_FAILURE_CONTAINMENT_DIR / "containment_report.json"
        )
        if containment_path.is_file():
            containment = json.loads(containment_path.read_text(encoding="utf-8"))
            st.markdown("#### 센서 실패 안전격리")
            containment_cols = st.columns(4)
            containment_cols[0].metric("격리 판정", containment["status"])
            containment_cols[1].metric(
                "성능 실패 조건", containment["failed_scenario_count"]
            )
            containment_cols[2].metric(
                "격리된 실패 조건",
                sum(
                    item["contained"]
                    for item in containment["failed_scenario_containment"]
                ),
            )
            containment_cols[3].metric(
                "깨끗한 배치", containment["clean_batch_status"]
            )
            if containment["status"] == "PASS":
                st.success(
                    "모델 성능이 실패한 강한 노이즈 조건은 모든 반복에서 배치 STOP과 "
                    "자동판정 금지로 격리됐습니다."
                )
            st.warning(
                "격리 성공은 모델 재현율 개선을 뜻하지 않습니다. 실제 장비 고장과 "
                "신규 lot에서 탐지율·대응시간을 전향 검증해야 합니다."
            )
            st.download_button(
                "센서 실패 안전격리 JSON",
                containment_path.read_bytes(),
                "secom_sensor_failure_containment.json",
                "application/json",
                key="secom_sensor_failure_containment_download",
            )
        st.caption(
            "결측·열 드롭아웃·노이즈·스파이크를 조건당 5회 주입했습니다. 교란 "
            "크기는 fixed train의 IQR로만 정했고 test 결과로 모델이나 임계값을 "
            "조정하지 않았습니다. 합성 교란은 실제 장비 고장 분포를 대체하지 않습니다."
        )
    else:
        st.caption(
            "robustness_results가 없습니다. evaluate_sensor_robustness.py를 실행하세요."
        )

    st.divider()
    st.subheader("시간순 검증 및 확률 보정")
    temporal_dir = ARTIFACTS_DIR / "temporal_results"
    if temporal_dir.exists():
        future_metrics = pd.read_csv(temporal_dir / "future_metrics.csv")
        temporal_selected = future_metrics.loc[
            future_metrics["selected"],
            [
                "model",
                "method",
                "pr_auc",
                "brier",
                "mean_predicted_probability",
                "observed_prevalence",
                "precision",
                "recall",
                "f2",
                "tp",
                "fp",
                "fn",
                "tn",
            ],
        ].rename(
            columns={
                "model": "모델",
                "method": "보정 방법",
                "pr_auc": "PR-AUC",
                "brier": "Brier",
                "mean_predicted_probability": "평균 예측확률",
                "observed_prevalence": "실제 불량률",
                "precision": "정밀도",
                "recall": "재현율",
                "f2": "F2",
            }
        )
        st.dataframe(temporal_selected, hide_index=True, width="stretch")
        st.image(
            str(temporal_dir / "temporal_defect_prevalence.png"),
            caption="시간에 따른 불량률 변화와 60/20/20 경계",
            width="stretch",
        )
        st.image(
            str(temporal_dir / "future_reliability_diagram.png"),
            caption="마지막 20% 미래 구간의 확률 보정 전후",
            width="stretch",
        )
        st.error(
            "시간순 미래 구간에서는 PR-AUC와 정밀도가 크게 하락했습니다. "
            "현재 모델은 연구용 프로토타입이며 공정 배포 준비가 완료된 상태가 아닙니다."
        )
    else:
        st.caption("temporal_results 폴더가 없어 시간순 결과를 표시할 수 없습니다.")

    st.divider()
    st.subheader("데이터 드리프트 진단")
    drift_dir = ARTIFACTS_DIR / "drift_results"
    if drift_dir.exists():
        drift_summary = json.loads(
            (drift_dir / "drift_summary.json").read_text(encoding="utf-8")
        )
        drift_table = pd.read_csv(drift_dir / "feature_drift.csv")
        drift_metrics = st.columns(4)
        drift_metrics[0].metric(
            "기간 판별 ROC-AUC", f"{drift_summary['domain_classifier_auc']:.4f}"
        )
        drift_metrics[1].metric(
            "High drift", drift_summary["severity_counts"].get("high", 0)
        )
        drift_metrics[2].metric(
            "Moderate drift", drift_summary["severity_counts"].get("moderate", 0)
        )
        drift_metrics[3].metric(
            "KS FDR 유의", drift_summary["ks_significant_after_fdr"]
        )
        st.dataframe(
            drift_table[
                [
                    "feature",
                    "psi",
                    "severity",
                    "fit_missing_rate",
                    "future_missing_rate",
                    "catboost_future_shap_rank",
                    "xgboost_future_shap_rank",
                    "monitor_priority_score",
                ]
            ].head(20).rename(
                columns={
                    "feature": "변수",
                    "psi": "PSI",
                    "severity": "드리프트 수준",
                    "fit_missing_rate": "초기 결측률",
                    "future_missing_rate": "미래 결측률",
                    "catboost_future_shap_rank": "CatBoost SHAP 순위",
                    "xgboost_future_shap_rank": "XGBoost SHAP 순위",
                    "monitor_priority_score": "감시 우선순위 점수",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.image(
            str(drift_dir / "top_feature_psi.png"),
            caption="분포 이동이 큰 센서 변수",
            width="stretch",
        )
        st.image(
            str(drift_dir / "drift_shap_priority.png"),
            caption="드리프트와 미래 SHAP 중요도를 결합한 감시 우선순위",
            width="stretch",
        )
        st.warning(
            "기간 판별 AUC 0.9993은 초기와 미래 데이터의 센서 분포가 거의 완전히 "
            "구분된다는 뜻입니다. 모델 성능 저하의 핵심 위험 신호로 봐야 합니다."
        )
    else:
        st.caption("drift_results 폴더가 없어 드리프트 결과를 표시할 수 없습니다.")

    st.divider()
    st.subheader("안정 변수 재학습 실험")
    robust_dir = ARTIFACTS_DIR / "drift_robust_results"
    if robust_dir.exists():
        robust_future = pd.read_csv(robust_dir / "future_strategy_metrics.csv")
        robust_display = robust_future[
            [
                "model",
                "strategy",
                "candidate_feature_count",
                "pr_auc",
                "brier",
                "precision",
                "recall",
                "f2",
                "selected_on_calibration",
            ]
        ].rename(
            columns={
                "model": "모델",
                "strategy": "변수 전략",
                "candidate_feature_count": "후보 변수 수",
                "pr_auc": "미래 PR-AUC",
                "brier": "미래 Brier",
                "precision": "미래 정밀도",
                "recall": "미래 재현율",
                "f2": "미래 F2",
                "selected_on_calibration": "보정 구간 선택",
            }
        )
        st.dataframe(robust_display, hide_index=True, width="stretch")
        st.image(
            str(robust_dir / "strategy_pr_auc_comparison.png"),
            caption="보정 구간에서 좋아진 안정 변수 전략이 미래 구간에는 일반화되지 않음",
            width="stretch",
        )
        st.warning(
            "미래 데이터를 보지 않고 선택한 안정 변수 모델이 기준 모델보다 좋아지지 "
            "않았습니다. 따라서 현재 기본 진단 모델은 교체하지 않았습니다."
        )
    else:
        st.caption("drift_robust_results 폴더가 없어 재학습 결과를 표시할 수 없습니다.")

    st.divider()
    st.subheader("Rolling-window walk-forward 실험")
    rolling_dir = ARTIFACTS_DIR / "rolling_results"
    if rolling_dir.exists():
        rolling_selection = pd.read_csv(
            rolling_dir / "strategy_selection_summary.csv"
        )
        rolling_future = pd.read_csv(rolling_dir / "future_evaluation.csv")
        st.markdown("**미래 데이터를 제외한 4개 순차 검증 fold의 전략 선택**")
        st.dataframe(
            rolling_selection.rename(
                columns={
                    "model": "모델",
                    "strategy": "학습 전략",
                    "mean_pr_auc": "평균 PR-AUC",
                    "std_pr_auc": "PR-AUC 표준편차",
                    "mean_roc_auc": "평균 ROC-AUC",
                    "selected": "선택",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**마지막 20% 미래 평가**")
        st.dataframe(
            rolling_future[
                [
                    "model",
                    "role",
                    "strategy",
                    "pr_auc",
                    "roc_auc",
                    "brier",
                    "precision",
                    "recall",
                    "f2",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                ]
            ].rename(
                columns={
                    "model": "모델",
                    "role": "역할",
                    "strategy": "학습 전략",
                    "pr_auc": "PR-AUC",
                    "roc_auc": "ROC-AUC",
                    "brier": "Brier",
                    "precision": "정밀도",
                    "recall": "재현율",
                    "f2": "F2",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.image(
            str(rolling_dir / "walk_forward_strategy_selection.png"),
            caption="네 개 순차 검증 fold에서 누적 학습이 선택됨",
            width="stretch",
        )
        st.error(
            "최근 300·500개 및 시간가중 학습은 누적 학습보다 안정적이지 않았습니다. "
            "미래 ROC-AUC도 0.5 미만으로 개념 드리프트 가능성이 있어 모델을 교체하지 "
            "않았습니다."
        )
    else:
        st.caption("rolling_results 폴더가 없어 walk-forward 결과를 표시할 수 없습니다.")

    render_secom_independent_validation(SECOM_INDEPENDENT_VALIDATION_DIR)

XAI_SCOPE_NOTE = (
    "SECOM은 트리 변수별 TreeSHAP, WM-811K는 같은 웨이퍼 형상의 "
    "'모든 die 정상' 기준과 비교한 Gradient SHAP으로 기여를 표현합니다. "
    "두 모듈 모두 기여도를 원인 확률이 아닌 모델 판단 근거로 한정합니다."
)
# Detailed WM-811K validation screens stay on the WM page (login required when
# SHAPGPT_REQUIRE_LOGIN=1); the project page links to them.
WM_VALIDATION_SECTIONS = (
    ("핵심 성능", "evaluation"),
    ("안전·OOD", "evaluation_safety"),
    ("SHAP 검증", "evaluation_xai"),
    ("독립 검증", "independent_validation"),
)


def render_project_xai_evidence() -> None:
    """SHAP/XAI validation evidence of both modules for the project page."""
    st.info(XAI_SCOPE_NOTE)
    render_secom_shap_reliability()
    st.divider()
    st.subheader("WM-811K 설명 검증")
    render_gradient_shap_validation()
    st.page_link(
        wm_page,
        label="WM-811K SHAP 검증 전체 보기 · 과거 IG·Grad-CAM 비교와 오분류 구조",
        icon=":material/arrow_forward:",
        query_params={"module": "wm811k", "section": "evaluation_xai"},
    )


def render_project_wm_evidence() -> None:
    """WM-811K validation summary for the project page, from saved results."""
    site.render_metric_cards(
        site.wm_metric_cards(collect_site_evidence()), key="site_metric_wm"
    )
    try:
        class_report = load_wafer_evaluation(
            str(WM_RESULTS_DIR / "run_summary.json"),
            str(WM_RESULTS_DIR / "test_confusion_matrix.csv"),
            str(WM_RESULTS_DIR / "test_classification_report.csv"),
        )[2]
    except (OSError, KeyError, ValueError):
        class_report = None
    if class_report is not None:
        with st.expander("클래스별 성능과 혼동행렬"):
            render_wm_class_performance(class_report)
    st.markdown("##### WM-811K 진단 페이지의 상세 검증 화면")
    link_columns = st.columns(len(WM_VALIDATION_SECTIONS))
    for column, (label, section) in zip(link_columns, WM_VALIDATION_SECTIONS):
        column.page_link(
            wm_page,
            label=label,
            icon=":material/arrow_forward:",
            query_params={"module": "wm811k", "section": section},
        )
    st.caption("로그인 필수 모드에서는 상세 화면이 로그인 후 열립니다.")


def render_wm_class_performance(class_report: pd.DataFrame) -> None:
    """Test confusion matrix image and per-class report, read from saved results."""
    detail_columns = st.columns((1.05, 1))
    with detail_columns[0]:
        st.image(
            str(WM_RESULTS_DIR / "test_confusion_matrix.png"),
            caption="Lot 비중복 고정 Test 혼동행렬",
            width="stretch",
        )
    with detail_columns[1]:
        display_report = class_report.rename(
            columns={
                "class": "클래스",
                "precision": "정밀도",
                "recall": "재현율",
                "f1-score": "F1",
                "support": "표본 수",
            }
        )
        st.dataframe(
            display_report,
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(format="%.3f")
                for name in ("정밀도", "재현율", "F1")
            }
            | {"표본 수": st.column_config.NumberColumn(format="%d")},
        )


def render_wm811k_dashboard() -> None:
    render_dashboard_header(
        "WM-811K 웨이퍼 맵 SHAP 진단",
        "Lot 단위로 분리한 실제 test 24,705개 성능과 CNN 예측을 확인합니다. "
        "Gradient SHAP은 결함 die의 위치별 기여를, Grad-CAM은 CNN의 주의 영역을 표시합니다.",
    )
    render_unified_xai_frame(
        explanation="Gradient SHAP 위치 기여 + Grad-CAM 대조",
        input_limit="형상·OOD·전처리 일치 확인",
    )

    required = {
        "체크포인트": WM_CHECKPOINT_PATH,
        "학습 요약": WM_RESULTS_DIR / "run_summary.json",
        "혼동행렬": WM_RESULTS_DIR / "test_confusion_matrix.csv",
        "클래스 보고서": WM_RESULTS_DIR / "test_classification_report.csv",
        "선택 기록": WM_RESULTS_DIR / "selection_protocol.json",
        "불확실성 정책": WM_RESULTS_DIR / "uncertainty_policy.json",
        "오분류 분석": WM_RESULTS_DIR / "error_analysis.json",
        "실제 test 예시": WM_DEMO_DIR / "manifest.json",
        "OOD 기준 정보": WM_OOD_DIR / "ood_reference.json",
        "OOD 기준 배열": WM_OOD_DIR / "ood_reference.npz",
        "OOD 평가 요약": WM_OOD_DIR / "ood_summary.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        st.error("WM-811K 결과 파일이 없습니다: " + ", ".join(missing))
        return

    summary, binary, class_report = load_wafer_evaluation(
        str(WM_RESULTS_DIR / "run_summary.json"),
        str(WM_RESULTS_DIR / "test_confusion_matrix.csv"),
        str(WM_RESULTS_DIR / "test_classification_report.csv"),
    )
    uncertainty_policy = json.loads(
        (WM_RESULTS_DIR / "uncertainty_policy.json").read_text(encoding="utf-8")
    )
    error_analysis = json.loads(
        (WM_RESULTS_DIR / "error_analysis.json").read_text(encoding="utf-8")
    )
    ood_summary = json.loads(
        (WM_OOD_DIR / "ood_summary.json").read_text(encoding="utf-8")
    )
    ood_metadata_path = WM_OOD_DIR / "ood_reference.json"
    ood_arrays_path = WM_OOD_DIR / "ood_reference.npz"
    ood_reference = load_wafer_ood_reference(
        str(ood_metadata_path),
        str(ood_arrays_path),
        ood_metadata_path.stat().st_mtime_ns,
        ood_arrays_path.stat().st_mtime_ns,
    )
    monitoring_reference_path = WM_MONITORING_DIR / "monitoring_reference.json"
    monitoring_reference = (
        load_monitoring_reference(monitoring_reference_path)
        if monitoring_reference_path.is_file()
        else None
    )
    artifact_hashes = {
        "CNN best_model.pt": load_file_sha256(
            str(WM_CHECKPOINT_PATH), WM_CHECKPOINT_PATH.stat().st_mtime_ns
        ),
        "OOD reference JSON": load_file_sha256(
            str(ood_metadata_path), ood_metadata_path.stat().st_mtime_ns
        ),
        "OOD reference NPZ": load_file_sha256(
            str(ood_arrays_path), ood_arrays_path.stat().st_mtime_ns
        ),
    }
    if monitoring_reference_path.is_file():
        artifact_hashes["Batch monitoring reference"] = load_file_sha256(
            str(monitoring_reference_path),
            monitoring_reference_path.stat().st_mtime_ns,
        )
        if (
            monitoring_reference["model_checkpoint_sha256"]
            != artifact_hashes["CNN best_model.pt"]
        ):
            st.error("배치 모니터링 기준과 현재 운영 체크포인트의 SHA-256이 다릅니다.")
            return
    wm_navigation = {
            "분석": (
                ("웨이퍼 진단·SHAP", "diagnosis"),
                ("배치 진단", "batch"),
                ("입력 안내", "guide"),
            ),
            "성능·검증": (
                ("핵심 성능", "evaluation"),
                ("안전·OOD", "evaluation_safety"),
                ("SHAP 검증", "evaluation_xai"),
                ("독립 검증", "independent_validation"),
            ),
        }
    if ACCESS.is_admin:
        wm_navigation["관리자"] = (("전체 실험 기록", "research_log"),)
    elif query_value("section", "") in dashboard_access.WM_ADMIN_SECTIONS:
        st.warning("관리자 권한이 필요한 화면입니다. 기본 화면으로 이동했습니다.")
    wm_section = render_section_navigation(
        wm_navigation,
        module="wm811k",
        default_section="batch",
        key_prefix="wm_nav",
    )

    def render_wm_performance_summary() -> None:
        st.subheader("운영 성능 요약")
        policy_test = uncertainty_policy["test_evaluation"]
        defect_report = class_report.loc[class_report["class"] != "none"].copy()
        best_classes = defect_report.loc[defect_report["support"] >= 100].nlargest(
            2, "recall"
        )
        st.success(
            "현재 운영 모델은 CNN Champion입니다. 모델 변경 없이 낮은 신뢰도와 "
            "분포 이탈 입력을 전문가 검토로 전환합니다."
        )
        headline = st.columns(5)
        headline[0].metric("운영 모델", "CNN Champion")
        headline[1].metric("Macro-F1", f"{summary['test_metrics']['macro_f1']:.3f}")
        headline[2].metric("불량 탐지 재현율", f"{binary['recall']:.1%}")
        headline[3].metric("자동 분류", f"{policy_test['automatic_coverage']:.1%}")
        headline[4].metric("오류 검토 포착", f"{policy_test['error_capture_rate']:.1%}")

        st.markdown("#### 가장 안정적으로 구분한 클래스")
        st.caption("고정 Test 표본 100개 이상인 불량 클래스 중 재현율 상위 2개입니다.")
        best_columns = st.columns(2)
        for column, (_, row) in zip(best_columns, best_classes.iterrows()):
            with column:
                with st.container(border=True):
                    st.metric(str(row["class"]), f"재현율 {row['recall']:.1%}")
                    st.caption(
                        f"F1 {row['f1-score']:.3f} · Test {int(row['support']):,}개"
                    )
        st.info("클래스별 오류는 아래 혼동행렬과 전체 성능표에서 확인할 수 있습니다.")

        summary_columns = st.columns(4)
        summary_columns[0].metric("Test 정확도", f"{summary['test_metrics']['accuracy']:.2%}")
        summary_columns[1].metric(
            "균형 정확도", f"{summary['test_metrics']['balanced_accuracy']:.2%}"
        )
        summary_columns[2].metric("정상 오탐", f"{binary['false_positive']:,}개")
        summary_columns[3].metric("불량 미탐", f"{binary['false_negative']:,}개")

        with st.expander("전체 클래스 성능과 혼동행렬 · 오류 확인"):
            render_wm_class_performance(class_report)

        selection = json.loads(
            (WM_RESULTS_DIR / "selection_protocol.json").read_text(encoding="utf-8")
        )
        strategy_labels = {
            "ce_sqrt_balanced": "교차엔트로피 · 제곱근 균형",
            "weighted_ce_shuffle": "가중 교차엔트로피 · 셔플",
            "focal_shuffle": "Focal loss · 셔플",
        }
        with st.expander("학습 전략과 재현성 근거"):
            st.markdown(
                f"**선택 전략:** {strategy_labels.get(selection['selected_strategy'], selection['selected_strategy'])}"
            )
            comparison = pd.read_csv(
                WM_RESULTS_DIR / "validation_strategy_comparison.csv"
            )[[
                "strategy",
                "validation_macro_f1",
                "validation_balanced_accuracy",
                "scratch_recall",
                "best_epoch",
            ]].rename(
                columns={
                    "strategy": "전략",
                    "validation_macro_f1": "검증 Macro-F1",
                    "validation_balanced_accuracy": "검증 균형 정확도",
                    "scratch_recall": "Scratch 재현율",
                    "best_epoch": "최적 epoch",
                }
            )
            comparison["전략"] = comparison["전략"].map(strategy_labels).fillna(
                comparison["전략"]
            )
            st.dataframe(
                comparison,
                hide_index=True,
                width="stretch",
                column_config={
                    name: st.column_config.NumberColumn(format="%.3f")
                    for name in ("검증 Macro-F1", "검증 균형 정확도", "Scratch 재현율")
                },
            )
            multiseed_path = WM_MULTISEED_DIR / "reproducibility_summary.json"
            if multiseed_path.is_file():
                multiseed = json.loads(multiseed_path.read_text(encoding="utf-8"))
                stats = multiseed["metrics"]["test_macro_f1"]
                st.caption(
                    f"시드 {len(multiseed['seeds'])}개 고정 Test Macro-F1 "
                    f"{stats['mean']:.4f} ± {stats['std']:.4f} · Test로 시드를 재선택하지 않음"
                )

        st.caption(
            "`운영 모델` · `Lot 비중복 Test` · `사후 분석` · `자동 모델 승격 없음`"
        )
        st.info(
            "정확도는 다수 클래스 none의 영향을 크게 받습니다. Macro-F1·클래스별 "
            "재현율·검토 포착률을 함께 해석해야 하며 신규 장비 성능을 보장하지 않습니다."
        )

    def render_wm_safety_summary() -> None:
        st.subheader("안전성·OOD")
        policy_test = uncertainty_policy["test_evaluation"]
        ood_test = next(
            item for item in ood_summary["split_metrics"] if item["split"] == "test"
        )
        ood_stress = ood_summary["stress_metrics"]
        st.info(
            "낮은 신뢰도는 검토로, 강한 분포 이탈은 판정 보류로 전환합니다. "
            "현재 기준은 Validation에서 고정했습니다."
        )
        policy_columns = st.columns(4)
        policy_columns[0].metric("자동 분류", f"{policy_test['automatic_coverage']:.1%}")
        policy_columns[1].metric("자동 구간 정확도", f"{policy_test['automatic_accuracy']:.2%}")
        policy_columns[2].metric("오류 검토 포착", f"{policy_test['error_capture_rate']:.1%}")
        policy_columns[3].metric("검토 대상", f"{policy_test['review_count']:,}개")

        st.markdown("#### 입력 분포 이탈")
        ood_columns = st.columns(4)
        ood_columns[0].metric("분포 내 자동 처리", f"{ood_test['automatic_coverage']:.1%}")
        ood_columns[1].metric("경계 검토", f"{ood_test['review_rate']:.1%}")
        ood_columns[2].metric("OOD 보류", f"{ood_test['ood_rate']:.1%}")
        ood_columns[3].metric("공간 셔플 탐지", f"{ood_stress['ood_detection_rate']:.1%}")
        with st.expander("OOD 점수 분포와 고정 임계값"):
            ood_detail = st.columns((1.2, 1))
            with ood_detail[0]:
                st.image(
                    str(WM_OOD_DIR / "ood_score_distribution.png"),
                    caption="Validation에서 고정한 특징 공간 OOD 경계",
                    width="stretch",
                )
            with ood_detail[1]:
                st.metric("경계 검토 점수", f"{ood_reference['thresholds']['review_threshold']:.4f}")
                st.metric("판정 보류 점수", f"{ood_reference['thresholds']['ood_threshold']:.4f}")
                st.caption("CNN pooled feature 256차원 · Test 임계값 선택 미사용")

        lot_summary_path = WM_LOT_AUDIT_DIR / "lot_generalization_summary.json"
        if lot_summary_path.is_file():
            st.markdown("#### Held-out lot")
            lot_summary = json.loads(lot_summary_path.read_text(encoding="utf-8"))
            lot_columns = st.columns(3)
            lot_columns[0].metric("고정 Test lot", f"{lot_summary['lot_count']:,}개")
            lot_columns[1].metric("전체 정확도", f"{lot_summary['global_accuracy']:.2%}")
            lot_columns[2].metric(
                "Lot bootstrap 95% 구간",
                f"{lot_summary['cluster_bootstrap_accuracy_ci95'][0]:.2%}~{lot_summary['cluster_bootstrap_accuracy_ci95'][1]:.2%}",
            )
            with st.expander("Lot·웨이퍼 크기 상세"):
                st.image(
                    str(WM_LOT_AUDIT_DIR / "lot_generalization_dashboard.png"),
                    caption="Lot·웨이퍼 크기·클래스별 사후 진단",
                    width="stretch",
                )

        robustness_path = WM_ROBUSTNESS_DIR / "robustness_summary.json"
        robustness_table_path = WM_ROBUSTNESS_DIR / "stress_test_summary.csv"
        if robustness_path.is_file() and robustness_table_path.is_file():
            st.markdown("#### 입력 오류 스트레스")
            robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
            robustness_table = pd.read_csv(robustness_table_path)
            robustness_columns = st.columns(3)
            robustness_columns[0].metric("평가 입력", f"{robustness['evaluated_rows']:,}개")
            robustness_columns[1].metric(
                "안전 기준 통과",
                f"{robustness['guardrail_pass_count']}/{robustness['scenario_count']}",
            )
            robustness_columns[2].metric(
                "최저 예측 유지율", f"{robustness_table['prediction_stability'].min():.1%}"
            )
            compact = robustness_table[[
                "scenario_label",
                "prediction_stability",
                "changed_prediction_capture",
                "guardrail_pass",
            ]].rename(
                columns={
                    "scenario_label": "교란 조건",
                    "prediction_stability": "예측 유지율",
                    "changed_prediction_capture": "변경 포착률",
                    "guardrail_pass": "안전 기준",
                }
            )
            compact["안전 기준"] = compact["안전 기준"].map(
                {True: "통과", False: "미통과"}
            )
            st.dataframe(
                compact,
                hide_index=True,
                width="stretch",
                column_config={
                    "예측 유지율": st.column_config.NumberColumn(format="percent"),
                    "변경 포착률": st.column_config.NumberColumn(format="percent"),
                },
            )
            failed = robustness_table.loc[
                ~robustness_table["guardrail_pass"], "scenario_label"
            ].tolist()
            if failed:
                st.error("자동 확정 금지 조건: " + ", ".join(failed))

        st.caption("`Validation 고정` · `Test 임계값 미사용` · `실제 외부 장비 검증 아님`")
        st.warning(
            "공간 셔플과 합성 교란은 안전장치 점검용입니다. 신규 장비·공정에서는 "
            "자동판정을 중단하고 별도 외부 검증이 필요합니다."
        )

    def render_wm_xai_summary() -> None:
        st.subheader("SHAP 설명 검증")
        st.info(XAI_SCOPE_NOTE)
        render_gradient_shap_validation()

        gradcam_path = WM_GRADCAM_AUDIT_DIR / "gradcam_faithfulness_summary.json"
        xai_path = WM_XAI_COMPARISON_DIR / "xai_method_summary.json"
        if not gradcam_path.is_file() or not xai_path.is_file():
            st.info("과거 설명법 비교 결과가 없습니다.")
            return
        gradcam = json.loads(gradcam_path.read_text(encoding="utf-8"))
        xai = json.loads(xai_path.read_text(encoding="utf-8"))
        matches_deployment = wm_xai_matches_deployment()
        paired_interval = xai["paired_ig_minus_gradcam_95ci"]
        with st.expander("과거 IG·Grad-CAM 방법 비교 증거 · Gradient SHAP 검증 아님"):
            st.info(
                "아래 수치는 0 기준 Integrated Gradients와 Grad-CAM을 비교한 과거 결과입니다. "
                "현재 단일 진단에 표시하는 Gradient SHAP의 검증 결과가 아닙니다."
                + (
                    " 비교 자체는 현재 배포 체크포인트로 수행했습니다."
                    if matches_deployment
                    else " 비교는 현재 배포 체크포인트와 다른 탐색 체크포인트 결과입니다."
                )
            )
            xai_columns = st.columns(4)
            xai_columns[0].metric(
                "Grad-CAM 상위−무작위",
                f"{xai['methods']['Grad-CAM']['macro_top_vs_random_advantage']:+.4f}",
            )
            xai_columns[1].metric(
                "IG 상위−무작위",
                f"{xai['methods']['Integrated Gradients']['macro_top_vs_random_advantage']:+.4f}",
            )
            xai_columns[2].metric(
                "IG−Grad-CAM", f"{xai['paired_ig_minus_gradcam']:+.4f}"
            )
            xai_columns[3].metric(
                "Grad-CAM 상위 > 무작위",
                f"{gradcam['top_drop_exceeds_random_rate']:.1%}",
            )
            st.image(
                str(WM_XAI_COMPARISON_DIR / "xai_method_comparison_dashboard.png"),
                caption="과거 IG·Grad-CAM 비교 · 고정 Test 균형 표본의 설명 충실도",
                width="stretch",
            )
            st.caption(
                f"IG−Grad-CAM 대응 bootstrap 95% 구간 "
                f"{paired_interval[0]:+.4f}~{paired_interval[1]:+.4f}"
            )

        st.markdown("#### 오분류 구조")
        st.caption(
            "특정 취약 클래스만 강조하지 않고, 전체 혼동 쌍을 건수와 "
            "검토 포착률로 함께 표시합니다."
        )
        confusion_pairs = pd.DataFrame(error_analysis["confusion_pairs"]).head(5)
        compact_pairs = confusion_pairs[[
            "true_class", "predicted_class", "count", "review_capture_rate"
        ]].rename(
            columns={
                "true_class": "실제",
                "predicted_class": "예측",
                "count": "건수",
                "review_capture_rate": "검토 포착률",
            }
        )
        st.dataframe(
            compact_pairs,
            hide_index=True,
            width="stretch",
            column_config={
                "건수": st.column_config.NumberColumn(format="%d"),
                "검토 포착률": st.column_config.NumberColumn(format="percent"),
            },
        )
        near_full_path = WM_NEAR_FULL_SHAPE_DIR / "near_full_global_shape_summary.json"
        if near_full_path.is_file():
            with st.expander("Near-full 전역 형태 근거"):
                near_full = json.loads(near_full_path.read_text(encoding="utf-8"))
                shape = near_full["metrics"]
                shape_columns = st.columns(4)
                shape_columns[0].metric("결함 die", f"{shape['defect_ratio']['mean']:.1%}")
                shape_columns[1].metric(
                    "최대 연결 영역", f"{shape['largest_component_fraction']['mean']:.1%}"
                )
                shape_columns[2].metric(
                    "경계 결함률", f"{shape['boundary_defect_coverage']['mean']:.1%}"
                )
                shape_columns[3].metric(
                    "결함 분포 범위", f"{shape['defect_bbox_fraction']['mean']:.1%}"
                )
                st.image(
                    str(WM_NEAR_FULL_SHAPE_DIR / "near_full_global_shape_dashboard.png"),
                    caption="고정 Test Near-full 21개의 전역 형태",
                    width="stretch",
                )
        st.caption("`사후 분석` · `물리적 원인 아님` · `모델 선택 미사용`")
        st.warning(
            "설명 지도는 모델의 위치 민감도입니다. 실제 원인이나 공정 조정 지시로 "
            "해석하지 말고 장비·레시피·센서 기록과 함께 검토해야 합니다."
        )

    def render_wm_independent_page() -> None:
        render_wm_independent_validation(
            WM_INDEPENDENT_VALIDATION_DIR, standalone=True
        )

    def render_wm_research_log() -> None:
        st.subheader("전체 실험 기록")
        st.caption(
            "운영 화면에서 생략한 후보 실험과 사후 분석을 보존한 감사용 상세 기록입니다."
        )
        registry_path = WM_MODEL_REGISTRY_DIR / "champion_manifest.json"
        registry_validation_path = WM_MODEL_REGISTRY_DIR / "registry_validation.json"
        if registry_path.is_file() and registry_validation_path.is_file():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry_validation = json.loads(
                registry_validation_path.read_text(encoding="utf-8")
            )
            deployment = registry["deployment"]
            challenger_gate = registry["latest_challenger_gate"]
            registry_columns = st.columns(4)
            registry_columns[0].metric("운영 모델 상태", "Champion 고정")
            registry_columns[1].metric(
                "학습 전략", deployment["strategy"].replace("_", " · ")
            )
            registry_columns[2].metric("고정 seed", deployment["seed"])
            registry_columns[3].metric(
                "최신 Challenger", "승격 거부"
                if not challenger_gate["eligible"] else "승격 검토"
            )
            st.success(
                "실제 추론은 기존 ce_sqrt_balanced seed 42 champion과 이에 연결된 "
                "불확실성·OOD 정책을 계속 사용합니다."
            )
            st.caption(
                f"레지스트리 검증: {registry_validation['status']} · "
                f"체크포인트 SHA-256: {deployment['checkpoint_sha256']} · "
                "자동 모델 승격 없음"
            )
            with st.expander("Champion·Challenger 결정 기록"):
                registry_decisions = pd.DataFrame(registry["candidate_decisions"])
                st.dataframe(
                    registry_decisions.rename(columns={
                        "candidate": "모델/후보", "role": "역할",
                        "status": "결정", "selection_scope": "평가 범위",
                        "reason": "근거",
                    }),
                    hide_index=True,
                    width="stretch",
                )
                st.warning(
                    "최신 baseline·robust 앙상블 후보는 validation 견고성 기준을 "
                    "통과했지만, 고정 test에서 정상 재현율 보호 기준을 넘어서 "
                    "승격하지 않았습니다. Test 결과로 재튜닝하지 않았으며 운영 "
                    "체크포인트는 기존 champion입니다. 이 후보는 "
                    "운영 체크포인트가 아닙니다."
                )
            st.divider()

        ssl_validation_path = WM_SSL_VALIDATION_DIR / "bundle_validation.json"
        if ssl_validation_path.is_file():
            ssl_result = json.loads(
                ssl_validation_path.read_text(encoding="utf-8")
            )
            with st.expander(
                "비라벨 자기지도 사전학습 후보 · validation 3시드 검증",
                expanded=False,
            ):
                ssl_columns = st.columns(4)
                ssl_columns[0].metric(
                    "Macro-F1",
                    f"{ssl_result['candidate_validation_metrics']['macro_f1']['mean']:.4f}",
                    delta=f"{ssl_result['metric_deltas']['macro_f1']:+.4f}",
                )
                ssl_columns[1].metric(
                    "균형 정확도",
                    f"{ssl_result['candidate_validation_metrics']['balanced_accuracy']['mean']:.4f}",
                    delta=f"{ssl_result['metric_deltas']['balanced_accuracy']:+.4f}",
                )
                ssl_columns[2].metric(
                    "정상(none) 재현율",
                    f"{ssl_result['candidate_validation_class_recall']['none']['mean']:.4f}",
                    delta=f"{ssl_result['metric_deltas']['none_recall']:+.4f}",
                    delta_color="normal",
                )
                ssl_columns[3].metric(
                    "검증한 예측",
                    f"{ssl_result['validation_prediction_rows_checked']:,}행",
                )
                ssl_rows = []
                for class_name in ("Edge-Loc", "Loc", "Scratch"):
                    ssl_rows.append(
                        {
                            "클래스": class_name,
                            "기존 recall": ssl_result[
                                "baseline_validation_class_recall"
                            ][class_name],
                            "자기지도 recall": ssl_result[
                                "candidate_validation_class_recall"
                            ][class_name]["mean"],
                            "변화": ssl_result["metric_deltas"][
                                f"{class_name}_recall"
                            ],
                        }
                    )
                st.dataframe(
                    pd.DataFrame(ssl_rows).style.format(
                        {
                            "기존 recall": "{:.4f}",
                            "자기지도 recall": "{:.4f}",
                            "변화": "{:+.4f}",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.warning(
                    "자기지도 후보는 Macro-F1·균형 정확도·취약 3종 recall을 "
                    "개선했지만 정상 recall 감소폭이 허용치 0.2%p를 넘어 "
                    "승격하지 않았습니다. 기존 champion을 유지하며 고정 test는 "
                    "다시 열지 않았습니다."
                )
                st.caption(
                    "비라벨 175,039개(라벨 train 포함) · train lot만 사용 · "
                    "seed 17/42/2026 · validation-only · 자동 배포 없음"
                )

        ssl_oof_path = WM_SSL_OOF_CALIBRATION_DIR / "bundle_validation.json"
        if ssl_oof_path.is_file():
            ssl_oof = json.loads(ssl_oof_path.read_text(encoding="utf-8"))
            with st.expander(
                "자기지도 후보 · train OOF 정상 클래스 보정",
                expanded=False,
            ):
                maximum_none = ssl_oof["maximum_none_recall_candidate"]
                oof_columns = st.columns(4)
                oof_columns[0].metric(
                    "OOF none recall",
                    f"{maximum_none['none_recall']:.4f}",
                    delta=f"{maximum_none['delta_none_recall']:+.4f}",
                )
                oof_columns[1].metric(
                    "OOF 약한 결함 recall",
                    f"{maximum_none['weak_recall']:.4f}",
                    delta=f"{maximum_none['delta_weak_recall']:+.4f}",
                    delta_color="normal",
                )
                oof_columns[2].metric(
                    "OOF Macro-F1",
                    f"{maximum_none['macro_f1']:.4f}",
                    delta=f"{maximum_none['delta_macro_f1']:+.4f}",
                )
                oof_columns[3].metric(
                    "선택된 bias",
                    "없음",
                )
                st.dataframe(
                    pd.DataFrame(ssl_oof["fold_metrics"])[
                        [
                            "fold",
                            "holdout_rows",
                            "holdout_lots",
                            "holdout_balanced_accuracy",
                            "holdout_macro_f1",
                        ]
                    ].rename(
                        columns={
                            "fold": "fold",
                            "holdout_rows": "OOF 행",
                            "holdout_lots": "OOF lot",
                            "holdout_balanced_accuracy": "균형 정확도",
                            "holdout_macro_f1": "Macro-F1",
                        }
                    ).style.format(
                        {"균형 정확도": "{:.4f}", "Macro-F1": "{:.4f}"}
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.warning(
                    f"train OOF에서 none recall 목표 "
                    f"{ssl_oof['target_none_recall']:.4f}를 만족하면서 약한 "
                    f"결함 recall 감소를 "
                    f"{ssl_oof['weak_recall_tolerance']:.2%}p 이내로 제한하는 "
                    f"bias가 없었습니다. 최대 none recall도 "
                    f"{maximum_none['none_recall']:.4f}였고 약한 결함 recall은 "
                    f"{abs(maximum_none['delta_weak_recall']):.2%}p 감소했습니다. "
                    "기준을 사후 완화하지 않고 기존 champion을 유지했습니다."
                )
                st.caption(
                    f"train OOF {ssl_oof['oof_rows_checked']:,}행 · "
                    f"{ssl_oof['oof_lots_checked']:,} lot · "
                    f"{ssl_oof['oof_folds']}-fold · "
                    "validation 미사용 · 고정 test 미사용 · 자동 배포 없음"
                )

        metrics = st.columns(5)
        metrics[0].metric("Test 정확도", f"{summary['test_metrics']['accuracy']:.2%}")
        metrics[1].metric(
            "균형 정확도", f"{summary['test_metrics']['balanced_accuracy']:.2%}"
        )
        metrics[2].metric("Macro-F1", f"{summary['test_metrics']['macro_f1']:.3f}")
        metrics[3].metric("불량 탐지 재현율", f"{binary['recall']:.2%}")
        metrics[4].metric("불량 탐지 F1", f"{binary['f1']:.3f}")

        st.caption(
            f"T4 GPU · {summary['loss']} · {summary['sampling']} · "
            f"최적 {summary['best_epoch']} epoch · "
            f"총 {summary['epochs_ran']} epoch · test {binary['test_rows']:,}개"
        )
        left, right = st.columns((1.15, 1))
        with left:
            st.image(
                str(WM_RESULTS_DIR / "test_confusion_matrix.png"),
                caption="Lot 비중복 test의 클래스별 정규화 혼동행렬",
                width="stretch",
            )
        with right:
            display_report = class_report.rename(
                columns={
                    "class": "클래스",
                    "precision": "정밀도",
                    "recall": "재현율",
                    "f1-score": "F1",
                    "support": "표본 수",
                }
            )
            st.dataframe(
                display_report,
                hide_index=True,
                width="stretch",
                column_config={
                    "정밀도": st.column_config.NumberColumn(format="%.3f"),
                    "재현율": st.column_config.NumberColumn(format="%.3f"),
                    "F1": st.column_config.NumberColumn(format="%.3f"),
                    "표본 수": st.column_config.NumberColumn(format="%d"),
                },
            )
            st.markdown(
                f"""
                **정상/불량 이진 관점**

                - 불량 정탐: {binary['true_positive']:,}개
                - 정상 오탐: {binary['false_positive']:,}개
                - 불량 미탐: {binary['false_negative']:,}개
                - 정밀도: {binary['precision']:.2%}
                """
            )
        st.info(
            "학습·검증·테스트 lot가 겹치지 않도록 분리했습니다. 정확도만 보면 다수 "
            "클래스인 none의 영향을 크게 받으므로 Macro-F1과 클래스별 재현율을 함께 봅니다."
        )
        st.divider()
        st.subheader("학습 전략 선택")
        selection = json.loads(
            (WM_RESULTS_DIR / "selection_protocol.json").read_text(encoding="utf-8")
        )
        comparison = pd.read_csv(
            WM_RESULTS_DIR / "validation_strategy_comparison.csv"
        ).rename(
            columns={
                "strategy": "전략",
                "validation_macro_f1": "Validation Macro-F1",
                "validation_balanced_accuracy": "Validation 균형 정확도",
                "validation_accuracy": "Validation 정확도",
                "scratch_recall": "Scratch 재현율",
                "loc_recall": "Loc 재현율",
                "near_full_recall": "Near-full 재현율",
                "best_epoch": "최적 epoch",
            }
        )
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.image(
            str(WM_RESULTS_DIR / "validation_strategy_comparison.png"),
            caption="Test를 사용하지 않은 validation 전략 비교",
            width="stretch",
        )
        st.success(
            f"Validation Macro-F1 기준으로 {selection['selected_strategy']}를 선택했습니다."
        )
        st.caption(
            "selection_uses_test=false로 기록했으며 test 지표는 선택을 고정한 뒤 "
            "참고용으로만 확인했습니다."
        )
        st.divider()
        st.subheader("다중 시드 재현성 검증")
        multiseed_summary_path = WM_MULTISEED_DIR / "reproducibility_summary.json"
        if multiseed_summary_path.is_file():
            multiseed = json.loads(multiseed_summary_path.read_text(encoding="utf-8"))
            multiseed_rows = pd.read_csv(WM_MULTISEED_DIR / "seed_metrics.csv")
            seed_stats = multiseed["metrics"]
            seed_columns = st.columns(3)
            seed_columns[0].metric("반복 학습 시드", f"{len(multiseed['seeds'])}개")
            seed_columns[1].metric("Validation Macro-F1 평균 ± 표준편차", f"{seed_stats['validation_macro_f1']['mean']:.4f} ± {seed_stats['validation_macro_f1']['std']:.4f}")
            seed_columns[2].metric("고정 Test Macro-F1 평균 ± 표준편차", f"{seed_stats['test_macro_f1']['mean']:.4f} ± {seed_stats['test_macro_f1']['std']:.4f}")
            seed_display = multiseed_rows[["seed", "validation_macro_f1", "test_macro_f1", "test_balanced_accuracy", "best_epoch"]].rename(columns={"seed": "시드", "validation_macro_f1": "Validation Macro-F1", "test_macro_f1": "고정 Test Macro-F1", "test_balanced_accuracy": "고정 Test 균형 정확도", "best_epoch": "최적 epoch"})
            st.dataframe(seed_display, hide_index=True, width="stretch", column_config={name: st.column_config.NumberColumn(format="%.4f") for name in ("Validation Macro-F1", "고정 Test Macro-F1", "고정 Test 균형 정확도")})
            st.image(str(WM_MULTISEED_DIR / "reproducibility_metrics.png"), caption="선택 전략과 분할을 고정하고 난수 시드만 변경", width="stretch")
            class_recall = pd.DataFrame(json.loads((WM_MULTISEED_DIR / "class_recall_summary.json").read_text(encoding="utf-8")))
            with st.expander("클래스별 시드 변동 확인 (고정 Test 사후 분석)"):
                class_display = class_recall.loc[class_recall["split"] == "test", ["class_name", "support_per_seed", "recall_mean", "recall_std", "recall_min", "recall_max"]].rename(columns={"class_name": "클래스", "support_per_seed": "동일 Test 표본 수", "recall_mean": "재현율 평균", "recall_std": "표본 표준편차", "recall_min": "최저 재현율", "recall_max": "최고 재현율"})
                st.dataframe(class_display, hide_index=True, width="stretch", column_config={name: st.column_config.NumberColumn(format="percent") for name in ("재현율 평균", "표본 표준편차", "최저 재현율", "최고 재현율")})
                st.caption("클래스별 변동은 전체 Macro-F1보다 큽니다. Near-full은 동일한 Test 21개를 반복 평가하므로 소수 표본 한계가 있습니다.")
            st.info("동일한 분할의 3회 초기값 안정성 평가입니다. 표준편차는 신뢰구간이 아니며 신규 lot·외부 공정 성능을 보장하지 않습니다. 가장 좋은 시드를 새로 선택하지 않았고 기존 시드 42 배포 모델과 보정·OOD 정책은 유지했습니다.")
            st.caption("Validation Macro-F1로 epoch를 선택했습니다. Test는 고정 후 평가에만 사용했으며, 이 실험은 각 시드의 보정·OOD 정책 재검증을 포함하지 않습니다.")
            st.download_button("재현성 검증 기록 다운로드", (WM_MULTISEED_DIR / "검증결과.md").read_text(encoding="utf-8"), file_name="wm811k_multiseed_validation.md", mime="text/markdown", key="wm811k_multiseed_report")
        else:
            st.caption("다중 시드 Colab 결과를 검증한 후 표시합니다.")
        st.divider()
        st.subheader("Held-out lot 일반화 감사")
        lot_summary_path = WM_LOT_AUDIT_DIR / "lot_generalization_summary.json"
        if lot_summary_path.is_file():
            lot_summary = json.loads(lot_summary_path.read_text(encoding="utf-8"))
            lot_metrics = pd.read_csv(WM_LOT_AUDIT_DIR / "lot_metrics.csv")
            lot_columns = st.columns(4)
            lot_columns[0].metric("고정 Test lot", f"{lot_summary['lot_count']:,}개")
            lot_columns[1].metric("전체 정확도", f"{lot_summary['global_accuracy']:.2%}")
            lot_columns[2].metric("Lot bootstrap 95% 구간", f"{lot_summary['cluster_bootstrap_accuracy_ci95'][0]:.2%}~{lot_summary['cluster_bootstrap_accuracy_ci95'][1]:.2%}")
            lot_columns[3].metric("20개 이상 lot 10% 분위수", f"{lot_summary['eligible_lot_accuracy_p10']:.2%}")
            st.image(str(WM_LOT_AUDIT_DIR / "lot_generalization_dashboard.png"), caption="고정 Test를 lot·웨이퍼 크기·클래스별로 나눈 사후 진단", width="stretch")
            with st.expander("오류가 많은 held-out lot 확인"):
                lot_display = lot_metrics.loc[lot_metrics["support"] >= 20].head(15)[["lot_name", "support", "error_count", "accuracy", "accuracy_ci95_low", "accuracy_ci95_high", "review_or_ood_rate"]].rename(columns={"lot_name": "lot", "support": "표본 수", "error_count": "오류 수", "accuracy": "정확도", "accuracy_ci95_low": "정확도 95% 하한", "accuracy_ci95_high": "정확도 95% 상한", "review_or_ood_rate": "검토·OOD 비율"})
                st.dataframe(lot_display, hide_index=True, width="stretch", column_config={name: st.column_config.NumberColumn(format="percent") for name in ("정확도", "정확도 95% 하한", "정확도 95% 상한", "검토·OOD 비율")})
            st.caption(f"표본 100개 이상 웨이퍼 크기 그룹 중 최저는 {lot_summary['lowest_eligible_geometry']} ({lot_summary['lowest_eligible_geometry_support']}개, 정확도 {lot_summary['eligible_geometry_accuracy_min']:.2%})입니다. 클래스 구성 차이가 섞여 있으므로 크기의 인과효과가 아닙니다.")
            st.warning(f"개별 lot은 최대 {lot_summary['lot_support_max']}개로 작아 최저 lot 순위를 확정적 품질 순위로 해석할 수 없습니다. 동일 고정 Test의 사후 분석이며 신규 장비·공정 외부 검증이 아닙니다.")
            st.caption("used_for_model_selection=false, used_for_threshold_selection=false. 이 결과로 모델이나 검토·OOD 임계값을 변경하지 않았습니다.")
            geometry_summary_path = WM_GEOMETRY_DIR / "geometry_candidate_summary.json"
            if geometry_summary_path.is_file():
                geometry_summary = json.loads(geometry_summary_path.read_text(encoding="utf-8"))
                st.markdown("#### 웨이퍼 크기 구성 보정")
                geometry_columns = st.columns(3)
                geometry_columns[0].metric("Validation 후보", ", ".join(geometry_summary["selected_geometries"]))
                geometry_columns[1].metric("후보의 고정 Test 정확도", f"{geometry_summary['selected_test_accuracy']:.2%}")
                geometry_columns[2].metric("후보에 포함된 Test 오류", f"{geometry_summary['selected_test_error_count']}개 ({geometry_summary['selected_test_error_share']:.2%})")
                st.image(str(WM_GEOMETRY_DIR / "geometry_adjusted_dashboard.png"), caption="각 크기의 클래스 구성을 감안한 Validation·고정 Test 정확도 격차", width="stretch")
                st.warning("Validation에서 찾은 후보 안전 규칙이며 현재 판정에는 적용하지 않았습니다. 기준은 데이터 분석 전에 사전 등록되지 않았고, 웨이퍼 크기는 장비·공정 차이의 대리변수일 수 있습니다.")
                st.caption(f"Feature OOD와 가상 결합 시 검토율 {geometry_summary['hypothetical_combined_review_rate']:.2%}, 오류 포착률 {geometry_summary['hypothetical_combined_error_capture_rate']:.2%}입니다. 실제 적용 전 검토 비용과 신규 lot 검증이 필요합니다.")
        else:
            st.caption("held-out lot 일반화 감사 결과가 없습니다.")
        st.divider()
        st.subheader("불확실성 검토 정책")
        policy_test = uncertainty_policy["test_evaluation"]
        policy_metrics = st.columns(4)
        policy_metrics[0].metric(
            "자동 분류 비율", f"{policy_test['automatic_coverage']:.2%}"
        )
        policy_metrics[1].metric(
            "자동 분류 구간 정확도", f"{policy_test['automatic_accuracy']:.2%}"
        )
        policy_metrics[2].metric(
            "오류 검토 포착률", f"{policy_test['error_capture_rate']:.2%}"
        )
        policy_metrics[3].metric(
            "검토 대상", f"{policy_test['review_count']:,}개"
        )
        st.caption(
            f"Validation에서 자동 분류 목표 90%로 정한 보정 신뢰도 임계값은 "
            f"{uncertainty_policy['review_threshold']:.2%}입니다. Test는 기준을 "
            "고정한 뒤 평가에만 사용했습니다."
        )
        st.info(
            "이 정책은 낮은 신뢰도 입력을 전문가 검토 대상으로 보내는 연구용 "
            "안전장치입니다. 실제 공정에 적용하려면 신규 lot로 재검증해야 합니다."
        )
        st.divider()
        st.subheader("입력 분포 이탈(OOD) 안전 기준")
        ood_test = next(
            item for item in ood_summary["split_metrics"] if item["split"] == "test"
        )
        ood_stress = ood_summary["stress_metrics"]
        ood_metrics = st.columns(4)
        ood_metrics[0].metric(
            "Test 분포 내 자동 처리", f"{ood_test['automatic_coverage']:.2%}"
        )
        ood_metrics[1].metric("Test 경계 검토", f"{ood_test['review_rate']:.2%}")
        ood_metrics[2].metric("Test OOD 보류", f"{ood_test['ood_rate']:.2%}")
        ood_metrics[3].metric(
            "공간 셔플 OOD 탐지", f"{ood_stress['ood_detection_rate']:.2%}"
        )
        ood_columns = st.columns((1.2, 1))
        with ood_columns[0]:
            st.image(
                str(WM_OOD_DIR / "ood_score_distribution.png"),
                caption="Validation에서 고정한 특징 공간 OOD 경계",
                width="stretch",
            )
        with ood_columns[1]:
            st.markdown(
                f"""
                **고정 기준**

                - 특징: CNN pooled feature 256차원
                - 기준 적합: train {ood_summary['split_rows']['train']:,}개
                - 임계값 고정: validation {ood_summary['split_rows']['validation']:,}개
                - 경계 검토 점수: {ood_reference['thresholds']['review_threshold']:.4f}
                - OOD 판정 보류 점수: {ood_reference['thresholds']['ood_threshold']:.4f}
                - test_used_for_threshold: `false`
                """
            )
            st.warning(
                "공간 셔플 stress는 실제 외부 장비 데이터가 아닙니다. OOD 기준은 "
                "안전 보조 장치이며 신규 장비·lot의 외부 검증을 대체하지 않습니다."
            )
        st.divider()
        st.subheader("입력 오류 강건성 스트레스 테스트")
        wm_robustness_path = WM_ROBUSTNESS_DIR / "robustness_summary.json"
        wm_robustness_table_path = WM_ROBUSTNESS_DIR / "stress_test_summary.csv"
        if wm_robustness_path.is_file() and wm_robustness_table_path.is_file():
            wm_robustness = json.loads(wm_robustness_path.read_text(encoding="utf-8"))
            wm_robustness_table = pd.read_csv(wm_robustness_table_path)
            robustness_metrics = st.columns(4)
            robustness_metrics[0].metric("평가된 교란 입력", f"{wm_robustness['evaluated_rows']:,}개")
            robustness_metrics[1].metric("Guardrail 통과", f"{wm_robustness['guardrail_pass_count']}/{wm_robustness['scenario_count']}")
            robustness_metrics[2].metric("최저 예측 유지율", f"{wm_robustness_table['prediction_stability'].min():.1%}")
            changed = wm_robustness_table["changed_prediction_count"] > 0
            capture = wm_robustness_table.loc[changed, "changed_prediction_capture"]
            robustness_metrics[3].metric("최저 변경 포착률", "변경 없음" if capture.empty else f"{capture.min():.1%}")
            dashboard_path = WM_ROBUSTNESS_DIR / "robustness_dashboard.png"
            if dashboard_path.is_file():
                st.image(str(dashboard_path), caption="파란색은 사전 기준 통과, 빨간색은 취약 조건", width="stretch")
            display_robustness = wm_robustness_table[["scenario_label", "prediction_stability", "changed_prediction_count", "changed_prediction_capture", "unreviewed_changed_count", "review_or_hold_rate", "ood_hold_rate", "guardrail_pass"]].rename(columns={"scenario_label": "교란 조건", "prediction_stability": "예측 유지율", "changed_prediction_count": "변경 건수", "changed_prediction_capture": "변경 포착률", "unreviewed_changed_count": "미포착 변경", "review_or_hold_rate": "검토·보류율", "ood_hold_rate": "OOD 보류율", "guardrail_pass": "Guardrail 통과"})
            st.dataframe(display_robustness, hide_index=True, width="stretch", column_config={column: st.column_config.NumberColumn(format="percent") for column in ("예측 유지율", "변경 포착률", "검토·보류율", "OOD 보류율")})
            if wm_robustness["guardrail_fail_count"]:
                failed_labels = wm_robustness_table.loc[~wm_robustness_table["guardrail_pass"], "scenario_label"].tolist()
                st.error("취약 조건: " + ", ".join(failed_labels) + ". 해당 입력에서는 자동 확정하지 말고 원본 웨이퍼 맵과 수집 상태를 확인해야 합니다.")
            st.caption("모델 선택 후 실제 test 정성 예시 중 기존 27개(클래스별 고신뢰 정답·검토 대상 정답·대표 오분류)에 조건별 5회 합성 교란을 주입했습니다. 이후 추가한 전형·경계 예시는 이 점검에 포함하지 않았습니다. 이는 전체 데이터의 대표 성능이나 실제 장비 고장 검증이 아니며, 모델·임계값 조정에 사용하지 않았습니다.")
            st.caption(
                f"원본부터 검토·보류 대상인 예시는 {wm_robustness['baseline_review_count']}/27개입니다. "
                "포착률에는 기존 검토 대상도 포함되므로 새로운 오류 탐지 성능으로 해석하지 않습니다. "
                "예측 변경은 정답 오류와 같은 의미가 아닙니다."
            )
        else:
            st.caption("robustness_results가 없습니다. evaluate_wm811k_robustness.py를 실행하세요.")

        st.divider()
        st.subheader("Grad-CAM 충실도·sanity 점검")
        gradcam_summary_path = WM_GRADCAM_AUDIT_DIR / "gradcam_faithfulness_summary.json"
        if gradcam_summary_path.is_file():
            gradcam_summary = json.loads(gradcam_summary_path.read_text(encoding="utf-8"))
            gradcam_columns = st.columns(4)
            gradcam_columns[0].metric("활성 die attribution mass", f"{gradcam_summary['mean_active_attribution_mass']:.1%}")
            gradcam_columns[1].metric("상위 10% 제거 신뢰도 변화", f"{gradcam_summary['mean_top10_confidence_drop']:+.3f}")
            gradcam_columns[2].metric("무작위 10% 제거 신뢰도 변화", f"{gradcam_summary['mean_random10_confidence_drop']:+.3f}")
            gradcam_columns[3].metric("상위 제거 > 무작위", f"{gradcam_summary['top_drop_exceeds_random_rate']:.1%}")
            st.image(str(WM_GRADCAM_AUDIT_DIR / "gradcam_faithfulness_dashboard.png"), caption="상위 Grad-CAM 영역과 동일 크기 무작위·하위 영역 제거 비교", width="stretch")
            if gradcam_summary["mean_top_vs_random_advantage"] <= 0:
                st.error("상위 Grad-CAM 영역 제거가 무작위 제거보다 평균적으로 더 큰 신뢰도 하락을 만들지 못했습니다. 현재 결과는 강한 설명 충실도 근거를 제공하지 않습니다.")
            st.caption(f"상위 영역 제거가 하위 영역보다 더 민감했던 예시는 {gradcam_summary['top_drop_exceeds_bottom_rate']:.1%}였습니다.")
            st.warning("클래스별로 고른 기존 27개 시연 예시(고신뢰 정답·검토 대상 정답·대표 오분류)의 사후 점검입니다. 전체 Test 대표 결과가 아니며 die 제거는 실제 공정 개입이 아닙니다. Grad-CAM은 물리적 원인이나 인과관계를 증명하지 않습니다.")
            st.caption(f"분류기 head만 무작위화한 heatmap 상관 평균은 {gradcam_summary['randomized_head_heatmap_correlation_mean']:.3f}입니다. 완전한 모델 무작위화 검정은 아닙니다.")
        else:
            st.caption("Grad-CAM 충실도 점검 결과가 없습니다.")

        st.divider()
        st.subheader("과거 설명법 비교 증거 · Grad-CAM·Integrated Gradients 비교")
        st.info(
            "0 기준 Integrated Gradients와 Grad-CAM을 비교한 과거 결과입니다. 현재 단일 "
            "진단에 표시하는 Gradient SHAP의 검증 결과가 아니며, Gradient SHAP 검증은 "
            "`성능·검증 → SHAP 검증`에 따로 표시합니다."
        )
        xai_matches_deployment = False
        xai_summary_path = WM_XAI_COMPARISON_DIR / "xai_method_summary.json"
        if xai_summary_path.is_file():
            xai_summary = json.loads(xai_summary_path.read_text(encoding="utf-8"))
            deployed_hash = load_file_sha256(
                str(WM_CHECKPOINT_PATH), WM_CHECKPOINT_PATH.stat().st_mtime_ns
            ).lower()
            xai_matches_deployment = (
                xai_summary.get("checkpoint_sha256") == deployed_hash
            )
            if not xai_matches_deployment:
                st.warning(
                    "이 XAI 비교는 배포 체크포인트와 다른 탐색 실험 결과입니다. "
                    "현재 모델의 설명 성능 근거로 사용하지 않습니다. 배포 체크포인트를 "
                    "직접 업로드하는 수정된 07번 Colab 노트북으로 재검증이 필요합니다."
                )
            method_columns = st.columns(2)
            for column, method in zip(method_columns, ("Grad-CAM", "Integrated Gradients")):
                values = xai_summary["methods"][method]
                with column:
                    st.markdown(f"#### {method}")
                    st.metric("상위-무작위 제거 효과", f"{values['macro_top_vs_random_advantage']:+.4f}")
                    st.metric("상위 제거 > 무작위", f"{values['top_exceeds_random_rate']:.1%}")
                    interval = values["top_vs_random_95ci"]
                    st.caption(f"표본 단위 bootstrap 참고 구간: {interval[0]:+.4f}~{interval[1]:+.4f}")
            paired_interval = xai_summary["paired_ig_minus_gradcam_95ci"]
            st.metric(
                "IG − Grad-CAM 상위·무작위 제거 효과",
                f"{xai_summary['paired_ig_minus_gradcam']:+.4f}",
            )
            st.caption(f"대응 표본 bootstrap 95% 구간: {paired_interval[0]:+.4f}~{paired_interval[1]:+.4f}")
            if paired_interval[0] > 0 and xai_matches_deployment:
                st.success("과거 비교의 이 고정 표본에서는 Integrated Gradients가 Grad-CAM보다 상위 attribution 영역을 더 충실하게 식별했습니다.")
            st.image(str(WM_XAI_COMPARISON_DIR / "xai_method_comparison_dashboard.png"), caption="고정 test 표본의 클래스별 설명 충실도 비교", width="stretch")
            st.warning("클래스별 최대 50개를 뽑은 균형 표본 결과입니다. Near-full은 test 전체 21개만 포함됐습니다. 실제 test 클래스 비율의 전체 평균이 아니며, 두 설명법 모두 물리적 원인이나 인과관계를 증명하지 않습니다.")
        else:
            st.info("정식 비교 결과가 아직 없습니다. 노트북/WM811K_07_XAI_방법비교_Colab.ipynb을 T4 GPU에서 실행한 뒤 결과 ZIP을 가져오면 표시됩니다.")

        near_full_summary_path = WM_NEAR_FULL_DIR / "near_full_summary.json"
        if near_full_summary_path.is_file():
            st.markdown("#### Near-full 집중 진단")
            near_full = json.loads(near_full_summary_path.read_text(encoding="utf-8"))
            near_columns = st.columns(4)
            deployed_near = next(
                item for item in error_analysis["class_analysis"]
                if item["class_name"] == "Near-full"
            )
            near_columns[0].metric("배포 모델 재현율", f"{deployed_near['recall']:.1%}")
            near_columns[1].metric("Test 표본", f"{near_full['test_sample_count']}개")
            near_columns[2].metric("신뢰도 검토", f"{deployed_near['review_count']}개")
            near_columns[3].metric("배포 모델 오분류", f"{deployed_near['error_count']}개")
            st.image(str(WM_NEAR_FULL_DIR / "near_full_diagnostic_dashboard.png"), caption="Near-full 전체 21개의 국소 10% 제거 민감도", width="stretch")
            st.info(
                f"현재 배포 모델은 Near-full 21개 중 {deployed_near['correct_count']}개를 맞혔고 "
                f"신뢰도 정책으로 {deployed_near['review_count']}개를 검토합니다. "
                + (
                    "국소 삭제 그래프도 같은 배포 체크포인트로 재검증했습니다. "
                    if xai_matches_deployment else
                    "국소 삭제 그래프는 다른 탐색 체크포인트 결과라 분리해서 봐야 합니다. "
                )
                + "전역 형태 지표는 모델과 무관합니다."
            )
            st.warning(
                "Near-full은 웨이퍼 전반에 퍼진 전역 패턴이라 무작위 10% 제거도 중요한 die를 포함하기 쉽습니다. "
                "이 지표가 낮다는 이유만으로 분류 모델 실패로 판단하지 않으며 모델·임계값은 변경하지 않습니다."
            )
            global_shape_path = (
                WM_NEAR_FULL_SHAPE_DIR / "near_full_global_shape_summary.json"
            )
            if global_shape_path.is_file():
                global_shape = json.loads(
                    global_shape_path.read_text(encoding="utf-8")
                )
                shape_metrics = global_shape["metrics"]
                st.markdown("##### Near-full 전역 형태 근거")
                shape_columns = st.columns(5)
                shape_columns[0].metric(
                    "결함 die 비율",
                    f"{shape_metrics['defect_ratio']['mean']:.1%}",
                )
                shape_columns[1].metric(
                    "최대 연결 영역",
                    f"{shape_metrics['largest_component_fraction']['mean']:.1%}",
                )
                shape_columns[2].metric(
                    "웨이퍼 경계 결함률",
                    f"{shape_metrics['boundary_defect_coverage']['mean']:.1%}",
                )
                shape_columns[3].metric(
                    "결함 분포 범위",
                    f"{shape_metrics['defect_bbox_fraction']['mean']:.1%}",
                )
                shape_columns[4].metric(
                    "연결 영역 중앙값",
                    f"{shape_metrics['component_count_8']['median']:.0f}개",
                )
                st.image(
                    str(
                        WM_NEAR_FULL_SHAPE_DIR
                        / "near_full_global_shape_dashboard.png"
                    ),
                    caption="고정 test Near-full 21개의 전역 형태 지표",
                    width="stretch",
                )
                st.success(
                    f"결함 die 평균 {shape_metrics['defect_ratio']['mean']:.1%}, "
                    f"최대 연결 영역 평균 "
                    f"{shape_metrics['largest_component_fraction']['mean']:.1%}, "
                    f"결함 분포 범위 평균 "
                    f"{shape_metrics['defect_bbox_fraction']['mean']:.1%}로 Near-full이 "
                    "넓게 이어진 전역 패턴임을 확인했습니다."
                )
                st.caption(
                    "고정 test 21개의 사후 형상 분석입니다. 지표는 모델·임계값 선택에 "
                    "사용하지 않았으며 물리적 공정 원인이나 인과관계를 증명하지 않습니다."
                )

        st.divider()
        st.subheader("클래스별 오분류 분석")
        class_diagnostics = pd.DataFrame(error_analysis["class_analysis"])
        chart_data = class_diagnostics.melt(
            id_vars="class_name",
            value_vars=("recall", "error_capture_rate"),
            var_name="metric",
            value_name="value",
        )
        chart_data["metric"] = chart_data["metric"].map(
            {"recall": "클래스 재현율", "error_capture_rate": "오류 검토 포착률"}
        )
        diagnostic_chart = px.bar(
            chart_data,
            x="class_name",
            y="value",
            color="metric",
            barmode="group",
            labels={"class_name": "실제 클래스", "value": "비율", "metric": "지표"},
            title="클래스별 재현율과 불확실성 검토 포착률",
        )
        diagnostic_chart.update_yaxes(tickformat=".0%", range=(0, 1.05))
        st.plotly_chart(diagnostic_chart, width="stretch")

        display_classes = class_diagnostics[
            [
                "class_name",
                "support",
                "error_count",
                "recall",
                "review_rate",
                "error_capture_rate",
                "automatic_accuracy",
            ]
        ].copy()
        class_rate_columns = (
            "recall",
            "review_rate",
            "error_capture_rate",
            "automatic_accuracy",
        )
        display_classes.loc[:, list(class_rate_columns)] *= 100.0
        display_classes = display_classes.rename(
            columns={
                "class_name": "실제 클래스",
                "support": "표본 수",
                "error_count": "오분류 수",
                "recall": "재현율 (%)",
                "review_rate": "검토 대상 비율 (%)",
                "error_capture_rate": "오류 검토 포착률 (%)",
                "automatic_accuracy": "자동 분류 정확도 (%)",
            }
        )
        st.dataframe(
            display_classes,
            hide_index=True,
            width="stretch",
            column_config={
                "표본 수": st.column_config.NumberColumn(format="%d"),
                "오분류 수": st.column_config.NumberColumn(format="%d"),
                "재현율 (%)": st.column_config.NumberColumn(format="%.2f"),
                "검토 대상 비율 (%)": st.column_config.NumberColumn(format="%.2f"),
                "오류 검토 포착률 (%)": st.column_config.NumberColumn(format="%.2f"),
                "자동 분류 정확도 (%)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        st.markdown("**빈도가 높은 실제 → 예측 오분류 쌍**")
        confusion_pairs = pd.DataFrame(error_analysis["confusion_pairs"]).head(10)
        display_pairs = confusion_pairs[
            [
                "true_class",
                "predicted_class",
                "count",
                "share_of_all_errors",
                "mean_confidence",
                "review_capture_rate",
            ]
        ].copy()
        pair_rate_columns = (
            "share_of_all_errors",
            "mean_confidence",
            "review_capture_rate",
        )
        display_pairs.loc[:, list(pair_rate_columns)] *= 100.0
        display_pairs = display_pairs.rename(
            columns={
                "true_class": "실제 클래스",
                "predicted_class": "예측 클래스",
                "count": "건수",
                "share_of_all_errors": "전체 오류 비중 (%)",
                "mean_confidence": "평균 보정 신뢰도 (%)",
                "review_capture_rate": "검토 포착률 (%)",
            }
        )
        st.dataframe(
            display_pairs,
            hide_index=True,
            width="stretch",
            column_config={
                "건수": st.column_config.NumberColumn(format="%d"),
                "전체 오류 비중 (%)": st.column_config.NumberColumn(format="%.2f"),
                "평균 보정 신뢰도 (%)": st.column_config.NumberColumn(format="%.2f"),
                "검토 포착률 (%)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        top_pair = confusion_pairs.iloc[0]
        scratch = class_diagnostics.loc[
            class_diagnostics["class_name"] == "Scratch"
        ].iloc[0]
        st.info(
            f"가장 많은 혼동은 {top_pair['true_class']} → "
            f"{top_pair['predicted_class']} {int(top_pair['count'])}건입니다. "
            f"Scratch 재현율은 {scratch['recall']:.2%}입니다. "
            "이 표는 선택이 끝난 모델의 test 사후 진단이며 모델 선택에는 사용하지 않았습니다."
        )

        weak_summary_path = WM_WEAK_CLASS_DIR / "weak_class_summary.json"
        weak_table_path = WM_WEAK_CLASS_DIR / "weak_class_summary.csv"
        if weak_summary_path.is_file() and weak_table_path.is_file():
            st.markdown("#### Scratch·Loc·Edge-Loc 집중 분석")
            weak_summary = json.loads(weak_summary_path.read_text(encoding="utf-8"))
            weak_table = pd.read_csv(weak_table_path)
            weak_metrics = st.columns(4)
            weak_metrics[0].metric("세 클래스 오류", f"{weak_summary['weak_class_errors']}건")
            weak_metrics[1].metric("none으로 오분류", f"{weak_summary['errors_to_none_share']:.1%}")
            weak_metrics[2].metric("오류 검토 포착", f"{weak_summary['review_error_capture_rate']:.1%}")
            weak_metrics[3].metric("미포착 자동 오류", f"{weak_summary['automatic_error_count']}건")
            st.image(
                str(WM_WEAK_CLASS_DIR / "weak_class_dashboard.png"),
                caption="취약 클래스 결과·보정 신뢰도·설명 충실도",
                width="stretch",
            )
            display_weak = weak_table[
                [
                    "class_name", "support", "error_count", "recall",
                    "errors_to_none", "errors_to_none_share",
                    "error_capture_rate", "automatic_error_count",
                    "integrated_gradients_top_vs_random",
                ]
            ].rename(columns={
                "class_name": "클래스", "support": "표본 수",
                "error_count": "오류", "recall": "재현율",
                "errors_to_none": "none 오분류",
                "errors_to_none_share": "오류 중 none 비중",
                "error_capture_rate": "오류 검토 포착률",
                "automatic_error_count": "미포착 오류",
                "integrated_gradients_top_vs_random": "IG 상위-무작위 제거 효과",
            })
            st.dataframe(
                display_weak,
                hide_index=True,
                width="stretch",
                column_config={
                    "재현율": st.column_config.NumberColumn(format="percent"),
                    "오류 중 none 비중": st.column_config.NumberColumn(format="percent"),
                    "오류 검토 포착률": st.column_config.NumberColumn(format="percent"),
                    "IG 상위-무작위 제거 효과": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            st.info(
                "세 클래스 오류 310건 중 208건이 none 예측입니다. 다음 비교 실험은 "
                "train에만 약한 결함 보존 증강과 none hard-negative sampling을 적용하고 "
                "validation으로 선택하며 test는 최종 확인까지 잠급니다."
            )
            st.caption(
                "선택이 끝난 test의 사후 진단입니다. lot·형상별 비율은 표본 5개 이상만 "
                "비교하며, 이 결과로 현재 모델이나 임계값을 다시 선택하지 않습니다."
            )

        weak_validation_summary_path = WM_WEAK_VALIDATION_DIR / "experiment_summary.json"
        weak_validation_table_path = WM_WEAK_VALIDATION_DIR / "validation_candidate_comparison.csv"
        weak_bundle_report_path = WM_WEAK_VALIDATION_DIR / "bundle_validation.json"
        if (
            weak_validation_summary_path.is_file()
            and weak_validation_table_path.is_file()
            and weak_bundle_report_path.is_file()
        ):
            st.markdown("#### 취약 클래스 개선 후보 · Validation 전용 비교")
            candidate_summary = json.loads(
                weak_validation_summary_path.read_text(encoding="utf-8")
            )
            bundle_report = json.loads(weak_bundle_report_path.read_text(encoding="utf-8"))
            candidate_table = pd.read_csv(weak_validation_table_path)
            baseline_row = candidate_table.loc[candidate_table["candidate"].eq("baseline")].iloc[0]
            selected_row = candidate_table.loc[
                candidate_table["candidate"].eq(candidate_summary["selected_candidate"])
            ].iloc[0]
            candidate_metrics = st.columns(4)
            candidate_metrics[0].metric("선택 후보", candidate_summary["selected_candidate"])
            candidate_metrics[1].metric(
                "취약 3종 평균 재현율",
                f"{selected_row['validation_weak_recall']:.1%}",
                f"{selected_row['validation_weak_recall'] - baseline_row['validation_weak_recall']:+.1%}",
            )
            candidate_metrics[2].metric(
                "Validation macro-F1", f"{selected_row['validation_macro_f1']:.3f}"
            )
            candidate_metrics[3].metric(
                "검증 표본", f"{bundle_report['validation_rows_per_candidate']:,}건"
            )
            st.image(
                str(WM_WEAK_VALIDATION_DIR / "validation_candidate_dashboard.png"),
                caption="Validation-only 후보 성능과 안전 기준 통과 여부",
                width="stretch",
            )
            display_candidates = candidate_table[
                [
                    "candidate", "validation_macro_f1", "validation_weak_recall",
                    "validation_none_recall", "macro_f1_guardrail",
                    "none_recall_guardrail", "eligible",
                ]
            ].rename(columns={
                "candidate": "후보",
                "validation_macro_f1": "Macro-F1",
                "validation_weak_recall": "취약 3종 평균 재현율",
                "validation_none_recall": "none 재현율",
                "macro_f1_guardrail": "Macro-F1 기준",
                "none_recall_guardrail": "none 기준",
                "eligible": "선택 가능",
            })
            st.dataframe(
                display_candidates,
                hide_index=True,
                width="stretch",
                column_config={
                    "Macro-F1": st.column_config.NumberColumn(format="%.4f"),
                    "취약 3종 평균 재현율": st.column_config.NumberColumn(format="percent"),
                    "none 재현율": st.column_config.NumberColumn(format="percent"),
                },
            )
            st.warning(
                "약한 결함 thinning 후보들은 Macro-F1 안전 기준을 통과하지 못했습니다. "
                "따라서 baseline을 유지하며, 이 실험 체크포인트는 배포에 사용하지 않습니다."
            )
            st.caption(
                "Colab 결과 ZIP의 경로·파일 목록·원본 validation split·예측 확률·재계산 지표·"
                "선택 규칙을 로컬에서 교차검증했습니다. Test 평가는 잠금 상태입니다."
            )

        weak_margin_summary_path = WM_WEAK_MARGIN_DIR / "experiment_summary.json"
        weak_margin_table_path = WM_WEAK_MARGIN_DIR / "validation_margin_candidate_comparison.csv"
        weak_margin_validation_path = WM_WEAK_MARGIN_DIR / "bundle_validation.json"
        if (
            weak_margin_summary_path.is_file()
            and weak_margin_table_path.is_file()
            and weak_margin_validation_path.is_file()
        ):
            st.markdown("#### 취약 클래스 margin 후보 · Validation 전용")
            margin_summary = json.loads(weak_margin_summary_path.read_text(encoding="utf-8"))
            margin_validation = json.loads(weak_margin_validation_path.read_text(encoding="utf-8"))
            margin_table = pd.read_csv(weak_margin_table_path)
            baseline_margin = margin_table.loc[margin_table["candidate"].eq("baseline")].iloc[0]
            selected_margin = margin_table.loc[
                margin_table["candidate"].eq(margin_summary["selected_candidate"])
            ].iloc[0]
            margin_metrics = st.columns(4)
            margin_metrics[0].metric("선택 후보", margin_summary["selected_candidate"])
            margin_metrics[1].metric(
                "취약 3종 평균 재현율",
                f"{selected_margin['validation_weak_recall']:.1%}",
                f"{selected_margin['validation_weak_recall'] - baseline_margin['validation_weak_recall']:+.1%}",
            )
            margin_metrics[2].metric(
                "Validation macro-F1",
                f"{selected_margin['validation_macro_f1']:.4f}",
                f"{selected_margin['validation_macro_f1'] - baseline_margin['validation_macro_f1']:+.4f}",
            )
            margin_metrics[3].metric(
                "none 재현율",
                f"{selected_margin['validation_none_recall']:.2%}",
                f"{selected_margin['validation_none_recall'] - baseline_margin['validation_none_recall']:+.2%}",
            )
            st.image(
                str(WM_WEAK_MARGIN_DIR / "validation_margin_dashboard.png"),
                caption="Margin 후보의 validation 성능과 안전 기준",
                width="stretch",
            )
            display_margin = margin_table[
                [
                    "candidate", "weak_margin_lambda", "validation_macro_f1",
                    "validation_weak_recall", "validation_none_recall", "eligible",
                ]
            ].rename(columns={
                "candidate": "후보", "weak_margin_lambda": "Margin λ",
                "validation_macro_f1": "Macro-F1",
                "validation_weak_recall": "취약 3종 평균 재현율",
                "validation_none_recall": "none 재현율", "eligible": "안전 기준 통과",
            })
            st.dataframe(
                display_margin,
                hide_index=True,
                width="stretch",
                column_config={
                    "Margin λ": st.column_config.NumberColumn(format="%.2f"),
                    "Macro-F1": st.column_config.NumberColumn(format="%.4f"),
                    "취약 3종 평균 재현율": st.column_config.NumberColumn(format="percent"),
                    "none 재현율": st.column_config.NumberColumn(format="percent"),
                },
            )
            st.success(
                "λ=0.05 후보가 validation 안전 기준을 통과하면서 취약 클래스 재현율과 "
                "Macro-F1을 함께 개선했습니다."
            )
            st.warning(
                "현재는 seed 42 한 번의 결과입니다. 다중 seed 재현 전에는 체크포인트를 "
                "배포하거나 test 평가에 사용하지 않습니다."
            )
            st.caption(
                f"로컬 번들 검증 상태: {margin_validation['status']} · "
                f"baseline 체크포인트 일치: {margin_validation['baseline_reuse_sha_matches']}"
            )

        margin_multiseed_summary_path = (
            WM_WEAK_MARGIN_MULTISEED_DIR / "reproducibility_summary.json"
        )
        margin_multiseed_metrics_path = (
            WM_WEAK_MARGIN_MULTISEED_DIR / "metric_summary.csv"
        )
        margin_multiseed_paired_path = (
            WM_WEAK_MARGIN_MULTISEED_DIR / "paired_seed_deltas.csv"
        )
        margin_multiseed_validation_path = (
            WM_WEAK_MARGIN_MULTISEED_DIR / "bundle_validation.json"
        )
        if all(path.is_file() for path in (
            margin_multiseed_summary_path,
            margin_multiseed_metrics_path,
            margin_multiseed_paired_path,
            margin_multiseed_validation_path,
        )):
            st.markdown("#### Margin 다중 seed 재현성 검증 · Validation 전용")
            multiseed_summary = json.loads(
                margin_multiseed_summary_path.read_text(encoding="utf-8")
            )
            multiseed_validation = json.loads(
                margin_multiseed_validation_path.read_text(encoding="utf-8")
            )
            multiseed_metrics = pd.read_csv(margin_multiseed_metrics_path).set_index(
                "strategy"
            )
            multiseed_paired = pd.read_csv(margin_multiseed_paired_path)
            baseline_multi = multiseed_metrics.loc["baseline"]
            margin_multi = multiseed_metrics.loc["weak_none_margin_005"]
            weak_delta = (
                margin_multi["validation_weak_recall_mean"]
                - baseline_multi["validation_weak_recall_mean"]
            )
            stability_delta = (
                margin_multi["validation_weak_recall_std"]
                - baseline_multi["validation_weak_recall_std"]
            )
            multiseed_columns = st.columns(4)
            multiseed_columns[0].metric(
                "다중 seed 최종 선택", multiseed_summary["selected_candidate"]
            )
            multiseed_columns[1].metric(
                "Baseline 취약 재현율 평균",
                f"{baseline_multi['validation_weak_recall_mean']:.2%}",
            )
            multiseed_columns[2].metric(
                "Margin 취약 재현율 평균",
                f"{margin_multi['validation_weak_recall_mean']:.2%}",
                f"{weak_delta:+.2%}",
            )
            multiseed_columns[3].metric(
                "Margin 취약 재현율 표준편차",
                f"{margin_multi['validation_weak_recall_std']:.2%}",
                f"{stability_delta:+.2%}",
                delta_color="inverse",
            )
            st.image(
                str(WM_WEAK_MARGIN_MULTISEED_DIR / "multiseed_margin_dashboard.png"),
                caption="seed 17·42·2026 쌍 비교와 margin-baseline 차이",
                width="stretch",
            )
            paired_display = multiseed_paired[[
                "seed", "delta_validation_weak_recall",
                "delta_validation_macro_f1", "delta_validation_none_recall",
                "delta_validation_balanced_accuracy",
            ]].rename(columns={
                "seed": "Seed",
                "delta_validation_weak_recall": "취약 재현율 변화",
                "delta_validation_macro_f1": "Macro-F1 변화",
                "delta_validation_none_recall": "none 재현율 변화",
                "delta_validation_balanced_accuracy": "Balanced accuracy 변화",
            })
            st.dataframe(
                paired_display,
                hide_index=True,
                width="stretch",
                column_config={
                    column: st.column_config.NumberColumn(format="%+.4f")
                    for column in paired_display.columns if column != "Seed"
                },
            )
            failed_guardrails = [
                label for key, label in {
                    "macro_f1_guardrail_all_seeds": "Macro-F1",
                    "none_recall_guardrail_all_seeds": "none 재현율",
                    "balanced_accuracy_guardrail_all_seeds": "Balanced accuracy",
                }.items()
                if not multiseed_summary["checks"][key]
            ]
            st.warning(
                "Margin λ=0.05는 취약 클래스 평균 재현율을 개선하고 seed 간 편차를 "
                "줄였지만, 모든 seed에서 지켜야 하는 안전 기준을 통과하지 못했습니다: "
                + ", ".join(failed_guardrails)
                + ". 따라서 기존 baseline을 유지합니다."
            )
            st.caption(
                f"로컬 검증 상태: {multiseed_validation['status']} · "
                f"검증 예측 {multiseed_validation['validation_prediction_rows_checked']:,}행 · "
                "Test 미사용 · 배포 모델 변경 없음"
            )

    def render_wm_diagnosis() -> None:
        st.subheader("1. 샘플 선택과 입력 신뢰도")
        source = st.radio(
            "입력 방식",
            ("실제 test 예시", "파일 업로드", "인공 데모 맵"),
            horizontal=True,
            key="wm811k_source",
        )
        wafer = None
        source_name = ""
        demo_metadata = None
        if source == "실제 test 예시":
            demo_manifest_path = WM_DEMO_DIR / "manifest.json"
            demo_manifest = load_wafer_demo_manifest(
                str(demo_manifest_path), demo_manifest_path.stat().st_mtime_ns
            )
            st.markdown("#### 대표 샘플 선택")
            selected_true_class = st.selectbox(
                "실제 클래스",
                wm_gallery.class_order(demo_manifest),
                key="wm811k_demo_class",
            )
            class_cards = wm_gallery.cards_for_class(demo_manifest, selected_true_class)
            st.caption(
                f"{selected_true_class}의 실제 test 대표 샘플 {len(class_cards)}개를 한 번에 "
                "보여 줍니다. 카드의 예측·점수·검토 여부는 저장된 값이며, 아래 CNN·Gradient "
                "SHAP·OOD 상세 계산은 선택한 한 개에만 실행합니다."
            )
            selected_sample_id = wm_gallery.render_sample_gallery(
                class_cards,
                state_key="wm811k_selected_sample",
                thumbnail=lambda card: demo_wafer_png(
                    str(WM_DEMO_DIR / card.npy_file),
                    (WM_DEMO_DIR / card.npy_file).stat().st_mtime_ns,
                    wafer_view.THUMBNAIL_MIN_SIDE,
                ),
                score_formatter=format_model_score,
            )
            demo_metadata = next(
                item for item in demo_manifest["samples"]
                if item["sample_id"] == selected_sample_id
            )
            selected_case = demo_metadata["selection_case"]
            selected_case_label = wm_gallery.CASE_LABELS.get(selected_case, selected_case)
            sample_path = WM_DEMO_DIR / demo_metadata["npy_file"]
            wafer = parse_wafer_bytes(sample_path.read_bytes(), sample_path.name)
            source_name = (
                f"실제 test · {selected_true_class} · {selected_case_label}"
            )
            st.info(
                f"분석 대상: {selected_case_label} · {demo_metadata['sample_id']} · "
                f"실제 라벨: {demo_metadata['true_class']} · 저장 예측: "
                f"{demo_metadata['predicted_class']} · 보정 신뢰도: "
                f"{format_model_score(demo_metadata['calibrated_confidence'])} · lot: "
                f"{demo_metadata['lot_name']}"
            )
            per_class = demo_manifest.get("samples_per_class_range") or ()
            per_class_text = (
                f"클래스별 {per_class[0]}~{per_class[1]}개 선정"
                if len(per_class) == 2
                else "클래스별 대표 샘플"
            )
            st.caption(
                f"선정 근거: {demo_manifest.get('selection_rules', {}).get(selected_case, '모델 선택 후 test에서 고정한 예시')} · "
                f"이 클래스의 실제 test 대표 샘플 {len(class_cards)}개({per_class_text}) 중 하나이며, "
                "모델 선택 후 고정한 정성 시연 예시로 성능 평가에는 사용하지 않습니다."
            )
        elif source == "파일 업로드":
            uploaded = st.file_uploader(
                "웨이퍼 맵 한 개를 선택하세요",
                type=("npy", "csv", "txt", "data"),
                key="wm811k_upload",
            )
            if uploaded is None:
                st.info("0·1·2 값으로 구성된 2차원 웨이퍼 맵 파일을 업로드하세요.")
            else:
                try:
                    wafer = parse_wafer_bytes(uploaded.getvalue(), uploaded.name)
                    source_name = uploaded.name
                except Exception as error:
                    st.error(f"웨이퍼 맵을 읽지 못했습니다: {error}")
        else:
            wafer = artificial_demo_map()
            source_name = "인공 Edge-Ring 형태 예시"
            st.warning(
                "이 입력은 화면 기능 확인용 인공 패턴입니다. 실제 test 성능 근거로 "
                "사용하거나 새로운 실측 데이터로 간주하면 안 됩니다."
            )

        if wafer is not None:
            preview_columns = st.columns((1.1, 1))
            input_png, input_scale = wafer_view.categorical_png(wafer)
            with preview_columns[0]:
                with st.container(key="wafer_map_input"):
                    wafer_view.show_png(
                        input_png,
                        caption=f"{source_name} · 입력 웨이퍼 맵(범주형 0·1·2)",
                    )
                st.markdown(wafer_view.category_legend_html(), unsafe_allow_html=True)
            with preview_columns[1]:
                preview_shape = canonical_wafer_shape_features(wafer)
                preview_rows = [
                    {"항목": "입력 크기", "값": f"{wafer.shape[0]}×{wafer.shape[1]}"},
                ]
                if demo_metadata is not None:
                    preview_rows.append({
                        "항목": "원본 die 격자",
                        "값": f"{demo_metadata['original_height']}×{demo_metadata['original_width']}",
                    })
                preview_rows.extend([
                    {
                        "항목": "화면 표시",
                        "값": f"{input_scale['height']}×{input_scale['width']} px · "
                        + (
                            f"{input_scale['factor']}배 블록 확대"
                            if input_scale["step"] == 1
                            else f"{input_scale['step']}칸 간격 축소"
                        ),
                    },
                    {"항목": "값 구성", "값": ", ".join(map(str, sorted(np.unique(wafer))))},
                    {"항목": "결함 die 비율", "값": f"{preview_shape['defect_ratio']:.1%}"},
                    {"항목": "입력 해시", "값": canonical_wafer_sha256(wafer)[:18] + "…"},
                ])
                st.dataframe(pd.DataFrame(preview_rows), hide_index=True, width="stretch")
                st.caption(
                    "파일은 0(웨이퍼 밖), 1(정상 die), 2(결함 die) 범주형 값으로 "
                    "검증하고 64×64로 최근접 보간합니다. "
                    + wafer_view.scale_note(wafer.shape, input_scale)
                )
                if demo_metadata is not None:
                    st.caption(
                        "저장된 실제 test 샘플은 원본 die 격자를 학습과 같은 방식으로 "
                        "64×64 최근접 변환한 모델 입력입니다."
                    )
            st.divider()
            st.subheader("2. 모델 판정과 SHAP 근거")
            try:
                with st.spinner("CNN 예측과 Gradient SHAP 계산 중..."):
                    model, class_names, checkpoint = load_wafer_model(
                        str(WM_CHECKPOINT_PATH), WM_CHECKPOINT_PATH.stat().st_mtime_ns
                    )
                    result = predict_with_gradcam(
                        model,
                        class_names,
                        wafer,
                        temperature=float(uncertainty_policy["temperature"]),
                    )
                    gradient_shap_result = explain_wafer_gradient_shap(
                        str(WM_CHECKPOINT_PATH),
                        WM_CHECKPOINT_PATH.stat().st_mtime_ns,
                        wafer,
                        float(uncertainty_policy["temperature"]),
                    )
                    ood_result = assess_wafer_ood_batch(
                        model, class_names, [wafer], ood_reference
                    )[0]
                    shape_features = canonical_wafer_shape_features(wafer)
                predicted = str(result["predicted_label"])
                if demo_metadata is not None:
                    if predicted != demo_metadata["predicted_class"]:
                        raise RuntimeError(
                            "저장 예시의 예측과 현재 체크포인트 예측이 다릅니다."
                        )
                    confidence_difference = abs(
                        float(result["confidence"])
                        - float(demo_metadata["calibrated_confidence"])
                    )
                    if confidence_difference > 1e-4:
                        raise RuntimeError(
                            "저장 예시의 보정 신뢰도와 현재 계산값이 다릅니다."
                        )
                status = "정상" if predicted == "none" else f"불량 · {predicted}"
                low_confidence = float(result["confidence"]) < float(
                    uncertainty_policy["review_threshold"]
                )
                ood_status = str(ood_result["ood_status"])
                needs_review = low_confidence or ood_status != "in_distribution"
                review_status = (
                    "자동판정 보류"
                    if ood_status == "out_of_distribution"
                    else "전문가 검토 필요"
                    if needs_review
                    else "자동 분류 가능"
                )
                metric_columns = st.columns(5)
                metric_columns[0].metric("CNN 판정", status)
                metric_columns[1].metric(
                    "보정 모델 점수", format_model_score(result["confidence"])
                )
                metric_columns[2].metric(
                    "입력 데이터 상태", WM_OOD_LABELS[ood_status]
                )
                metric_columns[3].metric("검토 정책", review_status)
                metric_columns[4].metric(
                    "체크포인트",
                    f"Epoch {checkpoint['best_epoch']}",
                    delta=f"Validation Macro-F1 {checkpoint['best_validation_macro_f1']:.3f}",
                    delta_color="off",
                )
                st.caption(
                    f"OOD 점수 {ood_result['ood_score']:.4f} · 가까운 train 기준 클래스 "
                    f"{ood_result['nearest_reference_class']} · 웨이퍼 면적 "
                    f"{ood_result['wafer_coverage']:.2%} · 결함 die "
                    f"{ood_result['defect_ratio']:.2%}"
                )
                with st.expander("보정 모델 점수는 어떻게 계산했나요?"):
                    st.markdown(
                        f"""
                        - CNN logit에 Validation에서 학습한 temperature **{uncertainty_policy['temperature']:.4f}**를 적용한 뒤 softmax 최댓값을 표시합니다.
                        - 전문가 검토 기준 **{uncertainty_policy['review_threshold']:.2%}**는 Validation 자동 분류 비율 90% 목표로 고정했습니다.
                        - Test는 temperature와 기준값 선택에 사용하지 않았습니다.
                        - **99.9% 초과는 모델 출력이 큰 것이지, 실제 원인이나 정답이 100% 확실하다는 뜻이 아닙니다.**
                        """
                    )
                if ood_result["geometry_warnings"]:
                    st.warning(" · ".join(ood_result["geometry_warnings"]))
                with st.expander("전역 형태 지표", expanded=predicted == "Near-full"):
                    shape_columns = st.columns(6)
                    shape_columns[0].metric(
                        "결함 die 비율", f"{shape_features['defect_ratio']:.1%}"
                    )
                    shape_columns[1].metric(
                        "연결 영역", f"{shape_features['component_count_8']:,}개"
                    )
                    shape_columns[2].metric(
                        "최대 영역 점유",
                        f"{shape_features['largest_component_fraction']:.1%}",
                    )
                    shape_columns[3].metric(
                        "가장자리 결함",
                        f"{shape_features['edge_defect_fraction']:.1%}",
                    )
                    shape_columns[4].metric(
                        "경계 결함률",
                        f"{shape_features['boundary_defect_coverage']:.1%}",
                    )
                    shape_columns[5].metric(
                        "결함 분포 범위",
                        f"{shape_features['defect_bbox_fraction']:.1%}",
                    )
                    st.caption(
                        "연결 영역은 8-이웃 기준입니다. 가장자리는 활성 die 중심에서 "
                        "정규화 반경 0.70 이상이며, 결함 분포 범위는 결함 bounding box가 "
                        "활성 die bounding box에서 차지하는 비율입니다. 여러 지표를 함께 "
                        "형상 설명에 사용하고 공정 원인의 인과 증거로 해석하지 마세요."
                    )
                if ood_status == "out_of_distribution":
                    st.error(
                        "이 입력은 CNN 특징 공간에서 학습 분포를 벗어났습니다. 모델 "
                        "확률과 관계없이 자동판정을 보류하고 입력 형식·장비·공정을 확인하세요."
                    )
                elif needs_review:
                    st.warning(
                        "보정 신뢰도 또는 입력 분포 기준에 따라 전문가 검토가 필요합니다. "
                        "이 결과를 자동 확정하지 마세요."
                    )
                else:
                    st.success(
                        "Validation에서 정한 신뢰도 기준을 통과했습니다. 다만 신규 장비·"
                        "공정 데이터에서는 별도 검증이 필요합니다."
                    )

                cause_guidance = build_wafer_cause_guidance(
                    predicted,
                    shape_features,
                    ood_status=ood_status,
                )
                st.markdown("#### 가능한 불량 원인과 우선 점검")
                st.caption(
                    f"해석 등급: {cause_guidance['evidence_level']} · "
                    + " · ".join(cause_guidance["observations"])
                )
                if cause_guidance["candidates"]:
                    cause_table = pd.DataFrame(
                        [
                            {
                                "순위": item["priority"],
                                "원인 후보": item["cause"],
                                "패턴 연관성": item["rationale"],
                                "우선 확인": item["check"],
                            }
                            for item in cause_guidance["candidates"]
                        ]
                    )
                    st.dataframe(
                        cause_table,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "순위": st.column_config.NumberColumn(width="small"),
                            "원인 후보": st.column_config.TextColumn(width="medium"),
                            "패턴 연관성": st.column_config.TextColumn(width="large"),
                            "우선 확인": st.column_config.TextColumn(width="large"),
                        },
                    )
                    st.warning(cause_guidance["disclaimer"])
                else:
                    st.info(cause_guidance["disclaimer"])

                st.markdown("#### 예측 클래스의 위치 기여")
                st.caption(
                    "아래 두 지도는 연속값 설명 지도입니다. 범주형 입력 맵(0·1·2)과 달리 "
                    "die마다 0~1로 정규화한 기여 크기를 색 농도로 표시하며, die 한 칸을 "
                    "한 색 블록으로 그려 칸 사이 값을 만들지 않습니다."
                )
                with st.container(key="wafer_map_explanations"):
                    image_columns = st.columns(2)
                    with image_columns[0]:
                        wafer_view.show_png(
                            wafer_view.explanation_png(
                                gradient_shap_result["heatmap"],
                                gradient_shap_result["resized_map"],
                            )[0],
                            caption=(
                                f"{gradient_shap_result['selected_label']} 판단 Gradient SHAP · "
                                "진할수록 모델 기여가 큰 위치"
                            ),
                        )
                    with image_columns[1]:
                        wafer_view.show_png(
                            wafer_view.explanation_png(
                                result["heatmap"], result["resized_map"]
                            )[0],
                            caption=(
                                f"{result['selected_label']} 판단 Grad-CAM · "
                                "진할수록 영향이 큰 위치"
                            ),
                        )
                st.markdown(
                    wafer_view.explanation_scale_html(
                        low_label="0 · 기여 작음", high_label="1 · 이 지도의 최대 기여"
                    ),
                    unsafe_allow_html=True,
                )
                attempts = " → ".join(
                    str(item["samples"]) for item in gradient_shap_result["sample_attempts"]
                )
                additivity_text = (
                    f"원시 기여 합계 {gradient_shap_result['raw_attribution_sum']:+.3f} · "
                    f"로짓 변화 {gradient_shap_result['selected_logit_delta']:+.3f} · "
                    f"잔차 {gradient_shap_result['raw_additivity_residual']:+.3f} "
                    f"(상대 {gradient_shap_result['relative_additivity_residual']:.1%}, "
                    f"허용 ±{gradient_shap_result['additivity_tolerance']:.3f}) · "
                    f"경로 표본 {attempts}"
                )
                if gradient_shap_result["additivity_check_passed"]:
                    st.success("Gradient SHAP 가산성 점검 통과 · " + additivity_text)
                else:
                    st.warning(
                        "Gradient SHAP 가산성 점검 미통과 · "
                        + additivity_text
                        + " · 유한 표본 근사 오차가 허용 범위를 넘었으므로 이 설명 지도는 "
                        "검증 통과로 보지 않습니다. 위치 참고로만 보고 전문가 검토를 우선하세요."
                    )
                st.caption(
                    "Gradient SHAP은 같은 웨이퍼 외형에서 모든 활성 die가 정상인 기준과 현재 "
                    "입력 사이 경로를 층화 표본으로 근사한 Expected Gradients입니다. 기여도는 "
                    "보정 전 로짓 단위의 원시 값이며 로짓 변화에 맞춰 사후 조정하지 않습니다. "
                    "유한 표본 근사라 원시 합계와 로짓 변화 사이에 잔차가 남을 수 있어, 잔차가 "
                    "로짓 변화의 5%를 넘으면 표본을 두 배로 늘려 다시 계산하고(최대 256개) 모든 "
                    "시도를 기록합니다. 색 농도만 화면 표시용으로 0~1로 정규화했습니다. "
                    "해상도가 낮은 영역은 개별 die의 인과효과로 해석하지 않습니다."
                )

                probability_table = pd.DataFrame(
                    {
                        "클래스": class_names,
                        "보정 모델 점수": result["probabilities"],
                    }
                ).sort_values("보정 모델 점수", ascending=False)
                figure = px.bar(
                    probability_table,
                    x="보정 모델 점수",
                    y="클래스",
                    orientation="h",
                    title="클래스별 모델 점수",
                )
                figure.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(figure, width="stretch")
                single_audit = pd.DataFrame(
                    [
                        {
                            "파일명": source_name,
                            "입력 SHA-256": canonical_wafer_sha256(wafer),
                            "판정": "정상" if predicted == "none" else "불량",
                            "예측 클래스": predicted,
                            "보정 신뢰도": float(result["confidence"]),
                            "입력 신뢰도": WM_OOD_LABELS[ood_status],
                            "OOD 점수": float(ood_result["ood_score"]),
                            "검토 필요": "예" if needs_review else "아니오",
                            "처리 권고": review_status,
                            "원인 해석": cause_guidance["interpretation"],
                            "주요 원인 후보": " | ".join(
                                item["cause"]
                                for item in cause_guidance["candidates"]
                            ),
                            "우선 확인 항목": " | ".join(
                                item["check"]
                                for item in cause_guidance["candidates"]
                            ),
                            "형상 근거": " · ".join(
                                cause_guidance["observations"]
                            ),
                            "입력 품질 경고": " · ".join(
                                ood_result["geometry_warnings"]
                            ),
                            "결함 die 비율": float(shape_features["defect_ratio"]),
                            "8-이웃 연결 영역 수": int(
                                shape_features["component_count_8"]
                            ),
                            "최대 연결 영역 점유율": float(
                                shape_features["largest_component_fraction"]
                            ),
                            "가장자리 결함 구성비": float(
                                shape_features["edge_defect_fraction"]
                            ),
                            "웨이퍼 경계 결함률": float(
                                shape_features["boundary_defect_coverage"]
                            ),
                            "결함 분포 범위": float(
                                shape_features["defect_bbox_fraction"]
                            ),
                        }
                    ]
                )
                st.download_button(
                    "이 진단의 감사 HTML 보고서 다운로드",
                    data=build_wafer_audit_report(
                        source_name=source_name,
                        results=single_audit,
                        artifact_hashes=artifact_hashes,
                    ),
                    file_name="wm811k_single_diagnosis_report.html",
                    mime="text/html",
                    key="wm811k_single_audit_download",
                )
                st.warning(
                    gradient_shap_evidence_statement(wm_gradient_shap_evidence())
                    + " "
                    + ("과거 IG·Grad-CAM 방법 비교는 같은 배포 체크포인트로 수행했지만 "
                       "Gradient SHAP 검증이 아닙니다. "
                       if wm_xai_matches_deployment() else
                       "과거 IG·Grad-CAM 방법 비교는 탐색 체크포인트 결과이며 Gradient SHAP "
                       "검증이 아닙니다. ")
                    + "설명 지도는 모델의 위치 민감도만 보여줍니다. 물리적 불량 원인이나 공정 "
                    "조정의 인과 근거로 사용하지 말고 실제 판정은 전문가가 검토해야 합니다."
                )
            except Exception as error:
                st.error(f"CNN 진단에 실패했습니다: {error}")

    def render_wm_batch() -> None:
        st.subheader("생산 웨이퍼 배치 진단")
        st.caption(
            "여러 웨이퍼를 한 번에 분류하고 낮은 신뢰도 결과를 전문가 검토 목록의 "
            "위쪽에 배치합니다. Gradient SHAP·Grad-CAM 상세 확인은 단일 진단 탭에서 진행하세요."
        )
        demo_manifest_path = WM_DEMO_DIR / "manifest.json"
        demo_manifest = load_wafer_demo_manifest(
            str(demo_manifest_path), demo_manifest_path.stat().st_mtime_ns
        )
        demo_count = len(demo_manifest["samples"])
        demo_batch_label = f"실제 test 예시 {demo_count}개"
        batch_source = st.radio(
            "배치 데이터",
            (demo_batch_label, "여러 파일 업로드"),
            horizontal=True,
            key="wm811k_batch_source",
        )
        batch_uploads = None
        if batch_source == "여러 파일 업로드":
            batch_uploads = st.file_uploader(
                "웨이퍼 맵 파일을 여러 개 선택하세요 (최대 100개)",
                type=("npy", "csv", "txt", "data"),
                accept_multiple_files=True,
                key="wm811k_batch_upload",
            )
            if batch_uploads and len(batch_uploads) > 100:
                st.error("한 번에 최대 100개까지만 진단할 수 있습니다.")
        else:
            st.info(
                f"모델 선택이 끝난 뒤 고정한 test 예시 {demo_count}개(클래스별 4~5개)로 "
                "배치 화면을 시연합니다. 이 결과는 새로운 성능 평가가 아닙니다."
            )

        can_run_batch = batch_source == demo_batch_label or bool(batch_uploads)
        if st.button(
            "일괄 진단 실행",
            type="primary",
            disabled=not can_run_batch
            or bool(batch_uploads and len(batch_uploads) > 100),
            key="wm811k_batch_run",
        ):
            batch_items: list[dict[str, object]] = []
            batch_errors: list[dict[str, str]] = []
            if batch_source == demo_batch_label:
                for metadata in demo_manifest["samples"]:
                    sample_path = WM_DEMO_DIR / metadata["npy_file"]
                    batch_items.append(
                        {
                            "파일명": sample_path.name,
                            "실제 클래스": metadata["true_class"],
                            "wafer": parse_wafer_bytes(
                                sample_path.read_bytes(), sample_path.name
                            ),
                        }
                    )
            else:
                for uploaded_file in batch_uploads or []:
                    try:
                        batch_items.append(
                            {
                                "파일명": uploaded_file.name,
                                "실제 클래스": "-",
                                "wafer": parse_wafer_bytes(
                                    uploaded_file.getvalue(), uploaded_file.name
                                ),
                            }
                        )
                    except Exception as error:
                        batch_errors.append(
                            {"파일명": uploaded_file.name, "오류": str(error)}
                        )

            if batch_items:
                with st.spinner(f"웨이퍼 {len(batch_items)}개 일괄 진단 중..."):
                    model, class_names, _ = load_wafer_model(
                        str(WM_CHECKPOINT_PATH), WM_CHECKPOINT_PATH.stat().st_mtime_ns
                    )
                    batch_predictions = predict_wafer_batch(
                        model,
                        class_names,
                        [item["wafer"] for item in batch_items],
                        temperature=float(uncertainty_policy["temperature"]),
                    )
                    batch_ood = assess_wafer_ood_batch(
                        model,
                        class_names,
                        [item["wafer"] for item in batch_items],
                        ood_reference,
                    )

                confidence_threshold = float(uncertainty_policy["review_threshold"])
                rows = []
                for item, prediction, ood_item in zip(
                    batch_items, batch_predictions, batch_ood
                ):
                    predicted = str(prediction["predicted_label"])
                    confidence = float(prediction["confidence"])
                    ood_status = str(ood_item["ood_status"])
                    needs_review = (
                        confidence < confidence_threshold
                        or ood_status != "in_distribution"
                    )
                    recommendation = (
                        "자동판정 보류 · 입력 확인"
                        if ood_status == "out_of_distribution"
                        else "전문가 검토"
                        if needs_review
                        else "자동 분류 가능"
                    )
                    shape_features = canonical_wafer_shape_features(item["wafer"])
                    cause_guidance = build_wafer_cause_guidance(
                        predicted,
                        shape_features,
                        ood_status=ood_status,
                    )
                    rows.append(
                        {
                            "파일명": item["파일명"],
                            "입력 SHA-256": canonical_wafer_sha256(item["wafer"]),
                            "실제 클래스": item["실제 클래스"],
                            "판정": "정상" if predicted == "none" else "불량",
                            "예측 클래스": predicted,
                            "보정 신뢰도": confidence,
                            "입력 신뢰도": WM_OOD_LABELS[ood_status],
                            "OOD 점수": float(ood_item["ood_score"]),
                            "가까운 기준 클래스": ood_item["nearest_reference_class"],
                            "검토 필요": "예" if needs_review else "아니오",
                            "처리 권고": recommendation,
                            "원인 해석": cause_guidance["interpretation"],
                            "주요 원인 후보": " | ".join(
                                candidate["cause"]
                                for candidate in cause_guidance["candidates"]
                            ),
                            "우선 확인 항목": " | ".join(
                                candidate["check"]
                                for candidate in cause_guidance["candidates"]
                            ),
                            "형상 근거": " · ".join(
                                cause_guidance["observations"]
                            ),
                            "입력 품질 경고": " · ".join(
                                ood_item["geometry_warnings"]
                            ),
                            "결함 die 비율": float(shape_features["defect_ratio"]),
                            "8-이웃 연결 영역 수": int(
                                shape_features["component_count_8"]
                            ),
                            "최대 연결 영역 점유율": float(
                                shape_features["largest_component_fraction"]
                            ),
                            "가장자리 결함 구성비": float(
                                shape_features["edge_defect_fraction"]
                            ),
                            "웨이퍼 경계 결함률": float(
                                shape_features["boundary_defect_coverage"]
                            ),
                            "결함 분포 범위": float(
                                shape_features["defect_bbox_fraction"]
                            ),
                        }
                    )
                batch_result = pd.DataFrame(rows)
                batch_monitoring = None
                if batch_source == "여러 파일 업로드" and monitoring_reference is not None:
                    batch_monitoring = assess_batch_drift(
                        predicted_labels=[
                            str(item["predicted_label"]) for item in batch_predictions
                        ],
                        ood_statuses=[str(item["ood_status"]) for item in batch_ood],
                        ood_scores=[float(item["ood_score"]) for item in batch_ood],
                        reference=monitoring_reference,
                    )
                    if batch_monitoring["status"] == "HOLD":
                        batch_result["검토 필요"] = "예"
                        batch_result["처리 권고"] = (
                            "배치 자동판정 보류 · 분포 변화 검토"
                        )
                batch_result["_검토순서"] = batch_result["검토 필요"].eq("예")
                batch_result = batch_result.sort_values(
                    ["_검토순서", "보정 신뢰도"], ascending=[False, True]
                ).drop(columns="_검토순서")

                batch_metrics = st.columns(5)
                batch_metrics[0].metric("처리 완료", f"{len(batch_result):,}개")
                batch_metrics[1].metric(
                    "불량 판정", f"{int(batch_result['판정'].eq('불량').sum()):,}개"
                )
                batch_metrics[2].metric(
                    "전문가 검토",
                    f"{int(batch_result['검토 필요'].eq('예').sum()):,}개",
                )
                batch_metrics[3].metric(
                    "OOD 판정 보류",
                    f"{int(batch_result['입력 신뢰도'].eq(WM_OOD_LABELS['out_of_distribution']).sum()):,}개",
                )
                batch_metrics[4].metric(
                    "평균 신뢰도", f"{batch_result['보정 신뢰도'].mean():.2%}"
                )
                if batch_source == demo_batch_label:
                    st.caption(
                        f"이 {demo_count}개는 클래스별로 선별한 시연 표본이므로 생산 배치 "
                        "분포 감시에 사용하지 않습니다. 분포 감시는 실제 업로드 배치에만 "
                        "적용합니다."
                    )
                elif monitoring_reference is None:
                    st.warning("배치 모니터링 기준이 없어 분포 변화를 판정하지 못했습니다.")
                elif batch_monitoring is not None:
                    st.markdown("#### 배치 분포 변화 감지")
                    monitoring_metrics = batch_monitoring["metrics"]
                    monitoring_columns = st.columns(4)
                    monitoring_columns[0].metric(
                        "배치 상태", batch_monitoring["status"]
                    )
                    monitoring_columns[1].metric(
                        "클래스 구성 JSD",
                        f"{monitoring_metrics['predicted_class_js_divergence']:.4f}",
                        f"기준 {monitoring_metrics['predicted_class_js_threshold']:.4f}",
                    )
                    monitoring_columns[2].metric(
                        "검토·OOD 비율",
                        f"{monitoring_metrics['review_or_ood_rate']:.1%}",
                        f"기준 {monitoring_metrics['review_or_ood_rate_threshold']:.1%}",
                    )
                    monitoring_columns[3].metric(
                        "Severe OOD 비율",
                        f"{monitoring_metrics['severe_ood_rate']:.1%}",
                        f"기준 {monitoring_metrics['severe_ood_rate_threshold']:.1%}",
                    )
                    if batch_monitoring["status"] == "HOLD":
                        st.error(
                            "배치 분포 변화 감지: 개별 예측이 정상처럼 보여도 전체 배치 "
                            "자동판정을 보류하고 입력 형식·장비·lot 변화를 검토하세요."
                        )
                    elif batch_monitoring["status"] == "STABLE":
                        st.success(
                            "배치 분포가 validation 기준의 99% 모의 변동 범위 안에 있습니다."
                        )
                    else:
                        st.info(
                            f"최소 {batch_monitoring['minimum_batch_size']}개 미만이므로 "
                            "배치 분포는 판정하지 않고 개별 OOD·신뢰도 기준만 적용합니다."
                        )
                    st.caption(
                        " · ".join(str(reason) for reason in batch_monitoring["reasons"])
                        + " · 정답 없는 감시이므로 성능 저하 확정이나 자동 재학습에 사용하지 않습니다."
                    )
                    st.download_button(
                        "배치 모니터링 JSON 다운로드",
                        data=json.dumps(
                            batch_monitoring, ensure_ascii=False, indent=2
                        ).encode("utf-8"),
                        file_name="wm811k_batch_monitoring.json",
                        mime="application/json",
                        key="wm811k_batch_monitoring_download",
                    )
                batch_display_columns = [
                    "파일명",
                    "예측 클래스",
                    "보정 신뢰도",
                    "입력 신뢰도",
                    "처리 권고",
                    "주요 원인 후보",
                    "우선 확인 항목",
                ]
                st.dataframe(
                    batch_result[batch_display_columns],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "보정 신뢰도": st.column_config.NumberColumn(format="percent"),
                        "주요 원인 후보": st.column_config.TextColumn(width="large"),
                        "우선 확인 항목": st.column_config.TextColumn(width="large"),
                    },
                )
                csv_bytes = batch_result.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "배치 진단 결과 CSV 다운로드",
                    data=csv_bytes,
                    file_name="wm811k_batch_diagnosis.csv",
                    mime="text/csv",
                    key="wm811k_batch_download",
                )
                st.download_button(
                    "서식 적용 Excel 다운로드",
                    data=build_wafer_excel_report(batch_result),
                    file_name="wm811k_batch_diagnosis.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="wm811k_batch_excel_download",
                )
                st.download_button(
                    "배치 감사 HTML 보고서 다운로드",
                    data=build_wafer_audit_report(
                        source_name=batch_source,
                        results=batch_result,
                        artifact_hashes=artifact_hashes,
                    ),
                    file_name="wm811k_batch_audit_report.html",
                    mime="text/html",
                    key="wm811k_batch_audit_download",
                )
                st.warning(
                    "배치 결과는 연구·시연용 우선순위입니다. 저신뢰 결과와 신규 장비·"
                    "공정 데이터는 자동 확정하지 말고 전문가가 확인해야 합니다."
                )
            if batch_errors:
                st.error(f"형식 오류로 제외된 파일이 {len(batch_errors)}개 있습니다.")
                st.dataframe(pd.DataFrame(batch_errors), hide_index=True, width="stretch")

    def render_wm_guide() -> None:
        demo = artificial_demo_map()
        demo_csv = pd.DataFrame(demo).to_csv(index=False, header=False).encode("utf-8")
        st.download_button(
            "인공 입력 예시 CSV 다운로드",
            data=demo_csv,
            file_name="wm811k_artificial_demo.csv",
            mime="text/csv",
        )

        st.markdown(
            """
            - 한 파일에는 웨이퍼 맵 한 개만 넣습니다.
            - 값 `0`은 웨이퍼 외부, `1`은 정상 die, `2`는 결함 die입니다.
            - `.csv`는 헤더와 행 번호 없이 숫자 행렬만 사용합니다.
            - `.npy`는 pickle을 사용하지 않는 2차원 숫자 배열이어야 합니다.
            - 원본 크기는 달라도 되며 학습과 동일하게 64×64 최근접 보간을 적용합니다.
            - 보정 신뢰도가 97.50% 미만이면 전문가 검토 대상으로 표시합니다.
            - OOD 점수가 1.1764 이상이면 분포 경계 검토, 1.5755 이상이면 자동판정을 보류합니다.
            - 웨이퍼 면적·결함 die 비율이 train 범위를 벗어나도 전문가 검토로 전환합니다.
            - 불량 원인 후보는 예측 패턴에 따른 점검 순서이며 원인 확률이나 인과 판정이 아닙니다.
            - 실제 원인은 장비·레시피·센서·유지보수·재측정 기록을 함께 확인해야 합니다.

            업로드 형식이 맞더라도 촬영·검사 장비, 값 정의, 전처리 방식이 학습 데이터와
            다르면 예측을 신뢰할 수 없습니다.
            """
        )

    wm_renderers = {
        "evaluation": render_wm_performance_summary,
        "evaluation_safety": render_wm_safety_summary,
        "evaluation_xai": render_wm_xai_summary,
        "independent_validation": render_wm_independent_page,
        "research_log": render_wm_research_log,
        "diagnosis": render_wm_diagnosis,
        "batch": render_wm_batch,
        "guide": render_wm_guide,
    }
    if not dashboard_access.section_allowed(
        wm_section, dashboard_access.WM_ADMIN_SECTIONS, ACCESS
    ):
        st.error("관리자 권한이 필요한 화면입니다.")
        return
    wm_renderers[wm_section]()


# Website navigation. Each page is a marker file under dashboard_ui/site_pages;
# this script renders the page st.navigation returns, so the SECOM analysis
# below keeps running as top-level code. Diagnosis pages need a signed-in user
# when SHAPGPT_REQUIRE_LOGIN=1; the administrator page exists only for users
# the access rules mark as administrators, so /admin falls back to the home
# page for everyone else.
REPOSITORY_URL = os.environ.get("SHAPGPT_REPOSITORY_URL", "").strip()
if not REPOSITORY_URL.startswith(("https://", "http://")):
    REPOSITORY_URL = "https://github.com/UJUNGKIM/Semiconductor-Ai-Project"
SITE_PAGES_DIR = PROJECT_DIR / "dashboard_ui" / "site_pages"
home_page = st.Page(SITE_PAGES_DIR / "home.py", title="홈", default=True)
secom_page = st.Page(SITE_PAGES_DIR / "secom.py", title="SECOM 진단", url_path="secom")
wm_page = st.Page(SITE_PAGES_DIR / "wm811k.py", title="WM-811K 진단", url_path="wm811k")
project_page = st.Page(
    SITE_PAGES_DIR / "project.py", title="프로젝트·검증", url_path="project"
)
site_pages = [home_page, secom_page, wm_page, project_page]
account_page = None
if ACCESS.login_required:
    account_page = st.Page(
        SITE_PAGES_DIR / "account.py",
        title="계정" if ACCESS.logged_in else "로그인",
        url_path="account",
    )
    site_pages.append(account_page)
admin_page = None
if ACCESS.is_admin:
    admin_page = st.Page(SITE_PAGES_DIR / "admin.py", title="관리자", url_path="admin")
    site_pages.append(admin_page)
site.render_logo()
current_page = st.navigation(site_pages, position="top")
current_page.run()

auth_ready = dashboard_access.auth_configured()
site_links = {"secom": secom_page, "wm811k": wm_page, "project": project_page}
# SECOM model and augmentation validation moved from the diagnosis page to the
# project page; old links to those sections open their new location.
SECOM_MOVED_SECTIONS = {
    "validation": "secom_validation",
    "augmentation": "secom_augmentation",
}
moved_secom_section = SECOM_MOVED_SECTIONS.get(query_value("section", ""))
site.render_account_strip(ACCESS)
if current_page is home_page:
    legacy_module = query_value("module", "")
    if legacy_module == "secom" and moved_secom_section is not None:
        st.switch_page(project_page, query_params={"section": moved_secom_section})
    legacy_page = {"secom": secom_page, "wm811k": wm_page}.get(legacy_module)
    if legacy_page is not None:
        # Old ?module=...&section=... links open the matching page and section.
        st.switch_page(legacy_page, query_params=st.query_params.to_dict())
    site.render_home(
        access=ACCESS,
        auth_ready=auth_ready,
        evidence=collect_site_evidence(),
        pages=site_links,
    )
    site.render_footer(repository_url=REPOSITORY_URL, project_page=project_page)
    st.stop()
if current_page is secom_page and moved_secom_section is not None:
    st.switch_page(project_page, query_params={"section": moved_secom_section})
if current_page is project_page:
    site.render_project(
        evidence=collect_site_evidence(),
        pages=site_links,
        area_renderers={
            "secom_validation": render_secom_validation,
            "secom_augmentation": render_secom_augmentation,
            "xai": render_project_xai_evidence,
            "wm811k": render_project_wm_evidence,
        },
    )
    site.render_footer(repository_url=REPOSITORY_URL, project_page=project_page)
    st.stop()
if account_page is not None and current_page is account_page:
    site.render_account(
        access=ACCESS,
        auth_ready=auth_ready,
        provider_label=(
            "Google 계정(OIDC)"
            if dashboard_access.configured_login_provider() == dashboard_access.GOOGLE_PROVIDER
            else "OIDC"
        ),
        admin_page=admin_page,
    )
    site.render_footer(repository_url=REPOSITORY_URL, project_page=project_page)
    st.stop()
if admin_page is not None and current_page is admin_page:
    admin_emails, admin_subjects = dashboard_access.admin_allowlists(
        os.environ, dashboard_access.streamlit_secrets()
    )
    site.render_admin_hub(
        access=ACCESS,
        auth_ready=auth_ready,
        allowlist_counts=(len(admin_emails), len(admin_subjects)),
        links=(
            site.AdminLink(
                "SECOM 운영 모니터링",
                "배치 운영 기록과 안전 신호 추세, 변조 감지형 감사 원장, 실제 검수 기반 "
                "성능 감시를 확인합니다.",
                secom_page,
                {"module": "secom", "section": "monitoring"},
            ),
            site.AdminLink(
                "SECOM 릴리스 준비도",
                "시연·생산 배포 판정과 외부 검증 준비, 센서 사전, 모델 무결성 점검을 "
                "확인합니다.",
                secom_page,
                {"module": "secom", "section": "release"},
            ),
            site.AdminLink(
                "WM-811K 전체 실험 기록",
                "학습 전략 선택, 다중 시드 재현성, lot 일반화, OOD·강건성 실험 기록 전체를 "
                "봅니다.",
                wm_page,
                {"module": "wm811k", "section": "research_log"},
            ),
        ),
    )
    site.render_footer(repository_url=REPOSITORY_URL, project_page=project_page)
    st.stop()
if ACCESS.login_required and not ACCESS.logged_in:
    site.render_login_gate(auth_ready=auth_ready)
    st.stop()
if current_page is wm_page:
    render_wm811k_dashboard()
    site.render_footer(repository_url=REPOSITORY_URL, project_page=project_page)
    st.stop()


render_dashboard_header(
    "SECOM 공정 센서 SHAP 진단",
    "CatBoost 균형형 후보와 XGBoost 고재현율 후보를 나란히 확인합니다. "
    "모델 점수와 SHAP 설명은 공정 엔지니어의 판단을 보조하는 용도입니다.",
)
render_unified_xai_frame(
    explanation="TreeSHAP 센서 기여도로 위험 방향 확인",
    input_limit="결측률·범위 이탈률·OOD 확인",
)
st.warning(
    "데이터 한계: 이 SECOM 공개 데이터는 센서명·단위·공정 시점이 "
    "익명화되었고 결측과 클래스 불균형이 큽니다. 성능 변동과 원인 설명의 "
    "한계는 모델보다 원천 데이터의 제약에서 비롯됩니다."
)

with st.sidebar:
    st.header("진단 설정")
    with st.form("secom_diagnosis_form", border=False):
        source = st.radio("데이터 소스", ("프로젝트 샘플", "파일 업로드"))
        profile_label = st.selectbox("운영 임계값", tuple(PROFILE_LABELS))
        top_n_input = st.slider("표시할 SHAP 변수 수", 3, 15, 8)
        if source == "프로젝트 샘플":
            sample_limit = st.slider("샘플 행 수", 5, 300, 100, step=5)
            uploaded = None
        else:
            sample_limit = None
            uploaded = st.file_uploader(
                "SECOM 호환 센서 파일", type=("data", "txt", "csv")
            )
        diagnosis_submitted = st.form_submit_button(
            "진단 실행", type="primary", width="stretch"
        )
        st.caption("설정 변경은 이 버튼을 누른 뒤 결과에 반영됩니다.")
    st.download_button(
        "CSV 입력 양식 다운로드",
        data=sample_csv_template(),
        file_name="secom_input_template.csv",
        mime="text/csv",
        width="stretch",
    )
    with st.expander("입력 형식 안내"):
        st.caption("공백 파일: 헤더 없이 센서값 590개를 입력합니다.")
        st.caption("CSV: feature_0~feature_589 헤더를 사용합니다.")
        st.caption("ID·timestamp·label 부가 열은 자동으로 제외합니다.")

try:
    loading_manifest_path = RELEASE_READINESS_DIR / "release_manifest.json"
    registry_path = MODEL_FAILOVER_DIR / "model_registry.json"
    if not loading_manifest_path.is_file() or not registry_path.is_file():
        raise FileNotFoundError("릴리스 manifest 또는 모델 레지스트리가 없습니다.")
    loading_manifest = json.loads(loading_manifest_path.read_text(encoding="utf-8"))
    loading_hashes = loading_manifest["component_hashes"]
    bundles = load_bundles(
        str(registry_path),
        registry_path.stat().st_mtime_ns,
        loading_hashes["model_registry"],
    )
except Exception as error:
    st.error(f"모델을 불러오지 못했습니다: {error}")
    st.stop()

run_requested = diagnosis_submitted or "secom_diagnosis_run" not in st.session_state
if run_requested:
    try:
        if source == "프로젝트 샘플":
            X_raw = load_sample_rows(sample_limit)
            source_name = f"secom.data 앞부분 {len(X_raw)}행"
            input_info = {
                "format": "공백 구분 · 헤더 없음",
                "rows": len(X_raw),
                "sensor_columns": X_raw.shape[1],
                "ignored_columns": [],
            }
        elif uploaded is not None:
            X_raw, input_info = parse_sensor_bytes(uploaded.getvalue())
            source_name = uploaded.name
        else:
            st.info("진단할 파일을 선택한 뒤 **진단 실행**을 눌러주세요.")
            st.stop()

        profile = PROFILE_LABELS[profile_label]
        with st.spinner("두 모델로 진단 중..."):
            results, prepared = run_predictions(X_raw, bundles, profile)

        advanced_path = ADVANCED_DIAGNOSTICS_DIR / "advanced_reference.joblib"
        advanced_reference = None
        ood_results = None
        review_queue = None
        batch_gate_result = None
        if advanced_path.is_file():
            advanced_reference = load_advanced_reference(
                str(advanced_path),
                advanced_path.stat().st_mtime_ns,
                loading_hashes["advanced_reference"],
            )
            ood_results = assess_ood(X_raw, advanced_reference)
            review_queue = build_review_queue(results, ood_results)
            gate_reference_path = BATCH_GATE_DIR / "batch_gate_reference.json"
            if gate_reference_path.is_file() and profile == "balanced_f2":
                gate_reference = json.loads(
                    gate_reference_path.read_text(encoding="utf-8")
                )
                batch_gate_result = evaluate_batch_gate(
                    results, ood_results, gate_reference
                )

        st.session_state["secom_diagnosis_run"] = {
            "run_id": f"secom-{pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%dT%H%M%S%f')}",
            "X_raw": X_raw,
            "input_info": input_info,
            "source_name": source_name,
            "profile_label": profile_label,
            "profile": profile,
            "top_n": top_n_input,
            "results": results,
            "prepared": prepared,
            "advanced_reference": advanced_reference,
            "ood_results": ood_results,
            "review_queue": review_queue,
            "batch_gate_result": batch_gate_result,
            "ran_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    except Exception as error:
        st.error(f"진단을 실행하지 못했습니다: {error}")
        st.markdown("**확인할 항목**")
        for item in diagnosis_error_guidance(str(error)):
            st.markdown(f"- {item}")
        st.caption("왼쪽의 CSV 입력 양식을 내려받아 원본 파일과 바로 비교할 수 있습니다.")
        if "secom_diagnosis_run" not in st.session_state:
            st.stop()

diagnosis_run = st.session_state["secom_diagnosis_run"]
if "run_id" not in diagnosis_run:
    diagnosis_run["run_id"] = (
        f"secom-{pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%dT%H%M%S%f')}"
    )
diagnosis_run_id = diagnosis_run["run_id"]
X_raw = diagnosis_run["X_raw"]
input_info = diagnosis_run["input_info"]
source_name = diagnosis_run["source_name"]
profile_label = diagnosis_run["profile_label"]
profile = diagnosis_run["profile"]
top_n = diagnosis_run["top_n"]
results = diagnosis_run["results"]
prepared = diagnosis_run["prepared"]
advanced_reference = diagnosis_run["advanced_reference"]
ood_results = diagnosis_run["ood_results"]
review_queue = diagnosis_run["review_queue"]
batch_gate_result = diagnosis_run["batch_gate_result"]
diagnosis_ran_at = diagnosis_run["ran_at"]

st.sidebar.success(
    f"최근 실행 · {input_info['rows']}행 × 센서 {input_info['sensor_columns']}개"
)
st.sidebar.caption(f"인식 형식: {input_info['format']}")
st.sidebar.caption(f"실행 시각: {diagnosis_ran_at}")
if input_info["ignored_columns"]:
    st.sidebar.caption(
        "자동 제외한 부가 열: " + ", ".join(input_info["ignored_columns"])
    )

render_run_context(
    source_name=source_name,
    rows=input_info["rows"],
    profile_label=profile_label,
    ran_at=diagnosis_ran_at,
)
if batch_gate_result is not None:
    if batch_gate_result["status"] == "STOP":
        st.error(
            "배치 안전 차단기 STOP · 자동판정을 중단했습니다. 입력 센서와 측정계를 "
            "확인하고 배치 전체를 전문가 검토하세요."
        )
    elif batch_gate_result["status"] == "CAUTION":
        st.warning("배치 안전 차단기 CAUTION · 자동 처리 전 추가 확인이 필요합니다.")
both_defect = int((results["종합 판정"] == "두 모델 모두 불량").sum())
one_defect = int(
    results["종합 판정"].isin(("CatBoost만 불량", "XGBoost만 불량")).sum()
)
both_normal = int((results["종합 판정"] == "두 모델 모두 정상").sum())
agreement = float(
    (
        (results["CatBoost 판정"] == results["XGBoost 판정"])
    ).mean()
)

# Model and augmentation validation live on the project page
# (SECOM_MOVED_SECTIONS); the diagnosis page keeps the tools for the current run.
secom_navigation = {
    "진단": (
        ("검토 대기열", "diagnosis"),
        ("개별 SHAP 설명", "explanation"),
        ("안전·What-if", "safety"),
    ),
    "안내": (("통합 분석 흐름", "process"), ("사용 안내", "guide")),
}
if ACCESS.is_admin:
    secom_navigation["관리자"] = (
        ("운영 모니터링", "monitoring"),
        ("릴리스 준비도", "release"),
    )
elif query_value("section", "") in dashboard_access.SECOM_ADMIN_SECTIONS:
    st.warning("관리자 권한이 필요한 화면입니다. 기본 화면으로 이동했습니다.")
with st.container(key="secom_navigation"):
    secom_section = render_section_navigation(
        secom_navigation,
        module="secom",
        default_section="diagnosis",
        key_prefix="secom_nav",
    )

if secom_section == "diagnosis":
    metric_columns = st.columns(4)
    with metric_columns[0]:
        with st.container(key="secom_metric_primary"):
            st.metric(
                "두 모델 모두 불량",
                both_defect,
                help="CatBoost와 XGBoost가 모두 불량으로 판정한 행입니다.",
            )
    with metric_columns[1]:
        with st.container(key="secom_metric_attention"):
            st.metric(
                "한 모델만 불량",
                one_defect,
                help="두 모델의 판정이 달라 전문가 확인이 필요한 행입니다.",
            )
    with metric_columns[2]:
        with st.container(key="secom_metric_normal"):
            st.metric(
                "두 모델 모두 정상",
                both_normal,
                help="두 모델이 모두 정상으로 판정한 행입니다. OOD 여부는 별도로 확인합니다.",
            )
    with metric_columns[3]:
        with st.container(key="secom_metric_agreement"):
            st.metric(
                "모델 판정 일치율",
                f"{agreement:.1%}",
                help="전체 행에서 두 모델의 정상·불량 판정이 같았던 비율입니다.",
            )

def open_row_explanation(row: int) -> None:
    """on_click callback: runs before the rerun builds any widget, so the row
    selector's key can still be set and the section switch lands cleanly."""
    st.session_state["secom_explanation_row"] = row
    navigate_to_section(module="secom", section="explanation", row=row)


def render_secom_diagnosis() -> None:
    queue = prepare_review_queue(
        review_queue.copy() if review_queue is not None else results.copy()
    )
    review_mask = queue["처리 권고"] != "자동 처리 후보"
    filter_options = review_filter_options(queue)
    filter_labels = list(filter_options)
    if st.session_state.get("secom_review_filter") not in filter_labels:
        st.session_state["secom_review_filter"] = filter_labels[0]

    filter_col, search_col = st.columns((3.2, 1))
    with filter_col:
        filter_label = st.segmented_control(
            "표시 범위",
            filter_labels,
            key="secom_review_filter",
            width="stretch",
        ) or filter_labels[0]
    with search_col:
        search_term = st.text_input(
            "행 검색",
            placeholder="행 번호·판정·권고",
            key="secom_review_search",
        )
    filtered = filter_review_queue(
        queue, filter_options[filter_label], search=search_term
    )
    if filtered.empty:
        st.info("조건과 검색어에 해당하는 행이 없습니다. 검색어를 지우거나 범위를 바꿔주세요.")
    st.caption(
        f"검토 대상 {int(review_mask.sum()):,}개 · "
        f"자동판정 보류 {int(queue['처리 권고'].str.contains('보류', na=False).sum()):,}개 · "
        f"현재 표시 {len(filtered):,}개 · "
        "행 선택 시 상세·SHAP 근거·검토 도구가 표 아래에 표시됩니다. "
        "● 긴급 · ◆ 추가 확인 · ○ 일반"
    )

    display_columns = [
        column
        for column in (
            "행 번호",
            "우선도",
            "처리 권고",
            "종합 판정",
            "입력 신뢰도",
            "CatBoost 점수",
            "XGBoost 점수",
            "검토 우선순위 점수",
        )
        if column in filtered.columns
    ]
    selected_review_row = st.session_state.get("secom_selected_row")
    table_event = st.dataframe(
        filtered[display_columns] if not filtered.empty else filtered,
        width="stretch",
        height=430,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="secom_review_table",
        column_config={
            "CatBoost 점수": st.column_config.NumberColumn(format="%.4f"),
            "XGBoost 점수": st.column_config.NumberColumn(format="%.4f"),
            "검토 우선순위 점수": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    selected_positions = table_event.selection.rows
    if selected_positions:
        selected_review_row = int(filtered.iloc[selected_positions[0]]["행 번호"])
        st.session_state["secom_selected_row"] = selected_review_row

    if selected_review_row is not None and selected_review_row in queue["행 번호"].values:
        selected_review_row = int(selected_review_row)
        selected = queue.loc[queue["행 번호"] == selected_review_row].iloc[0]
        st.markdown(
            f"""
            <div class="review-detail-card">
                <h4>{selected_review_row}번 행 · {selected['우선도']}</h4>
                <div class="review-detail-grid">
                    <span><b>처리 권고</b>{selected['처리 권고']}</span>
                    <span><b>종합 판정</b>{selected['종합 판정']}</span>
                    <span><b>입력 신뢰도</b>{selected.get('입력 신뢰도', '확인 정보 없음')}</span>
                    <span><b>모델 점수</b>CatBoost {selected['CatBoost 점수']:.4f} · XGBoost {selected['XGBoost 점수']:.4f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        evidence_col, review_col = st.columns((1.35, 1), gap="large")
        with evidence_col:
            st.markdown("#### 주요 SHAP 근거")
            preview_parts = []
            for model_name in MODEL_NAMES:
                preview = explain_selected_row(
                    model_name,
                    prepared[model_name],
                    selected_review_row,
                    min(3, top_n),
                ).copy()
                preview["근거"] = model_name + " · " + preview["변수"].astype(str)
                preview_parts.append(preview)
            preview_frame = pd.concat(preview_parts, ignore_index=True)
            preview_chart = px.bar(
                preview_frame,
                x="SHAP 기여도",
                y="근거",
                orientation="h",
                color="방향",
                color_discrete_map={"불량 쪽": "#4b5563", "정상 쪽": "#a3a8b0"},
                hover_data=["입력값(결측 대체 후)"],
            )
            preview_chart.add_vline(x=0, line_width=1, line_color="#d1d5db")
            preview_chart.update_layout(
                height=275,
                margin=dict(t=10, l=5, r=5, b=5),
                showlegend=False,
                yaxis_title=None,
            )
            st.plotly_chart(preview_chart, width="stretch", key="selected_row_shap_preview")
            st.button(
                "전체 SHAP 근거 확인",
                type="primary",
                width="stretch",
                key="open_selected_shap",
                on_click=open_row_explanation,
                args=(selected_review_row,),
            )
        with review_col:
            st.markdown("#### 검토 기록")
            with st.form("inline_review_form", border=True):
                review_decision = st.selectbox(
                    "검토 결정",
                    ("확인 필요", "정상 확인", "불량 확인", "측정 재요청", "판정 보류"),
                )
                reviewer_id = st.text_input("검토자 ID", value="engineer_01")
                review_note = st.text_input("메모", placeholder="장비 상태·재측정 사유")
                save_review = st.form_submit_button("검토 기록 저장", width="stretch")
        if save_review:
            try:
                append_store_event(
                    RUNTIME_AUDIT_DATABASE,
                    event_type="MANUAL_NOTE",
                    actor=reviewer_id.strip() or "미입력",
                    batch_id=diagnosis_run_id,
                    payload={
                        "row_number": selected_review_row,
                        "decision": review_decision,
                        "note": review_note.strip()[:500],
                        "consensus": str(selected["종합 판정"]),
                        "input_confidence": str(selected.get("입력 신뢰도", "")),
                        "profile": profile,
                    },
                )
                st.success("검토 기록을 로컬 append-only 감사 저장소에 저장했습니다.")
            except Exception as error:
                st.error(f"검토 기록을 저장하지 못했습니다: {error}")

    counts = (
        results["종합 판정"]
        .value_counts()
        .rename_axis("판정")
        .reset_index(name="행 수")
    )
    chart = px.bar(
        counts,
        x="판정",
        y="행 수",
        color="판정",
        title="전체 판정 분포",
    )
    chart.update_layout(showlegend=False, height=300, margin=dict(t=55, l=20, r=20, b=20))
    st.plotly_chart(chart, width="stretch")

    profile_comparison = build_profile_comparison(
        prepared["CatBoost"]["probability"],
        prepared["XGBoost"]["probability"],
        bundles,
        PROFILE_LABELS,
    )
    with st.expander("운영 임계값별 결과 비교"):
        st.caption("모델을 다시 실행하지 않고 저장된 점수에 각 운영 임계값을 적용한 결과입니다.")
        st.dataframe(
            profile_comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "판정 일치율": st.column_config.NumberColumn(format="percent")
            },
        )
        comparison_long = profile_comparison.melt(
            id_vars="운영 기준",
            value_vars=("두 모델 불량", "모델 불일치", "두 모델 정상"),
            var_name="판정",
            value_name="행 수",
        )
        comparison_chart = px.bar(
            comparison_long,
            x="운영 기준",
            y="행 수",
            color="판정",
            barmode="group",
            title="임계값 변경 시 판정 구성",
        )
        comparison_chart.update_layout(height=330, margin=dict(t=55, l=20, r=20, b=20))
        st.plotly_chart(comparison_chart, width="stretch")

    with st.expander("실행·검토 기록"):
        result_digest = hashlib.sha256(
            pd.util.hash_pandas_object(results, index=True).values.tobytes()
        ).hexdigest()
        try:
            audit_ledger = read_store(RUNTIME_AUDIT_DATABASE)
            diagnosis_history = audit_events_frame(audit_ledger, "BATCH_DIAGNOSIS")
            review_history = audit_events_frame(audit_ledger, "MANUAL_NOTE")
            already_saved = (
                not diagnosis_history.empty
                and diagnosis_run_id in diagnosis_history["batch_id"].astype(str).values
            )
            if st.button(
                "현재 실행 기록 저장" if not already_saved else "현재 실행 저장 완료",
                disabled=already_saved,
                key="save_current_diagnosis_run",
            ):
                append_store_event(
                    RUNTIME_AUDIT_DATABASE,
                    event_type="BATCH_DIAGNOSIS",
                    actor="dashboard",
                    batch_id=diagnosis_run_id,
                    payload={
                        "ran_at": diagnosis_ran_at,
                        "source_name": Path(source_name).name[:120],
                        "profile": profile,
                        "profile_label": profile_label,
                        "row_count": len(results),
                        "both_defect": both_defect,
                        "one_defect": one_defect,
                        "both_normal": both_normal,
                        "agreement": agreement,
                        "review_target_count": int(review_mask.sum()),
                        "result_digest": result_digest,
                    },
                )
                st.success("현재 실행의 집계값을 로컬 감사 저장소에 저장했습니다.")
                audit_ledger = read_store(RUNTIME_AUDIT_DATABASE)
                diagnosis_history = audit_events_frame(audit_ledger, "BATCH_DIAGNOSIS")
                review_history = audit_events_frame(audit_ledger, "MANUAL_NOTE")

            verification = verify_store(RUNTIME_AUDIT_DATABASE)
            st.caption(
                f"SQLite 무결성 {verification['sqlite_integrity_check']} · "
                f"해시 체인 {verification['entry_count']:,}개 이벤트 · 원본 센서값 저장 안 함"
            )
            if not diagnosis_history.empty:
                history_columns = [
                    column
                    for column in (
                        "recorded_at",
                        "batch_id",
                        "profile_label",
                        "row_count",
                        "both_defect",
                        "one_defect",
                        "both_normal",
                        "agreement",
                    )
                    if column in diagnosis_history
                ]
                st.markdown("**최근 진단 실행**")
                st.dataframe(
                    diagnosis_history.sort_values("sequence", ascending=False)
                    .head(10)[history_columns]
                    .rename(
                        columns={
                            "recorded_at": "저장 시각",
                            "batch_id": "실행 ID",
                            "profile_label": "운영 기준",
                            "row_count": "행 수",
                            "both_defect": "두 모델 불량",
                            "one_defect": "모델 불일치",
                            "both_normal": "두 모델 정상",
                            "agreement": "일치율",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                    column_config={"일치율": st.column_config.NumberColumn(format="percent")},
                )
                if len(diagnosis_history) >= 2:
                    ordered_history = diagnosis_history.sort_values("sequence")
                    previous_run = ordered_history.iloc[-2]
                    latest_run = ordered_history.iloc[-1]
                    delta_columns = st.columns(3)
                    delta_columns[0].metric(
                        "최근 두 모델 불량",
                        int(latest_run["both_defect"]),
                        delta=int(latest_run["both_defect"] - previous_run["both_defect"]),
                        delta_color="inverse",
                    )
                    delta_columns[1].metric(
                        "최근 모델 불일치",
                        int(latest_run["one_defect"]),
                        delta=int(latest_run["one_defect"] - previous_run["one_defect"]),
                        delta_color="inverse",
                    )
                    delta_columns[2].metric(
                        "최근 판정 일치율",
                        f"{float(latest_run['agreement']):.1%}",
                        delta=f"{float(latest_run['agreement'] - previous_run['agreement']):.1%}",
                    )
            else:
                st.info("저장된 실행이 없습니다. 현재 실행 기록 저장을 누르면 비교 이력이 시작됩니다.")
            if not review_history.empty:
                st.markdown("**최근 전문가 검토**")
                review_columns = [
                    column
                    for column in (
                        "recorded_at",
                        "batch_id",
                        "row_number",
                        "actor",
                        "decision",
                        "note",
                    )
                    if column in review_history
                ]
                st.dataframe(
                    review_history.sort_values("sequence", ascending=False)
                    .head(20)[review_columns]
                    .rename(
                        columns={
                            "recorded_at": "기록 시각",
                            "batch_id": "실행 ID",
                            "row_number": "행 번호",
                            "actor": "검토자",
                            "decision": "결정",
                            "note": "메모",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
        except Exception as error:
            st.warning(f"로컬 감사 저장소를 사용할 수 없습니다: {error}")

    html_report = None
    if advanced_reference is not None and ood_results is not None and review_queue is not None:
        html_report = build_html_report(
            source_name=source_name,
            profile_label=profile_label,
            predictions=results,
            ood=ood_results,
            review_queue=review_queue,
            model_hashes=advanced_reference["model_hashes"],
        )
    diagnosis_bundle = build_diagnosis_bundle(
        results=results.drop(columns="우선 확인 점수", errors="ignore"),
        review_queue=review_queue,
        ood_results=ood_results,
        html_report=html_report,
        metadata={
            "run_id": diagnosis_run_id,
            "source_name": source_name,
            "profile_label": profile_label,
            "rows": len(results),
            "ran_at": diagnosis_ran_at,
            "batch_gate": batch_gate_result,
            "profile_comparison": profile_comparison.to_dict(orient="records"),
        },
    )
    download_col, secondary_col, filtered_col = st.columns(3)
    with download_col:
        st.download_button(
            "통합 진단 패키지 ZIP 다운로드",
            data=diagnosis_bundle,
            file_name="secom_diagnosis_package.zip",
            mime="application/zip",
            type="primary",
            width="stretch",
        )
    with secondary_col:
        csv_bytes = results.drop(columns="우선 확인 점수", errors="ignore").to_csv(
            index=False, encoding="utf-8-sig"
        ).encode("utf-8-sig")
        st.download_button(
            "결과표만 CSV 다운로드",
            data=csv_bytes,
            file_name="secom_dual_model_diagnosis.csv",
            mime="text/csv",
            width="stretch",
        )
    with filtered_col:
        st.download_button(
            "현재 표시 행 CSV 다운로드",
            data=filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="secom_filtered_review_queue.csv",
            mime="text/csv",
            width="stretch",
            disabled=filtered.empty,
        )

def render_secom_explanation() -> None:
    suspicious_order = [int(value) for value in results.sort_values(
        "우선 확인 점수", ascending=False
    )["행 번호"].tolist()]
    requested_row = st.session_state.get("secom_selected_row")
    query_row = query_value("row", "")
    if query_row.isdigit() and int(query_row) in suspicious_order:
        requested_row = int(query_row)
    if requested_row not in suspicious_order:
        requested_row = suspicious_order[0]
    if st.session_state.get("secom_explanation_row") not in suspicious_order:
        st.session_state["secom_explanation_row"] = requested_row
    selected_row = st.selectbox(
        "설명할 행 번호", suspicious_order, key="secom_explanation_row"
    )
    selected_row = int(selected_row)
    st.session_state["secom_selected_row"] = selected_row
    set_query_state(module="secom", section="explanation", row=selected_row)
    st.subheader("1. 입력 데이터 한계와 검토 우선순위")
    if review_queue is not None and selected_row in review_queue["행 번호"].values:
        row_quality = review_queue.loc[
            review_queue["행 번호"] == selected_row
        ].iloc[0]
        quality_columns = st.columns(5)
        quality_columns[0].metric("처리 권고", str(row_quality["처리 권고"]))
        quality_columns[1].metric(
            "데이터 신뢰도", str(row_quality["입력 신뢰도"])
        )
        quality_columns[2].metric(
            "검토 우선순위",
            f"{float(row_quality['검토 우선순위 점수']):.2f}",
        )
        quality_columns[3].metric(
            "결측률", f"{float(row_quality.get('결측률', 0.0)):.1%}"
        )
        quality_columns[4].metric(
            "범위 이탈률", f"{float(row_quality.get('범위 이탈률', 0.0)):.1%}"
        )
        st.caption("검토 사유: " + str(row_quality.get("검토 사유", "확인 정보 없음")))
    st.info(
        "여기의 '입력 데이터 신뢰도'는 제품이나 장비의 신뢰도가 아니라, "
        "학습 데이터 범위와 비교한 **입력 데이터의 한계**를 뜻합니다."
    )
    st.subheader("2. 모델 판정과 TreeSHAP 기여")
    st.markdown(f"**입력 행 {selected_row}의 판정**")
    cat_col, xgb_col = st.columns(2)
    with cat_col:
        display_score("CatBoost", prepared["CatBoost"], selected_row)
    with xgb_col:
        display_score("XGBoost", prepared["XGBoost"], selected_row)

    for model_name in MODEL_NAMES:
        st.markdown(f"#### {model_name} 상위 SHAP 기여 변수")
        explanation = explain_selected_row(
            model_name, prepared[model_name], selected_row, top_n
        )
        figure = px.bar(
            explanation,
            x="SHAP 기여도",
            y="변수",
            orientation="h",
            color="방향",
            color_discrete_map={"불량 쪽": "#d1495b", "정상 쪽": "#2878b5"},
            hover_data=["입력값(결측 대체 후)"],
        )
        figure.add_vline(x=0, line_width=1, line_color="#555555")
        figure.update_layout(height=max(330, top_n * 34), legend_title_text="기여 방향")
        st.plotly_chart(figure, width="stretch")
    st.warning(
        "SHAP은 모델 판단에 영향을 준 익명 변수의 기여도입니다. feature_123같은 "
        "번호는 장비·물리 센서명이 아니며, 센서 사전과 단위가 없으므로 이 화면만으로 "
        "실제 공정 원인을 완결하게 설명할 수 없습니다."
    )
    render_secom_shap_reliability()

def render_secom_safety() -> None:
    st.subheader("먼저 볼 입력 데이터 한계")
    if review_queue is not None and not review_queue.empty:
        trust_columns = [
            column for column in (
                "행 번호", "처리 권고", "입력 신뢰도", "검토 우선순위 점수",
                "결측률", "범위 이탈률", "검토 사유",
            ) if column in review_queue.columns
        ]
        trust_overview = review_queue.sort_values(
            "검토 우선순위 점수", ascending=False
        ).head(15)[trust_columns].rename(
            columns={"입력 신뢰도": "입력 데이터 상태"}
        )
        st.dataframe(
            trust_overview,
            hide_index=True,
            width="stretch",
            column_config={
                "검토 우선순위 점수": st.column_config.NumberColumn(format="%.3f"),
                "결측률": st.column_config.NumberColumn(format="percent"),
                "범위 이탈률": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(
            "검토 우선순위·결측률·범위 이탈률을 먼저 확인한 뒤 아래 모델 안전 지표와 "
            "What-if를 해석하세요. 이 판정은 입력 데이터의 한계이며 장비 이상 확정이 아닙니다."
        )
    else:
        st.info("입력 신뢰도 기준이 없어 모델 점수만으로 자동 확정하지 않습니다.")
    st.divider()
    safety_summary_path = SAFETY_POLICY_DIR / "safety_policy_summary.json"
    if safety_summary_path.is_file():
        safety_summary = json.loads(safety_summary_path.read_text(encoding="utf-8"))
        locked_test = safety_summary["test_locked_policy"]
        st.subheader("고정 테스트셋 안전 정책 검증")
        policy_columns = st.columns(4)
        policy_columns[0].metric(
            "전문가 검토 업무량", f"{locked_test['review_workload']:.1%}"
        )
        policy_columns[1].metric(
            "오분류 포착률", f"{locked_test['error_capture']:.1%}"
        )
        policy_columns[2].metric(
            "미탐 포착률", f"{locked_test['false_negative_capture']:.1%}"
        )
        policy_columns[3].metric(
            "자동 처리 정확도", f"{locked_test['auto_accuracy']:.1%}"
        )
        st.caption(
            f"고정 test {locked_test['rows']}행 중 {locked_test['reviewed_rows']}행 검토 · "
            f"오분류 {locked_test['captured_errors']}/{locked_test['errors']}건 포착 · "
            f"미탐 {locked_test['captured_false_negatives']}/"
            f"{locked_test['false_negatives']}건 포착. 임계값은 반복 OOF에서 정했고 "
            "test 라벨로 검토 규칙을 조정하지 않았습니다."
        )
        safety_curve_path = SAFETY_POLICY_DIR / "safety_policy_curve.png"
        if safety_curve_path.is_file():
            st.image(
                str(safety_curve_path),
                caption="검토 업무량을 늘릴 때 오분류·미탐 포착률의 변화",
            )
        tradeoff_path = SAFETY_POLICY_DIR / "workload_tradeoff.csv"
        if tradeoff_path.is_file():
            with st.expander("검토 예산별 상세 수치"):
                tradeoff = pd.read_csv(tradeoff_path)
                st.dataframe(
                    tradeoff[
                        [
                            "dataset",
                            "additional_normal_budget",
                            "review_workload",
                            "error_capture",
                            "false_negative_capture",
                            "auto_accuracy",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "additional_normal_budget": st.column_config.NumberColumn(
                            "정상 합의 추가 검토 비율", format="percent"
                        ),
                        "review_workload": st.column_config.NumberColumn(
                            "전체 검토 업무량", format="percent"
                        ),
                        "error_capture": st.column_config.NumberColumn(
                            "오분류 포착률", format="percent"
                        ),
                        "false_negative_capture": st.column_config.NumberColumn(
                            "미탐 포착률", format="percent"
                        ),
                        "auto_accuracy": st.column_config.NumberColumn(
                            "자동 처리 정확도", format="percent"
                        ),
                    },
                )
                st.warning(
                    "test 곡선은 사후 설명용입니다. 운영 검토량을 선택할 때 test 성능을 "
                    "기준으로 삼으면 데이터 누수가 발생합니다."
                )
        st.divider()

    rescue_summary = None
    rescue_summary_path = (
        UNCERTAINTY_RESCUE_DIR / "uncertainty_rescue_summary.json"
    )
    if rescue_summary_path.is_file():
        rescue_summary = json.loads(
            rescue_summary_path.read_text(encoding="utf-8")
        )
        standard_test = rescue_summary["test_standard"]
        safety_first_test = rescue_summary["test_safety_first"]
        rescue_delta = rescue_summary["test_delta"]
        st.subheader("미탐 예방 안전 우선 프로필")
        rescue_columns = st.columns(4)
        rescue_columns[0].metric(
            "검토 업무량",
            f"{safety_first_test['review_workload']:.1%}",
            delta=f"+{rescue_delta['review_workload']:.1%}p",
            delta_color="inverse",
        )
        rescue_columns[1].metric(
            "미탐 포착률",
            f"{safety_first_test['false_negative_capture']:.1%}",
            delta=f"+{rescue_delta['false_negative_capture']:.1%}p",
        )
        rescue_columns[2].metric(
            "오분류 포착률",
            f"{safety_first_test['error_capture']:.1%}",
            delta=f"+{rescue_delta['error_capture']:.1%}p",
        )
        rescue_columns[3].metric(
            "자동 처리 정확도",
            f"{safety_first_test['auto_accuracy']:.1%}",
            delta=f"+{rescue_delta['auto_accuracy']:.1%}p",
        )
        rescue_plot = UNCERTAINTY_RESCUE_DIR / "policy_comparison.png"
        if rescue_plot.is_file():
            st.image(
                str(rescue_plot),
                caption="표준 정책과 OOF에서 선택한 안전 우선 정책 비교",
            )
        st.caption(
            f"반복 OOF에서 미탐 포착 70% 이상을 만족하는 최소 정책으로 정상 합의 "
            f"후보의 상위 {rescue_summary['chosen_additional_normal_budget']:.0%}를 "
            f"추가 검토합니다. 배포 컷오프는 "
            f"{rescue_summary['normalized_risk_cutoff']:.4f}이며 test 라벨은 정책 "
            "선택에 사용하지 않았습니다."
        )
        st.warning(
            "고정 test 미탐은 9건뿐입니다. 안전 우선 포착률의 95% 신뢰구간은 "
            f"{safety_first_test['fn_capture_ci_low']:.1%}~"
            f"{safety_first_test['fn_capture_ci_high']:.1%}로 넓으므로 외부 lot 검증이 "
            "필요합니다."
        )
        st.divider()

    gate_summary_path = BATCH_GATE_DIR / "batch_gate_summary.json"
    gate_reference_path = BATCH_GATE_DIR / "batch_gate_reference.json"
    if gate_summary_path.is_file() and gate_reference_path.is_file():
        gate_summary = json.loads(gate_summary_path.read_text(encoding="utf-8"))
        gate_reference = json.loads(
            gate_reference_path.read_text(encoding="utf-8")
        )
        st.subheader("배치 안전 차단기")
        gate_columns = st.columns(4)
        gate_columns[0].metric(
            "깨끗한 test 배치", gate_summary["clean_test_decision"]["status"]
        )
        gate_columns[1].metric(
            "교란 배치 STOP",
            f"{gate_summary['stress_stop_count']}/{gate_summary['stress_batches']}",
        )
        gate_columns[2].metric(
            "최소 배치 크기", f"{gate_reference['minimum_batch_rows']}행"
        )
        gate_columns[3].metric(
            "실시간 배치 상태",
            batch_gate_result["status"]
            if batch_gate_result is not None
            else "균형형에서 사용",
        )
        gate_plot = BATCH_GATE_DIR / "gate_stress_validation.png"
        if gate_plot.is_file():
            st.image(
                str(gate_plot),
                caption="센서 교란 조건별 배치 차단 결정 비율",
                width="stretch",
            )
        if batch_gate_result is not None:
            gate_signal_rows = []
            signal_labels = {
                "ood_any_rate": "전체 OOD 비율",
                "ood_severe_rate": "Severe OOD 비율",
                "model_disagreement_rate": "모델 불일치율",
            }
            for signal, detail in batch_gate_result["signals"].items():
                gate_signal_rows.append(
                    {
                        "안전 신호": signal_labels[signal],
                        "현재값": detail["value"],
                        "경고 한계": detail["warning_limit"],
                        "중단 한계": detail["stop_limit"],
                    }
                )
            st.dataframe(
                pd.DataFrame(gate_signal_rows),
                hide_index=True,
                width="stretch",
                column_config={
                    column: st.column_config.NumberColumn(format="percent")
                    for column in ("현재값", "경고 한계", "중단 한계")
                },
            )
            st.caption(" · ".join(batch_gate_result["reasons"]))
        else:
            st.caption(
                "배치 차단기는 OOF로 검증된 `균형형 · F2 최대` 임계값에서만 "
                "활성화됩니다."
            )
        st.caption(
            "차단 기준은 fixed train의 leave-one-out OOD 비율과 반복 OOF 모델 "
            "불일치율로 정했습니다. test는 차단 기준 선택에 사용하지 않았습니다."
        )
        st.divider()

    st.subheader("상세 전문가 검토·What-if")
    if advanced_reference is None or ood_results is None or review_queue is None:
        st.caption(
            "고급 진단 기준이 없습니다. build_advanced_diagnostics.py를 실행하세요."
        )
    else:
        active_review_queue = review_queue.copy()
        policy_options = ["표준 · 임계값 25% 경계"]
        if rescue_summary is not None and profile == "balanced_f2":
            policy_options.append("안전 우선 · 미탐 예방")
        review_policy_label = st.radio(
            "전문가 검토 정책",
            policy_options,
            horizontal=True,
            key="secom_review_policy",
        )
        if review_policy_label.startswith("안전 우선"):
            cutoff = float(rescue_summary["normalized_risk_cutoff"])
            normalized_risk = np.maximum(
                active_review_queue["CatBoost 점수"]
                / active_review_queue["CatBoost 임계값"].clip(lower=1e-12),
                active_review_queue["XGBoost 점수"]
                / active_review_queue["XGBoost 임계값"].clip(lower=1e-12),
            )
            rescue_rows = (
                (active_review_queue["종합 판정"] == "두 모델 모두 정상")
                & (normalized_risk >= cutoff)
                & (active_review_queue["처리 권고"] == "자동 처리 후보")
            )
            active_review_queue.loc[
                rescue_rows, "처리 권고"
            ] = "미탐 예방 검토"
            active_review_queue.loc[
                rescue_rows, "검토 우선순위 점수"
            ] += 2.0 + normalized_risk[rescue_rows]
            active_review_queue = active_review_queue.sort_values(
                ["검토 우선순위 점수", "우선 확인 점수"], ascending=False
            ).reset_index(drop=True)
            st.info(
                f"현재 입력에서 정상 합의 {int(rescue_rows.sum())}행을 미탐 예방 "
                "검토 대상으로 추가했습니다."
            )
        elif rescue_summary is not None and profile != "balanced_f2":
            st.caption(
                "안전 우선 컷오프는 `균형형 · F2 최대` 임계값에서만 검증되어 현재 "
                "운영 임계값에서는 표준 정책을 사용합니다."
            )

        if batch_gate_result is not None:
            gate_status = batch_gate_result["status"]
            if gate_status == "STOP":
                active_review_queue["처리 권고"] = "배치 차단 · 전체 검토"
                active_review_queue["검토 우선순위 점수"] += 20.0
                active_review_queue = active_review_queue.sort_values(
                    ["검토 우선순위 점수", "우선 확인 점수"], ascending=False
                ).reset_index(drop=True)
                st.error(
                    "이 배치는 안전 한계를 넘어 모든 행의 자동판정을 금지했습니다. "
                    "원본 입력 검증·센서 재측정·전문가 검토 후 다시 실행하세요."
                )
            elif gate_status == "CAUTION":
                st.warning(
                    "배치 통계가 경고 범위입니다. 행 단위 판정과 함께 배치 원인을 "
                    "확인하세요."
                )

        severe_count = int(
            (ood_results["OOD 상태"] == "out_of_distribution").sum()
        )
        review_count = int((ood_results["OOD 상태"] == "review").sum())
        disagreement_count = int(
            results["종합 판정"].isin(
                ("CatBoost만 불량", "XGBoost만 불량")
            ).sum()
        )
        review_target_count = int(
            (active_review_queue["처리 권고"] != "자동 처리 후보").sum()
        )
        safety_metrics = st.columns(4)
        safety_metrics[0].metric("OOD 판정 보류", severe_count)
        safety_metrics[1].metric("분포 경계 검토", review_count)
        safety_metrics[2].metric("모델 불일치", disagreement_count)
        safety_metrics[3].metric("전체 검토 대상", review_target_count)

        queue_columns = [
            "행 번호",
            "처리 권고",
            "종합 판정",
            "입력 신뢰도",
            "검토 우선순위 점수",
            "CatBoost 점수",
            "XGBoost 점수",
            "결측률",
            "범위 이탈률",
            "검토 사유",
        ]
        st.dataframe(
            active_review_queue[queue_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "검토 우선순위 점수": st.column_config.NumberColumn(format="%.3f"),
                "CatBoost 점수": st.column_config.NumberColumn(format="%.4f"),
                "XGBoost 점수": st.column_config.NumberColumn(format="%.4f"),
                "결측률": st.column_config.NumberColumn(format="percent"),
                "범위 이탈률": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(
            "OOD는 train의 결측률·1~99% 센서 범위·PCA 최근접 거리를 함께 검사합니다. "
            "분포 이탈 행은 확률이 높거나 낮아도 자동판정을 보류합니다."
        )
        queue_csv = active_review_queue[queue_columns].to_csv(
            index=False, encoding="utf-8-sig"
        ).encode("utf-8-sig")
        st.download_button(
            "전문가 검토 대기열 CSV 다운로드",
            data=queue_csv,
            file_name="secom_human_review_queue.csv",
            mime="text/csv",
            width="stretch",
        )

        st.divider()
        st.subheader("불량 패턴 후보와 제한형 What-if")
        safety_row = int(
            st.selectbox(
                "고급 진단할 행 번호",
                active_review_queue["행 번호"].astype(int).tolist(),
                key="advanced_diagnosis_row",
            )
        )
        X_safety_row = X_raw.iloc[[safety_row]]
        row_ood = ood_results.loc[ood_results["행 번호"] == safety_row].iloc[0]
        signature = assign_failure_signature(
            X_safety_row,
            bundles["CatBoost"],
            advanced_reference,
        )
        row_result = results.loc[results["행 번호"] == safety_row].iloc[0]
        detail_columns = st.columns(4)
        detail_columns[0].metric("입력 신뢰도", row_ood["입력 신뢰도"])
        detail_columns[1].metric("가까운 불량 패턴", signature["pattern"])
        detail_columns[2].metric(
            "패턴 내 실제 불량", f"{signature['training_defect_rows']}행"
        )
        detail_columns[3].metric(
            "두 모델 판정", str(row_result["종합 판정"])
        )
        st.markdown(
            "**패턴의 주요 불량 방향 변수:** "
            + ", ".join(signature["top_risk_features"])
        )
        st.caption(
            "이 패턴은 CatBoost SHAP 기여 방향이 비슷한 실제 train 불량을 묶은 탐색적 "
            "후보이며, 알려진 고장명이나 인과 원인이 아닙니다."
        )

        what_if_disabled = row_ood["OOD 상태"] == "out_of_distribution"
        if st.button(
            "관측 범위 내 What-if 탐색",
            disabled=what_if_disabled,
            type="primary",
            key="run_secom_counterfactual",
        ):
            with st.spinner("두 모델 공통 위험을 낮추는 최소 센서 변경 탐색 중..."):
                st.session_state["secom_counterfactual"] = {
                    "row": safety_row,
                    "profile": profile,
                    "result": constrained_counterfactual(
                        X_safety_row,
                        bundles,
                        advanced_reference,
                        profile,
                    ),
                }
        if what_if_disabled:
            st.warning(
                "이 행은 학습 분포를 벗어나 What-if를 실행하지 않습니다. 먼저 입력값과 "
                "센서 상태를 확인해야 합니다."
            )

        saved_counterfactual = st.session_state.get("secom_counterfactual")
        if (
            saved_counterfactual
            and saved_counterfactual["row"] == safety_row
            and saved_counterfactual["profile"] == profile
        ):
            counterfactual = saved_counterfactual["result"]
            before_after = []
            for model_name in MODEL_NAMES:
                before_after.append(
                    {
                        "모델": model_name,
                        "변경 전 위험도": counterfactual["before"][model_name],
                        "변경 후 위험도": counterfactual["after"][model_name],
                        "임계값": counterfactual["thresholds"][model_name],
                        "변경 후 판정": (
                            "정상"
                            if counterfactual["after"][model_name]
                            < counterfactual["thresholds"][model_name]
                            else "불량"
                        ),
                    }
                )
            st.dataframe(
                pd.DataFrame(before_after), hide_index=True, width="stretch"
            )
            if counterfactual["changes"]:
                change_table = pd.DataFrame(counterfactual["changes"]).rename(
                    columns={
                        "feature": "센서 변수",
                        "from": "변경 전",
                        "to": "변경 후보",
                        "quantile": "train 분위수",
                        "normalized_change": "IQR 기준 변화량",
                    }
                )
                st.dataframe(change_table, hide_index=True, width="stretch")
            if counterfactual["success"]:
                st.success(
                    "관측 train 범위 안에서 두 모델이 모두 정상으로 바뀌는 후보를 찾았습니다."
                )
            else:
                st.warning(
                    "허용한 변수 수와 관측 범위 안에서는 두 모델 공통 정상 후보를 찾지 "
                    "못했습니다. 부분적으로 위험도를 낮춘 후보만 표시합니다."
                )
            st.error(counterfactual["warning"])

        st.markdown("**전문가 검토 결과 기록**")
        review_form_columns = st.columns((1, 1, 2))
        with review_form_columns[0]:
            reviewer_decision = st.selectbox(
                "검토 결정",
                ("확인 필요", "정상 확인", "불량 확인", "측정 재요청", "판정 보류"),
                key="secom_reviewer_decision",
            )
        with review_form_columns[1]:
            reviewer_id = st.text_input(
                "검토자 ID", value="engineer_01", key="secom_reviewer_id"
            )
        with review_form_columns[2]:
            reviewer_note = st.text_input(
                "검토 메모",
                placeholder="장비 상태, 재측정 여부 등",
                key="secom_reviewer_note",
            )
        if st.button("현재 행 검토 기록 추가", key="add_secom_review_record"):
            try:
                append_store_event(
                    RUNTIME_AUDIT_DATABASE,
                    event_type="MANUAL_NOTE",
                    actor=reviewer_id.strip() or "미입력",
                    batch_id=diagnosis_run_id,
                    payload={
                        "row_number": safety_row,
                        "decision": reviewer_decision,
                        "note": reviewer_note.strip()[:500],
                        "consensus": str(row_result["종합 판정"]),
                        "input_confidence": str(row_ood["입력 신뢰도"]),
                        "failure_pattern": signature["pattern"],
                        "profile": profile,
                    },
                )
                st.success("검토 기록을 로컬 append-only 감사 저장소에 저장했습니다.")
            except Exception as error:
                st.error(f"검토 기록을 저장하지 못했습니다: {error}")

        try:
            stored_reviews = audit_events_frame(
                read_store(RUNTIME_AUDIT_DATABASE), "MANUAL_NOTE"
            )
        except Exception as error:
            stored_reviews = pd.DataFrame()
            st.warning(f"저장된 검토 기록을 읽지 못했습니다: {error}")
        if not stored_reviews.empty:
            review_records_frame = stored_reviews.rename(
                columns={
                    "recorded_at": "기록 시각",
                    "row_number": "행 번호",
                    "actor": "검토자 ID",
                    "decision": "검토 결정",
                    "note": "검토 메모",
                    "consensus": "모델 종합 판정",
                    "input_confidence": "입력 신뢰도",
                    "failure_pattern": "불량 패턴 후보",
                    "profile": "운영 임계값",
                }
            )
            visible_review_columns = [
                column
                for column in (
                    "기록 시각",
                    "행 번호",
                    "검토자 ID",
                    "검토 결정",
                    "검토 메모",
                    "모델 종합 판정",
                    "입력 신뢰도",
                    "불량 패턴 후보",
                    "운영 임계값",
                )
                if column in review_records_frame
            ]
            review_records_frame = review_records_frame[visible_review_columns]
            st.dataframe(review_records_frame, hide_index=True, width="stretch")
            st.download_button(
                "전문가 검토 기록 CSV 다운로드",
                data=review_records_frame.to_csv(
                    index=False, encoding="utf-8-sig"
                ).encode("utf-8-sig"),
                file_name="secom_expert_review_log.csv",
                mime="text/csv",
                width="stretch",
            )
            st.caption(
                "검토 기록은 로컬 SQLite에 해시 체인으로 보존됩니다. 운영 환경에서는 "
                "접근 제어와 외부 백업이 있는 중앙 감사 저장소로 교체해야 합니다."
            )

        st.divider()
        pattern_table = pd.read_csv(
            ADVANCED_DIAGNOSTICS_DIR / "failure_signatures.csv"
        )
        pattern_left, pattern_right = st.columns((1.1, 1))
        with pattern_left:
            st.image(
                str(ADVANCED_DIAGNOSTICS_DIR / "failure_signature_map.png"),
                caption="실제 train 불량 83개의 SHAP 패턴 후보",
                width="stretch",
            )
        with pattern_right:
            st.dataframe(
                pattern_table[
                    [
                        "pattern",
                        "training_defect_rows",
                        "share",
                        "top_risk_features",
                    ]
                ].rename(
                    columns={
                        "pattern": "패턴",
                        "training_defect_rows": "실제 불량 수",
                        "share": "비중",
                        "top_risk_features": "주요 불량 방향 변수",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.info(
                "4개 패턴 · silhouette 0.335 · 80% 재표본 안정성 ARI 0.863. "
                "군집 구조는 비교적 안정적이지만 물리적 고장 라벨은 아닙니다."
            )

        report_bytes = build_html_report(
            source_name=source_name,
            profile_label=profile_label,
            predictions=results,
            ood=ood_results,
            review_queue=active_review_queue,
            model_hashes=advanced_reference["model_hashes"],
        )
        st.download_button(
            "감사 가능한 HTML 진단 보고서 다운로드",
            data=report_bytes,
            file_name="secom_diagnostic_audit_report.html",
            mime="text/html",
            width="stretch",
        )

def render_secom_process() -> None:
    st.subheader("SECOM–WM-811K 2단계 제조 의사결정 흐름")
    process_columns = st.columns(4)
    process_columns[0].markdown(
        "#### 1. 공정 감시\nSECOM 센서로 불량 위험과 입력 분포 이탈을 탐지합니다."
    )
    process_columns[1].markdown(
        "#### 2. 검토 우선순위\n모델 불일치·OOD·고위험 샘플을 전문가 대기열로 보냅니다."
    )
    process_columns[2].markdown(
        "#### 3. 후단 검사\nWM-811K 모듈이 웨이퍼 결함 패턴과 Grad-CAM 위치를 제시합니다."
    )
    process_columns[3].markdown(
        "#### 4. 사람의 판단\n공정 엔지니어가 두 근거와 장비 정보를 함께 검토합니다."
    )
    st.info("상단 메뉴의 ‘WM-811K 진단’에서 후단 검사 화면을 열 수 있습니다.")
    st.warning(
        "SECOM과 WM-811K는 동일 wafer ID로 연결된 데이터가 아닙니다. 따라서 이 화면은 "
        "제조 의사결정 시스템의 모듈 흐름을 보여주며, 두 예측을 같은 제품의 연속 측정값처럼 "
        "결합하지 않습니다."
    )
    st.markdown(
        """
        **시스템 안전 원칙**

        - SECOM OOD 행은 자동판정을 보류합니다.
        - SHAP 패턴과 What-if는 점검 후보이며 공정 조정 명령이 아닙니다.
        - WM-811K 저신뢰 예측은 전문가 검토로 전환합니다.
        - 모든 결과에는 모델 버전·입력 결과 해시·한계가 포함된 보고서를 남깁니다.
        """
    )

def render_secom_monitoring() -> None:
    st.subheader("배치 운영 기록과 안전 신호 추세")
    st.caption(
        "원본 센서값 없이 집계 지표와 결과 해시만 기록합니다. 브라우저 세션은 영구 저장소가 "
        "아니므로 CSV를 다운로드하고 다음 접속 때 다시 불러오세요. 서로 다른 모델·임계값 프로필의 "
        "추세는 직접 비교하지 마세요."
    )
    with st.expander("기존 모니터링 기록 불러오기"):
        prior_log = st.file_uploader(
            "모니터링 CSV",
            type=["csv"],
            key="monitoring_prior_csv",
            label_visibility="collapsed",
        )
        if prior_log is not None and st.button("CSV를 현재 기록으로 불러오기"):
            try:
                if prior_log.size > 5 * 1024 * 1024:
                    raise ValueError("모니터링 CSV는 5MB 이하만 불러올 수 있습니다.")
                imported = validate_monitoring_log(pd.read_csv(prior_log))
                if len(imported) > 100_000:
                    raise ValueError("모니터링 기록은 최대 100,000행까지 불러올 수 있습니다.")
                st.session_state["secom_monitoring_log"] = imported.drop_duplicates("batch_id", keep="last")
                st.success("이전 기록을 불러왔습니다.")
            except Exception as error:
                st.error(f"모니터링 CSV를 읽지 못했습니다: {error}")
    batch_col, operator_col, note_col = st.columns((1.05, 1, 1.4), gap="medium")
    with batch_col:
        batch_id = st.text_input(
            "배치 ID",
            value="",
            key="monitoring_batch_id",
            help="같은 ID를 다시 기록하면 최신 값으로 교체합니다.",
        )
    with operator_col:
        operator_id = st.text_input("검수자 식별명 · 선택", key="monitoring_operator")
    with note_col:
        note = st.text_input("점검 메모 · 선택", key="monitoring_note")
    confirm_col, defect_col, alerted_col = st.columns((1.05, 1, 1.35), gap="medium")
    with confirm_col:
        confirmed = st.checkbox("실제 검수 결과 확인", key="monitoring_confirmed")
    with defect_col:
        confirmed_count = st.number_input(
            "확인된 불량 수",
            min_value=0,
            max_value=len(results),
            value=0,
            disabled=not confirmed,
        )
    maximum_confirmed_alerted = min(int(confirmed_count), both_defect + one_defect)
    with alerted_col:
        confirmed_alerted_count = st.number_input(
            "불량 중 모델 경보 포함 수",
            min_value=0,
            max_value=maximum_confirmed_alerted,
            value=0,
            disabled=not confirmed,
            help="실제 불량과 모델 경보의 교집합입니다. 재현율·정밀도 계산에 필요합니다.",
        )
    st.caption("숫자 직접 입력은 참고 기록이며 재학습 검토 근거로 사용하지 않습니다.")
    if st.button("현재 진단 배치 기록 추가 · 직접 집계", disabled=ood_results is None):
        try:
            if not batch_id.strip():
                raise ValueError("배치 ID를 입력하세요.")
            queue = active_review_queue if "active_review_queue" in globals() else review_queue
            count = int((queue["처리 권고"] != "자동 처리 후보").sum()) if queue is not None else len(results)
            record = build_monitoring_record(
                results, ood_results, batch_id=batch_id, source_name=source_name,
                profile=profile, gate_result=batch_gate_result, review_target_count=count,
                model_hashes=(advanced_reference or {}).get("model_hashes", {}),
                outcome_confirmed=confirmed, confirmed_defects=int(confirmed_count),
                confirmed_alerted_defects=int(confirmed_alerted_count),
                operator_id=operator_id, note=note,
            )
            updated_history = append_monitoring_record(
                st.session_state.get("secom_monitoring_log"), record
            )
            updated_ledger = append_event(
                st.session_state.get("secom_audit_ledger"),
                event_type="BATCH_DIAGNOSIS",
                actor=operator_id,
                batch_id=batch_id,
                payload=monitoring_record_payload(record),
            )
            st.session_state["secom_monitoring_log"] = updated_history
            st.session_state["secom_audit_ledger"] = updated_ledger
            st.success("현재 배치와 해시 체인 감사 이벤트를 기록했습니다.")
        except Exception as error:
            st.error(f"기록하지 못했습니다: {error}")
    with st.expander("권장 · 행 단위 검수 CSV로 결과 반영"):
        st.markdown(
            "검수 양식을 내려받아 `actual_label`에 정상/불량 또는 0/1을 입력한 뒤 "
            "다시 올리세요. 예측·배치 해시 열이 바뀌면 거부하며, 전체 행이 검수된 "
            "파일만 성능 감시 근거로 인정합니다."
        )
        if batch_id.strip() and ood_results is not None and advanced_reference is not None:
            try:
                label_template = build_label_template(
                    results,
                    ood_results,
                    batch_id=batch_id,
                    profile=profile,
                    model_hashes=advanced_reference.get("model_hashes", {}),
                )
                st.download_button(
                    "행 단위 검수 양식 다운로드",
                    template_csv_bytes(label_template),
                    "secom_row_label_template.csv",
                    "text/csv",
                )
                completed_file = st.file_uploader(
                    "검수 완료 CSV 업로드",
                    type=["csv"],
                    key="row_label_feedback_csv",
                )
                if completed_file is not None:
                    if completed_file.size > 5 * 1024 * 1024:
                        raise ValueError("검수 CSV는 5MB 이하만 올릴 수 있습니다.")
                    completed_labels, row_feedback = validate_completed_feedback(
                        pd.read_csv(completed_file), label_template
                    )
                    row_cols = st.columns(4)
                    row_cols[0].metric("라벨 범위", f"{row_feedback['label_coverage']:.1%}")
                    row_cols[1].metric("실제 불량", row_feedback["confirmed_defects"])
                    row_cols[2].metric("놓친 불량", row_feedback["false_negative"])
                    row_cols[3].metric("오탐", row_feedback["false_positive"])
                    comparison = None
                    if not row_feedback["complete"]:
                        st.warning(
                            "부분 검수는 표본 선택 편향 가능성이 있어 성능 감시·재학습 검토에 "
                            "사용하지 않습니다."
                        )
                    else:
                        promotion_policy_path = (
                            CHAMPION_CHALLENGER_DIR / "promotion_policy.json"
                        )
                        if promotion_policy_path.is_file():
                            promotion_policy = json.loads(
                                promotion_policy_path.read_text(encoding="utf-8")
                            )
                            comparison = evaluate_champion_challenger(
                                completed_labels,
                                promotion_policy,
                                allow_promotion_review=True,
                                bootstrap_draws=500,
                            )
                            st.markdown("#### CatBoost–XGBoost 동일 행 비교")
                            comparison_table = pd.DataFrame(
                                {
                                    "CatBoost": comparison["champion_metrics"],
                                    "XGBoost": comparison["challenger_metrics"],
                                }
                            ).loc[
                                ["precision", "recall", "f2", "false_positive_rate"]
                            ]
                            st.dataframe(
                                comparison_table,
                                width="stretch",
                                column_config={
                                    "CatBoost": st.column_config.NumberColumn(format="percent"),
                                    "XGBoost": st.column_config.NumberColumn(format="percent"),
                                },
                            )
                            decision = comparison["decision"]
                            if decision == "REVIEW_PROMOTION":
                                st.warning("통계 조건을 충족했지만 사람의 모델 승격 검토가 필요합니다.")
                            elif decision == "KEEP_CHAMPION":
                                st.error("안전 guardrail 미통과 · 현재 CatBoost를 유지합니다.")
                            elif decision == "INSUFFICIENT_EVIDENCE":
                                st.info("독립 배치·행·불량·불일치 표본이 부족해 모델 승격을 판단하지 않습니다.")
                            else:
                                st.info("안전 기준은 확인했지만 모델 교체 근거를 더 수집해야 합니다.")
                            st.caption(" · ".join(comparison["reasons"]))
                            recall_delta = comparison[
                                "paired_bootstrap_delta_challenger_minus_champion"
                            ]["recall"]
                            fpr_delta = comparison[
                                "paired_bootstrap_delta_challenger_minus_champion"
                            ]["false_positive_rate"]
                            st.caption(
                                f"XGBoost−CatBoost 재현율 차이 95% CI "
                                f"{recall_delta['ci_low']:.1%}~{recall_delta['ci_high']:.1%} · "
                                f"정상 오탐률 차이 {fpr_delta['ci_low']:.1%}~"
                                f"{fpr_delta['ci_high']:.1%} · McNemar p="
                                f"{comparison['mcnemar_exact_p_value']:.4g}"
                            )
                            st.download_button(
                                "Champion–challenger 비교 JSON 다운로드",
                                comparison_json_bytes(comparison),
                                "secom_champion_challenger_comparison.json",
                                "application/json",
                            )
                    st.download_button(
                        "행 단위 오류 manifest 다운로드",
                        row_manifest_json_bytes(completed_labels, row_feedback),
                        "secom_row_error_manifest.json",
                        "application/json",
                    )
                    if st.button(
                        "전체 행 검수 결과로 배치 기록 추가",
                        disabled=not row_feedback["complete"],
                    ):
                        queue = active_review_queue if "active_review_queue" in globals() else review_queue
                        count = int((queue["처리 권고"] != "자동 처리 후보").sum()) if queue is not None else len(results)
                        record = build_monitoring_record(
                            results,
                            ood_results,
                            batch_id=batch_id,
                            source_name=source_name,
                            profile=profile,
                            gate_result=batch_gate_result,
                            review_target_count=count,
                            model_hashes=advanced_reference.get("model_hashes", {}),
                            outcome_confirmed=True,
                            confirmed_defects=row_feedback["confirmed_defects"],
                            confirmed_alerted_defects=row_feedback[
                                "confirmed_alerted_defects"
                            ],
                            feedback_evidence="row_level_complete",
                            operator_id=operator_id,
                            note=note,
                        )
                        updated_history = append_monitoring_record(
                            st.session_state.get("secom_monitoring_log"), record
                        )
                        updated_ledger = append_event(
                            st.session_state.get("secom_audit_ledger"),
                            event_type="LABEL_FEEDBACK",
                            actor=operator_id,
                            batch_id=batch_id,
                            payload={
                                "label_coverage": row_feedback["label_coverage"],
                                "confirmed_defects": row_feedback["confirmed_defects"],
                                "confirmed_alerted_defects": row_feedback[
                                    "confirmed_alerted_defects"
                                ],
                                "true_positive": row_feedback["true_positive"],
                                "false_positive": row_feedback["false_positive"],
                                "false_negative": row_feedback["false_negative"],
                                "true_negative": row_feedback["true_negative"],
                                "feedback_evidence": "row_level_complete",
                                "input_digest": record["input_digest"],
                            },
                        )
                        if comparison is not None:
                            updated_ledger = append_event(
                                updated_ledger,
                                event_type="MODEL_COMPARISON",
                                actor=operator_id,
                                batch_id=batch_id,
                                payload={
                                    "champion": comparison["champion"],
                                    "challenger": comparison["challenger"],
                                    "decision": comparison["decision"],
                                    "reasons": comparison["reasons"],
                                    "champion_metrics": comparison["champion_metrics"],
                                    "challenger_metrics": comparison[
                                        "challenger_metrics"
                                    ],
                                    "automatic_model_promotion": False,
                                },
                            )
                        st.session_state["secom_monitoring_log"] = updated_history
                        st.session_state["secom_audit_ledger"] = updated_ledger
                        st.success(
                            "전체 행 검수 결과와 모델 비교를 해시 체인 감사 원장에 기록했습니다."
                        )
            except Exception as error:
                st.error(f"행 단위 검수 파일을 처리하지 못했습니다: {error}")
        else:
            st.info("배치 ID를 입력하고 OOD 기준이 준비된 진단을 실행하면 양식을 만들 수 있습니다.")
    st.divider()
    st.subheader("변조 감지형 감사 원장")
    st.caption(
        "집계 지표만 SHA-256 해시 체인으로 기록합니다. 원본 센서값은 저장하지 않으며, "
        "세션 종료 전 JSONL을 내려받아 보관하세요."
    )
    with st.expander("기존 감사 원장 불러오기"):
        prior_ledger = st.file_uploader(
            "감사 원장 JSONL",
            type=["jsonl"],
            key="audit_ledger_jsonl",
            label_visibility="collapsed",
        )
        if prior_ledger is not None and st.button("감사 원장 검증 후 불러오기"):
            try:
                imported_ledger = parse_ledger_jsonl(prior_ledger.getvalue())
                st.session_state["secom_audit_ledger"] = imported_ledger
                st.success(f"해시 체인이 유효한 {len(imported_ledger):,}개 이벤트를 불러왔습니다.")
            except Exception as error:
                st.error(f"감사 원장을 불러오지 못했습니다: {error}")
    audit_ledger = st.session_state.get("secom_audit_ledger")
    if audit_ledger is not None and not audit_ledger.empty:
        try:
            audit_summary = verify_ledger(audit_ledger)
            audit_cols = st.columns(3)
            audit_cols[0].metric("체인 검증", "유효")
            audit_cols[1].metric("감사 이벤트", audit_summary["entry_count"])
            audit_cols[2].metric("최근 해시", audit_summary["latest_hash"][:12] + "…")
            audit_display = audit_ledger.drop(columns=["payload_json"]).copy()
            audit_display["previous_hash"] = audit_display["previous_hash"].str[:12] + "…"
            audit_display["event_hash"] = audit_display["event_hash"].str[:12] + "…"
            st.dataframe(audit_display, width="stretch", hide_index=True)
            st.download_button(
                "검증 가능한 감사 원장 JSONL 다운로드",
                ledger_jsonl_bytes(audit_ledger),
                "secom_audit_ledger.jsonl",
                "application/x-ndjson",
            )
        except Exception as error:
            st.error(f"현재 감사 원장의 해시 체인이 손상됐습니다: {error}")
    else:
        st.caption("감사 이벤트 없음 · 현재 진단 배치를 기록하면 자동 생성됩니다.")

    history = st.session_state.get("secom_monitoring_log")
    if history is not None and not history.empty:
        selected_profile = st.selectbox("추세를 확인할 프로필", sorted(history["profile"].unique()), key="monitoring_profile")
        selected = history.loc[history["profile"] == selected_profile]
        trend_summary = analyze_monitoring_history(selected)
        if trend_summary["status"] == "CRITICAL":
            st.error("최근 배치 STOP · 입력 측정계 확인과 전문가 검토가 필요합니다.")
        elif trend_summary["status"] == "WARNING":
            st.warning("최근 5개 배치에서 경고·차단 반복 또는 현재 경고가 발생했습니다.")
        elif trend_summary["status"] == "NOT_EVALUATED":
            st.warning("최근 배치의 안전 차단기가 미평가 상태입니다. 정상 운영으로 판단하지 마세요.")
        else:
            st.info("현재 안전 신호에 반복 경고가 없습니다. 실제 진단 정확도를 보장하는 지표는 아닙니다.")
        cols = st.columns(3)
        cols[0].metric("기록 배치", trend_summary["batch_count"])
        cols[1].metric("STOP 누적", trend_summary["stop_count"])
        cols[2].metric("연속 미통과·미평가", trend_summary["consecutive_nonpass"])
        st.plotly_chart(px.line(selected.sort_values("recorded_at"), x="recorded_at", y=["ood_any_rate", "ood_severe_rate", "model_disagreement_rate"], markers=True), width="stretch")
        display_history = history.drop(columns=["operator_id", "note"], errors="ignore")
        st.dataframe(display_history, width="stretch", hide_index=True)
        st.caption("검수자 식별명과 메모는 화면 표에서 숨기고 다운로드 CSV에만 포함합니다.")
        st.download_button("모니터링 기록 CSV 다운로드", monitoring_csv_bytes(history), "secom_monitoring_log.csv", "text/csv")
        feedback_reference_path = FEEDBACK_MONITORING_DIR / "feedback_reference.json"
        if feedback_reference_path.is_file():
            st.divider()
            st.subheader("실제 검수 기반 성능 감시")
            feedback_reference = json.loads(feedback_reference_path.read_text(encoding="utf-8"))
            try:
                feedback = analyze_feedback_history(selected, feedback_reference)
                if feedback["status"] == "REVIEW_RETRAINING":
                    st.error("재학습 검토 조건 충족 · 자동 재학습은 시작하지 않습니다.")
                elif feedback["status"] == "WATCH":
                    st.warning("성능 또는 실제 불량률 변화를 관찰해야 합니다.")
                elif feedback["status"] == "INSUFFICIENT":
                    st.info("검수 표본이 아직 부족해 성능 저하를 판정하지 않습니다.")
                elif feedback["status"] == "NO_LABELS":
                    st.info("실제 불량과 모델 경보의 교차 검수 결과를 입력하면 성능 감시를 시작합니다.")
                elif feedback["status"] == "REFERENCE_MISMATCH":
                    st.warning("현재 모델·프로필과 OOF 감시 기준이 달라 성능을 비교하지 않습니다. 기준을 다시 생성하세요.")
                elif feedback["status"] == "UNVERIFIED_LABELS":
                    st.warning("직접 입력한 집계값만 있어 성능 감시에 사용하지 않습니다. 행 단위 전체 검수 CSV가 필요합니다.")
                else:
                    st.success("현재 검수 표본은 사전 고정된 감시 기준 이내입니다.")
                st.caption(" · ".join(feedback["reasons"]))
                if feedback.get("confirmed_rows", 0):
                    feedback_cols = st.columns(4)
                    feedback_cols[0].metric("검수 배치", feedback["confirmed_batches"])
                    feedback_cols[1].metric("검수 행", feedback["confirmed_rows"])
                    feedback_cols[2].metric(
                        "경보 재현율",
                        "계산 불가" if pd.isna(feedback.get("recall")) else f"{feedback['recall']:.1%}",
                    )
                    feedback_cols[3].metric("놓친 불량", feedback.get("false_negative", 0))
                    if "recall_ci_low" in feedback:
                        st.caption(
                            f"재현율 95% Wilson CI {feedback['recall_ci_low']:.1%}~"
                            f"{feedback['recall_ci_high']:.1%} · 사전 감시 하한 "
                            f"{feedback['recall_guardrail']:.1%}"
                        )
                    st.download_button(
                        "재학습 후보 배치 manifest 다운로드",
                        manifest_json_bytes(feedback),
                        "secom_retraining_candidate_manifest.json",
                        "application/json",
                        help="센서 원본은 포함하지 않으며 자동 재학습도 시작하지 않습니다.",
                    )
            except Exception as error:
                st.error(f"검수 성능을 계산하지 못했습니다: {error}")
    else:
        st.caption("운영 추세 없음 · 이전 CSV를 불러오거나 현재 배치를 추가하세요.")
    demo_dir = ARTIFACTS_DIR / "monitoring_demo"
    if (demo_dir / "monitoring_trend.png").is_file():
        with st.expander("합성 스트레스 재생 데모 · 실제 운영 이력 아님"):
            st.caption("기존 센서 오류 스트레스 결과의 시간순 재생입니다. 실제 공정 변화나 실운영 성능을 의미하지 않습니다.")
            st.image(str(demo_dir / "monitoring_trend.png"))

def render_secom_release() -> None:
    st.subheader("SECOM 통합 릴리스 준비도")
    readiness_path = RELEASE_READINESS_DIR / "release_readiness.json"
    manifest_path = RELEASE_READINESS_DIR / "release_manifest.json"
    html_path = RELEASE_READINESS_DIR / "release_readiness.html"
    if readiness_path.is_file() and manifest_path.is_file():
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        release_columns = st.columns(4)
        release_columns[0].metric("Release ID", release_manifest["release_id"])
        release_columns[1].metric("대회 데모", readiness["demo_readiness"])
        release_columns[2].metric("실제 생산", readiness["production_readiness"])
        release_columns[3].metric("생산 차단 항목", readiness["production_blocker_count"])
        if readiness["demo_readiness"] == "READY_WITH_WARNINGS":
            st.warning("대회 시연은 가능하지만 강건성 경고와 데이터 한계를 함께 공개해야 합니다.")
        elif readiness["demo_readiness"] == "BLOCKED":
            st.error("대회 시연 필수 증거가 충족되지 않았습니다.")
        else:
            st.success("대회 시연 필수 증거가 준비됐습니다.")
        if readiness["production_readiness"] == "BLOCKED":
            st.error("실제 반도체 생산 배포는 승인되지 않았습니다.")
        matrix = pd.DataFrame(readiness["checks"])
        display_matrix = matrix.rename(
            columns={
                "check_id": "검사 ID",
                "title": "검사 항목",
                "status": "상태",
                "evidence": "근거",
                "required_action": "필요 조치",
            }
        )
        st.dataframe(display_matrix, width="stretch", hide_index=True)
        blocked = matrix.loc[matrix["status"] == "BLOCK"]
        if not blocked.empty:
            with st.expander("생산 배포 차단 해제에 필요한 조치"):
                for _, item in blocked.iterrows():
                    st.markdown(f"- **{item['title']}**: {item['required_action']}")
        chart_path = RELEASE_READINESS_DIR / "readiness_matrix.png"
        if chart_path.is_file():
            st.image(str(chart_path), caption="PASS·WARN·BLOCK 증거 행렬", width="stretch")
        download_columns = st.columns(3)
        download_columns[0].download_button(
            "준비도 JSON",
            readiness_path.read_bytes(),
            "secom_release_readiness.json",
            "application/json",
        )
        download_columns[1].download_button(
            "Release manifest",
            manifest_path.read_bytes(),
            "secom_release_manifest.json",
            "application/json",
        )
        if html_path.is_file():
            download_columns[2].download_button(
                "감사 보고서 HTML",
                html_path.read_bytes(),
                "secom_release_readiness.html",
                "text/html",
            )
        external_protocol_path = EXTERNAL_VALIDATION_DIR / "protocol.json"
        external_status_path = EXTERNAL_VALIDATION_DIR / "status.json"
        external_template_path = EXTERNAL_VALIDATION_DIR / "external_lot_template.csv"
        external_declarations_path = (
            EXTERNAL_VALIDATION_DIR / "declarations_template.json"
        )
        if all(
            path.is_file()
            for path in (
                external_protocol_path,
                external_status_path,
                external_template_path,
                external_declarations_path,
            )
        ):
            st.divider()
            st.subheader("외부 신규 lot 검증 준비")
            external_protocol = json.loads(
                external_protocol_path.read_text(encoding="utf-8")
            )
            external_status = json.loads(
                external_status_path.read_text(encoding="utf-8")
            )
            external_cols = st.columns(4)
            external_cols[0].metric("현재 상태", external_status["status"])
            external_cols[1].metric(
                "프로토콜 ID", external_protocol["protocol_id"][:12] + "…"
            )
            external_cols[2].metric(
                "최소 lot", external_protocol["minimum_evidence"]["distinct_lots"]
            )
            external_cols[3].metric(
                "최소 표본/불량",
                f"{external_protocol['minimum_evidence']['rows']}/"
                f"{external_protocol['minimum_evidence']['defects']}",
            )
            st.warning(
                "실제 독립 신규 데이터가 아직 없습니다. 모델·임계값·합격 기준만 사전 "
                "고정된 상태이며 기존 SECOM 행은 외부 검증으로 인정되지 않습니다."
            )
            st.caption(
                "선언 JSON은 모든 항목이 false로 시작합니다. 실제 수집·라벨 절차가 해당 "
                "조건을 충족했을 때만 true로 바꾸세요."
            )
            external_downloads = st.columns(3)
            external_downloads[0].download_button(
                "외부 lot CSV 양식",
                external_template_path.read_bytes(),
                "secom_external_lot_template.csv",
                "text/csv",
                key="secom_external_lot_template_download",
            )
            external_downloads[1].download_button(
                "독립성 선언 JSON",
                external_declarations_path.read_bytes(),
                "secom_external_validation_declarations.json",
                "application/json",
                key="secom_external_declarations_download",
            )
            external_downloads[2].download_button(
                "고정 검증 프로토콜",
                external_protocol_path.read_bytes(),
                "secom_external_validation_protocol.json",
                "application/json",
                key="secom_external_protocol_download",
            )
            with st.expander("독립 신규 lot 검증 실행", expanded=False):
                st.info(
                    "실제 독립 수집·라벨 절차가 완료된 데이터만 사용하세요. 업로드 원본은 "
                    "현재 Streamlit 세션 메모리에서만 평가하며 서버 파일로 저장하지 않습니다."
                )
                st.caption(
                    "공개 배포 주소에서는 파일이 Streamlit Cloud 서버로 전송됩니다. 기밀 "
                    "공정 데이터는 공개 앱이 아니라 로컬 대시보드에서만 검증하세요."
                )
                external_csv_upload = st.file_uploader(
                    "외부 lot CSV",
                    type=["csv"],
                    key="secom_external_validation_csv",
                )
                external_declaration_upload = st.file_uploader(
                    "완료된 독립성 선언 JSON",
                    type=["json"],
                    key="secom_external_validation_declarations",
                )
                run_external_validation = st.button(
                    "고정 프로토콜로 검증 실행",
                    type="primary",
                    disabled=(
                        external_csv_upload is None
                        or external_declaration_upload is None
                    ),
                    key="run_secom_external_validation",
                )
                if run_external_validation:
                    try:
                        external_frame = parse_external_csv_bytes(
                            external_csv_upload.getvalue()
                        )
                        external_declarations = parse_declarations_bytes(
                            external_declaration_upload.getvalue()
                        )
                        external_summary, external_per_lot = evaluate_external_frame(
                            external_frame,
                            external_declarations,
                            PROJECT_DIR,
                            external_protocol,
                        )
                    except (ValueError, KeyError, TypeError) as error:
                        st.error(f"외부 검증 중단: {error}")
                    else:
                        overall = external_summary["overall"]
                        if external_summary["validation_status"] == "PASS":
                            st.success(
                                "사전 등록된 모든 외부 검증 gate를 통과했습니다. "
                                "이 결과는 모델 재선택이나 임계값 조정에 사용되지 않았습니다."
                            )
                        else:
                            st.warning(
                                "외부 검증은 완료됐지만 하나 이상의 사전 등록 gate를 "
                                "통과하지 못했습니다. 운영 배포 근거로 사용할 수 없습니다."
                            )
                        result_cols = st.columns(5)
                        result_cols[0].metric("판정", external_summary["validation_status"])
                        result_cols[1].metric("행 / lot", f"{overall['rows']:,} / {external_summary['distinct_lots']}")
                        result_cols[2].metric("재현율", f"{overall['recall']:.1%}")
                        result_cols[3].metric("정상 오탐률", f"{overall['false_positive_rate']:.1%}")
                        result_cols[4].metric("검토율", f"{overall['review_rate']:.1%}")
                        gate_labels = {
                            "minimum_rows": "최소 표본 수",
                            "minimum_distinct_lots": "최소 독립 lot 수",
                            "minimum_defects": "최소 불량 수",
                            "no_known_secom_row_overlap": "공개 SECOM 행과 비중복",
                            "recall_lower_bound": "재현율 95% 하한",
                            "false_positive_upper_bound": "정상 오탐률 95% 상한",
                            "review_rate_upper_bound": "검토율 95% 상한",
                        }
                        gate_frame = pd.DataFrame(
                            [
                                {
                                    "검증 항목": gate_labels.get(name, name),
                                    "결과": "통과" if passed else "차단",
                                }
                                for name, passed in external_summary["gates"].items()
                            ]
                        )
                        st.dataframe(gate_frame, hide_index=True, width="stretch")
                        st.dataframe(
                            external_per_lot,
                            hide_index=True,
                            width="stretch",
                        )
                        result_downloads = st.columns(2)
                        result_downloads[0].download_button(
                            "검증 집계 JSON",
                            external_summary_json_bytes(external_summary),
                            "secom_external_validation_summary.json",
                            "application/json",
                            key="secom_external_summary_download",
                        )
                        result_downloads[1].download_button(
                            "lot별 집계 CSV",
                            per_lot_csv_bytes(external_per_lot),
                            "secom_external_validation_per_lot.csv",
                            "text/csv",
                            key="secom_external_per_lot_download",
                        )
                        st.caption(
                            "다운로드 결과에는 원시 센서 행·sample_id·행별 예측이 포함되지 "
                            "않으며 dataset digest와 집계 지표만 남습니다."
                        )

        sensor_status_path = SENSOR_SEMANTICS_DIR / "status.json"
        sensor_schema_path = SENSOR_SEMANTICS_DIR / "input_schema.json"
        if (
            sensor_status_path.is_file()
            and sensor_schema_path.is_file()
            and SENSOR_DICTIONARY_TEMPLATE_PATH.is_file()
        ):
            st.divider()
            st.subheader("센서 의미·단위 사전 준비")
            sensor_status = json.loads(
                sensor_status_path.read_text(encoding="utf-8")
            )
            sensor_cols = st.columns(4)
            sensor_cols[0].metric("현재 상태", sensor_status["status"])
            sensor_cols[1].metric(
                "담당자 검증", f"{sensor_status['verified_count']}/590"
            )
            sensor_cols[2].metric(
                "임시 매핑", sensor_status["provisional_count"]
            )
            sensor_cols[3].metric("미매핑", sensor_status["unmapped_count"])
            st.warning(
                "공개 SECOM 변수의 물리 의미는 알 수 없습니다. 실제 장비·공정 담당자가 "
                "근거 문서와 함께 590개 전체를 검증하기 전에는 운영 배포에 사용할 수 없습니다."
            )
            sensor_downloads = st.columns(2)
            sensor_downloads[0].download_button(
                "센서 사전 CSV 양식",
                SENSOR_DICTIONARY_TEMPLATE_PATH.read_bytes(),
                "secom_sensor_dictionary_template.csv",
                "text/csv",
                key="secom_sensor_dictionary_template_download",
            )
            sensor_downloads[1].download_button(
                "센서 사전 입력 규칙",
                sensor_schema_path.read_bytes(),
                "secom_sensor_dictionary_schema.json",
                "application/json",
                key="secom_sensor_dictionary_schema_download",
            )
            with st.expander("작성된 센서 사전 검증", expanded=False):
                st.info(
                    "이 화면은 작성 내용을 검증만 하며 운영 사전을 자동 설치하지 않습니다. "
                    "590개 전체가 담당자 근거와 함께 VERIFIED여야 운영 gate 후보가 됩니다."
                )
                st.caption(
                    "공개 배포 주소에서는 파일이 Streamlit Cloud 서버로 전송됩니다. 장비·공정 "
                    "정보가 기밀이면 로컬 대시보드에서만 사용하세요."
                )
                sensor_dictionary_upload = st.file_uploader(
                    "작성된 센서 사전 CSV",
                    type=["csv"],
                    key="secom_sensor_dictionary_validation_csv",
                )
                validate_sensor_dictionary_upload = st.button(
                    "센서 사전 검증",
                    type="primary",
                    disabled=sensor_dictionary_upload is None,
                    key="run_secom_sensor_dictionary_validation",
                )
                if validate_sensor_dictionary_upload:
                    try:
                        normalized_dictionary, sensor_report = (
                            parse_sensor_dictionary_bytes(
                                sensor_dictionary_upload.getvalue()
                            )
                        )
                    except (ValueError, KeyError, TypeError) as error:
                        st.error(f"센서 사전 검증 중단: {error}")
                    else:
                        if sensor_report["validation_status"] == "VERIFIED":
                            st.success(
                                "590개 센서가 모두 담당자 검증 조건을 충족했습니다. 운영 설치와 "
                                "릴리스 갱신은 로컬 승인 절차에서 별도로 수행하세요."
                            )
                        else:
                            st.warning(
                                "파일 계약은 유효하지만 전체 담당자 검증이 끝나지 않았습니다. "
                                "현재 상태로는 운영 배포 gate를 통과할 수 없습니다."
                            )
                        validation_cols = st.columns(5)
                        validation_cols[0].metric(
                            "검증 상태", sensor_report["validation_status"]
                        )
                        validation_cols[1].metric(
                            "VERIFIED", sensor_report["verified_count"]
                        )
                        validation_cols[2].metric(
                            "PROVISIONAL", sensor_report["provisional_count"]
                        )
                        validation_cols[3].metric(
                            "UNMAPPED", sensor_report["unmapped_count"]
                        )
                        validation_cols[4].metric(
                            "검증률", f"{sensor_report['verified_coverage']:.1%}"
                        )
                        st.code(
                            f"dictionary_digest={sensor_report['dictionary_digest']}",
                            language="text",
                        )
                        dictionary_downloads = st.columns(2)
                        dictionary_downloads[0].download_button(
                            "센서 사전 검증 보고서",
                            sensor_report_json_bytes(sensor_report),
                            "secom_sensor_dictionary_validation.json",
                            "application/json",
                            key="secom_sensor_validation_report_download",
                        )
                        dictionary_downloads[1].download_button(
                            "정규화된 센서 사전",
                            normalized_dictionary_csv_bytes(normalized_dictionary),
                            "secom_sensor_dictionary_normalized.csv",
                            "text/csv",
                            key="secom_sensor_normalized_dictionary_download",
                        )
        failover_path = MODEL_FAILOVER_DIR / "failover_drill.json"
        registry_path = MODEL_FAILOVER_DIR / "model_registry.json"
        if failover_path.is_file() and registry_path.is_file():
            st.divider()
            st.subheader("모델 무결성·장애 복구 훈련")
            failover = json.loads(failover_path.read_text(encoding="utf-8"))
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            failover_cols = st.columns(4)
            failover_cols[0].metric("훈련 판정", failover["status"])
            failover_cols[1].metric("운영 모델", failover["active_model"])
            failover_cols[2].metric("검증 대기 모델", failover["standby_model"])
            failover_cols[3].metric(
                "판정 불일치율",
                f"{failover['model_prediction_disagreement_rate']:.1%}",
            )
            if failover["simulated_active_corruption_detected"]:
                st.success("원본 모델을 수정하지 않은 모의 손상에서 SHA-256 불일치를 탐지했습니다.")
            else:
                st.error("모의 모델 손상을 탐지하지 못했습니다.")
            registry_table = pd.DataFrame(registry["models"])[
                [
                    "model_name",
                    "role",
                    "config_id",
                    "primary_threshold",
                    "selected_feature_count",
                    "artifact_sha256",
                ]
            ].rename(
                columns={
                    "model_name": "모델",
                    "role": "역할",
                    "config_id": "설정 ID",
                    "primary_threshold": "고정 임계값",
                    "selected_feature_count": "선택 변수",
                    "artifact_sha256": "SHA-256",
                }
            )
            registry_table["SHA-256"] = registry_table["SHA-256"].str[:16] + "…"
            st.dataframe(registry_table, width="stretch", hide_index=True)
            st.error(
                "모델 손상 시 자동 진단을 중단하고 영향 배치를 격리해야 합니다. XGBoost "
                "대기 모델은 자동 전환하지 않으며, 무결성·임계값·검토 업무량 확인 후 책임자 "
                "승인이 필요합니다."
            )
            failover_downloads = st.columns(2)
            failover_downloads[0].download_button(
                "모델 레지스트리 JSON",
                registry_path.read_bytes(),
                "secom_model_registry.json",
                "application/json",
            )
            failover_downloads[1].download_button(
                "장애 복구 훈련 JSON",
                failover_path.read_bytes(),
                "secom_failover_drill.json",
                "application/json",
            )
        lineage_path = DATA_LINEAGE_DIR / "lineage_validation.json"
        lineage_manifest_path = DATA_LINEAGE_DIR / "data_lineage_manifest.json"
        if lineage_path.is_file() and lineage_manifest_path.is_file():
            st.divider()
            st.subheader("데이터 계보·오염 탐지 검증")
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            lineage_cols = st.columns(5)
            lineage_cols[0].metric("계보 판정", lineage["status"])
            lineage_cols[1].metric("원본 행", f"{lineage['raw_rows']:,}")
            lineage_cols[2].metric("원본 변수", lineage["raw_features"])
            lineage_cols[3].metric("유지 변수", lineage["retained_features"])
            lineage_cols[4].metric(
                "고정 분할", f"{lineage['train_rows']:,}/{lineage['test_rows']:,}"
            )
            lineage_table = pd.DataFrame(
                [
                    {"검증 항목": key, "통과": value}
                    for key, value in lineage["checks"].items()
                ]
            )
            st.dataframe(lineage_table, width="stretch", hide_index=True)
            if lineage["status"] == "PASS":
                st.success(
                    "원본에서 저장 train/test까지 행·열·값·라벨·median을 재현했고, "
                    "센서·라벨 파일의 모의 오염을 탐지했습니다."
                )
            else:
                st.error("데이터 계보 재현 또는 오염 탐지 검증에 실패했습니다.")
            st.warning(
                "SHA-256은 파일 변경을 탐지하지만 데이터 수집 당시 측정값의 진실성이나 "
                "익명 센서의 물리적 의미를 증명하지 않습니다."
            )
            lineage_downloads = st.columns(2)
            lineage_downloads[0].download_button(
                "데이터 계보 manifest",
                lineage_manifest_path.read_bytes(),
                "secom_data_lineage_manifest.json",
                "application/json",
            )
            lineage_downloads[1].download_button(
                "계보 검증 결과",
                lineage_path.read_bytes(),
                "secom_lineage_validation.json",
                "application/json",
            )
        environment_validation_path = (
            ENVIRONMENT_PROVENANCE_DIR / "environment_validation.json"
        )
        environment_manifest_path = (
            ENVIRONMENT_PROVENANCE_DIR / "environment_manifest.json"
        )
        software_bom_path = ENVIRONMENT_PROVENANCE_DIR / "software_bom.json"
        if (
            environment_validation_path.is_file()
            and environment_manifest_path.is_file()
            and software_bom_path.is_file()
        ):
            st.divider()
            st.subheader("실행환경 재현성·소프트웨어 명세")
            environment_validation = json.loads(
                environment_validation_path.read_text(encoding="utf-8")
            )
            environment_manifest = json.loads(
                environment_manifest_path.read_text(encoding="utf-8")
            )
            environment_cols = st.columns(4)
            environment_cols[0].metric(
                "환경 판정", environment_validation["status"]
            )
            environment_cols[1].metric(
                "Python", environment_validation["python_version"]
            )
            environment_cols[2].metric(
                "직접 의존성", environment_validation["direct_dependency_count"]
            )
            environment_cols[3].metric(
                "설치 패키지", environment_validation["installed_distribution_count"]
            )
            environment_table = pd.DataFrame(
                [
                    {"검증 항목": key, "통과": value}
                    for key, value in environment_validation["checks"].items()
                ]
            )
            st.dataframe(environment_table, width="stretch", hide_index=True)
            if environment_validation["status"] == "PASS":
                st.success(
                    "정확히 고정된 직접 의존성과 현재 설치 버전을 대조했고, "
                    "모의 버전 이탈 및 실행 파일 변경 탐지 경로를 확인했습니다."
                )
            else:
                st.error("실행환경 또는 프로젝트 파일 재현성 검증에 실패했습니다.")
            st.caption(
                "환경 지문: "
                f"{environment_manifest['environment_fingerprint_sha256'][:20]}…"
            )
            st.warning(
                "이 SBOM 생성 자체에서는 취약점 스캔은 수행하지 않았으므로, 아래의 별도 "
                "온라인 의존성 감사 결과와 함께 해석해야 합니다."
            )
            environment_downloads = st.columns(3)
            environment_downloads[0].download_button(
                "환경 manifest",
                environment_manifest_path.read_bytes(),
                "secom_environment_manifest.json",
                "application/json",
            )
            environment_downloads[1].download_button(
                "환경 검증 결과",
                environment_validation_path.read_bytes(),
                "secom_environment_validation.json",
                "application/json",
            )
            environment_downloads[2].download_button(
                "소프트웨어 SBOM",
                software_bom_path.read_bytes(),
                "secom_software_bom.json",
                "application/json",
            )
        dependency_lock_manifest_path = (
            DEPENDENCY_LOCK_DIR / "windows_lock_manifest.json"
        )
        windows_lock_path = PROJECT_DIR / "pylock.windows.toml"
        if dependency_lock_manifest_path.is_file() and windows_lock_path.is_file():
            dependency_lock = json.loads(
                dependency_lock_manifest_path.read_text(encoding="utf-8")
            )
            st.divider()
            st.subheader("Windows 전이 의존성 잠금")
            lock_cols = st.columns(4)
            lock_cols[0].metric("잠금 판정", dependency_lock["status"])
            lock_cols[1].metric("전체 package", dependency_lock["package_count"])
            lock_cols[2].metric("해시 wheel", dependency_lock["wheel_count"])
            lock_cols[3].metric("sdist", dependency_lock["sdist_count"])
            if dependency_lock["status"] == "PASS":
                st.success(
                    "현재 Windows 테스트 환경의 전이 의존성 버전과 설치 wheel SHA-256을 "
                    "PEP 751 형식으로 고정했습니다."
                )
            else:
                st.error("전이 의존성 잠금 검증에 실패했습니다.")
            st.warning(
                "이 잠금은 Windows x86-64와 Python 3.11.9 전용이며 pip lock은 아직 "
                "experimental입니다. Linux Streamlit Cloud와 GitHub Actions에는 그대로 "
                "사용하지 않습니다."
            )
            st.caption(
                f"형식: {dependency_lock['format']} · pip "
                f"{dependency_lock['generator_version']} · 잠금 SHA-256 "
                f"{dependency_lock['lock_sha256'][:20]}… · 취약점 감사는 별도 수행"
            )
            lock_downloads = st.columns(2)
            lock_downloads[0].download_button(
                "Windows pylock 다운로드",
                windows_lock_path.read_bytes(),
                "pylock.windows.toml",
                "application/toml",
                key="secom_windows_pylock_download",
            )
            lock_downloads[1].download_button(
                "잠금 manifest 다운로드",
                dependency_lock_manifest_path.read_bytes(),
                "secom_windows_lock_manifest.json",
                "application/json",
                key="secom_windows_lock_manifest_download",
            )
        linux_lock_manifest_path = (
            LINUX_DEPENDENCY_LOCK_DIR / "linux_lock_manifest.json"
        )
        linux_ci_lock_path = PROJECT_DIR / "pylock.github-actions.toml"
        streamlit_lock_path = PROJECT_DIR / "pylock.streamlit.toml"
        if all(
            path.is_file()
            for path in (
                linux_lock_manifest_path,
                linux_ci_lock_path,
                streamlit_lock_path,
            )
        ):
            linux_locks = json.loads(
                linux_lock_manifest_path.read_text(encoding="utf-8")
            )
            github_lock = linux_locks["github_actions"]
            streamlit_lock = linux_locks["streamlit"]
            st.divider()
            st.subheader("Linux CI·Streamlit 의존성 잠금")
            linux_cols = st.columns(4)
            linux_cols[0].metric("Linux manifest", linux_locks["status"])
            linux_cols[1].metric("CI package", github_lock["package_count"])
            linux_cols[2].metric("Streamlit package", streamlit_lock["package_count"])
            linux_cols[3].metric(
                "sdist",
                github_lock["sdist_count"] + streamlit_lock["sdist_count"],
            )
            st.success(
                "GitHub Actions lock은 Ubuntu/Python 3.11.9에서 실제 설치된 모든 버전과 "
                "일치하며 CI가 이 해시 잠금으로 설치됩니다."
            )
            st.warning(
                "Streamlit lock은 Ubuntu에서 wheel 해석과 SHA-256 고정까지만 검증했습니다. "
                "Community Cloud가 named pylock을 자동 적용한다고 가정하지 않으며, 현재 "
                "배포는 requirements.txt를 사용합니다."
            )
            runtime_report = build_runtime_environment_report(PROJECT_DIR)
            runtime_expected = runtime_report["expected"]
            runtime_actual = runtime_report["runtime"]
            runtime_checks = runtime_report["checks"]
            st.markdown("#### 현재 실행 프로세스 대조")
            runtime_cols = st.columns(4)
            runtime_cols[0].metric("런타임 판정", runtime_report["status"])
            runtime_cols[1].metric(
                "Python",
                runtime_actual["python_version"],
                f"기준 {runtime_expected['python_version']}",
            )
            runtime_cols[2].metric(
                "직접 의존성",
                (
                    f"{runtime_expected['direct_dependency_count']}/"
                    f"{runtime_expected['direct_dependency_count']}"
                    if runtime_checks["direct_dependencies_match"]
                    else "불일치"
                ),
            )
            runtime_cols[3].metric(
                "잠금 의존성",
                (
                    f"{runtime_expected['locked_dependency_count']}/"
                    f"{runtime_expected['locked_dependency_count']}"
                    if runtime_checks["locked_dependencies_match"]
                    else "불일치"
                ),
            )
            if runtime_report["status"] == "PASS":
                st.success(
                    "현재 Streamlit 프로세스의 Python·Linux 플랫폼·직접 및 전이 의존성 "
                    "버전이 커밋된 Streamlit 잠금과 모두 일치합니다."
                )
            else:
                st.warning(
                    "현재 프로세스가 Streamlit 잠금과 완전히 같지 않습니다. 로컬 Windows "
                    "실행에서는 정상적인 경고이며, 공개 배포에서 경고가 보이면 Python "
                    "버전과 설치 로그를 확인해야 합니다."
                )
                drift_rows = (
                    [
                        {
                            "범위": "직접",
                            "패키지": item["package"],
                            "기준": item["required"],
                            "실행": item["installed"],
                        }
                        for item in runtime_report["direct_drift"]["mismatched"]
                    ]
                    + [
                        {
                            "범위": "잠금",
                            "패키지": item["package"],
                            "기준": item["required"],
                            "실행": item["installed"],
                        }
                        for item in runtime_report["lock_drift"]["mismatched"]
                    ]
                )
                missing = sorted(
                    set(runtime_report["direct_drift"]["missing"])
                    | set(runtime_report["lock_drift"]["missing"])
                )
                with st.expander("런타임 불일치 상세"):
                    if drift_rows:
                        st.dataframe(pd.DataFrame(drift_rows).drop_duplicates())
                    if missing:
                        st.write("누락 패키지:", ", ".join(missing))
                    if not runtime_checks["python_version_matches"]:
                        st.write(
                            "Python:",
                            runtime_actual["python_version"],
                            "→ 기준",
                            runtime_expected["python_version"],
                        )
            st.caption(
                "런타임 지문: "
                f"{runtime_report['runtime_fingerprint_sha256']} · "
                "패키지 metadata만 사용하며 secrets·환경변수·호스트명은 수집하지 않습니다."
            )
            st.caption(
                f"CI wheel {github_lock['wheel_count']}개 · Streamlit wheel "
                f"{streamlit_lock['wheel_count']}개 · pip {linux_locks['generator_version']} "
                "experimental"
            )
            linux_downloads = st.columns(4)
            linux_downloads[0].download_button(
                "GitHub Actions pylock",
                linux_ci_lock_path.read_bytes(),
                "pylock.github-actions.toml",
                "application/toml",
                key="secom_linux_ci_pylock_download",
            )
            linux_downloads[1].download_button(
                "Streamlit pylock",
                streamlit_lock_path.read_bytes(),
                "pylock.streamlit.toml",
                "application/toml",
                key="secom_streamlit_pylock_download",
            )
            linux_downloads[2].download_button(
                "Linux 잠금 manifest",
                linux_lock_manifest_path.read_bytes(),
                "secom_linux_lock_manifest.json",
                "application/json",
                key="secom_linux_lock_manifest_download",
            )
            linux_downloads[3].download_button(
                "현재 런타임 대조 JSON",
                runtime_report_json_bytes(runtime_report),
                "secom_streamlit_runtime_report.json",
                "application/json",
                key="secom_streamlit_runtime_report_download",
            )
        dependency_security_path = (
            DEPENDENCY_SECURITY_DIR / "dependency_security_audit.json"
        )
        if dependency_security_path.is_file():
            dependency_security = json.loads(
                dependency_security_path.read_text(encoding="utf-8")
            )
            st.divider()
            st.subheader("Python 의존성 취약점 감사")
            security_cols = st.columns(5)
            security_cols[0].metric("보안 판정", dependency_security["status"])
            security_cols[1].metric(
                "검사 패키지", dependency_security["audited_distribution_count"]
            )
            security_cols[2].metric(
                "취약 패키지", dependency_security["vulnerable_package_count"]
            )
            security_cols[3].metric(
                "알려진 취약점", dependency_security["known_vulnerability_count"]
            )
            security_cols[4].metric("pip", dependency_security["pip_version"])
            if dependency_security["status"] == "PASS":
                st.success(
                    f"pip-audit {dependency_security['scanner_version']}와 PyPI advisory 기준으로 "
                    "requirements.txt의 해석된 의존성 그래프에서 검사 시점에 알려진 "
                    "Python 취약점이 발견되지 않았습니다."
                )
            else:
                st.error("알려진 Python 의존성 취약점이 있어 릴리스가 차단됩니다.")
            st.caption(
                f"검사 시각: {dependency_security['audited_at']} · 범위: requirements.txt 해석 그래프 · "
                "이 결과는 시점성 조회이며 "
                "운영체제·드라이버·CUDA·미공개 취약점까지 안전하다는 보증이 아닙니다."
            )
            st.download_button(
                "의존성 보안 감사 JSON",
                dependency_security_path.read_bytes(),
                "secom_dependency_security_audit.json",
                "application/json",
                key="secom_dependency_security_audit_download",
            )
        source_security_path = SOURCE_SECURITY_DIR / "source_security_audit.json"
        if source_security_path.is_file():
            source_security = json.loads(source_security_path.read_text(encoding="utf-8"))
            st.divider()
            st.subheader("Python 소스 보안 정적분석")
            source_cols = st.columns(5)
            source_cols[0].metric("보안 판정", source_security["status"])
            source_cols[1].metric("검사 파일", source_security["scanned_file_count"])
            source_cols[2].metric("Python LOC", source_security["lines_of_code"])
            source_cols[3].metric("중간 이상 이슈", source_security["issue_count"])
            source_cols[4].metric("nosec 억제", source_security["suppression_count"])
            if source_security["status"] == "PASS":
                st.success(
                    f"Bandit {source_security['scanner_version']} 기준으로 app.py와 SECOM "
                    "Python 코드에서 중간 이상 심각도·신뢰도 이슈가 발견되지 않았습니다."
                )
            else:
                st.error("소스 보안 이슈·스캐너 오류 또는 nosec 억제가 있어 릴리스가 차단됩니다.")
            st.caption(
                f"검사 시각: {source_security['scanned_at']} · AST 규칙 검사이며 "
                "동적 취약점과 WM-811K 코드까지 안전하다는 보증은 아닙니다."
            )
            st.download_button(
                "소스 보안 감사 JSON",
                source_security_path.read_bytes(),
                "secom_source_security_audit.json",
                "application/json",
                key="secom_source_security_audit_download",
            )
        st.caption(readiness["interpretation"])
    else:
        st.info("release_readiness 결과가 없습니다. build_release_readiness_report.py를 실행하세요.")

def render_secom_guide() -> None:
    st.markdown(
        """
        1. 공백으로 구분된 센서 변수 590개짜리 파일을 업로드합니다.
        2. **균형형**은 CatBoost의 F2 균형을 확인할 때 적합합니다.
        3. **목표 재현율 80~90%**는 불량 누락을 줄이지만 오탐이 크게 늘어납니다.
        4. `두 모델 모두 불량`을 우선 검토하고, 한 모델만 불량인 행은 2차 확인합니다.
        5. 개별 SHAP 그래프에서 양수는 불량 판단을, 음수는 정상 판단을 강화합니다.

        표시되는 점수는 불균형 보정 가중치로 학습한 모델의 출력입니다. 확률 보정 전이므로
        실제 불량 발생 확률과 동일하다고 해석하지 마세요.
        """
    )


secom_renderers = {
    "diagnosis": render_secom_diagnosis,
    "explanation": render_secom_explanation,
    "safety": render_secom_safety,
    "process": render_secom_process,
    "monitoring": render_secom_monitoring,
    "release": render_secom_release,
    "guide": render_secom_guide,
}
if not dashboard_access.section_allowed(
    secom_section, dashboard_access.SECOM_ADMIN_SECTIONS, ACCESS
):
    st.error("관리자 권한이 필요한 화면입니다.")
    st.stop()
secom_renderers[secom_section]()
site.render_footer(repository_url=REPOSITORY_URL, project_page=project_page)
