"""Public website pages around the SHAPGPT analysis dashboards.

The home, project, account and administrator pages only present evidence
that the dashboard already produced: every number comes from a result file
passed in by ``app.py`` and nothing here recomputes models or thresholds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape

import streamlit as st

from dashboard_ui import access as dashboard_access
from dashboard_ui.components import render_dashboard_header
from dashboard_ui.navigation import render_flat_navigation


LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="150" height="32" '
    'viewBox="0 0 150 32" role="img" aria-label="SHAPGPT">'
    '<text x="0" y="24" font-family="Segoe UI, Pretendard, Apple SD Gothic Neo, '
    'Arial, sans-serif" font-size="22" font-weight="800" letter-spacing="0.6" '
    'fill="#0F766E">SHAP<tspan fill="#0B2545">GPT</tspan></text></svg>'
)
HERO_SENTENCE = (
    "SHAPGPT는 반도체 공정 센서(SECOM)와 웨이퍼 맵(WM-811K)을 AI로 진단하고, "
    "각 판정의 근거를 SHAP으로 함께 보여 주는 설명 가능한 AI 프로젝트입니다."
)
FLOW_STEPS = (
    (
        "입력 신뢰도",
        "입력이 학습 데이터와 비슷한지 먼저 봅니다. 결측·범위 이탈과 분포 이탈(OOD)을 확인합니다.",
    ),
    (
        "모델 판정",
        "모델 점수를 미리 고정한 임계값과 비교해 정상·불량과 결함 유형을 판정합니다.",
    ),
    (
        "SHAP 기여",
        "어떤 센서와 어느 위치의 die가 판정을 얼마나 밀었는지 보여 줍니다. 원인 확률은 아닙니다.",
    ),
    (
        "전문가 검토",
        "점수가 낮거나 입력이 낯설면 자동 확정하지 않고 전문가 검토로 넘깁니다.",
    ),
)
USABLE_SCOPE = (
    "공개 데이터(SECOM·WM-811K)로 학습한 모델의 판정과 SHAP 근거를 확인하는 연구·교육용 시연",
    "과거 test 대표 샘플과 직접 올린 센서·웨이퍼 맵 파일의 진단 보조",
    "신뢰도가 낮거나 분포를 벗어난 입력을 전문가 검토로 넘기는 안전장치 확인",
)
GLOSSARY = (
    ("SHAP", "각 입력(센서·die 위치)이 모델 점수를 얼마나 올리거나 내렸는지 나눠 보여 주는 설명 방법입니다."),
    ("TreeSHAP", "트리 모델(CatBoost·XGBoost)의 SHAP 값을 정확히 계산하는 방법으로, SECOM 센서별 기여를 구합니다."),
    ("Gradient SHAP", "같은 웨이퍼 외형에서 모든 die가 정상인 기준 맵과 비교해 die 위치별 기여를 근사합니다."),
    ("OOD(분포 이탈)", "학습 데이터와 크게 다른 입력입니다. 모델 점수와 관계없이 자동 판정을 보류하고 검토합니다."),
    ("보정 모델 점수", "Validation 데이터로 온도(temperature)를 맞춘 CNN 점수입니다. 높아도 정답이 확실하다는 뜻은 아닙니다."),
    ("Macro-F1", "클래스마다 정밀도와 재현율의 조화평균을 구해 단순 평균한 값으로, 드문 결함 유형도 같은 비중으로 반영합니다."),
    ("재현율", "실제 불량 가운데 모델이 불량으로 찾아낸 비율입니다."),
    ("PR-AUC", "정밀도-재현율 곡선 아래 면적입니다. 불량이 드문 데이터에서 정확도보다 성능을 잘 드러냅니다."),
    ("가산성", "SHAP 기여를 모두 더하면 기준 대비 모델 점수(로짓) 변화와 같아야 한다는 성질입니다."),
)
CHECK_STATUS_LABELS = {"PASS": "통과", "WARN": "경고", "BLOCK": "생산 전 필요 조건"}
DEMO_READINESS_LABELS = {
    "READY": "시연 가능",
    "READY_WITH_WARNINGS": "시연 가능",
    "BLOCKED": "시연 보류",
}
PRODUCTION_READINESS_LABELS = {
    "READY": "생산 배포 가능",
    "READY_WITH_WARNINGS": "생산 배포 가능 · 경고 있음",
    "BLOCKED": "생산 배포 승인 전",
}
LOGIN_UNAVAILABLE_MESSAGE = (
    "OIDC 로그인 설정이 없어 로그인할 수 없습니다. 배포 관리자가 Streamlit "
    "secrets의 [auth] 항목을 설정해야 합니다."
)

SITE_CSS = """
<style>
:root {
    --site-navy: #0b2545;
    --site-navy-2: #13315c;
    --site-teal: #0f766e;
    --site-teal-soft: #e6f4f2;
    --site-ink: #102a43;
    --site-muted: #52606d;
    --site-line: #e2e8f0;
    --site-soft: #f5f8fa;
    --site-alert: #b42318;
    --site-alert-soft: #fef3f2;
}

.st-key-site_hero {
    padding: 2.4rem 2.2rem 2rem;
    margin-bottom: 1.4rem;
    border: 1px solid var(--site-line);
    border-radius: 1rem;
    background: #ffffff;
}

.site-eyebrow {
    margin: 0 0 0.6rem;
    color: var(--site-teal);
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 0.12em;
}

[data-testid="stMarkdownContainer"] h1.site-hero__title {
    margin: 0 0 0.8rem;
    padding: 0;
    color: var(--site-navy);
    font-size: clamp(1.7rem, 3.2vw, 2.55rem);
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.03em;
}

.site-hero__lead {
    max-width: 46rem;
    margin: 0 0 1.4rem;
    color: var(--site-muted);
    font-size: 1.05rem;
    line-height: 1.65;
}

[data-testid="stMarkdownContainer"] h2.site-section-title {
    margin: 2.2rem 0 0.35rem;
    padding: 0;
    color: var(--site-navy);
    font-size: 1.35rem;
    font-weight: 780;
    line-height: 1.3;
    letter-spacing: -0.02em;
}

.st-key-site_cta_primary [data-testid="stPageLink"] a,
.st-key-site_cta_secondary [data-testid="stPageLink"] a {
    justify-content: center;
    padding: 0.5rem 1rem;
    border: 1px solid var(--site-teal);
    border-radius: 0.6rem;
}

.st-key-site_cta_primary [data-testid="stPageLink"] a {
    background: var(--site-teal);
}

.st-key-site_cta_primary [data-testid="stPageLink"] a p {
    color: #ffffff;
    font-weight: 700;
}

.st-key-site_cta_secondary [data-testid="stPageLink"] a p {
    color: var(--site-teal);
    font-weight: 700;
}

/* Cards in one row share the row height. */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"]:has(
    > [data-testid="stLayoutWrapper"] > [class*="st-key-site_"]
) {
    height: 100%;
}

[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(
    > [class*="st-key-site_"]
) {
    flex: 1 1 auto;
}

[data-testid="stLayoutWrapper"] > [class*="st-key-site_metric"],
[data-testid="stLayoutWrapper"] > [class*="st-key-site_card"],
[data-testid="stLayoutWrapper"] > [class*="st-key-site_scope"],
[data-testid="stLayoutWrapper"] > [class*="st-key-site_admin"] {
    height: 100%;
}

[class*="st-key-site_metric"] [data-testid="stMetric"] {
    min-height: 0;
    padding: 0;
    border: 0;
    background: transparent;
    box-shadow: none;
}

[class*="st-key-site_metric"] [data-testid="stMetricLabel"] p {
    overflow: visible;
    white-space: normal;
}

.site-section-lead {
    margin: 0 0 1rem;
    color: var(--site-muted);
    font-size: 0.92rem;
    line-height: 1.6;
}

.site-card h3 {
    margin: 0.4rem 0 0.45rem;
    color: var(--site-navy);
    font-size: 1.12rem;
    font-weight: 780;
}

.site-card p,
.site-card li {
    color: var(--site-muted);
    font-size: 0.9rem;
    line-height: 1.6;
}

.site-card ul {
    margin: 0.4rem 0 0.2rem;
    padding-left: 1.1rem;
}

.site-card {
    min-height: 13.5rem;
}

.site-flow {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.8rem;
}

.site-step {
    position: relative;
    padding: 1rem 1.05rem;
    border: 1px solid var(--site-line);
    border-radius: 0.85rem;
    background: #ffffff;
}

.site-step__number {
    color: var(--site-teal);
    font-size: 0.78rem;
    font-weight: 780;
    letter-spacing: 0.08em;
}

.site-step h4 {
    margin: 0.25rem 0 0.35rem;
    padding: 0;
    color: var(--site-navy);
    font-size: 1rem;
}

.site-step p {
    margin: 0;
    color: var(--site-muted);
    font-size: 0.86rem;
    line-height: 1.55;
}

.site-step:not(:last-child)::after {
    content: "\\2192";
    position: absolute;
    top: 50%;
    right: -0.72rem;
    z-index: 1;
    color: var(--site-teal);
    font-weight: 800;
    transform: translateY(-50%);
}

[role="radiogroup"][aria-label="검증 자료 영역"] {
    display: flex;
    justify-content: flex-start;
    gap: 1.35rem;
    min-height: 2.6rem;
    margin: 0.1rem 0 1rem;
    padding: 0;
    overflow-x: auto;
    border: 0;
    border-bottom: 1px solid var(--site-line);
    border-radius: 0;
    background: transparent;
    scrollbar-width: thin;
}

[role="radiogroup"][aria-label="검증 자료 영역"] > button {
    flex: 0 0 auto !important;
    min-height: 2.45rem;
    padding: 0.42rem 0.1rem 0.52rem !important;
    border: 0 !important;
    border-radius: 0 !important;
    color: #737983 !important;
    background: transparent !important;
    box-shadow: none !important;
    font-size: 0.9rem;
    font-weight: 650;
}

[role="radiogroup"][aria-label="검증 자료 영역"] > button:hover {
    color: var(--site-navy) !important;
}

[role="radiogroup"][aria-label="검증 자료 영역"] > button[data-selected="true"] {
    color: var(--site-navy) !important;
    box-shadow: inset 0 -2px 0 var(--site-teal) !important;
    font-weight: 780;
}

.site-badge {
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border: 1px solid var(--site-line);
    border-radius: 999px;
    color: var(--site-navy-2);
    background: var(--site-soft);
    font-size: 0.74rem;
    font-weight: 700;
}

.site-badge--teal {
    border-color: #b7e1db;
    color: var(--site-teal);
    background: var(--site-teal-soft);
}

.site-condition {
    padding: 0.75rem 0 0.7rem;
    border-top: 1px solid var(--site-line);
}

.site-condition h4 {
    margin: 0 0 0.3rem;
    padding: 0;
    color: var(--site-navy);
    font-size: 0.95rem;
}

.site-condition p {
    margin: 0.1rem 0;
    color: var(--site-muted);
    font-size: 0.84rem;
    line-height: 1.55;
}

.site-condition b {
    color: var(--site-ink);
}

.site-source {
    color: #7b8794;
    font-size: 0.72rem;
}

.st-key-site_account_strip {
    margin: -0.4rem 0 0.6rem;
}

.st-key-site_account_strip [data-testid="stCaptionContainer"] {
    text-align: right;
}

.st-key-site_footer {
    margin-top: 3rem;
    padding: 1.4rem 0 0.4rem;
    border-top: 1px solid var(--site-line);
}

.site-footer__brand {
    margin: 0;
    color: var(--site-navy);
    font-weight: 800;
    letter-spacing: 0.02em;
}

.site-footer__copy {
    margin: 0.25rem 0 0;
    color: var(--site-muted);
    font-size: 0.82rem;
    line-height: 1.55;
}

.site-footer__copy a {
    color: var(--site-teal);
}

.wafer-legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem 1rem;
    margin: 0.35rem 0 0.2rem;
    color: var(--site-muted);
    font-size: 0.8rem;
}

.wafer-legend__item {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.wafer-legend__swatch {
    display: inline-block;
    width: 0.9rem;
    height: 0.9rem;
    border: 1px solid rgba(11, 37, 69, 0.25);
    border-radius: 0.2rem;
}

.explanation-scale {
    display: inline-block;
    width: 9rem;
    height: 0.7rem;
    border: 1px solid rgba(11, 37, 69, 0.25);
    border-radius: 0.2rem;
}

[class*="st-key-wafer_map"] img,
[class*="st-key-wm_card_"] img {
    max-width: 100%;
    height: auto;
    image-rendering: crisp-edges;
    image-rendering: pixelated;
    border: 1px solid var(--site-line);
    border-radius: 0.3rem;
}

[class*="st-key-wm_card_"] [data-testid="stElementContainer"]:has([data-testid="stImage"]) {
    align-self: center;
}

[class*="st-key-wm_card_selected_"] {
    border: 2px solid var(--site-teal) !important;
    box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14);
}

.wm-card__badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin: 0.1rem 0 0.45rem;
}

.wm-badge {
    padding: 0.14rem 0.5rem;
    border: 1px solid var(--site-line);
    border-radius: 999px;
    color: var(--site-navy-2);
    background: var(--site-soft);
    font-size: 0.72rem;
    font-weight: 700;
}

.wm-badge--selected {
    border-color: var(--site-teal);
    color: #ffffff;
    background: var(--site-teal);
}

.wm-card__facts {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.35rem 0.6rem;
    margin: 0 0 0.3rem;
}

.wm-card__facts dt {
    color: #7b8794;
    font-size: 0.7rem;
    font-weight: 650;
}

.wm-card__facts dd {
    margin: 0;
    color: var(--site-ink);
    font-size: 0.84rem;
    font-weight: 700;
    overflow-wrap: anywhere;
}

.wm-card__facts dd.wm-card__review {
    color: var(--site-alert);
}

@media (max-width: 1100px) {
    .st-key-wm_gallery [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
    }

    .st-key-wm_gallery [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 calc(50% - 1rem) !important;
        min-width: 15rem !important;
    }

    .site-flow {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .site-step:nth-child(2)::after {
        display: none;
    }
}

@media (max-width: 640px) {
    .st-key-site_hero {
        padding: 1.5rem 1.1rem 1.3rem;
    }

    .site-flow {
        grid-template-columns: minmax(0, 1fr);
    }

    .site-step::after {
        display: none;
    }

    .site-card {
        min-height: 0;
    }

    .st-key-site_account_strip [data-testid="stCaptionContainer"] {
        text-align: left;
    }
}
</style>
"""


WM_EVALUATION = "WM-811K test evaluation"
WM_REVIEW_POLICY = "WM-811K review policy"
WM_SHAP_VALIDATION = "WM-811K Gradient SHAP validation"
SECOM_REPEATED_CV = "SECOM repeated CV"
WM_RESULTS = "결과물/wm811k/selected_model_results"
SECOM_RESULTS = "결과물/secom/dual_model_results"


@dataclass(frozen=True)
class MetricCard:
    label: str
    value: str
    caption: str
    help: str
    source_label: str
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class ReadinessSummary:
    demo_label: str
    production_label: str
    warning_count: int
    blockers: tuple[dict, ...]
    checks: tuple[dict, ...]
    interpretation: str


def inject_site_style() -> None:
    st.markdown(SITE_CSS, unsafe_allow_html=True)


def render_logo() -> None:
    st.logo(LOGO_SVG, size="large")


def readiness_summary(report: Mapping[str, object] | None) -> ReadinessSummary | None:
    """Translate the release readiness report for public pages."""
    if not report:
        return None
    checks = tuple(dict(item) for item in report.get("checks", ()))
    return ReadinessSummary(
        demo_label=DEMO_READINESS_LABELS.get(
            str(report.get("demo_readiness")), str(report.get("demo_readiness"))
        ),
        production_label=PRODUCTION_READINESS_LABELS.get(
            str(report.get("production_readiness")),
            str(report.get("production_readiness")),
        ),
        warning_count=sum(item.get("status") == "WARN" for item in checks),
        blockers=tuple(item for item in checks if item.get("status") == "BLOCK"),
        checks=checks,
        interpretation=str(report.get("interpretation") or ""),
    )


def _shap_card(shap: Mapping[str, object] | None) -> MetricCard:
    sources = (
        "결과물/wm811k/xai_gradient_shap_validation/xai_method_summary.json",
        "결과물/wm811k/xai_gradient_shap_validation/xai_method_records.csv",
    )
    help_text = (
        "가산성: SHAP 기여를 모두 더하면 기준 대비 모델 점수(로짓) 변화와 같아야 합니다. "
        "그 차이가 허용오차 안인지 웨이퍼마다 확인했습니다."
    )
    if not shap or shap.get("status") != "verified":
        return MetricCard(
            "Gradient SHAP 원시 가산성",
            "검증 증거 없음",
            "현재 배포 체크포인트와 일치하는 검증 결과가 없어 수치를 표시하지 않습니다.",
            help_text,
            WM_SHAP_VALIDATION,
            sources,
        )
    return MetricCard(
        "Gradient SHAP 원시 가산성",
        f"{shap['pass_count']}/{shap['sample_count']}",
        f"WM-811K · 기여 합계와 점수 변화의 차이가 {shap['relative_tolerance']:.0%} "
        "이내인 고정 test 웨이퍼 수",
        help_text,
        WM_SHAP_VALIDATION,
        sources,
    )


def _wm_headline_cards(wm: Mapping[str, object]) -> list[MetricCard]:
    return [
        MetricCard(
            "WM-811K Macro-F1",
            f"{wm['macro_f1']:.3f}",
            f"Lot 비중복 고정 test {wm['test_rows']:,}개 · {wm['class_count']}개 클래스 평균",
            GLOSSARY[5][1],
            WM_EVALUATION,
            (f"{WM_RESULTS}/run_summary.json",),
        ),
        MetricCard(
            "WM-811K 불량 탐지 재현율",
            f"{wm['recall']:.1%}",
            "실제 불량 웨이퍼 가운데 불량으로 찾아낸 비율(정상·불량 이진 기준)",
            GLOSSARY[6][1],
            WM_EVALUATION,
            (f"{WM_RESULTS}/test_confusion_matrix.csv",),
        ),
    ]


def home_metric_cards(evidence: Mapping[str, object]) -> list[MetricCard]:
    """Four headline numbers, each read from an existing result file."""
    cards: list[MetricCard] = []
    wm = evidence.get("wm")
    if wm:
        cards.extend(_wm_headline_cards(wm))
    cards.append(_shap_card(evidence.get("shap")))
    secom = {row["model"]: row for row in evidence.get("secom") or ()}
    if "CatBoost" in secom:
        row = secom["CatBoost"]
        cards.append(
            MetricCard(
                "SECOM PR-AUC · CatBoost",
                f"{row['pr_auc']:.3f}",
                f"반복 교차검증({row['folds']}-fold × {row['repeats']}회) · "
                f"정상:불량 약 {row['normal_per_defect']:.0f}:1 불균형 데이터",
                GLOSSARY[7][1],
                SECOM_REPEATED_CV,
                (
                    f"{SECOM_RESULTS}/oof_operating_points.csv",
                    f"{SECOM_RESULTS}/repeated_cv_fold_auc.csv",
                ),
            )
        )
    return cards


def wm_metric_cards(evidence: Mapping[str, object]) -> list[MetricCard]:
    """WM-811K test and review-policy numbers for the project page."""
    wm = evidence.get("wm")
    if not wm:
        return []
    return _wm_headline_cards(wm) + [
        MetricCard(
            "WM-811K test 정확도",
            f"{wm['accuracy']:.2%}",
            "다수 클래스(none)의 영향을 크게 받으므로 Macro-F1과 함께 봅니다.",
            "전체 test 웨이퍼 중 클래스를 맞힌 비율입니다.",
            WM_EVALUATION,
            (f"{WM_RESULTS}/run_summary.json",),
        ),
        MetricCard(
            "WM-811K 균형 정확도",
            f"{wm['balanced_accuracy']:.2%}",
            "클래스별 재현율의 평균",
            "클래스마다 맞힌 비율을 구해 평균한 값으로, 드문 클래스도 같은 비중입니다.",
            WM_EVALUATION,
            (f"{WM_RESULTS}/run_summary.json",),
        ),
        MetricCard(
            "자동 분류 비율",
            f"{wm['automatic_coverage']:.1%}",
            "보정 점수가 검토 기준 이상이라 자동 분류한 test 비율",
            "나머지는 전문가 검토로 넘깁니다. 기준은 Validation에서 고정했습니다.",
            WM_REVIEW_POLICY,
            (f"{WM_RESULTS}/uncertainty_policy.json",),
        ),
        MetricCard(
            "오류 검토 포착률",
            f"{wm['error_capture_rate']:.1%}",
            "모델이 틀린 test 웨이퍼 가운데 검토 대상으로 넘어간 비율",
            "검토 정책이 오류를 얼마나 걸러 내는지 보여 줍니다.",
            WM_REVIEW_POLICY,
            (f"{WM_RESULTS}/uncertainty_policy.json",),
        ),
    ]


def render_metric_cards(cards: Sequence[MetricCard], *, key: str) -> None:
    """Cards name their source briefly; exact files sit in one expander below."""
    if not cards:
        st.info("표시할 검증 결과 파일이 없습니다.")
        return
    for start in range(0, len(cards), 4):
        columns = st.columns(4, gap="small")
        for index, (column, card) in enumerate(zip(columns, cards[start : start + 4])):
            with column, st.container(border=True, key=f"{key}_{start + index}"):
                st.metric(card.label, card.value, help=card.help)
                st.caption(card.caption)
                st.markdown(
                    f'<span class="site-source">출처: {escape(card.source_label)}</span>',
                    unsafe_allow_html=True,
                )
    with st.expander("검증 근거 보기"):
        st.dataframe(
            [
                {
                    "지표": card.label,
                    "값": card.value,
                    "출처": card.source_label,
                    "근거 파일": ", ".join(card.source_files),
                    "조건": card.caption,
                }
                for card in cards
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "근거 파일은 저장소 기준 경로입니다. 화면의 값은 이 파일을 읽어 표시하며 "
            "다시 계산하지 않습니다."
        )


def _section_title(title: str, lead: str | None = None) -> None:
    html = f'<h2 class="site-section-title">{escape(title)}</h2>'
    if lead:
        html += f'<p class="site-section-lead">{escape(lead)}</p>'
    st.markdown(html, unsafe_allow_html=True)


def render_login_button(*, auth_ready: bool, key: str, width: str = "content") -> None:
    st.button(
        "Google로 로그인",
        type="primary",
        key=key,
        on_click=dashboard_access.start_login,
        disabled=not auth_ready,
        width=width,
    )


def render_login_gate(*, auth_ready: bool) -> None:
    """Shown instead of a diagnosis page to viewers who are not signed in."""
    render_dashboard_header(
        "SHAPGPT 로그인",
        "진단 기능은 로그인한 사용자만 사용할 수 있습니다. 홈과 프로젝트·검증 "
        "페이지는 로그인 없이 볼 수 있습니다. 관리자 화면은 로그인한 계정이 관리자 "
        "허용 목록에 있을 때만 열립니다.",
    )
    render_login_button(auth_ready=auth_ready, key="gate_login")
    if not auth_ready:
        st.error(LOGIN_UNAVAILABLE_MESSAGE)


def render_account_strip(access: dashboard_access.AccessContext) -> None:
    """One compact line with the signed-in account and a logout button."""
    if access.login_required and access.logged_in:
        with st.container(key="site_account_strip"):
            columns = st.columns((5, 1), vertical_alignment="center")
            columns[0].caption(f"로그인: {access.user_label} · {access.role_label}")
            columns[1].button(
                "로그아웃", key="strip_logout", on_click=st.logout, type="tertiary"
            )
    elif access.is_admin:
        with st.container(key="site_account_strip"):
            st.caption("로컬 개발 관리자 모드 · 로그인 없음")


def _flow_html() -> str:
    steps = "".join(
        '<div class="site-step">'
        f'<span class="site-step__number">STEP {index}</span>'
        f"<h4>{escape(title)}</h4><p>{escape(body)}</p></div>"
        for index, (title, body) in enumerate(FLOW_STEPS, start=1)
    )
    return f'<div class="site-flow" aria-label="분석 4단계 흐름">{steps}</div>'


def _project_card(
    *, tag: str, title: str, body: str, facts: Sequence[str], page, link_label: str, key: str
) -> None:
    with st.container(border=True, key=key):
        items = "".join(f"<li>{escape(fact)}</li>" for fact in facts)
        st.markdown(
            '<div class="site-card">'
            f'<span class="site-badge">{escape(tag)}</span>'
            f"<h3>{escape(title)}</h3><p>{escape(body)}</p><ul>{items}</ul></div>",
            unsafe_allow_html=True,
        )
        st.page_link(page, label=link_label, icon=":material/arrow_forward:")


def _render_project_cards(evidence: Mapping[str, object], pages: Mapping[str, object]) -> None:
    wm = evidence.get("wm") or {}
    class_text = f"{wm['class_count']}개 유형" if wm else "결함 유형"
    columns = st.columns(2, gap="medium")
    with columns[0]:
        _project_card(
            tag="공정 센서 · 표 데이터",
            title="SECOM 공정 센서 진단",
            body=(
                "익명화된 공정 센서 590개 값으로 웨이퍼 불량 여부를 판정하고, "
                "두 모델의 점수와 센서별 기여를 나란히 보여 줍니다."
            ),
            facts=(
                "모델: CatBoost 균형형 · XGBoost 고재현율",
                "설명: TreeSHAP 센서별 기여",
                "안전장치: 결측·범위 이탈·OOD 확인과 검토 대기열",
            ),
            page=pages["secom"],
            link_label="SECOM 진단 열기",
            key="site_card_secom",
        )
    with columns[1]:
        _project_card(
            tag="웨이퍼 맵 · 이미지",
            title="WM-811K 웨이퍼 맵 진단",
            body=(
                f"웨이퍼 맵의 결함 die 배치로 정상을 포함한 {class_text}을 분류하고, "
                "판정에 기여한 die 위치를 지도로 보여 줍니다."
            ),
            facts=(
                "모델: CNN · Validation 온도 보정 점수",
                "설명: Gradient SHAP 위치 기여 · Grad-CAM 대조",
                "안전장치: 보정 점수 검토 기준과 OOD 판정 보류",
            ),
            page=pages["wm811k"],
            link_label="WM-811K 진단 열기",
            key="site_card_wm811k",
        )


def _render_scope(readiness: ReadinessSummary | None) -> None:
    columns = st.columns(2, gap="medium")
    with columns[0], st.container(border=True, key="site_scope_usable"):
        badge = ""
        if readiness:
            warnings = (
                f" · 경고 {readiness.warning_count}건" if readiness.warning_count else ""
            )
            badge = (
                f'<span class="site-badge site-badge--teal">'
                f"{escape(readiness.demo_label)}{warnings}</span>"
            )
        items = "".join(f"<li>{escape(item)}</li>" for item in USABLE_SCOPE)
        st.markdown(
            f'<div class="site-card">{badge}<h3>지금 사용할 수 있는 범위</h3>'
            f"<ul>{items}</ul><p>모든 판정은 공정 엔지니어의 검토를 돕는 참고 정보입니다.</p></div>",
            unsafe_allow_html=True,
        )
    with columns[1], st.container(border=True, key="site_scope_production"):
        if readiness is None:
            st.markdown(
                '<div class="site-card"><h3>생산 적용 한계</h3>'
                "<p>릴리스 준비도 결과 파일이 없어 생산 적용 조건을 표시할 수 없습니다.</p></div>",
                unsafe_allow_html=True,
            )
            return
        conditions = "".join(
            '<div class="site-condition">'
            f"<h4>{escape(str(item.get('title', item.get('check_id', ''))))}</h4>"
            f"<p><b>현재</b> {escape(str(item.get('evidence', '')))}</p>"
            f"<p><b>필요</b> {escape(str(item.get('required_action', '')))}</p></div>"
            for item in readiness.blockers
        )
        st.markdown(
            f'<div class="site-card"><span class="site-badge">{escape(readiness.production_label)}'
            f" · 조건 {len(readiness.blockers)}건</span><h3>생산 적용 전 필요한 조건</h3>"
            "<p>실제 생산 라인의 자동 판정에는 아직 쓸 수 없습니다. 아래 조건은 모델 오류가 "
            "아니라 실제 장비 정보, 새 데이터, 운영 인프라가 있어야 채울 수 있는 승인 전 "
            f"조건입니다.</p>{conditions}</div>",
            unsafe_allow_html=True,
        )
    if readiness and readiness.interpretation:
        st.caption(readiness.interpretation)


def render_home(
    *,
    access: dashboard_access.AccessContext,
    auth_ready: bool,
    evidence: Mapping[str, object],
    pages: Mapping[str, object],
) -> None:
    with st.container(key="site_hero"):
        st.markdown(
            '<p class="site-eyebrow">SEMICONDUCTOR EXPLAINABLE AI</p>'
            '<h1 class="site-hero__title">설명 가능한 AI로 반도체 공정 데이터를 진단합니다</h1>'
            f'<p class="site-hero__lead">{escape(HERO_SENTENCE)}</p>',
            unsafe_allow_html=True,
        )
        actions = st.columns((1, 1, 3), vertical_alignment="center")
        if access.login_required and not access.logged_in:
            with actions[0]:
                render_login_button(auth_ready=auth_ready, key="home_login", width="stretch")
            with actions[1], st.container(key="site_cta_secondary"):
                st.page_link(pages["project"], label="프로젝트·검증 보기", width="stretch")
            if not auth_ready:
                st.caption(
                    "로그인 설정 전입니다. 배포 관리자가 Google OIDC(secrets의 [auth])를 "
                    "설정하면 버튼이 활성화됩니다."
                )
        else:
            with actions[0], st.container(key="site_cta_primary"):
                st.page_link(pages["secom"], label="SECOM 진단 시작", width="stretch")
            with actions[1], st.container(key="site_cta_secondary"):
                st.page_link(pages["wm811k"], label="WM-811K 진단 시작", width="stretch")
            if not access.login_required:
                st.caption("로그인 없는 데모 모드로 실행 중입니다.")

    _section_title("두 가지 진단", "공정 센서 기록과 웨이퍼 맵, 서로 다른 두 데이터를 같은 방식으로 설명합니다.")
    _render_project_cards(evidence, pages)

    _section_title("분석 4단계", "입력 신뢰도 → 모델 판정 → SHAP 기여 → 전문가 검토 순서로 판단합니다.")
    st.markdown(_flow_html(), unsafe_allow_html=True)

    _section_title(
        "핵심 검증 지표",
        "이미 계산해 저장한 검증 결과에서 읽은 값입니다. 새 데이터에서의 성능을 보장하지 않습니다.",
    )
    render_metric_cards(home_metric_cards(evidence), key="site_metric_home")

    readiness = readiness_summary(evidence.get("readiness"))
    _section_title("사용 범위와 생산 적용 한계")
    _render_scope(readiness)


PROJECT_AREAS = (
    ("SECOM 모델 검증", "secom_validation"),
    ("SECOM 증강 검증", "secom_augmentation"),
    ("SHAP/XAI 검증 근거", "xai"),
    ("WM-811K 검증 자료", "wm811k"),
)


def render_project(
    *,
    evidence: Mapping[str, object],
    pages: Mapping[str, object],
    area_renderers: Mapping[str, Callable[[], None]],
) -> None:
    render_dashboard_header(
        "프로젝트·검증",
        "SHAPGPT가 무엇을 판정하고 어디까지 검증했는지 요약합니다. 수치는 모두 저장된 "
        "검증 결과 파일에서 읽었습니다.",
    )
    _section_title("프로젝트 개요")
    st.markdown(
        "SHAPGPT는 반도체 공정의 두 데이터, 공정 센서 기록(SECOM)과 웨이퍼 맵(WM-811K)에서 "
        "불량을 판정하고 그 근거를 SHAP으로 설명합니다. 모델은 공정 엔지니어의 판단을 "
        "보조하며, 신뢰도가 낮거나 입력이 학습 분포를 벗어나면 자동 확정 대신 전문가 "
        "검토로 넘깁니다."
    )
    _render_project_cards(evidence, pages)

    _section_title("분석 4단계")
    st.markdown(_flow_html(), unsafe_allow_html=True)

    _section_title("핵심 검증 지표", "요약 수치입니다. 근거 파일은 카드 아래 펼침 메뉴에 있습니다.")
    render_metric_cards(home_metric_cards(evidence), key="site_metric_project")

    readiness = readiness_summary(evidence.get("readiness"))
    _section_title("사용 범위와 생산 적용 한계")
    _render_scope(readiness)
    if readiness is not None:
        with st.expander("릴리스 준비도 전체 점검 항목"):
            st.dataframe(
                [
                    {
                        "점검": item.get("title", item.get("check_id")),
                        "판정": CHECK_STATUS_LABELS.get(item.get("status"), item.get("status")),
                        "근거": item.get("evidence"),
                    }
                    for item in readiness.checks
                ],
                hide_index=True,
                width="stretch",
            )

    _section_title("용어 설명")
    with st.expander("처음 보는 용어를 짧게 설명합니다", expanded=False):
        st.markdown("\n".join(f"- **{term}**: {text}" for term, text in GLOSSARY))

    _section_title(
        "검증 자료",
        "기술 검증 자료를 영역별로 나눠 둡니다. 선택한 영역만 불러오며, 수치·표·그래프는 "
        "저장된 검증 결과를 그대로 보여 줍니다.",
    )
    area = render_flat_navigation(PROJECT_AREAS, key="project_area", label="검증 자료 영역")
    with st.container(key="site_project_area"):
        area_renderers[area]()


def render_account(
    *,
    access: dashboard_access.AccessContext,
    auth_ready: bool,
    provider_label: str,
    admin_page=None,
) -> None:
    if not access.logged_in:
        render_dashboard_header(
            "로그인",
            "Google 계정으로 로그인하면 SECOM·WM-811K 진단을 사용할 수 있습니다. 이 서비스는 "
            "비밀번호를 저장하지 않습니다.",
        )
        render_login_button(auth_ready=auth_ready, key="account_login")
        if not auth_ready:
            st.error(LOGIN_UNAVAILABLE_MESSAGE)
        return
    render_dashboard_header("계정", "로그인한 계정과 권한을 확인합니다.")
    with st.container(border=True, key="site_account_card"):
        columns = st.columns(3)
        columns[0].metric("이름", access.display_name or "제공되지 않음")
        columns[1].metric("이메일", access.email or "제공되지 않음")
        columns[2].metric("역할", access.role_label)
        st.caption(
            f"로그인 방식: {provider_label} · 관리자 권한은 배포 관리자가 설정한 허용 목록으로만 "
            "부여됩니다."
        )
        st.button("로그아웃", key="account_logout", on_click=st.logout)
    if access.is_admin and admin_page is not None:
        st.page_link(admin_page, label="관리자 페이지 열기", icon=":material/arrow_forward:")


@dataclass(frozen=True)
class AdminLink:
    title: str
    description: str
    page: object
    query_params: dict


def render_admin_hub(
    *,
    access: dashboard_access.AccessContext,
    auth_ready: bool,
    allowlist_counts: tuple[int, int],
    links: Sequence[AdminLink],
) -> None:
    render_dashboard_header(
        "관리자",
        "관리자 허용 목록에 있는 계정에만 보이는 운영 화면입니다. 일반 사용자에게는 이 "
        "메뉴가 표시되지 않습니다.",
    )
    status = st.columns(4)
    status[0].metric(
        "권한 근거",
        dashboard_access.ADMIN_BASIS_LABELS.get(access.admin_basis, access.admin_basis),
    )
    status[1].metric("로그인 방식", "OIDC 로그인 필수" if access.login_required else "로그인 없음")
    status[2].metric(
        "허용 목록", f"이메일 {allowlist_counts[0]}개 · 주체 {allowlist_counts[1]}개"
    )
    status[3].metric("OIDC 설정", "완료" if auth_ready else "없음")
    if access.notice:
        st.caption(access.notice)
    _section_title("관리자 화면")
    columns = st.columns(len(links) or 1, gap="small")
    for column, link in zip(columns, links):
        with column, st.container(border=True, key=f"site_admin_{link.query_params['section']}"):
            st.markdown(
                f'<div class="site-card"><h3>{escape(link.title)}</h3>'
                f"<p>{escape(link.description)}</p></div>",
                unsafe_allow_html=True,
            )
            st.page_link(
                link.page,
                label="열기",
                icon=":material/arrow_forward:",
                query_params=link.query_params,
            )


def render_footer(*, repository_url: str, project_page) -> None:
    with st.container(key="site_footer"):
        columns = st.columns((3, 1), vertical_alignment="center")
        columns[0].markdown(
            '<p class="site-footer__brand">SHAPGPT</p>'
            '<p class="site-footer__copy">반도체 공정 데이터를 설명 가능한 AI로 진단하는 프로젝트 · '
            "데이터: UCI SECOM, WM-811K(LSWMD) · 연구·교육용 시연이며 생산 판정을 대신하지 "
            f'않습니다. · <a href="{escape(repository_url, quote=True)}" target="_blank" '
            'rel="noopener noreferrer">GitHub 저장소</a></p>',
            unsafe_allow_html=True,
        )
        columns[1].page_link(project_page, label="프로젝트·검증")
