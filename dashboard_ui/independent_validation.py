"""Compact dashboard views for independently reproduced benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


DESCRIPTOR_LABELS = {
    "largest_component_ratio": "가장 큰 결함 덩어리 비율",
    "radial_bin_3": "중간 반경 결함 비율",
    "radial_bin_0": "중심부 결함 비율",
    "radial_bin_7": "가장자리 결함 비율",
    "centroid_offset": "결함 중심 편심",
    "component_count": "분리된 결함 덩어리 수",
    "radius_mean": "결함의 평균 반경",
    "largest_extent": "가장 큰 덩어리 밀집도",
    "largest_boundary_ratio": "가장자리 접촉 비율",
    "component_size_mean": "평균 결함 덩어리 크기",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_wm_independent_validation(
    result_dir: Path, *, standalone: bool = False
) -> None:
    """Show the shape-descriptor control without presenting it as deployment."""
    final_dir = result_dir / "final_comparison"
    comparison_path = final_dir / "model_comparison.csv"
    bootstrap_path = final_dir / "paired_bootstrap.json"
    recall_path = final_dir / "class_recall_comparison.csv"
    descriptor_path = final_dir / "descriptor_glossary_top10.csv"
    disagreement_path = final_dir / "class_disagreement_summary.csv"
    eda_path = result_dir / "eda" / "eda_summary.json"
    required = (
        comparison_path,
        bootstrap_path,
        recall_path,
        descriptor_path,
        disagreement_path,
        eda_path,
    )
    if not all(path.is_file() for path in required):
        if standalone:
            st.info("독립 검증 요약 파일을 준비한 뒤 표시합니다.")
        return

    if standalone:
        st.subheader("독립 검증 · 형상 대조와 데이터 구성")
        st.caption("다른 표현 방식으로 같은 고정 Test를 재현한 검증입니다.")
        section = st.container()
    else:
        section = st.expander(
            "독립 검증 · 형상 대조와 데이터 구성", expanded=False
        )

    with section:
        st.info(
            "배포 모델은 CNN입니다. 아래 결과는 같은 lot 비중복 Test를 다른 표현 방식으로 "
            "재현한 대조 실험이며 모델 교체 근거가 아닙니다."
        )
        comparison = pd.read_csv(comparison_path)
        bootstrap = _read_json(bootstrap_path)
        shape = bootstrap["observed"]["shape"]
        cnn = bootstrap["observed"]["cnn"]
        macro = bootstrap["overall_difference"]["macro_f1"]
        balanced = bootstrap["overall_difference"]["balanced_accuracy"]

        metrics = st.columns(4)
        metrics[0].metric("형태 모델 Macro-F1", f"{shape['macro_f1']:.4f}")
        metrics[1].metric("CNN Macro-F1", f"{cnn['macro_f1']:.4f}")
        metrics[2].metric(
            "Macro-F1 차이",
            f"{macro['observed_difference']:+.4f}",
            help="형태 서술자 − CNN",
        )
        metrics[3].metric(
            "Balanced accuracy 차이",
            f"{balanced['observed_difference']:+.4f}",
            help="형태 서술자 − CNN",
        )
        st.caption(
            f"대응 bootstrap {bootstrap['resamples']:,}회 · seed={bootstrap['seed']} · "
            f"Macro-F1 차이 95% 구간 {macro['ci95_low']:+.4f}~{macro['ci95_high']:+.4f} · "
            f"Balanced accuracy 차이 95% 구간 {balanced['ci95_low']:+.4f}~{balanced['ci95_high']:+.4f}"
        )
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.warning(
            "두 전체 지표의 차이 구간은 모두 0을 포함합니다. 형상 모델이 CNN보다 우수하다고 "
            "결론내리지 않고, 서로 다른 방식으로 비슷한 결과가 재현됐다는 근거로만 사용합니다."
        )

        recall = pd.read_csv(recall_path)
        recall_display = recall[
            ["클래스", "test 표본수", "형태서술자 recall", "기준선CNN recall", "형태-CNN 차이"]
        ]
        st.markdown("#### 클래스별 재현율")
        st.dataframe(
            recall_display,
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="%.4f")
                for column in ("형태서술자 recall", "기준선CNN recall", "형태-CNN 차이")
            },
        )
        st.caption(
            "Near-full·Donut·Random·Scratch는 Test 표본이 200개 미만입니다. 클래스별 차이는 "
            "표본 수와 paired bootstrap 구간을 함께 확인해야 합니다."
        )

        disagreement = pd.read_csv(disagreement_path)
        mcnemar = bootstrap["mcnemar"]
        st.markdown("#### 두 모델이 서로 보완한 위치")
        disagreement_metrics = st.columns(3)
        disagreement_metrics[0].metric(
            "형태 모델만 정답", f"{mcnemar['shape_only_correct']:,}개"
        )
        disagreement_metrics[1].metric(
            "CNN만 정답", f"{mcnemar['cnn_only_correct']:,}개"
        )
        disagreement_metrics[2].metric(
            "둘 다 오답", f"{mcnemar['both_wrong']:,}개"
        )
        disagreement_display = disagreement.rename(
            columns={
                "class": "클래스",
                "support": "표본 수",
                "shape_only_correct": "형태만 정답",
                "cnn_only_correct": "CNN만 정답",
                "both_wrong": "둘 다 오답",
                "net_shape_minus_cnn": "형태−CNN 순정답",
            }
        )[
            ["클래스", "표본 수", "형태만 정답", "CNN만 정답", "둘 다 오답", "형태−CNN 순정답"]
        ]
        st.dataframe(disagreement_display, hide_index=True, width="stretch")
        st.caption(
            "Edge-Loc·Edge-Ring은 형태 모델이 더 많이 보완했고, none·Loc은 CNN이 더 많이 "
            "보완했습니다. 같은 Test의 사후 분석이므로 앙상블 선택이나 임계값 조정에는 사용하지 않습니다."
        )

        descriptor = pd.read_csv(descriptor_path)
        descriptor.insert(
            1,
            "형상 지표",
            descriptor["서술자"].map(DESCRIPTOR_LABELS).fillna(descriptor["서술자"]),
        )
        descriptor = descriptor.rename(columns={"서술자": "내부 변수"})
        st.markdown("#### 사람이 읽을 수 있는 형상 근거")
        st.dataframe(
            descriptor[
                ["순위", "형상 지표", "묶음", "사람이 읽는 뜻", "macro-F1 하락폭"]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "macro-F1 하락폭": st.column_config.NumberColumn(format="%.4f")
            },
        )
        if standalone:
            with st.expander("내부 변수명 확인"):
                st.dataframe(
                    descriptor[["형상 지표", "내부 변수"]],
                    hide_index=True,
                    width="stretch",
                )
        st.caption(
            "형상 중요도는 예측에 사용된 모양 단서이며 물리적 불량 원인이나 공정 조정 지시가 아닙니다."
        )

        eda = _read_json(eda_path)
        st.markdown("#### 원본 데이터 구성")
        eda_metrics = st.columns(4)
        eda_metrics[0].metric("전체 웨이퍼", f"{eda['n_total']:,}개")
        eda_metrics[1].metric("라벨 보유", f"{eda['n_labeled']:,}개", f"{eda['n_labeled']/eda['n_total']:.1%}")
        eda_metrics[2].metric("라벨 lot", f"{eda['n_lots_labeled']:,}개")
        eda_metrics[3].metric("lot당 중앙값", f"{eda['wafers_per_lot_labeled']['median']:.0f}개")
        class_counts = pd.Series(eda["failure_type_counts"], name="웨이퍼 수").sort_values()
        st.bar_chart(class_counts)
        st.caption(
            "라벨 없는 638,507개는 지도학습 평가에서 제외했고, 9개 클래스를 모두 유지했습니다. "
            "분할은 lot 단위로 고정해 같은 lot가 학습과 Test에 동시에 들어가지 않습니다."
        )


def render_secom_independent_validation(result_dir: Path) -> None:
    """Show the ExtraTrees benchmark as a non-deployed independent check."""
    comparison_path = result_dir / "baseline_comparison.csv"
    candidates_path = result_dir / "candidate_oof_summary.csv"
    selected_path = result_dir / "selected_model.json"
    if not all(path.is_file() for path in (comparison_path, candidates_path, selected_path)):
        return

    with st.expander("독립 검증 · ExtraTrees 대조 모델", expanded=False):
        selected = _read_json(selected_path)
        comparison = pd.read_csv(comparison_path)
        candidates = pd.read_csv(candidates_path)
        alternative = comparison.loc[comparison["출처"] == "독립 검증"].iloc[0]
        baseline = comparison.loc[
            comparison["모델"].astype(str).str.contains("CatBoost 튜닝")
        ].iloc[0]
        metrics = st.columns(4)
        metrics[0].metric("ExtraTrees PR-AUC", f"{alternative['pr_auc']:.4f}")
        metrics[1].metric("CatBoost PR-AUC", f"{baseline['pr_auc']:.4f}")
        metrics[2].metric("ExtraTrees F2", f"{alternative['f2']:.4f}")
        metrics[3].metric("CatBoost F2", f"{baseline['f2']:.4f}")
        st.caption(
            f"선택 규칙: {selected['selection_rule']} · 임계값 {selected['threshold']:.6f} · "
            "후보와 임계값은 반복 OOF에서 고정하고 Test는 마지막 확인에만 사용했습니다."
        )
        st.dataframe(
            comparison[["출처", "모델", "pr_auc", "precision", "recall", "f2", "tp", "fp", "fn"]],
            hide_index=True,
            width="stretch",
        )
        st.warning(
            "ExtraTrees는 불량 1건을 더 찾았지만 오탐이 13건 늘었고, PR-AUC와 F2도 "
            "CatBoost보다 높지 않습니다. 배포 후보가 아니라 독립 재현 참고값입니다."
        )
        st.markdown("#### OOF 후보 비교")
        st.dataframe(
            candidates[
                ["candidate", "oof_pr_auc", "fold_pr_auc_mean", "fold_pr_auc_std", "oof_recall", "oof_f2"]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "모델 바이너리는 저장소에 포함하지 않았습니다. 현재 환경에서 재학습하고 SHA-256을 "
            "다시 검증하기 전에는 추론에 사용하지 않습니다."
        )
