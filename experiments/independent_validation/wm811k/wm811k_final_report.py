"""WM-811K 최종 통합·검증 리포트 — 새 학습 없이 기존 산출물만 읽는다.

이 스크립트는 **모델을 학습하지 않습니다.** 이미 만들어진 세 가지 결과를
각자의 저장 파일에서 다시 읽어, 숫자가 서로 맞는지 독립적으로 검증하고
하나의 비교표·그림으로 묶습니다.

읽는 원본 (모두 읽기 전용):

1. 형태 서술자 모델 — `결과/wm811k/shape_model/metrics.json`
   그리고 같은 폴더의 `test_predictions.csv` / `validation_predictions.csv`,
   `*_confusion_matrix.csv`. 저장된 예측에서 지표를 **다시 계산**해
   metrics.json과 일치하는지 봅니다. 값을 하드코딩하지 않습니다.
2. 메타데이터 대조 모델 — `결과/wm811k/geometry_control/metrics.json`
3. 기준선 CNN — `결과물/wm811k/selected_model_results/test_classification_report.csv`
   (기준선 폴더는 읽기만 하고 절대 쓰지 않습니다.)

세 결과가 **같은 고정 test**를 본 것인지도 검사합니다. 클래스별 support가
9개 클래스 모두에서 일치하고 합이 24,705여야 통과합니다.

실행:
    .venv\\Scripts\\python.exe "experiments\\independent_validation\\wm811k\\wm811k_final_report.py"
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager, rcParams  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_DIR, to_markdown_table, write_json, read_json  # noqa: E402
from wm811k_common import (  # noqa: E402
    BASELINE_WM_DIR,
    CLASS_ORDER,
    WM_RESULT_DIR,
    evaluate,
)

SHAPE_DIR = WM_RESULT_DIR / "shape_model"
CONTROL_DIR = WM_RESULT_DIR / "geometry_control"
FINAL_DIR = WM_RESULT_DIR / "final_comparison"
BASELINE_REPORT = (
    BASELINE_WM_DIR / "selected_model_results" / "test_classification_report.csv"
)

EXPECTED_TEST_ROWS = 24_705
EXPECTED_CLASS_COUNT = 9
# test 표본이 이보다 적은 클래스는 recall 한 건이 크게 흔들린다.
SMALL_SAMPLE_THRESHOLD = 200
# 요구된 중점 비교 대상.
FOCUS_CLASSES = ("Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full")

MODEL_LABELS = {
    "control": "메타데이터 대조 (화소 미사용)",
    "shape": "52개 형태 서술자 + HistGB (CPU)",
    "cnn": "기준선 CNN (GPU)",
}


# --------------------------------------------------------------- 서술자 사전
def descriptor_meaning(name: str) -> str:
    """서술자 이름을 사람이 읽는 설명으로 바꾼다.

    주의: 이것은 **무엇을 측정한 값인지**에 대한 설명이지, 공정 원인에 대한
    설명이 아니다. 중요도가 높다고 해서 그 성질이 결함을 '일으켰다'는 뜻이
    전혀 아니다.
    """
    fixed = {
        "die_count": "웨이퍼 안에 있는 전체 die 개수",
        "fail_count": "불량 die 개수",
        "fail_ratio": "전체 die 중 불량 die 비율",
        "wafer_fill_ratio": "64×64 격자에서 웨이퍼가 차지하는 면적 비율",
        "centroid_offset": "불량 die 중심이 웨이퍼 중심에서 벗어난 정도(편심)",
        "radius_mean": "불량 die가 웨이퍼 중심에서 떨어진 평균 거리",
        "radius_std": "불량 die의 중심 거리 흩어짐(반경 방향 퍼짐)",
        "edge_fail_ratio": "바깥 20% 고리 안에서의 불량 비율",
        "core_fail_ratio": "중심 30% 원 안에서의 불량 비율",
        "edge_to_core_ratio": "가장자리 불량이 중심 불량보다 얼마나 많은지",
        "angular_resultant": "불량이 한 방향으로 쏠린 정도(0=고루, 1=한쪽)",
        "angular_entropy": "불량이 각도 방향으로 고르게 퍼진 정도",
        "angular_max": "12방향 중 불량이 가장 많은 방향의 비율",
        "angular_min": "12방향 중 불량이 가장 적은 방향의 비율",
        "component_count": "서로 떨어진 불량 덩어리의 개수",
        "largest_component_ratio": "가장 큰 연결 불량 덩어리가 전체 불량에서 차지하는 비율",
        "second_component_ratio": "두 번째로 큰 불량 덩어리의 비율",
        "component_size_mean": "불량 덩어리 하나의 평균 크기",
        "largest_elongation": "가장 큰 덩어리가 가늘고 긴 정도(1에 가까우면 선 모양)",
        "largest_extent": "가장 큰 덩어리가 자기 경계상자를 채운 정도",
        "largest_boundary_ratio": "가장 큰 덩어리 중 웨이퍼 가장자리에 닿은 비율",
        "longest_run_horizontal": "가로로 이어진 가장 긴 불량 구간 길이",
        "longest_run_vertical": "세로로 이어진 가장 긴 불량 구간 길이",
        "longest_run_diagonal": "대각으로 이어진 가장 긴 불량 구간 길이",
        "bbox_fill_ratio": "불량 전체를 감싼 사각형을 불량이 채운 정도",
    }
    if name in fixed:
        return fixed[name]
    if name.startswith("radial_bin_"):
        index = int(name.rsplit("_", 1)[1])
        low, high = index * 12.5, (index + 1) * 12.5
        where = "중심" if index <= 1 else ("가장자리" if index >= 6 else "중간 고리")
        return (
            f"중심→가장자리 {index + 1}번째 고리(반경 {low:.0f}~{high:.0f}%)의 불량 비율 — {where} 쪽"
        )
    if name.startswith("angular_sorted_"):
        index = int(name.rsplit("_", 1)[1])
        return f"12방향 불량 비율을 큰 순서로 세웠을 때 {index + 1}번째 값(회전 무관)"
    if name.startswith("hu_"):
        index = int(name.rsplit("_", 1)[1])
        return f"Hu 불변 모멘트 {index + 1}번 — 회전·크기가 변해도 유지되는 모양 지문"
    return "설명 미정의"


def descriptor_group(name: str) -> str:
    if name.startswith("radial_bin_") or name in {
        "radius_mean",
        "radius_std",
        "edge_fail_ratio",
        "core_fail_ratio",
        "edge_to_core_ratio",
        "centroid_offset",
    }:
        return "반경·위치"
    if name.startswith("angular"):
        return "각도 분포"
    if name.startswith("hu_"):
        return "불변 모멘트"
    if "component" in name or "largest" in name or "run" in name or name == "bbox_fill_ratio":
        return "덩어리 모양"
    return "전체 규모"


# ------------------------------------------------------------------ 읽기·검증
def _recompute(predictions: pd.DataFrame, labels: list[str]) -> dict:
    return evaluate(
        predictions["actual"].to_numpy(),
        predictions["predicted"].to_numpy(),
        labels,
    )


def verify_shape_model() -> dict:
    """저장된 예측에서 지표를 다시 계산해 metrics.json과 대조한다."""
    metrics = read_json(SHAPE_DIR / "metrics.json")
    labels = [c for c in CLASS_ORDER]
    checks: list[dict] = []

    for split in ("validation", "test"):
        predictions = pd.read_csv(SHAPE_DIR / f"{split}_predictions.csv")
        recomputed = _recompute(predictions, labels)
        stored = metrics[split]

        for key in ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"):
            checks.append(
                {
                    "항목": f"{split}.{key}",
                    "저장값": stored[key],
                    "재계산값": recomputed[key],
                    "일치": bool(np.isclose(stored[key], recomputed[key], atol=1e-9)),
                }
            )
        for name in CLASS_ORDER:
            checks.append(
                {
                    "항목": f"{split}.recall.{name}",
                    "저장값": stored["per_class_recall"][name],
                    "재계산값": recomputed["per_class_recall"][name],
                    "일치": bool(
                        np.isclose(
                            stored["per_class_recall"][name],
                            recomputed["per_class_recall"][name],
                            atol=1e-9,
                        )
                    ),
                }
            )

        # 혼동행렬 행 합 == support
        confusion = pd.read_csv(
            SHAPE_DIR / f"{split}_confusion_matrix.csv", index_col=0
        )
        for name in CLASS_ORDER:
            checks.append(
                {
                    "항목": f"{split}.confusion_rowsum.{name}",
                    "저장값": float(stored["support"][name]),
                    "재계산값": float(confusion.loc[f"실제_{name}"].sum()),
                    "일치": int(confusion.loc[f"실제_{name}"].sum())
                    == int(stored["support"][name]),
                }
            )

    # fold 요약도 fold_scores.csv에서 다시 평균 내 본다.
    folds = pd.read_csv(SHAPE_DIR / "fold_scores.csv")
    for key in ("macro_f1", "balanced_accuracy"):
        checks.append(
            {
                "항목": f"cv.{key}_mean",
                "저장값": metrics["cv"][f"{key}_mean"],
                "재계산값": float(folds[key].mean()),
                "일치": bool(
                    np.isclose(metrics["cv"][f"{key}_mean"], folds[key].mean(), atol=1e-9)
                ),
            }
        )

    return {
        "metrics": metrics,
        "folds": folds,
        "checks": pd.DataFrame(checks),
        "all_passed": bool(pd.DataFrame(checks)["일치"].all()),
    }


def load_baseline_cnn() -> dict:
    """기준선 CNN 지표를 저장 파일에서 직접 읽는다(하드코딩 없음)."""
    if not BASELINE_REPORT.is_file():
        raise FileNotFoundError(f"기준선 CNN 리포트가 없습니다: {BASELINE_REPORT}")
    report = pd.read_csv(BASELINE_REPORT, index_col=0)
    return {
        "source": BASELINE_REPORT.relative_to(PROJECT_DIR).as_posix(),
        "macro_f1": float(report.loc["macro avg", "f1-score"]),
        # macro avg recall == balanced accuracy (클래스별 recall의 단순 평균)
        "balanced_accuracy": float(report.loc["macro avg", "recall"]),
        "accuracy": float(report.loc["accuracy", "f1-score"]),
        "weighted_f1": float(report.loc["weighted avg", "f1-score"]),
        "per_class_recall": {
            name: float(report.loc[name, "recall"]) for name in CLASS_ORDER
        },
        "per_class_f1": {
            name: float(report.loc[name, "f1-score"]) for name in CLASS_ORDER
        },
        "support": {name: int(report.loc[name, "support"]) for name in CLASS_ORDER},
    }


def check_same_test_split(sources: dict[str, dict]) -> dict:
    """세 결과가 같은 고정 test를 봤는지 클래스별 support로 확인한다."""
    supports = {key: value["support"] for key, value in sources.items()}
    reference = supports["shape"]

    mismatches = []
    for key, value in supports.items():
        if value != reference:
            mismatches.append(key)

    totals = {key: sum(value.values()) for key, value in supports.items()}
    return {
        "reference_support": reference,
        "totals": totals,
        "class_count": {key: len(value) for key, value in supports.items()},
        "mismatched_sources": mismatches,
        "all_same_split": (
            not mismatches
            and all(total == EXPECTED_TEST_ROWS for total in totals.values())
            and all(
                len(value) == EXPECTED_CLASS_COUNT for value in supports.values()
            )
        ),
    }


def wilson_interval(recall: float, support: int, z: float = 1.96) -> tuple[float, float]:
    """recall의 95% Wilson 신뢰구간. 표본이 작을수록 넓어진다."""
    if support <= 0:
        return (float("nan"), float("nan"))
    denominator = 1.0 + z**2 / support
    center = (recall + z**2 / (2 * support)) / denominator
    margin = (
        z
        * np.sqrt(recall * (1 - recall) / support + z**2 / (4 * support**2))
        / denominator
    )
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


# --------------------------------------------------------------------- 표 만들기
def build_model_comparison(sources: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key in ("control", "shape", "cnn"):
        value = sources[key]
        rows.append(
            {
                "모델": MODEL_LABELS[key],
                "macro-F1": round(value["macro_f1"], 4),
                "balanced accuracy": round(value["balanced_accuracy"], 4),
                "accuracy": round(value["accuracy"], 4),
                "weighted F1": round(value.get("weighted_f1", float("nan")), 4),
                "test 건수": sum(value["support"].values()),
            }
        )
    return pd.DataFrame(rows)


def build_class_recall(sources: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name in CLASS_ORDER:
        support = sources["shape"]["support"][name]
        shape_recall = sources["shape"]["per_class_recall"][name]
        cnn_recall = sources["cnn"]["per_class_recall"][name]
        low, high = wilson_interval(shape_recall, support)
        cnn_low, cnn_high = wilson_interval(cnn_recall, support)
        rows.append(
            {
                "클래스": name,
                "test 표본수": support,
                "표본 적음": "⚠" if support < SMALL_SAMPLE_THRESHOLD else "",
                "대조모델 recall": round(sources["control"]["per_class_recall"][name], 4),
                "형태서술자 recall": round(shape_recall, 4),
                "기준선CNN recall": round(cnn_recall, 4),
                "형태-CNN 차이": round(shape_recall - cnn_recall, 4),
                "형태 recall 95%CI": f"{low:.3f}–{high:.3f}",
                "CNN recall 95%CI": f"{cnn_low:.3f}–{cnn_high:.3f}",
                "CI 겹침": "예"
                if not (high < cnn_low or cnn_high < low)
                else "아니오",
                "중점비교": "●" if name in FOCUS_CLASSES else "",
            }
        )
    return pd.DataFrame(rows)


def build_descriptor_table(top: int = 10) -> pd.DataFrame:
    importance = pd.read_csv(SHAPE_DIR / "descriptor_importance.csv")
    head = importance.head(top).copy()
    head.insert(0, "순위", range(1, len(head) + 1))
    head["묶음"] = head["descriptor"].map(descriptor_group)
    head["사람이 읽는 뜻"] = head["descriptor"].map(descriptor_meaning)
    head["macro-F1 하락폭"] = head["importance_mean"].round(5)
    head["표준편차"] = head["importance_std"].round(5)
    return head[
        ["순위", "descriptor", "묶음", "사람이 읽는 뜻", "macro-F1 하락폭", "표준편차"]
    ].rename(columns={"descriptor": "서술자"})


# ----------------------------------------------------------------------- 그림
def _use_korean_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"):
        if candidate in available:
            rcParams["font.family"] = candidate
            rcParams["axes.unicode_minus"] = False
            return candidate
    rcParams["axes.unicode_minus"] = False
    return rcParams["font.family"][0]


def plot_class_recall(table: pd.DataFrame, path: Path) -> None:
    """9개 클래스 recall 3모델 비교 + 중점 6개 클래스 차이."""
    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(13.5, 10.5), gridspec_kw={"height_ratios": [1.35, 1.0]}
    )

    positions = np.arange(len(table))
    width = 0.26
    series = [
        ("대조모델 recall", "메타데이터 대조 (화소 미사용)", "#b0bec5"),
        ("형태서술자 recall", "52개 형태 서술자 + HistGB", "#e67e22"),
        ("기준선CNN recall", "기준선 CNN", "#2e86c1"),
    ]
    for offset, (column, label, color) in zip((-width, 0.0, width), series):
        upper.bar(
            positions + offset,
            table[column],
            width=width,
            label=label,
            color=color,
            edgecolor="white",
        )

    # 주의: Malgun Gothic에 ⚠(U+26A0)와 −(U+2212) 글리프가 없어 그림에서는 쓰지 않는다.
    ticks = [
        f"{row['클래스']}\n(n={row['test 표본수']:,}){' *소표본' if row['표본 적음'] else ''}"
        for _, row in table.iterrows()
    ]
    upper.set_xticks(positions)
    upper.set_xticklabels(ticks, fontsize=9)
    upper.set_ylim(0, 1.24)
    upper.set_ylabel("고정 test recall")
    upper.set_title(
        "WM-811K 고정 test(24,705장) 클래스별 recall — 같은 분할·같은 지표\n"
        f"*소표본 = test 표본 {SMALL_SAMPLE_THRESHOLD}장 미만, recall 한 건의 변동이 큼",
        fontsize=12,
        pad=34,
    )
    # 막대가 1.0까지 차므로 범례를 그림 안에 두면 가린다. 축 위로 뺀다.
    upper.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    upper.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        upper.spines[spine].set_visible(False)

    # --- 아래: 중점 6개 클래스의 형태-CNN 차이 (Wilson CI 포함) ------------
    focus = table[table["클래스"].isin(FOCUS_CLASSES)].reset_index(drop=True)
    focus_positions = np.arange(len(focus))
    colors = ["#e67e22" if value > 0 else "#2e86c1" for value in focus["형태-CNN 차이"]]
    lower.barh(focus_positions, focus["형태-CNN 차이"], color=colors, edgecolor="white")
    lower.axvline(0.0, color="#424949", linewidth=1)

    for position, (_, row) in zip(focus_positions, focus.iterrows()):
        value = row["형태-CNN 차이"]
        note = f"{value:+.3f}  (n={row['test 표본수']:,}, CI 겹침 {row['CI 겹침']})"
        lower.text(
            value + (0.004 if value >= 0 else -0.004),
            position,
            note,
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    lower.set_yticks(focus_positions)
    lower.set_yticklabels(
        [
            f"{row['클래스']} (n={row['test 표본수']:,}){' *소표본' if row['표본 적음'] else ''}"
            for _, row in focus.iterrows()
        ],
        fontsize=10,
    )
    lower.invert_yaxis()
    span = float(np.abs(focus["형태-CNN 차이"]).max())
    lower.set_xlim(-span * 2.1, span * 2.1)
    lower.set_xlabel("형태 서술자 recall - 기준선 CNN recall  (오른쪽=형태가 높음)")
    lower.set_title(
        "중점 비교 6개 클래스 — 우열이 아니라 서로 다른 강약점을 본다\n"
        "95% Wilson 신뢰구간이 겹치면 이 표본에서 차이를 단정할 수 없다",
        fontsize=12,
    )
    lower.grid(axis="x", alpha=0.25)
    for spine in ("top", "right", "left"):
        lower.spines[spine].set_visible(False)

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_descriptor_importance(table: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.5, 6.4))
    ordered = table.iloc[::-1]
    positions = np.arange(len(ordered))
    palette = {
        "반경·위치": "#e67e22",
        "덩어리 모양": "#16a085",
        "각도 분포": "#8e44ad",
        "불변 모멘트": "#7f8c8d",
        "전체 규모": "#c0392b",
    }
    axis.barh(
        positions,
        ordered["macro-F1 하락폭"],
        xerr=ordered["표준편차"],
        color=[palette.get(group, "#95a5a6") for group in ordered["묶음"]],
        edgecolor="white",
        error_kw={"ecolor": "#424949", "capsize": 3, "lw": 1},
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [
            f"{row['서술자']}\n{row['사람이 읽는 뜻']}"
            for _, row in ordered.iterrows()
        ],
        fontsize=8.5,
    )
    axis.set_xlabel("섞었을 때의 macro-F1 하락폭 (순열 중요도, 클수록 중요)")
    axis.set_title(
        "형태 서술자 순열 중요도 TOP 10 — 모델이 무엇을 보고 판단했는가\n"
        "이것은 '측정한 모양 성질'이지 공정 원인이 아니다",
        fontsize=12,
    )
    axis.grid(axis="x", alpha=0.25)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    # TOP 10에 실제로 나온 묶음만 범례에 둔다.
    present = [group for group in palette if group in set(ordered["묶음"])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[group]) for group in present]
    axis.legend(handles, present, fontsize=9, loc="lower right", title="서술자 묶음")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# ------------------------------------------------------------------- 문서 쓰기
def write_summary(
    comparison: pd.DataFrame,
    recall_table: pd.DataFrame,
    descriptors: pd.DataFrame,
    verification: dict,
    split_check: dict,
    sources: dict[str, dict],
) -> None:
    shape, cnn = sources["shape"], sources["cnn"]
    focus = recall_table[recall_table["클래스"].isin(FOCUS_CLASSES)]
    shape_better = focus[focus["형태-CNN 차이"] > 0]["클래스"].tolist()
    cnn_better = focus[focus["형태-CNN 차이"] < 0]["클래스"].tolist()
    small = recall_table[recall_table["test 표본수"] < SMALL_SAMPLE_THRESHOLD]

    lines = [
        "# WM-811K 최종 비교 — 대조 · 형태 서술자 · 메인 CNN",
        "",
        "> **프로젝트의 메인 모델은 CNN + Grad-CAM입니다.** 여기의 메타데이터 대조 모델과",
        "> 형태 서술자 모델은 메인 모델을 독립적으로 재검증하기 위한 **대조 실험**이며,",
        "> 최종 모델이나 대표 모델로 승격되지 않습니다.",
        "",
        "세 결과 모두 **같은 lot 비중복 고정 test 24,705장**에서 나온 값입니다.",
        "이 문서의 숫자는 새로 학습한 것이 아니라 각 모델의 저장 파일에서 다시 읽은 것입니다.",
        "",
        "차이의 **통계적 유의성은 `paired_bootstrap_summary.md`**(같은 웨이퍼를 함께",
        "복원추출한 paired bootstrap)를 근거로 합니다. 이 문서의 Wilson 구간은 기술",
        "통계로만 읽어 주세요.",
        "",
        "## 검증 결과",
        "",
        f"- 형태 모델 저장 지표 ↔ 저장 예측 재계산: **{len(verification['checks'])}개 항목 전부 "
        f"{'일치' if verification['all_passed'] else '불일치 있음'}**",
        f"- 세 결과의 클래스별 test support 일치: **{'예' if split_check['all_same_split'] else '아니오'}**",
        "- 각 출처 test 합계: "
        + ", ".join(f"{key} {value:,}" for key, value in split_check["totals"].items()),
        f"- 클래스 수: {EXPECTED_CLASS_COUNT}개 (모든 출처 동일)",
        f"- 기준선 CNN 수치 출처: `{BASELINE_REPORT.name}` (읽기 전용, 수정하지 않음)",
        "",
        "## 3모델 비교 (고정 test)",
        "",
        to_markdown_table(comparison),
        "",
        f"- 형태 서술자와 CNN의 macro-F1 차이: **{shape['macro_f1'] - cnn['macro_f1']:+.4f}** "
        f"(CNN {cnn['macro_f1']:.4f} vs 형태 {shape['macro_f1']:.4f})",
        f"- balanced accuracy 차이: **{shape['balanced_accuracy'] - cnn['balanced_accuracy']:+.4f}** "
        f"(CNN {cnn['balanced_accuracy']:.4f} vs 형태 {shape['balanced_accuracy']:.4f})",
        f"- accuracy 차이: **{shape['accuracy'] - cnn['accuracy']:+.4f}**",
        "",
        "두 모델은 지표마다 방향이 갈립니다. macro-F1과 accuracy는 CNN이 앞서고,",
        "balanced accuracy는 형태 서술자가 앞섭니다. 어느 한쪽이 전반적으로 우수하다고",
        "말할 근거가 아니라, **다수 클래스 정확도(CNN)와 소수 클래스 균형(형태 서술자)의",
        "trade-off**로 읽는 것이 맞습니다.",
        "",
        "메타데이터 대조 모델은 화소를 하나도 보지 않는 하한선입니다. 두 화소 기반 모델이",
        "이 값보다 크게 높다는 것이 '모양을 실제로 봤다'는 증거이며, 대조 모델 자체를",
        "결함 진단에 쓰면 안 됩니다.",
        "",
        "## 클래스별 recall",
        "",
        to_markdown_table(recall_table),
        "",
        f"- 중점 6개 중 형태 서술자가 높은 클래스: {', '.join(shape_better) if shape_better else '없음'}",
        f"- 중점 6개 중 기준선 CNN이 높은 클래스: {', '.join(cnn_better) if cnn_better else '없음'}",
        "",
        "> ⚠ **위 `CI 겹침` 열만으로 차이를 판단하지 마세요.** Wilson 구간은 모델을 따로",
        "> 놓고 본 **기술 통계**입니다. 두 모델은 같은 웨이퍼를 봤기 때문에 오차가 서로",
        "> 상관되어 있어, 구간이 겹쳐도 차이가 유의할 수 있고 그 반대도 가능합니다.",
        "> **차이의 유의성 판단은 `paired_bootstrap_summary.md`의 paired bootstrap을",
        "> 근거로 삼습니다.** 실제로 `Loc`는 Wilson 구간이 겹치지만 paired bootstrap",
        "> 95% 신뢰구간은 0을 포함하지 않습니다.",
        "",
        "이 열은 비교를 위해 그대로 남겨 둡니다(삭제하지 않음).",
        "",
        "### 표본이 적은 클래스 주의",
        "",
        to_markdown_table(
            small[["클래스", "test 표본수", "형태서술자 recall", "형태 recall 95%CI"]]
        ),
        "",
        f"특히 `Near-full`은 test 표본이 {recall_table.loc[recall_table['클래스'] == 'Near-full', 'test 표본수'].iloc[0]}장뿐입니다. "
        "recall 1.000도 95% 신뢰구간은 "
        f"{recall_table.loc[recall_table['클래스'] == 'Near-full', '형태 recall 95%CI'].iloc[0]}로 넓습니다. "
        "한 장만 틀려도 값이 크게 흔들리므로 '완벽하게 잡는다'로 읽으면 안 됩니다.",
        "",
        "## 형태 서술자 중요도 TOP 10",
        "",
        to_markdown_table(descriptors),
        "",
        "이 표는 **모델이 무엇을 보고 판단했는가**를 이름으로 말해 줍니다.",
        "그러나 중요도가 높다고 해서 그 성질이 결함의 **원인**이라는 뜻은 아닙니다.",
        "순열 중요도는 예측에 대한 기여도이지 인과관계의 증거가 아니며,",
        "공정 조정 지시의 근거로 쓸 수 없습니다.",
        "",
        "## 그림",
        "",
        "- `class_recall_comparison.png` — 9개 클래스 3모델 recall + 중점 6개 클래스 차이",
        "- `descriptor_importance_top10.png` — 순열 중요도 상위 10개와 그 뜻",
        "",
        "## 한계",
        "",
        "- 고정 test는 기준선 실험에서 이미 여러 번 관찰된 세트입니다. 새로운 외부 검증이 아닙니다.",
        "- 64×64 축소 과정에서 작은 결함이 뭉개집니다. 원본 웨이퍼 크기가 제각각이라",
        "  같은 결함도 크기에 따라 다르게 보일 수 있습니다.",
        "- 형태 서술자는 정의하지 않은 패턴을 볼 수 없습니다. CNN이 스스로 찾는 표현과 다릅니다.",
        "- 웨이퍼 맵 분류 결과는 물리적 고장 원인이나 공정 조정 지시를 증명하지 않습니다.",
        "",
    ]
    (FINAL_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    if not (SHAPE_DIR / "metrics.json").is_file():
        raise SystemExit(
            "형태 서술자 모델 결과가 없습니다. 먼저 -Mode wmshape 로 학습하세요."
        )
    if not (CONTROL_DIR / "metrics.json").is_file():
        raise SystemExit("메타데이터 대조 모델 결과가 없습니다. 먼저 -Mode wm 을 실행하세요.")

    print("1) 형태 서술자 모델 저장 결과 독립 검증", flush=True)
    verification = verify_shape_model()
    failed = verification["checks"][~verification["checks"]["일치"]]
    if len(failed):
        print(failed.to_string(index=False), flush=True)
        raise SystemExit("저장된 지표와 재계산 결과가 다릅니다.")
    print(f"   {len(verification['checks'])}개 항목 전부 일치", flush=True)

    print("2) 기준선 CNN · 메타데이터 대조 결과 읽기 (하드코딩 없음)", flush=True)
    control_metrics = read_json(CONTROL_DIR / "metrics.json")
    sources = {
        "control": control_metrics["test"],
        "shape": verification["metrics"]["test"],
        "cnn": load_baseline_cnn(),
    }

    print("3) 같은 고정 test 인지 검사", flush=True)
    split_check = check_same_test_split(sources)
    if not split_check["all_same_split"]:
        raise SystemExit(f"세 결과의 test 구성이 다릅니다: {split_check}")
    print(
        f"   9개 클래스 support 일치, 합계 {split_check['totals']['shape']:,}장",
        flush=True,
    )

    print("4) 비교표·그림 생성", flush=True)
    comparison = build_model_comparison(sources)
    recall_table = build_class_recall(sources)
    descriptors = build_descriptor_table()

    comparison.to_csv(
        FINAL_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )
    recall_table.to_csv(
        FINAL_DIR / "class_recall_comparison.csv", index=False, encoding="utf-8-sig"
    )
    descriptors.to_csv(
        FINAL_DIR / "descriptor_glossary_top10.csv", index=False, encoding="utf-8-sig"
    )
    verification["checks"].to_csv(
        FINAL_DIR / "verification_checks.csv", index=False, encoding="utf-8-sig"
    )

    font = _use_korean_font()
    plot_class_recall(recall_table, FINAL_DIR / "class_recall_comparison.png")
    plot_descriptor_importance(
        descriptors, FINAL_DIR / "descriptor_importance_top10.png"
    )

    write_json(
        FINAL_DIR / "final_comparison.json",
        {
            "purpose": "새 학습 없이 기존 산출물만 읽어 만든 WM-811K 3모델 비교",
            "test_rows": split_check["totals"]["shape"],
            "class_count": EXPECTED_CLASS_COUNT,
            "same_test_split": split_check["all_same_split"],
            "verification": {
                "checked_items": int(len(verification["checks"])),
                "all_passed": verification["all_passed"],
                "shape_metrics_source": (SHAPE_DIR / "metrics.json").relative_to(PROJECT_DIR).as_posix(),
                "shape_predictions_source": "로컬 재현용 비추적 파일: "
                + (SHAPE_DIR / "test_predictions.csv").relative_to(PROJECT_DIR).as_posix(),
                "baseline_cnn_source": BASELINE_REPORT.relative_to(PROJECT_DIR).as_posix(),
                "control_source": (CONTROL_DIR / "metrics.json").relative_to(PROJECT_DIR).as_posix(),
            },
            "models": {
                key: {
                    "label": MODEL_LABELS[key],
                    "macro_f1": sources[key]["macro_f1"],
                    "balanced_accuracy": sources[key]["balanced_accuracy"],
                    "accuracy": sources[key]["accuracy"],
                    "per_class_recall": sources[key]["per_class_recall"],
                    "support": sources[key]["support"],
                }
                for key in ("control", "shape", "cnn")
            },
            "shape_cv": verification["metrics"]["cv"],
            "focus_classes": list(FOCUS_CLASSES),
            "small_sample_threshold": SMALL_SAMPLE_THRESHOLD,
            "small_sample_classes": {
                row["클래스"]: int(row["test 표본수"])
                for _, row in recall_table.iterrows()
                if row["test 표본수"] < SMALL_SAMPLE_THRESHOLD
            },
            "descriptor_top10": [
                {
                    "rank": int(row["순위"]),
                    "descriptor": row["서술자"],
                    "group": row["묶음"],
                    "meaning": row["사람이 읽는 뜻"],
                    "importance_mean": float(row["macro-F1 하락폭"]),
                    "importance_std": float(row["표준편차"]),
                }
                for _, row in descriptors.iterrows()
            ],
            "font": font,
            "python": platform.python_version(),
        },
    )
    write_summary(
        comparison, recall_table, descriptors, verification, split_check, sources
    )

    print(f"\n결과 저장: {FINAL_DIR}", flush=True)
    print(to_markdown_table(comparison), flush=True)


if __name__ == "__main__":
    main()


