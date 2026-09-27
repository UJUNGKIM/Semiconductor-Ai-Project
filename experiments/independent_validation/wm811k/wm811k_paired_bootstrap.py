"""WM-811K paired bootstrap — 메인 CNN vs 형태 서술자 대조 모델.

**모델을 새로 학습하거나 추론하지 않습니다.** 두 모델이 같은 고정 test 24,705장에
남긴 예측 파일만 읽습니다.

## 왜 paired인가

클래스별 recall의 신뢰구간을 모델마다 따로 구해 "겹치는가"로 판단하면 틀리기 쉽습니다.
두 모델은 **같은 웨이퍼**를 봤기 때문에 오차가 서로 상관되어 있습니다. 두 구간이
겹쳐도 차이가 유의할 수 있고, 반대도 가능합니다. 그래서 차이 자체의 분포를
직접 만들어야 합니다.

여기서는 고정된 test 24,705장을 **웨이퍼 단위로 복원추출**하고, 뽑힌 같은 웨이퍼
집합에서 두 모델의 지표를 동시에 계산해 그 차이를 기록합니다. 같은 인덱스를 두
모델에 동시에 적용하는 것이 pairing입니다.

## 예측 파일 출처 (둘 다 읽기 전용)

- 메인 CNN: `결과물/wm811k/ood_results/ood_score_records.csv`의 `split == "test"`
  행 24,705개. 이 파일이 선택된 메인 모델의 것인지 확인하기 위해, 여기서 만든
  혼동행렬이 `결과물/wm811k/selected_model_results/test_confusion_matrix.csv`와
  **한 칸도 빠짐없이 같은지** 검사합니다. 다르면 즉시 중단합니다.
  (`colab_가져오기/wm811k_cnn_results/test_predictions.csv`에도 웨이퍼별 예측이
  있지만 그것은 focal loss로 돌린 **다른 실험**이라 쓰지 않습니다.)
- 형태 서술자 대조 모델: `결과물/wm811k/independent_validation/shape_model/test_predictions.csv`

## 재현

난수 시드와 반복 횟수를 결과 JSON에 기록합니다. 같은 시드·같은 입력이면 항상
같은 구간이 나옵니다.

실행:
    .venv\\Scripts\\python.exe "experiments\\independent_validation\\wm811k\\wm811k_paired_bootstrap.py"
    .venv\\Scripts\\python.exe "experiments\\independent_validation\\wm811k\\wm811k_paired_bootstrap.py" --resamples 2000
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_DIR, to_markdown_table, write_json  # noqa: E402
from wm811k_common import (  # noqa: E402
    BASELINE_WM_DIR,
    CLASS_ORDER,
    WM_RESULT_DIR,
    load_split_metadata,
)
from wm811k_final_report import (  # noqa: E402
    FINAL_DIR,
    FOCUS_CLASSES,
    SMALL_SAMPLE_THRESHOLD,
    _use_korean_font,
)

SHAPE_PREDICTIONS = WM_RESULT_DIR / "shape_model" / "test_predictions.csv"
CNN_PREDICTIONS = BASELINE_WM_DIR / "ood_results" / "ood_score_records.csv"
CNN_CONFUSION = BASELINE_WM_DIR / "selected_model_results" / "test_confusion_matrix.csv"
CNN_RUN_SUMMARY = BASELINE_WM_DIR / "selected_model_results" / "run_summary.json"

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 42
EXPECTED_TEST_ROWS = 24_705
# CNN 쪽 label_id 순서. run_summary.json의 class_names와 같아야 한다.
CNN_LABEL_ORDER = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
)


# ------------------------------------------------------------------ 입력 정렬
def load_paired_predictions() -> pd.DataFrame:
    """두 모델의 웨이퍼별 예측을 array_index로 맞춰 한 표로 만든다."""
    from common import read_json

    shape = pd.read_csv(SHAPE_PREDICTIONS)
    if len(shape) != EXPECTED_TEST_ROWS:
        raise SystemExit(f"형태 모델 test 행이 {len(shape)}개입니다(기대 {EXPECTED_TEST_ROWS}).")

    records = pd.read_csv(CNN_PREDICTIONS)
    cnn = records[records["split"] == "test"].copy()
    if len(cnn) != EXPECTED_TEST_ROWS:
        raise SystemExit(f"CNN test 행이 {len(cnn)}개입니다(기대 {EXPECTED_TEST_ROWS}).")

    class_names = read_json(CNN_RUN_SUMMARY).get("class_names")
    if class_names is not None and tuple(class_names) != CNN_LABEL_ORDER:
        raise SystemExit(f"CNN 클래스 순서가 다릅니다: {class_names}")
    names = np.array(CNN_LABEL_ORDER)
    cnn["cnn_actual"] = names[cnn["true_label_id"].to_numpy()]
    cnn["cnn_predicted"] = names[cnn["predicted_label_id"].to_numpy()]

    merged = shape.merge(
        cnn[["array_index", "cnn_actual", "cnn_predicted"]], on="array_index", how="inner"
    )
    if len(merged) != EXPECTED_TEST_ROWS:
        raise SystemExit(
            f"두 예측을 array_index로 맞추니 {len(merged)}행만 남았습니다. 같은 test가 아닙니다."
        )
    if merged["array_index"].duplicated().any():
        raise SystemExit("array_index가 중복됩니다.")
    if not (merged["actual"] == merged["cnn_actual"]).all():
        raise SystemExit("두 파일의 정답 라벨이 서로 다릅니다.")

    # 분할 메타데이터와도 한 번 더 맞춰 본다(정답의 제3의 출처).
    frame = load_split_metadata()
    expected = frame[frame["split"] == "test"][["array_index", "failure_type"]]
    checked = merged.merge(expected, on="array_index", how="inner")
    if len(checked) != EXPECTED_TEST_ROWS or not (
        checked["actual"] == checked["failure_type"]
    ).all():
        raise SystemExit("정답 라벨이 기준선 분할 메타데이터와 다릅니다.")

    return merged.sort_values("array_index").reset_index(drop=True)


def verify_cnn_is_selected_model(paired: pd.DataFrame) -> dict:
    """이 예측이 정말 '선택된 메인 CNN'의 것인지 저장 혼동행렬로 확인한다."""
    stored = pd.read_csv(CNN_CONFUSION, index_col=0)
    labels = list(stored.columns)
    rebuilt = pd.crosstab(
        paired["cnn_actual"], paired["cnn_predicted"]
    ).reindex(index=labels, columns=labels, fill_value=0)
    stored_matrix = stored.reindex(index=labels, columns=labels).to_numpy()
    identical = bool(np.array_equal(rebuilt.to_numpy(), stored_matrix))
    if not identical:
        raise SystemExit(
            "CNN 예측이 selected_model_results의 혼동행렬과 다릅니다. "
            "다른 실험의 예측일 수 있어 중단합니다."
        )
    return {
        "source": CNN_PREDICTIONS.relative_to(PROJECT_DIR).as_posix(),
        "checked_against": CNN_CONFUSION.relative_to(PROJECT_DIR).as_posix(),
        "confusion_matrix_identical": identical,
        "cells_compared": int(stored_matrix.size),
        "total_test_rows": int(stored_matrix.sum()),
    }


def build_disagreement_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate where only one model is correct without storing row predictions."""
    frame = paired.copy()
    frame["shape_correct"] = frame["predicted"] == frame["actual"]
    frame["cnn_correct"] = frame["cnn_predicted"] == frame["actual"]
    rows = []
    for class_name in CLASS_ORDER:
        subset = frame[frame["actual"] == class_name]
        if subset.empty:
            continue
        both_correct = int((subset["shape_correct"] & subset["cnn_correct"]).sum())
        shape_only = int((subset["shape_correct"] & ~subset["cnn_correct"]).sum())
        cnn_only = int((~subset["shape_correct"] & subset["cnn_correct"]).sum())
        both_wrong = int((~subset["shape_correct"] & ~subset["cnn_correct"]).sum())
        support = int(len(subset))
        rows.append(
            {
                "class": class_name,
                "support": support,
                "both_correct": both_correct,
                "shape_only_correct": shape_only,
                "cnn_only_correct": cnn_only,
                "both_wrong": both_wrong,
                "shape_accuracy": (both_correct + shape_only) / support,
                "cnn_accuracy": (both_correct + cnn_only) / support,
                "net_shape_minus_cnn": shape_only - cnn_only,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- 지표 (벡터 연산)
def _metrics_from_counts(counts: np.ndarray) -> tuple[float, float, np.ndarray]:
    """9x9 혼동행렬 카운트에서 macro-F1 · balanced accuracy · 클래스별 recall.

    부트스트랩을 수만 번 돌리므로 sklearn 대신 카운트로 직접 계산한다.
    zero_division=0 규칙은 sklearn과 같게 맞췄다.
    """
    diagonal = np.diag(counts).astype(float)
    actual = counts.sum(axis=1).astype(float)
    predicted = counts.sum(axis=0).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(actual > 0, diagonal / actual, 0.0)
        precision = np.where(predicted > 0, diagonal / predicted, 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)

    present = actual > 0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    balanced_accuracy = float(recall[present].mean()) if present.any() else 0.0
    return macro_f1, balanced_accuracy, recall


def _counts(pair_codes: np.ndarray, sample: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(
        pair_codes[sample], minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes)


def paired_bootstrap(
    paired: pd.DataFrame, resamples: int, seed: int
) -> dict:
    """같은 웨이퍼 인덱스를 두 모델에 동시에 적용하는 복원추출."""
    labels = list(CLASS_ORDER)
    n_classes = len(labels)
    code = {name: index for index, name in enumerate(labels)}

    actual = paired["actual"].map(code).to_numpy()
    shape_pred = paired["predicted"].map(code).to_numpy()
    cnn_pred = paired["cnn_predicted"].map(code).to_numpy()
    if np.isnan(actual).any() or np.isnan(shape_pred).any() or np.isnan(cnn_pred).any():
        raise SystemExit("CLASS_ORDER에 없는 라벨이 있습니다.")

    shape_codes = (actual * n_classes + shape_pred).astype(np.int64)
    cnn_codes = (actual * n_classes + cnn_pred).astype(np.int64)
    n = len(paired)

    # 관측값(부트스트랩 전 원본)
    observed_shape = _metrics_from_counts(
        np.bincount(shape_codes, minlength=n_classes**2).reshape(n_classes, n_classes)
    )
    observed_cnn = _metrics_from_counts(
        np.bincount(cnn_codes, minlength=n_classes**2).reshape(n_classes, n_classes)
    )

    rng = np.random.default_rng(seed)
    macro_difference = np.empty(resamples)
    balanced_difference = np.empty(resamples)
    recall_difference = np.empty((resamples, n_classes))
    class_absent = np.zeros(n_classes, dtype=int)

    for step in range(resamples):
        # 같은 sample을 두 모델에 함께 쓴다 — 이것이 pairing이다.
        sample = rng.integers(0, n, n)
        shape_counts = _counts(shape_codes, sample, n_classes)
        cnn_counts = _counts(cnn_codes, sample, n_classes)

        shape_macro, shape_balanced, shape_recall = _metrics_from_counts(shape_counts)
        cnn_macro, cnn_balanced, cnn_recall = _metrics_from_counts(cnn_counts)

        macro_difference[step] = shape_macro - cnn_macro
        balanced_difference[step] = shape_balanced - cnn_balanced
        recall_difference[step] = shape_recall - cnn_recall
        class_absent += (shape_counts.sum(axis=1) == 0).astype(int)

    return {
        "labels": labels,
        "observed": {
            "shape": {
                "macro_f1": observed_shape[0],
                "balanced_accuracy": observed_shape[1],
                "per_class_recall": dict(zip(labels, observed_shape[2])),
            },
            "cnn": {
                "macro_f1": observed_cnn[0],
                "balanced_accuracy": observed_cnn[1],
                "per_class_recall": dict(zip(labels, observed_cnn[2])),
            },
        },
        "macro_difference": macro_difference,
        "balanced_difference": balanced_difference,
        "recall_difference": recall_difference,
        "class_absent": class_absent,
    }


def summarise(values: np.ndarray, observed: float) -> dict:
    """백분위 95% 구간과 0을 기준으로 한 양측 부트스트랩 p값."""
    low, high = np.percentile(values, [2.5, 97.5])
    below = float((values <= 0).mean())
    above = float((values >= 0).mean())
    p_value = float(min(1.0, 2 * min(below, above)))
    return {
        "observed_difference": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
        "bootstrap_p_two_sided": p_value,
        "bootstrap_mean": float(values.mean()),
        "bootstrap_std": float(values.std(ddof=1)),
    }


def mcnemar(paired: pd.DataFrame) -> dict:
    """전체 정오 일치/불일치에 대한 정확 McNemar 검정.

    부트스트랩과 다른 각도의 paired 검정이다. 두 모델이 **어느 웨이퍼에서**
    갈렸는지를 직접 세므로, 지표 요약이 가리는 부분을 보완한다.
    """
    shape_correct = (paired["actual"] == paired["predicted"]).to_numpy()
    cnn_correct = (paired["actual"] == paired["cnn_predicted"]).to_numpy()

    both = int((shape_correct & cnn_correct).sum())
    shape_only = int((shape_correct & ~cnn_correct).sum())
    cnn_only = int((~shape_correct & cnn_correct).sum())
    neither = int((~shape_correct & ~cnn_correct).sum())

    discordant = shape_only + cnn_only
    p_value = (
        float(stats.binomtest(shape_only, discordant, 0.5).pvalue)
        if discordant > 0
        else 1.0
    )
    return {
        "both_correct": both,
        "shape_only_correct": shape_only,
        "cnn_only_correct": cnn_only,
        "both_wrong": neither,
        "discordant": discordant,
        "exact_p_two_sided": p_value,
        "shape_accuracy": float(shape_correct.mean()),
        "cnn_accuracy": float(cnn_correct.mean()),
    }


# ----------------------------------------------------------------------- 표·그림
def build_class_table(
    result: dict, supports: dict[str, int]
) -> tuple[pd.DataFrame, dict]:
    """표시용 표(4자리 반올림)와 기계 판독용 원본 값을 함께 돌려준다.

    CSV는 사람이 보기 좋게 반올림하고, JSON에는 반올림 전 값을 담는다.
    두 곳에서 따로 반올림하면 `상한 - 하한`과 `폭`이 마지막 자리에서 어긋난다.
    """
    labels = result["labels"]
    rows = []
    raw: dict[str, dict] = {}
    for index, name in enumerate(labels):
        summary = summarise(
            result["recall_difference"][:, index],
            result["observed"]["shape"]["per_class_recall"][name]
            - result["observed"]["cnn"]["per_class_recall"][name],
        )
        support = supports[name]
        raw[name] = {
            "support": int(support),
            "small_sample": bool(support < SMALL_SAMPLE_THRESHOLD),
            "difference": summary["observed_difference"],
            "ci95_low": summary["ci95_low"],
            "ci95_high": summary["ci95_high"],
            "ci_width": summary["ci95_high"] - summary["ci95_low"],
            "excludes_zero": summary["excludes_zero"],
            "bootstrap_p_two_sided": summary["bootstrap_p_two_sided"],
        }
        rows.append(
            {
                "클래스": name,
                "test 표본수": support,
                "표본 적음": "주의" if support < SMALL_SAMPLE_THRESHOLD else "",
                "형태 recall": round(
                    result["observed"]["shape"]["per_class_recall"][name], 4
                ),
                "CNN recall": round(
                    result["observed"]["cnn"]["per_class_recall"][name], 4
                ),
                "차이(형태-CNN)": round(summary["observed_difference"], 4),
                "95% CI 하한": round(summary["ci95_low"], 4),
                "95% CI 상한": round(summary["ci95_high"], 4),
                "CI 폭": round(summary["ci95_high"] - summary["ci95_low"], 4),
                "0 제외": "예" if summary["excludes_zero"] else "아니오",
                "부트스트랩 p": round(summary["bootstrap_p_two_sided"], 4),
                "중점비교": "●" if name in FOCUS_CLASSES else "",
            }
        )
    return pd.DataFrame(rows), raw


def build_overall_table(result: dict) -> tuple[pd.DataFrame, dict]:
    overall = {
        "macro_f1": summarise(
            result["macro_difference"],
            result["observed"]["shape"]["macro_f1"]
            - result["observed"]["cnn"]["macro_f1"],
        ),
        "balanced_accuracy": summarise(
            result["balanced_difference"],
            result["observed"]["shape"]["balanced_accuracy"]
            - result["observed"]["cnn"]["balanced_accuracy"],
        ),
    }
    rows = []
    for label, key in (("macro-F1", "macro_f1"), ("balanced accuracy", "balanced_accuracy")):
        summary = overall[key]
        rows.append(
            {
                "지표": label,
                "형태 서술자": round(result["observed"]["shape"][key], 4),
                "메인 CNN": round(result["observed"]["cnn"][key], 4),
                "차이(형태-CNN)": round(summary["observed_difference"], 4),
                "95% CI 하한": round(summary["ci95_low"], 4),
                "95% CI 상한": round(summary["ci95_high"], 4),
                "0 제외": "예" if summary["excludes_zero"] else "아니오",
                "부트스트랩 p": round(summary["bootstrap_p_two_sided"], 4),
            }
        )
    return pd.DataFrame(rows), overall


def plot_forest(
    class_table: pd.DataFrame, overall_table: pd.DataFrame, path: Path
) -> None:
    """차이와 95% 신뢰구간을 한 장에 세운다(forest plot)."""
    figure, (upper, lower) = plt.subplots(
        2,
        1,
        figsize=(13.0, 10.0),
        gridspec_kw={"height_ratios": [3.1, 1.0]},
    )

    table = class_table.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(table))
    for position, (_, row) in zip(positions, table.iterrows()):
        excludes = row["0 제외"] == "예"
        color = "#e67e22" if row["차이(형태-CNN)"] > 0 else "#2e86c1"
        alpha = 1.0 if excludes else 0.45
        upper.plot(
            [row["95% CI 하한"], row["95% CI 상한"]],
            [position, position],
            color=color,
            linewidth=3.0,
            alpha=alpha,
            solid_capstyle="round",
        )
        upper.plot(
            row["차이(형태-CNN)"],
            position,
            marker="D" if excludes else "o",
            markersize=9 if excludes else 7,
            color=color,
            alpha=alpha,
        )
        note = (
            f"{row['차이(형태-CNN)']:+.3f}  "
            f"[{row['95% CI 하한']:+.3f}, {row['95% CI 상한']:+.3f}]"
        )
        upper.text(row["95% CI 상한"] + 0.012, position, note, va="center", fontsize=9)

    upper.axvline(0.0, color="#424949", linewidth=1.2)
    upper.set_yticks(positions)
    upper.set_yticklabels(
        [
            f"{row['클래스']} (n={row['test 표본수']:,})"
            + (" *소표본" if row["표본 적음"] else "")
            + (" ●" if row["중점비교"] else "")
            for _, row in table.iterrows()
        ],
        fontsize=10,
    )
    upper.set_xlim(-0.42, 0.62)
    upper.set_xlabel("클래스별 recall 차이  (형태 서술자 - 메인 CNN),  오른쪽=형태가 높음")
    upper.set_title(
        "Paired bootstrap — 같은 test 웨이퍼를 함께 복원추출한 recall 차이의 95% 신뢰구간\n"
        "속이 찬 마름모 = 구간이 0을 포함하지 않음,  옅은 원 = 0을 포함(차이 단정 불가)",
        fontsize=12,
    )
    upper.grid(axis="x", alpha=0.25)
    for spine in ("top", "right", "left"):
        upper.spines[spine].set_visible(False)

    # --- 아래: 전체 지표 -------------------------------------------------
    overall = overall_table.iloc[::-1].reset_index(drop=True)
    overall_positions = np.arange(len(overall))
    for position, (_, row) in zip(overall_positions, overall.iterrows()):
        excludes = row["0 제외"] == "예"
        color = "#e67e22" if row["차이(형태-CNN)"] > 0 else "#2e86c1"
        lower.plot(
            [row["95% CI 하한"], row["95% CI 상한"]],
            [position, position],
            color=color,
            linewidth=3.0,
            solid_capstyle="round",
        )
        lower.plot(
            row["차이(형태-CNN)"],
            position,
            marker="D" if excludes else "o",
            markersize=9,
            color=color,
        )
        lower.text(
            row["95% CI 상한"] + 0.0012,
            position,
            f"{row['차이(형태-CNN)']:+.4f}  "
            f"[{row['95% CI 하한']:+.4f}, {row['95% CI 상한']:+.4f}]  p={row['부트스트랩 p']:.3f}",
            va="center",
            fontsize=9,
        )
    lower.axvline(0.0, color="#424949", linewidth=1.2)
    lower.set_yticks(overall_positions)
    lower.set_yticklabels(list(overall["지표"]), fontsize=10)
    lower.set_xlim(-0.030, 0.045)
    lower.set_xlabel("전체 지표 차이 (형태 서술자 - 메인 CNN)")
    lower.set_title("전체 지표 — 방향이 갈린다", fontsize=12)
    lower.grid(axis="x", alpha=0.25)
    for spine in ("top", "right", "left"):
        lower.spines[spine].set_visible(False)

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# ------------------------------------------------------------------- 문서 쓰기
def write_summary(
    class_table: pd.DataFrame,
    overall_table: pd.DataFrame,
    mcnemar_result: dict,
    payload: dict,
) -> None:
    significant = class_table[class_table["0 제외"] == "예"]
    inconclusive = class_table[class_table["0 제외"] == "아니오"]
    small = class_table[class_table["표본 적음"] == "주의"]

    lines = [
        "# WM-811K paired bootstrap — 메인 CNN vs 형태 서술자 대조 모델",
        "",
        "> **메인 모델은 CNN입니다.** 형태 서술자 모델은 사람이 정의한 형상 특징만으로",
        "> 비슷한 수준이 재현되는지 확인하기 위한 대조 실험이며, 메인 모델을 대체하지",
        "> 않습니다.",
        "",
        "## 방법",
        "",
        f"- 고정 test {payload['n_test']:,}장을 **웨이퍼 단위로 복원추출**하고, 같은 인덱스를",
        "  두 모델에 동시에 적용해 지표 차이를 계산했습니다(pairing).",
        f"- 반복 횟수 `resamples` = **{payload['resamples']:,}**, 난수 시드 `seed` = **{payload['seed']}**",
        "  (`numpy.random.default_rng`). 같은 입력·같은 시드면 항상 같은 값이 나옵니다.",
        "- 신뢰구간은 백분위법(2.5% ~ 97.5%)입니다.",
        "- 두 모델이 같은 웨이퍼를 봤으므로 오차가 서로 상관됩니다. **모델별 신뢰구간이",
        "  겹치는지로 판단하지 않고, 차이 자체의 분포를 만들어 0을 포함하는지 봅니다.**",
        "",
        "### 예측 파일 출처 검증",
        "",
        f"- 메인 CNN 예측: `{Path(payload['cnn_source']['source']).name}`의 `split == \"test\"` "
        f"{payload['n_test']:,}행",
        f"- 이 예측으로 다시 만든 혼동행렬이 `test_confusion_matrix.csv`와 "
        f"{payload['cnn_source']['cells_compared']}칸 전부 일치 → 선택된 메인 모델의 예측이 맞습니다.",
        "- 정답 라벨은 형태 모델 예측 파일·CNN 예측 파일·분할 메타데이터 세 곳이 모두 같습니다.",
        "",
        "## 전체 지표 차이",
        "",
        to_markdown_table(overall_table),
        "",
        "## 클래스별 recall 차이",
        "",
        to_markdown_table(class_table),
        "",
        f"- **95% 신뢰구간이 0을 포함하지 않는 클래스: "
        f"{', '.join(significant['클래스']) if len(significant) else '없음'}**",
        f"- 0을 포함해 차이를 단정할 수 없는 클래스: "
        f"{', '.join(inconclusive['클래스']) if len(inconclusive) else '없음'}",
        "",
        "### 표본이 적은 클래스",
        "",
        to_markdown_table(
            small[["클래스", "test 표본수", "차이(형태-CNN)", "95% CI 하한", "95% CI 상한", "CI 폭"]]
        ),
        "",
        "표본이 적을수록 구간이 넓습니다. 표에서 `CI 폭`을 보면 바로 드러납니다.",
        "이 클래스들의 차이는 **방향조차 확정할 수 없습니다.** 순위나 우열의 근거로",
        "쓰면 안 됩니다.",
        "",
        "## McNemar 정확검정 (전체 정오 기준)",
        "",
        "부트스트랩과 다른 각도의 paired 검정입니다. 두 모델이 어느 웨이퍼에서 갈렸는지 직접 셉니다.",
        "",
        f"- 둘 다 맞음 {mcnemar_result['both_correct']:,} · "
        f"둘 다 틀림 {mcnemar_result['both_wrong']:,}",
        f"- 형태만 맞음 {mcnemar_result['shape_only_correct']:,} · "
        f"CNN만 맞음 {mcnemar_result['cnn_only_correct']:,} "
        f"(불일치 {mcnemar_result['discordant']:,}건)",
        f"- 양측 정확검정 p = **{mcnemar_result['exact_p_two_sided']:.3g}**",
        "",
        "## 읽는 법",
        "",
        "- 전체 지표와 클래스별 recall의 방향이 갈립니다. 한쪽이 전반적으로 우수하다는",
        "  결론은 이 데이터에서 나오지 않습니다.",
        "- 차이가 통계적으로 확인되는 항목은 소수이고, 그 크기도 작습니다.",
        "- **메인 모델은 CNN으로 유지합니다.** 이 분석의 목적은 순위 매기기가 아니라,",
        "  사람이 정의한 형상 특징만으로도 비슷한 수준이 나오는지 확인해 메인 모델의",
        "  성능이 특정 구현에만 의존한 결과가 아님을 보이는 것입니다.",
        "",
    ]
    (FINAL_DIR / "paired_bootstrap_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="WM-811K paired bootstrap")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    print("1) 두 모델의 웨이퍼별 test 예측 정렬", flush=True)
    paired = load_paired_predictions()
    print(f"   {len(paired):,}장 정렬 완료", flush=True)

    print("2) CNN 예측이 선택된 메인 모델의 것인지 검증", flush=True)
    cnn_source = verify_cnn_is_selected_model(paired)
    print(
        f"   혼동행렬 {cnn_source['cells_compared']}칸 전부 일치", flush=True
    )

    print(
        f"3) paired bootstrap (resamples={arguments.resamples:,}, seed={arguments.seed})",
        flush=True,
    )
    result = paired_bootstrap(paired, arguments.resamples, arguments.seed)

    supports = paired["actual"].value_counts().to_dict()
    class_table, class_difference = build_class_table(result, supports)
    overall_table, overall = build_overall_table(result)
    mcnemar_result = mcnemar(paired)
    disagreement = build_disagreement_summary(paired)

    class_table.to_csv(
        FINAL_DIR / "paired_bootstrap_class_recall.csv", index=False, encoding="utf-8-sig"
    )
    overall_table.to_csv(
        FINAL_DIR / "paired_bootstrap_overall.csv", index=False, encoding="utf-8-sig"
    )
    disagreement.to_csv(
        FINAL_DIR / "class_disagreement_summary.csv", index=False, encoding="utf-8-sig"
    )

    font = _use_korean_font()
    plot_forest(class_table, overall_table, FINAL_DIR / "paired_bootstrap.png")

    payload = {
        "purpose": "메인 CNN과 형태 서술자 대조 모델의 차이에 대한 paired bootstrap 불확실성",
        "main_model": "기준선 CNN (프로젝트 메인 모델)",
        "control_model": "52개 형태 서술자 + HistGradientBoosting (독립 검증용 대조 모델)",
        "method": "고정 test 웨이퍼 단위 복원추출, 같은 인덱스를 두 모델에 동시 적용",
        "ci_method": "percentile (2.5, 97.5)",
        "resamples": arguments.resamples,
        "seed": arguments.seed,
        "rng": "numpy.random.default_rng",
        "n_test": int(len(paired)),
        "class_count": len(result["labels"]),
        "small_sample_threshold": SMALL_SAMPLE_THRESHOLD,
        "cnn_source": cnn_source,
        "shape_source": "로컬 재현용 비추적 파일: "
        + SHAPE_PREDICTIONS.relative_to(PROJECT_DIR).as_posix(),
        "observed": {
            "shape": {
                "macro_f1": result["observed"]["shape"]["macro_f1"],
                "balanced_accuracy": result["observed"]["shape"]["balanced_accuracy"],
                "per_class_recall": {
                    name: float(value)
                    for name, value in result["observed"]["shape"]["per_class_recall"].items()
                },
            },
            "cnn": {
                "macro_f1": result["observed"]["cnn"]["macro_f1"],
                "balanced_accuracy": result["observed"]["cnn"]["balanced_accuracy"],
                "per_class_recall": {
                    name: float(value)
                    for name, value in result["observed"]["cnn"]["per_class_recall"].items()
                },
            },
        },
        "overall_difference": overall,
        # 반올림 전 원본. CSV는 표시용으로 4자리에서 반올림한다.
        "class_difference": class_difference,
        "mcnemar": mcnemar_result,
        "font": font,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    write_json(FINAL_DIR / "paired_bootstrap.json", payload)
    write_summary(class_table, overall_table, mcnemar_result, payload)

    print(f"\n결과 저장: {FINAL_DIR}", flush=True)
    print(to_markdown_table(overall_table), flush=True)
    print("", flush=True)
    print(
        to_markdown_table(
            class_table[
                ["클래스", "test 표본수", "차이(형태-CNN)", "95% CI 하한", "95% CI 상한", "0 제외"]
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


