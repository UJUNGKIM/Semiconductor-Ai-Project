"""독립 검증 후보 비교 · 검증 · 고정 test 확인 실행 스크립트.

검증 절차(기준선 `코드/secom/tune_weighted_models.py`와 같은 골격을 써서
숫자를 나란히 놓을 수 있게 맞췄다):

1. 원본에서 기준선과 동일한 고정 분할을 재현한다(train 1253 / test 314).
2. train에서만 RepeatedStratifiedKFold 5-fold × 3회로 OOF 확률을 만든다.
   전처리·변수선택·이상점수는 모두 fold 학습 데이터에서만 적합한다.
3. OOF 확률에서 recall >= 0.60 조건으로 F2를 최대화하는 임계값을 고른다.
4. OOF F2로 후보를 고른다. test는 이 단계에서 보지 않는다.
5. 선택된 후보만 train 전체로 다시 학습하고, 3단계 임계값 그대로 고정 test를 한 번 본다.

실행:
    .venv\\Scripts\\python.exe "experiments\\independent_validation\\secom\\train_validate.py"
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    MIN_RECALL,
    PROJECT_DIR,
    RANDOM_STATE,
    RESULT_DIR,
    file_sha256,
    load_fixed_split,
    RAW_DIR,
    score_at_threshold,
    select_threshold,
    to_markdown_table,
    write_json,
)
from models import CANDIDATE_DESCRIPTIONS, build_candidates  # noqa: E402

N_SPLITS = 5
N_REPEATS = 3


def _setup_korean_font() -> str:
    """그래프에 한글이 깨지지 않게 폰트를 맞춘다. 없으면 영문 라벨로 둔다."""
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    matplotlib.rcParams["axes.unicode_minus"] = False
    return ""


def run_oof(
    name: str, pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[np.ndarray, list[dict]]:
    """반복 교차검증 OOF 확률(반복 평균)과 fold별 기록을 만든다."""
    y = y_train.to_numpy()
    oof = np.full((len(y), N_REPEATS), np.nan)
    fold_rows: list[dict] = []

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X_train, y)):
        repeat = fold_index // N_SPLITS
        model = clone(pipeline)
        model.fit(X_train.iloc[train_idx], y[train_idx])
        proba = model.predict_proba(X_train.iloc[valid_idx])[:, 1]
        oof[valid_idx, repeat] = proba
        fold_rows.append(
            {
                "candidate": name,
                "repeat": repeat + 1,
                "fold": fold_index % N_SPLITS + 1,
                "valid_rows": int(len(valid_idx)),
                "valid_positives": int(y[valid_idx].sum()),
                "pr_auc": float(average_precision_score(y[valid_idx], proba)),
            }
        )

    if np.isnan(oof).any():
        raise RuntimeError("OOF 확률에 빈 칸이 있습니다. 분할을 확인하세요.")
    return oof.mean(axis=1), fold_rows


def evaluate_candidates(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    candidates = build_candidates()
    summary_rows: list[dict] = []
    fold_records: list[dict] = []
    oof_scores: dict[str, np.ndarray] = {}
    y = y_train.to_numpy()

    for name, pipeline in candidates.items():
        started = time.perf_counter()
        print(f"[OOF] {name} 실행 중 ...", flush=True)
        oof, folds = run_oof(name, pipeline, X_train, y_train)
        elapsed = time.perf_counter() - started

        choice = select_threshold(y, oof, MIN_RECALL)
        metrics = score_at_threshold(y, oof, choice.threshold)
        fold_pr = np.array([row["pr_auc"] for row in folds])

        oof_scores[name] = oof
        fold_records.extend(folds)
        summary_rows.append(
            {
                "candidate": name,
                "description": CANDIDATE_DESCRIPTIONS[name],
                "oof_pr_auc": metrics["pr_auc"],
                "oof_roc_auc": metrics["roc_auc"],
                "fold_pr_auc_mean": float(fold_pr.mean()),
                "fold_pr_auc_std": float(fold_pr.std(ddof=1)),
                "oof_threshold": metrics["threshold"],
                "oof_precision": metrics["precision"],
                "oof_recall": metrics["recall"],
                "oof_f1": metrics["f1"],
                "oof_f2": metrics["f2"],
                "recall_constraint_met": choice.constraint_met,
                "fit_seconds": round(elapsed, 1),
            }
        )
        print(
            f"       PR-AUC={metrics['pr_auc']:.4f} "
            f"recall={metrics['recall']:.4f} F2={metrics['f2']:.4f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    # 조건을 못 지킨 후보는 아래로 내린다. 조건 충족이 우선, 그다음 F2.
    summary = summary.sort_values(
        ["recall_constraint_met", "oof_f2"], ascending=[False, False]
    ).reset_index(drop=True)
    return summary, pd.DataFrame(fold_records), oof_scores


def make_plots(
    summary: pd.DataFrame,
    selected: str,
    y_train: np.ndarray,
    oof: np.ndarray,
    y_test: np.ndarray,
    test_proba: np.ndarray,
    threshold: float,
    result_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt

    from common import fbeta

    saved: list[str] = []

    # 1) 후보별 OOF F2 비교
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = summary.sort_values("oof_f2")
    colors = ["#c0392b" if n == selected else "#7f8c8d" for n in order["candidate"]]
    ax.barh(order["candidate"], order["oof_f2"], color=colors)
    for y_pos, value in enumerate(order["oof_f2"]):
        ax.text(value + 0.004, y_pos, f"{value:.4f}", va="center", fontsize=9)
    ax.set_xlabel("OOF F2 (5-fold x 3회 반복)")
    ax.set_title("독립 검증 후보 비교 — 선택 모델은 붉은색")
    ax.set_xlim(0, max(order["oof_f2"]) * 1.2)
    fig.tight_layout()
    path = result_dir / "candidate_oof_f2.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path.name)

    # 2) OOF / test PR 곡선
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for label, y_true, score, style in (
        ("OOF (train 1253건)", y_train, oof, "-"),
        ("고정 test (314건)", y_test, test_proba, "--"),
    ):
        precision, recall, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax.plot(recall, precision, style, label=f"{label} · PR-AUC={ap:.4f}")
    base_rate = float(np.mean(y_test))
    ax.axhline(base_rate, color="#999999", lw=1, ls=":", label=f"무작위 기준 {base_rate:.4f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"독립 검증 선택 모델 PR 곡선 — {selected}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = result_dir / "selected_pr_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path.name)

    # 3) 임계값 민감도 (고정 test)
    grid = np.linspace(0.01, 0.99, 197)
    rows = []
    for value in grid:
        pred = (test_proba >= value).astype(int)
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append((value, precision, recall, fbeta(precision, recall)))
    sweep = np.array(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(sweep[:, 0], sweep[:, 1], label="precision")
    ax.plot(sweep[:, 0], sweep[:, 2], label="recall")
    ax.plot(sweep[:, 0], sweep[:, 3], label="F2")
    ax.axvline(
        threshold, color="#c0392b", ls="--", lw=1.2, label=f"OOF 고정 임계값 {threshold:.4f}"
    )
    ax.set_xlabel("확률 임계값")
    ax.set_ylabel("점수")
    ax.set_title("고정 test 임계값 민감도 — 임계값은 OOF에서만 결정")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = result_dir / "test_threshold_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path.name)

    np.savetxt(
        result_dir / "test_threshold_sweep.csv",
        sweep,
        delimiter=",",
        header="threshold,precision,recall,f2",
        comments="",
        fmt="%.6f",
    )
    return saved


def write_summary(
    result_dir: Path,
    summary: pd.DataFrame,
    selected_row: pd.Series,
    test_metrics: dict,
    font_name: str,
) -> None:
    selected = selected_row["candidate"]
    table = summary[
        [
            "candidate",
            "oof_pr_auc",
            "fold_pr_auc_mean",
            "fold_pr_auc_std",
            "oof_precision",
            "oof_recall",
            "oof_f2",
            "oof_threshold",
            "recall_constraint_met",
        ]
    ].round(4)

    lines = [
        "# 독립 검증 대안 실험 검증 결과 (SECOM)",
        "",
        "이 문서는 `결과물/secom/independent_validation/`에 보관하는 독립 실험 결과입니다.",
        "학습·후보 선정은 운영 모델 구현을 import하지 않고 원본에서 별도로 수행했습니다.",
        "최종 비교 단계만 저장된 기준 지표를 읽기 전용으로 사용합니다.",
        "",
        "## 데이터와 절차",
        "",
        "- 원본: UCI SECOM 1567행 × 590변수, 불량 104건",
        "- 분할: 기준선과 동일한 고정 분할을 원본에서 재현 (train 1253 / test 314, seed 42)",
        "- 전처리는 기준선 산출물을 쓰지 않고 독립 검증 코드에서 다시 계산",
        "- 제거 기준: 결측률 ≥ 0.45, 최빈값 비율 ≥ 0.995 (모두 **학습 fold 안에서만** 판정)",
        "- 결측 지시 변수 추가, 중앙값 대치, QuantileTransformer(정규분포) 적용",
        f"- 검증: RepeatedStratifiedKFold {N_SPLITS}-fold × {N_REPEATS}회, OOF 확률은 반복 평균",
        f"- 임계값: OOF에서 recall ≥ {MIN_RECALL:.2f} 조건의 F2 최대점",
        "- 고정 test는 후보 선택이 끝난 뒤 1회만 확인",
        "",
        "## 후보 비교 (OOF, test 미사용)",
        "",
        to_markdown_table(table),
        "",
        "후보 설명:",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(f"- `{row['candidate']}`: {row['description']}")

    lines += [
        "",
        "## 선택 모델",
        "",
        f"- 후보: **{selected}**",
        f"- 설명: {selected_row['description']}",
        f"- OOF PR-AUC: {selected_row['oof_pr_auc']:.4f}",
        f"- OOF precision: {selected_row['oof_precision']:.4f}",
        f"- OOF recall: {selected_row['oof_recall']:.4f}",
        f"- OOF F2: {selected_row['oof_f2']:.4f}",
        f"- fold PR-AUC 평균 ± 표본 표준편차: "
        f"{selected_row['fold_pr_auc_mean']:.4f} ± {selected_row['fold_pr_auc_std']:.4f}",
        f"- OOF에서 고정한 임계값: {selected_row['oof_threshold']:.6f}",
        "",
        "## 고정 test 확인 (1회)",
        "",
        f"- PR-AUC: {test_metrics['pr_auc']:.4f}",
        f"- ROC-AUC: {test_metrics['roc_auc']:.4f}",
        f"- precision: {test_metrics['precision']:.4f}",
        f"- recall: {test_metrics['recall']:.4f}",
        f"- F1: {test_metrics['f1']:.4f}",
        f"- F2: {test_metrics['f2']:.4f}",
        f"- 혼동행렬: TP={test_metrics['tp']}, FP={test_metrics['fp']}, "
        f"FN={test_metrics['fn']}, TN={test_metrics['tn']}",
        "",
        "## 한계",
        "",
        "- 고정 test 314건에는 불량이 21건뿐입니다. 지표 한 개의 차이가 웨이퍼 1~2건에서 나옵니다.",
        "  따라서 이 숫자만으로 기준선과의 우열을 단정하지 않습니다.",
        "- 이 test 분할은 기준선 실험에서 이미 여러 번 관찰된 세트입니다.",
        "  완전히 새로운 외부 검증이 아니라 '같은 조건에서의 확인'으로만 읽어야 합니다.",
        "- SECOM 센서는 익명이므로 어떤 변수가 선택되어도 물리적 고장 원인이나",
        "  공정 조정 지시로 해석할 수 없습니다.",
        "- 여기 결과는 기존 모델을 대체하자는 제안이 아니라, 최종 비교·선택 단계에서",
        "  나란히 놓고 보기 위한 대안 후보입니다.",
        "",
        "## 재현",
        "",
        "```powershell",
        '.venv\\Scripts\\python.exe "experiments\\independent_validation\\secom\\train_validate.py"',
        "```",
        "",
    ]
    if not font_name:
        lines += [
            "> 그래프 생성 시 한글 폰트를 찾지 못해 일부 라벨이 깨질 수 있습니다.",
            "",
        ]
    (result_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    font_name = _setup_korean_font()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_fixed_split()
    print(
        f"고정 분할 재현: train={len(X_train)} (불량 {int(y_train.sum())}), "
        f"test={len(X_test)} (불량 {int(y_test.sum())})",
        flush=True,
    )

    summary, folds, oof_scores = evaluate_candidates(X_train, y_train)
    summary.to_csv(RESULT_DIR / "candidate_oof_summary.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(RESULT_DIR / "candidate_fold_scores.csv", index=False, encoding="utf-8-sig")

    selected_row = summary.iloc[0]
    selected = str(selected_row["candidate"])
    threshold = float(selected_row["oof_threshold"])
    print(f"\n선택 후보: {selected} (OOF F2={selected_row['oof_f2']:.4f})", flush=True)

    # 선택이 끝난 뒤에만 고정 test를 본다.
    final_model = build_candidates()[selected]
    final_model.fit(X_train, y_train.to_numpy())
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_metrics = score_at_threshold(y_test.to_numpy(), test_proba, threshold)
    print(
        f"고정 test: PR-AUC={test_metrics['pr_auc']:.4f} "
        f"recall={test_metrics['recall']:.4f} F2={test_metrics['f2']:.4f}",
        flush=True,
    )

    model_path = RESULT_DIR / "independent_extra_trees.joblib"
    joblib.dump(final_model, model_path, compress=3)

    pd.DataFrame(
        {
            "row_id": X_test.index,
            "label": y_test.to_numpy(),
            "probability": test_proba,
            "predicted": (test_proba >= threshold).astype(int),
        }
    ).to_csv(RESULT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    plots = make_plots(
        summary,
        selected,
        y_train.to_numpy(),
        oof_scores[selected],
        y_test.to_numpy(),
        test_proba,
        threshold,
        RESULT_DIR,
    )

    write_json(
        RESULT_DIR / "selected_model.json",
        {
            "candidate": selected,
            "description": CANDIDATE_DESCRIPTIONS[selected],
            "selection_rule": f"OOF recall >= {MIN_RECALL} 조건에서 F2 최대",
            "oof_metrics": {
                key: (
                    float(selected_row[key])
                    if key != "recall_constraint_met"
                    else bool(selected_row[key])
                )
                for key in (
                    "oof_pr_auc",
                    "oof_roc_auc",
                    "fold_pr_auc_mean",
                    "fold_pr_auc_std",
                    "oof_precision",
                    "oof_recall",
                    "oof_f1",
                    "oof_f2",
                    "oof_threshold",
                    "recall_constraint_met",
                )
            },
            "threshold": threshold,
            "expected_local_model_file": model_path.name,
            "model_sha256": file_sha256(model_path),
            "artifact_committed": False,
            "artifact_policy": "현재 환경에서 재학습하고 SHA-256을 재검증한 뒤에만 사용",
        },
    )

    write_json(
        RESULT_DIR / "test_confirmation.json",
        {
            "candidate": selected,
            "threshold_source": "OOF에서 고정, test로 조정하지 않음",
            "test_metrics": test_metrics,
            "test_note": (
                "고정 test는 기준선 실험에서 이미 관찰된 세트이므로 확인용이며 "
                "새로운 외부 검증이 아닙니다. 불량 21건 기준이라 지표 변동이 큽니다."
            ),
        },
    )

    write_json(
        RESULT_DIR / "run_metadata.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "random_state": RANDOM_STATE,
            "cv": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS},
            "min_recall": MIN_RECALL,
            "rows": {
                "train": int(len(X_train)),
                "train_positive": int(y_train.sum()),
                "test": int(len(X_test)),
                "test_positive": int(y_test.sum()),
            },
            "raw_data_sha256": {
                "secom.data": file_sha256(RAW_DIR / "secom.data"),
                "secom_labels.data": file_sha256(RAW_DIR / "secom_labels.data"),
            },
            "packages": _package_versions(),
            "generated_local_plots": plots,
            "plots_committed": False,
            "matplotlib_font": font_name or "기본값(한글 미지원 가능)",
        },
    )

    write_summary(RESULT_DIR, summary, selected_row, test_metrics, font_name)
    print(f"\n결과 저장: {RESULT_DIR.relative_to(PROJECT_DIR)}", flush=True)


def _package_versions() -> dict:
    import importlib

    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib"):
        module_name = "sklearn" if name == "scikit-learn" else name
        try:
            module = importlib.import_module(module_name)
            versions[name] = getattr(module, "__version__", "?")
        except Exception:  # pragma: no cover - 설치 환경에 따라 다름
            versions[name] = "미설치"
    return versions


if __name__ == "__main__":
    main()

