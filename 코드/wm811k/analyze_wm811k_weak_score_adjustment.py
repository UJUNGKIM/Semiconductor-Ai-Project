"""Audit conservative weak-class score adjustments on validation predictions only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score


WEAK_LABELS = (2, 4, 7)  # Edge-Loc, Loc, Scratch
NONE_LABEL = 8
MULTIPLIERS = (1.00, 1.05, 1.10, 1.15, 1.20)
PROBABILITY_COLUMNS = (
    "probability_center", "probability_donut", "probability_edge_loc",
    "probability_edge_ring", "probability_loc", "probability_near_full",
    "probability_random", "probability_scratch", "probability_none",
)


def adjusted_predictions(probabilities: np.ndarray, multiplier: float) -> np.ndarray:
    adjusted = np.asarray(probabilities, dtype=np.float64).copy()
    adjusted[:, WEAK_LABELS] *= float(multiplier)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted.argmax(axis=1)


def metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(true, predicted),
        "balanced_accuracy": balanced_accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, average="macro"),
        "weak_recall": recall_score(true, predicted, labels=list(WEAK_LABELS), average="macro"),
        "none_recall": recall_score(true, predicted, labels=[NONE_LABEL], average="macro"),
    }


def stable_lot_folds(lot_names: pd.Series, folds: int = 5) -> np.ndarray:
    if folds < 2:
        raise ValueError("folds must be at least two")
    return lot_names.fillna("missing").astype(str).map(
        lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % folds
    ).to_numpy(dtype=int)


def choose_candidate(
    candidates: pd.DataFrame,
    macro_f1_tolerance: float = 0.005,
    none_recall_tolerance: float = 0.002,
) -> tuple[float, pd.DataFrame]:
    baseline = candidates.loc[candidates["multiplier"].eq(1.0)].iloc[0]
    result = candidates.copy()
    result["macro_f1_guardrail"] = result["macro_f1"] >= baseline["macro_f1"] - macro_f1_tolerance
    result["none_recall_guardrail"] = result["none_recall"] >= baseline["none_recall"] - none_recall_tolerance
    result["eligible"] = result["macro_f1_guardrail"] & result["none_recall_guardrail"]
    selected = result.loc[result["eligible"]].sort_values(
        ["weak_recall", "macro_f1", "multiplier"], ascending=[False, False, True]
    ).iloc[0]
    return float(selected["multiplier"]), result


def run(predictions_path: Path, output_dir: Path, folds: int = 5) -> dict[str, object]:
    frame = pd.read_csv(predictions_path)
    if frame.empty or set(frame["split"].astype(str)) != {"validation"}:
        raise ValueError("Only non-empty validation predictions are accepted")
    missing = sorted(set(PROBABILITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    true = frame["label_id"].to_numpy(dtype=int)
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5
    ):
        raise ValueError("Invalid probability matrix")

    candidate_rows = []
    for multiplier in MULTIPLIERS:
        candidate_rows.append({
            "multiplier": multiplier,
            **metrics(true, adjusted_predictions(probabilities, multiplier)),
        })
    selected, candidates = choose_candidate(pd.DataFrame(candidate_rows))
    baseline = candidates.loc[candidates["multiplier"].eq(1.0)].iloc[0]
    selected_row = candidates.loc[candidates["multiplier"].eq(selected)].iloc[0]

    fold_ids = stable_lot_folds(frame["lot_name"], folds)
    fold_rows = []
    for fold in range(folds):
        mask = fold_ids == fold
        baseline_metrics = metrics(true[mask], adjusted_predictions(probabilities[mask], 1.0))
        adjusted_metrics = metrics(true[mask], adjusted_predictions(probabilities[mask], selected))
        fold_rows.append({
            "fold": fold,
            "rows": int(mask.sum()),
            "lots": int(frame.loc[mask, "lot_name"].nunique()),
            "multiplier": selected,
            **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
            **{f"adjusted_{key}": value for key, value in adjusted_metrics.items()},
            **{f"delta_{key}": adjusted_metrics[key] - baseline_metrics[key] for key in baseline_metrics},
        })
    fold_audit = pd.DataFrame(fold_rows)
    weak_improved_folds = int((fold_audit["delta_weak_recall"] > 0).sum())
    weak_not_worse_folds = int((fold_audit["delta_weak_recall"] >= 0).sum())
    fold_macro_guardrail_passes = int((fold_audit["delta_macro_f1"] >= -0.005).sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "score_adjustment_candidates.csv", index=False)
    fold_audit.to_csv(output_dir / "lot_stability_audit.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    candidates.plot(
        x="multiplier", y=["macro_f1", "weak_recall", "none_recall"],
        marker="o", ax=axes[0], ylim=(0.84, 1.0),
    )
    axes[0].axvline(selected, color="black", linestyle="--", alpha=0.5)
    axes[0].set_title("Validation-only score adjustment")
    axes[0].set_ylabel("score")
    fold_audit.plot.bar(
        x="fold", y=["delta_weak_recall", "delta_macro_f1", "delta_none_recall"],
        ax=axes[1], rot=0,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"Lot-fold stability at x{selected:.2f}")
    axes[1].set_ylabel("adjusted - baseline")
    figure.tight_layout()
    figure.savefig(output_dir / "score_adjustment_dashboard.png", dpi=170)
    plt.close(figure)

    summary: dict[str, object] = {
        "schema_version": 1,
        "purpose": "exploratory_validation_only_weak_score_adjustment",
        "source_predictions": str(predictions_path),
        "source_split": "validation",
        "rows": len(frame),
        "lots": int(frame["lot_name"].nunique()),
        "test_evaluated": False,
        "deployment_changed": False,
        "candidate_multipliers": list(MULTIPLIERS),
        "selected_multiplier": selected,
        "selection_metric": "weak_recall_with_macro_f1_and_none_recall_guardrails",
        "macro_f1_tolerance": 0.005,
        "none_recall_tolerance": 0.002,
        "baseline": {key: float(baseline[key]) for key in ("accuracy", "balanced_accuracy", "macro_f1", "weak_recall", "none_recall")},
        "selected": {key: float(selected_row[key]) for key in ("accuracy", "balanced_accuracy", "macro_f1", "weak_recall", "none_recall")},
        "deltas": {key: float(selected_row[key] - baseline[key]) for key in ("accuracy", "balanced_accuracy", "macro_f1", "weak_recall", "none_recall")},
        "lot_folds": folds,
        "weak_recall_improved_folds": weak_improved_folds,
        "weak_recall_not_worse_folds": weak_not_worse_folds,
        "macro_f1_guardrail_pass_folds": fold_macro_guardrail_passes,
        "decision": "evidence_for_margin_training_not_for_deployment",
        "next_step": "train_validation_only_weak_vs_none_margin_candidates",
    }
    (output_dir / "score_adjustment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# WM-811K 취약 클래스 점수 보정 탐색\n\n"
        f"Validation 예측에서 취약 클래스 점수를 {selected:.2f}배로 조정했을 때 "
        f"평균 재현율이 {baseline['weak_recall']:.2%}에서 {selected_row['weak_recall']:.2%}로 변했습니다.\n\n"
        f"Lot 5-fold 중 개선 {weak_improved_folds}개, 비열화 {weak_not_worse_folds}개이며 "
        f"Macro-F1 fold 안전 기준은 {fold_macro_guardrail_passes}/5개가 통과했습니다.\n\n"
        "이는 다음 margin-loss 학습 후보를 뒷받침하는 탐색 결과일 뿐입니다. "
        "Test를 평가하지 않았고 추론 점수나 배포 모델을 변경하지 않았습니다.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    run(args.predictions, args.output_dir, args.folds)


if __name__ == "__main__":
    main()
