"""Diagnose the deployed WM-811K model's three weakest supported classes."""

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


WEAK_CLASSES = ("Scratch", "Loc", "Edge-Loc")
PROBABILITY_COLUMNS = {
    "Center": "probability_center",
    "Donut": "probability_donut",
    "Edge-Loc": "probability_edge_loc",
    "Edge-Ring": "probability_edge_ring",
    "Loc": "probability_loc",
    "Near-full": "probability_near_full",
    "Random": "probability_random",
    "Scratch": "probability_scratch",
    "none": "probability_none",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrated_probabilities(frame: pd.DataFrame, temperature: float) -> np.ndarray:
    raw = frame[list(PROBABILITY_COLUMNS.values())].to_numpy(dtype=float)
    if not np.isfinite(raw).all() or (raw < 0).any() or not np.allclose(raw.sum(axis=1), 1, atol=2e-5):
        raise ValueError("예측 확률 열이 유효한 확률 분포가 아닙니다.")
    scaled = np.power(np.clip(raw, 1e-12, 1.0), 1.0 / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def analyze(
    predictions_path: Path,
    error_analysis_path: Path,
    policy_path: Path,
    ood_path: Path,
    xai_records_path: Path,
    xai_summary_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    predictions = pd.read_csv(predictions_path)
    error_analysis = json.loads(error_analysis_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    ood = pd.read_csv(ood_path)
    xai = pd.read_csv(xai_records_path)
    xai_summary = json.loads(xai_summary_path.read_text(encoding="utf-8"))

    if len(predictions) != error_analysis["total_rows"] or int((~predictions["correct"].astype(bool)).sum()) != error_analysis["total_errors"]:
        raise ValueError("표본별 예측이 배포 오류 요약과 일치하지 않습니다.")
    checkpoint_hash = file_sha256(checkpoint_path)
    if xai_summary.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("XAI 기록이 배포 체크포인트와 다릅니다.")

    temperature = float(policy["temperature"])
    review_threshold = float(policy["review_threshold"])
    probabilities = calibrated_probabilities(predictions, temperature)
    ordered = np.sort(probabilities, axis=1)
    predictions = predictions.copy()
    predictions["calibrated_confidence"] = ordered[:, -1]
    predictions["confidence_margin"] = ordered[:, -1] - ordered[:, -2]
    predictions["normalized_entropy"] = -(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))
    ).sum(axis=1) / np.log(probabilities.shape[1])
    predictions["review_required"] = predictions["calibrated_confidence"] < review_threshold

    test_ood = ood.loc[ood["split"].eq("test"), ["array_index", "ood_score", "ood_status"]]
    predictions = predictions.merge(test_ood, on="array_index", validate="one_to_one")
    weak = predictions.loc[predictions["failure_type"].isin(WEAK_CLASSES)].copy()
    weak["error_group"] = np.select(
        [
            weak["correct"].astype(bool),
            weak["predicted_failure_type"].eq("none"),
        ],
        ["correct", "predicted_none"],
        default="predicted_other_defect",
    )
    weak["automatic_error"] = ~weak["correct"].astype(bool) & ~weak["review_required"]
    weak["geometry"] = weak["original_height"].astype(str) + "x" + weak["original_width"].astype(str)

    reference = {row["class_name"]: row for row in error_analysis["class_analysis"]}
    summary_rows = []
    for class_name in WEAK_CLASSES:
        part = weak.loc[weak["failure_type"].eq(class_name)]
        errors = part.loc[~part["correct"].astype(bool)]
        to_none = errors["predicted_failure_type"].eq("none")
        xai_part = xai.loc[xai["true_class"].eq(class_name)]
        row = {
            "class_name": class_name,
            "support": len(part),
            "correct_count": int(part["correct"].astype(bool).sum()),
            "error_count": len(errors),
            "recall": float(part["correct"].astype(bool).mean()),
            "errors_to_none": int(to_none.sum()),
            "errors_to_none_share": float(to_none.mean()),
            "mean_confidence_correct": float(part.loc[part["correct"].astype(bool), "calibrated_confidence"].mean()),
            "mean_confidence_error": float(errors["calibrated_confidence"].mean()),
            "review_count": int(part["review_required"].sum()),
            "error_capture_count": int(errors["review_required"].sum()),
            "error_capture_rate": float(errors["review_required"].mean()),
            "automatic_error_count": int(errors["automatic_error"].sum()),
            "ood_review_or_hold_count": int(part["ood_status"].ne("in_distribution").sum()),
            "xai_sample_count": int(xai_part["array_index"].nunique()),
        }
        for method, suffix in (("Grad-CAM", "gradcam"), ("Integrated Gradients", "integrated_gradients")):
            method_part = xai_part.loc[xai_part["method"].eq(method)]
            row[f"{suffix}_top_vs_random"] = float(method_part["top_vs_random_advantage"].mean())
            row[f"{suffix}_active_attribution_lift"] = float(method_part["active_attribution_lift"].mean())
        expected = reference[class_name]
        if (row["support"], row["error_count"]) != (expected["support"], expected["error_count"]):
            raise ValueError(f"{class_name} 집계가 배포 오류 요약과 다릅니다.")
        summary_rows.append(row)
    class_summary = pd.DataFrame(summary_rows)

    pairs = (
        weak.loc[~weak["correct"].astype(bool)]
        .groupby(["failure_type", "predicted_failure_type"], as_index=False)
        .agg(
            count=("array_index", "size"),
            mean_confidence=("calibrated_confidence", "mean"),
            review_capture_rate=("review_required", "mean"),
        )
    )
    pairs["share_within_true_class_errors"] = pairs["count"] / pairs.groupby("failure_type")["count"].transform("sum")
    pairs = pairs.sort_values(["count", "failure_type"], ascending=[False, True])

    def grouped_rate(column: str) -> pd.DataFrame:
        result = weak.groupby(["failure_type", column], as_index=False).agg(
            support=("array_index", "size"),
            error_count=("correct", lambda values: int((~values.astype(bool)).sum())),
            review_count=("review_required", "sum"),
            mean_confidence=("calibrated_confidence", "mean"),
        )
        result["error_rate"] = result["error_count"] / result["support"]
        result["review_rate"] = result["review_count"] / result["support"]
        result["rate_reliable"] = result["support"] >= 5
        return result.sort_values(["failure_type", "error_rate", "support"], ascending=[True, False, False])

    lot_summary = grouped_rate("lot_name")
    geometry_summary = grouped_rate("geometry")

    output_dir.mkdir(parents=True, exist_ok=True)
    weak.sort_values(["failure_type", "array_index"]).to_csv(output_dir / "weak_class_sample_diagnostics.csv", index=False)
    class_summary.to_csv(output_dir / "weak_class_summary.csv", index=False)
    pairs.to_csv(output_dir / "weak_class_confusion_pairs.csv", index=False)
    lot_summary.to_csv(output_dir / "weak_class_lot_summary.csv", index=False)
    geometry_summary.to_csv(output_dir / "weak_class_geometry_summary.csv", index=False)

    total_errors = int(class_summary["error_count"].sum())
    total_to_none = int(class_summary["errors_to_none"].sum())
    total_automatic = int(class_summary["automatic_error_count"].sum())
    summary = {
        "schema_version": 1,
        "purpose": "post_selection_weak_class_diagnostic",
        "used_for_model_selection": False,
        "thresholds_retuned": False,
        "test_rows": len(predictions),
        "weak_classes": list(WEAK_CLASSES),
        "weak_class_rows": len(weak),
        "weak_class_errors": total_errors,
        "errors_to_none": total_to_none,
        "errors_to_none_share": total_to_none / total_errors,
        "automatic_error_count": total_automatic,
        "reviewed_error_count": total_errors - total_automatic,
        "review_error_capture_rate": (total_errors - total_automatic) / total_errors,
        "temperature": temperature,
        "review_threshold": review_threshold,
        "checkpoint_sha256": checkpoint_hash,
        "prediction_source_sha256": file_sha256(predictions_path),
        "xai_source_sha256": file_sha256(xai_records_path),
        "class_summary": summary_rows,
        "primary_failure_mode": "defect_to_none_collapse",
        "recommended_next_experiment": "train_only_subtle_defect_augmentation_and_none_hard_negative_sampling",
        "experiment_guardrail": "validation_only_for_selection_keep_test_locked",
        "limitations": [
            "test 사후 진단이며 현재 모델 선택이나 임계값 조정에 사용하지 않았습니다.",
            "lot·geometry 오류율은 표본 5개 이상 그룹만 비교 대상으로 봅니다.",
            "XAI는 클래스별 최대 50개 고정 표본의 민감도이며 공정 원인이나 인과관계를 증명하지 않습니다.",
        ],
    }
    (output_dir / "weak_class_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    positions = np.arange(len(class_summary))
    axes[0].bar(positions - 0.18, class_summary["recall"], width=0.36, label="Recall", color="#4c78a8")
    axes[0].bar(positions + 0.18, class_summary["errors_to_none_share"], width=0.36, label="Error to none share", color="#e15759")
    axes[0].set_xticks(positions, class_summary["class_name"]); axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8); axes[0].set_title("Weak-class outcomes")
    axes[1].bar(positions - 0.18, class_summary["mean_confidence_correct"], width=0.36, label="Correct", color="#59a14f")
    axes[1].bar(positions + 0.18, class_summary["mean_confidence_error"], width=0.36, label="Error", color="#f28e2b")
    axes[1].set_xticks(positions, class_summary["class_name"]); axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8); axes[1].set_title("Calibrated confidence")
    axes[2].bar(positions - 0.18, class_summary["gradcam_top_vs_random"], width=0.36, label="Grad-CAM", color="#4c78a8")
    axes[2].bar(positions + 0.18, class_summary["integrated_gradients_top_vs_random"], width=0.36, label="Integrated Gradients", color="#f28e2b")
    axes[2].axhline(0, color="black", linewidth=0.8); axes[2].set_xticks(positions, class_summary["class_name"]); axes[2].legend(fontsize=8); axes[2].set_title("Top-minus-random masking")
    fig.tight_layout(); fig.savefig(output_dir / "weak_class_dashboard.png", dpi=170); plt.close(fig)

    lines = [
        "# WM-811K 취약 클래스 진단",
        "",
        f"Scratch·Loc·Edge-Loc 오류 {total_errors}건 중 {total_to_none}건({total_to_none / total_errors:.1%})이 none 예측입니다.",
        f"고정 신뢰도 정책은 이 오류 중 {total_errors - total_automatic}건({(total_errors - total_automatic) / total_errors:.1%})을 검토 대상으로 포착했습니다.",
        "",
        "다음 실험은 train 표본에만 약한 결함 보존 증강과 none hard-negative sampling을 적용하고 validation으로만 선택합니다. Test는 최종 확인 전까지 잠급니다.",
        "",
        "이 결과는 test 사후 진단이며 현재 모델이나 임계값을 변경하지 않습니다.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--error-analysis", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--ood", type=Path, required=True)
    parser.add_argument("--xai-records", type=Path, required=True)
    parser.add_argument("--xai-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.predictions, args.error_analysis, args.policy, args.ood, args.xai_records, args.xai_summary, args.checkpoint, args.output_dir), ensure_ascii=False, indent=2))
