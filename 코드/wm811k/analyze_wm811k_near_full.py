"""Analyze the 21 Near-full test wafers without changing the deployed model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("Grad-CAM", "Integrated Gradients")
SEED = 42


def bootstrap_ci(values: np.ndarray, seed: int = SEED, repeats: int = 5000) -> list[float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap 입력은 2개 이상의 유한값이어야 합니다.")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def analyze(
    records_path: Path,
    selected_path: Path,
    ood_path: Path,
    policy_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    records = pd.read_csv(records_path)
    selected = pd.read_csv(selected_path)
    ood = pd.read_csv(ood_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    near_records = records.loc[records["true_class"].eq("Near-full")].copy()
    near_selected = selected.loc[selected["label_id"].eq(5)].copy()
    near_ood = ood.loc[ood["split"].eq("test") & ood["true_label_id"].eq(5)].copy()
    if len(near_selected) != 21 or near_selected["array_index"].nunique() != 21:
        raise ValueError("Near-full test 표본은 중복 없이 21개여야 합니다.")
    if len(near_records) != 42 or set(near_records["method"]) != set(METHODS):
        raise ValueError("Near-full 설명 기록은 두 방법 각각 21개여야 합니다.")
    for method in METHODS:
        indices = set(near_records.loc[near_records["method"].eq(method), "array_index"])
        if indices != set(near_selected["array_index"]):
            raise ValueError(f"{method} 표본이 선정표와 다릅니다.")
    if set(near_ood["array_index"]) != set(near_selected["array_index"]):
        raise ValueError("Near-full OOD 기록이 test 표본과 다릅니다.")

    value_columns = [
        "active_attribution_mass",
        "active_attribution_lift",
        "top10_confidence_drop",
        "random10_confidence_drop_mean",
        "bottom10_confidence_drop",
        "top_vs_random_advantage",
        "top_vs_bottom_advantage",
        "top10_prediction_changed",
    ]
    wide = near_records.pivot(index="array_index", columns="method", values=value_columns)
    wide.columns = [f"{metric}_{'gradcam' if method == 'Grad-CAM' else 'integrated_gradients'}" for metric, method in wide.columns]
    prediction = near_records.loc[near_records["method"].eq("Grad-CAM"), ["array_index", "predicted_class", "correct", "baseline_confidence"]]
    metadata_columns = ["array_index", "source_row", "source_split", "lot_name", "wafer_index", "original_height", "original_width"]
    diagnostic = near_selected[metadata_columns].merge(prediction, on="array_index", validate="one_to_one")
    diagnostic = diagnostic.merge(near_ood[["array_index", "ood_score", "ood_status"]], on="array_index", validate="one_to_one")
    diagnostic = diagnostic.merge(wide.reset_index(), on="array_index", validate="one_to_one")
    review_threshold = float(policy["review_threshold"])
    diagnostic["low_confidence_review"] = diagnostic["baseline_confidence"] < review_threshold
    diagnostic["ood_review"] = diagnostic["ood_status"] != "in_distribution"
    diagnostic["review_required"] = diagnostic["low_confidence_review"] | diagnostic["ood_review"]
    diagnostic["geometry"] = diagnostic["original_height"].astype(str) + "x" + diagnostic["original_width"].astype(str)

    method_summary: dict[str, dict[str, object]] = {}
    for offset, method in enumerate(METHODS):
        part = near_records.loc[near_records["method"].eq(method)]
        advantage = part["top_vs_random_advantage"].to_numpy(dtype=float)
        method_summary[method] = {
            "mean_top10_confidence_drop": float(part["top10_confidence_drop"].mean()),
            "mean_random10_confidence_drop": float(part["random10_confidence_drop_mean"].mean()),
            "mean_bottom10_confidence_drop": float(part["bottom10_confidence_drop"].mean()),
            "mean_top_vs_random_advantage": float(part["top_vs_random_advantage"].mean()),
            "top_vs_random_95ci": bootstrap_ci(advantage, SEED + offset),
            "mean_top_vs_bottom_advantage": float(part["top_vs_bottom_advantage"].mean()),
            "top_exceeds_random_rate": float((part["top_vs_random_advantage"] > 0).mean()),
            "top_exceeds_bottom_rate": float((part["top_vs_bottom_advantage"] > 0).mean()),
            "mean_active_attribution_lift": float(part["active_attribution_lift"].mean()),
            "prediction_change_rate_after_top10_mask": float(part["top10_prediction_changed"].astype(bool).mean()),
        }

    paired = (
        near_records.pivot(index="array_index", columns="method", values="top_vs_random_advantage")
        .assign(delta=lambda frame: frame["Integrated Gradients"] - frame["Grad-CAM"])
    )
    summary = {
        "schema_version": 1,
        "purpose": "post_selection_near_full_explanation_diagnostic",
        "used_for_model_selection": False,
        "thresholds_retuned": False,
        "deployment_model_changed": False,
        "test_sample_count": len(diagnostic),
        "test_accuracy": float(diagnostic["correct"].astype(bool).mean()),
        "mean_calibrated_confidence": float(diagnostic["baseline_confidence"].mean()),
        "review_threshold": review_threshold,
        "low_confidence_review_count": int(diagnostic["low_confidence_review"].sum()),
        "ood_review_count": int(diagnostic["ood_review"].sum()),
        "combined_review_count": int(diagnostic["review_required"].sum()),
        "unique_lot_count": int(diagnostic["lot_name"].nunique()),
        "geometry_count": int(diagnostic["geometry"].nunique()),
        "methods": method_summary,
        "paired_ig_minus_gradcam_top_vs_random": float(paired["delta"].mean()),
        "paired_ig_minus_gradcam_95ci": bootstrap_ci(paired["delta"].to_numpy(), SEED + 100),
        "interpretation": "Near-full은 21개 중 19개를 맞혔으며 패턴이 국소 결함이 아니라 넓게 분포하므로 활성 die의 국소 10% 제거는 설명 충실도를 과소평가할 수 있습니다.",
        "recommended_action": "모델·임계값은 유지하고 Near-full 설명에는 국소 삭제 점수보다 전체 결함 면적·연결성·경계 유지 지표를 함께 사용합니다.",
        "limitations": [
            "Near-full test가 21개뿐이므로 정밀한 모집단 추정이 아닙니다.",
            "10% die 제거는 실제 공정 개입이 아닙니다.",
            "확산형 패턴에서는 무작위 삭제도 중요한 영역을 포함하기 쉬워 top-minus-random 지표가 보수적입니다.",
        ],
    }

    geometry = diagnostic.groupby("geometry", as_index=False).agg(
        support=("array_index", "size"),
        lots=("lot_name", "nunique"),
        mean_confidence=("baseline_confidence", "mean"),
        review_rate=("review_required", "mean"),
        mean_ood_score=("ood_score", "mean"),
        gradcam_top_vs_random=("top_vs_random_advantage_gradcam", "mean"),
        integrated_gradients_top_vs_random=("top_vs_random_advantage_integrated_gradients", "mean"),
    ).sort_values(["support", "geometry"], ascending=[False, True])

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic.sort_values("array_index").to_csv(output_dir / "near_full_sample_diagnostics.csv", index=False)
    geometry.to_csv(output_dir / "near_full_geometry_summary.csv", index=False)
    (output_dir / "near_full_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    method_frame = pd.DataFrame(method_summary).T
    method_frame[["mean_top10_confidence_drop", "mean_random10_confidence_drop", "mean_bottom10_confidence_drop"]].plot(
        kind="bar", ax=axes[0], color=["#e15759", "#4c78a8", "#59a14f"]
    )
    axes[0].set(title="Near-full: confidence drop after 10% masking", xlabel="Explanation method", ylabel="Mean confidence drop")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(["Top attribution", "Random", "Bottom attribution"], fontsize=8)
    for method, color in zip(METHODS, ("#4c78a8", "#f28e2b")):
        part = near_records.loc[near_records["method"].eq(method)]
        axes[1].scatter(part["baseline_confidence"], part["top_vs_random_advantage"], label=method, color=color, alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvline(review_threshold, color="#e15759", linestyle="--", linewidth=1, label="Review threshold")
    axes[1].set(title="Confidence and local deletion faithfulness", xlabel="Calibrated confidence", ylabel="Top-minus-random confidence drop")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "near_full_diagnostic_dashboard.png", dpi=170)
    plt.close(fig)

    lines = [
        "# WM-811K Near-full 설명 진단",
        "",
        f"- test 표본: {summary['test_sample_count']}개 (전부 사용)",
        f"- 분류 정확도: {summary['test_accuracy']:.1%}",
        f"- 평균 보정 신뢰도: {summary['mean_calibrated_confidence']:.1%}",
        f"- 전문가 검토 대상: {summary['combined_review_count']}개",
        f"- OOD 경계 검토: {summary['ood_review_count']}개",
        "",
    ]
    for method, values in method_summary.items():
        ci = values["top_vs_random_95ci"]
        lines += [
            f"## {method}",
            "",
            f"- 상위-무작위 10% 제거 효과: {values['mean_top_vs_random_advantage']:+.4f}",
            f"- bootstrap 95% 구간: {ci[0]:+.4f}~{ci[1]:+.4f}",
            f"- 활성 die attribution lift: {values['mean_active_attribution_lift']:.3f}",
            f"- 상위 10% 제거 후 예측 변경: {values['prediction_change_rate_after_top10_mask']:.1%}",
            "",
        ]
    lines += [
        "Near-full은 국소 결함보다 웨이퍼 전반에 퍼진 패턴입니다. 무작위로 활성 die 10%를 제거해도 중요한 영역을 건드릴 가능성이 높으므로 top-minus-random 지표가 낮은 현상만으로 분류 모델 실패라고 판단할 수 없습니다.",
        "",
        "모델과 임계값은 변경하지 않습니다. Near-full 설명에는 전체 결함 면적, 연결성, 경계 유지 같은 전역 형태 지표를 추가하는 것이 다음 개선 방향입니다.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--ood", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.records, args.selected, args.ood, args.policy, args.output_dir), ensure_ascii=False, indent=2))
