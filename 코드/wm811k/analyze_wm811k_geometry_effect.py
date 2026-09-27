"""Estimate class-composition-adjusted WM-811K geometry gaps without changing policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def adjusted_geometry_metrics(rows: pd.DataFrame, minimum_support: int = 100) -> pd.DataFrame:
    """Compare each geometry with leave-one-geometry-out recall within each class."""
    if minimum_support < 2:
        raise ValueError("minimum_support는 2 이상이어야 합니다.")
    rows = rows.copy()
    rows["correct"] = rows["label_id"] == rows["predicted_label_id"]
    class_totals = rows.groupby("label_id")["correct"].agg(["sum", "count"])
    results = []
    for (height, width), group in rows.groupby(["original_height", "original_width"]):
        if len(group) < minimum_support:
            continue
        group_counts = group.groupby("label_id")["correct"].agg(["sum", "count"])
        expected_correct = 0.0
        valid = True
        for label_id, item in group_counts.iterrows():
            rest_count = int(class_totals.loc[label_id, "count"] - item["count"])
            if rest_count <= 0:
                valid = False
                break
            rest_correct = float(class_totals.loc[label_id, "sum"] - item["sum"])
            expected_correct += float(item["count"]) * rest_correct / rest_count
        if not valid:
            continue
        observed = float(group["correct"].mean())
        expected = expected_correct / len(group)
        results.append({"height": int(height), "width": int(width), "geometry": f"{int(height)}x{int(width)}", "support": len(group), "class_count": int(group["label_id"].nunique()), "observed_accuracy": observed, "class_mix_expected_accuracy": expected, "class_adjusted_gap": observed - expected, "error_count": int((~group["correct"]).sum())})
    return pd.DataFrame(results).sort_values(["class_adjusted_gap", "support", "geometry"]).reset_index(drop=True)


def analyze(assignments_path: Path, records_path: Path, output_dir: Path, *, minimum_support: int = 100, gap_cutoff: float = -0.05) -> dict:
    assignments = pd.read_csv(assignments_path)
    records = pd.read_csv(records_path)
    records = records.loc[records["split"].isin(["validation", "test"])].copy()
    if records["array_index"].duplicated().any():
        raise ValueError("validation/test 예측 기록의 array_index가 중복되었습니다.")
    merged = assignments.merge(records, on="array_index", how="inner", validate="one_to_one", suffixes=("", "_record"))
    merged = merged.loc[merged["split"] == merged["split_record"]].copy()
    if not np.array_equal(merged["label_id"], merged["true_label_id"]):
        raise ValueError("분할 라벨과 예측 기록 라벨이 다릅니다.")
    split_metrics = {}
    tables = []
    for split in ("validation", "test"):
        table = adjusted_geometry_metrics(merged.loc[merged["split"] == split], minimum_support)
        table.insert(0, "split", split)
        split_metrics[split] = table
        tables.append(table)
    validation = split_metrics["validation"]
    test = split_metrics["test"]
    selected = validation.loc[validation["class_adjusted_gap"] <= gap_cutoff, "geometry"].tolist()
    if not selected:
        raise ValueError("validation 기준을 만족하는 geometry 후보가 없습니다.")

    test_rows = merged.loc[merged["split"] == "test"].copy()
    test_rows["geometry"] = test_rows["original_height"].astype(int).astype(str) + "x" + test_rows["original_width"].astype(int).astype(str)
    test_rows["error"] = test_rows["label_id"] != test_rows["predicted_label_id"]
    test_rows["feature_ood_review"] = test_rows["ood_status"] != "in_distribution"
    test_rows["candidate_geometry_review"] = test_rows["geometry"].isin(selected)
    candidate = test_rows.loc[test_rows["candidate_geometry_review"]]
    existing_captured = int((test_rows["error"] & test_rows["feature_ood_review"]).sum())
    combined_review = test_rows["feature_ood_review"] | test_rows["candidate_geometry_review"]
    combined_captured = int((test_rows["error"] & combined_review).sum())
    selected_test = test.loc[test["geometry"].isin(selected)].copy()

    summary = {
        "schema_version": 1,
        "purpose": "post_selection_geometry_confounding_analysis",
        "minimum_support": minimum_support,
        "validation_gap_cutoff": gap_cutoff,
        "threshold_preregistered_before_dataset_analysis": False,
        "selection_split": "validation",
        "test_used_for_candidate_selection": False,
        "candidate_only": True,
        "deployed": False,
        "selected_geometries": selected,
        "selected_geometry_count": len(selected),
        "selected_test_rows": len(candidate),
        "selected_test_review_rate": float(len(candidate) / len(test_rows)),
        "selected_test_accuracy": float((~candidate["error"]).mean()),
        "selected_test_error_count": int(candidate["error"].sum()),
        "selected_test_error_share": float(candidate["error"].sum() / test_rows["error"].sum()),
        "all_selected_geometries_negative_on_test": bool((selected_test["class_adjusted_gap"] < 0).all() and len(selected_test) == len(selected)),
        "test_geometry_results": selected_test[["geometry", "support", "observed_accuracy", "class_mix_expected_accuracy", "class_adjusted_gap", "error_count"]].to_dict("records"),
        "feature_ood_review_rate": float(test_rows["feature_ood_review"].mean()),
        "feature_ood_error_capture_rate": float(existing_captured / test_rows["error"].sum()),
        "hypothetical_combined_review_rate": float(combined_review.mean()),
        "hypothetical_combined_error_capture_rate": float(combined_captured / test_rows["error"].sum()),
        "hypothetical_added_review_rows": int((test_rows["candidate_geometry_review"] & ~test_rows["feature_ood_review"]).sum()),
        "hypothetical_added_captured_errors": combined_captured - existing_captured,
        "limitations": ["후보 기준은 validation에서 정했지만 데이터 분석 전에 사전 등록된 기준은 아닙니다.", "웨이퍼 크기는 장비·공정·결함 형태 등 관측되지 않은 요인의 대리변수일 수 있습니다.", "feature OOD와의 결합 수치는 가상 정책 평가이며 현재 대시보드 판정 규칙에 적용하지 않습니다."],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(output_dir / "geometry_adjusted_metrics.csv", index=False)
    (output_dir / "geometry_candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = validation.merge(test, on="geometry", suffixes=("_validation", "_test"))
    comparison = comparison.sort_values("class_adjusted_gap_validation").head(12).sort_values("class_adjusted_gap_validation")
    y = np.arange(len(comparison))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y - 0.18, comparison["class_adjusted_gap_validation"], 0.36, label="Validation", color="#4c78a8")
    ax.barh(y + 0.18, comparison["class_adjusted_gap_test"], 0.36, label="Fixed test", color="#f28e2b")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(gap_cutoff, color="#e15759", linestyle="--", linewidth=1, label="Validation candidate cutoff")
    ax.set_yticks(y, comparison["geometry"])
    ax.set(xlabel="Observed accuracy - class-mix expected accuracy", title="Geometry performance after class-mix adjustment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "geometry_adjusted_dashboard.png", dpi=160)
    plt.close(fig)

    lines = ["# WM-811K 웨이퍼 크기 클래스 구성 보정 분석", "", f"Validation에서 support ≥ {minimum_support}, 클래스 구성 보정 격차 ≤ {gap_cutoff:.0%}인 후보: {', '.join(selected)}", f"고정 test에서 후보 {len(candidate):,}개의 정확도: {summary['selected_test_accuracy']:.2%}", f"후보에 포함된 test 오류: {summary['selected_test_error_count']}개 / 전체 오류의 {summary['selected_test_error_share']:.2%}", "", "두 후보 모두 고정 test에서도 클래스 구성 보정 후 음의 격차를 보였습니다. 다만 기준은 데이터 분석 전에 사전 등록되지 않았고 크기는 다른 공정 요인의 대리변수일 수 있습니다.", "", f"후보를 feature OOD 검토와 가상 결합하면 검토율 {summary['hypothetical_combined_review_rate']:.2%}, 오류 포착률 {summary['hypothetical_combined_error_capture_rate']:.2%}입니다. 이 정책은 배포하지 않았습니다."]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.assignments, args.records, args.output_dir), ensure_ascii=False, indent=2))
