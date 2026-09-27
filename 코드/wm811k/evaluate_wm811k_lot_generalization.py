"""Audit selected WM-811K predictions across held-out lots and wafer geometries."""

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("correct와 total이 유효하지 않습니다.")
    rate = correct / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return float(center - margin), float(center + margin)


def cluster_bootstrap_accuracy(lot_counts: pd.DataFrame, repeats: int, seed: int) -> tuple[float, float]:
    if repeats < 100:
        raise ValueError("cluster bootstrap repeats는 100 이상이어야 합니다.")
    counts = lot_counts[["correct_count", "support"]].to_numpy(dtype=np.int64)
    if len(counts) < 2 or np.any(counts[:, 1] <= 0):
        raise ValueError("cluster bootstrap에 유효한 lot가 2개 이상 필요합니다.")
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = counts[rng.integers(0, len(counts), size=len(counts))]
        values[index] = sample[:, 0].sum() / sample[:, 1].sum()
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def evaluate(assignments_path: Path, records_path: Path, output_dir: Path, *, bootstrap_repeats: int = 2000, seed: int = 42) -> dict:
    assignments = pd.read_csv(assignments_path)
    records = pd.read_csv(records_path)
    test_assignments = assignments.loc[assignments["split"] == "test"].copy()
    test_records = records.loc[records["split"] == "test"].copy()
    if test_assignments["array_index"].duplicated().any() or test_records["array_index"].duplicated().any():
        raise ValueError("test array_index가 중복되었습니다.")
    rows = test_assignments.merge(test_records, on="array_index", how="inner", validate="one_to_one", suffixes=("", "_record"))
    if len(rows) != len(test_assignments) or len(rows) != len(test_records):
        raise ValueError("고정 test 분할과 OOD 예측 행이 정확히 대응하지 않습니다.")
    if not np.array_equal(rows["label_id"], rows["true_label_id"]):
        raise ValueError("고정 test 라벨과 예측 기록 라벨이 다릅니다.")
    rows["correct"] = rows["label_id"] == rows["predicted_label_id"]
    rows["ood_hold"] = rows["ood_status"] == "out_of_distribution"
    rows["review_or_ood"] = rows["ood_status"] != "in_distribution"

    lot_rows = []
    for lot_name, group in rows.groupby("lot_name", sort=True):
        support = len(group)
        correct = int(group["correct"].sum())
        low, high = wilson_interval(correct, support)
        lot_rows.append({"lot_name": lot_name, "support": support, "class_count": int(group["label_id"].nunique()), "correct_count": correct, "error_count": support - correct, "accuracy": correct / support, "accuracy_ci95_low": low, "accuracy_ci95_high": high, "review_or_ood_rate": float(group["review_or_ood"].mean()), "ood_hold_rate": float(group["ood_hold"].mean())})
    lot_metrics = pd.DataFrame(lot_rows).sort_values(["accuracy", "support", "lot_name"], ascending=[True, False, True])
    bootstrap_low, bootstrap_high = cluster_bootstrap_accuracy(lot_metrics, bootstrap_repeats, seed)

    geometry_rows = []
    for (height, width), group in rows.groupby(["original_height", "original_width"], sort=True):
        support = len(group)
        correct = int(group["correct"].sum())
        low, high = wilson_interval(correct, support)
        geometry_rows.append({"height": int(height), "width": int(width), "geometry": f"{int(height)}x{int(width)}", "support": support, "correct_count": correct, "error_count": support - correct, "accuracy": correct / support, "accuracy_ci95_low": low, "accuracy_ci95_high": high, "review_or_ood_rate": float(group["review_or_ood"].mean()), "ood_hold_rate": float(group["ood_hold"].mean())})
    geometry_metrics = pd.DataFrame(geometry_rows).sort_values(["support", "geometry"], ascending=[False, True])

    class_rows = []
    for (label_id, class_name), group in rows.groupby(["label_id", "failure_type"], sort=True):
        per_lot = group.groupby("lot_name")["correct"].agg(["size", "sum"])
        per_lot["recall"] = per_lot["sum"] / per_lot["size"]
        class_rows.append({"label_id": int(label_id), "class_name": class_name, "support": len(group), "lot_count": len(per_lot), "weighted_recall": float(group["correct"].mean()), "median_lot_recall": float(per_lot["recall"].median()), "p10_lot_recall": float(per_lot["recall"].quantile(0.10)), "min_lot_recall": float(per_lot["recall"].min()), "max_lot_recall": float(per_lot["recall"].max()), "lots_with_errors": int((per_lot["recall"] < 1).sum()), "error_free_lot_rate": float((per_lot["recall"] == 1).mean())})
    class_stability = pd.DataFrame(class_rows).sort_values("label_id")

    output_dir.mkdir(parents=True, exist_ok=True)
    lot_metrics.to_csv(output_dir / "lot_metrics.csv", index=False)
    geometry_metrics.to_csv(output_dir / "geometry_metrics.csv", index=False)
    class_stability.to_csv(output_dir / "class_lot_stability.csv", index=False)

    eligible_lots = lot_metrics.loc[lot_metrics["support"] >= 20]
    eligible_geometries = geometry_metrics.loc[geometry_metrics["support"] >= 100]
    worst_geometry = eligible_geometries.sort_values(["accuracy", "support"]).iloc[0]
    summary = {
        "schema_version": 1,
        "purpose": "post_selection_test_lot_generalization_diagnostics",
        "strategy": "ce_sqrt_balanced",
        "source_split": "test",
        "used_for_model_selection": False,
        "used_for_threshold_selection": False,
        "rows": len(rows),
        "lot_count": len(lot_metrics),
        "lot_support_min": int(lot_metrics["support"].min()),
        "lot_support_median": float(lot_metrics["support"].median()),
        "lot_support_max": int(lot_metrics["support"].max()),
        "global_accuracy": float(rows["correct"].mean()),
        "unweighted_mean_lot_accuracy": float(lot_metrics["accuracy"].mean()),
        "median_lot_accuracy": float(lot_metrics["accuracy"].median()),
        "cluster_bootstrap_accuracy_ci95": [bootstrap_low, bootstrap_high],
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": seed,
        "eligible_lot_definition": "support >= 20",
        "eligible_lot_count": len(eligible_lots),
        "eligible_lot_accuracy_p10": float(eligible_lots["accuracy"].quantile(0.10)),
        "eligible_lot_perfect_rate": float((eligible_lots["accuracy"] == 1).mean()),
        "geometry_count": len(geometry_metrics),
        "eligible_geometry_definition": "support >= 100",
        "eligible_geometry_count": len(eligible_geometries),
        "eligible_geometry_accuracy_min": float(eligible_geometries["accuracy"].min()),
        "lowest_eligible_geometry": str(worst_geometry["geometry"]),
        "lowest_eligible_geometry_support": int(worst_geometry["support"]),
        "test_review_or_ood_rate": float(rows["review_or_ood"].mean()),
        "test_ood_hold_rate": float(rows["ood_hold"].mean()),
        "assignments_sha256": sha256_file(assignments_path),
        "ood_records_sha256": sha256_file(records_path),
        "limitations": ["각 lot은 최대 25개로 작아 개별 lot 순위의 불확실성이 큽니다.", "웨이퍼 크기 그룹은 결함 클래스 구성 차이의 영향을 함께 받으므로 크기의 인과효과가 아닙니다.", "동일한 고정 test의 사후 진단이며 신규 장비·공정 외부 검증이 아닙니다.", "test 결과로 모델이나 검토/OOD 임계값을 변경하지 않습니다."],
    }
    (output_dir / "lot_generalization_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hist(eligible_lots["accuracy"], bins=np.linspace(0, 1, 21), color="#4c78a8", edgecolor="white")
    axes[0].set(title="Held-out lots (support >= 20)", xlabel="Accuracy", ylabel="Lot count", xlim=(0, 1))
    top_geometry = geometry_metrics.head(12).sort_values("accuracy")
    axes[1].barh(top_geometry["geometry"], top_geometry["accuracy"], color="#59a14f")
    axes[1].set(title="12 most common geometries", xlabel="Accuracy", xlim=(0, 1))
    positions = np.arange(len(class_stability))
    axes[2].bar(positions - 0.18, class_stability["weighted_recall"], 0.36, label="Overall recall", color="#f28e2b")
    axes[2].bar(positions + 0.18, class_stability["p10_lot_recall"], 0.36, label="10th percentile lot", color="#e15759")
    axes[2].set_xticks(positions, class_stability["class_name"], rotation=50, ha="right")
    axes[2].set(title="Class recall across lots", ylim=(0, 1))
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "lot_generalization_dashboard.png", dpi=160)
    plt.close(fig)

    lines = ["# WM-811K held-out lot 일반화 감사", "", f"- 고정 test: {len(rows):,}개 / held-out lot: {len(lot_metrics):,}개", f"- 전체 정확도: {summary['global_accuracy']:.4%}", f"- lot cluster bootstrap 95% 구간: {bootstrap_low:.4%}~{bootstrap_high:.4%}", f"- support 20 이상 lot의 정확도 10% 분위수: {summary['eligible_lot_accuracy_p10']:.4%}", f"- support 100 이상 크기 그룹 최저: {summary['lowest_eligible_geometry']} / {summary['eligible_geometry_accuracy_min']:.4%} / {summary['lowest_eligible_geometry_support']}개", "", "개별 lot은 최대 25개로 작으므로 최저 lot 순위를 확정적 품질 순위로 해석하지 않습니다. 웨이퍼 크기별 차이에는 결함 클래스 구성 차이가 섞여 있어 크기의 인과효과가 아닙니다. 이 결과는 고정 test의 사후 진단이며 모델·임계값 선택에 사용하지 않았습니다."]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.assignments, args.records, args.output_dir, bootstrap_repeats=args.bootstrap_repeats, seed=args.seed), ensure_ascii=False, indent=2))
