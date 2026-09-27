"""Measure global shape descriptors for all fixed-test Near-full wafers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wafer_shape import wafer_shape_features


NEAR_FULL_LABEL_ID = 5
EXPECTED_TEST_SAMPLES = 21


def analyze(
    maps_path: Path,
    labels_path: Path,
    assignments_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    maps = np.load(maps_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    assignments = pd.read_csv(assignments_path)
    required = {"array_index", "split", "label_id"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"split assignment columns missing: {sorted(missing)}")
    selected = assignments.loc[
        assignments["split"].eq("test")
        & assignments["label_id"].eq(NEAR_FULL_LABEL_ID)
    ].copy()
    if (
        len(selected) != EXPECTED_TEST_SAMPLES
        or selected["array_index"].nunique() != EXPECTED_TEST_SAMPLES
    ):
        raise ValueError("fixed test split must contain 21 unique Near-full wafers")

    rows: list[dict[str, object]] = []
    metadata_columns = [
        column
        for column in (
            "array_index",
            "source_row",
            "lot_name",
            "wafer_index",
            "original_height",
            "original_width",
        )
        if column in selected.columns
    ]
    for assignment in selected.sort_values("array_index").to_dict("records"):
        index = int(assignment["array_index"])
        if not 0 <= index < len(maps) or int(labels[index]) != NEAR_FULL_LABEL_ID:
            raise ValueError(f"assignment and array labels disagree at index {index}")
        row = {column: assignment[column] for column in metadata_columns}
        row.update(wafer_shape_features(maps[index]))
        rows.append(row)

    records = pd.DataFrame(rows)
    metric_columns = [
        "defect_ratio",
        "component_count_8",
        "largest_component_fraction",
        "mean_normalized_radius",
        "center_defect_fraction",
        "edge_defect_fraction",
        "boundary_defect_coverage",
        "defect_bbox_fraction",
        "defect_centroid_offset",
        "radial_entropy_4bin",
    ]
    summary_metrics = {
        column: {
            "mean": float(records[column].mean()),
            "median": float(records[column].median()),
            "min": float(records[column].min()),
            "max": float(records[column].max()),
        }
        for column in metric_columns
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "purpose": "post_selection_near_full_global_shape_diagnostic",
        "used_for_model_selection": False,
        "thresholds_retuned": False,
        "deployment_model_changed": False,
        "split": "test",
        "class_name": "Near-full",
        "class_id": NEAR_FULL_LABEL_ID,
        "sample_count": len(records),
        "connectivity": "8-neighbor",
        "center_band": "normalized radius <= 0.35",
        "edge_band": "normalized radius >= 0.70",
        "metrics": summary_metrics,
        "interpretation_rule": (
            "Use area, connectedness, radial distribution, and boundary coverage "
            "together; no single descriptor is a causal process diagnosis."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_dir / "near_full_global_shape_samples.csv", index=False)
    (output_dir / "near_full_global_shape_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].scatter(
        records["defect_ratio"], records["defect_bbox_fraction"], color="#4c78a8"
    )
    axes[0].set(
        xlabel="Defect ratio",
        ylabel="Defect bounding-box fraction",
        title="Area and global spread",
    )
    axes[1].scatter(
        records["component_count_8"],
        records["largest_component_fraction"],
        color="#f28e2b",
    )
    axes[1].set(
        xlabel="8-connected components",
        ylabel="Largest-component fraction",
        title="Connectedness",
    )
    axes[2].scatter(
        records["mean_normalized_radius"],
        records["boundary_defect_coverage"],
        color="#59a14f",
    )
    axes[2].set(
        xlabel="Mean normalized radius",
        ylabel="Boundary defect coverage",
        title="Radial and boundary distribution",
    )
    fig.suptitle("Near-full fixed-test global shape descriptors (n=21)")
    fig.tight_layout()
    fig.savefig(output_dir / "near_full_global_shape_dashboard.png", dpi=170)
    plt.close(fig)

    readme = [
        "# WM-811K Near-full 전역 형태 진단",
        "",
        "- 고정 test의 Near-full 21개를 모두 사용합니다.",
        "- 결함 면적, 8-이웃 연결성, 중심/가장자리 분포, 웨이퍼 경계 점유를 계산합니다.",
        "- 모델·임계값 선택이나 재조정에는 사용하지 않습니다.",
        "- 각 지표는 형상 설명이며 물리적 공정 원인이나 인과관계를 증명하지 않습니다.",
    ]
    (output_dir / "README.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            analyze(
                arguments.maps,
                arguments.labels,
                arguments.assignments,
                arguments.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
