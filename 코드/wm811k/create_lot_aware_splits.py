"""Create reproducible leakage-free WM-811K train/validation/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_COLUMNS = {
    "array_index",
    "source_row",
    "failure_type",
    "label_id",
    "source_split",
    "lot_name",
}
SPLIT_ORDER = ("train", "validation", "test")


def _safe_zip_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def load_metadata(path: Path) -> pd.DataFrame:
    """Load labeled metadata from a CSV or preprocessing receipt ZIP."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            unsafe = [name for name in archive.namelist() if not _safe_zip_member(name)]
            if unsafe:
                raise ValueError(f"Unsafe ZIP member paths: {unsafe}")
            if "labeled_metadata.csv" not in archive.namelist():
                raise ValueError("ZIP does not contain labeled_metadata.csv")
            with archive.open("labeled_metadata.csv") as stream:
                frame = pd.read_csv(stream)
    else:
        frame = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Metadata is empty")
    if frame["array_index"].duplicated().any():
        raise ValueError("array_index must be unique")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required metadata columns contain missing values")

    frame = frame.copy()
    frame["array_index"] = frame["array_index"].astype(np.int64)
    frame["source_row"] = frame["source_row"].astype(np.int64)
    frame["label_id"] = frame["label_id"].astype(np.int64)
    frame["failure_type"] = frame["failure_type"].astype(str)
    frame["lot_name"] = frame["lot_name"].astype(str)
    frame["source_split"] = frame["source_split"].astype(str)
    frame = frame.sort_values("array_index").reset_index(drop=True)
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["array_index"].to_numpy(), expected):
        raise ValueError("array_index must be contiguous and aligned with the .npy arrays")
    return frame


def _fold_score(
    labels: np.ndarray,
    candidate_indices: np.ndarray,
    target_fraction: float,
) -> tuple[float, float, float]:
    classes = np.unique(labels)
    global_counts = np.bincount(labels, minlength=int(classes.max()) + 1)
    candidate_counts = np.bincount(
        labels[candidate_indices], minlength=len(global_counts)
    )
    if np.any(candidate_counts[classes] == 0):
        return (float("inf"), float("inf"), float("inf"))
    global_rate = global_counts[classes] / global_counts[classes].sum()
    candidate_rate = candidate_counts[classes] / candidate_counts[classes].sum()
    prevalence_error = float(np.abs(candidate_rate - global_rate).mean())
    size_error = abs(len(candidate_indices) / len(labels) - target_fraction)
    maximum_class_error = float(np.abs(candidate_rate - global_rate).max())
    return (prevalence_error + size_error, maximum_class_error, size_error)


def _best_validation_fold(
    frame: pd.DataFrame,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, object]]:
    labels = frame["label_id"].to_numpy()
    groups = frame["lot_name"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    candidates: list[tuple[tuple[float, float, float], int, np.ndarray]] = []
    placeholder = np.zeros(len(frame), dtype=np.uint8)
    for fold_id, (_, holdout_indices) in enumerate(
        splitter.split(placeholder, labels, groups)
    ):
        score = _fold_score(labels, holdout_indices, 1.0 / n_splits)
        candidates.append((score, fold_id, holdout_indices))
    score, fold_id, indices = min(candidates, key=lambda item: item[0])
    return indices, {
        "selected_fold": fold_id,
        "score": score[0],
        "maximum_class_prevalence_error": score[1],
        "size_error": score[2],
    }


def create_splits(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
    outer_splits: int = 7,
    inner_splits: int = 6,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assign rows to approximately 71/14/14 splits without lot overlap."""
    if outer_splits < 3 or inner_splits < 2:
        raise ValueError("outer_splits >= 3 and inner_splits >= 2 are required")

    test_relative, test_selection = _best_validation_fold(
        frame, outer_splits, random_state
    )
    test_mask = np.zeros(len(frame), dtype=bool)
    test_mask[test_relative] = True
    remaining = frame.loc[~test_mask].copy()

    validation_relative, validation_selection = _best_validation_fold(
        remaining, inner_splits, random_state + 1
    )
    validation_indices = remaining.index.to_numpy()[validation_relative]

    split = np.full(len(frame), "train", dtype=object)
    split[test_relative] = "test"
    split[validation_indices] = "validation"
    result = frame.copy()
    result["split"] = split

    lot_sets = {
        name: set(result.loc[result["split"] == name, "lot_name"])
        for name in SPLIT_ORDER
    }
    overlaps = {
        "train_validation": len(lot_sets["train"] & lot_sets["validation"]),
        "train_test": len(lot_sets["train"] & lot_sets["test"]),
        "validation_test": len(lot_sets["validation"] & lot_sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Lot leakage detected: {overlaps}")

    all_classes = set(result["label_id"].unique())
    for name in SPLIT_ORDER:
        present = set(result.loc[result["split"] == name, "label_id"].unique())
        if present != all_classes:
            raise AssertionError(f"{name} is missing label IDs {sorted(all_classes - present)}")

    summary: dict[str, object] = {
        "random_state": random_state,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "total_rows": len(result),
        "total_lots": int(result["lot_name"].nunique()),
        "class_count": int(result["label_id"].nunique()),
        "lot_overlap": overlaps,
        "test_fold_selection": test_selection,
        "validation_fold_selection": validation_selection,
        "splits": {},
    }
    split_summary = summary["splits"]
    assert isinstance(split_summary, dict)
    for name in SPLIT_ORDER:
        subset = result.loc[result["split"] == name]
        split_summary[name] = {
            "rows": len(subset),
            "fraction": len(subset) / len(result),
            "lots": int(subset["lot_name"].nunique()),
            "class_counts": {
                label: int(count)
                for label, count in subset["failure_type"].value_counts().items()
            },
        }
    return result, summary


def save_results(
    assignments: pd.DataFrame,
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "split_assignments.csv", index=False)

    rows: list[dict[str, object]] = []
    for split_name in SPLIT_ORDER:
        subset = assignments.loc[assignments["split"] == split_name]
        counts = subset.groupby(["label_id", "failure_type"]).size()
        for (label_id, failure_type), count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "label_id": int(label_id),
                    "failure_type": failure_type,
                    "count": int(count),
                    "percent_within_split": 100 * count / len(subset),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "split_class_distribution.csv", index=False
    )
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outer-splits", type=int, default=7)
    parser.add_argument("--inner-splits", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_metadata(args.metadata)
    assignments, summary = create_splits(
        frame,
        random_state=args.random_state,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
    )
    save_results(assignments, summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
