"""Extract leakage-safe unlabeled WM-811K maps for self-supervised learning.

Only unlabeled wafers whose lotName belongs to the already frozen labeled
training split are eligible. Wafers from validation/test lots and wafers whose
lot cannot be tied to the training split are excluded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from preprocess_wm811k import (
    DEFAULT_MINIMUM_FREE_GB,
    available_memory_gb,
    resize_wafer_map,
    unwrap_scalar,
    validate_schema,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_assignments(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"lot_name", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Split assignments are missing columns: {sorted(missing)}")
    if not {"train", "validation", "test"}.issubset(set(frame["split"])):
        raise ValueError("Assignments must contain train, validation, and test rows")
    frame = frame.copy()
    frame["lot_name"] = frame["lot_name"].astype(str)
    lot_sets = {
        name: set(frame.loc[frame["split"] == name, "lot_name"])
        for name in ("train", "validation", "test")
    }
    if (
        lot_sets["train"] & lot_sets["validation"]
        or lot_sets["train"] & lot_sets["test"]
        or lot_sets["validation"] & lot_sets["test"]
    ):
        raise ValueError("Frozen assignments contain lot leakage")
    return frame


def select_train_lot_unlabeled_indices(
    source: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    maximum_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return deterministic source indices eligible for SSL pretraining."""
    if maximum_samples <= 0:
        raise ValueError("maximum_samples must be positive")
    validate_schema(source)
    train_lots = set(
        assignments.loc[assignments["split"] == "train", "lot_name"].astype(str)
    )
    validation_lots = set(
        assignments.loc[assignments["split"] == "validation", "lot_name"].astype(str)
    )
    test_lots = set(
        assignments.loc[assignments["split"] == "test", "lot_name"].astype(str)
    )

    labels = source["failureType"].map(unwrap_scalar)
    lots = source["lotName"].map(unwrap_scalar)
    lot_text = lots.fillna("").astype(str)
    unlabeled = labels.isna()
    in_train_lot = lot_text.isin(train_lots)
    eligible = np.flatnonzero((unlabeled & in_train_lot).to_numpy())

    rng = np.random.default_rng(seed)
    if len(eligible) > maximum_samples:
        selected = np.sort(
            rng.choice(eligible, size=maximum_samples, replace=False).astype(np.int64)
        )
    else:
        selected = eligible.astype(np.int64)

    selected_lots = set(lot_text.iloc[selected])
    leaked_validation = selected_lots & validation_lots
    leaked_test = selected_lots & test_lots
    if leaked_validation or leaked_test:
        raise AssertionError(
            "Held-out lot leakage detected: "
            f"validation={sorted(leaked_validation)}, test={sorted(leaked_test)}"
        )

    summary: dict[str, object] = {
        "source_rows": int(len(source)),
        "unlabeled_rows": int(unlabeled.sum()),
        "train_lot_count": int(len(train_lots)),
        "validation_lot_count": int(len(validation_lots)),
        "test_lot_count": int(len(test_lots)),
        "eligible_unlabeled_train_lot_rows": int(len(eligible)),
        "selected_rows_before_map_validation": int(len(selected)),
        "maximum_samples": int(maximum_samples),
        "selection_seed": int(seed),
        "held_out_lot_overlap": {
            "validation": int(len(leaked_validation)),
            "test": int(len(leaked_test)),
        },
        "unknown_or_non_train_lot_unlabeled_excluded": int(
            (unlabeled & ~in_train_lot).sum()
        ),
    }
    return selected, summary


def convert_selected_maps(
    source: pd.DataFrame,
    selected_indices: np.ndarray,
    output_dir: Path,
    *,
    image_size: int,
) -> tuple[pd.DataFrame, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"unlabeled_train_lot_maps_{image_size}.partial.npy"
    final_path = output_dir / f"unlabeled_train_lot_maps_{image_size}.npy"
    maps = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(selected_indices), image_size, image_size),
    )
    rows: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    valid_position = 0
    for source_index in selected_indices:
        row = source.iloc[int(source_index)]
        try:
            maps[valid_position] = resize_wafer_map(row["waferMap"], image_size)
        except ValueError as error:
            invalid.append({"source_row": int(source_index), "reason": str(error)})
            continue
        original = np.asarray(row["waferMap"])
        rows.append(
            {
                "array_index": valid_position,
                "source_row": int(source_index),
                "lot_name": unwrap_scalar(row["lotName"]),
                "wafer_index": unwrap_scalar(row["waferIndex"]),
                "original_height": int(original.shape[0]),
                "original_width": int(original.shape[1]),
                "eligibility": "unlabeled_train_lot_only",
            }
        )
        valid_position += 1
        if valid_position % 10_000 == 0:
            maps.flush()
            print(f"  converted={valid_position:,}/{len(selected_indices):,}")

    maps.flush()
    del maps
    if valid_position == 0:
        partial_path.unlink(missing_ok=True)
        raise ValueError("No selected wafer map was valid")

    if valid_position < len(selected_indices):
        source_maps = np.load(partial_path, mmap_mode="r")
        compact_path = output_dir / f"unlabeled_train_lot_maps_{image_size}.compact.npy"
        compact = np.lib.format.open_memmap(
            compact_path,
            mode="w+",
            dtype=np.uint8,
            shape=(valid_position, image_size, image_size),
        )
        compact[:] = source_maps[:valid_position]
        compact.flush()
        del source_maps, compact
        partial_path.unlink()
        compact_path.replace(final_path)
    else:
        partial_path.replace(final_path)

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "unlabeled_train_lot_metadata.csv", index=False)
    if invalid:
        pd.DataFrame(invalid).to_csv(output_dir / "invalid_unlabeled_rows.csv", index=False)
    return metadata, len(invalid)


def extract(args: argparse.Namespace) -> dict[str, object]:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.assignments.is_file():
        raise FileNotFoundError(args.assignments)
    if args.image_size <= 0:
        raise ValueError("image-size must be positive")
    free_gb = available_memory_gb()
    print(f"available_memory={free_gb:.2f} GB")
    if not args.skip_memory_check and free_gb < args.minimum_free_gb:
        raise MemoryError(
            f"Available RAM {free_gb:.2f} GB is below {args.minimum_free_gb:.2f} GB"
        )

    started = time.perf_counter()
    assignments = load_frozen_assignments(args.assignments)
    print("Loading the full LSWMD pickle once...")
    source = pd.read_pickle(args.input)
    selected, summary = select_train_lot_unlabeled_indices(
        source,
        assignments,
        maximum_samples=args.maximum_samples,
        seed=args.seed,
    )
    metadata, invalid_count = convert_selected_maps(
        source, selected, args.output_dir, image_size=args.image_size
    )
    del source
    gc.collect()

    selected_lots = set(metadata["lot_name"].astype(str))
    held_out_lots = set(
        assignments.loc[assignments["split"].isin(["validation", "test"]), "lot_name"]
        .astype(str)
    )
    if selected_lots & held_out_lots:
        raise AssertionError("Converted metadata contains held-out lots")

    summary.update(
        {
            "schema_version": 1,
            "purpose": "self_supervised_pretraining_only",
            "eligibility_rule": "unlabeled AND lot_name in frozen labeled train lots",
            "validation_or_test_lots_used": False,
            "labels_used": False,
            "valid_selected_rows": int(len(metadata)),
            "invalid_selected_rows": int(invalid_count),
            "selected_lot_count": int(metadata["lot_name"].nunique()),
            "image_size": int(args.image_size),
            "map_dtype": "uint8",
            "source_pickle_sha256": sha256_file(args.input),
            "split_assignments_sha256": sha256_file(args.assignments),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "unlabeled_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-samples", type=int, default=100_000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-free-gb", type=float, default=DEFAULT_MINIMUM_FREE_GB)
    parser.add_argument("--skip-memory-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
