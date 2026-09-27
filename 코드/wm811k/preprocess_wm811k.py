"""Convert the large WM-811K pickle into lazy-loadable fixed-size arrays.

The source pickle must be fully deserialized once. A memory preflight prevents
accidental loading when available RAM is too low. Only labeled wafer maps are
resized, and the output ``.npy`` array can later be memory-mapped for training.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image


REQUIRED_COLUMNS = {
    "waferMap",
    "dieSize",
    "lotName",
    "waferIndex",
    "trianTestLabel",
    "failureType",
}
DEFAULT_MINIMUM_FREE_GB = 8.0
VALID_MAP_VALUES = {0, 1, 2}


def available_memory_gb() -> float:
    """Return currently available physical memory without extra dependencies."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Could not query Windows memory status")
        return status.available_physical / 1024**3

    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.is_file():
        for line in meminfo_path.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])
                return available_kb / 1024**2

    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return page_size * available_pages / 1024**3


def unwrap_scalar(value: object) -> str | None:
    """Extract one string from the nested arrays used by the LSWMD pickle."""
    if value is None:
        return None
    array = np.asarray(value, dtype=object)
    if array.size == 0:
        return None
    text = str(array.reshape(-1)[0]).strip()
    if not text or text.lower() in {"nan", "null"}:
        return None
    return text


def resize_wafer_map(wafer_map: object, image_size: int) -> np.ndarray:
    """Resize categorical die states with nearest-neighbour interpolation."""
    array = np.asarray(wafer_map)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"Invalid wafer map shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Wafer map contains non-finite values")
    unique_values = set(np.unique(array).astype(int).tolist())
    if not unique_values.issubset(VALID_MAP_VALUES):
        raise ValueError(f"Unexpected wafer-map values: {sorted(unique_values)}")
    image = Image.fromarray(array.astype(np.uint8))
    resized = image.resize((image_size, image_size), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def validate_schema(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"WM-811K required columns are missing: {missing}")


def convert_dataframe(
    frame: pd.DataFrame,
    output_dir: Path,
    image_size: int = 64,
) -> dict[str, object]:
    """Convert a loaded LSWMD DataFrame into arrays and compact metadata."""
    validate_schema(frame)
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_labels = frame["failureType"].map(unwrap_scalar)
    labeled_mask = failure_labels.notna()
    labeled_indices = np.flatnonzero(labeled_mask.to_numpy())
    if len(labeled_indices) == 0:
        raise ValueError("No labeled wafer maps were found")

    labeled_names = failure_labels.iloc[labeled_indices].astype(str)
    class_names = sorted(labeled_names.unique().tolist())
    label_to_id = {name: index for index, name in enumerate(class_names)}
    label_ids = labeled_names.map(label_to_id).to_numpy(dtype=np.uint8)

    partial_path = output_dir / f"wafer_maps_{image_size}.partial.npy"
    final_path = output_dir / f"wafer_maps_{image_size}.npy"
    maps = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(labeled_indices), image_size, image_size),
    )

    metadata_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    valid_position = 0
    for detected_position, source_index in enumerate(labeled_indices):
        row = frame.iloc[int(source_index)]
        try:
            maps[valid_position] = resize_wafer_map(row["waferMap"], image_size)
        except ValueError as error:
            invalid_rows.append(
                {"source_row": int(source_index), "reason": str(error)}
            )
            continue

        wafer_array = np.asarray(row["waferMap"])
        metadata_rows.append(
            {
                "array_index": valid_position,
                "source_row": int(source_index),
                "failure_type": labeled_names.iloc[detected_position],
                "label_id": int(label_ids[detected_position]),
                "source_split": unwrap_scalar(row["trianTestLabel"]),
                "lot_name": unwrap_scalar(row["lotName"]),
                "wafer_index": unwrap_scalar(row["waferIndex"]),
                "original_height": int(wafer_array.shape[0]),
                "original_width": int(wafer_array.shape[1]),
            }
        )
        valid_position += 1
        if valid_position % 10_000 == 0:
            maps.flush()
            print(f"  converted={valid_position:,}/{len(labeled_indices):,}")

    maps.flush()
    del maps
    if valid_position == 0:
        partial_path.unlink(missing_ok=True)
        raise ValueError("All labeled wafer maps were invalid")

    if valid_position < len(labeled_indices):
        source_maps = np.load(partial_path, mmap_mode="r")
        compact_path = output_dir / f"wafer_maps_{image_size}.compact.npy"
        compact_maps = np.lib.format.open_memmap(
            compact_path,
            mode="w+",
            dtype=np.uint8,
            shape=(valid_position, image_size, image_size),
        )
        compact_maps[:] = source_maps[:valid_position]
        compact_maps.flush()
        del source_maps, compact_maps
        partial_path.unlink()
        compact_path.replace(final_path)
    else:
        partial_path.replace(final_path)

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(output_dir / "labeled_metadata.csv", index=False)
    np.save(output_dir / "labels.npy", metadata["label_id"].to_numpy(dtype=np.uint8))
    distribution = (
        metadata.groupby(["label_id", "failure_type"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )
    distribution.to_csv(output_dir / "class_distribution.csv", index=False)
    if invalid_rows:
        pd.DataFrame(invalid_rows).to_csv(output_dir / "invalid_rows.csv", index=False)

    summary: dict[str, object] = {
        "source_rows": len(frame),
        "labeled_rows_detected": len(labeled_indices),
        "valid_labeled_rows": valid_position,
        "invalid_labeled_rows": len(invalid_rows),
        "image_size": image_size,
        "map_dtype": "uint8",
        "map_values": sorted(VALID_MAP_VALUES),
        "class_count": len(class_names),
        "label_mapping": label_to_id,
        "source_split_column": "trianTestLabel",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_to_id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--minimum-free-gb", type=float, default=DEFAULT_MINIMUM_FREE_GB
    )
    parser.add_argument(
        "--skip-memory-check",
        action="store_true",
        help="Skip RAM preflight only when the runtime memory is already known.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.image_size <= 0:
        raise ValueError("image-size must be positive")

    free_gb = available_memory_gb()
    source_gb = args.input.stat().st_size / 1024**3
    print(f"source={args.input}")
    print(f"source_size={source_gb:.3f} GB")
    print(f"available_memory={free_gb:.2f} GB")
    if not args.skip_memory_check and free_gb < args.minimum_free_gb:
        raise MemoryError(
            f"Available RAM {free_gb:.2f} GB is below the safety threshold "
            f"{args.minimum_free_gb:.2f} GB. Use a Colab high-RAM runtime or "
            "free memory before loading this pickle."
        )

    started = time.perf_counter()
    print("Loading the full pickle once...")
    frame = pd.read_pickle(args.input)
    print(f"loaded_shape={frame.shape}")
    summary = convert_dataframe(frame, args.output_dir, args.image_size)
    del frame
    gc.collect()
    summary["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
