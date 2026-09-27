"""Export selected WM-811K demo wafers from local preprocessed arrays.

This is the local equivalent of the export step in
``노트북/WM811K_04_실제샘플_추출_Colab.ipynb``. Every wafer is copied from
``wafer_maps_64.npy`` by its ``array_index``; nothing is synthesized. Existing
sample files are never overwritten with different bytes, so a previously
deployed sample either stays identical or the export stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from hash_evidence import text_hashes


PROJECT_DIR = Path(__file__).resolve().parents[2]
WM_DIR = PROJECT_DIR / "결과물" / "wm811k"
ALLOWED_VALUES = {0, 1, 2}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wafer_sha256(wafer: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(wafer, dtype=np.uint8).tobytes()).hexdigest()


def export_samples(args: argparse.Namespace) -> dict[str, object]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("source_split") != "test" or manifest.get("used_for_model_selection"):
        raise ValueError("모델 선택 후 test에서 고정한 manifest만 내보낼 수 있습니다.")
    maps = np.load(args.maps, mmap_mode="r", allow_pickle=False)
    labels = np.load(args.labels, mmap_mode="r", allow_pickle=False)
    if maps.ndim != 3 or maps.shape[1:] != (64, 64) or maps.dtype != np.uint8:
        raise ValueError(f"전처리 배열 규격이 다릅니다: {maps.shape}, {maps.dtype}")
    if len(maps) != len(labels):
        raise ValueError("웨이퍼 배열과 라벨 배열의 행 수가 다릅니다.")

    previous = {}
    previous_manifest_path = args.output_dir / "manifest.json"
    if previous_manifest_path.is_file():
        previous = {
            item["sample_id"]: item
            for item in json.loads(previous_manifest_path.read_text(encoding="utf-8"))[
                "samples"
            ]
        }

    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for item in manifest["samples"]:
        index = int(item["array_index"])
        if not 0 <= index < len(maps):
            raise ValueError(f"배열 인덱스 범위 오류: {index}")
        wafer = np.asarray(maps[index], dtype=np.uint8)
        if wafer.shape != (64, 64) or not set(np.unique(wafer)).issubset(ALLOWED_VALUES):
            raise ValueError(f"웨이퍼 규격 오류: {item['sample_id']}")
        if int(labels[index]) != int(item["true_label_id"]):
            raise ValueError(f"라벨 불일치: {item['sample_id']}")
        digest = wafer_sha256(wafer)
        if item["sample_id"] in previous and previous[item["sample_id"]]["wafer_sha256"] != digest:
            raise ValueError(f"기존 배포 샘플과 웨이퍼가 다릅니다: {item['sample_id']}")
        npy_name = f"{item['sample_id']}.npy"
        target = samples_dir / npy_name
        if target.is_file():
            existing = np.load(target, allow_pickle=False)
            if not np.array_equal(existing, wafer) or existing.dtype != np.uint8:
                raise ValueError(f"기존 샘플 파일을 다른 내용으로 덮어쓸 수 없습니다: {npy_name}")
        else:
            np.save(target, wafer, allow_pickle=False)
        exported_item = dict(item)
        exported_item["npy_file"] = f"samples/{npy_name}"
        exported_item["wafer_sha256"] = digest
        exported.append(exported_item)

    referenced = {Path(item["npy_file"]).name for item in exported}
    stale = sorted(path.name for path in samples_dir.glob("*.npy") if path.name not in referenced)
    if stale:
        raise ValueError(f"manifest에 없는 기존 샘플 파일이 있습니다: {stale}")
    missing_previous = sorted(set(previous) - {item["sample_id"] for item in exported})
    if missing_previous:
        raise ValueError(f"기존 배포 샘플이 새 manifest에서 빠졌습니다: {missing_previous}")

    export_manifest = dict(manifest)
    export_manifest["samples"] = exported
    export_manifest["exported_map_shape"] = [64, 64]
    export_manifest["allowed_values"] = sorted(ALLOWED_VALUES)
    export_manifest["export"] = {
        "tool": "코드/wm811k/export_wm811k_demo_samples.py",
        "source_arrays": "local wafer_maps_64.npy and labels.npy",
        "maps_sha256": file_sha256(args.maps),
        "labels_sha256": file_sha256(args.labels),
        "selection_manifest": {"file_name": args.manifest.name, **text_hashes(args.manifest)},
        "deployed_formats": ["npy"],
        "previous_samples_preserved": len(previous),
        "new_samples_added": len(exported) - len(previous),
    }
    previous_manifest_path.write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "sample_count": len(exported),
        "class_counts": dict(sorted(Counter(item["true_class"] for item in exported).items())),
        "previous_samples_preserved": len(previous),
        "new_samples_added": len(exported) - len(previous),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WM_DIR / "selected_model_results" / "demo_sample_manifest.json",
    )
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=WM_DIR / "demo_samples")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(export_samples(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
