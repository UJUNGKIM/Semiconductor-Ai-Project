"""독립 검증 WM-811K 공통 모듈 — 분할 로딩, lot 누출 검사, 지표.

기준선과 **같은 lot 비중복 분할**을 씁니다. 비교가 목적이므로 분할을 새로 만들면
숫자를 나란히 놓을 수 없습니다. 대신 그 분할에 실제로 lot 누출이 없는지
독립 검증가 독립적으로 다시 검사합니다.

평가 지표는 요구사항대로 macro-F1, balanced accuracy, 클래스별 recall입니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_DIR  # noqa: E402

WM_RESULT_DIR = PROJECT_DIR / "결과물" / "wm811k" / "independent_validation"
BASELINE_WM_DIR = PROJECT_DIR / "결과물" / "wm811k"
SPLIT_ASSIGNMENTS = BASELINE_WM_DIR / "split_results" / "split_assignments.csv"
DEMO_SAMPLE_DIR = BASELINE_WM_DIR / "demo_samples" / "samples"
DEMO_MANIFEST = BASELINE_WM_DIR / "demo_samples" / "demo_sample_manifest.json"

# Colab 전처리가 만든 64x64 uint8 배열이 있어야 픽셀 모델을 돌릴 수 있다.
# 저장소에는 없다(약 1.95GB 원본에서 파생). 아래 경로 중 하나에 두면 인식한다.
PIXEL_ARRAY_CANDIDATES = (
    PROJECT_DIR / "데이터" / "WM-811K(LSWMD) 데이터셋" / "wafer_maps_64.npy",
    PROJECT_DIR / "데이터" / "WM-811K(LSWMD) 데이터셋" / "processed" / "wafer_maps_64.npy",
)

SPLIT_ORDER = ("train", "validation", "test")
CLASS_ORDER = (
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
)
# 웨이퍼 맵 화소 값: 0=웨이퍼 밖, 1=정상 die, 2=불량 die
BACKGROUND, PASS_DIE, FAIL_DIE = 0, 1, 2


class MissingPixelData(FileNotFoundError):
    """64x64 웨이퍼 맵 배열이 없을 때."""


def load_split_metadata() -> pd.DataFrame:
    """기준선이 고정한 lot 비중복 분할 메타데이터를 읽는다(읽기 전용)."""
    if not SPLIT_ASSIGNMENTS.is_file():
        raise FileNotFoundError(f"분할 파일이 없습니다: {SPLIT_ASSIGNMENTS}")

    frame = pd.read_csv(SPLIT_ASSIGNMENTS)
    required = {
        "array_index",
        "failure_type",
        "label_id",
        "lot_name",
        "original_height",
        "original_width",
        "split",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"분할 파일에 열이 없습니다: {sorted(missing)}")

    frame["lot_name"] = frame["lot_name"].astype(str)
    frame["failure_type"] = frame["failure_type"].astype(str)
    frame["split"] = frame["split"].astype(str)
    return frame.sort_values("array_index").reset_index(drop=True)


def audit_lot_leakage(frame: pd.DataFrame) -> dict:
    """분할 사이에 같은 lot가 섞여 있는지 독립적으로 검사한다.

    lot 누출은 '같은 lot의 웨이퍼가 train과 test에 동시에 있는' 상태다.
    같은 lot는 같은 장비·시점을 공유하므로 성능이 실제보다 좋게 나온다.
    """
    lots = {name: set(group["lot_name"]) for name, group in frame.groupby("split")}
    overlaps = {}
    for i, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[i + 1 :]:
            shared = lots.get(left, set()) & lots.get(right, set())
            overlaps[f"{left}_{right}"] = {
                "shared_lot_count": len(shared),
                "examples": sorted(shared)[:5],
            }

    # array_index가 중복되면 같은 웨이퍼가 두 split에 들어간 것이다.
    duplicated = int(frame["array_index"].duplicated().sum())

    return {
        "total_rows": int(len(frame)),
        "total_lots": int(frame["lot_name"].nunique()),
        "duplicate_array_index": duplicated,
        "rows_per_split": {
            name: int(count) for name, count in frame["split"].value_counts().items()
        },
        "lots_per_split": {name: len(value) for name, value in lots.items()},
        "lot_overlap": overlaps,
        "leakage_free": all(
            item["shared_lot_count"] == 0 for item in overlaps.values()
        )
        and duplicated == 0,
    }


def class_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """split × 클래스 건수와 비율 표."""
    counts = (
        frame.groupby(["split", "failure_type"]).size().unstack(fill_value=0)
    )
    counts = counts.reindex(columns=[c for c in CLASS_ORDER if c in counts.columns])
    counts = counts.reindex(index=[s for s in SPLIT_ORDER if s in counts.index])
    ratios = counts.div(counts.sum(axis=1), axis=0)
    out = counts.copy()
    for column in counts.columns:
        out[f"{column}_비율"] = ratios[column].round(5)
    return out.reset_index()


def evaluate(y_true, y_pred, labels: list[str]) -> dict:
    """요구된 지표 세 가지를 한 번에 계산한다."""
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(report["accuracy"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class_recall": {
            name: float(value) for name, value in zip(labels, per_class_recall)
        },
        "per_class_f1": {
            name: float(report[name]["f1-score"]) for name in labels if name in report
        },
        "support": {
            name: int(report[name]["support"]) for name in labels if name in report
        },
    }


def confusion_frame(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=[f"실제_{l}" for l in labels], columns=[f"예측_{l}" for l in labels])


def find_pixel_array() -> Path:
    """64x64 웨이퍼 맵 배열을 찾는다. 없으면 어디에 두면 되는지 알려 준다."""
    for path in PIXEL_ARRAY_CANDIDATES:
        if path.is_file():
            return path
    locations = "\n".join(f"  - {p}" for p in PIXEL_ARRAY_CANDIDATES)
    raise MissingPixelData(
        "WM-811K 64x64 웨이퍼 맵 배열(wafer_maps_64.npy)을 찾지 못했습니다.\n"
        "원본 LSWMD.pkl(약 1.95GB)은 용량 때문에 저장소에 없습니다.\n"
        "노트북/WM811K_01_전처리_Colab.ipynb으로 만든 배열을 아래 중 한 곳에 두세요.\n"
        f"{locations}"
    )


def load_pixel_arrays(frame: pd.DataFrame) -> np.ndarray:
    """분할 메타데이터 순서에 맞는 웨이퍼 맵 배열을 메모리맵으로 연다."""
    path = find_pixel_array()
    array = np.load(path, mmap_mode="r")
    if array.ndim != 3 or array.shape[1:] != (64, 64):
        raise ValueError(f"배열 형상이 (N, 64, 64)가 아닙니다: {array.shape}")
    if len(array) <= int(frame["array_index"].max()):
        raise ValueError(
            f"배열 길이 {len(array)}가 array_index 최대값 "
            f"{int(frame['array_index'].max())}보다 작습니다."
        )
    return array


def load_demo_samples() -> tuple[np.ndarray, list[str]]:
    """저장소에 포함된 정성 예시(클래스별 4~5개)를 읽는다(파이프라인 점검용)."""
    if not DEMO_SAMPLE_DIR.is_dir():
        raise FileNotFoundError(f"데모 샘플 폴더가 없습니다: {DEMO_SAMPLE_DIR}")
    paths = sorted(DEMO_SAMPLE_DIR.glob("*.npy"))
    if not paths:
        raise FileNotFoundError("데모 샘플 .npy 파일이 없습니다.")
    maps = np.stack([np.load(path) for path in paths])
    names = [path.stem for path in paths]
    return maps, names



