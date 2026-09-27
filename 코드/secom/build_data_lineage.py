"""Build and verify a reproducible SECOM raw-to-split data lineage manifest."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_lineage import (
    build_file_inventory,
    canonical_hash,
    simulated_tamper_is_detected,
    validate_lineage_manifest,
)
from preprocess_secom import (
    DATA_DIR,
    OUTPUT_DIR,
    load_data,
    remove_unusable_features,
    split_and_impute,
)


PROJECT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT / "결과물" / "secom" / "data_lineage"


def _read_feature_split(name: str) -> pd.DataFrame:
    frame = pd.read_csv(OUTPUT_DIR / f"{name}.csv", index_col="row_id")
    frame.index = frame.index.astype(int)
    return frame


def _read_label_split(name: str) -> pd.Series:
    frame = pd.read_csv(OUTPUT_DIR / f"{name}.csv", index_col="row_id")
    frame.index = frame.index.astype(int)
    return frame.iloc[:, 0].astype("int8").rename("label")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] 원본에서 구조 필터·고정 분할·median 대체를 다시 재현합니다...")
    X_raw, y = load_data(DATA_DIR)
    X_reduced, high_missing, constant = remove_unusable_features(X_raw)
    X_train, X_test, y_train, y_test, imputer = split_and_impute(X_reduced, y)

    print("[2/4] 저장된 전처리 산출물과 행·열·값·라벨을 대조합니다...")
    stored_X_train = _read_feature_split("X_train")
    stored_X_test = _read_feature_split("X_test")
    stored_y_train = _read_label_split("y_train")
    stored_y_test = _read_label_split("y_test")
    stored_imputer = joblib.load(OUTPUT_DIR / "median_imputer.joblib")
    train_index_match = X_train.index.tolist() == stored_X_train.index.tolist()
    test_index_match = X_test.index.tolist() == stored_X_test.index.tolist()
    feature_order_match = (
        X_train.columns.tolist()
        == stored_X_train.columns.tolist()
        == X_test.columns.tolist()
        == stored_X_test.columns.tolist()
    )
    train_values_match = bool(
        np.allclose(X_train.to_numpy(), stored_X_train.to_numpy(), rtol=0, atol=1e-10)
    )
    test_values_match = bool(
        np.allclose(X_test.to_numpy(), stored_X_test.to_numpy(), rtol=0, atol=1e-10)
    )
    train_labels_match = y_train.equals(stored_y_train)
    test_labels_match = y_test.equals(stored_y_test)
    median_match = bool(
        np.allclose(
            np.asarray(imputer.statistics_, dtype=float),
            np.asarray(stored_imputer.statistics_, dtype=float),
            rtol=0,
            atol=1e-12,
        )
    )
    no_split_overlap = set(X_train.index).isdisjoint(set(X_test.index))
    full_row_coverage = set(X_train.index) | set(X_test.index) == set(X_raw.index)

    print("[3/4] 원본·처리 파일과 분할 fingerprint를 연결합니다...")
    inventory = build_file_inventory(
        PROJECT,
        [
            ("raw", "데이터/SECOM 데이터셋/raw/secom.data"),
            ("raw", "데이터/SECOM 데이터셋/raw/secom_labels.data"),
            ("raw_documentation", "데이터/SECOM 데이터셋/raw/secom.names"),
            ("processed", "데이터/SECOM 데이터셋/processed/X_train.csv"),
            ("processed", "데이터/SECOM 데이터셋/processed/X_test.csv"),
            ("processed", "데이터/SECOM 데이터셋/processed/y_train.csv"),
            ("processed", "데이터/SECOM 데이터셋/processed/y_test.csv"),
            ("preprocessor", "데이터/SECOM 데이터셋/processed/median_imputer.joblib"),
            ("metadata", "데이터/SECOM 데이터셋/processed/preprocessing_metadata.json"),
        ],
    )
    manifest = {
        "manifest_version": 2,
        "dataset": "UCI SECOM",
        "raw_sensor_values_embedded": False,
        "text_hash_normalization": "UTF-8-with-canonical-LF",
        "files": inventory,
        "fingerprints": {
            "raw_label_vector_sha256": canonical_hash(y.astype(int).tolist()),
            "retained_feature_schema_sha256": canonical_hash(X_reduced.columns.tolist()),
            "train_row_ids_sha256": canonical_hash(X_train.index.astype(int).tolist()),
            "test_row_ids_sha256": canonical_hash(X_test.index.astype(int).tolist()),
            "train_median_statistics_sha256": canonical_hash(
                [float(value) for value in imputer.statistics_]
            ),
        },
        "split_contract": {
            "method": "train_test_split_stratified",
            "random_state": 42,
            "test_size": 0.2,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
    }
    manifest_validation = validate_lineage_manifest(manifest, PROJECT)
    raw_data_entry = next(
        item for item in inventory if item["artifact_path"].endswith("secom.data")
    )
    raw_label_entry = next(
        item for item in inventory if item["artifact_path"].endswith("secom_labels.data")
    )
    data_tamper_detected = simulated_tamper_is_detected(
        PROJECT / raw_data_entry["artifact_path"], raw_data_entry["sha256"]
    )
    label_tamper_detected = simulated_tamper_is_detected(
        PROJECT / raw_label_entry["artifact_path"], raw_label_entry["sha256"]
    )

    checks = {
        "manifest_hashes_valid": manifest_validation["valid"],
        "train_row_order_reproduced": train_index_match,
        "test_row_order_reproduced": test_index_match,
        "feature_order_reproduced": feature_order_match,
        "train_values_reproduced": train_values_match,
        "test_values_reproduced": test_values_match,
        "train_labels_reproduced": train_labels_match,
        "test_labels_reproduced": test_labels_match,
        "train_fitted_medians_reproduced": median_match,
        "train_test_rows_disjoint": no_split_overlap,
        "all_raw_rows_covered_once": full_row_coverage,
        "raw_data_tamper_detected": data_tamper_detected,
        "raw_label_tamper_detected": label_tamper_detected,
    }
    status = "PASS" if all(checks.values()) else "BLOCK"
    validation = {
        "version": 1,
        "status": status,
        "validation_scope": "raw_to_saved_split_reconstruction",
        "raw_rows": len(X_raw),
        "raw_features": X_raw.shape[1],
        "retained_features": X_reduced.shape[1],
        "removed_high_missing_features": len(high_missing),
        "removed_constant_features": len(constant),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_class_counts": {
            str(key): int(value) for key, value in y_train.value_counts().sort_index().items()
        },
        "test_class_counts": {
            str(key): int(value) for key, value in y_test.value_counts().sort_index().items()
        },
        "checks": checks,
        "real_files_modified": False,
        "limitations": [
            "SHA-256은 파일 변경을 탐지하지만 데이터 수집 당시 측정의 진실성을 증명하지 않습니다.",
            "공개 SECOM 데이터에는 장비·lot 식별자와 센서 의미·단위가 없습니다.",
            "모의 오염은 메모리상 바이트 추가이며 실제 공급망 침해 훈련을 대체하지 않습니다.",
        ],
    }

    print("[4/4] 계보 manifest와 검증 보고서를 저장합니다...")
    (RESULT_DIR / "data_lineage_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "lineage_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(inventory).to_csv(
        RESULT_DIR / "file_inventory.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [{"check": key, "passed": value} for key, value in checks.items()]
    ).to_csv(RESULT_DIR / "reconstruction_checks.csv", index=False, encoding="utf-8-sig")
    (RESULT_DIR / "summary.md").write_text(
        "# SECOM 데이터 계보·오염 탐지 검증\n\n"
        f"- 판정: **{status}**\n"
        f"- 원본: {len(X_raw)}행 × {X_raw.shape[1]}변수\n"
        f"- 구조 필터 후: {X_reduced.shape[1]}변수\n"
        f"- 고정 train/test: {len(X_train)}/{len(X_test)}행\n"
        f"- 저장 분할 행·열·값·라벨·median 재현: {all(checks[key] for key in checks if 'tamper' not in key)}\n"
        f"- 원본 센서 파일 모의 오염 탐지: {data_tamper_detected}\n"
        f"- 라벨 파일 모의 오염 탐지: {label_tamper_detected}\n"
        "- Manifest 센서 원본값 포함: 아니오\n\n"
        "이 검증은 원본 파일부터 저장 전처리 분할까지의 재현성과 변경 탐지를 보여줍니다. "
        "데이터 수집 당시 측정값의 진실성, 익명 센서의 물리적 의미 또는 신규 공정 대표성을 "
        "증명하지는 않습니다.\n",
        encoding="utf-8",
    )
    if status != "PASS":
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"데이터 계보 검증 실패: {failed}")
    print(f"데이터 계보 검증 완료: status={status}, files={len(inventory)}")


if __name__ == "__main__":
    main()
