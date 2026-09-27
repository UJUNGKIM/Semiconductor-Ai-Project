"""Load and preprocess the UCI SECOM dataset.

The raw files are expected in the ``secom`` directory next to this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "데이터" / "SECOM 데이터셋" / "raw"
OUTPUT_DIR = PROJECT_DIR / "데이터" / "SECOM 데이터셋" / "processed"
RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPECTED_RAW_FEATURES = 590
EXPECTED_REDUCED_FEATURES = 446


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Load the space-delimited measurements and their binary labels."""
    data_path = data_dir / "secom.data"
    labels_path = data_dir / "secom_labels.data"

    missing_files = [path for path in (data_path, labels_path) if not path.is_file()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Required SECOM file(s) not found: {missing}")

    # The raw string avoids invalid escape-sequence SyntaxWarning messages.
    X = pd.read_csv(data_path, sep=r"\s+", header=None)
    labels = pd.read_csv(labels_path, sep=r"\s+", header=None)

    if len(X) != len(labels):
        raise ValueError(f"Row mismatch: data={len(X)}, labels={len(labels)}")
    if X.shape[1] != EXPECTED_RAW_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_RAW_FEATURES} raw features, got {X.shape[1]}"
        )

    X.columns = [f"feature_{column}" for column in X.columns]
    y = labels.iloc[:, 0].map({-1: 0, 1: 1})
    if y.isna().any():
        unexpected = sorted(labels.loc[y.isna(), 0].unique().tolist())
        raise ValueError(f"Unexpected label values: {unexpected}")
    y = y.astype("int8").rename("label")
    return X, y


def remove_unusable_features(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Remove columns with >=50% missing values, then constant columns."""
    missing_ratio = X.isna().mean()
    high_missing_columns = missing_ratio[missing_ratio >= 0.5].index.tolist()
    reduced = X.drop(columns=high_missing_columns)

    constant_mask = reduced.nunique(dropna=True) <= 1
    constant_columns = constant_mask[constant_mask].index.tolist()
    reduced = reduced.drop(columns=constant_columns)

    if reduced.shape[1] != EXPECTED_REDUCED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_REDUCED_FEATURES} retained features, "
            f"got {reduced.shape[1]}"
        )

    return reduced, high_missing_columns, constant_columns


def split_and_impute(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, SimpleImputer]:
    """Create a stratified split and apply train-fitted median imputation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Fit only on train data so test-set information cannot leak into medians.
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X.columns,
        index=X_test.index,
    )

    return X_train_imputed, X_test_imputed, y_train, y_test, imputer


def save_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    imputer: SimpleImputer,
    high_missing_columns: list[str],
    constant_columns: list[str],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Save processed splits, the fitted imputer, and preprocessing metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=True, index_label="row_id")
    X_test.to_csv(output_dir / "X_test.csv", index=True, index_label="row_id")
    y_train.to_csv(output_dir / "y_train.csv", index=True, index_label="row_id")
    y_test.to_csv(output_dir / "y_test.csv", index=True, index_label="row_id")
    joblib.dump(imputer, output_dir / "median_imputer.joblib")

    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "label_mapping": {"-1": 0, "1": 1},
        "raw_feature_count": EXPECTED_RAW_FEATURES,
        "high_missing_feature_count": len(high_missing_columns),
        "constant_feature_count": len(constant_columns),
        "retained_feature_count": X_train.shape[1],
        "high_missing_columns": high_missing_columns,
        "constant_columns": constant_columns,
    }
    (output_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    X, y = load_data()
    reduced, high_missing_columns, constant_columns = remove_unusable_features(X)
    X_train, X_test, y_train, y_test, imputer = split_and_impute(reduced, y)
    save_outputs(
        X_train,
        X_test,
        y_train,
        y_test,
        imputer,
        high_missing_columns,
        constant_columns,
    )

    print(f"Raw data: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Removed for >=50% missing: {len(high_missing_columns)}")
    print(f"Removed constants: {len(constant_columns)}")
    print(f"Retained features: {reduced.shape[1]}")
    print(f"Train/test rows: {len(X_train)}/{len(X_test)}")
    print(f"Train labels: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Test labels: {y_test.value_counts().sort_index().to_dict()}")
    print(
        "Remaining missing values: "
        f"train={int(X_train.isna().sum().sum())}, "
        f"test={int(X_test.isna().sum().sum())}"
    )
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
