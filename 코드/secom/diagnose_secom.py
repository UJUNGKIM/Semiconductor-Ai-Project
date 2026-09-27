"""Run both saved SECOM finalists and emit compact SHAP-based diagnostics.

Example:
    python 코드/secom/diagnose_secom.py --input "데이터/SECOM 데이터셋/raw/secom.data" --row-limit 5
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import shap


MODEL_NAMES = ("CatBoost", "XGBoost")
EXPECTED_RAW_FEATURES = 590
DEFAULT_PROFILE = "balanced_f2"
FEATURE_PATTERN = re.compile(r"^feature[_\- ]?(\d+)$", re.IGNORECASE)
OPTIONAL_COLUMN_NAMES = {
    "id",
    "index",
    "row_id",
    "sample_id",
    "wafer_id",
    "lot_id",
    "timestamp",
    "datetime",
    "date",
    "time",
    "label",
    "target",
    "class",
    "defect",
    "unnamed: 0",
    "아이디",
    "행 번호",
    "시간",
    "날짜",
    "라벨",
    "불량",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whitespace-delimited SECOM sensor rows with both models."
    )
    parser.add_argument("--input", required=True, help="Path to a 590-column data file")
    parser.add_argument(
        "--output",
        default="결과물/secom/dual_model_results/diagnosis_results.csv",
        help="Destination CSV path",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=(
            "balanced_f2",
            "recall_50",
            "recall_60",
            "recall_70",
            "recall_80",
            "recall_90",
        ),
        help="OOF-derived operating threshold profile",
    )
    parser.add_argument(
        "--top-drivers",
        type=int,
        default=5,
        help="Number of signed SHAP drivers to record per model",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Optional number of leading rows to process (useful for a smoke test)",
    )
    return parser.parse_args()


def _decode_sensor_file(raw_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("파일 문자를 UTF-8 또는 CP949로 해석할 수 없습니다.")


def _header_token_to_feature_index(token: object) -> int | None:
    text = str(token).strip()
    match = FEATURE_PATTERN.fullmatch(text)
    if match:
        index = int(match.group(1))
        return index if 0 <= index < EXPECTED_RAW_FEATURES else None
    if text.isdigit():
        index = int(text)
        return index if 0 <= index < EXPECTED_RAW_FEATURES else None
    return None


def parse_sensor_bytes(
    raw_bytes: bytes,
    row_limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Parse safe SECOM-compatible whitespace or comma-separated sensor rows."""
    text = _decode_sensor_file(raw_bytes)
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        raise ValueError("파일에 데이터가 없습니다.")

    first_line = nonempty_lines[0]
    comma_separated = "," in first_line
    separator = "," if comma_separated else r"\s+"
    if comma_separated:
        first_tokens = [token.strip() for token in first_line.split(",")]
    else:
        first_tokens = re.split(r"\s+", first_line.strip())
    has_header = any(
        FEATURE_PATTERN.fullmatch(token)
        or token.strip().lower() in OPTIONAL_COLUMN_NAMES
        for token in first_tokens
    )
    frame = pd.read_csv(
        StringIO(text),
        sep=separator,
        header=0 if has_header else None,
        nrows=row_limit,
        engine="python",
    )

    ignored_columns: list[str] = []
    if has_header:
        feature_columns: dict[int, object] = {}
        unknown_columns: list[str] = []
        for column in frame.columns:
            index = _header_token_to_feature_index(column)
            if index is not None:
                if index in feature_columns:
                    raise ValueError(f"센서 열 feature_{index}가 중복되었습니다.")
                feature_columns[index] = column
            elif str(column).strip().lower() in OPTIONAL_COLUMN_NAMES:
                ignored_columns.append(str(column))
            else:
                unknown_columns.append(str(column))

        expected = set(range(EXPECTED_RAW_FEATURES))
        missing = sorted(expected - set(feature_columns))
        if missing or unknown_columns:
            details = []
            if missing:
                preview = ", ".join(f"feature_{index}" for index in missing[:5])
                details.append(f"누락 센서 {len(missing)}개({preview})")
            if unknown_columns:
                details.append("알 수 없는 열 " + ", ".join(unknown_columns[:5]))
            raise ValueError(
                "590개 센서 헤더 feature_0~feature_589가 필요합니다: "
                + "; ".join(details)
            )
        frame = frame[
            [feature_columns[index] for index in range(EXPECTED_RAW_FEATURES)]
        ]
        frame.columns = [
            f"feature_{index}" for index in range(EXPECTED_RAW_FEATURES)
        ]
    else:
        if frame.shape[1] != EXPECTED_RAW_FEATURES:
            raise ValueError(
                f"헤더 없는 파일은 센서 열 590개가 필요하지만 "
                f"{frame.shape[1]}개가 발견됐습니다."
            )
        frame.columns = [
            f"feature_{index}" for index in range(EXPECTED_RAW_FEATURES)
        ]

    converted = frame.apply(pd.to_numeric, errors="coerce")
    invalid = converted.isna() & frame.notna()
    if invalid.any().any():
        row, column = np.argwhere(invalid.to_numpy())[0]
        value = frame.iat[int(row), int(column)]
        raise ValueError(
            "숫자로 변환할 수 없는 센서 값이 있습니다: "
            f"{frame.columns[int(column)]}={value!r}"
        )
    converted = converted.dropna(how="all").reset_index(drop=True)
    if converted.empty:
        raise ValueError("센서 데이터 행이 없습니다.")

    file_format = "CSV" if comma_separated else "공백 구분"
    header_format = "헤더 있음" if has_header else "헤더 없음"
    return converted, {
        "format": f"{file_format} · {header_format}",
        "rows": len(converted),
        "sensor_columns": converted.shape[1],
        "ignored_columns": ignored_columns,
    }


def load_rows(path: Path, row_limit: int | None) -> pd.DataFrame:
    rows, _ = parse_sensor_bytes(path.read_bytes(), row_limit)
    return rows


def shap_matrix(model, X_ready: np.ndarray) -> np.ndarray:
    raw = shap.TreeExplainer(model).shap_values(X_ready)
    if isinstance(raw, list):
        values = np.asarray(raw[-1])
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            values = values[:, :, -1]
    if values.shape != X_ready.shape:
        raise ValueError(f"Unexpected SHAP shape {values.shape}")
    return values


def format_drivers(
    shap_row: np.ndarray,
    feature_names: list[str],
    feature_values: np.ndarray,
    top_n: int,
) -> str:
    order = np.argsort(np.abs(shap_row))[::-1][:top_n]
    items = []
    for index in order:
        direction = "+" if shap_row[index] > 0 else "-"
        items.append(
            f"{feature_names[index]}={feature_values[index]:.6g}"
            f"({direction}{abs(shap_row[index]):.4f})"
        )
    return "; ".join(items)


def consensus_label(cat_prediction: int, xgb_prediction: int) -> str:
    if cat_prediction == 1 and xgb_prediction == 1:
        return "both_models_defect"
    if cat_prediction == 1:
        return "catboost_only_defect"
    if xgb_prediction == 1:
        return "xgboost_only_defect"
    return "both_models_normal"


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[2]
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_dir / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X_raw = load_rows(input_path, args.row_limit)
    results = pd.DataFrame({"input_row": np.arange(len(X_raw), dtype=int)})
    predictions = {}

    for model_name in MODEL_NAMES:
        bundle_path = (
            project_dir
            / "결과물"
            / "secom"
            / "dual_model_results"
            / f"{model_name.lower()}_model.joblib"
        )
        bundle = joblib.load(bundle_path)
        X_model = X_raw[bundle["input_features"]]
        X_imputed = bundle["imputer"].transform(X_model)
        X_ready = bundle["selector"].transform(X_imputed)
        probabilities = bundle["model"].predict_proba(X_ready)[:, 1]
        threshold = float(bundle["operating_thresholds"][args.profile])
        model_predictions = (probabilities >= threshold).astype(int)
        values = shap_matrix(bundle["model"], X_ready)
        key = model_name.lower()
        results[f"{key}_probability"] = probabilities
        results[f"{key}_threshold"] = threshold
        results[f"{key}_prediction"] = model_predictions
        results[f"{key}_top_shap_drivers"] = [
            format_drivers(
                values[row],
                bundle["selected_features"],
                X_ready[row],
                args.top_drivers,
            )
            for row in range(len(X_raw))
        ]
        predictions[model_name] = model_predictions

    results["consensus"] = [
        consensus_label(int(cat), int(xgb))
        for cat, xgb in zip(predictions["CatBoost"], predictions["XGBoost"])
    ]
    results.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"rows={len(results)}")
    print(f"profile={args.profile}")
    print("consensus_counts:")
    print(results["consensus"].value_counts().to_string())
    print(f"saved={output_path}")
    print(
        "note=SHAP drivers explain model predictions; anonymous feature numbers "
        "are not physical root-cause labels."
    )


if __name__ == "__main__":
    main()
