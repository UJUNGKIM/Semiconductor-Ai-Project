"""독립 검증 대안 실험의 공통 경로, 분할 재현, 지표 계산 모듈.

학습·후보 선정은 운영 모델 구현을 import하지 않고 원본 데이터에서 별도로 수행한다.
최종 비교 단계만 저장된 기준 지표를 읽기 전용으로 사용한다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# experiments/independent_validation/secom/common.py -> parents[3] == 프로젝트 루트
PROJECT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_DIR / "데이터" / "SECOM 데이터셋" / "raw"
BASELINE_RESULT_DIR = PROJECT_DIR / "결과물" / "secom"
RESULT_DIR = PROJECT_DIR / "결과물" / "secom" / "independent_validation"

# 기준선(코드/secom/preprocess_secom.py)과 동일한 고정 분할 조건.
# 같은 test 314건 위에서 비교하기 위해 이 값만 그대로 맞춘다.
RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPECTED_RAW_FEATURES = 590
EXPECTED_TRAIN_ROWS = 1253
EXPECTED_TEST_ROWS = 314

# 운영 목표: 불량 놓침을 줄이는 쪽이 중요하므로 recall 하한을 두고 F2를 최대화한다.
MIN_RECALL = 0.60


def load_raw() -> tuple[pd.DataFrame, pd.Series]:
    """UCI SECOM 원본 파일을 읽어 (X, y)를 돌려준다."""
    data_path = RAW_DIR / "secom.data"
    labels_path = RAW_DIR / "secom_labels.data"
    missing = [p for p in (data_path, labels_path) if not p.is_file()]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"SECOM 원본 파일을 찾을 수 없습니다: {joined}")

    X = pd.read_csv(data_path, sep=r"\s+", header=None)
    labels = pd.read_csv(labels_path, sep=r"\s+", header=None)
    if len(X) != len(labels):
        raise ValueError(f"행 수 불일치: data={len(X)}, labels={len(labels)}")
    if X.shape[1] != EXPECTED_RAW_FEATURES:
        raise ValueError(
            f"원본 변수 수가 {EXPECTED_RAW_FEATURES}가 아닙니다: {X.shape[1]}"
        )

    X.columns = [f"feature_{c}" for c in X.columns]
    y = labels.iloc[:, 0].map({-1: 0, 1: 1})
    if y.isna().any():
        raise ValueError("레이블에 -1/1 이외의 값이 있습니다.")
    return X, y.astype("int8").rename("label")


def load_fixed_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """원본에서 기준선과 동일한 고정 train/test 분할을 재현한다.

    train_test_split은 행 인덱스와 stratify 레이블만으로 분할하므로,
    전처리 방식이 달라도 같은 seed면 같은 행이 test로 간다.
    덕분에 독립 검증는 기준선의 전처리 산출물에 전혀 의존하지 않으면서도
    같은 test 314건 위에서 비교할 수 있다.
    """
    X, y = load_raw()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    if len(X_train) != EXPECTED_TRAIN_ROWS or len(X_test) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"분할 크기가 예상과 다릅니다: train={len(X_train)}, test={len(X_test)}"
        )
    return X_train, X_test, y_train, y_test


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


@dataclass
class ThresholdChoice:
    """확률 임계값 한 개와 그 지점의 지표."""

    threshold: float
    precision: float
    recall: float
    f2: float
    constraint_met: bool

    def to_dict(self) -> dict:
        return asdict(self)


def select_threshold(
    y_true: np.ndarray, y_score: np.ndarray, min_recall: float = MIN_RECALL
) -> ThresholdChoice:
    """recall >= min_recall 조건에서 F2를 최대화하는 임계값을 고른다.

    PR 곡선에는 '전부 불량이라 예측'하는 recall=1.0 지점이 늘 포함되므로,
    min_recall이 1.0 이하이면 조건은 사실상 항상 충족된다. 즉 이 조건은
    '달성 가능한가'를 판정하는 장치가 아니라, precision만 높고 불량을 대부분
    놓치는 임계값을 후보에서 걸러내는 필터로 쓴다.

    그래도 조건을 만족하는 지점이 하나도 없으면 조건을 풀고 F2만 최대화한 뒤
    constraint_met=False로 표시한다. 결과가 조용히 바뀌지 않게 하기 위함이다.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve는 마지막 점에 대응하는 임계값이 없다.
    precision, recall = precision[:-1], recall[:-1]
    if thresholds.size == 0:
        return ThresholdChoice(0.5, 0.0, 0.0, 0.0, False)

    scores = np.array([fbeta(p, r) for p, r in zip(precision, recall)])
    eligible = recall >= min_recall
    constraint_met = bool(eligible.any())
    pool = np.where(eligible)[0] if constraint_met else np.arange(scores.size)
    best = pool[int(np.argmax(scores[pool]))]
    return ThresholdChoice(
        threshold=float(thresholds[best]),
        precision=float(precision[best]),
        recall=float(recall[best]),
        f2=float(scores[best]),
        constraint_met=constraint_met,
    )


def score_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict:
    """고정 임계값에서의 전체 지표를 계산한다."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(fbeta(precision, recall)),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def to_markdown_table(frame: pd.DataFrame) -> str:
    """DataFrame을 마크다운 표로 만든다.

    pandas의 to_markdown은 tabulate 패키지를 요구한다. 독립 검증 하나를 위해
    프로젝트 의존성을 늘리지 않으려고 직접 만든다.
    """
    columns = [str(c) for c in frame.columns]
    rows = [[_cell(value) for value in record] for record in frame.itertuples(index=False)]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in rows)) if rows else len(columns[i])
        for i in range(len(columns))
    ]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    divider = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = [
        "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _cell(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "예" if value else "아니오"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

