"""독립 검증 독립 전처리기.

기준선(`코드/secom/preprocess_secom.py`)과 의도적으로 다른 점:

1. 제거할 변수(고결측·상수)를 전체 데이터가 아니라 **학습 fold 안에서만** 정한다.
   기준선은 train+test 전체를 보고 결정하므로 약한 정보 누수가 있다.
2. 결측 자체를 신호로 본다. 결측률이 일정 구간인 변수에는 결측 지시 변수를 추가한다.
3. StandardScaler 대신 QuantileTransformer(정규분포 출력)를 쓴다.
   SECOM 센서는 꼬리가 두껍고 이상치가 많아 분위수 변환이 더 안정적이다.

모든 통계는 fit에서 본 데이터에서만 계산하므로 Pipeline 안에 넣으면
교차검증 fold마다 자동으로 학습 데이터에만 적합된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer

# 결측률이 이 값 이상인 변수는 버린다(기준선은 0.50).
DROP_MISSING_RATIO = 0.45
# 한 값이 이 비율 이상을 차지하면 사실상 상수로 본다(기준선은 완전 상수만 제거).
QUASI_CONSTANT_RATIO = 0.995
# 결측 지시 변수를 만들 결측률 구간.
INDICATOR_MIN_RATIO = 0.02
INDICATOR_MAX_RATIO = DROP_MISSING_RATIO


class FriendPreprocessor(BaseEstimator, TransformerMixin):
    """결측 지시 변수 + 중앙값 대치 + 분위수 정규화를 묶은 변환기."""

    def __init__(
        self,
        drop_missing_ratio: float = DROP_MISSING_RATIO,
        quasi_constant_ratio: float = QUASI_CONSTANT_RATIO,
        indicator_min_ratio: float = INDICATOR_MIN_RATIO,
        add_missing_indicators: bool = True,
        random_state: int = 42,
    ) -> None:
        self.drop_missing_ratio = drop_missing_ratio
        self.quasi_constant_ratio = quasi_constant_ratio
        self.indicator_min_ratio = indicator_min_ratio
        self.add_missing_indicators = add_missing_indicators
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: N803 - sklearn 규약
        frame = self._as_frame(X)

        missing_ratio = frame.isna().mean()
        keep = missing_ratio[missing_ratio < self.drop_missing_ratio].index

        reduced = frame[keep]
        # 관측값 중 최빈값 비율이 임계 이상이면 정보가 거의 없는 변수로 본다.
        top_share = reduced.apply(self._top_value_share)
        keep = top_share[top_share < self.quasi_constant_ratio].index

        self.kept_columns_ = list(keep)
        if not self.kept_columns_:
            raise ValueError("남은 변수가 없습니다. 임계값을 확인하세요.")

        kept_missing = missing_ratio[self.kept_columns_]
        if self.add_missing_indicators:
            self.indicator_columns_ = [
                column
                for column in self.kept_columns_
                if self.indicator_min_ratio <= kept_missing[column] < INDICATOR_MAX_RATIO
            ]
        else:
            self.indicator_columns_ = []

        base = frame[self.kept_columns_]
        self.medians_ = base.median()
        # 관측값이 하나도 없는 변수는 median이 NaN이 되므로 0으로 둔다.
        self.medians_ = self.medians_.fillna(0.0)

        filled = base.fillna(self.medians_)
        matrix = self._stack(filled, frame)

        # n_quantiles가 표본 수보다 크면 sklearn이 경고와 함께 줄인다. 미리 맞춘다.
        n_quantiles = int(min(1000, max(2, matrix.shape[0])))
        self.quantile_ = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            subsample=100_000,
            random_state=self.random_state,
        )
        self.quantile_.fit(matrix)

        self.feature_names_out_ = list(self.kept_columns_) + [
            f"{column}__missing" for column in self.indicator_columns_
        ]
        self.n_features_in_ = frame.shape[1]
        self.input_columns_ = list(frame.columns)
        return self

    def transform(self, X):  # noqa: N803 - sklearn 규약
        frame = self._as_frame(X)
        missing_expected = [c for c in self.kept_columns_ if c not in frame.columns]
        if missing_expected:
            raise ValueError(
                f"입력에 필요한 변수가 없습니다(예: {missing_expected[:3]})."
            )
        base = frame[self.kept_columns_].fillna(self.medians_)
        matrix = self._stack(base, frame)
        return self.quantile_.transform(matrix)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    # ------------------------------------------------------------------
    def _stack(self, filled: pd.DataFrame, original: pd.DataFrame) -> np.ndarray:
        if not self.indicator_columns_:
            return filled.to_numpy(dtype=float)
        indicators = (
            original[self.indicator_columns_].isna().to_numpy(dtype=float)
        )
        return np.hstack([filled.to_numpy(dtype=float), indicators])

    @staticmethod
    def _top_value_share(column: pd.Series) -> float:
        observed = column.dropna()
        if observed.empty:
            return 1.0
        counts = observed.value_counts()
        return float(counts.iloc[0] / len(observed))

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(
            np.asarray(X, dtype=float),
            columns=[f"feature_{i}" for i in range(np.asarray(X).shape[1])],
        )


class AnomalyScoreAppender(BaseEstimator, TransformerMixin):
    """IsolationForest 이상 점수를 변수 한 개로 덧붙인다.

    레이블을 쓰지 않는 비지도 신호라서, 지도학습 분류기가 보지 못하는
    '평소와 다른 웨이퍼' 축을 하나 더 준다. 기준선의 GBDT 계열에는 없는 채널이다.
    """

    def __init__(self, n_estimators: int = 300, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: N803 - sklearn 규약
        from sklearn.ensemble import IsolationForest

        matrix = np.asarray(X, dtype=float)
        self.detector_ = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.detector_.fit(matrix)
        return self

    def transform(self, X):  # noqa: N803 - sklearn 규약
        matrix = np.asarray(X, dtype=float)
        score = self.detector_.score_samples(matrix).reshape(-1, 1)
        return np.hstack([matrix, score])



