"""독립 검증 후보 모델 정의.

기준선은 CatBoost / XGBoost / LightGBM 부스팅 트리 계열이다.
독립 검증는 일부러 다른 계열을 쓴다.

- 규제 선형 모델(L2 / L1 로지스틱): 고차원·소표본에서 분산이 작고 계수 해석이 쉽다.
- ExtraTrees: 부스팅과 달리 분산 감소형 배깅 계열이라 과적합 양상이 다르다.
- 비지도 이상 점수 결합: 불량이 104건뿐인 데이터에서 레이블을 쓰지 않는 축을 하나 더 준다.
- 두 계열 soft voting: 서로 다른 오류를 내는 모델을 평균해 안정성을 노린다.

모든 후보는 전처리·변수선택까지 Pipeline 안에 넣어, 교차검증에서
학습 fold 밖 정보가 새지 않게 한다.

solver 선택 메모: elasticnet(saga)도 시험했지만 50,000 iteration에서도 수렴하지
않고 fold당 2분 이상 걸려 제외했다. lbfgs(L2)와 liblinear(L1)는 100 iteration
안에서 수렴한다.
"""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocess import AnomalyScoreAppender, FriendPreprocessor

RANDOM_STATE = 42


def _mutual_info(X, y):
    """SelectKBest용 상호정보량 점수 함수(시드를 고정해 재현 가능하게 한다)."""
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)


def _preprocessor() -> FriendPreprocessor:
    return FriendPreprocessor(random_state=RANDOM_STATE)


def _l2_logistic(C: float = 0.05) -> LogisticRegression:
    return LogisticRegression(
        l1_ratio=0.0,
        C=C,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def _l1_logistic(C: float = 0.1) -> LogisticRegression:
    return LogisticRegression(
        l1_ratio=1.0,
        C=C,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def _extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_candidates() -> dict[str, Pipeline]:
    """후보 이름 -> 파이프라인 사전을 만든다."""
    return {
        "l2_logistic": Pipeline(
            [
                ("prep", _preprocessor()),
                ("select", SelectKBest(f_classif, k=150)),
                ("clf", _l2_logistic(C=0.05)),
            ]
        ),
        "l1_logistic": Pipeline(
            [
                ("prep", _preprocessor()),
                ("select", SelectKBest(f_classif, k=300)),
                ("clf", _l1_logistic(C=0.1)),
            ]
        ),
        "mi_logistic": Pipeline(
            [
                ("prep", _preprocessor()),
                ("select", SelectKBest(_mutual_info, k=100)),
                ("clf", _l2_logistic(C=0.1)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("prep", _preprocessor()),
                ("select", SelectKBest(f_classif, k=250)),
                ("clf", _extra_trees()),
            ]
        ),
        "anomaly_logistic": Pipeline(
            [
                ("prep", _preprocessor()),
                ("select", SelectKBest(f_classif, k=150)),
                ("anomaly", AnomalyScoreAppender(random_state=RANDOM_STATE)),
                ("clf", _l2_logistic(C=0.05)),
            ]
        ),
        "vote_blend": Pipeline(
            [
                ("prep", _preprocessor()),
                (
                    "clf",
                    VotingClassifier(
                        estimators=[
                            (
                                "linear",
                                Pipeline(
                                    [
                                        ("select", SelectKBest(f_classif, k=150)),
                                        ("model", _l2_logistic(C=0.05)),
                                    ]
                                ),
                            ),
                            (
                                "trees",
                                Pipeline(
                                    [
                                        ("select", SelectKBest(f_classif, k=250)),
                                        ("model", _extra_trees()),
                                    ]
                                ),
                            ),
                        ],
                        voting="soft",
                    ),
                ),
            ]
        ),
    }


CANDIDATE_DESCRIPTIONS = {
    "l2_logistic": "분위수 정규화 + ANOVA 150변수 + L2 로지스틱(C=0.05, class_weight=balanced)",
    "l1_logistic": "분위수 정규화 + ANOVA 300변수 + L1 로지스틱(C=0.1, 계수 희소화)",
    "mi_logistic": "분위수 정규화 + 상호정보량 100변수 + L2 로지스틱(C=0.1)",
    "extra_trees": "분위수 정규화 + ANOVA 250변수 + ExtraTrees 500그루(balanced_subsample)",
    "anomaly_logistic": "ANOVA 150변수 + IsolationForest 이상 점수 1개 추가 + L2 로지스틱",
    "vote_blend": "L2 로지스틱과 ExtraTrees의 soft voting 평균",
}



