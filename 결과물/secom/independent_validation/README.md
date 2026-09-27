# 독립 검증 대안 실험 검증 결과 (SECOM)

이 문서는 `결과물/secom/independent_validation/`에 보관하는 독립 실험 결과입니다.
학습·후보 선정은 운영 모델 구현을 import하지 않고 원본에서 별도로 수행했습니다.
최종 비교 단계만 저장된 기준 지표를 읽기 전용으로 사용합니다.

## 데이터와 절차

- 원본: UCI SECOM 1567행 × 590변수, 불량 104건
- 분할: 기준선과 동일한 고정 분할을 원본에서 재현 (train 1253 / test 314, seed 42)
- 전처리는 기준선 산출물을 쓰지 않고 독립 검증 코드에서 다시 계산
- 제거 기준: 결측률 ≥ 0.45, 최빈값 비율 ≥ 0.995 (모두 **학습 fold 안에서만** 판정)
- 결측 지시 변수 추가, 중앙값 대치, QuantileTransformer(정규분포) 적용
- 검증: RepeatedStratifiedKFold 5-fold × 3회, OOF 확률은 반복 평균
- 임계값: OOF에서 recall ≥ 0.60 조건의 F2 최대점
- 고정 test는 후보 선택이 끝난 뒤 1회만 확인

## 후보 비교 (OOF, test 미사용)

| candidate        | oof_pr_auc | fold_pr_auc_mean | fold_pr_auc_std | oof_precision | oof_recall | oof_f2 | oof_threshold | recall_constraint_met |
|------------------|------------|------------------|-----------------|---------------|------------|--------|---------------|-----------------------|
| extra_trees      | 0.1893     | 0.2121           | 0.0765          | 0.1330        | 0.6024     | 0.3531 | 0.0981        | 예                     |
| l1_logistic      | 0.1465     | 0.1657           | 0.0523          | 0.1066        | 0.6627     | 0.3243 | 0.2089        | 예                     |
| mi_logistic      | 0.1655     | 0.1617           | 0.0445          | 0.0926        | 0.8193     | 0.3189 | 0.1763        | 예                     |
| vote_blend       | 0.1749     | 0.1946           | 0.0687          | 0.0884        | 0.8916     | 0.3165 | 0.0633        | 예                     |
| l2_logistic      | 0.1468     | 0.1699           | 0.0644          | 0.0878        | 0.7831     | 0.3032 | 0.0837        | 예                     |
| anomaly_logistic | 0.1468     | 0.1700           | 0.0644          | 0.0878        | 0.7831     | 0.3032 | 0.0838        | 예                     |

후보 설명:

- `extra_trees`: 분위수 정규화 + ANOVA 250변수 + ExtraTrees 500그루(balanced_subsample)
- `l1_logistic`: 분위수 정규화 + ANOVA 300변수 + L1 로지스틱(C=0.1, 계수 희소화)
- `mi_logistic`: 분위수 정규화 + 상호정보량 100변수 + L2 로지스틱(C=0.1)
- `vote_blend`: L2 로지스틱과 ExtraTrees의 soft voting 평균
- `l2_logistic`: 분위수 정규화 + ANOVA 150변수 + L2 로지스틱(C=0.05, class_weight=balanced)
- `anomaly_logistic`: ANOVA 150변수 + IsolationForest 이상 점수 1개 추가 + L2 로지스틱

## 선택 모델

- 후보: **extra_trees**
- 설명: 분위수 정규화 + ANOVA 250변수 + ExtraTrees 500그루(balanced_subsample)
- OOF PR-AUC: 0.1893
- OOF precision: 0.1330
- OOF recall: 0.6024
- OOF F2: 0.3531
- fold PR-AUC 평균 ± 표본 표준편차: 0.2121 ± 0.0765
- OOF에서 고정한 임계값: 0.098109

## 고정 test 확인 (1회)

- PR-AUC: 0.2376
- ROC-AUC: 0.7538
- precision: 0.1373
- recall: 0.6667
- F1: 0.2276
- F2: 0.3763
- 혼동행렬: TP=14, FP=88, FN=7, TN=205

## 한계

- 고정 test 314건에는 불량이 21건뿐입니다. 지표 한 개의 차이가 웨이퍼 1~2건에서 나옵니다.
  따라서 이 숫자만으로 기준선과의 우열을 단정하지 않습니다.
- 이 test 분할은 기준선 실험에서 이미 여러 번 관찰된 세트입니다.
  완전히 새로운 외부 검증이 아니라 '같은 조건에서의 확인'으로만 읽어야 합니다.
- SECOM 센서는 익명이므로 어떤 변수가 선택되어도 물리적 고장 원인이나
  공정 조정 지시로 해석할 수 없습니다.
- 여기 결과는 기존 모델을 대체하자는 제안이 아니라, 최종 비교·선택 단계에서
  나란히 놓고 보기 위한 대안 후보입니다.

## 재현

```powershell
.venv\Scripts\python.exe "experiments\independent_validation\secom\train_validate.py"
```
