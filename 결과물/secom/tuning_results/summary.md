# 반복 교차검증 모델 최적화 결과

- 비교: XGBoost, CatBoost, LightGBM 가중치 모델
- 탐색: 모델별 6개 설정, 총 18개
- 검증: RepeatedStratifiedKFold 5-fold × 3회
- 목표: OOF recall ≥ 0.60 조건에서 F2 최대화
- median/변수선택은 각 fold 학습 데이터에서만 적합

## 선택 모델

- 모델: **CatBoost** (`catboost_01`)
- 선택 변수 수: 200
- OOF PR-AUC: 0.2018
- OOF precision: 0.1463
- OOF recall: 0.6627
- OOF F2: 0.3884
- OOF 기반 임계값: 0.0718

## 고정 test 확인

- PR-AUC: 0.2505
- ROC-AUC: 0.7011
- precision: 0.1477
- recall: 0.6190
- F2: 0.3779
- 혼동행렬: TP=13, FP=75, FN=8, TN=218

고정 test는 이전 실험에서 이미 결과를 확인했으므로 완전히 새로운 외부 검증 세트는 아닙니다.
최종 성능 주장은 반복 OOF 결과와 향후 외부 검증을 중심으로 판단해야 합니다.
