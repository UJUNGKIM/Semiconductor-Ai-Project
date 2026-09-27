# CatBoost와 XGBoost 설명가능성 분석

- 설명 대상: 고정 test 314개 샘플
- 방법: TreeSHAP
- SHAP 부호가 양수이면 모델의 불량 raw score를 높이고, 음수이면 낮춤
- CatBoost 선택 변수: 200개
- XGBoost 선택 변수: 100개

## 전역 중요 변수

- CatBoost 상위 10개: feature_59, feature_103, feature_33, feature_488, feature_477, feature_31, feature_577, feature_21, feature_587, feature_86
- XGBoost 상위 10개: feature_103, feature_488, feature_59, feature_587, feature_577, feature_21, feature_477, feature_511, feature_333, feature_31
- 상위 20개 중 공통 변수: 13개
- 공통 변수: feature_103, feature_180, feature_21, feature_213, feature_281, feature_31, feature_333, feature_477, feature_488, feature_511, feature_577, feature_587, feature_59
- 전체 중요도 Spearman 상관: 0.4835

## 예측 일치

- 균형형 임계값에서 두 모델 예측 일치율: 0.7771
- 불일치 샘플 수: 70개

## 해석상 주의

SECOM 변수명은 센서 의미가 없는 익명 번호입니다. 따라서 SHAP은 어떤 feature 번호가
예측을 올리거나 내렸는지는 보여주지만, 그 자체로 온도·압력 같은 실제 공정 원인을
증명하지 않습니다. 센서 사전 및 공정 엔지니어 검토를 연결해야 '원인 진단'으로 확장할
수 있습니다. 또한 이 test는 앞선 실험에서 이미 관찰했으므로 성능 선택용이 아니라 설명
예시용으로만 사용했습니다.
