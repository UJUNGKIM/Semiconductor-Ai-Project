# SECOM Champion–Challenger 승격 심사 데모

반복 train OOF 예측을 5개 가상 배치로 나눈 통계 검증 데모이며 실제 승격 근거가 아닙니다. 고정 test 라벨은 규칙 설정에 사용하지 않았습니다.

- decision: HISTORICAL_DEMO_ONLY
- XGBoost−CatBoost 재현율 차이 95% CI: [0.0964, 0.2651]
- 정상 오탐률 차이 95% CI: [0.1667, 0.2179]
- McNemar exact p-value: 0.000000
- 재현율 비열등성: True
- 정상 오탐률 guardrail: False
- 자동 승격: 금지
