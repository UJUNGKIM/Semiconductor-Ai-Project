# SECOM 생성형 증강 강화 결과

- 실제 데이터: 1567건 (불량 104건)
- 고정 test: 314건 (불량 21건), 합성 데이터 미사용
- fold 내부 처리: median → ANOVA 상위 50개 → 표준화 → DDPM 학습/품질 게이트
- 비교 모델: CatBoost, XGBoost
- DDPM 목표 불량/정상 비율: 0.25, 0.50, 0.75
- 채택 규칙: CV PR-AUC +0.005 이상이고 5개 fold 중 3개 이상 승리

## 결론

**retain_scale_pos_weight**  
기준: CatBoost / scale_pos_weight (CV PR-AUC 0.1954)  
최상 DDPM: CatBoost / DDPM_0.50 (CV PR-AUC 0.1616, 차이 -0.0338, fold 승리 1/5)

## 합성 데이터 품질 (5-fold 평균)

- KS 유사도: 0.7543
- 상관 유사도: 0.9221
- real-vs-synthetic 판별 AUC: 0.7069 (0.5에 가까울수록 구별 어려움)
- 실제 불량 커버리지: 0.8885
- 암기 위험률: 0.0000
- 종합 품질 점수: 0.7811

## 실제 test 참고 결과

CV 1위 구성 CatBoost / scale_pos_weight: PR-AUC 0.1596, precision 0.1613, recall 0.2381, F1 0.1923

모델/증강 선택은 test가 아니라 교차검증 결과로 결정했습니다. 합성 행 수는 실측 정보량과 같지 않으며, 외부 lot 검증 전에는 연구용 프로토타입으로 해석해야 합니다.
