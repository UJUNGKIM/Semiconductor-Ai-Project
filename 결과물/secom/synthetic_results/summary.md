# 생성형 증강 비교 결과

- 실제 입력: 1567행, 불량 104건
- test: 314행, 불량 21건 — 합성/증강 미적용
- fold 내부 전처리: median → ANOVA 상위 50개 → 표준화
- 생성 모델: CTGAN/TVAE, minority-only, 75 epochs
- 생성 목표 비율: 0.5, 0.75, 1.0
- 임계값: fold별 F1 최대, 최종은 fold 임계값 중앙값

## 전략군별 CV 1위와 실제 test 성능

| family   | model    | strategy         |   cv_pr_auc_mean |   pr_auc |   precision |   recall |     f1 |   tp |   fp |   fn |   tn |
|:---------|:---------|:-----------------|-----------------:|---------:|------------:|---------:|-------:|-----:|-----:|-----:|-----:|
| scale    | CatBoost | scale_pos_weight |           0.1954 |   0.1596 |      0.1613 |   0.2381 | 0.1923 |    5 |   26 |   16 |  267 |
| SMOTE    | CatBoost | SMOTE_1.00       |           0.187  |   0.1831 |      0.1579 |   0.1429 | 0.15   |    3 |   16 |   18 |  277 |
| TVAE     | XGBoost  | TVAE_0.75        |           0.1734 |   0.142  |      0.1538 |   0.0952 | 0.1176 |    2 |   11 |   19 |  282 |
| CTGAN    | CatBoost | CTGAN_0.50       |           0.1729 |   0.1571 |      0.125  |   0.1905 | 0.1509 |    4 |   28 |   17 |  265 |

## 전체 CV 1위

**CatBoost / scale_pos_weight**  
CV PR-AUC=0.1954, test PR-AUC=0.1596,
precision=0.1613, recall=0.2381, F1=0.1923

합성 행 수는 실측 정보량과 동일하지 않습니다. 품질 지표와 untouched real test의
downstream 성능을 함께 보고 사용 여부를 결정해야 합니다.
