# WM-811K 데이터 탐색·정리(EDA) — 확정된 기준

`코드/wm811k_eda.py`로 원본 `LSWMD.pkl` 전체(811,457건)를 탐색한 결과를 바탕으로,
"의미있는 데이터만 추려서 가공"하는 기준을 아래와 같이 확정했다.

## 확정된 정리 기준

1. **라벨(`failureType`)이 있는 172,950건만 사용한다.** (전체 811,457건 중 21.3%)
   - 근거: 라벨이 없는 638,507건은 `trianTestLabel`도 함께 비어 있어, 원본
     데이터셋 자체가 "검사·분류된 웨이퍼"와 "미검사 웨이퍼"를 이미 구분해
     놓은 것으로 확인됨 (`eda_summary.json`의 `train_test_flat_counts` 참고).
   - 이 기준은 `코드/wm811k_prepare_data.py`의 `load_lswmd_labeled()`가 이미
     그대로 구현하고 있으므로 **추가 코드 변경은 필요 없다.**
2. **추가적인 이상치 제거는 하지 않는다.**
   - 다이 개수(die_count) 100 미만: 172,950건 중 4건, 웨이퍼 맵 한 변이 10픽셀
     미만: 2건 — 무시해도 되는 수준이라 별도 규칙을 만들지 않기로 함.
3. **9개 failureType 클래스를 전부 유지한다** (표본이 적은 `Near-full`(149건)
   포함). 클래스 불균형은 데이터 정리 단계가 아니라 모델링 단계(클래스 균형
   샘플러 등, `코드/wm811k_train.py`에 이미 적용)에서 다룬다.
4. **lot 단위 구조는 그대로 둔다.** 라벨 있는 lot은 10,762개(전체 46,293개 중
   23%)이며 lot당 웨이퍼 수 중앙값은 23개로, 극단적으로 적은 lot(1~5개)도
   소수 존재하지만 별도로 제외하지 않는다 — 기존 프로젝트가 확정한
   lot-aware 분할(`결과물/wm811k/split_results/split_assignments.csv`)을 그대로
   재사용하므로, lot 구성 자체를 이 단계에서 바꾸지 않는 것이 비교
   가능성(baseline과 동일 분할) 요구에 맞다.

## 결론

`코드/wm811k_prepare_data.py`가 이미 위 기준대로 동작하고 있어 파이프라인
코드는 변경하지 않았다. 이 폴더(`결과/wm811k/eda/`)의 통계·그래프는 이 결정의
근거 자료로 보관한다.

- `eda_summary.json` — 전체 통계 요약
- `labeled_wafer_detail.csv` — 라벨 있는 172,950건의 메타데이터(lot명, 웨이퍼
  인덱스, failure_type, 웨이퍼 맵 크기, 다이/결함 개수) — 원본 웨이퍼 배열은
  포함하지 않음
- `failure_type_distribution.png`, `labeled_vs_unlabeled.png`,
  `die_count_distribution.png`, `wafers_per_lot_distribution.png` — 분포 그래프

