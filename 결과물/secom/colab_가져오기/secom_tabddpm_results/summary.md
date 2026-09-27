# SECOM 공식 TabDDPM Colab 결과

- 실행 모드: paper
- GPU: Tesla T4
- 학습 step: 30,000
- 학습 시간: 5.4분
- validation 선택: CatBoost / TabDDPM 1:1 균형 증강
- validation PR-AUC: 0.1512
- validation F2: 0.3731
- 임계값: validation F2 기준 0.0320
- test PR-AUC: 0.2474
- test F2: 0.3846
- test는 모델·전략·임계값 선택에 사용하지 않음

합성 데이터는 기준선보다 분류 성능을 높였지만, 합성 판별기 AUC가 1.0이므로 실제
불량과 합성 불량의 분포 차이가 매우 크다. 따라서 이 결과는 연구용 증강 실험으로
사용하고 실제 센서 데이터와 동일한 데이터로 표현하면 안 된다.
