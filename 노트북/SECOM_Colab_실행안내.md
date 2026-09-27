# SECOM TabDDPM Colab 실행 안내

실행 파일: `SECOM_TabDDPM_Colab.ipynb`

## 실행 순서

1. GitHub에서 노트북 파일을 내려받거나 Google Drive에 업로드합니다.
2. Google Colab에서 노트북을 엽니다.
3. `런타임` → `런타임 유형 변경` → `T4 GPU`를 선택합니다.
4. 먼저 설정 셀의 `RUN_MODE = 'smoke'`로 바꾸고 위에서부터 모두 실행합니다.
5. 마지막 셀까지 성공하면 `RUN_MODE = 'paper'`로 바꿉니다.
6. `런타임` → `세션 다시 시작` 후 위에서부터 모두 실행합니다.
7. 마지막에 내려받는 `secom_tabddpm_results.zip`을 원본 이름 그대로 전달합니다.

## 알아둘 점

- SECOM 원본은 UCI에서 자동 다운로드되므로 Kaggle API 토큰이 필요 없습니다.
- `SAVE_TO_DRIVE = True`이면 Google Drive 연결 승인이 한 번 필요합니다.
- 정식 실행 결과는 Drive의 `MyDrive/AI_project/SECOM_TabDDPM/secom_tabddpm_results_paper.zip`에도 저장됩니다.
- Colab 창을 닫아도 Drive에 복사된 최종 ZIP은 남지만, 학습 도중 런타임 연결이 끊기면 그 실행은 다시 시작해야 합니다.
- `smoke` 결과는 작동 확인용이며 최종 비교에는 `paper` 결과를 사용합니다.

## 전달할 파일

`secom_tabddpm_results.zip` 하나만 전달하면 됩니다. ZIP에는 설정, 학습 모델, 합성 불량 데이터, 합성 품질 지표, CatBoost·XGBoost 검증 결과와 실행 메타데이터가 포함됩니다.
