# WM-811K 취약 클래스 진단

Scratch·Loc·Edge-Loc 오류 310건 중 208건(67.1%)이 none 예측입니다.
고정 신뢰도 정책은 이 오류 중 285건(91.9%)을 검토 대상으로 포착했습니다.

다음 실험은 train 표본에만 약한 결함 보존 증강과 none hard-negative sampling을 적용하고 validation으로만 선택합니다. Test는 최종 확인 전까지 잠급니다.

이 결과는 test 사후 진단이며 현재 모델이나 임계값을 변경하지 않습니다.
