# SECOM 외부 신규 lot 전향 검증

프로토콜 `DC6F68673693A0FBB71A`는 외부 데이터를 보기 전에 모델 해시, 임계값, 최소 표본과 합격 기준을 고정합니다. 현재 실제 독립 데이터가 없어 상태는 `AWAITING_INDEPENDENT_DATA`입니다.

`external_lot_template.csv`의 헤더를 사용해 신규 데이터 파일을 만들고, 라벨 담당자가 모델 출력을 보지 않은 상태에서 라벨을 확정한 뒤 `declarations_template.json`의 항목을 사실인 경우에만 `true`로 바꾸고 평가 스크립트를 실행합니다. 기존 SECOM 행과 정확히 겹치면 통과할 수 없습니다. 결과에는 원시 센서 행을 저장하지 않습니다.
