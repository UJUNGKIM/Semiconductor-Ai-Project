# SECOM 실행환경 재현성·소프트웨어 공급망 검증

- 판정: **PASS**
- Python: 3.11.9
- exact-pinned 직접 의존성: 18개
- 설치 distribution: 103개
- pip check: 통과
- 모의 버전 이탈 탐지: True
- 모델 역직렬화 전 해시 게이트: True
- 취약점 스캔: 수행하지 않음

이 결과는 현재 실행환경 재현성과 무결성 확인 자료이며 보안 취약점이 없다는 증명이 아닙니다. Windows와 Linux CI lock은 별도 생성되며 Streamlit 설치 적용, 서명된 artifact와 정기 취약점 스캔은 별도 통제로 유지해야 합니다.
