# SECOM Linux 전이 의존성 잠금

- 판정: **PASS**
- 대상: Linux x86_64 / Python 3.11.9
- GitHub Actions: 81 package, 실제 설치 버전 일치
- Streamlit: 99 package, resolver 검증만 완료
- sdist: 0개, 모든 wheel SHA-256 기록

CI는 커밋된 lock으로 설치합니다. Streamlit lock은 Community Cloud 자동 적용을 가정하지 않으며 실제 설치 재현 검증 전까지 경고 상태입니다.
