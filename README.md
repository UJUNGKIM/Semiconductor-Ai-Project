# Semiconductor-Ai-Project

AI-based semiconductor defect analysis using SECOM sensor data and WM-811K wafer maps

**SHAPGPT**는 반도체 공정 센서(SECOM)와 웨이퍼 맵(WM-811K)을 AI로 진단하고, 각 판정의 근거를
SHAP으로 함께 보여 주는 설명 가능한 AI 프로젝트입니다. 이 저장소는 그 공개본입니다.

## 공개 저장소 안내

- 저장소: https://github.com/UJUNGKIM/Semiconductor-Ai-Project
- SECOM 데이터: `데이터/SECOM 데이터셋/`의 원본(`secom.data`, `secom_labels.data`,
  `secom.names`)은 UCI Machine Learning Repository의 SECOM 데이터셋(Michael McCann,
  Adrian Johnston, DOI 10.24432/C54305)이며 CC BY 4.0 라이선스로 배포됩니다. 대시보드
  시연과 회귀 테스트에 필요해 함께 포함합니다.
- WM-811K 데이터: 원본 `LSWMD.pkl`(약 1.95GB)은 포함하지 않습니다. 저장소에는 학습한 모델,
  학습·검증 결과 파일, 대시보드 시연용 대표 웨이퍼 45개(64×64)가 들어 있습니다. 원본이
  필요하면 아래 WM-811K 절의 안내에 따라 직접 내려받으세요.
- 비밀값: 로그인 정보, API 키, `.streamlit/secrets.toml`은 저장소에 없습니다.
  `.streamlit/secrets.toml.example`은 자리표시자만 담은 예시이며, 실제 값은 배포 환경의
  Secrets에만 입력합니다.
- 연구·교육용 시연이며 실제 생산 라인의 자동 판정을 대신하지 않습니다.

AI 티키타카 프로젝트의 통합 작업 폴더입니다. 현재 **SECOM 공정 센서 기반 불량
진단**과 **WM-811K 웨이퍼 맵 결함 분류**를 하나의 설명가능 AI 시스템으로 구현합니다.

## 폴더 구조

```text
Semiconductor-Ai-Project/
├─ app.py                 # 통합 Streamlit 웹사이트·대시보드
├─ dashboard_ui/          # 공개 페이지·접근 권한·샘플 갤러리·웨이퍼 맵 표시·화면 도우미
│  └─ site_pages/         # 상단 메뉴 페이지 등록용 표시 파일(화면은 app.py가 그림)
├─ run_dashboard.ps1      # 대시보드 실행
├─ 데이터/
│  ├─ SECOM 데이터셋/raw/          # UCI 원본 데이터
│  ├─ SECOM 데이터셋/processed/    # 전처리된 train/test와 imputer
│  └─ WM-811K(LSWMD) 데이터셋/     # 웨이퍼 맵 원본 데이터
├─ 코드/secom/            # SECOM 분석·학습·검증 코드
├─ 코드/wm811k/           # WM-811K 메모리 안전 전처리·학습 코드
├─ 노트북/                # Colab GPU·고용량 RAM 실행 노트북
└─ 결과물/secom, wm811k/  # 모델, 표, 보고서, 그래프
```

## 대시보드 실행

PowerShell에서 이 폴더로 이동한 다음 실행합니다.

```powershell
.\run_dashboard.ps1
```

대시보드는 외부 네트워크에 공개되지 않고 `http://127.0.0.1:8501`에서만 열립니다.

### 웹사이트 구조와 권한

상단 메뉴는 Streamlit `st.navigation(position="top")`으로 구성합니다.

| 페이지 | 주소 | 볼 수 있는 사용자 |
|---|---|---|
| 홈 | `/` | 모두. 프로젝트 소개, 분석 4단계, 저장된 검증 결과에서 읽은 핵심 지표, 사용 범위와 생산 적용 한계 |
| SECOM 진단 | `/secom` | 로그인 필수 모드에서는 로그인한 사용자. 검토 대기열·개별 SHAP 설명·안전·What-if와 안내 |
| WM-811K 진단 | `/wm811k` | 로그인 필수 모드에서는 로그인한 사용자 |
| 프로젝트·검증 | `/project` | 모두. 검증 요약, 사용 범위, 용어 설명과 검증 자료 4개 영역(SECOM 모델 검증, SECOM 증강 검증, SHAP/XAI 검증 근거, WM-811K 검증 자료) |
| 로그인 / 계정 | `/account` | 로그인 필수 모드에서만 메뉴에 표시. 이름·이메일·역할·로그아웃 |
| 관리자 | `/admin` | 관리자로 판정된 사용자에게만 등록. 다른 사용자가 주소를 입력하면 홈으로 이동 |

`dashboard_ui/site_pages/*.py`는 페이지 주소를 등록하는 표시 파일이고, 화면은
`app.py`가 그립니다. 예전 `?module=secom&section=...` 형식의 링크는 해당 페이지와
화면으로 이어집니다. SECOM `모델 검증`·`증강 검증`은 진단 화면에서 프로젝트·검증
페이지로 옮겼으며, 예전 `section=validation`·`section=augmentation` 링크는
`/project?section=secom_validation`·`secom_augmentation`으로 열립니다. 옮긴 화면의
내용·수치·표·그래프는 그대로입니다. 홈과 프로젝트·검증의 지표 카드에는 짧은 출처명만
표시하고, 정확한 근거 파일 경로는 카드 아래 `검증 근거 보기`에 둡니다. 수치는 모두
커밋된 결과 파일에서 읽으며 새로 계산하거나 코드에 적어 두지 않습니다. 푸터의
저장소 주소는 `SHAPGPT_REPOSITORY_URL` 환경 변수로 바꿀 수 있습니다.

진단 화면은 SECOM과 WM-811K를 공통 4단계인 `입력 신뢰도 → 모델 판정 →
SHAP 기여 → 검토 행동`으로 표시합니다. SECOM은 TreeSHAP, WM-811K는 같은
웨이퍼 형상의 모든 die가 정상인 기준과 비교하는 Gradient SHAP(Expected
Gradients 근사)을 사용합니다. 두 기여도 모두 모델 판단 근거이며 물리적 원인
확률이 아닙니다.

운영 모니터링, 릴리스 준비도, WM-811K 전체 실험 기록은 관리자 화면이며 관리자
페이지와 각 진단 페이지의 '관리자' 탭에만 보입니다. 메뉴를 숨기는 것과 별개로,
관리자 전용 화면의 딥링크(`?section=monitoring` 등)도 렌더링 전에 서버에서 권한을
다시 확인하고 권한이 없으면 기본 화면으로 돌려보냅니다. 판정 로직은
`dashboard_ui/access.py`에 있습니다.

- **로그인 없는 데모(기본값)**: 환경 변수를 설정하지 않으면 모든 페이지가 로그인
  없이 열리고 관리자 메뉴는 없습니다. 홈에는 로그인 대신 진단 시작 버튼이 보입니다.
- **Google 로그인 필수**: Streamlit 내장 OIDC(`st.login`)를 사용하며 이 앱은
  비밀번호나 회원 정보를 저장하지 않습니다. secrets의 `[auth]`(`redirect_uri`,
  `cookie_secret`)와 `[auth.google]`(`client_id`, `client_secret`,
  `server_metadata_url`)을 넣고 `SHAPGPT_REQUIRE_LOGIN=1`을 지정합니다. 제공자
  설정을 `[auth]`에 바로 넣는 형식도 인식합니다. 로그인하지 않은 사용자는 홈과
  프로젝트·검증만 볼 수 있고, 진단 페이지에서는 로그인 안내와 `Google로 로그인`
  버튼만 봅니다. 설정이 빠졌거나 불완전하면 버튼을 비활성화하고 안내만 표시하며,
  로그인하지 않은 상태를 로그인한 것처럼 처리하지 않습니다. `st.login`에는
  `Authlib`가 필요하므로 `requirements.txt`에 `Authlib==1.8.0`을 고정했습니다.
  로그인한 사용자 화면에는 이름·이메일·역할만 표시하고 원시 토큰, `sub`, 전체
  claims는 표시하지 않습니다.
- **관리자 권한**: 로그인한 사용자별로 판정하며 기본값은 관리자 아님입니다.
  `SHAPGPT_ADMIN_EMAILS`(쉼표 구분) 또는 secrets의 `[shapgpt] admin_emails`에 있는
  이메일이면서 ID 토큰의 `email_verified`가 true일 때, 또는
  `SHAPGPT_ADMIN_SUBJECTS`/`[shapgpt] admin_subjects`에 `발급자(iss)|주체(sub)`가
  정확히 일치할 때만 관리자입니다. `email_verified`를 주지 않는 제공자는
  `iss|sub` 목록을 사용합니다. 이메일 주소와 비밀값은 코드에 넣지 않으며
  `.streamlit/secrets.toml`은 Git에서 제외됩니다.
- **로컬 개발용 `SHAPGPT_ADMIN_MODE=1`**: 로그인 없는 모드이면서 Streamlit
  `server.address`가 루프백(`127.0.0.1`, `localhost`, `::1`)일 때만 적용됩니다.
  `run_dashboard.ps1`은 `127.0.0.1`에 바인딩하므로 로컬 개발에서는 동작하고,
  공개 주소로 실행하거나 `SHAPGPT_REQUIRE_LOGIN=1`과 함께 쓰면 무시됩니다.

현재 로컬 저장소에서는 로그인 화면과 관리자 권한 판정을 단위 테스트와 Streamlit
AppTest로 확인했지만, 실제 OIDC 제공자와의 로그인 왕복은 배포 환경의 secrets 설정
후 별도로 확인해야 합니다. 이 기능은 화면 접근 분리이며 사용자별 데이터 격리까지
완성한 B2C 권한 관리는 아닙니다.

#### Google 로그인 설정 순서(배포 관리자)

1. Google Cloud 콘솔에서 OAuth 동의 화면을 설정하고 **웹 애플리케이션** 유형의
   OAuth 클라이언트 ID를 만듭니다.
2. 승인된 리디렉션 URI에 `https://<배포-주소>/oauth2callback`을 등록합니다. 로컬에서
   확인하려면 `http://localhost:8501/oauth2callback`도 등록합니다.
3. `.streamlit/secrets.toml.example`을 참고해 배포 플랫폼의 Secrets에 `[auth]`,
   `[auth.google]`, `[shapgpt]` 값을 입력합니다. 로컬에서는 Git에서 제외되는
   `.streamlit/secrets.toml`에만 입력합니다. `cookie_secret`은 32자 이상의 임의
   문자열로 만듭니다.
4. 환경 변수 `SHAPGPT_REQUIRE_LOGIN=1`을 설정합니다.
5. 배포 후 로그인 왕복, 일반 계정에 관리자 메뉴가 없는지, 허용 목록 계정에 관리자
   메뉴가 보이는지, 일반 계정으로 `/admin`을 직접 열면 홈으로 돌아가는지 확인합니다.

### WM-811K 대표 샘플 갤러리와 웨이퍼 맵 표시

- 실제 클래스를 고르면 demo manifest에 있는 그 클래스의 대표 샘플을 한 화면에
  모두 카드로 보여 줍니다(현재 클래스당 5개, 한 줄 최대 4개, 넘치면 다음 줄).
  카드에는 웨이퍼 맵, 샘플 유형, 실제 클래스, 저장 예측, 보정 모델 점수, 검토 필요
  여부를 저장된 값으로 표시하고, CNN·Gradient SHAP·OOD 상세 계산은 `이 샘플 분석`
  으로 고른 한 개에만 실행합니다. 파일 업로드와 인공 데모 맵도 그대로 쓸 수 있습니다.
- WM-811K 값 0·1·2는 범주형이므로 각 칸을 정수 배율 블록으로 복제(최근접)해
  무손실 PNG로 표시합니다. 보간·안티앨리어싱·초해상도를 쓰지 않으며, 상세 맵은
  최소 512×512, 썸네일은 256×256으로 미리 만들고 CSS `image-rendering: pixelated`를
  적용합니다. 화면 확대는 정보량을 늘리지 않으며, 모델 입력 검증과 64×64 최근접
  전처리는 바뀌지 않았습니다.
- Gradient SHAP·Grad-CAM 지도는 연속값이므로 범주형 맵과 다른 순차 색 척도와
  범례를 사용합니다. 색 농도만 표시용으로 0~1 정규화하며 die 한 칸을 한 색
  블록으로 그립니다.

파일 업로드는 헤더 없는 590열 공백 파일과 `feature_0`~`feature_589` 헤더가
있는 CSV를 지원합니다. `row_id`, `timestamp`, `label` 같은 부가 열은 자동으로
제외하며, 화면에서 내려받을 수 있는 CSV 양식을 제공합니다. 센서 순서·단위가
학습 데이터와 다른 파일은 같은 590열이어도 유효한 입력으로 간주할 수 없습니다.

SECOM 설정은 `진단 실행`을 눌렀을 때만 새 결과에 반영됩니다. 기본 화면은 OOD,
모델 불일치, 경계값, 공통 불량을 합친 검토 대기열이며 표에서 선택한 행을 개별
SHAP 설명으로 이어서 볼 수 있습니다. 필터별 건수와 행 검색을 제공하며, 표에서
행을 선택하면 오른쪽에서 두 모델 점수·상위 SHAP 근거·검토 저장 도구를 함께
확인할 수 있습니다. 현재 모듈·화면·행 번호는 URL에 저장되어
새로고침하거나 링크를 공유해도 같은 위치로 복원됩니다. 결과표, 검토 대기열,
OOD 판정, 실행 메타데이터와 HTML 보고서는 통합 ZIP으로 한 번에 내려받을 수 있습니다.
화면은 선택한 섹션만 렌더링하므로 숨겨진 분석을 매번 다시 계산하지 않습니다.
운영 임계값 비교는 모델 추론을 반복하지 않고 이미 계산된 점수에 저장된 임계값을
적용합니다. 사용자가 저장한 실행 집계와 검토 결정은 원본 센서값 없이 로컬
append-only SQLite 감사 저장소에 기록되며, 최근 실행 간 변화도 화면에서 비교합니다.

## 환경 설치와 자동실행

```powershell
# 최초 1회: 가상환경 생성 및 패키지 설치
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip==26.2.1"
python -m pip install -r requirements.txt

# 빠른 환경·데이터 검사
.\run_pipeline.ps1 -Mode verify

# 데이터·모델·대시보드 자동 테스트만 실행
.\run_tests.ps1

# SECOM DDPM 증강 강화 실험만 재실행
.\run_pipeline.ps1 -Mode augment

# 전처리부터 모든 증강·모델·시간순 검증까지 전체 재실행
.\run_pipeline.ps1 -Mode full
```

`full` 모드는 CTGAN·TVAE·DDPM과 반복 교차검증을 포함하므로 실행 시간이 오래 걸립니다.
일상적인 확인에는 기본값인 `verify`를 사용합니다.

자동 테스트는 원본 데이터 형상, 446개 변수 전처리, 고정 train/test 분할,
CatBoost·XGBoost 저장 모델 예측, 진단 입력 및 대시보드 렌더링을 검사합니다.

## SECOM 고급 안전 진단

대시보드의 `안전·What-if` 탭은 성능 점수 외에 실제 운용을 위한 안전장치를
제공합니다.

- train의 결측률·센서 1~99% 범위·PCA 최근접 거리로 입력 OOD를 판정합니다.
- OOD, 모델 불일치, 경계값, 두 모델 공통 불량을 결합해 전문가 검토 순서를 만듭니다.
- CatBoost SHAP 패턴이 비슷한 실제 train 불량 83개를 4개 패턴 후보로 분류합니다.
- What-if는 SHAP 상위 변수, train 분위수, 최대 4개 변수, 변수당 IQR 3배 이내로 제한합니다.
- 고정 test 314행에서 5% train-IQR 노이즈를 3회 주입해 SHAP 상위 변수 순위와 부호
  안정성을 검사하고, 상위 SHAP 10개 변수의 중앙값 마스킹 효과를 같은 개수 무작위
  변수 마스킹과 비교합니다. 이 사후 감사는 모델·임계값 선택에 사용하지 않습니다.
- 검토자 결정과 메모는 로컬 append-only SQLite 감사 저장소에 보존하고 CSV로도
  내려받을 수 있습니다. SQLite는 중앙 접근 제어·외부 백업을 대신하지 않습니다.
- 입력 결과 해시와 모델 SHA-256이 포함된 독립 HTML 감사 보고서를 생성합니다.

불량 패턴과 What-if는 익명 센서에 대한 모델 기반 점검 후보이며 물리적 고장명,
인과관계 또는 공정 조정 지시가 아닙니다. `결과물/secom/advanced_diagnostics/`에
train-only 기준, 군집 지도와 재현 메타데이터가 저장됩니다.

`운영 모니터링` 탭은 진단 배치별 OOD 비율, 모델 불일치율, 불량 경보 수,
전문가 검토 수와 배치 차단 상태를 누적합니다. 센서 원본값 대신 진단 결과의
SHA-256만 기록하며, 이전 CSV를 다시 불러와 운영 추세를 이어갈 수 있습니다.
Streamlit Cloud의 브라우저 세션은 영구 저장소가 아니므로 작업 후 CSV를 반드시
다운로드해야 합니다. 함께 제공되는 합성 스트레스 재생은 화면 검증용이며 실제
운영 이력으로 해석하지 않습니다.

실제 검수가 끝난 배치는 실제 불량 수와 그중 모델 경보에 포함됐던 수를 함께
기록할 수 있습니다. `balanced_f2`와 현재 모델 SHA-256이 train 반복 OOF 기준과
일치할 때만 정밀도·재현율을 계산합니다. 3배치·200행·실제 불량 30건 전에는
성능 저하를 판정하지 않으며, 재현율 95% Wilson 신뢰구간과 연속 저하를 함께
확인합니다. 조건을 벗어나도 자동 재학습하지 않고 원본 저장소에서 다시 조회할
배치 ID manifest만 생성합니다.

숫자 직접 입력은 참고용 집계로만 남습니다. 권장 흐름은 대시보드에서 센서값이
포함되지 않은 행 단위 검수 양식을 내려받고 `actual_label`, 라벨 출처, 검수자,
검수 시각을 채워 다시 업로드하는 것입니다. 배치 해시·모델 해시·예측 열을 원본
진단과 대조하며, 전체 행 라벨이 확인된 경우에만 성능 감시 근거로 사용합니다.
부분 검수는 선택 편향 가능성을 표시하고 FP/FN 행 manifest 생성에만 사용합니다.

전체 행 검수 파일에서는 CatBoost를 champion, XGBoost를 challenger로 같은 행에서
비교합니다. 최소 3개 독립 배치·200행·불량 30건·예측 불일치 20건이 필요하며,
계층화 paired bootstrap으로 재현율·F2·정상 오탐률 차이의 95% 신뢰구간을 만들고
McNemar exact 검정도 함께 표시합니다. 재현율 비열등성 −5%와 정상 오탐률 증가
+5% guardrail을 모두 확인하며, 통과하더라도 자동 교체하지 않고 사람의 승격
검토만 허용합니다. OOF 결과는 통계 화면 시연용이며 실제 승격 근거가 아닙니다.

`릴리스 준비도` 탭은 전처리 계약, 모델 SHA-256, test 라벨 잠금, OOD/XAI 범위,
배치 차단기, 센서 오류 강건성, 사람 승인 정책을 하나의 결정적 Release ID로
묶습니다. 대회 데모 준비도와 실제 생산 배포 준비도를 별도로 표시하고 JSON,
manifest, HTML 감사 보고서를 내려받을 수 있습니다. 현재 대회 데모는 경고와 함께
가능하지만, 센서 의미표·독립 신규 lot 전향 검증·영구 감사 저장소가 없으므로 실제
생산 배포는 차단됩니다.

`통합 공정 흐름` 탭은 SECOM 전단 센서 감시 → 전문가 검토 우선순위 → WM-811K
후단 웨이퍼 검사 → 사람의 최종 판단을 연결합니다. 두 공개 데이터셋은 동일 wafer로
연결된 자료가 아니므로 개별 제품 수준의 결합 예측으로 표현하지 않습니다.

## 증강 상태

- SMOTE는 학습 세트만 1,253건에서 2,340건으로 늘려 비교했습니다.
- CTGAN·TVAE도 불량 클래스만 대상으로 0.50, 0.75, 1.00 비율을 비교했습니다.
- 경량 DDPM 계열 생성기는 fold 내부에서만 학습하고, 0.25, 0.50, 0.75 비율을 비교했습니다.
- DDPM 합성 데이터는 KS·상관·실제/합성 판별 AUC·커버리지·최근접 거리·암기 위험으로 품질을 검사했습니다.
- test 314건에는 합성·증강을 적용하지 않아 실제 데이터 기준으로 평가했습니다.
- DDPM은 평균 품질 점수 0.7811, 암기 위험률 0%였지만 CV PR-AUC가 기준보다 0.0338 낮았습니다.
- 따라서 최종 기본 후보는 증강 모델보다 안정적이었던 `scale_pos_weight` CatBoost입니다.
- 기존 결과는 `결과물/secom/synthetic_results/summary.md`, 강화 결과는 `결과물/secom/enhanced_synthetic_results/summary.md`에 저장되어 있습니다.

강건성 스트레스에서는 가우시안 노이즈 1.00 IQR의 모델 재현율 guardrail이
실패했으므로 모델 강건성 판정은 계속 WARN입니다. 다만 이 실패 조건의 5회 반복은
모두 배치 안전 차단기에서 STOP되고 자동판정이 금지됐습니다.
결과물/secom/sensor_failure_containment/는 강건성 원본 신호와 차단 결과를
일대일·해시로 교차검증합니다. 이는 모델 개선이나 실제 장비 고장 검증이 아니라
취약 입력을 자동 처리하지 않는 운영 안전장치 증거입니다.

## Colab TabDDPM 실험

`노트북/SECOM_TabDDPM_Colab.ipynb`을 Google Colab에 업로드하고 T4 GPU 런타임에서
위에서부터 실행합니다. 원본 `secom.data`, `secom_labels.data`는 UCI에서 자동으로
받으며 Kaggle 토큰은 필요하지 않습니다. 먼저 `smoke` 모드로 작동을 확인하고 새
세션에서 `paper` 모드로 정식 실행합니다. 공식 TabDDPM은 label-conditioned 합성 불량
데이터를 만들고 CatBoost·XGBoost 및 `scale_pos_weight` 기준선과 비교합니다. 자세한
순서는 `노트북/SECOM_Colab_실행안내.md`에 있습니다. 마지막에 내려받는
`secom_tabddpm_results.zip`은 `결과물/secom/colab_가져오기/`에 압축 상태 그대로 넣습니다.

수령한 정식 실행에서는 CatBoost + TabDDPM 1:1 증강이 test PR-AUC 0.2474, F2
0.3846을 기록했습니다. CatBoost `scale_pos_weight` 기준선보다 각각 높았지만 합성
판별기 AUC가 1.0이므로, 실제 분포 재현에는 강한 한계가 있습니다. 결과와 해석은
`결과물/secom/colab_가져오기/검증결과.md`에 기록되어 있습니다.

## SECOM 독립 대조 검증

`experiments/independent_validation/secom/`은 기준선과 같은 고정 분할을 원본에서 다시
만들고 ExtraTrees·희소 로지스틱 등 여섯 후보를 반복 5-fold × 3회 OOF로 비교합니다.
후보와 임계값은 Test를 보기 전에 고정합니다. 선택된 ExtraTrees는 고정 Test에서
PR-AUC 0.2376, F2 0.3763으로 tuned CatBoost(PR-AUC 0.2505, F2 0.3779)를 넘지
못했으므로 배포 모델이 아니라 독립 재현 참고값으로만 유지합니다.

모델 바이너리와 행 단위 예측은 저장소에 포함하지 않습니다. 재현이 필요하면 아래를
실행하고 새 모델의 SHA-256을 다시 검증해야 합니다.

```powershell
.venv\Scripts\python.exe "experiments\independent_validation\secom\train_validate.py"
.venv\Scripts\python.exe "experiments\independent_validation\secom\compare_with_baseline.py"
```

## Git 관리

- `.venv`, Python 캐시, 로컬 비밀 설정은 Git에서 제외합니다.
- `LSWMD.pkl`은 약 1.95GB이므로 Git에 넣지 않고 로컬 데이터 폴더에만 보관합니다.
- SECOM 코드·데이터·저장 모델·검증 결과는 대시보드를 바로 재현할 수 있도록 관리합니다.

## GitHub 자동 검증

`.github/workflows/project-ci.yml`은 모든 push와 pull request에서 Python 3.11.9,
정확히 고정된 의존성, `pip check`, SECOM 원본 계약, 릴리스 증거와 clean-clone 회귀 테스트를
검증합니다. 재학습과 외부 데이터 다운로드는 수행하지 않으므로 API 키나 Kaggle 토큰이
필요하지 않습니다. GitHub 토큰은 저장소 읽기 권한만 사용하고 공식 action은 릴리스
커밋 SHA로 고정했습니다. Linux CI에서는 디스크 사용량을 줄이기 위해 동일 버전의
PyTorch CPU wheel과 `xgboost-cpu`를 사용하며, GPU 학습 성능을 검증하는 단계는
Colab 실험에 남겨둡니다.

Git에서 의도적으로 제외한 WM-811K 원본 Colab ZIP·전략 비교 폴더가 필요한 테스트와
기존 Windows 원시 CSV 해시 증거 테스트 등 3개는 CI 로그에 이름과 사유를 표시하고
로컬 `run_tests.ps1`에서 검증합니다. 따라서 CI 통과가 대용량 Colab 결과까지 모두
재검증했다는 뜻은 아닙니다.

Actions 탭의 `Project regression tests`가 통과한 커밋만 `main`에 병합하는 것을
권장합니다. 브랜치 보호 규칙은 저장소 Settings에서 이 검사를 필수 상태 검사로 지정한
뒤 적용할 수 있습니다.

## Python 의존성 취약점 감사

CI는 회귀 테스트가 끝난 뒤 `pip-audit==2.10.1`로 `requirements.txt`의 해석된
의존성 그래프를 PyPI advisory에 조회합니다. 알려진 취약점이 하나라도 발견되면
workflow를 실패시키며, 성공·실패와
관계없이 정규화된 JSON과 요약을 30일간 Actions artifact로 보관합니다. 빌드에 사용하는
`pip`도 `26.2.1`로 고정합니다.

`결과물/secom/dependency_security/`의 보고서는 마지막 로컬 검사 시점의 증거이며,
대시보드 `릴리스 준비도`에서 확인할 수 있습니다. 이 감사는 Python distribution만
대상으로 하므로 운영체제, GPU 드라이버, CUDA, 네이티브 라이브러리와 미공개 취약점까지
안전하다는 보증은 아닙니다.

## Python 소스 보안 정적분석

CI는 `Bandit==1.9.4`로 통합 `app.py`와 `코드/secom`의 Python AST를 검사합니다.
중간 이상 심각도·신뢰도 이슈, 스캐너 오류 또는 `nosec` 억제가 하나라도 있으면
릴리스를 차단하고 정규화된 JSON 증거를 30일간 보관합니다. 원문 코드 조각은 결과물에
저장하지 않습니다. 매주 월요일에도 회귀·의존성·소스 보안 검사를 다시 실행합니다.

이 정적분석은 실행 중 동작과 모든 취약점을 증명하지 않으며, 별도 창에서 개발 중인
WM-811K 코드는 이번 SECOM 보안 gate 범위에 포함하지 않습니다.

## 변조 감지형 감사 원장

SECOM 대시보드에서 직접 집계 배치 기록 또는 행 단위 전체 검수 결과를 저장하면,
원본 센서값 없이 집계 지표·입력 결과 해시·모델 해시를 SHA-256 해시 체인으로
연결합니다. 이전 이벤트의 해시가 다음 이벤트에 포함되므로 중간 내용 수정, 중간 이벤트
삭제, 순서 변경은 JSONL 재검증에서 탐지됩니다. 이전 원장을 불러와 이어서 기록하거나
검증 가능한 `secom_audit_ledger.jsonl`로 내려받을 수 있습니다.

이 기능은 변경 탐지 기능이며 중앙 영구 보관 시스템은 아닙니다. 대시보드에서 저장한
실행·검토 이벤트는 로컬 SQLite에 남지만 서버 디스크 자체의 삭제 방지·접근통제·외부
백업·보존기간은 제공하지 않습니다. 실제 생산 배포 전에는 외부 영구 감사 저장소와
권한·백업 정책을 연결해야 합니다. 결정적 기능 증거는
`결과물/secom/audit_ledger/`에 있습니다.

로컬 내구성 검증을 위해 SQLite 기반 영구 감사 저장소 어댑터도 제공합니다. 각 이벤트는
`BEGIN IMMEDIATE` 트랜잭션으로 추가되고 UPDATE/DELETE trigger와 기존 SHA-256 체인으로
이중 검증됩니다. 프로세스를 종료하고 다시 연결한 뒤에도 기록이 복원되는지, trigger를
제거한 뒤 내용을 바꾸면 해시 검증이 실패하는지도 자동 테스트합니다. 운영 DB 자체는
민감한 감사 기록이므로 Git에 넣지 않습니다.

다만 로컬 SQLite는 SSO/RBAC, 승인된 보존기간, 암호화된 외부 백업·복원 훈련, 외부 불변
체인 앵커를 제공하지 않습니다. 따라서 `결과물/secom/production_audit_store/`는 기능
증거만 제공하며 해당 통제가 실제로 연결되기 전까지 운영 배포 상태는 계속 `BLOCKED`로
표시합니다.

## 외부 신규 lot 전향 검증

`결과물/secom/external_validation/`에는 외부 데이터를 보기 전에 고정한 검증 프로토콜과
590개 센서 입력 템플릿이 있습니다. CatBoost·XGBoost 모델 SHA-256, `balanced_f2`
임계값, OR 경보 규칙, 최소 500행·3개 lot·30개 불량, 재현율·오탐률·검토율의 Wilson
95% 구간 합격선을 미리 고정합니다. 평가 데이터로 모델 선택, 임계값 조정, calibration
재학습을 수행할 수 없습니다.

실제 파일은 `sample_id`, `lot_id`, 시간대가 포함된 `captured_at`, 0/1 `label`,
`feature_0~feature_589`를 포함해야 합니다. 기존 공개 SECOM 행과 정확히 겹치는 데이터는
외부 검증으로 인정하지 않으며, 라벨 담당자가 모델 출력을 보지 않았다는 선언도 필요합니다.
결과에는 원시 센서 행이나 행별 예측을 저장하지 않고 전체·lot별 집계만 남깁니다. 현재는
독립 신규 데이터가 없으므로 `AWAITING_INDEPENDENT_DATA`이고 운영 판정은 계속
`BLOCKED`입니다.

대시보드 `릴리스 준비도` 탭의 `독립 신규 lot 검증 실행`에서 CSV와 완료된 선언 JSON을
함께 올리면 등록된 모델 해시·임계값·합격선을 바꾸지 않고 평가합니다. 파일 크기와 행 수,
열 계약, 시간대, 중복 ID, 공개 데이터 행 중복을 검사하며 원본 업로드는 디스크에 저장하지
않습니다. 내려받는 결과도 전체·lot별 집계와 dataset digest만 포함합니다.

## 센서 의미·단위 사전

공개 SECOM의 `feature_0~feature_589`는 물리 의미가 공개되지 않았으므로 임의로 센서명을
생성하지 않습니다. `데이터/SECOM 데이터셋/sensor_dictionary_template.csv`에서 실제
장비·공정 담당자가 센서명, 단위, 공정 단계, 적용 장비, 설명, 물리 유효 범위, 책임자,
근거 문서와 검증 시각을 입력하도록 준비했습니다.

각 행은 `UNMAPPED → PROVISIONAL → VERIFIED` 상태를 가지며 `unknown`, `TBD`, `미정`
같은 placeholder는 매핑된 값으로 인정하지 않습니다. 운영 사전 설치는 590개 전체가
근거와 시간대가 있는 담당자 검증을 완료했을 때만 가능합니다. 현재는 실제 정보가 없어
0/590이므로 `SENSOR_SEMANTICS` 운영 gate는 계속 `BLOCKED`입니다.

대시보드의 `작성된 센서 사전 검증`에서 담당자가 채운 CSV를 검사할 수 있습니다. 검증은
590행·열 계약, feature 중복, 상태별 필수값, placeholder, 물리 유효 범위와 시간대를 확인하고
digest가 포함된 보고서와 정규화 CSV를 만듭니다. 대시보드는 운영 사전을 자동 설치하지
않으며, 기밀 장비·공정 정보는 공개 Streamlit 앱이 아닌 로컬 대시보드에서만 처리해야 합니다.

## 모델 무결성 및 장애 복구

CatBoost는 `ACTIVE`, XGBoost는 `VERIFIED_STANDBY`로 모델 레지스트리에 등록합니다.
각 모델은 파일 SHA-256, 입력 스키마 해시, 라벨 매핑 해시, 설정 ID와 고정 임계값으로
검증됩니다. 운영 모델 손상 훈련은 실제 파일을 수정하지 않고 메모리에서 불일치 해시를
만들어 탐지 경로만 점검합니다.

손상이 탐지되면 자동 진단을 중단하고 영향 배치를 격리합니다. 대기 모델로 자동 전환하지
않으며, 무결성·입력 계약·임계값·검토 업무량을 확인하고 책임자 승인을 받아야 합니다.
고정 test에서 두 모델 판정은 동일하지 않으므로 XGBoost를 CatBoost의 완전히 동일한
대체품으로 해석하지 않습니다. 증거는 `결과물/secom/model_failover/`에 있습니다.

## 데이터 계보 및 오염 탐지

원본 `secom.data`, `secom_labels.data`부터 구조 필터, 고정 stratified 분할, train
median 대체, 저장된 train/test CSV와 imputer까지 SHA-256과 스키마·행 ID fingerprint로
연결합니다. 검증 시 원본에서 전처리를 다시 실행해 저장 산출물의 행 순서, 변수 순서,
값, 라벨, median 통계가 동일한지 대조합니다.

원본 센서 파일과 라벨 파일의 변경 탐지는 실제 파일을 수정하지 않고 메모리상 모의
오염으로 점검합니다. Manifest에는 센서 원본값을 넣지 않습니다. 이 기능은 파일 변경을
탐지하지만 데이터 수집 당시 측정값의 진실성이나 신규 공정 대표성을 증명하지 않습니다.
증거는 `결과물/secom/data_lineage/`에 있습니다.

## 실행환경 재현성 및 SBOM

`requirements.txt`의 직접 의존성은 모두 정확한 버전으로 고정하고, 현재 가상환경과
일치하는지 및 `pip check` 통과 여부를 확인합니다. Python 버전과 직접 의존성으로 환경
지문을 만들고, 대시보드·실행 스크립트의 SHA-256과 모의 버전 이탈 탐지 결과를 함께
기록합니다. 모델·고급 진단 joblib 파일은 릴리스 manifest의 해시를 먼저 확인한 뒤에만
역직렬화합니다.

`pylock.windows.toml`은 현재 검증한 Windows x86-64/Python 3.11.9 환경의 전이
의존성과 PyPI wheel SHA-256을 PEP 751 형식으로 고정합니다. 생성 시 설치된 전체
잠금 버전과 현재 테스트 환경이 일치하는지도 확인하며, 결과는
`결과물/secom/dependency_lock/`에 기록됩니다. 재생성은 다음 명령으로 수행합니다.

```powershell
.\.venv\Scripts\python.exe .\코드\secom\build_windows_dependency_lock.py
```

pip 26.2.1의 `pip lock`은 experimental 기능이고 이 lock은 Windows 전용입니다.
Linux에는 Ubuntu Actions에서 생성한 `pylock.github-actions.toml`과
`pylock.streamlit.toml`을 별도로 둡니다. Actions는 첫 번째 lock에서 직접 설치하고
커밋된 lock의 전이 버전을 제약으로 다시 해석해 동일하게 재현되는지 검사합니다.
따라서 외부 패키지의 새 버전 공개만으로 일반 코드 CI가 깨지지 않습니다. 의존성을
의도적으로 갱신할 때만 Linux/Python 3.11.9 환경에서 builder에 `--refresh`를 지정하고
생성된 lock과 manifest를 검토해 커밋합니다. 두 번째 lock은
Streamlit용 Linux wheel 해석과 SHA-256을 기록하지만 Community Cloud가 named pylock을
자동 적용한다고 가정하지 않으며, 현재 공개 배포 설치 입력은 `requirements.txt`입니다.

Linux 잠금 manifest는 `결과물/secom/linux_dependency_lock/`에 있습니다. CI lock은
81개 package의 실제 설치 버전까지 대조했고 Streamlit lock은 99개 package의 해석만
검증했습니다. 두 lock은 `Authlib` 추가 후 Ubuntu 24.04/Python 3.11.9에서 `--refresh`
없이 다시 만들었습니다. 그래서 기존 lock의 버전은 그대로 두고 Authlib 관련 5개
package만 추가됐습니다.
두 범위는 대시보드와 릴리스 준비도에서 각각 PASS와 WARN으로 표시합니다.
대시보드의 릴리스 준비도 탭은 실행 중인 프로세스의 Python·플랫폼·직접 의존성
18개·잠금 전이 의존성 99개를 Streamlit 잠금과 즉시 대조합니다. 이 런타임
보고서는 패키지 metadata만 사용하며 환경변수, secrets, 사용자명과 호스트명은 수집하지
않습니다. 공개 배포에서 PASS가 표시돼야 설치 결과까지 일치한 것입니다.
wheel hash는 받은 파일의 무결성을 검증하지만 알려진 취약점이 없다는 뜻은 아닙니다.

`결과물/secom/environment_provenance/`의 SBOM은 현재 설치 패키지 스냅샷입니다.
취약점 스캔은 별도 pip-audit 증거로 수행합니다. 실제 생산 배포에는 Streamlit 잠금의
설치 적용 검증과 서명된 artifact·외부 투명성 로그가 추가로 필요합니다.

`app.py`, `dashboard_ui/` 또는 CI 핵심 파일을 수정했지만 의존성과 Python 환경은
바뀌지 않았다면 아래 순서로 코드 무결성 해시와 릴리스 준비도만 갱신합니다. 이 단계를
생략하면 GitHub Actions가 의도적으로 `실행환경 파일 무결성 불일치`로 중단됩니다.

```powershell
.\.venv\Scripts\python.exe .\코드\secom\build_environment_provenance.py --refresh-project-hashes-only
.\.venv\Scripts\python.exe .\코드\secom\build_release_readiness_report.py --skip-plot
.\.venv\Scripts\python.exe .\코드\verify_ci.py
```

`requirements.txt`, Python 버전 또는 설치 패키지가 바뀐 경우에는 축약 옵션을 사용하지
말고 정확히 고정된 Python 3.11.9 환경에서 `build_environment_provenance.py`를 전체
실행해 SBOM과 환경 검증 증거까지 새로 생성해야 합니다.

## WM-811K 시작 단계

원본 `LSWMD.pkl`은 약 1.95GB이며 전체 역직렬화에 큰 RAM이 필요합니다.
`노트북/WM811K_01_전처리_Colab.ipynb`은 Colab Secret의 `KAGGLE_API_TOKEN`을
사용해 Kaggle `qingyi/wm811k-wafer-map`에서 원본을 직접 내려받습니다. 로컬의 2GB
파일을 Drive에 올릴 필요가 없습니다. 라벨이 있는 웨이퍼맵은 64×64 `uint8`
메모리맵 배열로 변환해 Drive에 유지하고, `wm811k_preprocessing_receipt.zip`만
`결과물/wm811k/colab_가져오기/`에 가져옵니다.

전처리 영수증을 검증한 뒤 `코드/wm811k/create_lot_aware_splits.py`로 동일 lot가
서로 다른 세트에 섞이지 않는 약 71/14/14% train/validation/test 분할을 만듭니다.
`노트북/WM811K_02_CNN_학습_Colab.ipynb`은 Drive의 배열과 이 고정 분할을 사용해
T4 GPU에서 2채널 CNN을 학습합니다. class-balanced focal loss와 회전·반전 증강을
적용하고 validation macro-F1로 최적 epoch를 선택합니다. 마지막에 받는
`wm811k_cnn_results.zip`은 `결과물/wm811k/colab_가져오기/`에 넣습니다.

수령한 CNN은 lot 비중복 test 24,705개에서 정확도 96.78%, 균형 정확도 85.34%,
macro-F1 0.8280을 기록했습니다. 정상/불량 이진 관점의 불량 탐지 F1은 0.9238입니다.
`결과물/wm811k/colab_가져오기/wm811k_cnn_results/검증결과.md`에 ZIP 안전성,
분할 일치, 확률, 체크포인트와 클래스별 검증 내용을 기록했습니다.

상단 메뉴에서 `WM-811K 진단`을 열면 test 성능과
혼동행렬을 확인하고, 0·1·2 값으로 구성된 `.npy`, `.csv`, `.txt`, `.data`
웨이퍼 맵 한 개를 업로드해 CNN 예측과 Gradient SHAP·Grad-CAM 설명을 볼 수 있습니다.
기본 test 예시는 클래스별 5개(9개 클래스, 총 45개)의 실제 웨이퍼를 클래스별
갤러리 카드로 한 번에 표시하며, 고신뢰 정답·전형적 정답(신뢰도 중앙값 근처)·경계 근처 정답(검토 임계값
근처)·검토 대상 정답·대표 오분류라는 선정 근거를 함께 기록합니다. 선정 코드는
클래스별 4~5개를 지원하며 해당 사례가 없는 클래스는 4개가 되지만, 현재 test에서는
9개 클래스 모두 5개가 선정됐습니다. 이전에 배포한 27개(클래스별 고신뢰 정답·검토
대상 정답·대표 오분류)는 샘플 ID, 웨이퍼 배열, 예측 클래스가 그대로 유지되는 부분
집합이며, 보정 신뢰도 기록값만 CPU 재계산에 따라 최대 1.4×10⁻⁶ 달라졌습니다.
단일·배치 진단은 예측 패턴과 전역 형상 지표를 이용해 가능한 원인 후보 3개와
우선 점검 항목을 함께 제시합니다. 이 후보는 원인 확률이나 인과 판정이 아니며 실제
원인은 장비·레시피·센서·유지보수·재측정 기록과 공정 엔지니어 검토로 확인해야 합니다.

`노트북/WM811K_03_취약클래스_개선_Colab.ipynb`은 같은 분할에서 focal loss,
가중 cross-entropy, 제곱근 클래스 균형 샘플링을 비교합니다. 세 전략 모두 train에만
회전·반전 증강을 적용하고 validation macro-F1로 모델을 선택합니다. test 지표는
선택을 고정한 뒤 참고용으로만 저장합니다. 마지막에 받는
`wm811k_strategy_comparison_results.zip`은 `결과물/wm811k/colab_가져오기/`에
압축 상태로 넣습니다.

Validation macro-F1 기준으로 `ce_sqrt_balanced`가 선택됐습니다. 기존 focal 모델
대비 test macro-F1은 0.8280에서 0.8828, 불량 탐지 F1은 0.9238에서 0.9463으로
개선됐습니다. 대시보드는 `결과물/wm811k/selected_model_results/`의 선택 모델을
사용하며 기존 focal 결과는 비교 기준으로 보존합니다.

`experiments/independent_validation/wm811k/`은 같은 lot 비중복 분할에서 52개 형상
서술자와 HistGradientBoosting을 사용해 CNN을 독립적으로 대조합니다. 형상 모델의
Test Macro-F1은 0.8794, CNN은 0.8828이며, 10,000회 paired bootstrap에서 전체
Macro-F1과 balanced accuracy 차이의 95% 구간은 모두 0을 포함했습니다. 따라서
모델 교체 근거가 아니라 서로 다른 표현 방식으로 유사한 결과가 재현됐다는 검증으로
사용합니다. 두 모델의 단독 정답을 같은 Test에서 비교하면 형상 모델만 맞힌 샘플은
258개, CNN만 맞힌 샘플은 361개였습니다. 클래스별 형상 단서, 상호 보완 위치와 원본
811,457개 데이터의 라벨·lot EDA는 대시보드 `성능·검증 → 독립 검증`
화면에서 확인할 수 있습니다. 핵심 성능, 안전·OOD, SHAP 검증은 각각 별도
화면으로 분리하며, 후보 실험의 전체 표와 그래프는 관리자 모드에서만
`전체 실험 기록`으로 보존합니다.

```powershell
.venv\Scripts\python.exe "experiments\independent_validation\wm811k\wm811k_train_shape.py"
.venv\Scripts\python.exe "experiments\independent_validation\wm811k\wm811k_geometry_control.py"
.venv\Scripts\python.exe "experiments\independent_validation\wm811k\wm811k_paired_bootstrap.py"
```

재학습에는 로컬 `wafer_maps_64.npy`가 필요합니다. 생성되는 `.joblib`, 특징 캐시,
행 단위 예측은 Git에서 제외하고 JSON·CSV·Markdown 요약만 관리합니다.
대시보드 연결은 정적 계약 테스트와 실제 Streamlit 경로 렌더 테스트로 함께 검증합니다.

선택 모델의 validation 예측만 사용해 온도 보정과 불확실성 검토 정책도 고정했습니다.
보정 신뢰도 97.50% 미만은 `전문가 검토 필요`로 표시합니다. 이 기준을 고정한 test에서
90.41%를 자동 분류했고, 그 구간의 정확도는 99.84%였습니다. 전체 오분류 561개 중
525개(93.58%)가 검토 대상으로 포착됐습니다. 이 수치는 같은 데이터셋의 비중복 lot
평가이며 신규 장비·공정에 대한 성능 보장은 아닙니다.

Test 사후 진단에서는 `Edge-Loc → none` 92건과 `Loc → none` 82건이 가장 큰
혼동 유형이었습니다. 결함 클래스 중 `Scratch` 재현율이 71.76%로 가장 낮았습니다.
대시보드의 클래스별 오분류 분석에서 재현율, 검토 대상 비율, 오류 포착률과 주요
실제→예측 혼동 쌍을 확인할 수 있습니다. 이 분석은 모델 선택에 사용하지 않았습니다.

대표 샘플은 모델 선택이 끝난 뒤 test 예측에서 고른 정성 시연 자료입니다. 배포
모델의 행 단위 test 예측표는 Git에서 제외된 Colab 번들에만 있으므로, 로컬
`wafer_maps_64.npy`·`labels.npy`와 배포 체크포인트로 다시 계산한 뒤 커밋된 증거와
대조했습니다. `코드/wm811k/reproduce_wm811k_test_predictions.py`는 test 24,705개의
클래스별 성능표(최대 차이 1.1×10⁻¹⁶), 혼동행렬, 검토 정책 집계(자동 22,335·검토
2,370·오류 561·포착 525), 기존 27개 샘플의 예측과 보정 신뢰도가 모두 일치할 때만
예측표를 씁니다. 예측표는 `결과물/wm811k/local_reproduction/`(Git 제외)에 두고
해시와 대조 결과만 `selected_model_results/test_prediction_reproduction.json`에
기록합니다. 이어서 `select_demo_samples.py`가 manifest를 만들고
`export_wm811k_demo_samples.py`가 같은 배열에서 웨이퍼를 복사합니다. 기존 샘플
파일과 내용이 다르면 덮어쓰지 않고 중단하며, 배포에는 45개 NPY와 manifest만
포함합니다.

```powershell
.\.venv\Scripts\python.exe .\코드\wm811k\reproduce_wm811k_test_predictions.py --maps <wafer_maps_64.npy> --labels <labels.npy>
.\.venv\Scripts\python.exe .\코드\wm811k\select_demo_samples.py --test-predictions .\결과물\wm811k\local_reproduction\deployed_test_predictions.csv --policy .\결과물\wm811k\selected_model_results\uncertainty_policy.json --output .\결과물\wm811k\selected_model_results\demo_sample_manifest.json
.\.venv\Scripts\python.exe .\코드\wm811k\export_wm811k_demo_samples.py --maps <wafer_maps_64.npy> --labels <labels.npy>
```

`노트북/WM811K_04_실제샘플_추출_Colab.ipynb`은 같은 manifest를 받아 Drive의
`WM811K_processed/wafer_maps_64.npy`에서 클래스별 4~5개를 추출하는 Colab 경로이며,
27개 고정 조건 없이 클래스별 개수를 검사합니다.

단일 웨이퍼 진단은 Gradient SHAP과 Grad-CAM을 나란히 표시합니다. Gradient SHAP은
같은 웨이퍼 외형에서 모든 활성 die를 정상으로 바꾼 기준과 현재 입력 사이 경로를
층화 표본으로 근사한 Expected Gradients입니다. 기여도는 로짓 변화에 맞춰 사후
조정하지 않은 원시 값(보정 전 로짓 단위)이며, 화면의 색 밝기만 0~1로 정규화합니다.
원시 기여 합계와 로짓 변화의 차이(가산성 잔차)가 로짓 변화의 5%(최소 0.001)를
넘으면 경로 표본을 64→128→256개로 늘려 다시 계산하고, 그래도 넘으면 "가산성 점검
미통과"로 표시합니다. 유한 표본 근사이므로 잔차가 0이 되지 않을 수 있습니다.

`배치 진단` 탭에서는 실제 test 예시 45개 또는 사용자가 올린 웨이퍼 맵 최대 100개를
한 번에 분류합니다. Validation에서 고정한 보정 신뢰도 임계값에 따라 전문가 검토
대상을 먼저 표시하며, 전체 결과를 UTF-8 CSV로 내려받을 수 있습니다.

다음 안전성 실험은 `노트북/WM811K_05_OOD_기준생성_Colab.ipynb`에서 실행합니다.
train의 CNN pooled feature로 클래스별 기준을 적합하고 validation 점수의 95%·99%
분위수로 검토/OOD 임계값을 고정합니다. test는 고정 후 평가에만 사용합니다. 내려받은
`wm811k_ood_results.zip`은 압축을 풀지 않고 `결과물/wm811k/colab_가져오기/`에
넣습니다. 공간 셔플 stress는 파이프라인 점검용이며 실제 외부 장비 검증은 아닙니다.

수령한 OOD 결과는 pickle을 사용하지 않는 JSON·NPZ 형식, 안전한 ZIP 경로, 모델과
split SHA-256, validation-only 임계값 및 49,408개 validation/test 점수 상태를
검증했습니다. 대시보드의 단일·배치 진단은 저신뢰 확률뿐 아니라 CNN 특징 OOD와
train 기하 범위 이탈도 전문가 검토 사유로 사용합니다. test 자동 처리율은 95.54%,
OOD 보류율은 0.92%이며 공간 셔플 stress OOD 탐지율은 99.42%입니다.

단일 및 배치 진단 화면에서는 감사 가능한 HTML 보고서도 내려받을 수 있습니다. 보고서에
각 웨이퍼의 64×64 정규화 입력 SHA-256, 예측·보정 신뢰도·OOD·처리 권고와 CNN 및
OOD 기준 파일 SHA-256을 기록합니다. 사용자 파일명은 HTML escape 처리하며 보고서에도
이 모델이 물리적 원인이나 공정 조정 지시를 증명하지 않는다는 한계를 명시합니다.

선택이 끝난 모델에는 `코드/wm811k/evaluate_wm811k_robustness.py`로 입력 오류
스트레스 테스트를 적용합니다. 실제 test 정성 예시 중 기존 27개(클래스별 고신뢰 정답·
검토 대상 정답·대표 오분류, 이후 추가한 전형·경계 예시 제외)에 활성 die 누락, die 상태
반전, 블록 손실, 위치 이동을 조건별 5회 주입하고 예측 유지율과 변경된 예측의
검토·OOD 포착률을 기록합니다. 이 결과는 모델 선택이나 임계값 조정에 사용하지 않으며,
전체 데이터의 대표 성능이나 실제 장비·센서 고장 검증을 대체하지 않습니다.

모델 초기값에 따른 안정성은 `노트북/WM811K_06_다중시드_재현성_Colab.ipynb`에서
검증합니다. 고정된 lot 비중복 분할과 선택 전략 `ce_sqrt_balanced`를 유지하고 시드
17·42·2026만 바꿔 3회 학습합니다. 모델을 다시 선택하거나 가장 좋은 시드를 고르지
않고 validation 및 고정 test 지표의 평균·표본 표준편차를 보고합니다. 기존 시드 42
배포 모델은 변경하지 않습니다. 마지막에 받는
`wm811k_multiseed_reproducibility_results.zip`은 압축을 풀지 않고
`결과물/wm811k/colab_가져오기/`에 넣습니다.

수령한 세 시드 결과는 ZIP 경로·분할 SHA-256·클래스 support·F1 집계와 표본
표준편차를 검증했습니다. Validation macro-F1은 0.8916 ± 0.0082, 고정 test
macro-F1은 0.8815 ± 0.0024였습니다. 시드 42 체크포인트 해시는 현재 배포 모델과
동일합니다. 클래스별 재현율은 더 변동하므로 동일 test의 초기값 안정성 근거로만
해석합니다. `결과물/wm811k/multiseed_results/검증결과.md`에 한계를 기록했고,
대시보드의 `다중 시드 재현성 검증`에서 시드별 점수와 클래스별 변동을 확인합니다.

선택 모델의 고정 test 예측은 `코드/wm811k/evaluate_wm811k_lot_generalization.py`로
held-out lot와 원본 웨이퍼 크기별 편차를 추가 감사합니다. lot를 재표집한 cluster
bootstrap 구간과 클래스별 lot 재현율을 기록하며, 개별 lot은 최대 25개뿐이므로
최저 lot 순위를 확정적인 공정 품질 순위로 해석하지 않습니다. 이 사후 분석은 모델이나
검토·OOD 임계값 선택에 사용하지 않습니다.

웨이퍼 크기별 차이는 `코드/wm811k/analyze_wm811k_geometry_effect.py`에서 결함
클래스 구성을 leave-one-geometry-out 방식으로 보정합니다. Validation에서 찾은
`26x30`, `43x42` 후보를 고정 test에서 확인하지만 이는 사전 등록된 기준이 아닌
탐색적 안전 규칙이며 현재 진단 정책에는 적용하지 않습니다.

Grad-CAM은 `코드/wm811k/evaluate_wm811k_gradcam_faithfulness.py`로 클래스별
기존 정성 예시 27개에서 추가 점검합니다(대표 샘플을 45개로 늘린 뒤에도 같은 27개만
사용하며, 다시 실행하면 커밋된 결과가 1×10⁻⁶ 이내로 재현됩니다). 활성 die attribution mass, 상위 10% 영역 제거와
동일 크기 무작위·하위 영역 제거의 신뢰도 변화, 분류기 head 무작위화 후 heatmap
상관을 기록합니다. 이는 전체 test 대표 평가나 실제 공정 개입, 물리적 원인·인과관계
증명이 아닙니다.

설명 방법 비교는 `코드/wm811k/evaluate_wm811k_xai_methods.py`로 실행하며
`노트북/WM811K_07_XAI_방법비교_Colab.ipynb`은 같은 스크립트를 Colab GPU에서 실행합니다.
고정된 test split에서 클래스별 최대 50개(Near-full은 21개 전부, 총 421개)를 seed 42로
추출하고 validation에서 고정한 temperature 1.04717을 적용합니다. 클래스 균형
표본이므로 실제 test 분포의 전체 평균으로 해석하지 않으며, 모델 선택이나 진단
임계값 조정에는 사용하지 않습니다.

- **과거 IG·Grad-CAM 비교**(`결과물/wm811k/xai_comparison_results/`,
  `colab_가져오기/xai_deployed_model_results.zip`): 0 기준 Integrated Gradients와
  Grad-CAM만 비교한 결과입니다. 현재 대시보드의 Gradient SHAP 검증이 아니며 화면에도
  "Gradient SHAP 검증 아님"으로 표시합니다.
- **Gradient SHAP 검증**(`결과물/wm811k/xai_gradient_shap_validation/`): 같은 421개에서
  세 방법을 비교했습니다. 배포 체크포인트와 검증을 통과한 로컬 전처리 배열로 CPU에서
  실행했고, 같은 표본에서 과거 IG·Grad-CAM 수치를 다시 계산해 Colab 결과와 비교했습니다
  (Grad-CAM 상위−무작위 +0.0049로 동일, IG +0.1737). Gradient SHAP의 원시 가산성은
  417/421개(99.0%)가 5% 허용오차를 통과했고(상대 잔차 중앙값 0.97%, 최대 15.2%),
  미통과 4개는 통과로 세지 않았습니다. Gradient SHAP 기준과 같은 방향인 결함 die→정상
  die 교란에서 상위 10% 교란의 예측 점수 하락이 무작위보다 컸던 웨이퍼는 98.6%였습니다
  (클래스 평균 상위−무작위 +0.4615, IG +0.1567, Grad-CAM +0.1828). 이 점검은 모델
  민감도 확인이며 실제 공정 개입이나 원인 증명이 아닙니다.
  `validate_wm811k_gradient_shap_results.py`가 원자료에서 잔차·통과 여부·집계를 다시
  계산해 확인합니다.

과거 IG·Grad-CAM 결과 ZIP은 Windows에서 CRLF 줄바꿈 분할표의 원시 SHA-256을
기록했습니다. 저장소 분할표는 LF이므로 원시 해시는 다릅니다. 검증기는 원시 해시 일치
(`exact_byte_hash_match`)와 줄바꿈만 정규화한 CSV 행·필드 일치
(`canonical_content_match`)를 따로 보고하며, 현재 결과는 "줄바꿈 차이는 있으나 논리적
내용은 일치"입니다. 기존 manifest의 해시는 바꾸지 않았고, 새로 만드는 결과에는
`assignments_sha256`과 `assignments_canonical_sha256`을 함께 기록합니다.

`코드/wm811k/analyze_wm811k_near_full.py`는 test의 Near-full 21개 전부를
별도로 점검합니다. 분류 정확도, 기존 신뢰도·OOD 검토 정책, 두 설명법의 국소 삭제
효과를 함께 보며 확산형 결함에 국소 10% 제거 지표가 불리한 점을 분리합니다.
이 사후 분석으로 모델이나 임계값을 변경하지 않습니다.

`노트북/WM811K_12_견고성_2단계분류_Colab.ipynb`은 확인된 약점인 4px 위치 이동,
1% 활성 die 누락, `Edge-Loc`·`Loc`·`Scratch`의 정상 오인을 직접 겨냥합니다. 현재와
같은 flat CNN, 견고성 증강 flat CNN, 정상/불량 판별 뒤 8개 불량 유형을 분류하는
2단계 CNN을 동일한 lot-aware validation과 seed 17·42·2026에서 비교합니다. 모든
seed의 macro-F1·균형 정확도·정상 재현율 보호 조건과 이동·누락 성능 향상을 함께
통과해야 challenger로 선택합니다. 이 단계에서는 test 추론과 배포 모델 교체를 하지
않으며, 선택 후보가 있을 때만 별도의 고정 test 최종 평가로 넘어갑니다.

첫 실행에서 견고성 증강 flat CNN은 약한 클래스 평균 재현율과 macro-F1을 높였지만
정상 재현율 보호 조건을 통과하지 못했습니다. 세 seed의 validation 확률에 하나의 공통
none logit bias를 적용하는 `코드/wm811k/calibrate_wm811k_robust_flat_bias.py`로
재검증한 결과 bias `1.155`가 49개 합격 후보 중 가장 높은 약한 클래스 재현율을
기록했습니다. Baseline 대비 평균 약한 클래스 재현율 +4.39%p, macro-F1 +2.22%p,
균형 정확도 +3.21%p이며 정상 재현율 차이는 -0.025%p였습니다. 아직 test는 사용하지
않았고 배포 모델도 변경하지 않았습니다. 다음 단계는
`노트북/WM811K_13_보정후_견고성재검증_Colab.ipynb`에서 같은 bias를 고정하고 4px
이동·1% 활성 die 누락 validation 스트레스를 다시 추론하는 것입니다.

재검증 결과 bias `1.155`는 clean과 die 누락 조건에서는 모든 보호 기준을 통과했고,
4px 이동에서도 평균 약한 클래스 재현율 +6.81%p, macro-F1 +2.77%p를 기록했습니다.
그러나 seed 42의 이동 조건에서 정상 재현율 차이가 -0.389%p로 허용 기준
(-0.2%p)을 넘어서 최종 견고성 게이트는 통과하지 못했습니다. 따라서 test는 아직
열지 않았고 기존 배포 모델을 유지합니다. 다음 단계인
`노트북/WM811K_14_스트레스공동보정_Colab.ipynb`은 clean·이동·누락 validation을
동시에 사용해 공통 none bias를 다시 선택합니다. 모든 조건과 seed의 보호 기준을
통과한 경우에만 단 한 번의 고정 test 평가로 넘어갑니다.

공동 보정에서 bias `0~2`의 401개 값을 탐색했지만 모든 보호 조건을 동시에 통과한
값은 없었습니다. bias를 높이면 정상 재현율은 회복되지만 일부 seed의 정확도 또는
균형 정확도 보호 조건과 충돌하므로, 단일 robust 모델의 후처리만으로는 부족하다고
판단했습니다. test와 배포 모델은 계속 잠긴 상태입니다.
`노트북/WM811K_15_견고성앙상블선택_Colab.ipynb`은 정상 판별이 안정적인 baseline과
약한 불량 클래스에 강한 robust-flat의 확률을 혼합하고, 세 validation 스트레스에서
공통 혼합 비율과 none bias를 선택합니다. 동일한 보호 조건을 통과할 때만 고정 test
최종 평가로 넘어갑니다.

앙상블 탐색에서는 1,105개 조합 중 442개가 모든 보호 조건을 통과했고,
`baseline 50% + robust-flat 50%`, none bias `0.1`을 최종 선택했습니다. Baseline 대비
평균 약한 클래스 재현율은 clean +5.44%p, 4px 이동 +6.30%p, die 누락 +6.18%p였고,
각 조건의 macro-F1·균형 정확도·정확도도 개선됐습니다. 모든 seed의 정상 재현율
저하가 -0.2%p 이내여서 validation 견고성 게이트를 통과했습니다.
`노트북/WM811K_16_앙상블고정Test_Colab.ipynb`은 이 설정을 변경하지 않고 현재 실제
배포 checkpoint와 test에서 단 한 번 비교합니다. 같은 출력 폴더에 결과가 있으면
추론을 반복하지 않으며, test 결과로 혼합 비율이나 bias를 다시 선택하지 않습니다.

고정 test에서 앙상블은 기존 배포 모델보다 정확도 +0.30%p, 균형 정확도 +3.57%p,
Macro-F1 +2.09%p, 약한 클래스 평균 재현율 +7.31%p를 기록했습니다. 다만 정상
재현율이 -0.280%p로 사전 허용 기준 -0.2%p를 0.080%p 초과하여 최종 배포 게이트는
통과하지 못했습니다. 정상 웨이퍼 21,061개에서 기존보다 59개를 더 불량으로
오판한 차이입니다. Test를 본 뒤 bias를 다시 조정하지 않았으며 기존 champion과
이에 연결된 불확실성·OOD 정책을 계속 운영합니다.

운영 단계에서는 `코드/wm811k/build_wm811k_monitoring_reference.py`가 test를 제외한
validation 24,703개의 예측 클래스와 OOD 점수로 배치 감시 기준을 고정합니다. 실제
업로드가 20개 이상이면 같은 크기의 validation 배치를 4,000회 모의한 99% 상한과
예측 클래스 구성, 검토·OOD 비율, severe OOD 비율, 상위 OOD 점수 비율을 비교합니다.
하나라도 벗어나면 대시보드는 배치 전체 자동판정을 보류합니다. 이 신호는 정답 없는
분포 감시이므로 성능 저하 확정이나 자동 재학습 명령으로 해석하지 않습니다.

## 주요 명령

```powershell
# 전처리 다시 실행
python .\코드\secom\preprocess_secom.py

# 저장된 두 모델로 진단
python .\코드\secom\diagnose_secom.py --input ".\데이터\SECOM 데이터셋\raw\secom.data"
```
