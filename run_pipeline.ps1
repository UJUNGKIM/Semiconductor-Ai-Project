[CmdletBinding()]
param(
    [ValidateSet("verify", "preprocess", "augment", "core", "full")]
    [string]$Mode = "verify"
)

$ErrorActionPreference = "Stop"
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONIOENCODING = "utf-8"
Set-Location -LiteralPath $PSScriptRoot

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$scriptDir = Join-Path $PSScriptRoot "코드\secom"
$rawData = Join-Path $PSScriptRoot "데이터\SECOM 데이터셋\raw\secom.data"
$rawLabels = Join-Path $PSScriptRoot "데이터\SECOM 데이터셋\raw\secom_labels.data"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "가상환경을 찾을 수 없습니다: $pythonPath"
}
if (-not (Test-Path -LiteralPath $rawData)) {
    throw "SECOM 센서 데이터를 찾을 수 없습니다: $rawData"
}
if (-not (Test-Path -LiteralPath $rawLabels)) {
    throw "SECOM 라벨 데이터를 찾을 수 없습니다: $rawLabels"
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$ScriptName
    )

    $scriptPath = Join-Path $scriptDir $ScriptName
    if (-not (Test-Path -LiteralPath $scriptPath)) {
        throw "실행 파일을 찾을 수 없습니다: $scriptPath"
    }

    Write-Host ""
    Write-Host "[$Label] $ScriptName" -ForegroundColor Cyan
    & $pythonPath $scriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "$ScriptName 실행 실패 (exit code: $LASTEXITCODE)"
    }
}

function Test-ProjectEnvironment {
    Write-Host "[검증] Python 및 주요 라이브러리 확인" -ForegroundColor Cyan
    & $pythonPath -c "import pandas, numpy, sklearn, xgboost, catboost, lightgbm, torch, shap, dice_ml, joblib, matplotlib, scipy, imblearn, streamlit, plotly, ctgan; print('라이브러리 import: 정상')"
    if ($LASTEXITCODE -ne 0) {
        throw "필수 라이브러리 import 실패"
    }

    & $pythonPath -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "패키지 의존성 검사 실패"
    }

    & $pythonPath -c "import sys; from pathlib import Path; p=Path.cwd(); sys.path.insert(0, str(p/'코드'/'secom')); from train_compare_models import load_data; X,y,_=load_data(p); assert X.shape==(1567,590); assert y.value_counts().to_dict()=={0:1463,1:104}; print(f'SECOM 데이터: {X.shape[0]}행 x {X.shape[1]}변수, 정상 {(y==0).sum()}건, 불량 {(y==1).sum()}건')"
    if ($LASTEXITCODE -ne 0) {
        throw "SECOM 데이터 검증 실패"
    }

    Write-Host "프로젝트 검증 완료" -ForegroundColor Green
}

Test-ProjectEnvironment

Write-Host ""
Write-Host "[검증] 자동 회귀 테스트 실행" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "run_tests.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "자동 회귀 테스트 실패"
}

if ($Mode -eq "verify") {
    Write-Host "빠른 검증만 실행했습니다. 전체 재학습: .\run_pipeline.ps1 -Mode full" -ForegroundColor Green
    exit 0
}

if ($Mode -eq "augment") {
    Invoke-PythonStep "SECOM DDPM 증강 강화" "strengthen_synthetic_augmentation.py"
    Write-Host "SECOM 증강 강화 실험 완료" -ForegroundColor Green
    exit 0
}

Invoke-PythonStep "1/26 전처리" "preprocess_secom.py"

if ($Mode -eq "preprocess") {
    Write-Host "전처리 완료" -ForegroundColor Green
    exit 0
}

Invoke-PythonStep "2/26 기본 모델 및 SMOTE 비교" "train_compare_models.py"

if ($Mode -eq "full") {
    Invoke-PythonStep "3/26 CTGAN·TVAE 생성형 증강 비교" "compare_synthetic_augmentation.py"
    Invoke-PythonStep "4/26 DDPM 생성형 증강·품질 검증" "strengthen_synthetic_augmentation.py"
}
else {
    Write-Host "[3~4/26] 생성형 증강 비교 생략(core 모드)" -ForegroundColor DarkGray
}

Invoke-PythonStep "5/26 가중치 모델 튜닝" "tune_weighted_models.py"
Invoke-PythonStep "6/26 CatBoost·XGBoost 최종 비교" "compare_catboost_xgboost.py"
Invoke-PythonStep "7/26 SHAP 설명가능성 분석" "explain_dual_models.py"
Invoke-PythonStep "8/26 OOD·What-if·불량 패턴 안전장치" "build_advanced_diagnostics.py"
Invoke-PythonStep "9/26 전문가 검토 안전정책 검증" "evaluate_safety_policy.py"
Invoke-PythonStep "10/26 미탐 예방 안전우선 정책" "build_uncertainty_rescue_policy.py"
Invoke-PythonStep "11/26 센서 오류 강건성 검증" "evaluate_sensor_robustness.py"
Invoke-PythonStep "12/26 배치 안전 차단기" "build_batch_safety_gate.py"
Invoke-PythonStep "13/26 운영 모니터링 데모" "build_monitoring_demo.py"
Invoke-PythonStep "14/26 실제 검수 피드백 감시 기준" "build_feedback_monitoring_reference.py"
Invoke-PythonStep "15/26 Champion-challenger 승격 심사 데모" "build_champion_challenger_demo.py"

if ($Mode -eq "full") {
    Invoke-PythonStep "16/26 시간순 검증·확률 보정" "temporal_calibration_evaluation.py"
    Invoke-PythonStep "17/26 시간 드리프트 분석" "analyze_temporal_drift.py"
    Invoke-PythonStep "18/26 드리프트 대응 재학습" "retrain_drift_aware_models.py"
    Invoke-PythonStep "19/26 Rolling-window 검증" "rolling_window_retraining.py"
}
else {
    Write-Host "[16~19/26] 시간순·드리프트 분석 생략(core 모드)" -ForegroundColor DarkGray
}

Invoke-PythonStep "20/26 변조 감지형 감사 원장" "build_audit_ledger_demo.py"
Invoke-PythonStep "21/26 SHAP 설명 신뢰성 감사" "evaluate_shap_reliability.py"
Invoke-PythonStep "22/26 모델 무결성·장애 복구 훈련" "build_model_failover_drill.py"
Invoke-PythonStep "23/26 데이터 계보·오염 탐지" "build_data_lineage.py"
Invoke-PythonStep "24/26 Windows 전이 의존성 잠금" "build_windows_dependency_lock.py"
Invoke-PythonStep "25/26 실행환경 재현성·SBOM" "build_environment_provenance.py"
Invoke-PythonStep "26/26 통합 릴리스 준비도" "build_release_readiness_report.py"

Write-Host ""
Write-Host "파이프라인 완료. 대시보드 실행: .\run_dashboard.ps1" -ForegroundColor Green
