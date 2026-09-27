$ErrorActionPreference = "Stop"
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONIOENCODING = "utf-8"
Set-Location -LiteralPath $PSScriptRoot

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$testDir = Join-Path $PSScriptRoot "테스트"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "가상환경을 찾을 수 없습니다: $pythonPath"
}
if (-not (Test-Path -LiteralPath $testDir)) {
    throw "테스트 폴더를 찾을 수 없습니다: $testDir"
}

& $pythonPath -m unittest discover -s $testDir -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) {
    throw "자동 테스트 실패 (exit code: $LASTEXITCODE)"
}

Write-Host "자동 테스트를 모두 통과했습니다." -ForegroundColor Green
