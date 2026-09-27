$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "가상환경을 찾을 수 없습니다: $pythonPath"
}

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
& $pythonPath -m streamlit run (Join-Path $PSScriptRoot "app.py") `
    --server.headless true `
    --server.address 127.0.0.1 `
    --server.port 8501 `
    --browser.gatherUsageStats false

