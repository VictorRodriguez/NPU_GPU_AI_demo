$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Install Python 3.11 x64 from python.org, then retry." }
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& .\.venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency compatibility check failed." }
& .\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 requirements-installed.txt
Write-Host "Ready. Download models: .\.venv\Scripts\python.exe download_models.py"
