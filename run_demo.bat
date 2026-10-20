@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.ps1 first. See README.md.
  pause
  exit /b 1
)
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set DO_NOT_TRACK=1
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1
pause
