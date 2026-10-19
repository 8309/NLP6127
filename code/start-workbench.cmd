@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment missing. Run: uv sync --locked --group dev
  pause
  exit /b 1
)
echo Oil Evidence: http://127.0.0.1:8517/
echo Keep this window open while using the workbench. Ctrl+C stops the service.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8517 --server.headless true --browser.gatherUsageStats false
if errorlevel 1 (
  echo Startup failed. If port 8517 is in use, check the already-running workbench.
  pause
)
