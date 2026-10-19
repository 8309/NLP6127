@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run SETUP.cmd first. Python 3.12 and internet access are needed for dependency installation.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run_teacher.py
pause
