@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    python setup_teacher.py
) else (
    py -3.12 setup_teacher.py
)
if errorlevel 1 (
    echo Setup failed. Install Python 3.12 and follow README_TEACHER.md.
    pause
    exit /b 1
)
pause
