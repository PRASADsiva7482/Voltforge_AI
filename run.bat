@echo off
setlocal
cd /d "%~dp0"
echo ===================================================
echo  VoltForge AI Microservice Launcher (Port 2002)
echo ===================================================
echo.

if exist ".\.venv\Scripts\python.exe" (
    echo [✓] Starting VoltForge AI using Virtual Environment Python 3.12...
    ".\.venv\Scripts\python.exe" main.py
) else (
    echo [!] Warning: .venv not found. Running with default python...
    python main.py
)

pause
