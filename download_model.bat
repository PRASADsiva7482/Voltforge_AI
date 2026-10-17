@echo off
setlocal
cd /d "%~dp0"
echo ===================================================
echo  VoltForge AI - Clean Model Downloader & Setup
echo ===================================================
echo.

if exist ".\.venv\Scripts\python.exe" (
    echo [i] Using virtual environment Python: .\.venv\Scripts\python.exe
    ".\.venv\Scripts\python.exe" download_model.py %*
) else (
    echo [!] Virtual environment not found, falling back to system python...
    python download_model.py %*
)

echo.
pause
