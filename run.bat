@echo off
setlocal
cd /d "%~dp0"
python bootstrap.py --run %*
exit /b %errorlevel%
