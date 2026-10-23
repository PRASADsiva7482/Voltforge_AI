@echo off
setlocal
cd /d "%~dp0"
python bootstrap.py --verify %*
exit /b %errorlevel%
