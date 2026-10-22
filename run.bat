@echo off
setlocal
cd /d "%~dp0"
if exist "finsecure-ai-v2\run.py" (
    cd finsecure-ai-v2
)
python run.py %*
