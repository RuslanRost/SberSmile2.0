@echo off
setlocal
cd /d "%~dp0"

where winget >nul 2>nul
if errorlevel 1 goto check_tools

where git >nul 2>nul
if errorlevel 1 call :install_git

py -3.12 -V >nul 2>nul
if errorlevel 1 call :install_python

:check_tools
where git >nul 2>nul
if errorlevel 1 goto run
py -3.12 -V >nul 2>nul
if errorlevel 1 goto run

git fetch origin main >nul 2>nul
if errorlevel 1 goto run

for /f %%i in ('git rev-parse HEAD') do set LOCAL=%%i
for /f %%i in ('git rev-parse origin/main') do set REMOTE=%%i

if "%LOCAL%"=="%REMOTE%" goto run

git diff --quiet
if errorlevel 1 goto run
git diff --cached --quiet
if errorlevel 1 goto run

git pull origin main

:run
call ".venv312\Scripts\Activate.bat"
python main.py
endlocal
goto :eof

:install_git
powershell -NoProfile -Command "Start-Process winget -ArgumentList 'install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements' -Verb RunAs -Wait"
goto :eof

:install_python
powershell -NoProfile -Command "Start-Process winget -ArgumentList 'install --id Python.Python.3.12 -e --source winget --accept-source-agreements --accept-package-agreements' -Verb RunAs -Wait"
goto :eof
