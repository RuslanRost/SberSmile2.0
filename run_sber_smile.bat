@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -Command "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass" >nul 2>nul

set WINGET_OK=1
where winget >nul 2>nul
if errorlevel 1 (
  set WINGET_OK=0
  echo Winget not found. Skipping auto-install.
)

where git >nul 2>nul
if errorlevel 1 call :install_git

py -3.12 -V >nul 2>nul
if errorlevel 1 call :install_python

:check_tools
where git >nul 2>nul
if errorlevel 1 (
  if "%WINGET_OK%"=="0" (
    echo Git not found and winget is unavailable. Install Git manually.
    pause
  )
  echo Git not found. Continuing without update.
  goto run
)
py -3.12 -V >nul 2>nul
if errorlevel 1 (
  if "%WINGET_OK%"=="0" (
    echo Python 3.12 not found and winget is unavailable. Install Python manually.
    pause
  )
  echo Python 3.12 not found. Continuing without update.
  goto run
)

if not exist ".git" (
  echo No .git found. Initializing from remote...
  git init
  git remote add origin https://github.com/RuslanRost/SberSmile2.0
  git fetch origin main
  git reset --hard origin/main
)

git fetch origin main
if errorlevel 1 (
  echo Git fetch failed. Continuing without update.
  goto run
)

for /f %%i in ('git rev-parse HEAD') do set LOCAL=%%i
for /f %%i in ('git rev-parse origin/main') do set REMOTE=%%i

if "%LOCAL%"=="%REMOTE%" goto run

git diff --quiet
if errorlevel 1 (
  echo Local changes detected. Skipping pull.
  goto run
)
git diff --cached --quiet
if errorlevel 1 (
  echo Staged changes detected. Skipping pull.
  goto run
)

git pull origin main

:run
if not exist ".venv312\Scripts\python.exe" (
  call :create_venv
) else (
  ".venv312\Scripts\python.exe" -V >nul 2>nul
  if errorlevel 1 call :recreate_venv
)

if not exist ".venv312\Scripts\python.exe" (
  echo Venv creation failed.
  pause
  goto :eof
)

call :install_deps
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  goto :eof
)

".venv312\Scripts\python.exe" main.py
if errorlevel 1 (
  echo Script exited with error.
  pause
)
endlocal
goto :eof

:install_git
powershell -NoProfile -Command "Start-Process winget -ArgumentList 'install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements' -Verb RunAs -Wait"
if errorlevel 1 (
  echo Git install failed.
  pause
)
goto :eof

:install_python
powershell -NoProfile -Command "Start-Process winget -ArgumentList 'install --id Python.Python.3.12 -e --source winget --accept-source-agreements --accept-package-agreements' -Verb RunAs -Wait"
if errorlevel 1 (
  echo Python install failed.
  pause
)
goto :eof

:create_venv
py -3.12 -m venv .venv312
goto :eof

:recreate_venv
echo Recreating venv...
if exist ".venv312" rmdir /s /q ".venv312"
py -3.12 -m venv .venv312
goto :eof

:install_deps
".venv312\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
".venv312\Scripts\python.exe" -m pip install opencv-python mediapipe numpy tensorflow
goto :eof
