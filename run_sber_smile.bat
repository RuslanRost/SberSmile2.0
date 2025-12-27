@echo off
setlocal
cd /d "%~dp0"
call ".venv312\Scripts\Activate.bat"
python main.py
endlocal
