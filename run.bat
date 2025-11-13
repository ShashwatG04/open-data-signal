@echo off
REM ================================
REM Open-Data Signals Project Runner
REM ================================

cd /d https://github.com/ShashwatG04/open-data-signal
echo Activating virtual environment...
call venv311\Scripts\activate

echo.
echo 🚀 Starting full pipeline...
python run.py

echo.
echo ✅ Process complete.
pause

