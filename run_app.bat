@echo off
if not exist .venv (
    echo [!] Environment not found. Please run setup_all.bat first.
    pause
    exit /b
)
call .venv\Scripts\activate
echo 🚀 Starting Karaoke Studio...
python app.py
pause
