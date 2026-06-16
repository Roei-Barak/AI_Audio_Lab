@echo off
:: הרצה מקומית — בלי שירות, עם דפדפן
cd /d "%~dp0.."
echo [AI_Audio_Lab] מפעיל ממשק Web מקומי...
venv\Scripts\python main_web.py --host 127.0.0.1 --port 7860
pause
