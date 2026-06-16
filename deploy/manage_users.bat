@echo off
:: ניהול משתמשים מורשים
cd /d "%~dp0.."
echo.
echo [AI_Audio_Lab] ניהול משתמשים
echo.
venv\Scripts\python auth/manage_users.py %*
pause
