@echo off
:: הסרת השירות — הרץ כ-Administrator
net session >nul 2>&1
if errorlevel 1 ( echo חייב Administrator! & pause & exit /b 1 )

set SERVICE_NAME=KaraokeStudio
cd /d "%~dp0.."

echo עוצר ומסיר שירות %SERVICE_NAME%...
net stop %SERVICE_NAME% >nul 2>&1
"%~dp0nssm.exe" remove %SERVICE_NAME% confirm

netsh advfirewall firewall delete rule name="KaraokeStudio" >nul 2>&1

echo.
echo השירות הוסר בהצלחה.
pause
