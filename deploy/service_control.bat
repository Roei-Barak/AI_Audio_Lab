@echo off
:: בקרת שירות — הרץ כ-Administrator
set SERVICE_NAME=KaraokeStudio

echo.
echo  [1] הפעל שירות
echo  [2] עצור שירות
echo  [3] הפעל מחדש
echo  [4] סטטוס
echo  [5] הצג לוג
echo  [0] יציאה
echo.
set /p choice=בחר:

if "%choice%"=="1" net start %SERVICE_NAME%
if "%choice%"=="2" net stop %SERVICE_NAME%
if "%choice%"=="3" net stop %SERVICE_NAME% && net start %SERVICE_NAME%
if "%choice%"=="4" sc query %SERVICE_NAME%
if "%choice%"=="5" type "%~dp0service.log" | more
if "%choice%"=="0" exit /b

pause
