@echo off
:: =============================================
::   AI_Audio_Lab - התקנת Windows Service
::   חובה: הרץ כ-Administrator (Right Click → Run as administrator)
:: =============================================

:: בדיקת הרשאות Admin
net session >nul 2>&1
if errorlevel 1 (
    echo.
    echo  שגיאה: חייב הרשאות Administrator!
    echo  לחץ ימני על הקובץ → Run as administrator
    echo.
    pause & exit /b 1
)

cd /d "%~dp0.."
set APP_DIR=%CD%
set PYTHON=%APP_DIR%\venv\Scripts\python.exe
set SERVICE_NAME=KaraokeStudio
set PORT=7860

:: =============================================
::   הגדרות — ערוך כאן לפני ההרצה
:: =============================================
set USERNAME_AUTH=admin
set PASSWORD_AUTH=changeme123
:: HF_TOKEN נדרש רק לאיבריט-AI (Hugging Face token)
set HF_TOKEN=
:: =============================================

echo.
echo [AI_Audio_Lab] מתקין שירות Windows...

:: --- הורדת NSSM ---
if not exist "%APP_DIR%\deploy\nssm.exe" (
    echo [1/4] מוריד NSSM ^(מנהל שירותים^)...
    powershell -NoProfile -Command ^
        "Invoke-WebRequest 'https://nssm.cc/release/nssm-2.24.zip' -OutFile '$env:TEMP\nssm.zip' -UseBasicParsing; ^
         Expand-Archive '$env:TEMP\nssm.zip' -DestinationPath '$env:TEMP\nssm_extract' -Force; ^
         Copy-Item '$env:TEMP\nssm_extract\nssm-2.24\win64\nssm.exe' '%APP_DIR%\deploy\nssm.exe'"
    if errorlevel 1 ( echo שגיאה בהורדת NSSM & pause & exit /b 1 )
) else (
    echo [1/4] NSSM כבר קיים.
)

:: --- הסרת שירות קודם אם קיים ---
echo [2/4] מנקה שירות ישן ^(אם קיים^)...
"%APP_DIR%\deploy\nssm.exe" stop %SERVICE_NAME% >nul 2>&1
"%APP_DIR%\deploy\nssm.exe" remove %SERVICE_NAME% confirm >nul 2>&1

:: --- התקנת השירות ---
echo [3/4] מתקין שירות %SERVICE_NAME%...
"%APP_DIR%\deploy\nssm.exe" install %SERVICE_NAME% "%PYTHON%"
"%APP_DIR%\deploy\nssm.exe" set %SERVICE_NAME% AppParameters "main_web.py --host 0.0.0.0 --port %PORT% --no-browser --user %USERNAME_AUTH% --password %PASSWORD_AUTH%"
"%APP_DIR%\deploy\nssm.exe" set %SERVICE_NAME% AppDirectory "%APP_DIR%"
"%APP_DIR%\deploy\nssm.exe" set %SERVICE_NAME% Description "AI_Audio_Lab Karaoke Studio Web Server"
"%APP_DIR%\deploy\nssm.exe" set %SERVICE_NAME% Start SERVICE_AUTO_START
"%APP_DIR%\deploy\nssm.exe" set %SERVICE_NAME% AppStdout "%APP_DIR%\deploy\service.log"
"%APP_DIR%\deploy\nssm.exe" set %SERVICE_NAME% AppStderr "%APP_DIR%\deploy\service_err.log"
"%APP_DIR%\deploy\nssm.exe" set %SERVICE_NAME% AppRotateFiles 1
"%APP_DIR%\deploy\nssm.exe" set %SERVICE_NAME% AppRotateBytes 10485760

:: הגדרת HF_TOKEN אם קיים
if not "%HF_TOKEN%"=="" (
    "%APP_DIR%\deploy\nssm.exe" set %SERVICE_NAME% AppEnvironmentExtra "HF_TOKEN=%HF_TOKEN%"
)

:: --- פתיחת פורט בחומת אש ---
echo [4/4] פותח פורט %PORT% בחומת אש...
netsh advfirewall firewall delete rule name="KaraokeStudio" >nul 2>&1
netsh advfirewall firewall add rule name="KaraokeStudio" dir=in action=allow protocol=TCP localport=%PORT%

:: --- הפעלת השירות ---
net start %SERVICE_NAME%

echo.
echo ================================================
echo   השירות הותקן והופעל!
echo.
echo   גישה מהאינטרנט: http://YOUR_PUBLIC_IP:%PORT%
echo   גישה מקומית:    http://localhost:%PORT%
echo.
echo   שם משתמש: %USERNAME_AUTH%
echo   סיסמה:    %PASSWORD_AUTH%
echo.
echo   לוגים: deploy\service.log
echo.
echo   פקודות שימושיות:
echo     net start %SERVICE_NAME%   -- הפעל
echo     net stop  %SERVICE_NAME%   -- עצור
echo     sc query  %SERVICE_NAME%   -- סטטוס
echo ================================================
pause
