@echo off
:: =============================================
::   AI_Audio_Lab - Setup (הרץ פעם אחת בלבד)
:: =============================================
:: דרישות מוקדמות:
::   - Python 3.10+ מותקן
::   - NVIDIA GPU Driver עדכני
::   - FFmpeg ב-PATH (https://ffmpeg.org/download.html)

cd /d "%~dp0.."
echo.
echo [AI_Audio_Lab] מתחיל התקנה...
echo ================================================

:: --- 1. venv ---
echo.
echo [1/5] יוצר סביבה וירטואלית...
python -m venv venv
if errorlevel 1 ( echo שגיאה: Python לא נמצא & pause & exit /b 1 )

:: --- 2. Upgrade pip ---
echo.
echo [2/5] מעדכן pip...
venv\Scripts\python -m pip install --upgrade pip --quiet

:: --- 3. PyTorch + CUDA ---
echo.
echo [3/5] מתקין PyTorch עם תמיכת CUDA 12.1...
echo      (זה יקח כמה דקות - ~2.5GB)
venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 ( echo שגיאה בהתקנת PyTorch & pause & exit /b 1 )

:: --- 4. Requirements ---
echo.
echo [4/5] מתקין שאר חבילות...
venv\Scripts\pip install -r requirements.txt --quiet
if errorlevel 1 ( echo שגיאה בהתקנת requirements & pause & exit /b 1 )

:: --- 5. Verify GPU ---
echo.
echo [5/5] בדיקת GPU...
venv\Scripts\python -c "import torch; cuda=torch.cuda.is_available(); print(f'CUDA: {\"YES - \" + torch.cuda.get_device_name(0) if cuda else \"NO (CPU only)\"}')"

echo.
echo ================================================
echo   ההתקנה הושלמה!
echo.
echo   הרצה מקומית:
echo     deploy\run_local.bat
echo.
echo   התקנת שירות (גישה מהאינטרנט):
echo     הרץ כ-Administrator: deploy\2_install_service.bat
echo ================================================
pause
