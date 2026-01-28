@echo off
setlocal
echo ====================================================
echo 🎤 Karaoke Studio Pro - Universal Installer
echo ====================================================

:: 1. בדיקת פייתון
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python not found! Please install Python 3.10 or newer.
    pause
    exit /b
)

:: 2. יצירת סביבה וירטואלית
echo [*] Creating isolated environment (.venv)...
python -m venv .venv
call .venv\Scripts\activate

:: 3. שדרוג PIP
python -m pip install --upgrade pip

:: 4. זיהוי חומרה חכם (GPU vs CPU)
echo [*] Detecting hardware...
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo [+] NVIDIA GPU detected! Installing CUDA 12.1 version...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install onnxruntime-gpu
    pip install "audio-separator[gpu]"
) else (
    echo [-] No NVIDIA GPU detected. Installing CPU optimized version...
    pip install torch torchaudio
    pip install onnxruntime
    pip install "audio-separator"
)

:: 5. התקנת שאר הדרישות
echo [*] Installing additional dependencies...
pip install -r requirements.txt

echo ====================================================
echo ✅ Installation Complete!
echo To start the app, run 'run_app.bat'
echo ====================================================
pause
