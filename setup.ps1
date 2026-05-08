# setup.ps1 — Windows PowerShell setup script for AI_Audio_Lab
# Run from the project root:  .\setup.ps1
# If blocked by execution policy: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

param(
    [switch]$CPU,        # Force CPU-only install (no CUDA)
    [switch]$NoPyTorch,  # Skip PyTorch (install later manually)
    [switch]$Dev         # Install dev tools too (pytest, etc.)
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Karaoke Studio Pro — Windows Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Python version check ──────────────────────────────────────────────────
Write-Host "[1/7] בודק Python..." -ForegroundColor Yellow
try {
    $pyver = python --version 2>&1
    Write-Host "  ✅ $pyver" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Python לא נמצא. הורד מ: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# ── 2. Create / activate venv ────────────────────────────────────────────────
Write-Host "[2/7] יוצר סביבה וירטואלית..." -ForegroundColor Yellow
$VenvPath = Join-Path $ProjectRoot "venv"

if (-not (Test-Path $VenvPath)) {
    python -m venv $VenvPath
    Write-Host "  ✅ venv נוצר ב: $VenvPath" -ForegroundColor Green
} else {
    Write-Host "  ℹ️  venv קיים, משתמש בו" -ForegroundColor Cyan
}

$PipExe   = Join-Path $VenvPath "Scripts\pip.exe"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
Write-Host "[3/7] מעדכן pip..." -ForegroundColor Yellow
& $PipExe install --upgrade pip --quiet
Write-Host "  ✅ pip מעודכן" -ForegroundColor Green

# ── 4. Install base dependencies (no heavy ML yet) ───────────────────────────
Write-Host "[4/7] מתקין תלויות בסיס..." -ForegroundColor Yellow
$BaseDeps = @(
    "yt-dlp",
    "imageio-ffmpeg",
    "gradio>=4.0",
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
    "pandas",
    "arabic-reshaper",
    "python-bidi",
    "pysrt",
    "soundfile",
    "scipy",
    "numpy",
    "librosa"
)
foreach ($pkg in $BaseDeps) {
    Write-Host "  📦 $pkg" -ForegroundColor DarkGray
    & $PipExe install $pkg --quiet
}
Write-Host "  ✅ תלויות בסיס הותקנו" -ForegroundColor Green

# ── 5. PyTorch ───────────────────────────────────────────────────────────────
if (-not $NoPyTorch) {
    Write-Host "[5/7] מתקין PyTorch..." -ForegroundColor Yellow

    # Detect CUDA
    $HasNvidia = $false
    try {
        $nvout = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($nvout) { $HasNvidia = $true }
    } catch {}

    if ($CPU -or -not $HasNvidia) {
        Write-Host "  💻 מתקין PyTorch (CPU בלבד)" -ForegroundColor Cyan
        & $PipExe install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    } else {
        Write-Host "  🎮 GPU זוהה: $nvout" -ForegroundColor Green
        Write-Host "  ⚡ מתקין PyTorch + CUDA 12.1" -ForegroundColor Cyan
        & $PipExe install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    }

    # transformers + accelerate
    & $PipExe install transformers accelerate --quiet
    Write-Host "  ✅ PyTorch + transformers הותקנו" -ForegroundColor Green
} else {
    Write-Host "[5/7] PyTorch — דולג (--NoPyTorch)" -ForegroundColor DarkGray
}

# ── 6. audio-separator ───────────────────────────────────────────────────────
Write-Host "[6/7] מתקין audio-separator..." -ForegroundColor Yellow
Write-Host "  ⚠️  זה עלול לקחת כמה דקות..." -ForegroundColor DarkYellow
try {
    if ($CPU) {
        & $PipExe install "audio-separator[cpu]" --quiet
    } else {
        & $PipExe install "audio-separator[gpu]" --quiet
    }
    Write-Host "  ✅ audio-separator הותקן" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  audio-separator נכשל — תוכל להתקין ידנית מאוחר יותר" -ForegroundColor DarkYellow
    Write-Host "       pip install audio-separator[cpu]" -ForegroundColor DarkGray
}

# ── Dev tools ─────────────────────────────────────────────────────────────────
if ($Dev) {
    Write-Host "🔧 מתקין כלי פיתוח..." -ForegroundColor Yellow
    & $PipExe install pytest black isort --quiet
    Write-Host "  ✅ כלי פיתוח הותקנו" -ForegroundColor Green
}

# ── 7. KaraokeStudio.WPF (.NET 8) ────────────────────────────────────────────
Write-Host "[7/7] בונה KaraokeStudio.WPF (.NET 8)..." -ForegroundColor Yellow
$SlnPath = Join-Path $ProjectRoot "KaraokeStudio\KaraokeStudio.sln"

if (-not (Test-Path $SlnPath)) {
    Write-Host "  ⚠️  פרויקט WPF לא נמצא — דולג" -ForegroundColor DarkYellow
} else {
    $HasDotnet = $false
    try {
        $dotnetVer = dotnet --version 2>&1
        if ($dotnetVer -match "^\d") { $HasDotnet = $true }
    } catch {}

    if (-not $HasDotnet) {
        Write-Host "  ⚠️  .NET SDK לא נמצא." -ForegroundColor DarkYellow
        Write-Host "      הורד מ: https://dotnet.microsoft.com/download/dotnet/8.0" -ForegroundColor DarkGray
        Write-Host "      (אופציונלי — Gradio/CLI/API יעבדו בלעדיו)" -ForegroundColor DarkGray
    } else {
        Write-Host "  📦 dotnet $dotnetVer זוהה" -ForegroundColor Green
        try {
            Push-Location (Split-Path $SlnPath -Parent)
            & dotnet restore $SlnPath 2>&1 | Out-Null
            & dotnet build   $SlnPath -c Release --nologo 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✅ KaraokeStudio.WPF נבנה בהצלחה" -ForegroundColor Green
            } else {
                Write-Host "  ⚠️  בנייה נכשלה — בדוק build log" -ForegroundColor DarkYellow
            }
        } catch {
            Write-Host "  ⚠️  שגיאה בבנייה: $_" -ForegroundColor DarkYellow
        } finally {
            Pop-Location
        }
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ✅ ההתקנה הושלמה!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "כדי להפעיל את המערכת:" -ForegroundColor White
Write-Host ""
Write-Host "  .\run_gui.bat                    ← ממשק Gradio בדפדפן" -ForegroundColor Yellow
Write-Host "  .\run_api.bat                    ← שרת FastAPI" -ForegroundColor Yellow
Write-Host "  .\run_cli.bat pipeline <url>     ← CLI מלא" -ForegroundColor Yellow
Write-Host "  .\run_wpf.bat                    ← Windows native GUI (WPF)" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # או הפעל ישירות:" -ForegroundColor DarkGray
Write-Host "  venv\Scripts\python.exe gui\gradio_app.py" -ForegroundColor DarkGray
Write-Host "  KaraokeStudio\KaraokeStudio.WPF\bin\Release\net8.0-windows\KaraokeStudio.exe" -ForegroundColor DarkGray
Write-Host ""
