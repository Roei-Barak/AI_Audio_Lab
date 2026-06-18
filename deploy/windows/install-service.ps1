# KaraokeStudio — Windows Service Installer
# Installs the Python backend as a Windows service using NSSM.
# Run as Administrator.

#Requires -RunAsAdministrator
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Param(
    [string]$InstallDir  = "C:\KaraokeStudio",
    [string]$Port        = "8000",
    [string]$Domain      = "",           # e.g. "karaoke.example.com" — leave empty for LAN-only
    [string]$AuthMode    = "required",   # "required" or "none"
    [string]$JwtSecret   = "",           # leave empty to auto-generate
    [string]$VenvDir     = ""            # leave empty to create inside InstallDir
)

function Write-Step { param($msg) Write-Host "[INSTALL] $msg" -ForegroundColor Cyan  }
function Write-Ok   { param($msg) Write-Host "[  OK   ] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[ WARN  ] $msg" -ForegroundColor Yellow }
function Fail       { param($msg) Write-Host "[ ERROR ] $msg" -ForegroundColor Red; exit 1 }

$ScriptDir  = $PSScriptRoot
$RepoRoot   = Split-Path (Split-Path $ScriptDir -Parent) -Parent

Write-Step "Checking prerequisites..."

# ── Python ──────────────────────────────────────────────────────────────────
$Python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.([89]|1[0-9])") { $Python = $cmd; break }
    } catch {}
}
if (-not $Python) { Fail "Python 3.9+ not found. Install from https://python.org" }
Write-Ok "Python: $(& $Python --version)"

# ── NSSM ────────────────────────────────────────────────────────────────────
$Nssm = $null
foreach ($path in @("nssm", "C:\nssm\win64\nssm.exe", "C:\tools\nssm\nssm.exe")) {
    if (Get-Command $path -ErrorAction SilentlyContinue) { $Nssm = $path; break }
}
if (-not $Nssm) {
    Write-Warn "NSSM not found. Downloading to $env:TEMP\nssm..."
    $NssmZip = "$env:TEMP\nssm.zip"
    $NssmDir = "$env:TEMP\nssm"
    Invoke-WebRequest "https://nssm.cc/release/nssm-2.24.zip" -OutFile $NssmZip
    Expand-Archive $NssmZip -DestinationPath $NssmDir -Force
    $Nssm = (Get-ChildItem "$NssmDir" -Recurse -Filter "nssm.exe" |
             Where-Object { $_.FullName -match "win64" } | Select-Object -First 1).FullName
    if (-not $Nssm) { Fail "Could not extract NSSM" }
    Write-Ok "NSSM downloaded: $Nssm"
}
Write-Ok "NSSM: $Nssm"

# ── Copy project files ───────────────────────────────────────────────────────
Write-Step "Copying project to $InstallDir..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

$Exclude = @(".git", "__pycache__", ".venv", "venv", "node_modules",
             "KaraokeStudio", "installer", "deploy")
foreach ($item in (Get-ChildItem $RepoRoot)) {
    if ($Exclude -contains $item.Name) { continue }
    Copy-Item $item.FullName -Destination $InstallDir -Recurse -Force
}
Write-Ok "Files copied"

# ── Python venv + dependencies ───────────────────────────────────────────────
if (-not $VenvDir) { $VenvDir = Join-Path $InstallDir "venv" }

Write-Step "Creating Python venv at $VenvDir..."
& $Python -m venv $VenvDir
$PipExe = Join-Path $VenvDir "Scripts\pip.exe"
& $PipExe install --quiet -r (Join-Path $InstallDir "requirements.txt")
Write-Ok "Python dependencies installed"

# ── JWT secret ───────────────────────────────────────────────────────────────
if (-not $JwtSecret) {
    $JwtSecret = [System.Convert]::ToBase64String(
        [System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
    Write-Warn "Generated JWT secret. Save this: $JwtSecret"
}

# ── Environment file ─────────────────────────────────────────────────────────
$EnvFile = Join-Path $InstallDir ".env"
@"
KARAOKE_AUTH_MODE=$AuthMode
KARAOKE_JWT_SECRET=$JwtSecret
KARAOKE_PORT=$Port
KARAOKE_OUTPUT_DIR=$InstallDir\Karaoke_Output
"@ | Set-Content $EnvFile
Write-Ok ".env written to $EnvFile"

# ── NSSM service ─────────────────────────────────────────────────────────────
$ServiceName = "KaraokeStudio"
$PythonExe   = Join-Path $VenvDir "Scripts\python.exe"
$ServerScript = Join-Path $InstallDir "api\server.py"

Write-Step "Installing Windows service '$ServiceName'..."

# Remove existing service if present
$existing = & $Nssm status $ServiceName 2>&1
if ($existing -notmatch "Can't open") {
    Write-Warn "Removing existing service..."
    & $Nssm stop    $ServiceName 2>&1 | Out-Null
    & $Nssm remove  $ServiceName confirm 2>&1 | Out-Null
}

& $Nssm install  $ServiceName $PythonExe
& $Nssm set      $ServiceName AppParameters   "`"$ServerScript`" --port $Port"
& $Nssm set      $ServiceName AppDirectory    $InstallDir
& $Nssm set      $ServiceName AppEnvironmentExtra `
    "KARAOKE_AUTH_MODE=$AuthMode" `
    "KARAOKE_JWT_SECRET=$JwtSecret" `
    "KARAOKE_PORT=$Port" `
    "KARAOKE_OUTPUT_DIR=$InstallDir\Karaoke_Output"
& $Nssm set      $ServiceName DisplayName     "KaraokeStudio Backend"
& $Nssm set      $ServiceName Description     "KaraokeStudio AI audio processing backend"
& $Nssm set      $ServiceName Start           SERVICE_AUTO_START
& $Nssm set      $ServiceName AppStdout       "$InstallDir\logs\service.log"
& $Nssm set      $ServiceName AppStderr       "$InstallDir\logs\service-err.log"
& $Nssm set      $ServiceName AppRotateFiles  1
& $Nssm set      $ServiceName AppRotateBytes  10485760

New-Item -ItemType Directory -Force -Path "$InstallDir\logs" | Out-Null
& $Nssm start $ServiceName
Write-Ok "Service '$ServiceName' installed and started"

# ── Optional: Caddy for HTTPS ────────────────────────────────────────────────
if ($Domain) {
    $CaddyExe = $null
    foreach ($path in @("caddy", "C:\caddy\caddy.exe")) {
        if (Get-Command $path -ErrorAction SilentlyContinue) { $CaddyExe = $path; break }
    }
    if (-not $CaddyExe) {
        Write-Warn "Caddy not found. Install from https://caddyserver.com/download"
        Write-Warn "Then run: caddy run --config (Join-Path $InstallDir 'Caddyfile')"
    } else {
        $Caddyfile = Join-Path $InstallDir "Caddyfile"
        @"
$Domain {
    encode gzip
    reverse_proxy localhost:$Port
}
"@ | Set-Content $Caddyfile
        Write-Ok "Caddyfile written. Starting Caddy service..."
        & $Nssm install  CaddyKaraoke $CaddyExe
        & $Nssm set      CaddyKaraoke AppParameters "run --config `"$Caddyfile`""
        & $Nssm set      CaddyKaraoke Start         SERVICE_AUTO_START
        & $Nssm start    CaddyKaraoke
        Write-Ok "Caddy service installed for $Domain"
    }
}

# ── First-run admin ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor White
Write-Ok   "Installation complete!"
Write-Host "Backend URL : http://localhost:$Port" -ForegroundColor White
if ($Domain) {
    Write-Host "Public URL  : https://$Domain" -ForegroundColor White
}
Write-Host ""
Write-Host "Create the first admin user:" -ForegroundColor Yellow
Write-Host "  cd `"$InstallDir`"" -ForegroundColor Gray
Write-Host "  .\venv\Scripts\python.exe api\server.py create-user admin <password> --admin" -ForegroundColor Gray
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor White
