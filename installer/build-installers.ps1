# Build KaraokeStudio installers
Set-StrictMode -Version Latest; $ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$repoRoot  = Split-Path $scriptDir -Parent
$wpfDir    = Join-Path $repoRoot "KaraokeStudio\KaraokeStudio.WPF"

function Write-Step { param($msg) Write-Host "[BUILD] $msg" -ForegroundColor Cyan  }
function Write-Ok   { param($msg) Write-Host "[ OK ] $msg"  -ForegroundColor Green }

# Build web SPA
$webDir = Join-Path $repoRoot "web"
if (Test-Path $webDir) {
    Write-Step "Building React web SPA..."
    Push-Location $webDir; npm ci --silent; npm run build; Pop-Location
    Write-Ok "Web SPA built → web/dist"
}

# dotnet publish
Write-Step "Publishing WPF (self-contained, win-x64)..."
dotnet publish $wpfDir -c Release -r win-x64 --self-contained true `
    -p:PublishSingleFile=true -p:EnableCompressionInSingleFile=true
Write-Ok "WPF published"

# ISCC
$iscc = "ISCC.exe"
if (-not (Get-Command $iscc -ErrorAction SilentlyContinue)) {
    $iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
}
if (Test-Path $iscc) {
    Push-Location $scriptDir
    & $iscc "KaraokeStudio-Client.iss"
    Pop-Location
    Write-Ok "Installer built → installer\dist\"
} else {
    Write-Host "[WARN] Inno Setup not found — skipping installer" -ForegroundColor Yellow
}
