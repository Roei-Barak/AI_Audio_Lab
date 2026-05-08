@echo off
REM run_wpf.bat — Launch KaraokeStudio.WPF (Windows native GUI)
setlocal

set EXE=KaraokeStudio\KaraokeStudio.WPF\bin\Release\net8.0-windows\KaraokeStudio.exe

if not exist "%EXE%" (
    echo Building KaraokeStudio.WPF...
    pushd KaraokeStudio
    dotnet build KaraokeStudio.sln -c Release --nologo
    if errorlevel 1 (
        echo Build failed.
        popd
        exit /b 1
    )
    popd
)

start "" "%EXE%"
