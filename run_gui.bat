@echo off
chcp 65001 >nul
title Karaoke Studio Pro

echo.
echo  ============================================
echo   Karaoke Studio Pro - GUI
echo  ============================================
echo.

REM Use venv if it exists, otherwise system Python
if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else (
    set PYTHON=python
)

echo  מפעיל ממשק Gradio...
echo  פתח דפדפן בכתובת: http://localhost:7860
echo.

%PYTHON% gui\gradio_app.py %*
pause
