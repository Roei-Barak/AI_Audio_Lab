@echo off
chcp 65001 >nul
title Karaoke Studio Pro - Desktop

echo.
echo  ============================================
echo   Karaoke Studio Pro - Desktop App
echo  ============================================
echo.

if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else (
    set PYTHON=python
)

echo  מפעיל אפליקציית Desktop...
echo  הדפדפן יפתח אוטומטית.
echo.

%PYTHON% gui\desktop_app.py %*
pause
