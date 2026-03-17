@echo off
chcp 65001 >nul
title Karaoke Studio Pro - API Server

echo.
echo  ============================================
echo   Karaoke Studio Pro - API Server
echo  ============================================
echo.

if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else (
    set PYTHON=python
)

echo  מפעיל שרת FastAPI...
echo  API:  http://localhost:8000
echo  Docs: http://localhost:8000/docs
echo.

%PYTHON% api\server.py %*
pause
