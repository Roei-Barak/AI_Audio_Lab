@echo off
chcp 65001 >nul

if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else (
    set PYTHON=python
)

REM Pass all arguments directly to cli/main.py
REM Examples:
REM   run_cli.bat pipeline "https://youtu.be/..."
REM   run_cli.bat download "https://youtu.be/..." --format mp4
REM   run_cli.bat batch songs.txt --lang he
REM   run_cli.bat lecture "https://youtu.be/..." --lang en

%PYTHON% cli\main.py %*
