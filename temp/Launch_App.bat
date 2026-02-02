@echo off
REM Karaoke Studio Pro Launcher
REM This batch file runs the app without showing the terminal

cd /d "%~dp0"
start "" python web_ui.py
exit
