@echo off
title Lora Training Suite 2.0

:: Check that the venv exists
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo [ERROR] Virtual environment not found.
    echo         Please run INSTALL.bat first.
    echo.
    pause
    exit /b 1
)

:: Launch without a console window
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0main.py"
