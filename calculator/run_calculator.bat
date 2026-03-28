@echo off
title LoRA Calculator — Lora Training Suite 2.0

if not exist "%~dp0..\\.venv\\Scripts\\pythonw.exe" (
    echo [ERROR] Virtual environment not found.
    echo         Please run INSTALL.bat from the suite root first.
    echo.
    pause
    exit /b 1
)

start "" "%~dp0..\\.venv\\Scripts\\pythonw.exe" "%~dp0run_calculator.py"
