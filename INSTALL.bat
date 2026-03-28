@echo off
title Lora Training Suite 2.0 — Installer
color 0A

echo ================================================================================
echo   Lora Training Suite v2.0 — Installer
echo   Designed by: Zero  ^|  Built by: Jarvis
echo ================================================================================
echo.

set "SUITE_DIR=%~dp0"

:: ── Check Python ──────────────────────────────────────────────────────────────
echo [CHECKING] Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [NOT FOUND] Python is not installed or not on PATH.
    echo.
    echo Attempting automatic install via winget...
    winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo [ERROR] Automatic install failed.
        echo         Install Python manually: https://www.python.org/downloads/
        echo         IMPORTANT: Check "Add Python to PATH" during install.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Python installed. Please CLOSE and RE-RUN this installer.
    echo.
    pause
    exit /b 0
)
echo [OK] Python found:
python --version
echo.

:: ── Create virtual environment ─────────────────────────────────────────────
echo ================================================================================
echo   [1/3] Creating virtual environment
echo ================================================================================

if not exist "%SUITE_DIR%.venv\Scripts\python.exe" (
    echo Creating .venv...
    python -m venv "%SUITE_DIR%.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists, skipping.
)
echo.

:: ── Upgrade pip ───────────────────────────────────────────────────────────────
echo ================================================================================
echo   [2/3] Upgrading pip
echo ================================================================================
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet --disable-pip-version-check
echo [OK] pip up to date.
echo.

:: ── Install packages ──────────────────────────────────────────────────────────
echo ================================================================================
echo   [3/3] Installing packages
echo ================================================================================
echo.
echo Installing PySide6 (Qt 6 — bundled Chromium, ~150 MB)...
echo Note: PySide6 ^>=6.4 includes WebEngine — no separate package needed.
echo This may take several minutes on first install.
echo.
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install ^
    PySide6 ^
    Pillow ^
    numpy ^
    opencv-python ^
    PyYAML ^
    huggingface-hub ^
    transformers ^
    einops ^
    kornia ^
    timm ^
    realesrgan ^
    safetensors ^
    accelerate ^
    --quiet --disable-pip-version-check

if errorlevel 1 (
    echo.
    echo [ERROR] Package installation failed.
    echo         Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)
echo [OK] Core packages installed.
echo.

:: ── GPU acceleration (for WD14 tagger + JoyCaption) ───────────────────────────
echo ================================================================================
echo   GPU Acceleration  (WD14 tagger + JoyCaption)
echo ================================================================================
echo.
echo   [1] NVIDIA GPU   CUDA 12.1 - recommended if you have an NVIDIA card
echo   [2] AMD GPU      DirectML
echo   [3] CPU only     No GPU - slow for auto-tagging but fully functional
echo   [4] Skip         Install manually later
echo.
set /p GPU_CHOICE="   Enter choice (1/2/3/4): "
echo.

if "%GPU_CHOICE%"=="1" goto :nvidia
if "%GPU_CHOICE%"=="2" goto :amd
if "%GPU_CHOICE%"=="3" goto :cpu
if "%GPU_CHOICE%"=="4" goto :done
echo Invalid choice - skipping GPU setup.
goto :done

:nvidia
echo Installing NVIDIA packages (onnxruntime-gpu, torch, torchvision)...
echo NOTE: Download is approximately 2-3 GB. This will take a while.
echo.
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install ^
    onnxruntime-gpu ^
    torch torchvision ^
    --index-url https://download.pytorch.org/whl/cu121 ^
    --quiet --disable-pip-version-check
if errorlevel 1 ( echo [ERROR] NVIDIA install failed. ) else ( echo [OK] NVIDIA packages installed. )
goto :done

:amd
echo Installing AMD packages (onnxruntime-directml, torch-directml)...
echo.
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install ^
    onnxruntime-directml ^
    torch-directml ^
    --quiet --disable-pip-version-check
if errorlevel 1 ( echo [ERROR] AMD install failed. ) else ( echo [OK] AMD packages installed. )
goto :done

:cpu
echo Installing CPU packages (onnxruntime, torch, torchvision)...
echo NOTE: Download is approximately 2-3 GB. This will take a while.
echo.
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install ^
    onnxruntime ^
    torch torchvision ^
    --quiet --disable-pip-version-check
if errorlevel 1 ( echo [ERROR] CPU install failed. ) else ( echo [OK] CPU packages installed. )
goto :done

:done
echo.
echo ================================================================================
echo   Installation complete!
echo.
echo   Run the suite:  run.bat
echo.
echo   What's new in v2.0:
echo     - PySide6 + bundled Chromium (no Chrome/Edge dependency)
echo     - Tag Handler and Calculator are now embedded tabs (not separate windows)
echo     - Per-tab zoom via Qt compositor (no CSS hack)
echo     - Off-the-record browsing (no cookies on disk)
echo ================================================================================
echo.
pause
