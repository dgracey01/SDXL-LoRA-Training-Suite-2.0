@echo off
title Lora Training Suite 2.0 - Installer

echo ================================================================================
echo   Lora Training Suite v2.0 - Installer
echo   Designed by: Zero  ^|  Built by: Jarvis
echo ================================================================================
echo.

set "SUITE_DIR=%~dp0"

:: -- Check Python (try python, then py launcher) --
echo [CHECKING] Python...
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python"
    goto :python_ok
)
py --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py"
    goto :python_ok
)

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

:python_ok
echo [OK] Python found:
%PYTHON% --version
echo.

:: ============================================================
:: Ask GPU choice FIRST before creating venv or installing
:: ============================================================
echo ================================================================================
echo   GPU Acceleration  (WD14 tagger + JoyCaption)
echo ================================================================================
echo.
echo   [1] NVIDIA GPU   CUDA 12.1 - recommended if you have an NVIDIA card
echo   [2] AMD GPU      ROCm (AMD RX 5000+ on Windows via native ROCm PyTorch)
echo   [3] AMD GPU      DirectML (simpler alternative for AMD/Intel)
echo   [4] CPU only     No GPU - slow for auto-tagging but fully functional
echo   [5] Skip         Install manually later
echo.
set /p GPU_CHOICE="   Enter choice (1/2/3/4/5): "
echo.

:: -- Create virtual environment --
echo ================================================================================
echo   [1/3] Creating virtual environment
echo ================================================================================

if not exist "%SUITE_DIR%.venv\Scripts\python.exe" (
    echo Creating .venv...
    %PYTHON% -m venv "%SUITE_DIR%.venv"
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

:: -- Upgrade pip --
echo ================================================================================
echo   [2/3] Upgrading pip
echo ================================================================================
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
echo [OK] pip up to date.
echo.

:: -- Install PyTorch FIRST with correct backend --
echo ================================================================================
echo   [3/3] Installing packages
echo ================================================================================
echo.

if "%GPU_CHOICE%"=="1" goto :nvidia
if "%GPU_CHOICE%"=="2" goto :rocm
if "%GPU_CHOICE%"=="3" goto :directml
if "%GPU_CHOICE%"=="4" goto :cpu
goto :skip_gpu

:nvidia
echo Installing PyTorch CUDA 12.1...
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 ( echo [ERROR] NVIDIA torch install failed. ) else ( echo [OK] PyTorch CUDA 12.1 installed. )
echo Installing onnxruntime (CPU - WD14 tagger)...
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet onnxruntime
if errorlevel 1 ( echo [ERROR] onnxruntime failed. ) else ( echo [OK] onnxruntime installed. JoyCaption uses GPU via PyTorch. )
echo.
goto :core

:rocm
echo Installing PyTorch ROCm 6.2...
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
if errorlevel 1 ( echo [ERROR] ROCm torch install failed. ) else ( echo [OK] PyTorch ROCm installed. )
echo Installing onnxruntime (CPU - WD14 tagger)...
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet onnxruntime
if errorlevel 1 ( echo [ERROR] onnxruntime failed. ) else ( echo [OK] onnxruntime installed. )
echo.
goto :core

:directml
echo Installing onnxruntime-directml...
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet onnxruntime-directml
if errorlevel 1 ( echo [ERROR] DirectML install failed. ) else ( echo [OK] onnxruntime-directml installed. )
echo.
goto :core

:cpu
echo Installing PyTorch CPU...
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 ( echo [ERROR] CPU torch install failed. ) else ( echo [OK] PyTorch CPU installed. )
echo Installing onnxruntime...
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet onnxruntime
if errorlevel 1 ( echo [ERROR] onnxruntime failed. ) else ( echo [OK] onnxruntime installed. )
echo.
goto :core

:skip_gpu
echo Skipping GPU setup.
echo Installing onnxruntime CPU fallback...
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet onnxruntime
echo.

:core
echo Installing core packages...
echo This may take several minutes on first install.
echo.
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet ^
    PySide6 ^
    Pillow ^
    numpy ^
    opencv-python ^
    PyYAML ^
    huggingface-hub ^
    "transformers>=4.47" ^
    "tokenizers>=0.20" ^
    einops ^
    kornia ^
    timm ^
    realesrgan ^
    safetensors ^
    accelerate ^
    insightface ^
    gfpgan ^
    tiktoken ^
    sentencepiece

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

:done
echo ================================================================================
echo   Installation complete!
echo.
echo   Run the suite:  run.bat
echo ================================================================================
echo.
pause
