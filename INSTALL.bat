@echo off
chcp 65001 >nul
title Lora Training Suite 2.0 - Installer
color 0A

echo ================================================================================
echo   Lora Training Suite v2.0 - Installer
echo   Designed by: Zero  ^|  Built by: Jarvis
echo ================================================================================
echo.

set "SUITE_DIR=%~dp0"

:: -- Check Python --
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

:: -- Create virtual environment --
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

:: -- Upgrade pip --
echo ================================================================================
echo   [2/3] Upgrading pip
echo ================================================================================
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
echo [OK] pip up to date.
echo.

:: -- Install packages --
echo ================================================================================
echo   [3/3] Installing packages
echo ================================================================================
echo.
echo Installing PySide6, Pillow, numpy, transformers, and friends...
echo This may take several minutes on first install.
echo.
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet ^
    PySide6 ^
    Pillow ^
    numpy ^
    opencv-python ^
    PyYAML ^
    huggingface-hub ^
    "transformers==4.46.3" ^
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

:: -- GPU acceleration --
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

if "%GPU_CHOICE%"=="1" goto :nvidia
if "%GPU_CHOICE%"=="2" goto :rocm
if "%GPU_CHOICE%"=="3" goto :amd
if "%GPU_CHOICE%"=="4" goto :cpu
if "%GPU_CHOICE%"=="5" goto :done
echo Invalid choice - skipping GPU setup.
goto :done

:nvidia
echo Installing NVIDIA packages (onnxruntime-gpu, torch, torchvision)...
echo NOTE: Download is approximately 2-3 GB. This will take a while.
echo.
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet onnxruntime-gpu
if errorlevel 1 ( echo [ERROR] onnxruntime-gpu install failed. & goto :done )
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet ^
    torch torchvision ^
    --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 ( echo [ERROR] NVIDIA torch install failed. ) else ( echo [OK] NVIDIA packages installed. )
goto :done

:rocm
echo Installing AMD ROCm packages (native ROCm PyTorch)...
echo NOTE: Download is approximately 2-3 GB. This will take a while.
echo.
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet ^
    onnxruntime-gpu ^
    torch torchvision ^
    --index-url https://download.pytorch.org/whl/rocm6.2
if errorlevel 1 ( echo [ERROR] ROCm install failed. ) else ( echo [OK] ROCm packages installed. )
echo.
echo Launch with run.bat - no wrapper needed.
goto :done

:amd
echo Installing AMD packages (onnxruntime-directml)...
echo.
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet ^
    onnxruntime-directml
if errorlevel 1 ( echo [ERROR] AMD install failed. ) else ( echo [OK] AMD packages installed. )
goto :done

:cpu
echo Installing CPU packages (onnxruntime, torch, torchvision)...
echo NOTE: Download is approximately 2-3 GB. This will take a while.
echo.
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet onnxruntime
if errorlevel 1 ( echo [ERROR] onnxruntime install failed. & goto :done )
"%SUITE_DIR%.venv\Scripts\python.exe" -m pip install --quiet ^
    torch torchvision ^
    --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 ( echo [ERROR] CPU torch install failed. ) else ( echo [OK] CPU packages installed. )
goto :done

:done
echo.
echo ================================================================================
echo   Installation complete!
echo.
echo   Run the suite:  run.bat
echo ================================================================================
echo.
pause
