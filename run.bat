@echo off
setlocal enabledelayedexpansion
title Lora Training Suite 2.0
cd /d "%~dp0"

:: Check that the venv exists
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo [ERROR] Virtual environment not found.
    echo         Please run INSTALL.bat first.
    echo.
    pause
    exit /b 1
)

:: ── Git update check (best-effort; never blocks launch) ─────────────────────
::  - Requires git on PATH and this folder to be a git clone.
::  - skip-worktree tells git to ignore LOCAL edits to runtime-written configs,
::    so they neither get pushed nor block a pull (repo keeps its baseline copy).
::  - Updates are applied with --ff-only, which NEVER overwrites local changes:
::    if it can't cleanly fast-forward it warns and launches the current version.
where git >nul 2>&1
if not errorlevel 1 (
  if exist "%~dp0.git" (
    git update-index --skip-worktree health/health_config.json merge/merge_config.json >nul 2>&1

    echo Checking for updates...
    git fetch --quiet
    set "BEHIND="
    for /f %%i in ('git rev-list HEAD..@{u} --count 2^>nul') do set "BEHIND=%%i"
    if defined BEHIND (
      if not "!BEHIND!"=="0" (
        echo Update available: !BEHIND! commit^(s^). Applying...
        git pull --ff-only
        if errorlevel 1 (
          echo.
          echo [WARN] Could not auto-update ^(local changes or diverged history^).
          echo        Launching current version.
          timeout /t 3 >nul
        ) else (
          echo Update complete.
        )
      ) else (
        echo Already up to date.
      )
    )
  )
)

:: MIOpen conv kernel-find FAST mode: avoids the hires-fix 2.0x (~2048^2 conv) exhaustive-search stall
:: on this AMD ROCm-Windows box (a 2.0x render hung 25s+ on one upsampling conv, 2026-08-02). Inherited
:: by the launched pythonw + its children. Also set in main.py as the launch-method-independent catch-all.
set "MIOPEN_FIND_MODE=FAST"

:: Launch without a console window
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0main.py"
endlocal
