@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "ENV_PYTHON=%EMGFORCE_PYTHON%"
if defined ENV_PYTHON if exist "%ENV_PYTHON%" goto run

set "ENV_PYTHON=D:\miniconda\envs\emgforce\python.exe"
if exist "%ENV_PYTHON%" goto run
set "ENV_PYTHON=%USERPROFILE%\miniconda3\envs\emgforce\python.exe"
if exist "%ENV_PYTHON%" goto run
set "ENV_PYTHON=%USERPROFILE%\anaconda3\envs\emgforce\python.exe"
if exist "%ENV_PYTHON%" goto run

where conda >nul 2>nul
if not errorlevel 1 (
  conda run -n emgforce python main.py
  if errorlevel 1 pause
  exit /b %errorlevel%
)

echo [ERROR] Cannot find the emgforce Conda environment.
echo Create it with: conda env create -f environment.yml
echo Or set EMGFORCE_PYTHON to the environment's python.exe.
pause
exit /b 1

:run
set "PYTHONUTF8=1"
"%ENV_PYTHON%" main.py
if errorlevel 1 pause
exit /b %errorlevel%
