@echo off
rem MusicTool setup for Windows: creates .venv and installs everything.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
echo === MusicTool setup ===

rem Prefer the Python launcher with a known-good version, else fall back to python on PATH.
set "PY="
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo Python 3.10 or newer was not found.
  echo Install Python 3.12 from https://www.python.org/downloads/windows/
  echo and tick "Add python.exe to PATH" in the installer, then run this again.
  goto :fail
)
echo Using: %PY%

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"

echo Installing packages (first run downloads ~300 MB, this can take a few minutes)...
"%VPY%" -m pip install --upgrade pip || goto :fail
rem basic-pitch's own requirements pin TensorFlow; we only need its ONNX model.
"%VPY%" -m pip install --no-deps basic-pitch || goto :fail
"%VPY%" -m pip install -e . resampy mir_eval scikit-learn || goto :fail

echo.
echo Setup complete. Double-click start.bat to open MusicTool.
if not defined CI if not defined MUSICTOOL_NOPAUSE pause
exit /b 0

:fail
echo.
echo Setup failed - see the messages above.
if not defined CI if not defined MUSICTOOL_NOPAUSE pause
exit /b 1
