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

"%VPY%" -m pip install --upgrade pip || goto :fail
rem basic-pitch's own requirements pin TensorFlow; we only need its ONNX model.
"%VPY%" -m pip install --no-deps basic-pitch || goto :fail

rem NVIDIA GPU present (and not forced to CPU)? Install the neural models.
if defined MUSICTOOL_CPU goto :cpu
nvidia-smi >nul 2>&1 || goto :cpu

:gpu
echo NVIDIA GPU found - installing the neural models.
echo First run downloads about 3.5 GB, this can take a while...
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
"%VPY%" -m pip install torch torchaudio --index-url %TORCH_INDEX% || goto :fail
rem Pin the CUDA build so later packages can't swap in the CPU-only torch from PyPI.
set "CONSTRAINTS=%TEMP%\musictool-torch-constraints.txt"
"%VPY%" -c "import torch, torchaudio; print('torch==' + torch.__version__); print('torchaudio==' + torchaudio.__version__)" > "%CONSTRAINTS%" || goto :fail
rem onnxruntime and onnxruntime-gpu overwrite each other; keep only the GPU one.
"%VPY%" -m pip uninstall -y onnxruntime >nul 2>&1
"%VPY%" -m pip install -e ".[gpu]" resampy mir_eval scikit-learn -c "%CONSTRAINTS%" --extra-index-url %TORCH_INDEX% || goto :fail
"%VPY%" -c "import torch; print('CUDA available:', torch.cuda.is_available())"
goto :done

:cpu
echo No NVIDIA GPU - installing the CPU engines. Set MUSICTOOL_CPU=1 to force this.
"%VPY%" -m pip install -e ".[cpu]" resampy mir_eval scikit-learn || goto :fail

:done
echo.
echo Setup complete. Double-click start.bat to open MusicTool.
echo Models for stem separation download on first use, about 300 MB.
if not defined CI if not defined MUSICTOOL_NOPAUSE pause
exit /b 0

:fail
echo.
echo Setup failed - see the messages above.
if not defined CI if not defined MUSICTOOL_NOPAUSE pause
exit /b 1
