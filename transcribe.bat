@echo off
rem Drag and drop one or more audio files onto this file to get .mid files next to them.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  set MUSICTOOL_NOPAUSE=1
  call install.bat || (pause & exit /b 1)
)
if "%~1"=="" (
  echo Drag one or more audio files onto transcribe.bat to convert them to MIDI.
  pause
  exit /b 1
)
:next
if "%~1"=="" goto :done
echo.
echo === %~nx1 ===
".venv\Scripts\python.exe" -m musictool "%~1" --separate-tracks
shift
goto :next
:done
echo.
pause
