@echo off
rem Starts the MusicTool web app and opens it in your browser.
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  set MUSICTOOL_NOPAUSE=1
  call install.bat || (pause & exit /b 1)
)
".venv\Scripts\python.exe" -m musictool serve --open
pause
