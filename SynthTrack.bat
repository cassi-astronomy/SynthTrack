@echo off
rem SynthTrack - aplikace v samostatnem okne (bez prohlizece)
rem Dvojklik spusti okno; konzole se hned zavre. Log: %USERPROFILE%\SynthTrackData\synthtrack-app.log
cd /d "%~dp0"
if exist .venv\Scripts\synthtrack-app.exe (
  start "" .venv\Scripts\synthtrack-app.exe %*
) else if exist .venv\Scripts\pythonw.exe (
  start "" .venv\Scripts\pythonw.exe -m synthtrack app %*
) else (
  python -m synthtrack app %*
  pause
)
