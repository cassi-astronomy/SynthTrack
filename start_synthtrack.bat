@echo off
rem SynthTrack - spusteni GUI (Windows)
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
start "" http://localhost:8765
python -m synthtrack serve %*
pause
