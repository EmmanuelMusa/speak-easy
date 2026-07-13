@echo off
rem Launch Speak Easy with a visible console (logs shown).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m app %*
pause
