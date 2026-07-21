@echo off
rem Launch SpeakEasy with a visible console (logs shown).
rem The title is set here as well as from Python, so the taskbar reads
rem "SpeakEasy" during the seconds before the app finishes importing, rather
rem than the folder path.
title SpeakEasy
cd /d "%~dp0"
".venv\Scripts\python.exe" -m app %*
pause
