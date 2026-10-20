@echo off
cd /d "%~dp0"

rem Keys are read from C:\keydata (see config.json)
rem To use ComfyUI portable python, change PY below, e.g.
rem set PY=C:\ComfyUI_windows_portable\python_embeded\python.exe
set PY=python

%PY% main.py %*
pause
