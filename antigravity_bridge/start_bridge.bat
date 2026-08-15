@echo off
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python bridge_server.py
pause
