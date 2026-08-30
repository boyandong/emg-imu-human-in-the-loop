@echo off
call "D:\Users\qxy\anaconda3\Scripts\activate.bat" emgforce
python "%~dp0main.py"
if errorlevel 1 pause
