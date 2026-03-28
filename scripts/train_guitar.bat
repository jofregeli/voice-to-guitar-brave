@echo off
cd /d "%~dp0.."

call venv\Scripts\activate.bat

echo.
echo Starting BRAVE guitar model training...
echo Logs will appear below. Do NOT close this window.
echo To check on training later, look in runs\guitar_v1\
echo.

venv\Scripts\rave train ^
    --config C:\Users\Usuario\Documents\BRAVE\configs\c128_r10.gin ^
    --db_path data/rave_ready/guitar ^
    --name guitar_v1 ^
    --channels 1 ^
    --val_every 10000

echo.
echo Training finished or stopped.
pause
