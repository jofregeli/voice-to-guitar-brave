@echo off
cd /d "%~dp0.."

call venv\Scripts\activate.bat

echo.
echo Starting BRAVE guitar model training...
echo Logs will appear below. Do NOT close this window.
echo To check on training later, look in runs\guitar_v1\
echo.

:: Find the latest best.ckpt using PowerShell
for /f "delims=" %%f in ('powershell -NoProfile -Command "Get-ChildItem -Path runs -Recurse -Filter *.ckpt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set CKPT=%%f

if defined CKPT (
    echo Resuming from checkpoint: %CKPT%
    venv\Scripts\rave train ^
        --config C:\Users\Usuario\Documents\BRAVE\configs\c128_r10.gin ^
        --db_path data/rave_ready/guitar ^
        --name guitar_v1 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000 ^
        --ckpt "%CKPT%"
) else (
    echo No checkpoint found, starting from scratch.
    venv\Scripts\rave train ^
        --config C:\Users\Usuario\Documents\BRAVE\configs\c128_r10.gin ^
        --db_path data/rave_ready/guitar ^
        --name guitar_v1 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000
)

echo.
echo Training finished or stopped.
pause
