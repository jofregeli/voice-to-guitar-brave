@echo off
cd /d "%~dp0.."

call venv\Scripts\activate.bat

echo.
echo Starting BRAVE guitar model training...
echo Logs will appear below. Do NOT close this window.
echo To check on training later, look in runs\guitar_v2_*\
echo TensorBoard: venv\Scripts\tensorboard --logdir runs\guitar_v2_3415d67484
echo.

:: Find the latest checkpoint for guitar_v2 only
for /f "delims=" %%f in ('powershell -NoProfile -Command "Get-ChildItem -Path runs -Recurse -Filter *.ckpt | Where-Object { $_.FullName -like '*guitar_v2*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set CKPT=%%f

if defined CKPT (
    echo Resuming from checkpoint: %CKPT%
    venv\Scripts\rave train ^
        --config config\c128_r10_beta_fixed.gin ^
        --db_path data/rave_ready/guitar ^
        --name guitar_v2 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000 ^
        --ckpt "%CKPT%"
) else (
    echo No checkpoint found, starting from scratch.
    venv\Scripts\rave train ^
        --config config\c128_r10_beta_fixed.gin ^
        --db_path data/rave_ready/guitar ^
        --name guitar_v2 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000
)

echo.
echo Training finished or stopped.
pause
