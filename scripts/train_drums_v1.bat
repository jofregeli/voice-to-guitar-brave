@echo off
cd /d "%~dp0.."

call venv\Scripts\activate.bat

echo.
echo Starting BRAVE drums_v1 training...
echo Dataset: Groove MIDI Dataset audio, 10.86h, 1090 files (mono 16-bit @ 44100 Hz)
echo Config: c16_r10_beta_fixed.gin (BRAVE-paper defaults — drums proven to work at 2.8h)
echo Note: We have 4x more drums data than the BRAVE paper, default config should suffice.
echo Logs will appear below. Do NOT close this window.
echo To check on training later, look in runs\drums_v1_*\
echo TensorBoard: venv\Scripts\tensorboard --logdir runs\
echo.

:: Find the latest checkpoint for drums_v1 only
for /f "delims=" %%f in ('powershell -NoProfile -Command "Get-ChildItem -Path runs -Recurse -Filter *.ckpt | Where-Object { $_.FullName -like '*drums_v1*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set CKPT=%%f

if defined CKPT (
    echo Resuming from checkpoint: %CKPT%
    venv\Scripts\rave train ^
        --config config\c16_r10_beta_fixed.gin ^
        --db_path data/rave_ready/drums_v1 ^
        --name drums_v1 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000 ^
        --ckpt "%CKPT%"
) else (
    echo No checkpoint found, starting from scratch.
    venv\Scripts\rave train ^
        --config config\c16_r10_beta_fixed.gin ^
        --db_path data/rave_ready/drums_v1 ^
        --name drums_v1 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000
)

echo.
echo Training finished or stopped.
pause
