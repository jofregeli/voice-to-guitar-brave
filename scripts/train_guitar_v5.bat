@echo off
cd /d "%~dp0.."

call venv\Scripts\activate.bat

echo.
echo Starting BRAVE guitar_v5 training...
echo Dataset: GuitarSet + Guitar-TECHS + IDMT-SMT-Guitar dataset4 ~16h
echo Config: c16_r10_v5_balanced.gin (weakened discriminator, longer Phase 1)
echo NOTE: --augment flags removed (torchaudio.sox_effects no longer available
echo in our torchaudio version, breaks RAVE's augmentation code)
echo Logs will appear below. Do NOT close this window.
echo To check on training later, look in runs\guitar_v5_*\
echo TensorBoard: venv\Scripts\tensorboard --logdir runs\
echo.

:: Find the latest checkpoint for guitar_v5 only
for /f "delims=" %%f in ('powershell -NoProfile -Command "Get-ChildItem -Path runs -Recurse -Filter *.ckpt | Where-Object { $_.FullName -like '*guitar_v5*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set CKPT=%%f

if defined CKPT (
    echo Resuming from checkpoint: %CKPT%
    venv\Scripts\rave train ^
        --config config\c16_r10_v5_balanced.gin ^
        --db_path data/rave_ready/guitar_v5 ^
        --name guitar_v5 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000 ^
        --ckpt "%CKPT%"
) else (
    echo No checkpoint found, starting from scratch.
    venv\Scripts\rave train ^
        --config config\c16_r10_v5_balanced.gin ^
        --db_path data/rave_ready/guitar_v5 ^
        --name guitar_v5 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000
)

echo.
echo Training finished or stopped.
pause
