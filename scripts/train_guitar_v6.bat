@echo off
cd /d "%~dp0.."

call venv\Scripts\activate.bat

echo.
echo Starting BRAVE guitar_v6 training (FINAL ATTEMPT)...
echo.
echo Strategy: Use OFFICIAL RAVE v2 + causal configs (community-validated,
echo BRAVE-paper recipe) with minimal overrides for our dataset.
echo.
echo Key features that v6 has and previous attempts did NOT:
echo   - update_discriminator_every = 4 (D updates 4x slower than G)
echo   - MultiPeriodDiscriminator + MultiScaleDiscriminator combined
echo   - EncoderV2 / GeneratorV2 with amplitude_modulation=True
echo   - valid_signal_crop = True
echo   - relative feature_matching loss
echo.
echo Dataset: data/rave_ready/guitar_v5 (16h, already preprocessed)
echo Overrides: LATENT_SIZE=16, PHASE_1_DURATION=1.5M
echo.
echo Logs will appear below. Do NOT close this window.
echo To check on training later, look in runs\guitar_v6_*\
echo TensorBoard: venv\Scripts\tensorboard --logdir runs\
echo.

:: Find the latest checkpoint for guitar_v6 only
for /f "delims=" %%f in ('powershell -NoProfile -Command "Get-ChildItem -Path runs -Recurse -Filter *.ckpt | Where-Object { $_.FullName -like '*guitar_v6*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set CKPT=%%f

if defined CKPT (
    echo Resuming from checkpoint: %CKPT%
    venv\Scripts\rave train ^
        --config v2 ^
        --config causal ^
        --config config\guitar_v6_overrides.gin ^
        --db_path data/rave_ready/guitar_v5 ^
        --name guitar_v6 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000 ^
        --ckpt "%CKPT%"
) else (
    echo No checkpoint found, starting from scratch.
    venv\Scripts\rave train ^
        --config v2 ^
        --config causal ^
        --config config\guitar_v6_overrides.gin ^
        --db_path data/rave_ready/guitar_v5 ^
        --name guitar_v6 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000
)

echo.
echo Training finished or stopped.
pause
