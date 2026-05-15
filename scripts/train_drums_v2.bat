@echo off
cd /d "%~dp0.."

call venv\Scripts\activate.bat

echo.
echo Starting BRAVE drums_v2 training...
echo.
echo Strategy: Use OFFICIAL RAVE v2 + causal configs (same recipe that worked
echo for guitar_v6 as autoencoder). The BRAVE paper proved drums works with
echo a similar architecture at 2.8h. We have 10.86h.
echo.
echo Key features (vs drums_v1 which used custom c16_r10):
echo   - update_discriminator_every = 4 (prevents D dominance, the recurring failure)
echo   - MultiPeriodDiscriminator + MultiScaleDiscriminator combined
echo   - EncoderV2 / GeneratorV2 with amplitude_modulation = True
echo   - valid_signal_crop = True
echo.
echo Dataset: data/rave_ready/drums_v1 (10.86h Groove, already preprocessed)
echo Overrides: LATENT_SIZE=16, PHASE_1_DURATION=1.5M (same as guitar_v6)
echo.

:: Find the latest checkpoint for drums_v2 only
for /f "delims=" %%f in ('powershell -NoProfile -Command "Get-ChildItem -Path runs -Recurse -Filter *.ckpt | Where-Object { $_.FullName -like '*drums_v2*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set CKPT=%%f

if defined CKPT (
    echo Resuming from checkpoint: %CKPT%
    venv\Scripts\rave train ^
        --config v2 ^
        --config causal ^
        --config config\drums_v2_overrides.gin ^
        --db_path data/rave_ready/drums_v1 ^
        --name drums_v2 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000 ^
        --ckpt "%CKPT%"
) else (
    echo No checkpoint found, starting from scratch.
    venv\Scripts\rave train ^
        --config v2 ^
        --config causal ^
        --config config\drums_v2_overrides.gin ^
        --db_path data/rave_ready/drums_v1 ^
        --name drums_v2 ^
        --channels 1 ^
        --gpu 0 ^
        --val_every 10000
)

echo.
echo Training finished or stopped.
pause
