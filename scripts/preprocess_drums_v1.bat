@echo off
cd /d "%~dp0.."

call venv\Scripts\activate.bat

echo.
echo Preprocessing drums_v1 dataset (10.86h, 1090 files)
echo Source: data/raw/drums_v1_combined/ (mono 16-bit @ 44100 Hz)
echo Output: data/rave_ready/drums_v1/
echo.
echo This typically takes 10-20 minutes. DO NOT close this window.
echo.

if exist data\rave_ready\drums_v1 (
    echo Removing existing LMDB at data/rave_ready/drums_v1...
    rmdir /S /Q data\rave_ready\drums_v1
)

venv\Scripts\rave preprocess ^
    --input_path data/raw/drums_v1_combined ^
    --output_path data/rave_ready/drums_v1 ^
    --channels 1 ^
    --sampling_rate 44100

echo.
echo Done. Check data/rave_ready/drums_v1/metadata.yaml exists.
pause
