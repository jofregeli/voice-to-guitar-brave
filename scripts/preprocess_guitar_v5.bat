@echo off
cd /d "%~dp0.."

call venv\Scripts\activate.bat

echo.
echo Preprocessing guitar_v5 dataset (~8.9h, 899 files)
echo Source: data/raw/guitar_v5_combined/
echo Output: data/rave_ready/guitar_v5/
echo.
echo This typically takes 10-20 minutes. DO NOT close this window.
echo Progress will be shown below.
echo.

if exist data\rave_ready\guitar_v5 (
    echo Removing existing LMDB at data/rave_ready/guitar_v5...
    rmdir /S /Q data\rave_ready\guitar_v5
)

venv\Scripts\rave preprocess ^
    --input_path data/raw/guitar_v5_combined ^
    --output_path data/rave_ready/guitar_v5 ^
    --channels 1 ^
    --sampling_rate 44100

echo.
echo Done. Check data/rave_ready/guitar_v5/metadata.yaml exists.
pause
