"""
Preprocess raw audio datasets for BRAVE/RAVE training.

What this does:
  1. Finds all audio files in the raw data directory
  2. Resamples to 44100 Hz (RAVE standard)
  3. Converts to mono
  4. Trims leading/trailing silence
  5. Saves as 16-bit WAV in data/processed/<instrument>/

Usage:
    python scripts/preprocess.py --instrument guitar
    python scripts/preprocess.py --instrument drums
    python scripts/preprocess.py --instrument guitar --dry_run  # Preview without writing

After this, run RAVE's built-in preprocessor:
    rave preprocess --input_path data/processed/guitar --output_path data/rave_ready/guitar
"""

import argparse
import os
from pathlib import Path

import librosa          # audio loading and resampling
import numpy as np
import soundfile as sf  # audio saving
from tqdm import tqdm   # progress bar

# - Configuration -

TARGET_SR = 44100       # Sample rate expected by RAVE/BRAVE
TARGET_CHANNELS = 1     # Mono

# Silence trimming threshold (in dB below peak)
# Increase if you want more aggressive silence removal
SILENCE_THRESHOLD_DB = 40

# Minimum file duration to keep (in seconds)
# Files shorter than this are skipped
MIN_DURATION_SEC = 1.0

# Input directories per instrument
INPUT_DIRS = {
    "guitar": [
        "data/raw/guitarset",      # GuitarSet mic recordings (flat directory)
        "data/raw/guitartechs",    # Guitar-TECHS (recursive, DI channel only)
    ],
    "drums": [
        "data/raw/groove/audio",   # Groove MIDI Dataset audio
    ],
}

# File filter per directory
# GuitarSet: only solo (monophonic) files ? filenames contain '_solo'
# Guitar-TECHS: only direct-input channel ? path contains 'directinput'
GUITAR_FILTER = {
    "data/raw/guitarset": lambda p: "_solo" in p.stem,
    "data/raw/guitartechs": lambda p: "directinput" in str(p),
}

OUTPUT_DIR_TEMPLATE = "data/processed/{instrument}"

# - Helpers -

def find_audio_files(directory: Path, extension=(".wav", ".mp3", ".flac", ".ogg")):
    """Recursively find all audio files in a directory."""
    return [p for p in directory.rglob("*") if p.suffix.lower() in extension]


def preprocess_audio(input_path: Path, output_path: Path, sr: int = TARGET_SR):
    """
    Load, resample, convert to mono, trim silence, and save a single audio file.

    Returns True if the file was saved, False if it was skipped.
    """
    # Load audio (librosa handles most formats, resamples automatically)
    audio, original_sr = librosa.load(str(input_path), sr=sr, mono=True)

    # Trim silence from beginning and end
    audio_trimmed, _ = librosa.effects.trim(
        audio,
        top_db=SILENCE_THRESHOLD_DB,
        frame_length=2048,
        hop_length=512,
    )

    # Skip very short files
    duration_sec = len(audio_trimmed) / sr
    if duration_sec < MIN_DURATION_SEC:
        return False

    # Save as 16-bit WAV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_trimmed, sr, subtype="PCM_16")
    return True


# - Main -

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess audio datasets for BRAVE training."
    )
    parser.add_argument(
        "--instrument",
        choices=["guitar", "drums"],
        required=True,
        help="Which instrument's dataset to preprocess.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List files that would be processed without actually writing anything.",
    )
    args = parser.parse_args()

    output_dir = Path(OUTPUT_DIR_TEMPLATE.format(instrument=args.instrument))
    input_dirs = INPUT_DIRS[args.instrument]

    print(f"\nPreprocessing: {args.instrument}")
    print(f"Target sample rate: {TARGET_SR} Hz, mono")
    print(f"Output directory: {output_dir}")
    print()

    total_saved = 0
    total_skipped = 0
    total_duration_sec = 0.0

    for raw_dir_str in input_dirs:
        raw_dir = Path(raw_dir_str)

        if not raw_dir.exists():
            print(f"  [WARN] Input directory not found: {raw_dir}")
            print(f"    Run: python scripts/download_data.py --dataset {args.instrument}")
            continue

        audio_files = find_audio_files(raw_dir)
        print(f"  Found {len(audio_files)} files in {raw_dir}")

        # Apply per-directory filters (e.g., GuitarSet solo-only)
        filter_fn = GUITAR_FILTER.get(raw_dir_str, lambda p: True)
        audio_files = [f for f in audio_files if filter_fn(f)]
        print(f"  After filter: {len(audio_files)} files")

        for input_path in tqdm(audio_files, desc=f"  Processing {raw_dir.name}"):
            # Build output path: preserve relative filename, change extension to .wav
            rel_path = input_path.relative_to(raw_dir)
            output_path = output_dir / rel_path.with_suffix(".wav")

            if args.dry_run:
                print(f"    [DRY RUN] {input_path} -> {output_path}")
                continue

            try:
                saved = preprocess_audio(input_path, output_path)
                if saved:
                    # Measure duration of saved file
                    dur = librosa.get_duration(path=str(output_path))
                    total_duration_sec += dur
                    total_saved += 1
                else:
                    total_skipped += 1
            except Exception as e:
                print(f"    ERROR processing {input_path}: {e}")
                total_skipped += 1

    if not args.dry_run:
        total_min = total_duration_sec / 60
        total_hr = total_min / 60
        print(f"\n{'='*50}")
        print(f"Done!")
        print(f"  Files saved:   {total_saved}")
        print(f"  Files skipped: {total_skipped} (too short or errors)")
        print(f"  Total duration: {total_min:.1f} min ({total_hr:.2f} h)")
        print(f"  Output: {output_dir}/")
        print()
        print("Next step: run RAVE's preprocessor:")
        print(f"  rave preprocess --input_path {output_dir} "
              f"--output_path data/rave_ready/{args.instrument}")


if __name__ == "__main__":
    main()
