"""
Download training datasets for the voice-to-guitar TFG project.

Usage:
    python scripts/download_data.py --dataset guitarset
    python scripts/download_data.py --dataset guitartechs
    python scripts/download_data.py --dataset groove
    python scripts/download_data.py --dataset all

What gets downloaded:
    guitarset   → data/raw/guitarset/   (~8.2 GB total, we use the mic/ folder only)
    guitartechs → data/raw/guitartechs/ (~4 GB, we use the DI channel)
    groove      → data/raw/groove/      (~3.5 GB audio subset)
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Use the zenodo_get from the same Python environment running this script
ZENODO_GET = [sys.executable, "-m", "zenodo_get"]

# ── Dataset definitions ───────────────────────────────────────────────────────

DATASETS = {
    "guitarset": {
        "description": "GuitarSet — acoustic guitar, CC BY 4.0 (~8.2 GB)",
        # Zenodo record for GuitarSet audio files
        # We only need the annotation-only + audio_mono-mic files
        "zenodo_id": "3371780",
        "output_dir": "data/raw/guitarset",
        "notes": (
            "After download, we only use: audio_mono-mic/ (microphone recordings).\n"
            "Solo performances (filename contains '_solo') are monophonic — use those for training."
        ),
    },
    "guitartechs": {
        "description": "Guitar-TECHS — electric guitar, CC BY 4.0 (~4 GB)",
        "zenodo_id": "14963133",
        "output_dir": "data/raw/guitartechs",
        "notes": (
            "After download, use the DI (direct injection) channel for cleanest signal.\n"
            "Look for files in the 'DI/' subdirectory."
        ),
    },
    "groove": {
        "description": "Groove MIDI Dataset — drums audio, CC BY 4.0 (~3.5 GB audio subset)",
        # Groove audio is hosted by Google Magenta — manual download required
        "zenodo_id": None,
        "output_dir": "data/raw/groove",
        "manual_url": "https://storage.googleapis.com/magentadata/datasets/groove/groove-v0.0.2-midionly.zip",
        "notes": (
            "The audio version requires downloading from Google Magenta directly.\n"
            "Visit: https://magenta.tensorflow.org/datasets/groove\n"
            "Download the 'Full dataset (audio+MIDI)' version (~3.5 GB)."
        ),
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_zenodo_get():
    """Check that zenodo_get is installed."""
    try:
        subprocess.run(ZENODO_GET + ["--help"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_zenodo(zenodo_id: str, output_dir: str):
    """Download a Zenodo record using zenodo_get."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Downloading Zenodo record {zenodo_id} -> {output_dir}/")
    result = subprocess.run(
        ZENODO_GET + [zenodo_id, "-o", output_dir],
        check=True,
    )
    return result.returncode == 0


def download_dataset(name: str):
    """Download a single dataset by name."""
    if name not in DATASETS:
        print(f"ERROR: Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")
        sys.exit(1)

    ds = DATASETS[name]
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"  {ds['description']}")
    print(f"  Output: {ds['output_dir']}")
    print(f"  Notes: {ds['notes']}")
    print(f"{'='*60}")

    # Check if already downloaded
    out_path = Path(ds["output_dir"])
    real_files = [f for f in out_path.iterdir() if f.name != ".gitkeep"] if out_path.exists() else []
    if real_files:
        print(f"  Directory {out_path} already has files. Skipping download.")
        print("  (Delete the directory if you want to re-download.)")
        return

    if ds["zenodo_id"] is not None:
        # Zenodo download
        if not check_zenodo_get():
            print("  ERROR: zenodo_get not found. Install it with:")
            print("    pip install zenodo-get")
            sys.exit(1)
        download_zenodo(ds["zenodo_id"], ds["output_dir"])
        print(f"  [OK] Download complete: {ds['output_dir']}/")

    else:
        # Manual download required
        print(f"\n  ⚠ Manual download required for '{name}'.")
        print(f"  URL: {ds.get('manual_url', 'See notes above')}")
        print(f"  After downloading, extract files to: {ds['output_dir']}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download training datasets for the voice-to-guitar project."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        required=True,
        help="Which dataset to download ('all' downloads everything).",
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in targets:
        download_dataset(name)

    print("\nDone. Next step: run scripts/preprocess.py to prepare data for training.")


if __name__ == "__main__":
    main()
