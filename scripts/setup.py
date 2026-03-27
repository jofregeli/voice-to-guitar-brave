"""
One-command setup for the voice-to-guitar project.

Usage (from the repo root):
    python scripts/setup.py

What this does:
    1. Creates a virtual environment (venv/)
    2. Installs PyTorch with the right CUDA version for your GPU
    3. Installs all other dependencies
    4. Applies compatibility patches to acids-rave (scipy/numpy/pytorch-lightning fixes)
    5. Checks for ffmpeg and gives install instructions if missing

Tested on:
    - Windows 11, Python 3.14, RTX 5080 (sm_120), CUDA driver 13.2
    - Should work on Linux/Mac with adjustments noted below
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# - Config -

REPO_ROOT = Path(__file__).parent.parent
VENV_DIR = REPO_ROOT / "venv"
IS_WINDOWS = sys.platform == "win32"

# Pip executable inside the venv
PIP = str(VENV_DIR / ("Scripts/pip" if IS_WINDOWS else "bin/pip"))
PYTHON = str(VENV_DIR / ("Scripts/python" if IS_WINDOWS else "bin/python"))

# - Helpers -

def run(cmd, check=True, **kwargs):
    """Run a shell command, print it, and check for errors."""
    print(f"\n  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return subprocess.run(cmd, check=check, **kwargs)


def patch_file(path: Path, old: str, new: str, description: str):
    """Apply a single string replacement patch to a file."""
    content = path.read_text(encoding="utf-8")
    if old not in content:
        if new.strip() in content:
            print(f"  [SKIP] Already patched: {description}")
        else:
            print(f"  [WARN] Could not find patch target: {description}")
            print(f"         Expected to find: {old!r}")
        return
    patched = content.replace(old, new, 1)
    path.write_text(patched, encoding="utf-8")
    print(f"  [OK]   {description}")


def get_cuda_compute_capability() -> tuple[int, int] | None:
    """Try to detect the GPU's compute capability using nvidia-smi."""
    try:
        # nvidia-smi may not be in PATH on Windows
        nvsmi_candidates = ["nvidia-smi"]
        if IS_WINDOWS:
            nvsmi_candidates.append(
                r"C:\Windows\System32\nvidia-smi.exe"
            )
        for nvsmi in nvsmi_candidates:
            result = subprocess.run(
                [nvsmi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            cc_str = result.stdout.strip().split(".")
            return int(cc_str[0]), int(cc_str[1])
    except Exception:
        return None


def pick_torch_index_url(cc: tuple[int, int] | None) -> str:
    """
    Choose the right PyTorch wheel index URL based on compute capability.

    RTX 5000 series (Blackwell, sm_120+) needs cu128 or cu130.
    Older GPUs work with cu126.
    CPU fallback if no GPU detected.
    """
    if cc is None:
        print("  [WARN] Could not detect GPU. Installing CPU-only PyTorch.")
        print("         For GPU support, install PyTorch manually:")
        print("         https://pytorch.org/get-started/locally/")
        return "https://download.pytorch.org/whl/cpu"

    major, minor = cc
    sm = major * 10 + minor  # e.g. sm_120 for RTX 5080

    if sm >= 120:
        # Blackwell (RTX 5000 series) ? needs CUDA 12.8+
        print(f"  [INFO] Detected GPU compute capability sm_{sm} (Blackwell)")
        print("         Using PyTorch CUDA 12.8 wheels")
        return "https://download.pytorch.org/whl/cu128"
    else:
        print(f"  [INFO] Detected GPU compute capability sm_{sm}")
        print("         Using PyTorch CUDA 12.6 wheels")
        return "https://download.pytorch.org/whl/cu126"


# - Setup steps -

def step_create_venv():
    print("\n-- Step 1: Create virtual environment ----------------------------------")
    if VENV_DIR.exists():
        print("  [SKIP] venv/ already exists")
        return
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    print("  [OK]   venv created")


def step_install_pytorch():
    print("\n-- Step 2: Install PyTorch ---------------------------------------------")
    cc = get_cuda_compute_capability()
    index_url = pick_torch_index_url(cc)
    run([PIP, "install", "--upgrade", "pip"], check=False)
    run([PIP, "install", "torch", "torchaudio", "--index-url", index_url])
    print("  [OK]   PyTorch installed")


def step_install_dependencies():
    print("\n-- Step 3: Install project dependencies --------------------------------")
    req = REPO_ROOT / "requirements.txt"
    run([PIP, "install", "-r", str(req)])
    # Install acids-rave 2.3.1 without strict dep enforcement
    # (forces 2.3.x over the 2.1.x that the resolver would pick)
    run([PIP, "install", "acids-rave==2.3.1", "--no-deps"])
    # pytorch-lightning 2.x (the pinned 1.9 doesn't detect modern GPUs)
    run([PIP, "install", "pytorch-lightning>=2.0", "--no-deps"])
    print("  [OK]   Dependencies installed")


def step_apply_patches():
    """
    Apply compatibility patches to acids-rave for scipy >= 1.14 and numpy 2.x.
    These patches are documented in docs/setup_notes.md.
    """
    print("\n-- Step 4: Apply compatibility patches ---------------------------------")

    # Find the installed pqmf.py
    site_packages = Path(PYTHON).parent.parent / "Lib/site-packages"
    if not IS_WINDOWS:
        # Linux/Mac venv layout
        site_packages = VENV_DIR / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

    pqmf_path = site_packages / "rave" / "pqmf.py"
    model_path = site_packages / "rave" / "model.py"

    if not pqmf_path.exists():
        print(f"  [WARN] Could not find {pqmf_path}")
        print("         Make sure acids-rave installed correctly.")
        return

    # --- pqmf.py patch 1: kaiser moved to scipy.signal.windows ---
    patch_file(
        pqmf_path,
        old="from scipy.signal import firwin, kaiser, kaiser_beta, kaiserord",
        new=(
            "from scipy.signal import firwin, kaiser_beta, kaiserord\n"
            "from scipy.signal.windows import kaiser"
        ),
        description="pqmf.py: kaiser ? scipy.signal.windows.kaiser (scipy >= 1.14)",
    )

    # --- pqmf.py patch 2: kaiserord needs Python float (numpy 2.x) ---
    patch_file(
        pqmf_path,
        old="    N_, beta = kaiserord(atten, wc / np.pi)",
        new="    N_, beta = kaiserord(atten, float(np.asarray(wc).ravel()[0]) / np.pi)",
        description="pqmf.py: kaiserord wc must be Python float (numpy 2.x)",
    )

    # --- pqmf.py patch 3: firwin nyq ? fs (scipy >= 1.14) ---
    patch_file(
        pqmf_path,
        old="    h = firwin(N, wc, window=('kaiser', beta), scale=False, nyq=np.pi)",
        new="    h = firwin(N, wc, window=('kaiser', beta), scale=False, fs=2 * np.pi)",
        description="pqmf.py: firwin nyq= ? fs= (scipy >= 1.14)",
    )

    # --- model.py patch: validation_epoch_end ? on_validation_epoch_end (PL 2.0) ---
    if model_path.exists():
        patch_file(
            model_path,
            old="    def validation_epoch_end(self, out):",
            new="    def on_validation_epoch_end(self, out=[]):",
            description="model.py: validation_epoch_end ? on_validation_epoch_end (PL 2.0)",
        )
    else:
        print(f"  [WARN] Could not find {model_path}")

    print("  [OK]   All patches applied")


def step_check_ffmpeg():
    print("\n-- Step 5: Check ffmpeg ------------------------------------------------")

    # Check common locations
    ffmpeg_found = shutil.which("ffmpeg") is not None

    if not ffmpeg_found and IS_WINDOWS:
        # Check winget install location
        winget_path = Path(os.environ.get("LOCALAPPDATA", "")) / (
            "Microsoft/WinGet/Packages/"
            "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        )
        ffmpeg_candidates = list(winget_path.glob("**/ffmpeg.exe"))
        if ffmpeg_candidates:
            ffmpeg_dir = str(ffmpeg_candidates[0].parent)
            ffmpeg_found = True
            _write_ffmpeg_to_activate(ffmpeg_dir)
            print(f"  [OK]   ffmpeg found at {ffmpeg_dir}")
            print("         Added to venv/Scripts/activate")

    if ffmpeg_found and shutil.which("ffmpeg"):
        print("  [OK]   ffmpeg is in PATH")
    elif not ffmpeg_found:
        print("  [WARN] ffmpeg not found ? required for rave preprocess")
        print()
        if IS_WINDOWS:
            print("  Install with:")
            print("    winget install --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements")
            print("  Then re-run this script to add it to the venv activate script.")
        else:
            print("  Install with:")
            print("    Ubuntu/Debian: sudo apt install ffmpeg")
            print("    macOS:         brew install ffmpeg")
            print("    conda:         conda install ffmpeg")


def _write_ffmpeg_to_activate(ffmpeg_bin_dir: str):
    """Inject ffmpeg PATH into venv/Scripts/activate if not already there."""
    activate = VENV_DIR / "Scripts" / "activate"
    if not activate.exists():
        return
    content = activate.read_text(encoding="utf-8")
    marker = "# ffmpeg PATH (added by scripts/setup.py)"
    if marker in content:
        return  # already injected
    injection = (
        f"\n{marker}\n"
        f'export PATH="{ffmpeg_bin_dir}:$PATH"\n'
    )
    # Insert just before VIRTUAL_ENV_PROMPT line
    content = content.replace(
        "VIRTUAL_ENV_PROMPT=venv",
        injection + "VIRTUAL_ENV_PROMPT=venv",
        1,
    )
    activate.write_text(content, encoding="utf-8")


def step_verify():
    print("\n-- Step 6: Verify installation -----------------------------------------")
    result = run(
        [PYTHON, "-c",
         "import rave; import torch; "
         "print('PyTorch:', torch.__version__); "
         "print('CUDA available:', torch.cuda.is_available()); "
         "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); "
         "print('rave: OK')"],
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"  [WARN] Verification failed:\n{result.stderr.strip()}")
    else:
        print("  [OK]   Environment verified")


# - Main -

def main():
    print("=" * 60)
    print("Voice-to-Guitar TFG ? Environment Setup")
    print("=" * 60)
    print(f"  Repo root: {REPO_ROOT}")
    print(f"  Python:    {sys.version}")
    print(f"  Platform:  {sys.platform}")

    step_create_venv()
    step_install_pytorch()
    step_install_dependencies()
    step_apply_patches()
    step_check_ffmpeg()
    step_verify()

    print("\n" + "=" * 60)
    print("Setup complete!")
    print()
    print("To activate the environment:")
    if IS_WINDOWS:
        print("  source venv/Scripts/activate   (Git Bash)")
        print("  venv\\Scripts\\activate.bat      (Windows CMD)")
    else:
        print("  source venv/bin/activate")
    print()
    print("Next: download datasets")
    print("  python scripts/download_data.py --dataset guitarset")
    print("  python scripts/download_data.py --dataset guitartechs")
    print("  python scripts/download_data.py --dataset groove")
    print("=" * 60)


if __name__ == "__main__":
    main()
