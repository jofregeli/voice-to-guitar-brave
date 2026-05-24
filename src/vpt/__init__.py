"""Vocal Percussion Transcription (VPT) — Path A classifier.

Real-time kick/snare/hi-hat classification of beatbox sounds, deployed as a
TorchScript model loaded by `nn~` in Pure Data. Trained on the AVP dataset
(Delgado et al., 2019, https://doi.org/10.5281/zenodo.3250230).
"""

LABEL_MAP = {
    "kd": 0,   # kick drum
    "sd": 1,   # snare drum
    "hhc": 2,  # closed hi-hat
    "hho": 2,  # opened hi-hat (collapsed into hh class)
}
CLASS_NAMES = ["kick", "snare", "hh"]
N_CLASSES = 3

SAMPLE_RATE = 44100
WINDOW_SAMPLES = 1024  # ~23 ms causal context
