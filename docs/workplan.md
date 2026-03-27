# Project Workplan — Voice-to-Guitar TFG

**Deadline:** June 12, 2026

---

## Locked Decisions

| Decision | Choice |
|----------|--------|
| Primary task | Voice → Guitar (timbre transfer) |
| Secondary task | Beatbox → Drums (sanity check + secondary demo) |
| Model | BRAVE (streaming RAVE) |
| Real-time I/O | Pure Data + nn~ external |
| Guitar dataset | GuitarSet (solo/mic) + Guitar-TECHS (DI) = ~6.7h, CC BY 4.0 |
| Drums dataset | Groove MIDI Dataset audio, ~13h, CC BY 4.0 |
| GPU | NVIDIA RTX 5080 (local, 24/7 access) |
| Framework | Python, PyTorch, Docker |

---

## Phase Overview

| Phase | Weeks | Focus |
|-------|-------|-------|
| 2 | W1–W2 | State of the Art + thesis outline |
| 3 | W1–W2 | Environment & repo setup *(parallel with Phase 2)* |
| 4 | W3–W4 | Dataset download, preprocessing pipeline |
| 5 | W3–W4 | BRAVE configuration deep-dive |
| 6 | W5–W7 | Training: guitar model (+ drums in W7) |
| 7 | W8–W9 | Pure Data integration + real-time pipeline |
| 8 | W10 | Evaluation metrics + results |
| 9 | W11 | Demo polish + thesis finalization |

*Thesis writing runs in parallel from Week 3 onward.*

---

## Week-by-Week Detail

---

### Week 1 — Mar 27–Apr 2 | Setup + Orientation

| Task | Status |
|------|--------|
| Clone BRAVE repo, run their demo/example | ☐ |
| Verify PyTorch, Docker, Pure Data + `nn~` all work | ☐ |
| Download GuitarSet (check storage: ~8GB) | ☐ |
| Skim RAVE paper — concepts only, no math | ☐ |
| Initialize git repo + project structure | ✅ |

**Milestone:** BRAVE runs locally. Environment confirmed working. `nn~` works in Pure Data.

> **Risk:** `nn~` may not work on Windows without manual setup. Test this first — it's a known pain point.

---

### Week 2 — Apr 3–Apr 9 | State of the Art

| Task | Status |
|------|--------|
| Deep read: RAVE paper (arXiv:2111.05011) | ☐ |
| Deep read: BRAVE paper (arXiv:2503.11562) | ☐ |
| Skim: NSynth + DDSP + TimbreTron | ☐ |
| Draft thesis Section 2 (SoA) — bullet points | ☐ |

**Milestone:** Can explain what RAVE/BRAVE does in own words. SoA outline drafted.

---

### Week 3 — Apr 10–Apr 16 | Dataset Analysis + BRAVE Config

| Task | Status |
|------|--------|
| Download Guitar-TECHS, Groove datasets | ☐ |
| Explore GuitarSet: listen, check formats, total duration | ☐ |
| Write `scripts/preprocess.py` — resample, mono, trim silence | ☐ |
| Read BRAVE config files — understand all key parameters | ☐ |
| **Start thesis writing:** Introduction + SoA first draft | ☐ |

**Milestone:** Preprocessed audio ready in `data/processed/`. Config parameters understood.

---

### Week 4 — Apr 17–Apr 23 | Training Setup

| Task | Hours | Status |
|------|-------|--------|
| Finalize BRAVE config for guitar model | 2h | ☐ |
| Launch training run — verify no errors, first 15 min | 1.5h | ☐ |
| Set up logging / loss curve monitoring | 1h | ☐ |
| **Thesis:** Methodology section — architecture description | 1.5h | ☐ |

**Milestone:** Training running without errors. First loss values look reasonable.

---

### Week 5 — Apr 24–Apr 30 | Training Run #1 — Guitar

| Task | Status |
|------|--------|
| Monitor training: check loss curves at checkpoints | ☐ |
| Listen to intermediate outputs (every 50k steps) | ☐ |
| Debug any training issues | ☐ |
| **Thesis:** Dataset section + preprocessing description | ☐ |

**Milestone:** Guitar model training in progress. At least one checkpoint produces guitar-like output.

---

### Week 6 — May 1–May 7 | Evaluate Guitar Model

| Task | Status |
|------|--------|
| Listen to generated outputs — qualitative assessment | ☐ |
| Run evaluation metrics (FAD, mel-spectrogram MSE) | ☐ |
| Decide: continue training or adjust config + retrain | ☐ |
| **Thesis:** Training section | ☐ |

**Milestone:** Can describe what the guitar model does well and poorly. Go/no-go for retraining.

---

### Week 7 — May 8–May 14 | Training Run #2 + Drums

| Task | Status |
|------|--------|
| Guitar model: extend training or second run (if needed) | ☐ |
| Launch drums model training (Groove dataset) | ☐ |
| Export final guitar checkpoint for Pure Data deployment | ☐ |
| **Thesis:** Results section — first draft | ☐ |

**Milestone:** Final guitar model exported. Drums training in progress.

---

### Week 8 — May 15–May 21 | Pure Data Integration

| Task | Status |
|------|--------|
| Load guitar model via `nn~` in Pure Data | ☐ |
| Build PD patch: mic → model → audio out | ☐ |
| Test voice-to-guitar live — assess pitch handling | ☐ |
| **Thesis:** Real-time system section | ☐ |

**Milestone:** Live voice-to-guitar working in Pure Data (quality TBD). Drums model ready.

---

### Week 9 — May 22–May 28 | Demo + Secondary Experiment

| Task | Status |
|------|--------|
| Add drums model to Pure Data patch | ☐ |
| Polish PD patch: volume, latency, error handling | ☐ |
| Record demo audio/video for thesis appendix | ☐ |
| **Thesis:** Abstract + Conclusions first drafts | ☐ |

**Milestone:** Demo runs in one command. Both guitar and drums demos recorded.

---

### Week 10 — May 29–Jun 4 | Thesis Sprint

| Task | Status |
|------|--------|
| Assemble all thesis sections into full document | ☐ |
| Fill gaps, fix transitions, add figures and diagrams | ☐ |
| Send to advisor/peer for feedback | ☐ |

**Milestone:** Complete first draft submitted for review.

---

### Week 11 — Jun 5–Jun 12 | Final Polish

| Task | Status |
|------|--------|
| Incorporate advisor feedback | ☐ |
| Final prototype test — ensure demo is reproducible | ☐ |
| Submit thesis + code | ☐ |

**Milestone:** ✅ Submitted.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `nn~` doesn't work on Windows | Medium | High | Test Week 1; Docker fallback |
| Guitar model sounds bad | Medium | Medium | It's a known BRAVE limitation — document it, it's thesis content |
| Pitch rendering issues | High | Medium | Pitch preprocessor (CREPE/pyin) as optional Week 8 add-on |
| Scope creep | High | Medium | Flag anything new before adding it |
| Thesis writing left too late | Low | High | Writing in plan from Week 3 — stay on it |

---

## Thesis Chapter Structure

```
1. Introduction
2. State of the Art
   2.1 Neural Audio Synthesis
   2.2 Timbre Transfer and Voice Conversion
   2.3 Datasets
   2.4 Research Gap
3. Methodology
   3.1 System Architecture
   3.2 Datasets and Preprocessing
   3.3 Model Configuration and Training
4. Results and Evaluation
   4.1 Guitar Model
   4.2 Drums Model (secondary)
   4.3 Real-Time Performance
5. Conclusions and Future Work
Appendix A: Demo Setup Instructions
Appendix B: Audio Samples
```
