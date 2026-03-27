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

### Week 1 — Mar 27–Apr 2 | Setup + Orientation (6h)

| Task | Hours | Status |
|------|-------|--------|
| Clone BRAVE repo, run their demo/example | 2h | ☐ |
| Verify PyTorch, Docker, Pure Data + `nn~` all work | 1.5h | ☐ |
| Download GuitarSet (check storage: ~8GB) | 0.5h | ☐ |
| Skim RAVE paper — concepts only, no math | 1h | ☐ |
| Initialize git repo + project structure | 1h | ✅ |

**Milestone:** BRAVE runs locally. Environment confirmed working. `nn~` works in Pure Data.

> **Risk:** `nn~` may not work on Windows without manual setup. Test this first — it's a known pain point.

---

### Week 2 — Apr 3–Apr 9 | State of the Art (6h)

| Task | Hours | Status |
|------|-------|--------|
| Deep read: RAVE paper (arXiv:2111.05011) | 2h | ☐ |
| Deep read: BRAVE paper (arXiv:2503.11562) | 1.5h | ☐ |
| Skim: NSynth + DDSP + TimbreTron | 1h | ☐ |
| Draft thesis Section 2 (SoA) — bullet points | 1.5h | ☐ |

**Milestone:** Can explain what RAVE/BRAVE does in own words. SoA outline drafted.

---

### Week 3 — Apr 10–Apr 16 | Dataset Analysis + BRAVE Config (6h)

| Task | Hours | Status |
|------|-------|--------|
| Download Guitar-TECHS, Groove datasets | 0.5h | ☐ |
| Explore GuitarSet: listen, check formats, total duration | 1h | ☐ |
| Write `scripts/preprocess.py` — resample, mono, trim silence | 2h | ☐ |
| Read BRAVE config files — understand all key parameters | 1.5h | ☐ |
| **Start thesis writing:** Introduction + SoA first draft | 1h | ☐ |

**Milestone:** Preprocessed audio ready in `data/processed/`. Config parameters understood.

---

### Week 4 — Apr 17–Apr 23 | Training Setup (6h)

| Task | Hours | Status |
|------|-------|--------|
| Finalize BRAVE config for guitar model | 2h | ☐ |
| Launch training run — verify no errors, first 15 min | 1.5h | ☐ |
| Set up logging / loss curve monitoring | 1h | ☐ |
| **Thesis:** Methodology section — architecture description | 1.5h | ☐ |

**Milestone:** Training running without errors. First loss values look reasonable.

---

### Week 5 — Apr 24–Apr 30 | Training Run #1 — Guitar (6h)

| Task | Hours | Status |
|------|-------|--------|
| Monitor training: check loss curves at checkpoints | 2h | ☐ |
| Listen to intermediate outputs (every 50k steps) | 1h | ☐ |
| Debug any training issues | 1h | ☐ |
| **Thesis:** Dataset section + preprocessing description | 2h | ☐ |

**Milestone:** Guitar model training in progress. At least one checkpoint produces guitar-like output.

---

### Week 6 — May 1–May 7 | Evaluate Guitar Model (6h)

| Task | Hours | Status |
|------|-------|--------|
| Listen to generated outputs — qualitative assessment | 1h | ☐ |
| Run evaluation metrics (FAD, mel-spectrogram MSE) | 2h | ☐ |
| Decide: continue training or adjust config + retrain | 1h | ☐ |
| **Thesis:** Training section | 2h | ☐ |

**Milestone:** Can describe what the guitar model does well and poorly. Go/no-go for retraining.

---

### Week 7 — May 8–May 14 | Training Run #2 + Drums (6h)

| Task | Hours | Status |
|------|-------|--------|
| Guitar model: extend training or second run (if needed) | 2h | ☐ |
| Launch drums model training (Groove dataset) | 1h | ☐ |
| Export final guitar checkpoint for Pure Data deployment | 1h | ☐ |
| **Thesis:** Results section — first draft | 2h | ☐ |

**Milestone:** Final guitar model exported. Drums training in progress.

---

### Week 8 — May 15–May 21 | Pure Data Integration (6h)

| Task | Hours | Status |
|------|-------|--------|
| Load guitar model via `nn~` in Pure Data | 2h | ☐ |
| Build PD patch: mic → model → audio out | 2h | ☐ |
| Test voice-to-guitar live — assess pitch handling | 1h | ☐ |
| **Thesis:** Real-time system section | 1h | ☐ |

**Milestone:** Live voice-to-guitar working in Pure Data (quality TBD). Drums model ready.

---

### Week 9 — May 22–May 28 | Demo + Secondary Experiment (6h)

| Task | Hours | Status |
|------|-------|--------|
| Add drums model to Pure Data patch | 1h | ☐ |
| Polish PD patch: volume, latency, error handling | 1.5h | ☐ |
| Record demo audio/video for thesis appendix | 1.5h | ☐ |
| **Thesis:** Abstract + Conclusions first drafts | 2h | ☐ |

**Milestone:** Demo runs in one command. Both guitar and drums demos recorded.

---

### Week 10 — May 29–Jun 4 | Thesis Sprint (6h)

| Task | Hours | Status |
|------|-------|--------|
| Assemble all thesis sections into full document | 2h | ☐ |
| Fill gaps, fix transitions, add figures and diagrams | 2h | ☐ |
| Send to advisor/peer for feedback | 2h | ☐ |

**Milestone:** Complete first draft submitted for review.

---

### Week 11 — Jun 5–Jun 12 | Final Polish (6h)

| Task | Hours | Status |
|------|-------|--------|
| Incorporate advisor feedback | 2h | ☐ |
| Final prototype test — ensure demo is reproducible | 2h | ☐ |
| Submit thesis + code | 2h | ☐ |

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
