# GANTT Chart — Voice-to-Guitar TFG

**Author:** Jofre Geli de Fuenmayor | **Deadline:** June 12, 2026

---

## Timeline Overview

| Task | W1| W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 | W11 |
|------|----------|---------|----------|----------|----------|---------|---------|----------|----------|-----------|----------|
| **Environment setup** | ✅ | | | | | | | | | | |
| **Dataset download + preprocessing** | ✅ | | | | | | | | | | |
| **Guitar model training** | ✅ | ▶ | ▶ | ▶ | ▶ | | | | | | |
| **State of the Art** | ▶ | ▶ | | | | | | | | | |
| **Thesis: Introduction + SoA** | | ▶ | ▶ | | | | | | | | |
| **Evaluate guitar model** | | | | | ▶ | ▶ | | | | | |
| **Drums model training** | | | | | | | ▶ | | | | |
| **Thesis: Methodology + Training** | | | | ▶ | ▶ | ▶ | ▶ | | | | |
| **Pure Data integration** | | | | | | | | ▶ | ▶ | | |
| **Thesis: Results + Real-time** | | | | | | | ▶ | ▶ | ▶ | | |
| **Thesis: Abstract + Conclusions** | | | | | | | | | ▶ | ▶ | |
| **Demo recording** | | | | | | | | | ▶ | | |
| **Thesis final assembly + polish** | | | | | | | | | | ▶ | ▶ |
| **Submission** | | | | | | | | | | | ✅ |

**Legend:** ✅ Done &nbsp;|&nbsp; ▶ Active/Planned &nbsp;|&nbsp; (empty) Not started

---

## Task List

### In Progress
- [x] Reading: RAVE paper (arXiv:2111.05011)
- [ ] Reading: BRAVE paper (arXiv:2503.11562)
- [ ] Guitar model training guitar_v2 — Phase 1 (fixed beta, restarted after diagnosing posterior collapse in v1)

### Next (Weeks 2–4)
- [ ] Read: NSynth, DDSP, TimbreTron (skim)
- [ ] Draft thesis Section 2 — State of the Art
- [ ] Define thesis sections with content descriptions
- [ ] Listen to guitar model outputs as training progresses
- [ ] Begin thesis Introduction

### Weeks 5–7
- [ ] Evaluate guitar model — qualitative + FAD metric
- [ ] Decide: retrain or proceed to integration
- [ ] Launch drums model training (Groove dataset)
- [ ] Export final guitar model checkpoint (.ts)

### Weeks 8–9
- [ ] Build Pure Data patch: mic → BRAVE model → audio out
- [ ] Live voice-to-guitar demo — assess quality
- [ ] Record demo audio/video for thesis appendix

### Weeks 10–11
- [ ] Assemble full thesis document
- [ ] Incorporate advisor feedback
- [ ] Final reproducibility check
- [ ] Submit

---

## Key Milestones

| Milestone | Target | Status |
|-----------|--------|--------|
| Environment + pipeline working | Week 1 | ✅ Done |
| Guitar model training started | Week 1 | ✅ Done |
| Guitar model Phase 2 (GAN active) | Week 2 (~Apr 2) | ⚠ guitar_v1 had posterior collapse — guitar_v2 restarted Mar 31 with fixed beta (0.0001→0.1 over 1M steps). Phase 1 ends ~Apr 1 evening. |
| Guitar model evaluation | Week 6 | Pending |
| Live voice-to-guitar demo | Week 8 | Pending |
| Complete thesis first draft | Week 10 | Pending |
| Submission | Jun 12 | Pending |
