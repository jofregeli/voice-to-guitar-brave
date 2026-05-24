# Project timeline and key decisions

This document narrates the project's chronology, the planning structure, and the major engineering and research decisions made along the way. It is the canonical source for the **Introduction (motivation framing)**, **Conclusions (reflection on the process)**, and **Appendix D (Project Planning)** sections.

## Phase structure (UPF TFG guide alignment)

The project is divided into three phases following the UPF TFG guide (Hernández-Leo, Moreno, & Camps, 2012, §2 *Temporització*):

| Phase | Duration | Effort | Dates (2026) |
|---|---|---|---|
| **Initial** (planning + setup) | 3 weeks | ~12 % of ECTS | 24 March – 13 April |
| **Execution** (iterative training + analysis) | 6 weeks | ~58 % of ECTS | 14 April – 24 May |
| **Final** (writing + defence preparation) | ~3 weeks | ~30 % of ECTS | 25 May – 12 June |

## High-level chronology

| Week | Dates (2026) | Phase | Event |
|---|---|---|---|
| W1 | 24–29 March | Initial | Project kick-off. Project Charter and Gantt drafted with director (Appendix D). Workstation set up; `acids-rave 2.3.1` installed with four compatibility patches; Pure Data 0.55 + `nn~` external verified. |
| W2 | 30 March – 5 April | Initial | RAVE (Caillon & Esling, 2021) and BRAVE (Caspe et al., 2025) papers read in depth; key hyperparameters and training-phase structure understood. GuitarSet and Guitar-TECHS downloaded; Groove MIDI Dataset downloaded for the secondary case study. |
| W3 | 6–12 April | Initial | Preprocessing pipeline (`scripts/preprocess.py`) implemented and validated. Initial LMDB built for guitar baseline corpus. Diagnostic methodology drafted (§3.4 of memoir). |
| W4 | 13–19 April | Execution | **guitar_v1** launched (`c128_r10.gin`, unmodified). Diagnostic confirmed posterior collapse: KL fell 0.5 → 0.25, forward diff 0.006. Lesson logged: log-space β warmup required. |
| W5 | 20–26 April | Execution | **guitar_v2** launched (`c128_r10_beta_fixed.gin`, β warmup applied). Diagnosed latent underutilisation: KL/dim ≈ 0.005 nats, forward diff 0.003. Lesson logged: LATENT_SIZE 128 too large for 5.4 h corpus; reduce to 16. |
| W6 | 27 April – 3 May | Execution | **guitar_v3** launched (`c16_r10_beta_fixed.gin`). Phase 1 stable; Phase 2 generator collapse observed. IDMT-SMT-Guitar dataset 4 added, lifting consolidated corpus to ~16 h. |
| W7 | 4–10 May | Execution | **guitar_v4** launched on cleaner mic-only data (3.05 h). Counter-intuitive worsening: adversarial mode collapse, gap 5.65, silence > signal. **guitar_v5** launched with halved discriminator. Inverse failure: discriminator collapse, pred_real = −0.90, gap 0.80. |
| W8 | 11–17 May | Execution | Pivot away from `c16_r10` to community-validated configuration. **drums_v1** launched (`c16_r10_beta_fixed.gin`) and trained to epoch 755 as a secondary case study; diagnostic forward diff 0.0045 (below 0.01 threshold, partial transient response only). **guitar_v6** launched on `v2.gin` + `causal.gin` + two overrides (LATENT_SIZE = 16, PHASE_1_DURATION = 1.5 M). |
| W9 | 18–24 May | Execution | guitar_v6 completed (full 3 M-step trajectory). Phase 2 GAN gap reversed for the first time in the series. Mid-Phase-2 checkpoint (epoch 1935) selected as canonical `guitar_v6_best.ts`. **drums_v2** launched on the same v2.gin + causal recipe (15 May), trained to step 2.28 M (95 wall-clock hours) before being terminated as Phase 2 discriminator dominance reproduced (gap widened 9.8 → 21.9). External baseline test conducted using the official ACIDS-IRCAM percussion model (forward diff 1.75 vs drums_v1's 0.0045, ≈ 390× more responsive). STFT-magnitude-correlation cross-checkpoint calibration study run; finding: no checkpoint passes the 0.3 heuristic, the metric does not robustly discriminate working from collapsed checkpoints. |
| W10 | 25–31 May | Final | Writing phase begins. LaTeX scaffold and bibliography set up. Chapters 1–6 first drafts completed in Google Docs / Word for iterative revision before LaTeX migration. Path A hybrid pipeline (vocal-percussion classifier + sample triggering, AVP dataset) prototyped as a preliminary exploration; not pursued for thesis demonstration after listening evaluation. |
| W11 | 1–7 June | Final | Chapter revisions; figure generation (`scripts/generate_chapter5_figures.py` producing five figures from existing TensorBoard logs and the canonical model). Methodology calibration discovery (STFT metric) integrated into §3.4, §5.1.4 and §6.3. References, appendices, and front matter compiled. |
| W12 | 8–12 June | Final | LaTeX migration of all chapters; final proofreading; defence rehearsal; **submission on 12 June 2026**. |

## Key engineering decisions

### Decision 1 — Use BRAVE rather than RAVE

BRAVE's streaming-causal architecture brings sub-10 ms latency, essential for live performance. The standard RAVE architecture uses non-causal convolutions and a larger temporal context, giving ~50 ms latency. The trade-off is that streaming-causal RAVE has slightly lower audio quality, but the latency reduction is critical for the project's musical use case.

### Decision 2 — Pure Data via `nn~` over a custom audio engine

The ACIDS-IRCAM `nn~` Pure Data external is the standard, well-maintained deployment target for RAVE-family models. Writing a custom audio engine would have consumed time better spent on training and experimentation. Pure Data also gives free access to standard DSP operations (filtering, gating, gain) when building demo patches.

### Decision 3 — Mix of recording sources for guitar training

Initially we used GuitarSet (mic'd acoustic) and Guitar-TECHS (line-out DI) to maximise dataset size. The supervisor flagged that mixing recording methods would introduce noise-floor heterogeneity. Quantitative analysis confirmed this (35.8 dB spread between cleanest DI and loudest mic-captured sources; see §5.5). We accepted the limitation rather than narrowing to a single recording method, on the grounds that more data was empirically beneficial in v3 (5.79 h mixed) vs v4 (3.05 h mic-only): cleaner-but-smaller data made the failure worse, not better.

### Decision 4 — When v5 failed, switch architectures entirely instead of tuning further

After v3, v4, and v5 all failed with the custom c16_r10 architecture (in three distinct ways), it was clear that no Goldilocks parameter setting would emerge from incremental tuning. The architectural features missing from c16_r10 (`update_discriminator_every`, MultiPeriod discriminator, amplitude-modulating generator) were present in the official v2.gin used by the broader RAVE community. The switch to v2.gin + causal.gin produced the first working autoencoder. Lesson: prefer community-validated configurations over local tuning.

### Decision 5 — Select mid-Phase-2 v6 checkpoint as canonical rather than the final epoch

guitar_v6 trained from step 0 through step ~3 M. Listening tests indicated reconstruction quality peaked at mid-Phase 2 (epoch 1935, step ~2.3 M) and degraded slightly by the end (epoch 2511, step ~2.98 M). We did not stop training early because TensorBoard showed the GAN gap reversing direction (rather than collapsing) and we wanted to confirm the full trajectory for the failure-analysis chapter. The final canonical model `guitar_v6_best.ts` is the mid-Phase-2 checkpoint, not the final epoch.

### Decision 6 — Pivot the thesis framing mid-project

Mid-project, when voice-to-guitar was clearly not going to produce a strong perceptual result, the framing was pivoted from "voice-to-guitar prototype" to "voice-to-instrument resynthesis case studies + diagnostic methodology". This is an honest realignment: the autoencoder is guitar (in-distribution), the drums secondary case study is reported with its diagnostic numbers, and the failure analysis is the largest research contribution. The pivot was discussed with the supervisor and reflected in the final title: *BRAVE-Based Real-Time Voice-to-Instrument Resynthesis*.

### Decision 7 — Add an external baseline (IRCAM percussion) to disentangle architectural from training-recipe failure

To distinguish whether the OOD limitation observed in our models is structural to BRAVE or a consequence of training-recipe choices, the official ACIDS-IRCAM percussion model was downloaded (71 MB TorchScript) and tested with the same synthetic-input diagnostic of §3.4. The forward difference between silence and white-noise input was 1.75, versus 0.0045 for our drums_v1 — approximately 390× more responsive — and informal Pure Data listening tests indicated audibly modulated percussive output for voice input. This external baseline supports the conclusion of §6.2 that the OOD limitation is a training-recipe rather than architectural failure.

### Decision 8 — Acknowledge the STFT-correlation diagnostic as a calibration limitation rather than retracting it

A cross-checkpoint calibration of the STFT-magnitude-correlation diagnostic (§5.1.4) revealed that no checkpoint in the iteration sequence passes the 0.3 heuristic proposed initially in §3.4, and that collapsed checkpoints score equal to or slightly higher than the working guitar_v6 canonical checkpoint (Table 5.2). Rather than removing the metric, the methodology was retained with an explicit calibration limitation: collapsed near-silent outputs trivially correlate with input through shared spectral envelope structure. Output amplitude and informal listening evaluation became the operational evaluators for canonical-checkpoint selection. The 0.3 threshold is now framed as a working initial proposal rather than a validated criterion, with re-derivation flagged as future work (§6.3 / §6.4).

### Decision 9 — Hybrid Path A explored as preliminary study, not pursued for thesis demonstration

A hybrid pipeline combining symbolic vocal-percussion onset classification (small CNN trained on the AVP dataset) with sample triggering was prototyped as a preliminary exploration of how to bridge the cross-modal gap. The classifier reached ~87 % offline accuracy on calibrated user-specific data but the deployed Pure Data pipeline did not reach a perceptually compelling level for live demonstration. The exploration is mentioned in §6.4 as a candidate future direction; no quantitative results from this branch are reported in the memoir.

## Lessons about RAVE training in our setup

These would have saved significant time if known from the start. They go into the **Conclusions and Future Work** chapter:

1. **Constant beta from step 0 always collapses the posterior.** Use a log-space warmup from 0.0001 to the target value over Phase 1.

2. **`LATENT_SIZE=128` is too large for datasets under ~20 h.** Use 16 unless the dataset is very large. KL/dim around 0.04 is the target; below 0.01 is collapse.

3. **The default custom discriminator dominates Phase 2 in our setup.** The official v2 configuration's `update_discriminator_every=4` and combined MultiPeriod + MultiScale discriminator together resolve this for guitar. Manual halving of discriminator capacity is too coarse a tool — it produces the opposite failure (discriminator collapse). The v2.gin recipe did not transfer cleanly from guitar to drums under the same data scale (drums_v2 reproduced the dominance pattern).

4. **Phase 2 quality is non-monotonic.** Reconstruction can be best mid-Phase-2 and degrade by the end. Always listen to multiple checkpoints across the trajectory before selecting the final model.

5. **`rave preprocess` silently deadlocks on:**
   - 24-bit audio files
   - Stereo files (when `--channels 1` is passed)
   - Empty `.wav` files (e.g., `__MACOSX/` metadata files inside ZIP archives)
   - Pipe deadlocks when called from a script with limited stdout/stderr buffering

   Pre-validate every input file before passing it to `rave preprocess`.

6. **Pure Data sample rate must match training sample rate exactly.** A silent 44100 vs 48000 mismatch produces completely unusable output and is hard to diagnose without a sample-rate-aware mind on it. Always verify in `Media → Audio settings`.

7. **`torchaudio.sox_effects` augmentation is broken in newer torchaudio.** The `--augment` flag in `rave train` raises `AttributeError` on import. We trained without augmentation. The upstream `RandomGain` transform also contains a bug (`return x` instead of `return x_amp`); even if augmentation were enabled, gain augmentation would do nothing.

8. **Voice is structurally out-of-distribution for any instrument-trained autoencoder under our training conditions.** The IRCAM external baseline (Decision 7) shows that this is a training-recipe / data-scale limitation rather than an architectural one. Resolving it requires either (a) explicit pitch-conditioning preprocessing (DDSP-style), (b) including voice in the training data, (c) AdaIN-based cross-domain training (RAVE v3), or (d) a hybrid pipeline as briefly explored in Decision 9.

9. **STFT magnitude correlation does not robustly discriminate working from collapsed checkpoints when collapsed outputs are near-silent.** This was the calibration discovery of Decision 8. Future replication studies should re-derive the reconstruction-validation threshold against both working and collapsed reference models in the target setup.
