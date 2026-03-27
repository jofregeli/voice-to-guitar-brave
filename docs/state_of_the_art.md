# State of the Art — Thesis Section 2 Draft

> **Status:** Working draft. Rewrite in your own words before submission.
> Edit freely — this is a starting point, not a final text.

---

## 2. State of the Art

### 2.1 Neural Audio Synthesis

The synthesis of audio using neural networks has evolved substantially over the past decade, transitioning from rule-based signal processing approaches toward data-driven models capable of producing perceptually convincing musical instrument sounds.

**Early deep learning approaches.** One of the foundational contributions in this area is NSynth (Engel et al., 2017), which introduced the first large-scale neural audio synthesis system. NSynth uses a WaveNet autoencoder trained on a dataset of over 300,000 isolated musical notes across a wide range of instruments and pitches. The model learns a 16-dimensional embedding for each note and can interpolate between instrument timbres. However, NSynth is not designed for real-time operation: inference requires several minutes of compute time per second of audio, making it impractical for live performance applications.

**Differentiable signal processing.** A more interpretable and computationally efficient approach was introduced with DDSP (Differentiable Digital Signal Processing, Engel et al., 2020). Rather than generating raw audio samples or spectral frames, DDSP combines a neural network with explicit digital signal processing components — specifically, a harmonic synthesizer and a filtered noise model. The network predicts the control parameters (fundamental frequency, loudness, noise magnitudes) and the synthesizer generates audio from them. Because the synthesis model respects physical constraints of sound production, DDSP requires far less data and computation than purely data-driven approaches. It has been applied successfully to violin and flute synthesis, and later to voice-to-instrument conversion. The key limitation for our use case is that DDSP requires explicit pitch annotations during training (or a pitch tracker at inference time), whereas RAVE-based approaches learn pitch implicitly from raw audio. DDSP also lacks a dedicated real-time streaming architecture for Pure Data deployment.

**RAVE: Real-time variational autoencoder.** The most directly relevant prior work is RAVE (Realtime Audio Variational autoEncoder, Caillon & Esling, 2021). RAVE is a two-stage generative model based on a Variational Autoencoder (VAE). In the first stage, an encoder compresses audio into a compact latent representation and a decoder reconstructs audio from it. The encoder uses multi-band decomposition for efficiency, and the decoder operates in the spectral domain via a multi-scale STFT reconstruction loss rather than generating raw samples. In the second stage, a normalizing flow prior is trained over the latent space to enable unconditional generation. The key contribution of RAVE is that the encoder-decoder pair is compact enough to run in real-time: the authors demonstrate inference at 40+ times real-time speed on a consumer CPU, enabling live performance applications. RAVE is deployable in Pure Data via the `nn~` external, making it the standard choice for neural timbre transfer in live contexts.

**BRAVE: Streaming real-time RAVE.** BRAVE (Bravely Realtime Audio Variational autoEncoder, Caspe et al., 2025) extends RAVE with a *streaming encoder* — a causal architecture that processes audio in small temporal chunks rather than fixed-size windows. This architectural change reduces end-to-end latency significantly while preserving audio quality, making BRAVE better suited than RAVE for real-time performance scenarios where latency is perceptible. The authors evaluate BRAVE on two datasets: a solo saxophone corpus (Filosax, ~4.5 hours) and an electronic drum kit corpus (~2.8 hours). Results show that BRAVE achieves competitive perceptual quality with RAVE while providing lower latency. An important limitation noted in the paper is the difficulty of accurate pitch rendering for melodic instruments: while the model preserves relative pitch contours, absolute pitch accuracy degrades at extreme registers. This limitation is particularly relevant to our voice-to-guitar application and constitutes one of the research questions we investigate.

---

### 2.2 Timbre Transfer and Voice Conversion

**What is timbre?** Timbre refers to the perceptual quality that distinguishes two sounds with the same pitch and loudness — what makes a guitar sound different from a violin playing the same note. Acoustically, timbre is determined by the spectral envelope (distribution of energy across frequencies), the attack and decay characteristics of notes, and fine-grained temporal modulations. Neural timbre transfer aims to learn a mapping from the timbre of one instrument to another while preserving other musical properties such as pitch and rhythm.

**CycleGAN-based approaches.** TimbreTron (Huang et al., 2018) applied CycleGAN — a generative adversarial network designed for unpaired image-to-image translation — to the problem of timbre transfer. Operating on constant-Q transform (CQT) spectrograms, TimbreTron learns to convert between instrument timbres without requiring paired recordings (i.e., the same melody played on both instruments simultaneously). While the approach demonstrates promising results for piano-to-harpsichord and classical-to-jazz stylistic transfers, it suffers from characteristic GAN instabilities, requires significant data, and does not operate in real-time. The inference involves spectrogram-domain manipulation followed by WaveNet-based vocoding, making it unsuitable for live applications.

**Voice conversion.** Voice conversion (VC) is the closely related task of modifying a speaker's voice to sound like a different speaker while preserving linguistic content. Modern VC systems such as AutoVC (Qian et al., 2019) and SoundStream-based approaches use encoder-decoder architectures with speaker conditioning, achieving high-quality results in offline settings. The cross-modal variant — converting voice to a non-voice instrument — is less studied. The primary challenge is that the acoustic feature spaces of voice and instrument differ in fundamental ways: voice has formant structure driven by the vocal tract, while instruments have body resonance, mechanical excitation, and no phonetic content. RAVE-based approaches sidestep this mismatch by learning a shared latent space that captures timbral structure without explicitly modelling either source.

**Existing voice-to-instrument work.** [TODO: add 1-2 sentences about any relevant paper you find during your reading. Candidates: MIDI-DDSP, Differentiable Wavetable Synthesis, or any 2024-2025 work on voice-to-instrument with RAVE.]

---

### 2.3 Datasets for Neural Audio Synthesis

The quality and quantity of training data are critical factors in RAVE-based synthesis. The model requires continuous, single-instrument audio without silence or background noise, and benefits from diverse musical content that covers a wide range of pitches, dynamics, and playing techniques.

**GuitarSet** (Xi et al., 2018) is a dataset of 360 recordings of acoustic guitar performance by six players across five musical styles (rock, jazz, bossa nova, funk, singer-songwriter). Each recording (~30 seconds) is captured with both a hexaphonic pickup (individual string isolation) and a reference microphone. The dataset includes rich annotations: pitch contours (via the jAMS format), beat positions, chord labels, and playing technique markers. For our training, we use the reference microphone recordings of the solo (monophonic) performances only, yielding approximately 1.5 hours of clean monophonic acoustic guitar. GuitarSet is released under a CC BY 4.0 license.

**Guitar-TECHS** (2025) is a more recent dataset of electric guitar performances recorded in multiple environments by three professional guitarists. It covers six playing techniques (alternate picking, palm muting, vibrato, harmonics, pinch harmonics, string bends), musical excerpts, chord types, and major scale runs, totalling approximately 5.2 hours of audio. The dataset provides multiple simultaneous capture channels, including a direct-injection (DI) signal, amplified microphone recordings, and binaural room recordings. For RAVE training, we use the DI channel, which provides the cleanest isolated guitar signal. Guitar-TECHS is released under CC BY 4.0. Combined with GuitarSet, our guitar training corpus totals approximately 6.7 hours.

**Groove MIDI Dataset** (Gillick et al., 2019) is a dataset of human drum performances by professional drummers, captured as MIDI and audio. The audio portion consists of approximately 13 hours of drum kit recordings at 44.1 kHz. It covers a wide range of musical styles and tempos. For the secondary beatbox-to-drums experiment, we use the full audio split. The Groove MIDI Dataset is released under CC BY 4.0.

**Dataset selection rationale.** The RAVE paper recommends between 3 and 50 hours of training audio. Our guitar corpus (6.7h) sits comfortably within this range while remaining computationally tractable. The use of two complementary guitar datasets provides timbral diversity — acoustic texture from GuitarSet and electric texture from Guitar-TECHS — which we expect to help the model generalise across playing styles. The drums corpus (13h) exceeds the lower bound significantly, which should produce a more robust secondary model.

---

### 2.4 Research Gap and Motivation

The review above reveals a clear gap: while BRAVE has been applied to saxophone and drums, its application to guitar timbre transfer from voice has not been systematically investigated. The specific challenge of voice-to-guitar conversion raises open questions that the BRAVE authors acknowledge but do not address:

1. **Pitch rendering for plucked string instruments.** Guitar sound is characterized by a sharp percussive attack followed by exponential decay. Unlike saxophone, where breath pressure sustains a note, guitar notes cannot be sustained indefinitely. How this temporal envelope interacts with the streaming encoder's representation is unknown.

2. **Timbral gap between voice and guitar.** The acoustic feature space of singing voice differs substantially from guitar. Voice contains formant structure, vibrato, and phonetic content absent from guitar. It is unclear how much of this mismatch the BRAVE latent space can absorb.

3. **Multi-dataset training.** Training BRAVE on a combined acoustic + electric guitar corpus has not been explored. Whether this improves generalisation or degrades timbre fidelity is an empirical question.

This project addresses these gaps by training and evaluating BRAVE on the guitar corpus described above, documenting the qualitative and quantitative characteristics of the resulting timbre transfer, and building a real-time Pure Data system that demonstrates the approach in a live performance context. The secondary beatbox-to-drums experiment, replicating the drums evaluation from the original BRAVE paper on our setup, provides a methodological baseline confirming that our training pipeline is correct before interpreting the more novel guitar results.

---

## References

*(Format according to your university's citation style — these are APA approximate)*

- Caillon, A., & Esling, P. (2021). RAVE: A variational autoencoder for fast and high-quality neural audio synthesis. *arXiv preprint arXiv:2111.05011*.
- Caspe, F., et al. (2025). BRAVE: Bravely Realtime Audio Variational autoEncoder. *arXiv preprint arXiv:2503.11562*.
- Engel, J., et al. (2017). Neural audio synthesis of musical notes with WaveNet autoencoders. *Proceedings of ICML 2017*.
- Engel, J., et al. (2020). DDSP: Differentiable digital signal processing. *Proceedings of ICLR 2020*.
- Gillick, J., et al. (2019). Learning to groove with inverse kinematics. *Proceedings of ICML 2019*.
- Huang, C. Z. A., et al. (2018). TimbreTron: A WaveNet(CycleGAN(CQT(Audio))) pipeline for musical timbre transfer. *arXiv preprint arXiv:1811.09620*.
- Xi, Q., et al. (2018). GuitarSet: A dataset for guitar transcription. *Proceedings of ISMIR 2018*.
