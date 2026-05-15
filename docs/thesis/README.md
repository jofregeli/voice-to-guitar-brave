# Thesis LaTeX project

## Structure

```
docs/thesis/
├── main.tex                 # Master document, includes everything
├── references.bib           # BibTeX bibliography
├── chapters/
│   ├── 00_titlepage.tex
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_state_of_the_art.tex   (mostly filled from docs/state_of_the_art.md)
│   ├── 03_methodology.tex        (scaffolded — fill in)
│   ├── 04_implementation.tex     (scaffolded — fill in)
│   ├── 05_results.tex            (scaffolded — fill in, this is the longest)
│   ├── 06_conclusions.tex        (scaffolded — fill in)
│   ├── A_demo_setup.tex          (appendix)
│   ├── B_audio_samples.tex       (appendix)
│   └── C_configs.tex             (appendix — paste gin files)
└── figures/                 # Put figures here (PNG/PDF)
```

## Building

Requires a LaTeX distribution (TeX Live, MiKTeX) with `biber` for bibliography.

```bash
cd docs/thesis
pdflatex main
biber main
pdflatex main
pdflatex main
```

Or use a one-shot:

```bash
latexmk -pdf -bibtex main.tex
```

## VS Code

Recommended extension: **LaTeX Workshop**. It auto-builds on save.

## Writing order

Suggested for the next 4 weeks:

1. **Week 1:** Section 1 (Introduction), polish Section 2 (already filled).
2. **Week 2:** Section 3 (Methodology). Most content already exists in `docs/setup_notes.md`.
3. **Week 3:** Section 5 (Results) — the longest section. TensorBoard screenshots, tables of metrics, audio sample references.
4. **Week 4:** Section 4 (Implementation) and Section 6 (Conclusions). Final polish, references check.

## Status

| Section | Status |
|---------|--------|
| Title page | ✅ Done |
| Abstract | ✅ Drafted (250 words) |
| Introduction | ☐ Outline only |
| State of the Art | ✅ Largely filled from existing markdown draft |
| Methodology | ☐ Outline only |
| Implementation | ☐ Outline only |
| Results | ☐ Outline only |
| Conclusions | ☐ Outline only |
| References | ✅ Initial bibliography committed |
| Appendices | ☐ Outline only |
