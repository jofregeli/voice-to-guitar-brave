# Build the thesis PDF.
# Run from the repository root:
#   .\scripts\build_thesis.ps1
#
# Produces docs/thesis/thesis.pdf -- one-sided, continuous flow (no blank pages
# in the body, per the UPF template), uppercase-Roman front matter, literal
# Times New Roman. This is the submission file.
#
# Requires XeLaTeX (MiKTeX) and biber on PATH.
#
# (For a double-sided bound print copy with binding blanks, temporarily change
#  the \documentclass option 'oneside,openany' to 'twoside,openright' and run
#  the four passes below. Do not submit that version: the template forbids
#  blank pages between the table of contents and the conclusion.)

$ErrorActionPreference = "Stop"
$thesisDir = Join-Path $PSScriptRoot "..\docs\thesis"

Push-Location $thesisDir
try {
    Write-Host "=== Building thesis.pdf (one-sided, template-compliant) ===" -ForegroundColor Cyan
    xelatex -interaction=nonstopmode thesis.tex | Out-Null
    biber thesis | Out-Null
    xelatex -interaction=nonstopmode thesis.tex | Out-Null
    xelatex -interaction=nonstopmode thesis.tex | Out-Null
    Write-Host "  -> thesis.pdf ready" -ForegroundColor Green
} finally {
    Pop-Location
}
