# Build the thesis PDFs.
# Run from the repository root:
#   .\scripts\build_thesis.ps1
#
# Produces:
#   docs/thesis/thesis.pdf         - two-sided, openright, blank versos, uppercase
#                                    Roman front matter (TEMPLATE-COMPLIANT; submit this)
#   docs/thesis/thesis_screen.pdf  - one-sided, continuous flow (clean on-screen copy)
#
# Requires XeLaTeX (MiKTeX) and biber on PATH.

$ErrorActionPreference = "Stop"
$thesisDir = Join-Path $PSScriptRoot "..\docs\thesis"
$thesisTex = Join-Path $thesisDir "thesis.tex"

function Invoke-Build {
    param([string]$Label, [string]$OutputName, [bool]$RunBiber)
    Write-Host "=== Building $Label ===" -ForegroundColor Cyan
    Push-Location $thesisDir
    try {
        xelatex -interaction=nonstopmode thesis.tex | Out-Null
        if ($RunBiber) { biber thesis | Out-Null }
        xelatex -interaction=nonstopmode thesis.tex | Out-Null
        xelatex -interaction=nonstopmode thesis.tex | Out-Null
        if ($OutputName -ne "thesis.pdf") {
            Copy-Item -Force "thesis.pdf" $OutputName
        }
        Write-Host "  -> $OutputName" -ForegroundColor Green
    } finally {
        Pop-Location
    }
}

# 1. Two-sided template-compliant build (canonical, already set in thesis.tex)
Invoke-Build -Label "submission (two-sided, template-compliant)" -OutputName "thesis.pdf" -RunBiber $true

# 2. One-sided screen copy (temporary documentclass swap)
$content = Get-Content $thesisTex -Raw
$swapped = $content -replace '\\documentclass\[a4paper,11pt,twoside,openright\]\{book\}', '\documentclass[a4paper,11pt,oneside,openany]{book}'
Set-Content -Path $thesisTex -Value $swapped -NoNewline
try {
    Invoke-Build -Label "screen copy (one-sided)" -OutputName "thesis_screen.pdf" -RunBiber $false
} finally {
    Set-Content -Path $thesisTex -Value $content -NoNewline
    # Rebuild canonical two-sided thesis.pdf to restore submission state
    Invoke-Build -Label "restoring canonical thesis.pdf (two-sided)" -OutputName "thesis.pdf" -RunBiber $false
}

Write-Host "`nBoth PDFs ready in docs/thesis/  (submit thesis.pdf)" -ForegroundColor Green
