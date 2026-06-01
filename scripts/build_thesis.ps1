# Build both digital (oneside) and print (twoside) versions of the thesis.
# Run from the repository root:
#   .\scripts\build_thesis.ps1
#
# Produces:
#   docs/thesis/thesis.pdf        - oneside, continuous flow (digital submission)
#   docs/thesis/thesis_print.pdf  - twoside, openright with blank versos (printed bound copy)
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

# 1. Oneside (canonical, already set in thesis.tex)
Invoke-Build -Label "digital (oneside)" -OutputName "thesis.pdf" -RunBiber $true

# 2. Twoside (temporary documentclass swap)
$content = Get-Content $thesisTex -Raw
$swapped = $content -replace '\\documentclass\[a4paper,11pt,oneside,openany\]\{book\}', '\documentclass[a4paper,11pt,twoside,openright]{book}'
Set-Content -Path $thesisTex -Value $swapped -NoNewline
try {
    Invoke-Build -Label "print (twoside)" -OutputName "thesis_print.pdf" -RunBiber $false
} finally {
    Set-Content -Path $thesisTex -Value $content -NoNewline
    # Rebuild oneside once to restore thesis.pdf to canonical state
    Invoke-Build -Label "restoring canonical thesis.pdf (oneside)" -OutputName "thesis.pdf" -RunBiber $false
}

Write-Host "`nBoth PDFs ready in docs/thesis/" -ForegroundColor Green
