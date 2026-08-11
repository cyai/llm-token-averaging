#!/usr/bin/env bash
# Compile paper/main.tex → paper/main.pdf
# Run from the repo root: bash paper/compile_pdf.sh
# Removes LaTeX intermediates; keeps .tex, .pdf, and this script.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$ROOT/paper"
TEX="main"

cd "$PAPER"

pdflatex -interaction=nonstopmode "$TEX.tex"
bibtex "$TEX" || true
pdflatex -interaction=nonstopmode "$TEX.tex"
pdflatex -interaction=nonstopmode "$TEX.tex"

# Drop LaTeX junk; keep .tex, .pdf, and compile scripts
rm -f \
  "$TEX.aux" "$TEX.bbl" "$TEX.blg" "$TEX.log" "$TEX.out" \
  "$TEX.toc" "$TEX.nav" "$TEX.snm" "$TEX.vrb" "$TEX.fls" \
  "$TEX.fdb_latexmk" "$TEX.synctex.gz" missfont.log

echo "PDF written to: $PAPER/$TEX.pdf"
