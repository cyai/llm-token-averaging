#!/usr/bin/env bash
# Compile presentation/token_averaging.tex → presentation/token_averaging.pdf
# Run from the repo root: bash presentation/compile_pdf.sh
# Removes LaTeX intermediates; keeps .tex, .pdf, and this script.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRES="$ROOT/presentation"
TEX="token_averaging"

cd "$PRES"

pdflatex -interaction=nonstopmode "$TEX.tex"
bibtex "$TEX" || true
pdflatex -interaction=nonstopmode "$TEX.tex"
pdflatex -interaction=nonstopmode "$TEX.tex"

# Drop LaTeX junk; keep .tex, .pdf, and compile scripts
rm -f \
  "$TEX.aux" "$TEX.bbl" "$TEX.blg" "$TEX.log" "$TEX.out" \
  "$TEX.toc" "$TEX.nav" "$TEX.snm" "$TEX.vrb" "$TEX.fls" \
  "$TEX.fdb_latexmk" "$TEX.synctex.gz" missfont.log

echo "PDF written to: $PRES/$TEX.pdf"
