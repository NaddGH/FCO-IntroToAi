#!/usr/bin/env bash
# Rebuild the paper PDF (run `python run_all.py` first so that paper/generated/ holds REAL results).
set -e
cd "$(dirname "$0")/paper"
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode proposal.tex
