#!/bin/bash
# Regenerate every pedagogical figure in docs/pedagogy/ (a few minutes on the CPU; the model-structure figure renders
# three 96² images and marches a 96² × 160 cache). Run after a change of the model, the fitting stages, the dependencies
# or the timings, then copy the figures to Daniel's Dropbox folder (docs/pedagogy/README.md).
set -e
cd "$(dirname "$0")/../.."
for f in model_structure model_structure_riaf model_specification fitting_pipeline fitting_convergence dependencies profile; do
    echo "== $f"; nice -n 5 julia -t 6 --project=viz viz/pedagogy/$f.jl 2>&1 | grep -v "^\s*@\|└\|┌\|│" | tail -1
done
