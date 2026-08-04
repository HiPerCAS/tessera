#!/usr/bin/env bash
# Surrogate-driven multi-objective design-space exploration with Optuna.
#
# Searches signal/ground arrangements on fixed-topology grids, using the trained
# GNN surrogate as the (millisecond) objective oracle. NSGA-II vs. random are
# compared by Pareto hypervolume vs. number of surrogate evaluations.
#
# This is a small demo (2 grid sizes, 1 seed, 3k trials, no exhaustive baseline).
# For the full paper-scale sweep, drop the overrides and use config.yaml defaults
# (see `python -m tessera.optimize --help`).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

python -m tessera.optimize models/best_model.pth \
    --sizes 5 7 \
    --samplers nsga2 random \
    --seeds 0 \
    --n-trials 3000 \
    --no-exhaustive \
    --out-dir results/optimization_demo

echo
echo "Done. Results (hypervolume history, Pareto fronts, plots) -> results/optimization_demo/"
