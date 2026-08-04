# Tessera

**A unified electro-thermo-optimization framework for through-silicon-via (TSV) networks.**

Tessera replaces slow full-wave EM + thermal solvers with a physics-informed
graph neural network (GNN) **surrogate**: given a TSV array's geometry,
signal/ground arrangement, temperature, and frequency, it predicts the full
complex **S-matrix** in milliseconds. That speed makes large multi-objective
**design-space exploration** — searching arrangements for low crosstalk,
insertion loss, and reflection under electro-thermal coupling — practical.

This repository accompanies our IEEE TCAD paper (see [Citation](#citation)) and
contains everything needed to **run the trained model** and **reproduce the
Optuna-based design-space exploration**. The trained weights are included.

<p align="center">
  <img src="assets/tsv_coopt20x20_sref.gif" width="820"
       alt="TSV electro-thermal co-optimization: S-parameter matrix, temperature field, and coupled convergence over iterations">
</p>

*Above: an electro-thermal co-optimization loop on a 20×20 TSV array — the
S-parameter matrix (top-left), the steady-state temperature field (top-right),
and the coupled convergence of die/array temperature and dissipated power
(bottom). The copper conductivity σ<sub>Cu</sub>(T) is updated each step until the
temperature residual falls below tolerance.*

---

## What's included

| Capability | Entry point |
|---|---|
| **Spec → S-matrix inference** | `tessera.predict_s_matrix(design)` · demo: `examples/predict_smatrix.py` |
| **Optuna design-space exploration** | `python -m tessera.optimize` · demo: `examples/run_optimization.sh` |
| **Trained surrogate + scaler** | `models/best_model.pth`, `models/input_scaler.pt` |

The surrogate uses **7-D node features** `[Type, Radius, Pitch, Height, Liner,
Temperature, Freq]` with FiLM conditioning on the electro-thermal state
(frequency + temperature), a 4-layer graph transformer, and reciprocity-enforcing
dual output heads (node-level S21/S11, edge-level NEXT/FEXT). See the paper for
details.

## Installation

Requires Python ≥ 3.10. Install [PyTorch](https://pytorch.org/get-started/) for
your platform first (CPU is sufficient — a GPU only speeds up large batched
searches), then:

```bash
git clone https://github.com/HiPerCAS/tessera.git
cd tessera
pip install -e .
```

## Quickstart — predict an S-matrix

```python
import numpy as np
from tessera import predict_s_matrix

design = {
    "radius": 5e-6, "pitch": 60e-6, "height": 100e-6, "liner": 0.5e-6,  # metres
    "temperature": 300.0,   # K
    "freq": 15e9,           # Hz
    # +1 = signal via, -1 = ground via, 0 = empty cell
    "arrangement": np.array([[ 1, -1,  1],
                             [-1,  1, -1],
                             [ 1, -1,  1]], dtype=np.int8),
}

S = predict_s_matrix(design)     # complex ndarray, shape [2*n_signal, 2*n_signal]
print(S.shape)
```

Or run the full demo (prints metrics, saves `s_matrix.npy` and a dB heatmap):

```bash
python examples/predict_smatrix.py
```

To predict for many designs, load the model once and reuse it:

```python
from tessera import load_model, predict_s_matrix
model, scaler, device = load_model()
for d in designs:
    S = predict_s_matrix(d, model=model, scaler=scaler, device=device)
```

## Design-space exploration with Optuna

`tessera.optimize` runs an in-memory Optuna ask/tell loop (NSGA-II / TPE / QMC /
random) over signal/ground arrangements on a fixed-topology *G×G* grid, using the
surrogate as a millisecond objective oracle. It reports Pareto **hypervolume vs.
number of surrogate evaluations** and writes the final Pareto fronts and plots.

```bash
# small demo (2 grid sizes, 1 seed, 3k trials)
bash examples/run_optimization.sh

# or drive it directly
python -m tessera.optimize models/best_model.pth \
    --sizes 5 7 9 --samplers nsga2 random --seeds 0 1 2 --n-trials 100000
```

Objectives (all optimized simultaneously): maximize average S21 (transmission),
minimize worst-case S11 (reflection), NEXT and FEXT (near-/far-end crosstalk).
Fixed physical parameters for the search live under `optimization.fixed_params`
in `config.yaml`. Run `python -m tessera.optimize --help` for all options.

## Repository layout

```
tessera/                 the importable package
  inference.py           predict_s_matrix / load_model  (the high-level API)
  model.py               TSVPhysicsGNN (graph transformer + FiLM)
  graph_builder.py       design spec -> PyG graph (7-D nodes, 3-D edges)
  scaler.py              input normalisation
  smatrix.py             graph outputs -> complex S-matrix
  thermal.py             analytical equivalent thermal conductivities
  optimize.py            Optuna multi-objective design-space exploration
models/                  trained weights + input scaler
examples/                runnable demos + a sample design-spec CSV
assets/                  README figures
config.yaml              paths, device, and exploration settings
```

## Model & data

The **trained surrogate** (`models/`) is included so the demos run out of the
box. The **training dataset is not distributed** — the ground-truth S-parameters
were produced with commercial full-wave solvers and are not ours to redistribute.
The model here is a set of learned weights (a `state_dict`); it does not embed or
expose the training data. `examples/arrangements_sample.csv` shows the design-spec
input schema (grid arrangements + geometry), not simulation results.

## Citation

If you use Tessera in your research, please cite:

> M. Gharib, L. Popryho, and I. Partin-Vaisband, "From Physics to Surrogate
> Intelligence: A Unified Electro-Thermo-Optimization Framework for TSV
> Networks," *IEEE Transactions on Computer-Aided Design of Integrated Circuits
> and Systems*, 2026, doi: 10.1109/TCAD.2026.3718807.

Paper: https://ieeexplore.ieee.org/document/11631696

```bibtex
@ARTICLE{11631696,
  author={Gharib, Mohamed and Popryho, Leonid and Partin-Vaisband, Inna},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  title={From Physics to Surrogate Intelligence: A Unified Electro-Thermo-Optimization Framework for TSV Networks},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Through-silicon vias;Arrays;Modeling;Optimization;Design methodology;Joining processes;Scattering parameters;Training;Simulation;Couplings;Through-substrate vias (TSVs);package modeling;electro–thermal modeling;graph neural networks (GNNs);surrogate modeling;S-parameters;Pareto optimization;heterogeneous integration;2.5D/3D IC design automation},
  doi={10.1109/TCAD.2026.3718807}}
```

## License

Released under the [BSD 3-Clause License](LICENSE). Copyright (c) 2026, HiPerCAS Lab.
