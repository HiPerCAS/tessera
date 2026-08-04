"""Tessera — physics-informed GNN surrogate for TSV-network S-parameters,
with steady-state thermal, closed EM-thermal coupling, and Ansys sign-off.

Public API
----------
S-parameter surrogate:
    predict_s_matrix(design[, model, scaler, device]) -> np.ndarray (complex)
    load_model([model_path, scaler_path, device])     -> (model, scaler, device)
    TSVPhysicsGNN                                      -> the GNN module

Electro-thermal (tessera.electrothermal):
    steady_state_temperature(design[, p_per_tsv_W], ...)      -> dict
    electrothermal_loop(design[, model, scaler, device], ...) -> dict
    compute_per_tsv_power(s_matrix, n_signals, ...)           -> dict

Ansys HFSS <-> Mechanical sign-off (tessera.signoff; needs the [signoff] extra
and a machine with Ansys AEDT + PyAEDT):
    generate_signoff_script(design, out_path, ...) -> str  (standalone .py)
    run_signoff(design[, out_path], ...)           -> dict

See `examples/` for end-to-end demos: predict_smatrix.py, run_optimization.sh,
steady_state_temperature.py, electrothermal_loop.py, and signoff_generate.py.
"""

from tessera.inference import load_model, predict_s_matrix
from tessera.model import TSVPhysicsGNN
from tessera.electrothermal import (
    compute_per_tsv_power,
    electrothermal_loop,
    steady_state_temperature,
)
from tessera.signoff import generate_script as generate_signoff_script
from tessera.signoff import run_signoff

__all__ = [
    # S-parameter surrogate
    "predict_s_matrix",
    "load_model",
    "TSVPhysicsGNN",
    # Electro-thermal
    "steady_state_temperature",
    "electrothermal_loop",
    "compute_per_tsv_power",
    # Ansys sign-off
    "generate_signoff_script",
    "run_signoff",
]
__version__ = "1.1.0"
