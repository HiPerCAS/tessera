"""Tessera — physics-informed GNN surrogate for TSV-network S-parameters.

Public API:
    predict_s_matrix(design[, model, scaler, device]) -> np.ndarray  (complex)
    load_model([model_path, scaler_path, device])    -> (model, scaler, device)
    TSVPhysicsGNN                                     -> the GNN module

See `examples/predict_smatrix.py` for an end-to-end demo (design spec -> S-matrix)
and `examples/run_optimization.sh` for surrogate-driven Optuna design-space
exploration.
"""

from tessera.inference import load_model, predict_s_matrix
from tessera.model import TSVPhysicsGNN

__all__ = ["predict_s_matrix", "load_model", "TSVPhysicsGNN"]
__version__ = "1.0.0"
