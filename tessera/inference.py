"""High-level inference: a TSV design spec -> predicted complex S-matrix.

The surrogate takes a design *specification* (geometry + signal/ground
arrangement + temperature + frequency) and returns the full complex S-matrix in
milliseconds, replacing a full-wave EM solve. No training data or pre-built
dataset is required at inference time — the input graph is constructed on the fly
from the spec (see `tessera.graph_builder`).

Example
-------
    import numpy as np
    from tessera import predict_s_matrix

    design = {
        "radius": 5e-6, "pitch": 60e-6, "height": 100e-6, "liner": 0.5e-6,
        "temperature": 300.0, "freq": 15e9,
        "arrangement": np.array([[1, -1, 1],
                                 [-1, 1, -1],
                                 [1, -1, 1]], dtype=np.int8),  # +1 signal, -1 ground, 0 empty
    }
    S = predict_s_matrix(design)          # complex ndarray, shape [2*n_signal, 2*n_signal]
"""
from __future__ import annotations

import numpy as np
import torch

from tessera.config import get_device, load_config, resolve_path
from tessera.graph_builder import build_inference_data
from tessera.model import TSVPhysicsGNN
from tessera.scaler import InputScaler
from tessera.smatrix import convert_pyg_data_to_result


def load_model(model_path=None, scaler_path=None, device=None):
    """Load the trained surrogate and its input scaler.

    Paths default to the entries in ``config.yaml`` (``models/best_model.pth`` and
    ``models/input_scaler.pt``). Returns ``(model, scaler, device)``.
    """
    cfg = load_config()
    if device is None:
        device = get_device(cfg)
    if model_path is None:
        model_path = resolve_path(cfg["paths"]["model"])
    if scaler_path is None:
        scaler_path = resolve_path(cfg["paths"]["scaler"])

    model = TSVPhysicsGNN().to(device)
    model.load_state_dict(
        torch.load(str(model_path), map_location=device, weights_only=True)
    )
    model.eval()

    scaler = InputScaler()
    scaler.load(str(scaler_path))
    return model, scaler, device


def _normalize_design(design: dict) -> dict:
    """Coerce a user design dict into the graph-builder input schema."""
    arr = np.asarray(design["arrangement"], dtype=np.int8)
    return {
        "radius": float(design["radius"]),
        "pitch": float(design["pitch"]),
        "height": float(design["height"]),
        "liner": float(design["liner"]),
        "temperature": float(design.get("temperature", 300.0)),
        "freq": float(design["freq"]),
        "arrangement": arr,
        "id": design.get("id", 0),
    }


def predict_s_matrix(design: dict, model=None, scaler=None, device=None) -> np.ndarray:
    """Predict the complex S-matrix for one TSV design.

    Parameters
    ----------
    design : dict
        Keys: ``radius``, ``pitch``, ``height``, ``liner`` (metres),
        ``temperature`` (K, default 300), ``freq`` (Hz), and ``arrangement`` —
        a 2-D integer array where ``+1`` marks a signal via, ``-1`` a ground via,
        and ``0`` an empty cell.
    model, scaler, device : optional
        Reuse a model/scaler loaded once with :func:`load_model` to amortise
        loading across many predictions (e.g. inside an optimisation loop). If
        omitted they are loaded from ``config.yaml``.

    Returns
    -------
    numpy.ndarray
        Complex S-matrix of shape ``[2 * n_signal, 2 * n_signal]`` (two ports per
        signal via: input/output).
    """
    if model is None or scaler is None:
        model, scaler, device = load_model(device=device)
    if device is None:
        device = next(model.parameters()).device

    data = build_inference_data(_normalize_design(design))
    data_raw = data.clone()  # keep un-normalised copy for S-matrix reconstruction

    scaler.transform(data)
    data = data.to(device)
    with torch.no_grad():
        pred_node, pred_edge = model(data.x, data.edge_index, data.edge_attr)

    data_raw.y_node = pred_node.cpu()
    data_raw.y_edge = pred_edge.cpu()
    return convert_pyg_data_to_result(data_raw)["s_matrix"]
