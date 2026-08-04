"""Centralized configuration loader for paths, device, and hyperparameters."""

import os
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # tessera/config.py -> repo root


def load_config():
    """Load and return the project configuration from config.yaml.

    Honors a TESSERA_CUDA env var as a one-time override of
    device.cuda_device (lets ablation runners pin a GPU without editing
    config.yaml). Defaults are preserved when the env var is unset.
    """
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    override = os.environ.get("TESSERA_CUDA")
    if override is not None:
        cfg.setdefault("device", {})["cuda_device"] = int(override)
    return cfg


def resolve_path(relative_path):
    """Resolve a path relative to the project root into an absolute Path."""
    return PROJECT_ROOT / relative_path


def get_device(cfg):
    """Return the torch device specified in the configuration."""
    idx = cfg["device"]["cuda_device"]
    if idx < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{idx}")
