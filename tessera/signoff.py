"""Ansys HFSS <-> Mechanical electrothermal sign-off for a chosen TSV design.

This module turns a user-chosen "optimal" design (the same lowercase-SI design
dict used across :mod:`tessera.inference`, :mod:`tessera.thermal`, and
:mod:`tessera.electrothermal`) into a **standalone, self-contained PyAEDT
script** that drives a bidirectional HFSS <-> Mechanical steady-state
electrothermal loop in Ansys AEDT, and (optionally) runs it.

Two entry points
----------------
``generate_script(design, out_path, ...)``
    Write a standalone sign-off script (a copy of the validated runner template
    with only its ``CONFIG`` block rewritten for *design*). The generated file
    has no dependency on the ``tessera`` package -- copy it to any machine with
    Ansys AEDT 2026.1 + PyAEDT installed and run ``python <out_path>``.

``run_signoff(design, out_path=None, ...)``
    Generate the script *and* run it as a subprocess, then parse and return the
    result JSON. AEDT/PyAEDT is only ever imported inside that subprocess, so
    importing :mod:`tessera.signoff` (and this whole package) never requires
    PyAEDT to be installed.

The heavy PyAEDT logic lives in ``tessera/_signoff_runner.py`` (a lightly
parameterized, byte-faithful port of the validated
``hfss_mech_bidirectional_linux.py``). We only ever rewrite the ``CONFIG`` dict
between its ``# === TESSERA-SIGNOFF-CONFIG-{START,END} ===`` markers; every line
of validated PyAEDT machinery is preserved verbatim.

Notes
-----
* Solver: **Mechanical** (HFSS <-> Mechanical FEA). Icepak is intentionally not
  shipped; ``solver`` must be ``"mechanical"``.
* ``design["temperature"]`` (the GNN operating point) is *not* an Ansys input:
  the sign-off loop self-consistently solves temperature from ambient. The
  ambient / reference temperatures are ``t_ref_cu`` / ``die_t_ref_K`` (default
  300 K); override them via ``**params`` if your ambient differs.
"""

from __future__ import annotations

import copy
import json
import os
import pprint
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np

# Path to the parameterized runner template shipped inside the package.
_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "_signoff_runner.py")

_CONFIG_START = "# === TESSERA-SIGNOFF-CONFIG-START ==="
_CONFIG_END = "# === TESSERA-SIGNOFF-CONFIG-END ==="


# ---------------------------------------------------------------------------
# Canonical defaults (mirror the CONFIG block in _signoff_runner.py). These are
# the single source of truth for the public API; a design + **params override
# them. Geometry keys are filled in from the design at generation time.
# ---------------------------------------------------------------------------
DEFAULT_SIGNOFF_CONFIG: Dict[str, Any] = {
    # AEDT session
    "aedt_version": "2026.1",
    "non_graphical": True,
    "close_on_exit": True,
    # Output (None -> $TESSERA_SIGNOFF_OUTPUT_DIR or ./tessera_signoff_out next
    # to the generated script). run_signoff() always sets this explicitly.
    "output_dir": None,
    # Design names
    "hfss_design": "HFSS_TSV_BI",
    "mech_full_design": "Mech_TSV_full",
    "hfss_setup": "Setup1",
    "hfss_sweep": "LastAdaptive",
    # Geometry (radius_m/pitch_m/height_m/liner_m/freq_hz come from the design;
    # wid_subs_m/l_subs_m default to the array footprint = n*pitch).
    "geom": {
        "radius_m": 2.0e-6,
        "pitch_m": 25.0e-6,
        "height_m": 80.0e-6,
        "liner_m": 1.5e-6,
        "wid_subs_m": 100.0e-6,
        "l_subs_m": 100.0e-6,
        "freq_hz": 100.0e9,
    },
    "arrangement": [
        [1, 1, 1, 1],
        [1, 1, -1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ],
    # HFSS modelling toggles
    "solve_inside_cu": True,
    "use_edit_sources": True,
    # Excitation
    "pin_w": 1.0,
    "port_excited": 1,
    # Region material/padding (SMOKE Region path; REAL uses direct-face BCs)
    "region_material": "air",
    "region_padding_percent": 100.0,
    # Die + T-dependent leakage
    "die_enabled": True,
    "die_material": "custom_silicon",
    "die_power_W": 0.3,
    "die_alpha_W_per_K": 0.005,
    "die_t_ref_K": 300.0,
    "die_p_max_W": 10.0,
    "die_extent_mm": {
        "x0": -0.100, "x1": -0.050,
        "y0": -0.100, "y1": -0.050,
        "z0": 0.080, "z1": 0.090,
    },
    # Boundary conditions
    "heatsink_htc_W_per_m2K": 4.0e5,
    "air_side_htc_W_per_m2K": 10.0,
    "air_top_htc_W_per_m2K": 5.0,
    # Copper conductivity T-dependence
    "sigma_cu_ref": 5.8e7,
    "alpha_cu": 3.93e-3,
    "t_ref_cu": 300.0,
    # Loop control
    "max_iter": 20,
    "tol_T_K": 0.5,
    "save_per_iter": True,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a copy of *base* (dict values merge
    key-wise; everything else is replaced)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _arrangement_to_int_lists(arrangement) -> list:
    """Coerce an arrangement (numpy array or nested sequence) to a plain
    list-of-lists of Python ints (so it renders cleanly as a literal)."""
    arr = np.asarray(arrangement)
    if arr.ndim != 2:
        raise ValueError(f"arrangement must be 2-D, got shape {arr.shape}")
    return [[int(v) for v in row] for row in arr.tolist()]


def _config_from_design(design: Dict[str, Any],
                        params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a full CONFIG dict from DEFAULT_SIGNOFF_CONFIG, the user's design,
    and flat **params overrides.

    Precedence (low -> high): DEFAULT_SIGNOFF_CONFIG < design-derived geometry
    < params. ``params`` may override any top-level CONFIG key, and nested dicts
    (``geom``, ``die_extent_mm``) merge key-wise.
    """
    if "arrangement" not in design:
        raise KeyError("design must include a 2-D 'arrangement' array "
                       "(+1 signal, -1 ground, 0 empty)")

    arrangement = _arrangement_to_int_lists(design["arrangement"])
    n_rows = len(arrangement)
    n_cols = len(arrangement[0]) if arrangement else 0

    # Geometry from the design (SI). Missing keys fall back to the defaults.
    dg = DEFAULT_SIGNOFF_CONFIG["geom"]
    radius = float(design.get("radius", dg["radius_m"]))
    pitch = float(design.get("pitch", dg["pitch_m"]))
    height = float(design.get("height", dg["height_m"]))
    liner = float(design.get("liner", dg["liner_m"]))
    freq = float(design.get("freq", dg["freq_hz"]))
    # Array footprint = grid extent * pitch (the builder adds 3x-pitch margins).
    geom = {
        "radius_m": radius,
        "pitch_m": pitch,
        "height_m": height,
        "liner_m": liner,
        "wid_subs_m": n_cols * pitch,
        "l_subs_m": n_rows * pitch,
        "freq_hz": freq,
    }

    config = copy.deepcopy(DEFAULT_SIGNOFF_CONFIG)
    config["geom"] = geom
    config["arrangement"] = arrangement

    # Convenience: accept a SI-metre die_extent (like tessera.electrothermal)
    # and convert to the mm the runner expects.
    params = dict(params)  # shallow copy so we can pop
    die_extent_m = params.pop("die_extent", None)
    if die_extent_m is not None:
        config["die_extent_mm"] = {k: float(die_extent_m[k]) * 1.0e3
                                   for k in ("x0", "x1", "y0", "y1", "z0", "z1")}

    if params:
        config = _deep_merge(config, params)

    return config


def _render_config_literal(config: Dict[str, Any]) -> str:
    """Render CONFIG as a valid, readable Python source literal (insertion order
    preserved; True/False/None rendered as Python keywords)."""
    body = pprint.pformat(config, indent=4, width=88, sort_dicts=False)
    return "CONFIG = " + body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_script(design: Dict[str, Any],
                    out_path: str,
                    solver: str = "mechanical",
                    **params: Any) -> str:
    """Generate a standalone HFSS <-> Mechanical sign-off script for *design*.

    Parameters
    ----------
    design : dict
        Lowercase-SI design dict. Required: ``arrangement`` (2-D int array,
        +1 signal / -1 ground / 0 empty). Used if present: ``radius``,
        ``pitch``, ``height``, ``liner`` (metres), ``freq`` (Hz).
    out_path : str
        Where to write the generated ``.py`` script.
    solver : str
        Only ``"mechanical"`` (HFSS <-> Mechanical) is supported.
    **params
        Flat overrides for any top-level CONFIG key (e.g. ``max_iter=6``,
        ``die_power_W=0.5``, ``heatsink_htc_W_per_m2K=4e5``,
        ``output_dir="/path"``). Nested dicts (``geom``, ``die_extent_mm``)
        merge key-wise. Convenience: ``die_extent`` (SI metres) is converted to
        ``die_extent_mm``.

    Returns
    -------
    str
        Absolute path to the written script.
    """
    if solver != "mechanical":
        raise NotImplementedError(
            f"solver={solver!r} is not supported; only 'mechanical' "
            "(HFSS <-> Mechanical) sign-off is shipped."
        )

    with open(_TEMPLATE_PATH, "r") as f:
        template = f.read()

    if template.count(_CONFIG_START) != 1 or template.count(_CONFIG_END) != 1:
        raise RuntimeError(
            f"runner template {_TEMPLATE_PATH} is missing its unique CONFIG "
            "markers; refusing to generate."
        )

    config = _config_from_design(design, params)
    config_src = _render_config_literal(config)

    before, _, rest = template.partition(_CONFIG_START)
    _old_config, _, after = rest.partition(_CONFIG_END)
    new_content = (before + _CONFIG_START + "\n"
                   + config_src + "\n"
                   + _CONFIG_END + after)

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(new_content)
    return out_path


def run_signoff(design: Dict[str, Any],
                out_path: Optional[str] = None,
                *,
                python_exe: Optional[str] = None,
                timeout: Optional[float] = None,
                capture_output: bool = True,
                solver: str = "mechanical",
                **params: Any) -> Dict[str, Any]:
    """Generate a sign-off script for *design* and run it as a subprocess.

    PyAEDT / Ansys AEDT is imported only inside the subprocess, so this call is
    import-safe on machines without AEDT (it will simply fail at run time with a
    clear error if AEDT is unavailable).

    Parameters
    ----------
    design : dict
        See :func:`generate_script`.
    out_path : str, optional
        Path for the generated script. Defaults to
        ``./tessera_signoff/signoff_run.py`` under the current directory. The
        script's output directory is set to the script's directory so the result
        JSON lands predictably beside it.
    python_exe : str, optional
        Python interpreter to run the generated script with (default:
        ``sys.executable``). Point this at the AEDT-enabled environment.
    timeout : float, optional
        Seconds before the subprocess is killed (default: no timeout). A full
        HFSS <-> Mechanical loop can take 1-3 h.
    capture_output : bool
        Capture stdout/stderr (default True). Set False to stream live.
    solver : str
        Only ``"mechanical"`` is supported.
    **params
        Overrides forwarded to :func:`generate_script`.

    Returns
    -------
    dict
        ``{script_path, output_dir, returncode, stdout, stderr,
        result_json_path, result}`` where ``result`` is the parsed
        ``signoff_result.json`` (or ``None`` if it was not produced).
    """
    if out_path is None:
        out_path = os.path.join(os.getcwd(), "tessera_signoff", "signoff_run.py")
    out_path = os.path.abspath(out_path)
    output_dir = os.path.dirname(out_path)
    os.makedirs(output_dir, exist_ok=True)

    # Pin the generated script's output dir so we know where the result lands.
    params.setdefault("output_dir", output_dir)

    script_path = generate_script(design, out_path, solver=solver, **params)

    python_exe = python_exe or sys.executable
    proc = subprocess.run(
        [python_exe, script_path],
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )

    result_json_path = os.path.join(output_dir, "signoff_result.json")
    result: Optional[Dict[str, Any]] = None
    if os.path.exists(result_json_path):
        with open(result_json_path) as f:
            result = json.load(f)

    out = {
        "script_path": script_path,
        "output_dir": output_dir,
        "returncode": proc.returncode,
        "stdout": proc.stdout if capture_output else None,
        "stderr": proc.stderr if capture_output else None,
        "result_json_path": result_json_path,
        "result": result,
    }

    if proc.returncode != 0 and result is None:
        raise RuntimeError(
            f"sign-off run failed (returncode={proc.returncode}). "
            f"No result JSON at {result_json_path}.\n"
            f"--- stderr ---\n{proc.stderr or '(captured only when capture_output=True)'}"
        )
    return out


__all__ = [
    "DEFAULT_SIGNOFF_CONFIG",
    "generate_script",
    "run_signoff",
]
