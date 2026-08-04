"""Electro-thermal coupling: steady-state temperature and the closed loop.

This module turns the surrogate's S-matrix into a *thermal* answer and closes the
electro-thermal loop:

1. **EM -> dissipated power.** :func:`compute_per_tsv_power` converts a (passive)
   S-matrix into per-TSV absorbed power via power-wave balance
   ``P_abs = ||a||^2 - ||b||^2`` with ``b = S a``.
2. **Steady-state temperature.** :func:`steady_state_temperature` solves the 3-D
   anisotropic heat equation ``div(K . grad T) + q''' = 0`` on a finite-volume
   grid, using the homogenised anisotropic conductivity from
   :mod:`tessera.thermal` inside the TSV block and bulk silicon outside.
3. **Closed loop.** :func:`electrothermal_loop` alternates surrogate S-matrix
   prediction and the thermal solve until the array temperature converges. The
   published surrogate is *temperature-aware* (temperature is a node feature), so
   the copper-conductivity feedback sigma_Cu(T) is internalised by the model — the
   loop simply feeds the updated mean temperature back into the next prediction.

Example
-------
    import numpy as np
    from tessera.electrothermal import electrothermal_loop

    design = {
        "radius": 2e-6, "pitch": 25e-6, "height": 80e-6, "liner": 0.5e-6,
        "temperature": 300.0, "freq": 100e9,
        "arrangement": np.array([[1, 1, 1, 1],
                                 [1, 1, -1, 1],
                                 [1, 1, 1, 1],
                                 [1, 1, 1, 1]], dtype=np.int8),
    }
    res = electrothermal_loop(design, htc_W_per_m2K=4.0e5, sink_area_scale=3.0)
    print(res["converged"], res["T_max_K"], res["P_diss_W"])

The finite-volume solver requires SciPy; the surrogate coupling requires a loaded
model (see :func:`tessera.load_model`).
"""
from __future__ import annotations

import json
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tessera.thermal import K_SUBS, thermal_conductivity

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAVE_SCIPY = True
except ImportError:                                              # pragma: no cover
    sp = None
    spla = None
    _HAVE_SCIPY = False


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

TSV_COPPER_PREFIX = "TSV_Copper_"

# Grid-resolution defaults.  Tuned so 4x4 ... 9x9 arrays land in the
# few-thousand to few-tens-of-thousand cell regime — sub-second solves.
DEFAULT_CELLS_PER_PITCH = 3       # in-plane resolution inside the active region
DEFAULT_NZ = 6                    # through-thickness layers
DEFAULT_MAX_GRID_CELLS = 120_000  # safety cap; solver warns/coarsens above this
# Coarsening floor on the in-plane grid: each lateral direction always has
# *at most* this many cells.  Prevents extremely large sink_area_scale values
# from blowing the cell budget while preserving the 3-D physics.
_MAX_INPLANE_CELLS = 64

# Copper-conductivity diagnostic (reporting only).  The published surrogate is
# temperature-aware, so sigma_Cu(T) feedback is handled inside the model; these
# constants only drive the sigma_Cu value reported in the loop history.
_SIGMA_CU_REF = 5.8e7   # S/m at _T_REF_K
_ALPHA_CU = 0.00393     # 1/K (copper temperature coefficient)
_T_REF_K = 300.0


# =============================================================================
# Section A.  EM -> dissipated power
# =============================================================================

def _absorbed_per_port_from_a(s_2d: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Per-port absorbed power for an arbitrary complex incident-wave vector a.

    P_in = |a_j|^2 at each port; b = S a; |b_j|^2 is the outgoing power at port
    j.  Net absorbed at port j is |a_j|^2 - |b_j|^2; summing over j gives the
    total dissipated power ||a||^2 - ||b||^2.
    """
    a = np.asarray(a, dtype=complex)
    if a.shape[-1] != s_2d.shape[-1]:
        raise ValueError(f"len(a)={a.shape[-1]} != S size {s_2d.shape[-1]}")
    b = s_2d @ a
    return np.abs(a) ** 2 - np.abs(b) ** 2


def compute_per_tsv_power(s_matrix: np.ndarray,
                          n_signals: int,
                          p_in_W: float = 1.0,
                          excitation_mode: str = "all_signal",
                          excited_signal_idx: int = 0,
                          a_vector: Optional[np.ndarray] = None,
                          ) -> Dict[int, float]:
    """Per-signal-TSV dissipated power from a (passive) S-matrix.

    Parameters
    ----------
    s_matrix : ndarray
        2-D ``(n_ports, n_ports)`` or 3-D ``(n_freq, n_ports, n_ports)``.  For
        3-D the result is the frequency-averaged per-TSV absorbed power.
    n_signals : int
        Number of signal TSVs (``n_ports / 2``; two ports per signal via).
    p_in_W : float
        Incident power at each driven port, in Watts.
    excitation_mode : {"all_signal", "single_port", "custom"}
        ``all_signal`` drives every "top" signal port (port ``2k``) at once;
        ``single_port`` drives only ``excited_signal_idx``; ``custom`` uses the
        supplied ``a_vector`` (``p_in_W`` is then ignored).
    excited_signal_idx : int
        Signal TSV to excite when ``excitation_mode='single_port'``.
    a_vector : ndarray, optional
        Complex incident-wave vector of length ``n_ports`` for ``custom`` mode.

    Returns
    -------
    dict
        ``{signal_tsv_idx -> P_dissipated_W}`` for ``0 .. n_signals-1``.

    Notes
    -----
    For ``single_port`` the per-TSV attribution is unambiguous.  For
    ``all_signal`` / ``custom`` the only exact quantity is the global balance
    ``||a||^2 - ||b||^2``; positive per-TSV values are rescaled to match it (see
    the paper's electro-thermal coupling section).  An S-matrix with
    ``sigma_max > 1`` (mildly non-passive) can yield negative raw power under
    coherent multi-port drive — pass it through :func:`project_to_passive` first
    (the closed loop does this automatically).
    """
    s = np.asarray(s_matrix)
    if s.ndim not in (2, 3):
        raise ValueError(f"s_matrix must be 2D or 3D, got ndim={s.ndim}")
    n_ports = s.shape[-1]
    if n_ports != 2 * n_signals:
        raise ValueError(
            f"S-matrix n_ports={n_ports} != 2*n_signals ({2*n_signals})")

    # Build incident-wave vector.
    if excitation_mode == "all_signal":
        a = np.zeros(n_ports, dtype=complex)
        a[0:n_ports:2] = p_in_W ** 0.5
    elif excitation_mode == "single_port":
        if not (0 <= excited_signal_idx < n_signals):
            raise ValueError(
                f"excited_signal_idx={excited_signal_idx} out of "
                f"[0,{n_signals})")
        a = np.zeros(n_ports, dtype=complex)
        a[2 * excited_signal_idx] = p_in_W ** 0.5
    elif excitation_mode == "custom":
        if a_vector is None:
            raise ValueError("excitation_mode='custom' requires a_vector")
        a = np.asarray(a_vector, dtype=complex)
        if a.shape != (n_ports,):
            raise ValueError(f"a_vector shape {a.shape} != ({n_ports},)")
    else:
        raise ValueError(
            f"unknown excitation_mode={excitation_mode!r}; expected "
            f"'all_signal', 'single_port', or 'custom'")

    # Signed per-port |a|^2 - |b|^2 (no per-port clipping: a matched port
    # transmitting power out is a legitimate negative that cancels the in-flow
    # at the same TSV's other port).
    if s.ndim == 2:
        absorbed_per_port = _absorbed_per_port_from_a(s, a)
    else:
        per_freq = np.stack(
            [_absorbed_per_port_from_a(s[i], a) for i in range(s.shape[0])],
            axis=0)
        absorbed_per_port = per_freq.mean(axis=0)

    per_tsv_signed = np.array(
        [absorbed_per_port[2 * k] + absorbed_per_port[2 * k + 1]
         for k in range(n_signals)], dtype=float)

    # Global energy balance — the gold-standard total dissipation.
    a_abs2 = float((np.abs(a) ** 2).sum())
    b_abs2 = (
        float((np.abs(s @ a) ** 2).sum()) if s.ndim == 2
        else float(np.mean([(np.abs(s[i] @ a) ** 2).sum()
                            for i in range(s.shape[0])]))
    )
    P_global = max(0.0, a_abs2 - b_abs2)

    pos = np.where(per_tsv_signed < 0.0, 0.0, per_tsv_signed)
    if excitation_mode == "single_port":
        per_tsv = pos
    else:
        if pos.sum() > 0 and P_global > 0:
            per_tsv = pos * (P_global / pos.sum())
        elif P_global > 0:
            per_tsv = np.full(n_signals, P_global / n_signals)
        else:
            per_tsv = np.zeros(n_signals)

    return {k: float(per_tsv[k]) for k in range(n_signals)}


def compute_per_tsv_volumetric_heat(p_per_tsv_W: Dict[Any, float],
                                    radius_m: float,
                                    height_m: float) -> Dict[Any, float]:
    """Convert per-TSV dissipated power (W) to a volumetric source (W/m^3),
    treating each TSV as a uniform cylinder: ``q_v = P / (pi r^2 h)``."""
    if radius_m <= 0.0 or height_m <= 0.0:
        raise ValueError(f"radius_m and height_m must be positive "
                         f"(got r={radius_m}, h={height_m}).")
    volume_m3 = np.pi * (radius_m ** 2) * height_m
    return {k: float(p_W / volume_m3) for k, p_W in p_per_tsv_W.items()}


def map_signal_idx_to_copper_name(signal_idx: int,
                                  arrangement: np.ndarray) -> str:
    """Copper-body name (``TSV_Copper_{row}_{col}``) for the ``signal_idx``-th
    signal TSV, scanning the arrangement row-major over signal cells (== 1)."""
    arr = np.asarray(arrangement)
    if signal_idx < 0:
        raise IndexError(f"signal_idx must be >= 0, got {signal_idx}")
    count = 0
    n_rows, n_cols = arr.shape
    for r in range(n_rows):
        for c in range(n_cols):
            if arr[r, c] == 1:
                if count == signal_idx:
                    return f"{TSV_COPPER_PREFIX}{r}_{c}"
                count += 1
    raise IndexError(
        f"signal_idx={signal_idx} out of range; arrangement has "
        f"{count} signal TSV(s).")


def project_to_passive(s_matrix: np.ndarray
                       ) -> Tuple[np.ndarray, float, float]:
    """Project a 2-D S-matrix onto the nearest passive matrix by clipping its
    singular values to <= 1 (minimum Frobenius-norm perturbation).

    Returns ``(S_passive, sigma_max_before, sigma_max_after)``.  When the input
    is already passive it is returned unchanged.
    """
    S = np.asarray(s_matrix)
    if S.ndim != 2:
        raise ValueError(f"project_to_passive expects a 2-D S-matrix, "
                         f"got ndim={S.ndim}")
    sigma_max_pre = float(np.linalg.svd(S, compute_uv=False)[0])
    if sigma_max_pre <= 1.0:
        return S, sigma_max_pre, sigma_max_pre
    U, sv, Vh = np.linalg.svd(S, full_matrices=False)
    sv_clipped = np.minimum(sv, 1.0)
    return (U * sv_clipped) @ Vh, sigma_max_pre, float(sv_clipped[0])


def _coerce_power_dict(p_per_tsv_W: Optional[Dict[Any, float]],
                       arrangement: np.ndarray) -> Dict[str, float]:
    """Accept either ``{body_name: W}`` or ``{signal_idx: W}`` and return a
    body-name-keyed dict.  ``None`` maps to an empty dict (zero power)."""
    if not p_per_tsv_W:
        return {}
    keys = list(p_per_tsv_W.keys())
    if all(isinstance(k, str) for k in keys):
        return {str(k): float(v) for k, v in p_per_tsv_W.items()}
    return {map_signal_idx_to_copper_name(int(k), arrangement): float(v)
            for k, v in p_per_tsv_W.items()}


# =============================================================================
# Section B.  3-D steady-state anisotropic finite-volume thermal solver
# =============================================================================
#
# Solves  div(K . grad T) + q''' = 0  with  K = diag(K_x, K_y, K_z)
# discretised by 7-point finite volumes on a uniform Cartesian grid, with
# harmonic-mean face conductances.  Effective anisotropic K (from
# tessera.thermal) is used inside the active TSV bounding box; isotropic bulk
# silicon (K_SUBS) outside.  See the module docstring and the paper for the
# homogenisation and boundary-condition assumptions.


def _parse_geometry(geom_params: Dict[str, Any],
                    units: Optional[str] = None
                    ) -> Tuple[float, float, float, float, np.ndarray]:
    """Return ``(radius_m, pitch_m, height_m, liner_m, arrangement)`` in SI.

    ``units`` is ``'m'``, ``'mm'``, or ``None`` (auto-detect via
    ``pitch > 1e-3 -> mm``, with a warning).
    """
    for k in ("radius", "pitch", "height", "liner", "arrangement"):
        if k not in geom_params:
            raise KeyError(f"design missing required field: {k!r}")
    arrangement = np.asarray(geom_params["arrangement"], dtype=int)
    if arrangement.ndim != 2 or arrangement.size == 0:
        raise ValueError(
            f"'arrangement' must be a non-empty 2-D array, got "
            f"shape {arrangement.shape}")
    radius = float(geom_params["radius"])
    pitch = float(geom_params["pitch"])
    height = float(geom_params["height"])
    liner = float(geom_params["liner"])
    if units is None:
        if pitch > 1e-3:
            warnings.warn(
                f"Auto-detected mm units (pitch={pitch} > 1e-3); converting "
                "to metres.  Pass units='mm' explicitly to silence this.",
                stacklevel=3)
            units = "mm"
        else:
            units = "m"
    if units == "mm":
        radius *= 1e-3
        pitch *= 1e-3
        height *= 1e-3
        liner *= 1e-3
    elif units != "m":
        raise ValueError(f"Unknown units={units!r}; expected 'm' or 'mm'")

    if not (radius > 0 and pitch > 0 and height > 0):
        raise ValueError(
            f"Geometry must be strictly positive: r={radius}, "
            f"p={pitch}, H={height}")
    if 2.0 * (radius + liner) >= pitch:
        warnings.warn(
            f"Geometric overlap: 2*(radius+liner)={2*(radius+liner):.2e} m "
            f">= pitch={pitch:.2e} m.  Solver will run but the homogenised "
            f"K is no longer physically defensible.",
            stacklevel=3)
    return radius, pitch, height, liner, arrangement


def _list_copper_bodies(arrangement: np.ndarray) -> List[str]:
    """Row-major list of every non-empty cell as ``TSV_Copper_{i}_{j}``."""
    bodies: List[str] = []
    rows, cols = arrangement.shape
    for i in range(rows):
        for j in range(cols):
            if int(arrangement[i, j]) != 0:
                bodies.append(f"{TSV_COPPER_PREFIX}{i}_{j}")
    return bodies


class _Grid:
    """Uniform Cartesian 3-D grid descriptor (all lengths in SI metres)."""

    def __init__(self, *, nx: int, ny: int, nz: int,
                 dx: float, dy: float, dz: float,
                 x_origin: float, y_origin: float,
                 array_x0: float, array_y0: float,
                 array_x1: float, array_y1: float,
                 H_active: Optional[float] = None):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.x_origin = x_origin
        self.y_origin = y_origin
        self.array_x0 = array_x0
        self.array_y0 = array_y0
        self.array_x1 = array_x1
        self.array_y1 = array_y1
        self.xc = x_origin + (np.arange(nx) + 0.5) * dx
        self.yc = y_origin + (np.arange(ny) + 0.5) * dy
        self.zc = (np.arange(nz) + 0.5) * dz
        self.n = nx * ny * nz
        self.cell_vol = dx * dy * dz
        self.H_active = float(H_active) if H_active is not None else float(nz * dz)
        self.nz_active = int(np.sum(self.zc <= self.H_active + 1e-15))
        if self.nz_active < 1:
            self.nz_active = 1

    def lin(self, i: int, j: int, k: int) -> int:
        return ((i * self.ny) + j) * self.nz + k

    def tsv_center_to_ij(self, x: float, y: float) -> Tuple[int, int]:
        i = int((x - self.x_origin) // self.dx)
        j = int((y - self.y_origin) // self.dy)
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            raise ValueError(
                f"TSV at ({x:.3e},{y:.3e}) m lies outside grid "
                f"({self.x_origin:.3e}..{self.x_origin+self.nx*self.dx:.3e}, "
                f"{self.y_origin:.3e}..{self.y_origin+self.ny*self.dy:.3e})")
        return i, j


def _build_grid(arrangement: np.ndarray, pitch: float, height: float,
                sink_area_scale: float, cells_per_pitch: int, nz: int,
                max_cells: int, z_max: Optional[float] = None) -> _Grid:
    """Build the 3-D grid covering the active array plus a lateral margin sized
    by ``sink_area_scale`` (domain side = ``max(1, scale) * max_array_side``)."""
    n_rows, n_cols = arrangement.shape
    L_x = n_cols * pitch
    L_y = n_rows * pitch
    L_max = max(L_x, L_y)
    scale = max(1.0, float(sink_area_scale))
    L_dom = scale * L_max
    L_dom_x = max(L_dom, L_x)
    L_dom_y = max(L_dom, L_y)

    dx_wish = pitch / max(1, int(cells_per_pitch))
    dx = max(dx_wish, L_dom_x / float(_MAX_INPLANE_CELLS))
    dy = max(dx_wish, L_dom_y / float(_MAX_INPLANE_CELLS))
    nz = max(2, int(nz))
    Z_total = float(z_max) if z_max is not None else float(height)
    if Z_total < float(height):
        raise ValueError(f"z_max={Z_total} must be >= height={height}")
    dz = Z_total / nz

    nx = max(2, int(round(L_dom_x / dx)))
    ny = max(2, int(round(L_dom_y / dy)))
    dx = L_dom_x / nx
    dy = L_dom_y / ny

    total = nx * ny * nz
    if total > max_cells:
        scale_factor = math.sqrt(total / max_cells)
        nx = max(2, int(math.ceil(nx / scale_factor)))
        ny = max(2, int(math.ceil(ny / scale_factor)))
        dx = L_dom_x / nx
        dy = L_dom_y / ny
        total = nx * ny * nz
        warnings.warn(
            f"Grid coarsened to nx={nx}, ny={ny}, nz={nz} "
            f"(={total} cells) to stay under max_cells={max_cells}.",
            stacklevel=3)

    x_origin = -(L_dom_x - L_x) / 2.0
    y_origin = -(L_dom_y - L_y) / 2.0
    return _Grid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                 x_origin=x_origin, y_origin=y_origin,
                 array_x0=0.0, array_y0=0.0,
                 array_x1=L_x, array_y1=L_y,
                 H_active=float(height))


def _build_K_field(grid: _Grid, K_eff: Tuple[float, float, float]
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell anisotropic conductivity: effective K inside the active TSV
    bounding box (z <= H_active), isotropic silicon K_SUBS elsewhere."""
    Kx_eff, Ky_eff, Kz_eff = float(K_eff[0]), float(K_eff[1]), float(K_eff[2])
    Kx = np.full((grid.nx, grid.ny, grid.nz), K_SUBS, dtype=np.float64)
    Ky = np.full_like(Kx, K_SUBS)
    Kz = np.full_like(Kx, K_SUBS)
    inside_x = (grid.xc >= grid.array_x0) & (grid.xc <= grid.array_x1)
    inside_y = (grid.yc >= grid.array_y0) & (grid.yc <= grid.array_y1)
    mask2d = inside_x[:, None] & inside_y[None, :]
    nz_a = grid.nz_active
    Kx[mask2d, :nz_a] = Kx_eff
    Ky[mask2d, :nz_a] = Ky_eff
    Kz[mask2d, :nz_a] = Kz_eff
    return Kx, Ky, Kz


def _build_heat_source(grid: _Grid,
                       arrangement: np.ndarray,
                       pitch: float,
                       p_per_tsv_W: Dict[str, float],
                       die_power_W: float,
                       die_extent: Optional[Dict[str, float]] = None,
                       ) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], float]:
    """Per-cell heat input in Watts.

    Each signal TSV's power is deposited uniformly along its column (the
    interposer-thickness layers) into the in-plane cell containing its centre.
    Optional ``die_power_W`` is spread over ``die_extent`` (a 3-D box in metres)
    if given, else uniformly across the top face.
    """
    Q = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float64)
    tsv_ij: Dict[str, Tuple[int, int]] = {}
    total_deposited = 0.0

    rows, cols = arrangement.shape
    nz_a = grid.nz_active
    for r in range(rows):
        for c in range(cols):
            if int(arrangement[r, c]) == 0:
                continue
            name = f"{TSV_COPPER_PREFIX}{r}_{c}"
            x = (c + 0.5) * pitch
            y = (r + 0.5) * pitch
            i, j = grid.tsv_center_to_ij(x, y)
            tsv_ij[name] = (i, j)
            P = float(p_per_tsv_W.get(name, 0.0))
            if P != 0.0:
                Q[i, j, :nz_a] += P / nz_a
                total_deposited += P

    if die_power_W and die_power_W > 0.0:
        if die_extent is not None:
            for k in ("x0", "x1", "y0", "y1", "z0", "z1"):
                if k not in die_extent:
                    raise ValueError(
                        f"die_extent missing required key {k!r}; "
                        f"expected x0,x1,y0,y1,z0,z1 (all in metres)")
            x0, x1 = float(die_extent["x0"]), float(die_extent["x1"])
            y0, y1 = float(die_extent["y0"]), float(die_extent["y1"])
            z0, z1 = float(die_extent["z0"]), float(die_extent["z1"])
            if x1 <= x0 or y1 <= y0 or z1 <= z0:
                raise ValueError(
                    f"die_extent ranges must satisfy x1>x0, y1>y0, z1>z0; "
                    f"got x=[{x0},{x1}] y=[{y0},{y1}] z=[{z0},{z1}]")
            mask_x = (grid.xc >= x0) & (grid.xc <= x1)
            mask_y = (grid.yc >= y0) & (grid.yc <= y1)
            mask_z = (grid.zc >= z0) & (grid.zc <= z1)
            die_mask = (mask_x[:, None, None]
                        & mask_y[None, :, None]
                        & mask_z[None, None, :])
            n_die = int(np.count_nonzero(die_mask))
            if n_die == 0:
                raise ValueError(
                    f"die_extent does not overlap any grid cell. Grid domain "
                    f"is x={grid.x_origin*1e6:.1f}.."
                    f"{(grid.x_origin+grid.nx*grid.dx)*1e6:.1f} um, "
                    f"y={grid.y_origin*1e6:.1f}.."
                    f"{(grid.y_origin+grid.ny*grid.dy)*1e6:.1f} um, "
                    f"z=0..{(grid.nz*grid.dz)*1e6:.1f} um. For a die above the "
                    f"interposer, pass z_max >= die_extent['z1'].")
            Q[die_mask] += float(die_power_W) / n_die
            total_deposited += float(die_power_W)
        else:
            n_face_cells = grid.nx * grid.ny
            Q[:, :, grid.nz - 1] += float(die_power_W) / n_face_cells
            total_deposited += float(die_power_W)
    return Q, tsv_ij, total_deposited


def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise harmonic mean; 0 where either operand is 0 (a void carries
    no conduction across that face)."""
    s = a + b
    out = np.zeros_like(a)
    mask = s > 0
    out[mask] = 2.0 * a[mask] * b[mask] / s[mask]
    return out


def _assemble_and_solve(grid: _Grid,
                        Kx: np.ndarray, Ky: np.ndarray, Kz: np.ndarray,
                        Q: np.ndarray,
                        htc_bot: float, htc_top: float, htc_side: float,
                        t_amb: float) -> np.ndarray:
    """Assemble the 7-point FV system and solve ``A T = b``.

    Face conductance ``G = K_harmonic * A_face / d``.  Robin BC at a face of
    area A with film coefficient h adds ``h*A`` to the diagonal and ``h*A*t_amb``
    to the RHS.  ``htc_bot >= 1e10`` collapses to the Dirichlet limit
    ``T(z=0) = t_amb`` at the backside.
    """
    if not _HAVE_SCIPY:
        raise ImportError(
            "SciPy is required for the 3-D FV solver — `pip install scipy`.")

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    N = nx * ny * nz
    A_yz = dy * dz
    A_xz = dx * dz
    A_xy = dx * dy

    dirichlet_bottom = htc_bot >= 1e10

    row_idx: List[np.ndarray] = []
    col_idx: List[np.ndarray] = []
    val: List[np.ndarray] = []
    b = np.zeros(N, dtype=np.float64)
    diag = np.zeros((nx, ny, nz), dtype=np.float64)

    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                          indexing="ij")
    LIN = ((I * ny) + J) * nz + K              # (nx, ny, nz)

    if nx >= 2:
        K_face = _harmonic_mean(Kx[:-1, :, :], Kx[1:, :, :])
        G = K_face * A_yz / dx
        row_idx.append(LIN[:-1, :, :].ravel())
        col_idx.append(LIN[1:, :, :].ravel())
        val.append(-G.ravel())
        row_idx.append(LIN[1:, :, :].ravel())
        col_idx.append(LIN[:-1, :, :].ravel())
        val.append(-G.ravel())
        diag[:-1, :, :] += G
        diag[1:, :, :] += G

    if ny >= 2:
        K_face = _harmonic_mean(Ky[:, :-1, :], Ky[:, 1:, :])
        G = K_face * A_xz / dy
        row_idx.append(LIN[:, :-1, :].ravel())
        col_idx.append(LIN[:, 1:, :].ravel())
        val.append(-G.ravel())
        row_idx.append(LIN[:, 1:, :].ravel())
        col_idx.append(LIN[:, :-1, :].ravel())
        val.append(-G.ravel())
        diag[:, :-1, :] += G
        diag[:, 1:, :] += G

    if nz >= 2:
        K_face = _harmonic_mean(Kz[:, :, :-1], Kz[:, :, 1:])
        G = K_face * A_xy / dz
        row_idx.append(LIN[:, :, :-1].ravel())
        col_idx.append(LIN[:, :, 1:].ravel())
        val.append(-G.ravel())
        row_idx.append(LIN[:, :, 1:].ravel())
        col_idx.append(LIN[:, :, :-1].ravel())
        val.append(-G.ravel())
        diag[:, :, :-1] += G
        diag[:, :, 1:] += G

    def _apply_robin_face(mask3d: np.ndarray, area_face: float,
                          h: float, t_inf: float):
        if h <= 0.0:
            return
        coef = h * area_face
        diag[mask3d] += coef
        b[LIN[mask3d]] += coef * t_inf

    mask_bot = np.zeros((nx, ny, nz), dtype=bool); mask_bot[:, :, 0] = True
    if not dirichlet_bottom:
        _apply_robin_face(mask_bot, A_xy, htc_bot, t_amb)

    mask_top = np.zeros((nx, ny, nz), dtype=bool); mask_top[:, :, nz - 1] = True
    _apply_robin_face(mask_top, A_xy, htc_top, t_amb)

    if htc_side > 0.0:
        mask = np.zeros((nx, ny, nz), dtype=bool); mask[0, :, :] = True
        _apply_robin_face(mask, A_yz, htc_side, t_amb)
        mask = np.zeros((nx, ny, nz), dtype=bool); mask[nx - 1, :, :] = True
        _apply_robin_face(mask, A_yz, htc_side, t_amb)
        mask = np.zeros((nx, ny, nz), dtype=bool); mask[:, 0, :] = True
        _apply_robin_face(mask, A_xz, htc_side, t_amb)
        mask = np.zeros((nx, ny, nz), dtype=bool); mask[:, ny - 1, :] = True
        _apply_robin_face(mask, A_xz, htc_side, t_amb)

    b += Q.ravel()

    row_idx.append(LIN.ravel()); col_idx.append(LIN.ravel())
    val.append(diag.ravel())

    rows = np.concatenate(row_idx).astype(np.int64)
    cols = np.concatenate(col_idx).astype(np.int64)
    vals = np.concatenate(val).astype(np.float64)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    if dirichlet_bottom:
        bot_idx = LIN[:, :, 0].ravel()
        A = A.tolil()
        for r in bot_idx:
            A.rows[r] = [r]
            A.data[r] = [1.0]
        A = A.tocsr()
        b[bot_idx] = t_amb

    try:
        T_flat = spla.spsolve(A, b)
    except Exception:
        d = A.diagonal()
        d[d == 0] = 1.0
        M = sp.diags(1.0 / d)
        T_flat, info = spla.cg(A, b, M=M, atol=1e-9, maxiter=20_000)
        if info != 0:
            raise RuntimeError(f"CG fallback failed: info={info}")
    return T_flat.reshape((nx, ny, nz))


def _extract_tsv_T(T_field: np.ndarray,
                   tsv_ij: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
    """Column-averaged temperature at each TSV's in-plane location."""
    return {name: float(T_field[i, j, :].mean())
            for name, (i, j) in tsv_ij.items()}


def _validate(T_field: np.ndarray, t_amb_K: float,
              p_per_tsv_W: Dict[str, float], die_power_W: float,
              total_deposited: float) -> List[str]:
    """Return a list of physical-sanity warnings (empty if all checks pass)."""
    msgs: List[str] = []
    if not np.all(np.isfinite(T_field)):
        msgs.append("non-finite temperatures in solution")
    if np.any(T_field < t_amb_K - 1e-6):
        msgs.append(
            f"T < t_amb by more than 1e-6 K at min={float(T_field.min()):.6f} K "
            f"vs t_amb={t_amb_K} K — sign / BC bug suspected")
    declared = float(sum(p_per_tsv_W.values()) + float(die_power_W))
    if not math.isclose(total_deposited, declared, rel_tol=1e-9, abs_tol=1e-12):
        msgs.append(
            f"deposited heat {total_deposited:.6e} W != declared "
            f"{declared:.6e} W")
    if declared == 0.0:
        dev = float(np.max(np.abs(T_field - t_amb_K)))
        if dev > 1e-6:
            msgs.append(f"zero-power solve deviates from t_amb by {dev:.3e} K")
    return msgs


def steady_state_temperature(design: Dict[str, Any],
                             p_per_tsv_W: Optional[Dict[Any, float]] = None,
                             *,
                             htc_W_per_m2K: float = 100.0,
                             t_amb_K: float = 300.0,
                             sink_area_scale: float = 20.0,
                             die_power_W: float = 0.0,
                             units: str = "m",
                             cells_per_pitch: int = DEFAULT_CELLS_PER_PITCH,
                             nz: int = DEFAULT_NZ,
                             max_cells: int = DEFAULT_MAX_GRID_CELLS,
                             htc_top_W_per_m2K: float = 0.0,
                             htc_side_W_per_m2K: float = 0.0,
                             die_extent: Optional[Dict[str, float]] = None,
                             z_max: Optional[float] = None,
                             out_json: Optional[str] = None,
                             verbose: bool = False,
                             ) -> dict:
    """3-D steady-state anisotropic finite-volume thermal solve.

    Parameters
    ----------
    design : dict
        Geometry keys ``radius``, ``pitch``, ``height``, ``liner`` (metres by
        default; see ``units``) and ``arrangement`` (2-D integer array,
        ``+1`` signal, ``-1`` ground, ``0`` empty).  Extra keys are ignored, so
        the same dict used for :func:`tessera.predict_s_matrix` works here.
    p_per_tsv_W : dict, optional
        Per-TSV dissipated power in Watts, keyed by **body name**
        (``TSV_Copper_{r}_{c}``) or by **signal index** (0-based, row-major over
        signal cells).  ``None`` means zero TSV power (e.g. a die-only solve).
    htc_W_per_m2K : float
        Backside convection coefficient (W/m^2/K); ``>= 1e10`` gives the
        Dirichlet limit ``T(z=0)=t_amb_K``.
    t_amb_K : float
        Ambient temperature (K).
    sink_area_scale : float
        Lateral domain extension factor (domain side = ``max(1, scale) * L_max``).
    die_power_W : float
        External die power (W); spread over ``die_extent`` if given, else the
        top face.
    units : {'m', 'mm'}
        Units of the geometry in ``design`` (default metres).
    cells_per_pitch, nz, max_cells : int
        Grid controls (in-plane resolution, through-thickness layers, cell cap).
    htc_top_W_per_m2K, htc_side_W_per_m2K : float
        Optional convection on the top / lateral faces (default adiabatic).
    die_extent : dict, optional
        ``{'x0','x1','y0','y1','z0','z1'}`` (metres) box for ``die_power_W``.
    z_max : float, optional
        Total vertical grid extent (m); defaults to ``height`` (auto-extended to
        a ``die_extent`` ceiling above the interposer).
    out_json : str, optional
        If given, write the scalar results (no 3-D field) to this JSON path.
    verbose : bool
        Print grid / solve statistics.

    Returns
    -------
    dict
        ``T_per_TSV`` (dict name->K), ``T_max``/``T_min``/``T_mean`` (per-TSV),
        ``T_field`` (3-D ndarray), volume/top-face aggregates, ``K_eff``,
        ``grid_nx/ny/nz``, ``deposited_W``, ``wallclock_s``, ``warnings``.
    """
    t0 = time.perf_counter()

    radius, pitch, height, liner, arrangement = _parse_geometry(
        design, units=units)
    p_bodies = _coerce_power_dict(p_per_tsv_W, arrangement)

    known = set(_list_copper_bodies(arrangement))
    unknown = [k for k in p_bodies if k not in known]
    if unknown:
        raise ValueError(
            f"Unknown copper body name(s) in p_per_tsv_W: {unknown}. "
            f"Known: {sorted(known)}")

    Kx_eff, Ky_eff, Kz_eff = thermal_conductivity(
        radius, pitch, liner, arrangement)
    K_eff = (float(Kx_eff), float(Ky_eff), float(Kz_eff))

    z_max_eff = z_max
    if die_extent is not None and "z1" in die_extent:
        z1 = float(die_extent["z1"])
        if z_max_eff is None and z1 > height:
            z_max_eff = z1
        elif z_max_eff is not None and z1 > z_max_eff:
            raise ValueError(
                f"die_extent.z1={z1} exceeds z_max={z_max_eff}; "
                f"raise z_max or lower the die ceiling.")
    grid = _build_grid(arrangement, pitch, height,
                       sink_area_scale=sink_area_scale,
                       cells_per_pitch=cells_per_pitch, nz=nz,
                       max_cells=max_cells, z_max=z_max_eff)
    if verbose:
        print(f"[thermal] grid nx={grid.nx} ny={grid.ny} nz={grid.nz} "
              f"(={grid.n} cells)  dx={grid.dx*1e6:.2f}um  "
              f"dy={grid.dy*1e6:.2f}um  dz={grid.dz*1e6:.2f}um", flush=True)
        print(f"[thermal] K_eff (Kx,Ky,Kz)=({K_eff[0]:.2f}, {K_eff[1]:.2f}, "
              f"{K_eff[2]:.2f}) W/m/K", flush=True)

    Kx, Ky, Kz = _build_K_field(grid, K_eff)
    Q, tsv_ij, deposited = _build_heat_source(
        grid, arrangement, pitch, p_bodies,
        die_power_W=float(die_power_W), die_extent=die_extent)
    T_field = _assemble_and_solve(
        grid, Kx, Ky, Kz, Q,
        htc_bot=float(htc_W_per_m2K),
        htc_top=float(htc_top_W_per_m2K),
        htc_side=float(htc_side_W_per_m2K),
        t_amb=float(t_amb_K))

    T_per_TSV = _extract_tsv_T(T_field, tsv_ij)
    if not T_per_TSV:
        T_max = T_min = T_mean = float("nan")
    else:
        v = np.array(list(T_per_TSV.values()), dtype=float)
        T_max, T_min, T_mean = float(v.max()), float(v.min()), float(v.mean())

    msgs = _validate(T_field, t_amb_K=float(t_amb_K), p_per_tsv_W=p_bodies,
                     die_power_W=float(die_power_W), total_deposited=deposited)
    if verbose and msgs:
        for m in msgs:
            print(f"[thermal] WARN: {m}", flush=True)

    wall = time.perf_counter() - t0
    result = {
        "T_per_TSV": T_per_TSV,
        "T_max": T_max,
        "T_min": T_min,
        "T_mean": T_mean,
        "wallclock_s": wall,
        "grid_nx": grid.nx,
        "grid_ny": grid.ny,
        "grid_nz": grid.nz,
        "K_eff": K_eff,
        "deposited_W": float(deposited),
        "warnings": msgs,
        "T_field": T_field,
        "T_vol_max": float(T_field.max()),
        "T_vol_min": float(T_field.min()),
        "T_vol_mean": float(T_field.mean()),
        "T_top_max": float(T_field[:, :, -1].max()),
        "T_top_min": float(T_field[:, :, -1].min()),
        "T_top_mean": float(T_field[:, :, -1].mean()),
        "H_active": grid.H_active,
        "nz_active": grid.nz_active,
    }

    if out_json:
        serialisable = {k: v for k, v in result.items() if k != "T_field"}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)

    return result


# =============================================================================
# Section C.  Closed electro-thermal loop
# =============================================================================

def electrothermal_loop(design: Dict[str, Any],
                        *,
                        model=None,
                        scaler=None,
                        device=None,
                        # --- EM -> power ---
                        p_in_W_per_port: float = 1.0,
                        excitation_mode: str = "all_signal",
                        excited_signal_idx: int = 0,
                        passivity_project: bool = True,
                        # --- thermal boundary conditions ---
                        htc_W_per_m2K: float = 100.0,
                        t_amb_K: float = 300.0,
                        sink_area_scale: float = 20.0,
                        die_power_W: float = 0.0,
                        htc_top_W_per_m2K: float = 0.0,
                        htc_side_W_per_m2K: float = 0.0,
                        die_extent: Optional[Dict[str, float]] = None,
                        z_max: Optional[float] = None,
                        cells_per_pitch: int = DEFAULT_CELLS_PER_PITCH,
                        nz: int = DEFAULT_NZ,
                        max_cells: int = DEFAULT_MAX_GRID_CELLS,
                        units: str = "m",
                        # --- loop control ---
                        max_iter: int = 10,
                        tol_T_K: float = 1.0,
                        t_init_K: Optional[float] = None,
                        verbose: bool = True,
                        ) -> dict:
    """Run the closed EM-thermal loop with the temperature-aware surrogate.

    Each iteration: (1) predict the S-matrix at the current mean array
    temperature, (2) project it onto the nearest passive matrix, (3) convert to
    per-TSV dissipated power, (4) solve the 3-D steady-state temperature field,
    (5) feed the new mean temperature back.  Because the published surrogate
    takes temperature as a node feature, the copper-conductivity feedback
    sigma_Cu(T) is captured by the model itself — the loop only carries the mean
    temperature between EM and thermal steps.  Iteration stops when the change in
    mean temperature falls below ``tol_T_K``.

    Parameters
    ----------
    design : dict
        Same schema as :func:`tessera.predict_s_matrix` (geometry, ``freq``,
        ``arrangement``; ``temperature`` sets the initial guess if
        ``t_init_K`` is not given).
    model, scaler, device : optional
        A surrogate loaded once via :func:`tessera.load_model`.  If omitted, the
        model is loaded from ``config.yaml`` on the first call.
    p_in_W_per_port, excitation_mode, excited_signal_idx, passivity_project :
        EM-to-power controls (see :func:`compute_per_tsv_power`).  Passivity
        projection is required for ``all_signal`` mode.
    htc_W_per_m2K ... units :
        Thermal boundary conditions and grid controls (see
        :func:`steady_state_temperature`).
    max_iter : int
        Maximum EM-thermal iterations.
    tol_T_K : float
        Convergence tolerance on the change in mean array temperature (K).
    t_init_K : float, optional
        Initial temperature (K).  Defaults to ``design['temperature']`` or 300.
    verbose : bool
        Print a one-line summary per iteration.

    Returns
    -------
    dict
        ``converged`` (bool), ``n_iter``, ``converged_at_iter``,
        ``T_per_TSV``/``T_mean_K``/``T_max_K``/``T_min_K`` (final),
        ``T_field`` (final 3-D field), ``K_eff``, ``s_matrix`` (final, raw),
        ``s_matrix_passive`` (final, projected), ``P_diss_W``,
        ``P_per_TSV_W`` (body-name keyed), ``grid`` (nx,ny,nz), and
        ``history`` (per-iteration diagnostics).
    """
    # Local import avoids any import-time coupling to the inference stack.
    from tessera.inference import load_model, predict_s_matrix

    if model is None or scaler is None:
        model, scaler, device = load_model(device=device)

    arrangement = np.asarray(design["arrangement"], dtype=int)
    n_signals = int(np.sum(arrangement == 1))
    if n_signals == 0:
        raise ValueError("design 'arrangement' has no signal TSVs (cells == 1)")
    n_ports = 2 * n_signals

    if t_init_K is None:
        t_init_K = float(design.get("temperature", 300.0))

    # Per-TSV temperature state for every non-empty cell.
    T_per_TSV = {f"{TSV_COPPER_PREFIX}{r}_{c}": float(t_init_K)
                 for r in range(arrangement.shape[0])
                 for c in range(arrangement.shape[1])
                 if arrangement[r, c] != 0}

    history: List[dict] = []
    T_mean_prev = float(t_init_K)
    converged = False
    converged_at = None
    # Carry final-iteration artefacts out of the loop.
    S = S_for_power = th = None
    p_bodies: Dict[str, float] = {}
    sum_P = 0.0

    for k in range(max_iter):
        T_mean = float(np.mean(list(T_per_TSV.values())))

        # ---- 1. EM: surrogate S-matrix at the current mean temperature ----
        gnn_t0 = time.perf_counter()
        S = predict_s_matrix({**design, "temperature": T_mean},
                             model=model, scaler=scaler, device=device)
        wall_gnn = time.perf_counter() - gnn_t0
        if S.shape != (n_ports, n_ports):
            raise RuntimeError(
                f"predicted S shape {S.shape} != ({n_ports}, {n_ports})")

        # ---- 2. Passivity projection (needed for coherent multi-port drive) ----
        if passivity_project:
            S_for_power, sig_pre, sig_post = project_to_passive(S)
        else:
            S_for_power = S
            sig_pre = sig_post = float(np.linalg.svd(S, compute_uv=False)[0])

        # ---- 3. EM -> per-TSV dissipated power ----
        p_per_signal = compute_per_tsv_power(
            S_for_power, n_signals=n_signals, p_in_W=p_in_W_per_port,
            excitation_mode=excitation_mode,
            excited_signal_idx=excited_signal_idx)
        p_bodies = {map_signal_idx_to_copper_name(i, arrangement): float(P)
                    for i, P in p_per_signal.items()}
        sum_P = float(sum(p_bodies.values()))

        # ---- 4. Thermal: 3-D steady-state solve ----
        th_t0 = time.perf_counter()
        th = steady_state_temperature(
            design, p_bodies,
            htc_W_per_m2K=htc_W_per_m2K, t_amb_K=t_amb_K,
            sink_area_scale=sink_area_scale, die_power_W=die_power_W,
            htc_top_W_per_m2K=htc_top_W_per_m2K,
            htc_side_W_per_m2K=htc_side_W_per_m2K,
            die_extent=die_extent, z_max=z_max, units=units,
            cells_per_pitch=cells_per_pitch, nz=nz, max_cells=max_cells,
            verbose=False)
        wall_th = time.perf_counter() - th_t0

        # ---- 5. Update per-TSV temperatures (keep last value if missing) ----
        T_new = {n: float(v) for n, v in th["T_per_TSV"].items()
                 if isinstance(v, (int, float)) and not math.isnan(v)}
        for n in T_per_TSV:
            T_new.setdefault(n, T_per_TSV[n])
        T_per_TSV = T_new

        T_mean_new = float(np.mean(list(T_per_TSV.values())))
        dT_mean = abs(T_mean_new - T_mean_prev)
        # Reported copper conductivity (the model internalises the feedback).
        sigma_Cu = _SIGMA_CU_REF / (1.0 + _ALPHA_CU * (T_mean - _T_REF_K))

        rec = dict(
            iter=k,
            T_drive_K=T_mean,
            sigma_Cu=float(sigma_Cu),
            P_diss_W=sum_P,
            T_mean_substrate_K=float(th["T_mean"]),
            T_max_substrate_K=float(th["T_max"]),
            T_min_substrate_K=float(th["T_min"]),
            max_abs_S=float(np.max(np.abs(S))),
            sigma_max_pre_S=float(sig_pre),
            sigma_max_post_S=float(sig_post),
            passivity_projected=bool(passivity_project and sig_pre > 1.0),
            dT_mean_K=dT_mean,
            K_eff=th.get("K_eff"),
            wall_gnn_s=wall_gnn,
            wall_thermal_s=wall_th,
        )
        history.append(rec)
        if verbose:
            print(f"[iter {k}] T_drive={T_mean:7.2f}K  P_TSV={sum_P:.3e}W  "
                  f"|S|max={rec['max_abs_S']:.3f}  "
                  f"sigma_max={sig_pre:.3f}->{sig_post:.3f}  "
                  f"T_sub(mean/max)={float(th['T_mean']):.2f}/"
                  f"{float(th['T_max']):.2f}K  dT_mean={dT_mean:.3f}K  "
                  f"gnn={wall_gnn:.2f}s  th={wall_th:.2f}s", flush=True)

        if k >= 1 and dT_mean < tol_T_K:
            converged = True
            converged_at = k
            rec["converged"] = True
            if verbose:
                print(f"[converged] iter {k}: dT_mean={dT_mean:.3f}K "
                      f"< tol={tol_T_K}K", flush=True)
            break
        T_mean_prev = T_mean_new

    if not converged and verbose:
        print(f"[done] hit max_iter={max_iter} without converging "
              f"(last dT_mean={history[-1]['dT_mean_K']:.3f}K)", flush=True)

    v = np.array(list(T_per_TSV.values()), dtype=float)
    return {
        "converged": converged,
        "n_iter": len(history),
        "converged_at_iter": converged_at,
        "T_per_TSV": T_per_TSV,
        "T_mean_K": float(v.mean()),
        "T_max_K": float(v.max()),
        "T_min_K": float(v.min()),
        "T_field": th["T_field"] if th is not None else None,
        "K_eff": th["K_eff"] if th is not None else None,
        "s_matrix": S,
        "s_matrix_passive": S_for_power,
        "P_diss_W": sum_P,
        "P_per_TSV_W": p_bodies,
        "grid": (th["grid_nx"], th["grid_ny"], th["grid_nz"]) if th else None,
        "history": history,
    }


# =============================================================================
# Lightweight self-check (no file writes)
# =============================================================================

def _self_check() -> int:
    """Physical sanity invariants for the FV solver (no I/O, no /tmp)."""
    arrangement = np.array([[1, -1, 1, -1],
                            [-1, 1, -1, 1],
                            [1, -1, 1, -1],
                            [-1, 1, -1, 1]], dtype=int)
    design = {"radius": 2e-6, "pitch": 25e-6, "height": 80e-6,
              "liner": 0.5e-6, "arrangement": arrangement}
    bodies = {f"{TSV_COPPER_PREFIX}{i}_{j}": 1.0
              for i in range(4) for j in range(4) if arrangement[i, j] == 1}

    r1 = steady_state_temperature(design, bodies, htc_W_per_m2K=1e10,
                                  t_amb_K=300.0, sink_area_scale=1.0)
    r0 = steady_state_temperature(design, {k: 0.0 for k in bodies},
                                  htc_W_per_m2K=1e10, t_amb_K=300.0,
                                  sink_area_scale=1.0)
    r2 = steady_state_temperature(design, {k: 2.0 for k in bodies},
                                  htc_W_per_m2K=1e10, t_amb_K=300.0,
                                  sink_area_scale=1.0)

    # (1) zero power -> ambient
    assert abs(r0["T_mean"] - 300.0) < 1e-6, "zero-power drift"
    # (2) linearity: 2x power -> 2x dT
    dT1, dT2 = r1["T_mean"] - 300.0, r2["T_mean"] - 300.0
    assert abs(dT2 / max(dT1, 1e-12) - 2.0) < 1e-3, "linearity"
    # (3) energy conservation
    assert math.isclose(sum(bodies.values()), r1["deposited_W"], rel_tol=1e-9), \
        "energy conservation"
    # (4) 180-deg rotation symmetry of the chequerboard
    Td = r1["T_per_TSV"]
    d = abs(Td[f"{TSV_COPPER_PREFIX}0_0"] - Td[f"{TSV_COPPER_PREFIX}3_3"])
    assert d < 5e-3, f"symmetry break {d*1e3:.3f} mK"
    print("[self-check] FV solver sanity invariants PASS "
          f"(dT_mean@1W={dT1:.3f}K, grid={r1['grid_nx']}x{r1['grid_ny']}"
          f"x{r1['grid_nz']})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
