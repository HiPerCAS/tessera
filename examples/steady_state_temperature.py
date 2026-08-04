"""End-to-end demo: TSV design spec -> steady-state temperature field.

This chains the two published physics pieces for a single (open-loop) pass:

    1. predict the S-matrix with the GNN surrogate,
    2. turn it into per-TSV ohmic power (energy balance  P = ||a||^2 - ||b||^2),
    3. solve the 3-D anisotropic steady-state heat equation on the homogenised
       substrate (analytical K_eff from tessera.thermal) with a die + heatsink.

It prints T_mean / T_max / per-TSV temperatures and writes a top-surface
temperature map. For the *self-consistent* version (copper heating raises T,
which changes the S-matrix, which changes the heating...) see
`examples/electrothermal_loop.py`.

Run (from the repository root, after `pip install -e .`):
    python examples/steady_state_temperature.py

The boundary conditions below reproduce the validated 4x4 @ 100 GHz Ansys-matched
case and mirror the `electrothermal:` section of config.yaml.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tessera import load_model, predict_s_matrix
from tessera.electrothermal import (
    compute_per_tsv_power,
    project_to_passive,
    steady_state_temperature,
)

# ── Design specification (metres / K / Hz) ──────────────────────────
DESIGN = {
    "radius": 2e-6,
    "pitch": 25e-6,
    "height": 80e-6,
    "liner": 1.5e-6,          # in-distribution (training liner band [1, 3] um)
    "temperature": 300.0,     # operating point at which we predict the S-matrix
    "freq": 100e9,
    # 4x4 array: 15 signals + 1 ground at (row 1, col 2)
    "arrangement": np.array([
        [ 1,  1,  1,  1],
        [ 1,  1, -1,  1],
        [ 1,  1,  1,  1],
        [ 1,  1,  1,  1],
    ], dtype=np.int8),
}

# A 50x50 um die sits on the substrate top corner (z = 80 -> 90 um). Metres.
DIE_EXTENT = {"x0": -100e-6, "x1": -50e-6,
              "y0": -100e-6, "y1": -50e-6,
              "z0":   80e-6, "z1":  90e-6}

# Boundary conditions (= config.yaml `electrothermal:` defaults).
BCS = dict(
    htc_W_per_m2K=4.0e5,      # substrate-bottom heatsink
    htc_top_W_per_m2K=5.0,    # top-face convection
    htc_side_W_per_m2K=10.0,  # lateral convection
    t_amb_K=300.0,
    sink_area_scale=3.0,
    die_power_W=0.5,          # constant die power over DIE_EXTENT
    die_extent=DIE_EXTENT,
    z_max=DIE_EXTENT["z1"],
    cells_per_pitch=3,
    nz=6,
)

OUT_FIG = "temperature_top.png"


def main():
    arr = DESIGN["arrangement"]
    n_sig = int((arr == 1).sum())
    print(f"Arrangement {arr.shape[0]}x{arr.shape[1]}  |  signals={n_sig}  "
          f"grounds={int((arr == -1).sum())}")

    # 1) predict S-matrix, 2) per-TSV ohmic power (all signal-tops driven).
    model, scaler, device = load_model()
    s_matrix = predict_s_matrix(DESIGN, model=model, scaler=scaler, device=device)
    # Enforce passivity (singular values <= 1) before the energy-balance power
    # calc, exactly as the closed loop does: a raw GNN prediction can be
    # marginally non-passive, which would make ||b|| >= ||a|| and zero out the
    # absorbed power.
    s_passive, sig_pre, sig_post = project_to_passive(s_matrix)
    p_per_tsv = compute_per_tsv_power(s_passive, n_signals=n_sig,
                                      p_in_W=1.0, excitation_mode="all_signal")
    print(f"\nPredicted {s_matrix.shape[0]}-port S-matrix; "
          f"passivity sigma_max {sig_pre:.3f} -> {sig_post:.3f}")
    print(f"total copper dissipation = {sum(p_per_tsv.values()):.4e} W "
          f"(sum over {len(p_per_tsv)} signal TSVs)")

    # 3) steady-state temperature solve (per-TSV power keyed by signal index is
    #    accepted directly; it is mapped onto the copper bodies internally).
    th = steady_state_temperature(DESIGN, p_per_tsv, **BCS)

    kx, ky, kz = th["K_eff"]
    print(f"\nGrid {th['grid_nx']}x{th['grid_ny']}x{th['grid_nz']}  "
          f"K_eff = ({kx:.1f}, {ky:.1f}, {kz:.1f}) W/mK  "
          f"deposited = {th['deposited_W']:.4e} W")
    print(f"  T_mean = {th['T_mean']:7.2f} K  ({th['T_mean'] - 273.15:6.2f} degC)")
    print(f"  T_max  = {th['T_max']:7.2f} K  ({th['T_max'] - 273.15:6.2f} degC)")
    print(f"  T_min  = {th['T_min']:7.2f} K")
    if th["warnings"]:
        print("  warnings:", "; ".join(th["warnings"]))

    print("\nPer-TSV temperature (K):")
    for name, T in sorted(th["T_per_TSV"].items()):
        print(f"    {name:>18s} : {T:7.2f}")

    # Top-surface temperature map.
    top = th["T_field"][:, :, -1].T   # [ny, nx] for imshow row=y
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(top - 273.15, origin="lower", cmap="inferno", aspect="equal")
    ax.set_title("Steady-state top-surface temperature")
    ax.set_xlabel("x cell"); ax.set_ylabel("y cell")
    fig.colorbar(im, ax=ax, label="T [degC]", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"\nSaved top-surface temperature map -> {OUT_FIG}")


if __name__ == "__main__":
    main()
