"""End-to-end demo: self-consistent closed EM<->thermal loop for a TSV design.

Copper ohmic loss heats the substrate; the higher temperature changes copper
conductivity and hence the S-matrix and the loss; iterate to a fixed point. The
published GNN is temperature-aware (temperature is a node feature), so each outer
iteration is just a re-prediction at the updated mean temperature -- no bespoke
conductivity model is needed on the ML side.

    for k in range(max_iter):
        S      = predict_s_matrix(design @ current mean T)   # GNN
        S      = project_to_passive(S)                        # sigma <= 1
        P_tsv  = per-TSV power from S                         # energy balance
        T      = steady_state_temperature(design, P_tsv)     # 3-D FV solve
        if |dT_mean| < tol: break

Run (from the repository root, after `pip install -e .`):
    python examples/electrothermal_loop.py

Boundary conditions mirror the `electrothermal:` section of config.yaml (the
validated 4x4 @ 100 GHz Ansys-matched case). Converges in a few iterations.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tessera import electrothermal_loop, load_model

# ── Design specification (metres / K / Hz) ──────────────────────────
DESIGN = {
    "radius": 2e-6,
    "pitch": 25e-6,
    "height": 80e-6,
    "liner": 1.5e-6,          # in-distribution (training liner band [1, 3] um)
    "temperature": 300.0,     # loop starts here; updated each iteration
    "freq": 100e9,
    "arrangement": np.array([
        [ 1,  1,  1,  1],
        [ 1,  1, -1,  1],
        [ 1,  1,  1,  1],
        [ 1,  1,  1,  1],
    ], dtype=np.int8),
}

DIE_EXTENT = {"x0": -100e-6, "x1": -50e-6,
              "y0": -100e-6, "y1": -50e-6,
              "z0":   80e-6, "z1":  90e-6}

OUT_FIG = "electrothermal_convergence.png"


def main():
    model, scaler, device = load_model()

    res = electrothermal_loop(
        DESIGN, model=model, scaler=scaler, device=device,
        excitation_mode="all_signal", passivity_project=True,
        # boundary conditions (= config.yaml electrothermal: defaults)
        htc_W_per_m2K=4.0e5, t_amb_K=300.0, sink_area_scale=3.0,
        die_power_W=0.5, htc_top_W_per_m2K=5.0, htc_side_W_per_m2K=10.0,
        die_extent=DIE_EXTENT, z_max=DIE_EXTENT["z1"],
        cells_per_pitch=3, nz=6,
        max_iter=10, tol_T_K=1.0, verbose=True,
    )

    kx, ky, kz = res["K_eff"]
    print(f"\nConverged={res['converged']} at iter {res['converged_at_iter']} "
          f"({res['n_iter']} iterations)")
    print(f"  T_mean = {res['T_mean_K']:7.2f} K  ({res['T_mean_K'] - 273.15:6.2f} degC)")
    print(f"  T_max  = {res['T_max_K']:7.2f} K  ({res['T_max_K'] - 273.15:6.2f} degC)")
    print(f"  total copper dissipation P_diss = {res['P_diss_W']:.4e} W")
    print(f"  K_eff = ({kx:.1f}, {ky:.1f}, {kz:.1f}) W/mK")

    # Convergence trajectory.
    hist = res["history"]
    it = [r["iter"] for r in hist]
    tmean = [r["T_mean_substrate_K"] for r in hist]
    tmax = [r["T_max_substrate_K"] for r in hist]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(it, tmean, "o-", label="T_mean (substrate)")
    ax.plot(it, tmax, "s--", label="T_max (substrate)")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("temperature [K]")
    ax.set_title("Closed EM<->thermal loop convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"\nSaved convergence plot -> {OUT_FIG}")


if __name__ == "__main__":
    main()
