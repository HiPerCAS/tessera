"""End-to-end demo: TSV design spec -> predicted complex S-matrix.

Given fixed geometry, a temperature, a frequency, and a signal/ground
arrangement, this loads the trained surrogate, predicts the full S-matrix, prints
a few summary metrics, saves the complex matrix to .npy, and writes a dB heatmap.

Run (from the repository root, after `pip install -e .`):
    python examples/predict_smatrix.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tessera import load_model, predict_s_matrix
from tessera.thermal import thermal_conductivity

# ── Design specification ────────────────────────────────────────────
# Geometry in metres; temperature in K; frequency in Hz.
DESIGN = {
    "radius": 5e-6,
    "pitch": 60e-6,
    "height": 100e-6,
    "liner": 0.5e-6,
    "temperature": 300.0,
    "freq": 15e9,
    # +1 = signal via, -1 = ground via, 0 = empty cell
    "arrangement": np.array([
        [ 1,  1, -1,  1, -1],
        [-1, -1,  1, -1,  1],
        [-1,  1, -1,  1, -1],
        [-1,  1, -1,  1,  1],
        [ 1, -1,  1, -1, -1],
    ], dtype=np.int8),
}

OUT_NPY = "s_matrix.npy"
OUT_FIG = "s_matrix.png"


def _db(z):
    return 20.0 * np.log10(np.abs(z) + 1e-12)


def main():
    arr = DESIGN["arrangement"]
    n_sig = int((arr == 1).sum())
    n_gnd = int((arr == -1).sum())
    print(f"Arrangement {arr.shape[0]}x{arr.shape[1]}  |  signals={n_sig}  grounds={n_gnd}")
    print(arr)

    # Load once, then predict (reuse model/scaler for many designs in a loop).
    model, scaler, device = load_model()
    print(f"\nModel loaded on {device}. Predicting S-matrix...")
    s_matrix = predict_s_matrix(DESIGN, model=model, scaler=scaler, device=device)

    n_ports = s_matrix.shape[0]
    s_db = _db(s_matrix)
    # crude port-level summaries straight off the matrix
    diag = np.diagonal(s_db)
    off = s_db[~np.eye(n_ports, dtype=bool)]
    print(f"\nS-matrix: {n_ports}x{n_ports} complex  (2 ports per signal via)")
    print(f"  mean |S_ii| (reflection) : {diag.mean():7.2f} dB")
    print(f"  max  |S_ij| (i!=j)       : {off.max():7.2f} dB   (worst coupling)")

    # bonus: analytical equivalent thermal conductivities for this array
    kx, ky, kz = thermal_conductivity(
        r_via=DESIGN["radius"], pitch=DESIGN["pitch"],
        t_ox=DESIGN["liner"], arrangement=arr)
    print(f"  equivalent K (x,y,z)     : {kx:.1f}, {ky:.1f}, {kz:.1f} W/mK")

    np.save(OUT_NPY, s_matrix)
    print(f"\nSaved complex S-matrix -> {OUT_NPY}")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(s_db, cmap="viridis", aspect="equal")
    ax.set_title(f"Predicted |S| (dB), {n_ports}-port")
    ax.set_xlabel("port j"); ax.set_ylabel("port i")
    fig.colorbar(im, ax=ax, label="|S| [dB]", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"Saved heatmap        -> {OUT_FIG}")


if __name__ == "__main__":
    main()
