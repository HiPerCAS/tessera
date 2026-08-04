"""Generate an Ansys HFSS <-> Mechanical sign-off script for a chosen design.

Once design-space exploration (tessera.optimize) has produced an "optimal" TSV
arrangement, this turns that design into a **standalone PyAEDT script** that runs
the full bidirectional HFSS <-> Mechanical steady-state electrothermal loop in
Ansys AEDT -- the sign-off ground truth for the surrogate's prediction.

This example only *generates* the script; it does not need Ansys or PyAEDT and
runs anywhere. Copy the generated file to a machine with Ansys AEDT 2026.1 +
PyAEDT and run it there (see the printed instructions, or examples/run_signoff.sh).

Run (from the repository root, after `pip install -e .`):
    python examples/signoff_generate.py

To generate *and* run in one call on an AEDT machine, use tessera.run_signoff()
instead (see the commented block at the bottom).
"""
import os

import numpy as np

from tessera import generate_signoff_script

# ── The chosen "optimal" design (e.g. a Pareto point from tessera.optimize) ──
DESIGN = {
    "radius": 2e-6,
    "pitch": 25e-6,
    "height": 80e-6,
    "liner": 1.5e-6,
    "temperature": 300.0,     # ignored by the sign-off (T is solved from ambient)
    "freq": 100e9,
    "arrangement": np.array([
        [ 1,  1,  1,  1],
        [ 1,  1, -1,  1],
        [ 1,  1,  1,  1],
        [ 1,  1,  1,  1],
    ], dtype=np.int8),
    "id": "optimal_candidate",
}

OUT_SCRIPT = os.path.join("signoff", "signoff_optimal_candidate.py")


def main():
    # Any top-level CONFIG key can be overridden here. A shorter loop and a
    # constant die (die_alpha_W_per_K=0) keep the first sign-off run bounded.
    script_path = generate_signoff_script(
        DESIGN,
        OUT_SCRIPT,
        solver="mechanical",              # HFSS <-> Mechanical (only option)
        pin_w=1.0,                        # incident power per signal-top port (W)
        heatsink_htc_W_per_m2K=4.0e5,
        air_side_htc_W_per_m2K=10.0,
        air_top_htc_W_per_m2K=5.0,
        die_power_W=0.5,
        die_alpha_W_per_K=0.0,            # 0 = constant die power (no T-leakage)
        max_iter=6,
        tol_T_K=1.0,
        # die footprint in SI metres (auto-converted to the mm AEDT uses)
        die_extent={"x0": -100e-6, "x1": -50e-6,
                    "y0": -100e-6, "y1": -50e-6,
                    "z0":   80e-6, "z1":  90e-6},
    )

    print(f"Generated standalone sign-off script:\n    {script_path}\n")
    print("This file is self-contained (no dependency on the tessera package).")
    print("Run it on a machine with Ansys AEDT 2026.1 + PyAEDT:\n")
    print(f"    python {script_path}\n")
    print("Outputs (per-iter JSON, Touchstone snapshots, and a final")
    print("signoff_result.json) are written next to the script.")

    # --- Generate AND run in one call (uncomment on an AEDT machine) ---------
    # from tessera import run_signoff
    # out = run_signoff(DESIGN, OUT_SCRIPT, solver="mechanical",
    #                   die_alpha_W_per_K=0.0, max_iter=6, tol_T_K=1.0,
    #                   die_extent={"x0": -100e-6, "x1": -50e-6,
    #                               "y0": -100e-6, "y1": -50e-6,
    #                               "z0":   80e-6, "z1":  90e-6})
    # r = out["result"]
    # print("converged at iter", r.get("converged_at_iter"))
    # print("iters run:", len(r.get("iters", [])))


if __name__ == "__main__":
    main()
