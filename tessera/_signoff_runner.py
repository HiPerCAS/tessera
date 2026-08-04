#!/usr/bin/env python3
"""
hfss_mech_bidirectional_linux.py
================================
Linux + AEDT 2026.1 BIDIRECTIONAL HFSS <-> Mechanical electrothermal coupling.

Adapted from the Windows one-way script:
  windows_setup/latest_scripts/HFSS_RUN_SINGLE_LINE_THERMAL_MECH_REGION_orig.py

What this script does
---------------------
For a single CSV row (or hard-coded geometry) it runs:

  build HFSS design (TSV array)
  build Mechanical "FULL" design (copy HFSS solids + Region + BCs)
  for k in range(MAX_ITER):
      a) (re-)solve HFSS at the current copper conductivity
      b) compute total absorbed power P_diss from S (energy balance)
      c) re-trigger EM-loss mapping  HFSS -> Mechanical
      d) solve Mechanical steady-state thermal
      e) extract T_mean, T_max, T_min, per-TSV T from Mechanical
      f) update copper material conductivity:
           sigma_new = sigma_0 / (1 + alpha * (T_mean - T_ref))
      g) check convergence:  |T_mean(k+1) - T_mean(k)|  < TOL_T

After every iteration we dump:
    iter{k}_summary.json   (scalar metrics)
    iter{k}.s{N}p          (Touchstone snapshot, post-solve)

Purpose
-------
- Get the bidirectional EM<->thermal loop running on Linux.
- Test whether PyAEDT 0.27 + AEDT 2026.1 chokes on Mechanical the way it did
  on Icepak (Icepak attach-by-PID failed and poisoned the HFSS gRPC session).

Differences from the Windows original
-------------------------------------
- Linux paths only (no Windows drive paths anywhere)
- AEDT 2026.1
- Outer iteration loop with copper-sigma update + convergence
- HOMOG (ETC-block) design dropped for the first pass to keep this focused on
  whether the FULL-geometry loop works at all
- Per-iter persistence (JSON + Touchstone)
- Verbose error logging at every PyAEDT call site
"""

import os
import sys
import time
import math
import json
import uuid
import tempfile
import traceback
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# Resilient, import-safe PyAEDT import. On AEDT 2026.1 the package is
# `ansys.aedt.core` (formerly `pyaedt`). Reading / py_compiling this module on
# a machine without AEDT must not fail: the names fall back to None and only an
# actual solve (run_bidirectional_loop) requires a working install.
try:
    from ansys.aedt.core import Desktop, Hfss, Mechanical
    from ansys.aedt.core import constants
    _HAVE_AEDT = True
except ImportError:  # pragma: no cover - exercised only off the AEDT box
    try:
        from pyaedt import Desktop, Hfss, Mechanical  # legacy package name
        from pyaedt import constants
        _HAVE_AEDT = True
    except ImportError:
        Desktop = Hfss = Mechanical = None
        constants = None
        _HAVE_AEDT = False


# ============================================================================
# USER CONTROLS  (CONFIG-driven)
# ============================================================================
# Everything a caller needs to change lives in the CONFIG dict below, between
# the two markers. tessera.signoff.generate_script() rewrites ONLY that dict for
# a user-chosen design; the validated PyAEDT machinery further down is untouched.
# You can also edit CONFIG here and run this file directly on an AEDT machine:
#     python _signoff_runner.py
#
# === TESSERA-SIGNOFF-CONFIG-START ===
CONFIG = {
    # --- AEDT session ---
    "aedt_version":  "2026.1",
    "non_graphical": True,
    "close_on_exit": True,

    # --- Output (None -> $TESSERA_SIGNOFF_OUTPUT_DIR or ./tessera_signoff_out) ---
    "output_dir": None,

    # --- Design names ---
    "hfss_design":      "HFSS_TSV_BI",
    "mech_full_design": "Mech_TSV_full",
    "hfss_setup":       "Setup1",
    "hfss_sweep":       "LastAdaptive",

    # --- TSV geometry (SI, metres / Hz). wid_subs_m,l_subs_m are the ARRAY
    #     footprint; the builder adds 3x-pitch substrate margins around it. ---
    "geom": {
        "radius_m":    2.0e-6,
        "pitch_m":    25.0e-6,
        "height_m":   80.0e-6,
        "liner_m":     1.5e-6,
        "wid_subs_m": 100.0e-6,
        "l_subs_m":   100.0e-6,
        "freq_hz":   100.0e9,
    },
    # --- Arrangement: +1 signal, -1 ground, 0 empty (row-major grid) ---
    "arrangement": [
        [ 1, 1, 1, 1],
        [ 1, 1,-1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
    ],

    # --- HFSS modelling toggles ---
    "solve_inside_cu":  True,   # volumetric J^2*rho ohmic loss inside Cu
    "use_edit_sources": True,   # raw-COM EditSources (IncludePortPostProcessing=False)

    # --- Excitation ---
    "pin_w":        1.0,        # incident power per signal-top port (W)
    "port_excited": 1,          # 1-based; only the per-column P_diss diagnostic

    # --- Region material/padding (SMOKE Region path; REAL uses direct-face BCs) ---
    "region_material":        "air",
    "region_padding_percent": 100.0,

    # --- Silicon die on top of the substrate + T-dependent leakage ---
    #     P_die(T) = die_power_W + die_alpha_W_per_K * (T_die - die_t_ref_K)
    #     Set die_alpha_W_per_K = 0.0 for a constant die power.
    "die_enabled":       True,
    "die_material":      "custom_silicon",
    "die_power_W":       0.3,
    "die_alpha_W_per_K": 0.005,
    "die_t_ref_K":       300.0,
    "die_p_max_W":       10.0,
    "die_extent_mm": {          # model units are mm
        "x0": -0.100, "x1": -0.050,
        "y0": -0.100, "y1": -0.050,
        "z0":  0.080, "z1":  0.090,
    },

    # --- Boundary conditions (applied directly to substrate + die faces) ---
    "heatsink_htc_W_per_m2K": 4.0e5,   # substrate bottom (calibrated heatsink)
    "air_side_htc_W_per_m2K": 10.0,    # substrate sides (natural convection)
    "air_top_htc_W_per_m2K":  5.0,     # die/substrate top (weak convection)

    # --- Copper conductivity T-dependence: sigma(T)=sigma_ref/(1+alpha*(T-Tref)) ---
    "sigma_cu_ref": 5.8e7,
    "alpha_cu":     3.93e-3,
    "t_ref_cu":     300.0,

    # --- Outer-loop control ---
    "max_iter":      20,
    "tol_T_K":       0.5,
    "save_per_iter": True,
}
# === TESSERA-SIGNOFF-CONFIG-END ===

# ----------------------------------------------------------------------------
# Derive module constants from CONFIG. Do NOT edit below; change CONFIG above.
# ----------------------------------------------------------------------------
AEDT_VERSION  = CONFIG["aedt_version"]
NON_GRAPHICAL = CONFIG["non_graphical"]
CLOSE_ON_EXIT = CONFIG["close_on_exit"]

# --- Paths: portable, no hard-coded repo root, no /tmp ---
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = (
    CONFIG.get("output_dir")
    or os.environ.get("TESSERA_SIGNOFF_OUTPUT_DIR")
    or os.path.join(_HERE, "tessera_signoff_out")
)
PROJECT_DIR = os.path.join(OUTPUT_DIR, "aedt_project")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR, exist_ok=True)

HFSS_DESIGN      = CONFIG["hfss_design"]
MECH_FULL_DESIGN = CONFIG["mech_full_design"]
HFSS_SETUP       = CONFIG["hfss_setup"]
HFSS_SWEEP       = CONFIG["hfss_sweep"]

# The parameterized runner always follows the validated REAL_CASE path (die on
# top + direct-face BCs). SMOKE_TEST is retained (False) for the machinery's
# mesh-density branch; there is no separate smoke geometry here.
SMOKE_TEST = False
REAL_CASE  = True

GEOM = {
    "radius_m":   float(CONFIG["geom"]["radius_m"]),
    "pitch_m":    float(CONFIG["geom"]["pitch_m"]),
    "height_m":   float(CONFIG["geom"]["height_m"]),
    "liner_m":    float(CONFIG["geom"]["liner_m"]),
    "wid_subs_m": float(CONFIG["geom"]["wid_subs_m"]),
    "l_subs_m":   float(CONFIG["geom"]["l_subs_m"]),
    "freq_hz":    float(CONFIG["geom"]["freq_hz"]),
}
ARRANGEMENT = [list(row) for row in CONFIG["arrangement"]]

SOLVE_INSIDE_CU  = bool(CONFIG["solve_inside_cu"])
USE_EDIT_SOURCES = bool(CONFIG["use_edit_sources"])

PIN_W        = float(CONFIG["pin_w"])
PORT_EXCITED = int(CONFIG["port_excited"])

# --- Region / BCs (SMOKE Region path retained; unused in REAL_CASE) ---
REGION_NAME            = "Region"
REGION_MATERIAL        = CONFIG.get("region_material", "air")
REGION_PADDING_PERCENT = float(CONFIG.get("region_padding_percent", 100.0))
H_TOP_W_PER_M2K        = 500.0
H_SIDES_W_PER_M2K      = 0.0
H_BOTTOM_W_PER_M2K     = 200.0
BOTTOM_MODE            = "adiabatic"
AMBIENT_TEMP_STR       = "300kel"

# --- REAL_CASE die + direct-face BCs ---
DIE_ENABLED       = bool(CONFIG["die_enabled"])
DIE_NAME          = "Die_Chip"
DIE_MATERIAL      = CONFIG["die_material"]
DIE_POWER_W       = float(CONFIG["die_power_W"])
DIE_ALPHA_W_PER_K = float(CONFIG["die_alpha_W_per_K"])
DIE_T_REF_K       = float(CONFIG["die_t_ref_K"])
DIE_P_MAX_W       = float(CONFIG["die_p_max_W"])
DIE_EXTENT_MM     = dict(CONFIG["die_extent_mm"])

HEATSINK_HTC_W_PER_M2K = float(CONFIG["heatsink_htc_W_per_m2K"])
AIR_SIDE_HTC_W_PER_M2K = float(CONFIG["air_side_htc_W_per_m2K"])
AIR_TOP_HTC_W_PER_M2K  = float(CONFIG["air_top_htc_W_per_m2K"])

# --- Naming conventions ---
SUBSTRATE_NAME = "Silicon_Substrate"
COPPER_PREFIX  = "TSV_Copper_"
LINER_PREFIXES = ("TSV_Liner_", "TSV_Oxide_", "TSV_Ins_")
SUBTRACT_FILTER_PREFIXES = ("TSV_", "RDL_", "Microbump_", "UBM_")

# --- Cu T-dependence ---
SIGMA_CU_REF = float(CONFIG["sigma_cu_ref"])
ALPHA_CU     = float(CONFIG["alpha_cu"])
T_REF_CU     = float(CONFIG["t_ref_cu"])

# --- Loop control ---
MAX_ITER      = int(CONFIG["max_iter"])
TOL_T_K       = float(CONFIG["tol_T_K"])
SAVE_PER_ITER = bool(CONFIG["save_per_iter"])

# --- Touchstone export base ---
TS_BASENAME = "TSV_iter"


# ============================================================================
# Helpers (ported from the Windows original with light cleanup)
# ============================================================================

def _stamp(tag: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {tag}", flush=True)


def _safe_save(app, label: str = "") -> None:
    try:
        app.save_project()
        if label:
            print(f"  saved project ({label})")
    except Exception as e:
        print(f"  WARN: save failed ({label}): {e}")


def _tail_messages(app, n: int = 60) -> List[str]:
    try:
        return app.odesktop.GetMessages(app.project_name, app.design_name, 0)[-n:]
    except Exception:
        return []


def _delete_object_if_exists(app, obj_name: str) -> None:
    try:
        if obj_name in getattr(app.modeler, "solid_names", []):
            app.modeler.delete(obj_name)
            print(f"  cleaned object: {obj_name}")
    except Exception:
        pass


def _get_bbox(app, obj_name: str) -> Tuple[float, float, float, float, float, float]:
    mdl = app.modeler
    if hasattr(mdl, "get_object_bounding_box"):
        bb = mdl.get_object_bounding_box(obj_name)
        if bb and len(bb) == 6:
            return tuple(float(x) for x in bb)
    ed = mdl.oeditor
    for name in ("GetObjectBoundingBox", "GetBoundingBox", "GetObjectBBox"):
        if hasattr(ed, name):
            try:
                bb = getattr(ed, name)(obj_name)
                if bb and len(bb) == 6:
                    return tuple(float(x) for x in bb)
            except Exception:
                pass
    raise RuntimeError(f"Cannot get bounding box for '{obj_name}'.")


def _union_bbox(app, obj_names: List[str]):
    bbs = [_get_bbox(app, n) for n in obj_names]
    xmin = min(bb[0] for bb in bbs); ymin = min(bb[1] for bb in bbs); zmin = min(bb[2] for bb in bbs)
    xmax = max(bb[3] for bb in bbs); ymax = max(bb[4] for bb in bbs); zmax = max(bb[5] for bb in bbs)
    return xmin, ymin, zmin, xmax, ymax, zmax


def _get_face_groups_by_extents(app, obj_name: str, tol: float = 1e-6) -> Dict[str, List[int]]:
    xmin, ymin, zmin, xmax, ymax, zmax = _get_bbox(app, obj_name)
    faces = app.modeler.get_object_faces(obj_name)
    groups = {"x_min": [], "x_max": [], "y_min": [], "y_max": [],
              "z_min": [], "z_max": []}
    for fid in faces:
        c = app.modeler.get_face_center(fid)
        if not c:
            continue
        x, y, z = c
        if   abs(x - xmin) < tol: groups["x_min"].append(fid)
        elif abs(x - xmax) < tol: groups["x_max"].append(fid)
        elif abs(y - ymin) < tol: groups["y_min"].append(fid)
        elif abs(y - ymax) < tol: groups["y_max"].append(fid)
        elif abs(z - zmin) < tol: groups["z_min"].append(fid)
        elif abs(z - zmax) < tol: groups["z_max"].append(fid)
    return groups


def _activate_design(app) -> None:
    """Make `app.design_name` the active design in AEDT (best-effort).

    PyAEDT 0.27 caches odefinition_manager / oanalysis inside the Hfss/
    Mechanical wrapper, so this alone is NOT enough to make material edits
    or post queries work after switching designs. Use raw-COM helpers
    (_edit_material_raw, _scalar_via_fields_reporter) for those.
    """
    # SetActiveProject/Design may fail via gRPC if the design is already
    # active or PyAEDT's session state disagrees; harmless in either case.
    # Silence the warning to avoid log spam during per-body extraction.
    try:
        app.odesktop.SetActiveProject(app.project_name)
    except Exception:
        pass
    try:
        app.odesktop.SetActiveDesign(app.design_name)
    except Exception:
        pass


def _set_solve_inside_raw(hfss: Hfss, body_names: List[str]) -> int:
    """Set 'Solve Inside' = True on each named body via raw oEditor.ChangeProperty,
    matching the GUI's recorded script verbatim. Returns count of bodies set.
    Use this instead of PyAEDT's `obj.solve_inside = True` if PyAEDT silently
    skips (the underlying COM call is identical but PyAEDT may not refresh
    the model state)."""
    try:
        _activate_design(hfss)
        oeditor = hfss.odesign.SetActiveEditor("3D Modeler")
    except Exception as e:
        print(f"  ERROR _set_solve_inside_raw: cannot get editor: {e}")
        return 0
    n_ok = 0
    for name in body_names:
        try:
            oeditor.ChangeProperty([
                "NAME:AllTabs",
                [
                    "NAME:Geometry3DAttributeTab",
                    ["NAME:PropServers", name],
                    [
                        "NAME:ChangedProps",
                        ["NAME:Solve Inside", "Value:=", True],
                    ],
                ],
            ])
            n_ok += 1
        except Exception as e:
            print(f"  WARN solve_inside on '{name}': {e}")
    print(f"  Solve Inside (raw COM): set True on {n_ok}/{len(body_names)} bodies")
    return n_ok


def _edit_sources_raw(hfss: Hfss, top_names: List[str], bot_names: List[str],
                      pin_w: float, include_port_post_processing: bool = False
                      ) -> int:
    """Drive every signal-top port at pin_w (W) and zero every signal-bot
    port, via raw oModule('Solutions').EditSources(...). Matches the GUI's
    recorded script verbatim.

    The GUI recording sets IncludePortPostProcessing=False, which PyAEDT's
    edit_sources defaults to True. That single flag-flip is the difference
    between a working solve and the silent "Error in Solving Setup" we hit.
    """
    _activate_design(hfss)
    try:
        omodule = hfss.odesign.GetModule("Solutions")
    except Exception as e:
        print(f"  ERROR _edit_sources_raw: cannot get Solutions module: {e}")
        return 0

    pin_str = f"{float(pin_w)}W"
    top_set = set(top_names)
    bot_set = set(bot_names)
    ex_avail = list(getattr(hfss, "excitation_names", []))

    # Build the EditSources payload exactly like the GUI recording:
    # first element = global options dict, then one entry per port.
    payload = [[
        "IncludePortPostProcessing:=", bool(include_port_post_processing),
        "UseElementPatternMode:=",     False,
        "SpecifySystemPower:=",        False,
    ]]
    n_top = 0
    for nm in ex_avail:
        port_id = nm.split(":", 1)[0] if ":" in nm else nm
        if port_id in top_set:
            payload.append(["Name:=", nm, "Magnitude:=", pin_str, "Phase:=", "0deg"])
            n_top += 1
        elif port_id in bot_set:
            payload.append(["Name:=", nm, "Magnitude:=", "0W", "Phase:=", "0deg"])
        else:
            payload.append(["Name:=", nm, "Magnitude:=", "0W", "Phase:=", "0deg"])
    try:
        omodule.EditSources(payload)
    except Exception as e:
        print(f"  ERROR _edit_sources_raw: EditSources failed: {e}")
        return 0
    print(f"  EditSources (raw COM, IncludePortPostProc={include_port_post_processing}): "
          f"{n_top} tops driven at {pin_str}; {len(ex_avail) - n_top} matched at 0W")
    return n_top


def _edit_material_raw(app, mat_name: str, conductivity: float) -> bool:
    """Edit copper conductivity via raw AEDT COM, bypassing PyAEDT's stale
    caches. Targets the built-in `copper` material (clones into a
    project-local override on first edit). Spec includes the standard
    copper thermal / mechanical properties so Mechanical's SteadyState
    analyse keeps working after the edit.
    """
    try:
        _activate_design(app)
        oproject = app.odesktop.GetActiveProject()
        defmgr = oproject.GetDefinitionManager()
        spec = [
            f"NAME:{mat_name}",
            "CoordinateSystemType:=", "Cartesian",
            "BulkOrSurfaceType:=", 1,
            ["NAME:PhysicsTypes",
             "set:=", ["Electromagnetic", "Thermal", "Structural"]],
            "permittivity:=",            "1",
            "permeability:=",            "0.999991",
            "conductivity:=",            str(float(conductivity)),
            "thermal_conductivity:=",    "400",
            "mass_density:=",            "8933",
            "specific_heat:=",           "385",
            "youngs_modulus:=",          "115000000000",
            "poissons_ratio:=",          "0.34",
            "thermal_expansion_coefficient:=", "1.77e-05",
        ]
        defmgr.EditMaterial(mat_name, spec)
        return True
    except Exception as e:
        print(f"  ERROR _edit_material_raw({mat_name}): {e}")
        return False


def _list_fields_reporter_quantities(app, verbose: bool = False) -> List[str]:
    """Best-effort enumeration of valid Field Calculator quantity names
    in the active design. Helpful when EnterQty('Temperature') fails."""
    out: List[str] = []
    try:
        fr = app.odesign.GetModule("FieldsReporter")
        for getter in ("GetChildNames", "GetNames", "GetQuantityNames"):
            if hasattr(fr, getter):
                try:
                    names = getattr(fr, getter)()
                    if names:
                        out.extend(list(names))
                except Exception:
                    pass
    except Exception:
        pass
    if verbose:
        print(f"    FR quantity names discovered: {out}")
    return out


def _scalar_via_fields_reporter(app, body_name: str,
                                quantity: str = "Temp",
                                agg: str = "Mean",
                                solution: str = "SteadyState1 : SteadyState",
                                verbose: bool = False,
                                enter_method: str = "EnterQty",
                                ) -> Optional[float]:
    """Compute a body-averaged scalar via the AEDT Field Calculator.

    AEDT API gotcha: CalculatorWrite's FIRST arg is the output FILE PATH,
    and the solution name goes inside the SECOND arg as
    ``["Solution:=", "<setup> : <sweep>"]``. We write a temp file then
    parse the scalar out of it.

    `enter_method` is one of EnterQty / EnterField / EnterScalarFunc -- some
    Mechanical/IcepakFEA builds use different entry method names.
    For IcepakFEA Mechanical on AEDT 2026.1, `EnterQty("Temp")` works.
    """
    op = {"Mean": "Mean", "Max": "Maximum", "Min": "Minimum",
          "Integrate": "Integrate"}.get(agg, agg)
    try:
        _activate_design(app)
        fr = app.odesign.GetModule("FieldsReporter")
        try:
            fr.ClearAllNamedExpr()
        except Exception as e:
            if verbose: print(f"    FR.ClearAllNamedExpr: {e}")
        try:
            getattr(fr, enter_method)(quantity)
        except Exception as e:
            if verbose:
                print(f"    FR.{enter_method}({quantity!r}) EXC: {e}")
            return None
        try:
            fr.EnterVol(body_name)
        except Exception as e:
            if verbose:
                print(f"    FR.EnterVol({body_name!r}) EXC: {e}")
            return None
        try:
            fr.CalcOp(op)
        except Exception as e:
            if verbose:
                print(f"    FR.CalcOp({op!r}) EXC: {e}")
            return None
        # CORRECT signature:
        #   CalculatorWrite(<filename>,
        #                   ["Solution:=", "<setup> : <sweep>"],
        #                   [variations...])
        # The scalar is written to <filename> as plain text.
        import tempfile, os as _os
        tf = tempfile.NamedTemporaryFile(prefix="fr_", suffix=".txt",
                                          delete=False, mode="w")
        tf_path = tf.name
        tf.close()
        try:
            fr.CalculatorWrite(tf_path, ["Solution:=", solution], [])
        except Exception as e:
            if verbose:
                print(f"    FR.CalculatorWrite(<file>,sol={solution!r}) EXC: {e}")
            try: _os.unlink(tf_path)
            except Exception: pass
            return None
        # Parse the output file - typically a single scalar number per line
        try:
            with open(tf_path, "r") as f:
                data = f.read().strip()
        finally:
            try: _os.unlink(tf_path)
            except Exception: pass
        if verbose:
            preview = data.replace("\n", " | ")[:120]
            print(f"    FR[{enter_method}({quantity}),{op},{body_name}] "
                  f"sol={solution!r} -> file: {preview!r}")
        if not data:
            return None
        # Take the last whitespace-separated token (often it's "value [unit]")
        tokens = data.split()
        for tok in tokens:
            try:
                return float(tok)
            except ValueError:
                continue
        return None
    except Exception as e:
        if verbose:
            print(f"    FR[{enter_method}({quantity}),{op},{body_name}] "
                  f"sol={solution!r} OUTER EXC: {e}")
        return None


def _list_field_solutions(app) -> List[str]:
    """Discover valid solution context names for the active design's
    Field Calculator. Helpful when you don't know the exact format
    AEDT expects (e.g. 'SteadyState1 : SteadyState' vs 'Setup1 : Solution')."""
    out: List[str] = []
    try:
        # Try the analysis module first
        oa = app.odesign.GetModule("AnalysisSetup")
        for name in ("GetSweeps", "GetSetups"):
            try:
                seq = getattr(oa, name)()
                if seq:
                    out.extend(list(seq))
            except Exception:
                pass
    except Exception:
        pass
    return out


def _delete_boundaries_by_prefix(app, prefixes_lower: List[str]) -> None:
    try:
        bnd_mod = app.odesign.GetModule("BoundarySetup")
        to_del = []
        for b in getattr(app, "boundaries", []):
            nm = getattr(b, "name", "")
            if any(nm.lower().startswith(p) for p in prefixes_lower):
                to_del.append(nm)
        if to_del:
            bnd_mod.DeleteBoundaries(to_del)
            print(f"  cleaned BCs: {to_del}")
    except Exception as e:
        print(f"  WARN clean BCs: {e}")


def _mapfreq_candidates(freq_hz: float) -> List[str]:
    ghz = freq_hz / 1e9
    out, seen = [], set()
    for c in (f"{ghz:.14f}GHz", f"{ghz:.15g}GHz", f"{ghz}GHz",
              f"{ghz:.6f}GHz", f"{freq_hz:.15g}Hz"):
        if c not in seen:
            seen.add(c); out.append(c)
    return out


# ============================================================================
# HFSS build (compact port of build_tsv_array from the original)
# ============================================================================

def _ensure_custom_materials(hfss: Hfss) -> None:
    """Match the Windows original: use built-in `copper` (cloned implicitly
    when we access it) + built-in `silicon_dioxide` + a new `custom_silicon`
    cloned from `silicon` with conductivity bumped to 10 S/m (lossy Si).

    Inheriting the built-in `copper` material means it already has thermal
    conductivity / mass density / specific heat / Young's modulus etc.,
    which Mechanical's SteadyState analyse needs. No manual thermal-props
    hacks required.
    """
    mats = hfss.materials
    if "custom_silicon" not in mats.material_keys:
        base = mats["silicon"]
        mats.add_material("custom_silicon")
        new = mats["custom_silicon"]
        for attr in ("permittivity", "permeability", "conductivity",
                     "dielectric_loss_tangent", "magnetic_loss_tangent",
                     "thermal_conductivity", "mass_density", "specific_heat",
                     "youngs_modulus", "poissons_ratio",
                     "thermal_expansion_coefficient"):
            try:
                getattr(new, attr).value = getattr(base, attr).value
            except Exception:
                pass
        try:
            new.conductivity.value = 10.0          # lossy Si (paper default)
        except Exception:
            pass
        print("  custom_silicon: cloned from silicon, conductivity=10 S/m")
    # Touch `copper` so it becomes a project-local entry editable per iter
    _ = mats["copper"]
    print("  using built-in materials: copper, silicon_dioxide, "
          "and custom_silicon")


def build_hfss_design(hfss: Hfss, geom: Dict[str, Any], arrangement: List[List[int]],
                     pin_w: float) -> Tuple[int, int]:
    """Build the substrate + TSV array + RECTANGLE lumped ports + full
    top/bottom PEC planes. This is a Linux-side, faithful port of
    `build_tsv_array` from the Windows
    HFSS_RUN_SINGLE_LINE_THERMAL_MECH_REGION_orig.py:

      - units: model_units = "mm"; CSV meters are multiplied by 1e3
      - substrate: 3x margins around the TSV array, full height, material
        custom_silicon (cloned from silicon, sigma=10 S/m)
      - PEC top/bot full-substrate planes at z = height + pec_thickness
        and z = -pec_thickness
      - copper inner cylinders (built-in `copper`)
      - silicon_dioxide outer cylinders (built-in `silicon_dioxide`),
        radius = r + liner
      - rectangle ports in the ZX plane, sizes = [pec_thickness, 2*radius]
      - integration line: AxisDir.ZNeg (top) / ZPos (bot)
      - boolean cleanup: liner -= copper (-> annulus); substrate -= liners
        (-> hollows); both keep_originals=True
      - 10x extent Radiation_Box

    Returns (n_signals, n_ports).
    """
    # --- Clean any previous content ---
    try:
        if hfss.modeler.solid_names:
            hfss.modeler.delete(hfss.modeler.solid_names)
        if hfss.modeler.sheet_names:
            hfss.modeler.delete(hfss.modeler.sheet_names)
        if hfss.boundaries:
            hfss.boundaries.clear()
        # hfss.setups is a list in PyAEDT 0.27 (was a dict in older versions)
        for s in list(hfss.setups):
            name = s.name if hasattr(s, "name") else s
            try:
                hfss.delete_setup(name, sweep=True)
            except TypeError:
                hfss.delete_setup(name)
    except Exception as e:
        print(f"  WARN cleanup: {e}")

    # Same units convention as the original Windows script: model in mm,
    # all geometry inputs are CSV-meters * 1e3.
    hfss.modeler.model_units = "mm"

    # Materials (built-in copper + silicon_dioxide; clone custom_silicon)
    _ensure_custom_materials(hfss)

    rows, cols = len(arrangement), len(arrangement[0])
    radius     = float(geom["radius_m"])  * 1e3        # m -> mm
    pitch      = float(geom["pitch_m"])   * 1e3
    height     = float(geom["height_m"])  * 1e3
    liner      = float(geom["liner_m"])   * 1e3
    pec_thickness = 0.5e-3                              # = 0.5 um (original convention)

    substrate_size_x = cols * pitch
    substrate_size_y = rows * pitch

    # Substrate: 3x footprint margins on -X, -Y AND +X, +Y
    hfss.modeler.create_box(
        [-substrate_size_x, -substrate_size_y, 0.0],
        [3 * substrate_size_x, 3 * substrate_size_y, height],
        name=SUBSTRATE_NAME, material="custom_silicon",
        color=(192, 192, 192),
    )

    # Full top + bottom PEC planes (cover only the TSV-array footprint, NOT
    # the full 3x substrate — matches the original)
    top_plane = hfss.modeler.create_rectangle(
        orientation=constants.Plane.XY,
        origin=[0, 0, height + pec_thickness],
        sizes=[substrate_size_x, substrate_size_y],
    )
    bot_plane = hfss.modeler.create_rectangle(
        orientation=constants.Plane.XY,
        origin=[0, 0, -pec_thickness],
        sizes=[substrate_size_x, substrate_size_y],
    )
    hfss.assign_perfecte_to_sheets(top_plane.name, "TOP_PLANE")
    hfss.assign_perfecte_to_sheets(bot_plane.name, "BOT_PLANE")

    cu_names: List[str] = []
    ox_names: List[str] = []
    signal_ports: List[Tuple[int, int]] = []

    next_port = 1
    for r in range(rows):
        for c in range(cols):
            v = arrangement[r][c]
            if v == 0:
                continue
            is_signal = (v == 1)
            x = (c + 0.5) * pitch
            y = (r + 0.5) * pitch

            cu = f"{COPPER_PREFIX}{r}_{c}"
            ox = f"TSV_Liner_{r}_{c}"

            # Copper inner cylinder (Solve Inside set in a batch below
            # via raw COM, matching the GUI's recorded script behaviour).
            cu_obj = hfss.modeler.create_cylinder(
                orientation="Z", origin=[x, y, 0],
                radius=radius, height=height,
                name=cu, material="copper",
                color=(255, 0, 0) if is_signal else (0, 0, 255),
            )
            cu_names.append(cu)

            # SiO2 outer cylinder (will become an annulus after subtract)
            hfss.modeler.create_cylinder(
                orientation="Z", origin=[x, y, 0],
                radius=radius + liner, height=height,
                name=ox, material="silicon_dioxide",
                color=(255, 0, 0) if is_signal else (0, 0, 255),
            )
            ox_names.append(ox)

            # Rectangle port sheets: ZX plane, thin in Z, full TSV diameter in X
            top_rect = hfss.modeler.create_rectangle(
                orientation=constants.Plane.ZX,
                origin=[x - radius, y, height],
                sizes=[pec_thickness, 2 * radius],
            )
            bot_rect = hfss.modeler.create_rectangle(
                orientation=constants.Plane.ZX,
                origin=[x - radius, y, 0],
                sizes=[-pec_thickness, 2 * radius],
            )

            if is_signal:
                hfss.lumped_port(
                    assignment=top_rect, integration_line=hfss.axis_directions.ZNeg,
                    impedance=50, name=f"{next_port}",
                    renormalize=True, deembed=False,
                )
                hfss.lumped_port(
                    assignment=bot_rect, integration_line=hfss.axis_directions.ZPos,
                    impedance=50, name=f"{next_port + 1}",
                    renormalize=True, deembed=False,
                )
                signal_ports.append((next_port, next_port + 1))
                next_port += 2
            else:
                hfss.assign_perfecte_to_sheets(top_rect.name,
                                                f"PerfectE_Top_{r}_{c}")
                hfss.assign_perfecte_to_sheets(bot_rect.name,
                                                f"PerfectE_Bot_{r}_{c}")

    # Boolean cleanup (matches the original ordering):
    # 1) liner -= copper -> hollow ring (keep copper)
    for ox in ox_names:
        cu = ox.replace("TSV_Liner_", COPPER_PREFIX)
        if cu in cu_names:
            hfss.modeler.subtract(blank_list=[ox], tool_list=[cu],
                                  keep_originals=True)
    # 2) substrate -= liners -> cylindrical holes (keep liners)
    if SUBSTRATE_NAME in hfss.modeler.solid_names and ox_names:
        hfss.modeler.subtract(blank_list=[SUBSTRATE_NAME], tool_list=ox_names,
                              keep_originals=True)

    # Radiation box (10x extent on each side)
    hfss.modeler.create_box(
        [-5 * substrate_size_x, -5 * substrate_size_y, -5 * height],
        [10 * substrate_size_x, 10 * substrate_size_y, 10 * height],
        name="Radiation_Box", material="air",
    )
    hfss.assign_radiation_boundary_to_objects("Radiation_Box")

    # Solve setup at the validation frequency (single-point discrete sweep)
    freq_ghz = geom["freq_hz"] / 1e9
    setup = hfss.create_setup(HFSS_SETUP)
    setup.props["Frequency"]          = f"{freq_ghz}GHz"
    # Smoke-test mode loosens convergence + caps passes so HFSS finishes in
    # minutes, not hours.
    setup.props["MaxDeltaS"]          = 0.05 if SMOKE_TEST else 0.02
    setup.props["MaximumPasses"]      = 5    if SMOKE_TEST else 10
    setup.props["MinimumConvergedPasses"] = 1
    setup.props["SaveFields"]         = True
    setup.props["SaveRadFieldsOnly"]  = False
    setup.update()

    # single-point Discrete sweep
    try:
        sweep = setup.add_sweep(name="Sweep1", sweep_type="Discrete")
    except TypeError:
        sweep = setup.add_sweep(sweepname="Sweep1", sweeptype="Discrete")
    sweep.props["RangeType"]  = "SinglePoints"
    sweep.props["RangeStart"] = f"{freq_ghz}GHz"
    sweep.props["RangeEnd"]   = f"{freq_ghz}GHz"
    sweep.update()

    n_signals = len(signal_ports)
    n_ports   = 2 * n_signals
    top_names = [str(p[0]) for p in signal_ports]
    bot_names = [str(p[1]) for p in signal_ports]
    print(f"  HFSS built: n_signals={n_signals}, n_ports={n_ports}, "
          f"freq={freq_ghz}GHz")
    print(f"  HFSS top port names: {top_names[:6]}"
          + (f" ... ({len(top_names)} total)" if len(top_names) > 6 else ""))

    # SOLVE INSIDE on copper TSVs -- via raw COM ChangeProperty to match
    # the GUI's recorded script exactly. PyAEDT's `obj.solve_inside = True`
    # setter is supposed to do the same thing but the GUI-recorded path is
    # known-good. Doing this in a batch after all bodies exist is also
    # faster than per-cylinder setter calls.
    if SOLVE_INSIDE_CU and cu_names:
        _set_solve_inside_raw(hfss, cu_names)

    # CRITICAL: do an INITIAL HFSS solve with default excitations BEFORE
    # the loop's first edit_sources call. The Windows original does this
    # at line 467 of build_tsv_array. Without it, calling edit_sources on
    # a never-solved setup leaves AEDT in a state where analyze_setup
    # errors with "Error in Solving Setup" and produces no field data.
    # The loop's per-iter analyze_setup will then re-use the adaptive
    # mesh from this initial solve (much faster than re-meshing).
    print(f"  HFSS: initial adaptive solve (default all-ports excitation)...")
    t0 = time.perf_counter()
    hfss.analyze()
    print(f"  HFSS initial solve done in {time.perf_counter() - t0:.1f}s")

    return n_signals, n_ports, top_names, bot_names


# ============================================================================
# HFSS: set excitations + (re-)solve + Touchstone export
# ============================================================================

def hfss_set_excitations_and_solve(hfss: Hfss, n_ports: int, pin_w: float,
                                   port_excited: int,
                                   top_names: List[str],
                                   bot_names: List[str]) -> None:
    """Drive ALL signal-TOP ports at `pin_w` (W) simultaneously, leave all
    signal-BOTTOM ports at 0 W (matched 50-ohm loads), then re-solve.

    We build the assignment dict directly from top_names/bot_names
    (returned by build_hfss_design). NO suffix-stripping or convention
    matching -- that previously over-matched and silently drove every
    excitation, which HFSS then rejected as degenerate.
    """
    ex_avail = list(getattr(hfss, "excitation_names", []))
    print(f"  HFSS excitation_names from PyAEDT: {ex_avail[:8]}"
          + (f" ... ({len(ex_avail)} total)" if len(ex_avail) > 8 else ""))

    # PyAEDT 0.27 returns names as "<PortName>:<ModeNum>" (e.g. "1:1") --
    # our build named the ports "1".."30", so port-name PREFIX (before ':')
    # is what we match against top_names/bot_names. Bot ports' suffix is
    # also "1" (the mode number), so suffix-stripping would be wrong;
    # always use the prefix.
    top_set = set(top_names)
    bot_set = set(bot_names)

    # PyAEDT 0.27 expects string values with explicit units (e.g. "1W",
    # "0deg") -- NOT bare floats. The docstring example uses tuples like
    # ("1W", "90deg"). Bare floats might be interpreted differently or
    # fail validation silently, leaving HFSS in a bad state.
    pin_str = f"{float(pin_w)}W"
    assign: Dict[str, tuple] = {}
    n_top = 0
    unknown: List[str] = []
    for nm in ex_avail:
        port_id = nm.split(":", 1)[0] if ":" in nm else nm
        if port_id in top_set:
            assign[nm] = (pin_str, "0deg"); n_top += 1
        elif port_id in bot_set:
            assign[nm] = ("0W", "0deg")
        else:
            unknown.append(nm)

    if n_top == 0:
        raise RuntimeError(
            f"No top ports matched. excitation_names={ex_avail[:8]}... "
            f"Expected to split on ':' and find prefixes in top_set={list(top_set)[:6]}..."
        )
    if unknown:
        print(f"  WARN unknown excitations (left at default): {unknown[:6]}")

    print(f"  HFSS excitations: would drive {n_top}/{len(ex_avail)} top "
          f"ports at {pin_w} W each (total incident = {n_top * pin_w:.2f} W); "
          f"bots at 0 (matched).")
    if USE_EDIT_SOURCES:
        # Use raw-COM EditSources matching the GUI recording (with
        # IncludePortPostProcessing=False). PyAEDT's edit_sources defaults
        # IncludePortPostProcessing=True which breaks DrivenModal solves
        # in PyAEDT 0.27 + AEDT 2026.1.
        _edit_sources_raw(hfss, top_names, bot_names, pin_w,
                          include_port_post_processing=False)
    else:
        print(f"  USE_EDIT_SOURCES=False -> keeping HFSS default excitations "
              f"(all 30 ports at 1W). Network S is unchanged regardless of "
              f"which excitation pattern HFSS solved.")
    t0 = time.perf_counter()
    hfss.analyze_setup(HFSS_SETUP)
    print(f"  HFSS solved in {time.perf_counter() - t0:.1f}s")


def hfss_export_touchstone(hfss: Hfss, n_ports: int, out_path: str) -> str:
    """Export Touchstone using the PyAEDT 0.27 signature:
        export_touchstone(setup, sweep, output_file, ...).
    PyAEDT may log "exported correctly" even on partial failure; we verify
    the file actually exists and, if it doesn't, search the AEDT project
    dir for any sNp matching the expected port count and copy it over.
    """
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    try:
        ret = hfss.export_touchstone(setup=HFSS_SETUP, sweep="Sweep1",
                                     output_file=out_path)
    except Exception as e:
        raise RuntimeError(f"export_touchstone raised: {e}")

    # PyAEDT sometimes returns the actual written path as a string; prefer it.
    actual = ret if isinstance(ret, str) and os.path.isfile(ret) else out_path

    if not os.path.isfile(actual):
        # Fall back to searching the AEDT project tree for the file PyAEDT
        # actually produced. AEDT sometimes ignores output_file= and dumps
        # the Touchstone into <project>.aedtresults/<setup>.s{N}p
        from glob import glob
        candidates = (
            glob(f"{out_dir}/**/*.s{n_ports}p", recursive=True)
            + glob(f"{PROJECT_DIR}/**/*.s{n_ports}p", recursive=True)
            + glob(f"{os.path.expanduser('~')}/Ansoft/**/*.s{n_ports}p", recursive=True)
        )
        candidates = [p for p in candidates if os.path.isfile(p)]
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        if not candidates:
            raise RuntimeError(
                f"export_touchstone reported success but produced no file. "
                f"Looked at {out_path} and project subtree. "
                f"PyAEDT return value was: {ret!r}"
            )
        import shutil
        shutil.copy(candidates[0], out_path)
        print(f"  WARN: PyAEDT wrote Touchstone to {candidates[0]}, "
              f"copied -> {out_path}")
        actual = out_path

    print(f"  Touchstone -> {actual}")
    return actual


def parse_touchstone_S(path: str, n_ports: int, target_freq_hz: float
                       ) -> np.ndarray:
    """Read a Touchstone file and return the complex S(n_ports x n_ports)
    matrix at the frequency closest to target_freq_hz."""
    unit, fmt = "GHZ", "MA"
    toks: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("!"):
                continue
            if s.startswith("#"):
                for p in s.split():
                    up = p.upper()
                    if up in ("HZ", "KHZ", "MHZ", "GHZ"):
                        unit = up
                    if up in ("RI", "MA", "DB"):
                        fmt = up
                continue
            if s.startswith("+"):
                s = s[1:].strip()
            if "!" in s:
                s = s.split("!", 1)[0].strip()
            if s:
                toks.extend(s.split())

    f_scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}[unit]
    needed = 1 + 2 * n_ports * n_ports
    if len(toks) < needed:
        raise RuntimeError(f"Touchstone too short for {n_ports} ports.")

    def _pair(a: float, b: float) -> complex:
        if fmt == "RI":
            return complex(a, b)
        if fmt == "MA":
            return complex(a * math.cos(math.radians(b)), a * math.sin(math.radians(b)))
        if fmt == "DB":
            m = 10 ** (a / 20.0)
            return complex(m * math.cos(math.radians(b)), m * math.sin(math.radians(b)))
        raise ValueError(fmt)

    points = []
    idx = 0
    while idx + needed <= len(toks):
        f0 = float(toks[idx]) * f_scale; idx += 1
        vals = list(map(float, toks[idx: idx + 2 * n_ports * n_ports])); idx += 2 * n_ports * n_ports
        S = np.zeros((n_ports, n_ports), dtype=complex)
        p = 0
        for r in range(n_ports):
            for c in range(n_ports):
                S[r, c] = _pair(vals[p], vals[p + 1]); p += 2
        points.append((f0, S))
    f_best, S_best = min(points, key=lambda t: abs(t[0] - target_freq_hz))
    print(f"  parsed S({n_ports}x{n_ports}) at f={f_best/1e9:.4f} GHz "
          f"(req {target_freq_hz/1e9:.4f})")
    return S_best


def p_dissipated_from_s(S: np.ndarray, n_ports: int, port_excited: int,
                        pin_w: float) -> float:
    """Single-port excitation: P_diss = Pin * (1 - sum_i |S(i,k)|^2).
    NOTE: only valid when exactly one port is driven. For the all-tops
    excitation in this script, use `p_dissipated_global_balance` below
    instead (the column-based number underestimates the true total)."""
    k = port_excited - 1
    col = S[:, k]
    return max(pin_w * (1.0 - float(np.sum(np.abs(col)**2))), 0.0)


def p_dissipated_global_balance(S: np.ndarray, n_ports: int,
                                pin_w: float) -> float:
    """Energy balance with ALL signal-top ports driven at sqrt(pin_w) and
    all signal-bottom ports matched (a=0):  P_diss = ||a||^2 - ||S a||^2.

    Tops are odd-indexed (1, 3, 5, ...) in our port-naming scheme; in
    0-based S-matrix indices that's columns 0, 2, 4, ... = 2*k (top of
    signal k). So we set a[2k] = sqrt(pin_w) for k = 0..n_signals-1.
    """
    n_signals = n_ports // 2
    a = np.zeros(n_ports, dtype=complex)
    a[0::2] = np.sqrt(pin_w)
    b = S @ a
    p = float((np.abs(a) ** 2).sum() - (np.abs(b) ** 2).sum())
    return max(p, 0.0)


# ============================================================================
# Mechanical: build FULL design (copy HFSS solids + Region + BCs)
# ============================================================================

def _is_air_solid(name: str) -> bool:
    nml = name.lower()
    return nml.startswith("region") or "air" in nml or "radiation" in nml


def _delete_all_air_regions(mech: Mechanical) -> None:
    sol = list(getattr(mech.modeler, "solid_names", []))
    to_del = [s for s in sol if _is_air_solid(s)]
    if to_del:
        try:
            mech.modeler.delete(to_del)
            print(f"  Mech: cleaned air solids: {to_del}")
        except Exception as e:
            print(f"  WARN Mech air-clean: {e}")


def _create_mech_region(mech: Mechanical, padding_percent: float) -> str:
    _delete_object_if_exists(mech, REGION_NAME)
    xmin, ymin, zmin, xmax, ymax, zmax = _union_bbox(mech, [SUBSTRATE_NAME])
    sx = xmax - xmin; sy = ymax - ymin; sz = zmax - zmin
    px = sx * padding_percent / 100.0
    py = sy * padding_percent / 100.0
    pz = sz * padding_percent / 100.0
    mech.modeler.create_box(
        origin=[xmin - px, ymin - py, zmin - pz],
        sizes=[sx + 2 * px, sy + 2 * py, sz + 2 * pz],
        name=REGION_NAME, material=REGION_MATERIAL,
    )
    # subtract substrate + cylinders so Region is the air volume
    cutters = [s for s in mech.modeler.solid_names
               if s == SUBSTRATE_NAME
               or any(s.startswith(p) for p in SUBTRACT_FILTER_PREFIXES)]
    cutters = [c for c in cutters if c != REGION_NAME]
    if cutters:
        mech.modeler.subtract(REGION_NAME, cutters, keep_originals=True)
    print(f"  Mech: created Region (pad={padding_percent}%, cutters={len(cutters)})")
    return REGION_NAME


def _assign_region_bcs(mech: Mechanical, region_name: str, run_tag: str) -> None:
    fg = _get_face_groups_by_extents(mech, region_name)
    side = fg["x_min"] + fg["x_max"] + fg["y_min"] + fg["y_max"]
    top, bot = fg["z_max"], fg["z_min"]

    if top:
        mech.assign_uniform_convection(
            assignment=top, convection_value=H_TOP_W_PER_M2K,
            convection_unit="w_per_m2kel", temperature=AMBIENT_TEMP_STR,
            name=f"TopConv_{run_tag}")
    if side:
        mech.assign_uniform_convection(
            assignment=side, convection_value=H_SIDES_W_PER_M2K,
            convection_unit="w_per_m2kel", temperature=AMBIENT_TEMP_STR,
            name=f"SideConv_{run_tag}")
    if bot and BOTTOM_MODE.lower() == "convection":
        mech.assign_uniform_convection(
            assignment=bot, convection_value=H_BOTTOM_W_PER_M2K,
            convection_unit="w_per_m2kel", temperature=AMBIENT_TEMP_STR,
            name=f"BottomConv_{run_tag}")
    else:
        print("  Mech: bottom face is ADIABATIC.")


def _select_loss_solids(mech: Mechanical) -> Tuple[List[str], List[str]]:
    """assignment = copper + liners + substrate; surface metals = copper only."""
    sol = list(getattr(mech.modeler, "solid_names", []))
    cu  = [s for s in sol if s.startswith(COPPER_PREFIX)]
    ln  = sorted({s for p in LINER_PREFIXES for s in sol if s.startswith(p)})
    sub = [SUBSTRATE_NAME] if SUBSTRATE_NAME in sol else []
    if not cu:
        raise RuntimeError(f"No copper TSV bodies in Mechanical: prefix='{COPPER_PREFIX}'")
    if not sub:
        raise RuntimeError(f"Substrate '{SUBSTRATE_NAME}' missing in Mechanical.")
    assignment = sorted(set(cu + ln + sub))
    surface_metals = sorted(cu)
    print(f"  Mech loss-map: cu={len(cu)}, liner={len(ln)}, substrate=1 -> total={len(assignment)}")
    return assignment, surface_metals


def _ensure_mech_setup(mech: Mechanical, name: str = "SteadyState1") -> str:
    if name in mech.setups:
        return name
    mech.create_setup(name=name)
    return name


def _add_die_body(mech: Mechanical) -> None:
    """Add a small silicon die on top of the substrate (REAL_CASE mode).
    Die spans DIE_EXTENT_MM in model coords (mm). Position is chosen to
    match the user's Icepak GUI walkthrough (one corner of the substrate
    top face)."""
    _delete_object_if_exists(mech, DIE_NAME)
    e = DIE_EXTENT_MM
    sx = e["x1"] - e["x0"]
    sy = e["y1"] - e["y0"]
    sz = e["z1"] - e["z0"]
    mech.modeler.create_box(
        [e["x0"], e["y0"], e["z0"]],
        [sx, sy, sz],
        name=DIE_NAME, material=DIE_MATERIAL,
        color=(0, 200, 0),
    )
    print(f"  Mech: created die '{DIE_NAME}' at "
          f"({e['x0']*1e3:.0f},{e['y0']*1e3:.0f},{e['z0']*1e3:.0f}) -> "
          f"({e['x1']*1e3:.0f},{e['y1']*1e3:.0f},{e['z1']*1e3:.0f}) um, "
          f"material={DIE_MATERIAL}, will dissipate {DIE_POWER_W} W")


def _assign_substrate_bcs_real(mech: Mechanical, run_tag: str) -> None:
    """Apply BCs directly to substrate faces for REAL_CASE:
        - bottom face (z=0): HEATSINK (high HTC, ambient 300K)
        - 4 lateral faces:   AIR convection
        - top face (z=H):    weak convection (NOTE: top face is partially
                             covered by the die in our stackup; we apply
                             the BC over the full top face, the die area
                             will see negligible contribution since the
                             die itself has a separate top BC).
    The Region is NOT created in REAL_CASE -- avoids the air-conduction
    shield that would kill the heatsink BC."""
    fg = _get_face_groups_by_extents(mech, SUBSTRATE_NAME)
    bottom = fg["z_min"]
    top    = fg["z_max"]
    sides  = fg["x_min"] + fg["x_max"] + fg["y_min"] + fg["y_max"]

    if bottom:
        mech.assign_uniform_convection(
            assignment=bottom, convection_value=HEATSINK_HTC_W_PER_M2K,
            convection_unit="w_per_m2kel", temperature=AMBIENT_TEMP_STR,
            name=f"Heatsink_SubBot_{run_tag}")
        print(f"  Mech BC: substrate BOTTOM heatsink h={HEATSINK_HTC_W_PER_M2K:.2e} W/m2K")
    if sides:
        mech.assign_uniform_convection(
            assignment=sides, convection_value=AIR_SIDE_HTC_W_PER_M2K,
            convection_unit="w_per_m2kel", temperature=AMBIENT_TEMP_STR,
            name=f"AirSide_Sub_{run_tag}")
        print(f"  Mech BC: substrate SIDES air conv h={AIR_SIDE_HTC_W_PER_M2K} W/m2K")
    if top:
        mech.assign_uniform_convection(
            assignment=top, convection_value=AIR_TOP_HTC_W_PER_M2K,
            convection_unit="w_per_m2kel", temperature=AMBIENT_TEMP_STR,
            name=f"AirTop_Sub_{run_tag}")
        print(f"  Mech BC: substrate TOP weak conv h={AIR_TOP_HTC_W_PER_M2K} W/m2K")

    # Die: top + sides weak convection (its bottom touches substrate top => no BC)
    if DIE_ENABLED and DIE_NAME in mech.modeler.solid_names:
        fg_die = _get_face_groups_by_extents(mech, DIE_NAME)
        die_top = fg_die["z_max"]
        die_sides = (fg_die["x_min"] + fg_die["x_max"]
                     + fg_die["y_min"] + fg_die["y_max"])
        if die_top:
            mech.assign_uniform_convection(
                assignment=die_top, convection_value=AIR_TOP_HTC_W_PER_M2K,
                convection_unit="w_per_m2kel", temperature=AMBIENT_TEMP_STR,
                name=f"AirTop_Die_{run_tag}")
        if die_sides:
            mech.assign_uniform_convection(
                assignment=die_sides, convection_value=AIR_SIDE_HTC_W_PER_M2K,
                convection_unit="w_per_m2kel", temperature=AMBIENT_TEMP_STR,
                name=f"AirSide_Die_{run_tag}")
        print(f"  Mech BC: die top + sides air convection")


def build_mech_full(mech: Mechanical, hfss: Hfss, run_tag: str) -> None:
    """One-shot Mechanical FULL geometry: copy HFSS solids, (optionally
    Region + die), apply BCs. EM-loss mapping + solve happen inside the
    per-iter loop. In REAL_CASE the Region is skipped and BCs are applied
    directly to substrate + die faces (heatsink on bottom, air on sides).
    """
    if hasattr(mech, "copy_solid_bodies_from"):
        try:
            mech.copy_solid_bodies_from(hfss)
            print("  Mech: copied solid bodies from HFSS")
        except Exception as e:
            print(f"  WARN copy_solid_bodies_from: {e}")

    _delete_all_air_regions(mech)
    _delete_object_if_exists(mech, REGION_NAME)
    _delete_boundaries_by_prefix(mech,
        prefixes_lower=["topconv_", "sideconv_", "bottomconv_",
                        "heatsink_", "airside_", "airtop_", "emloss",
                        "diepower_", "dieheat_"])

    if SUBSTRATE_NAME not in mech.modeler.solid_names:
        raise RuntimeError(f"Substrate '{SUBSTRATE_NAME}' not in Mechanical solids "
                           f"after copy. Have: {mech.modeler.solid_names[:10]}...")

    if REAL_CASE:
        # Stackup mode: add die, apply BCs to substrate + die faces directly.
        if DIE_ENABLED:
            _delete_object_if_exists(mech, DIE_NAME)
            _add_die_body(mech)
        _assign_substrate_bcs_real(mech, run_tag)
    else:
        # Original Windows-style: air Region + convection on Region faces.
        _create_mech_region(mech, REGION_PADDING_PERCENT)
        _assign_region_bcs(mech, REGION_NAME, run_tag)

    _ensure_mech_setup(mech, "SteadyState1")
    print("  Mech FULL: build complete")


# ============================================================================
# Mechanical: per-iter EM-loss mapping + solve
# ============================================================================

def mech_remap_and_solve(mech: Mechanical, hfss: Hfss, freq_hz: float,
                        run_tag: str, k: int,
                        die_power_W: Optional[float] = None
                        ) -> Tuple[bool, Optional[str]]:
    """(Re-)apply EM-loss mapping from HFSS, (re-)apply die heat generation
    if REAL_CASE, and solve the Mechanical setup.

    `die_power_W` overrides the module-level DIE_POWER_W for this iter (used
    by the T-dependent-leakage loop). If None, falls back to DIE_POWER_W.
    """
    die_power_this_iter = (
        float(die_power_W) if die_power_W is not None else float(DIE_POWER_W)
    )
    _delete_boundaries_by_prefix(mech,
        prefixes_lower=["emloss", "diepower_", "dieheat_"])

    assignment, surface_metals = _select_loss_solids(mech)
    mf_used: Optional[str] = None
    for mf in _mapfreq_candidates(freq_hz):
        try:
            mech.assign_em_losses(
                design=HFSS_DESIGN, setup=HFSS_SETUP, sweep=HFSS_SWEEP,
                map_frequency=mf, assignment=assignment,
                surface_objects=surface_metals,
            )
            mf_used = mf
            print(f"  Mech iter {k}: EM-loss mapped @ {mf}")
            break
        except Exception as e:
            print(f"  WARN EM-loss map @ {mf}: {e}")

    if mf_used is None:
        return False, None

    # REAL_CASE: assign die heat generation (volumetric, total W spread over body).
    # Note: `die_power_W` is the value for THIS iter (may differ across iters
    # when the T-dependent leakage model is active).
    if REAL_CASE and DIE_ENABLED and DIE_NAME in mech.modeler.solid_names:
        try:
            mech.assign_heat_generation(
                assignment=[DIE_NAME],
                value=f"{die_power_this_iter}W",
                name=f"DieHeat_{run_tag}_iter{k}",
            )
            print(f"  Mech iter {k}: die heat generation = {die_power_this_iter:.4f} W")
        except Exception as e:
            print(f"  WARN die heat generation failed: {e}")

    t0 = time.perf_counter()
    ok = bool(mech.analyze("SteadyState1"))
    print(f"  Mech iter {k}: solve ok={ok} ({time.perf_counter() - t0:.1f}s)")
    return ok, mf_used


# ============================================================================
# Mechanical: extract per-TSV temperature + global T_mean
# ============================================================================

def extract_T_from_mech(mech: Mechanical, n_signals: int,
                       arrangement: List[List[int]]) -> Dict[str, Any]:
    """Use Mechanical's post-processing to compute mean(Temperature) per body
    and over the substrate. Returns dict with T_per_TSV (dict), T_mean_substrate,
    T_max_substrate, T_min_substrate (all in K)."""
    out: Dict[str, Any] = {"T_per_TSV_K": {}, "T_mean_substrate_K": float("nan"),
                           "T_max_substrate_K": float("nan"),
                           "T_min_substrate_K": float("nan"),
                           "extraction_errors": []}

    # Activate Mechanical design AND set active setup -- both are required
    # before post.get_solution_data works (PyAEDT 0.27 reads
    # active_setup + setup_sweeps_names from the active design).
    _activate_design(mech)
    try:
        if mech.setups:
            first_setup = list(mech.setups.keys())[0] if hasattr(mech.setups, "keys") else mech.setups[0].name
            try:
                mech.active_setup = first_setup
            except Exception:
                pass
    except Exception as e:
        out["extraction_errors"].append(f"set active_setup: {e}")

    sol = list(getattr(mech.modeler, "solid_names", []))
    tsv_bodies = [s for s in sol if s.startswith(COPPER_PREFIX)]

    # Use raw FieldsReporter (Field Calculator). For Mechanical/IcepakFEA
    # the calculator quantity name and entry method differ from HFSS --
    # we try EnterQty + EnterField + EnterScalarFunc across several common
    # quantity-name candidates and several setup-name formats.
    try:
        sols = _list_field_solutions(mech)
        print(f"  Mech extract: discovered solutions = {sols}")
        qty_names = _list_fields_reporter_quantities(mech, verbose=True)
    except Exception:
        qty_names = []

    # Probe order: most-likely first. We already learned in smoke test #7
    # that EnterQty("Temp") accepts the quantity (no exception). The
    # remaining axis is the solution-name format.
    solution_candidates = [
        "SteadyState1 : SteadyState",
        "SteadyState1 : Steady-State",
        "SteadyState1 : Solution",
        "SteadyState1 : Last Solution",
        "SteadyState1 : Solution1",
        "SteadyState1",
    ]
    quantity_candidates = ["Temp", "Temperature"]
    enter_candidates = ["EnterQty"]

    _first_probe = {"count": 0}

    def _try_extract(body: str, agg: str) -> Optional[float]:
        """Returns temperature in KELVIN (converts from the Field
        Calculator's degC output by adding 273.15)."""
        for em in enter_candidates:
            for q in quantity_candidates:
                for sol_name in solution_candidates:
                    vbs = _first_probe["count"] < 8
                    _first_probe["count"] += 1
                    v_C = _scalar_via_fields_reporter(
                        mech, body, q, agg, sol_name,
                        verbose=vbs, enter_method=em)
                    if v_C is not None and math.isfinite(v_C):
                        # AEDT Mechanical/IcepakFEA returns Temp in degC by
                        # default. Convert to Kelvin for our loop's rho_Cu
                        # update (which expects an absolute temperature).
                        v_K = float(v_C) + 273.15
                        if vbs:
                            print(f"    ** WORKING combo: {em}({q!r}) on "
                                  f"{sol_name!r} -> {v_C} degC = {v_K:.3f} K")
                        return v_K
        return None

    # Per-TSV (copper bodies)
    for b in tsv_bodies:
        t = _try_extract(b, "Mean")
        if t is not None:
            out["T_per_TSV_K"][b] = t
        else:
            out["extraction_errors"].append(f"failed Mean(T) on {b}")

    # Substrate aggregates
    for tag, agg in (("T_mean_substrate_K", "Mean"),
                     ("T_max_substrate_K",  "Max"),
                     ("T_min_substrate_K",  "Min")):
        v = _try_extract(SUBSTRATE_NAME, agg)
        if v is not None:
            out[tag] = v
        else:
            out["extraction_errors"].append(f"failed {agg}(T) on {SUBSTRATE_NAME}")

    print(f"  Mech extract: per-TSV={len(out['T_per_TSV_K'])}/{len(tsv_bodies)}, "
          f"T_mean_sub={out['T_mean_substrate_K']:.3f}K, errors={len(out['extraction_errors'])}")
    return out


# ============================================================================
# HFSS: update copper conductivity from a scalar temperature
# ============================================================================

def update_cu_sigma(hfss: Hfss, T_drive_K: float) -> float:
    """sigma_new = sigma_0 / (1 + alpha * (T_drive - T_ref)).  Returns new sigma.

    Mirrors set_copper_conductivity_at_T from the Windows
    HFSS_Interface_not_sweep.py. Uses the simple PyAEDT path first
    (mat.conductivity.value = X) and falls back to raw COM EditMaterial
    on the built-in `copper` if PyAEDT's cached handle is stale (the
    Mech-took-over-the-active-design failure mode).
    """
    boost = 1.0 + ALPHA_CU * (T_drive_K - T_REF_CU)
    boost = max(boost, 1e-3)
    sigma_new = SIGMA_CU_REF / boost
    # First try the simple PyAEDT path
    try:
        _activate_design(hfss)
        cu = hfss.materials["copper"]
        cu.conductivity.value = sigma_new
        print(f"  Cu update (PyAEDT path): T_drive={T_drive_K:.2f} K  ->  "
              f"sigma={sigma_new:.4e} S/m (boost={boost:.4f})")
        return sigma_new
    except Exception as e:
        print(f"  WARN PyAEDT Cu update failed ({type(e).__name__}: {e}); "
              f"falling back to raw COM EditMaterial.")
    # Raw COM fallback
    if not _edit_material_raw(hfss, "copper", sigma_new):
        raise RuntimeError("Both PyAEDT and raw-COM Cu updates failed.")
    print(f"  Cu update (raw COM): T_drive={T_drive_K:.2f} K  ->  "
          f"sigma={sigma_new:.4e} S/m (boost={boost:.4f})")
    return sigma_new


# ============================================================================
# Main: bidirectional loop runner
# ============================================================================

def run_bidirectional_loop() -> Dict[str, Any]:
    if not _HAVE_AEDT:
        raise RuntimeError(
            "PyAEDT (ansys-aedt-core) is not installed. Install the 'signoff' "
            "extra and run on a machine with Ansys AEDT 2026.1: "
            "pip install 'tessera-tsv[signoff]'."
        )
    run_tag = ("smoke_" if SMOKE_TEST else "") + uuid.uuid4().hex[:6]
    project_path = os.path.join(PROJECT_DIR, f"hfss_mech_bi_{run_tag}.aedt")
    out_summary = {
        "run_tag": run_tag,
        "project_path": project_path,
        "aedt_version": AEDT_VERSION,
        "geometry": dict(GEOM),
        "arrangement": ARRANGEMENT,
        "loop": {"max_iter": MAX_ITER, "tol_T_K": TOL_T_K,
                 "alpha_Cu": ALPHA_CU, "T_ref_K": T_REF_CU,
                 "sigma_Cu_ref": SIGMA_CU_REF, "pin_w": PIN_W,
                 "port_excited": PORT_EXCITED},
        "iters": [],
        "aborted_reason": None,
    }

    # Temp isolation
    temp_dir = tempfile.mkdtemp(prefix=f"hfss_mech_bi_{run_tag}_",
                                dir=os.environ.get("TMPDIR", OUTPUT_DIR))
    os.environ["ANSYS_TEMP_DIRECTORY"] = temp_dir
    os.environ["PYAEDT_PROJECT_DIRECTORY"] = OUTPUT_DIR

    _stamp(f"start AEDT {AEDT_VERSION} (non_graphical={NON_GRAPHICAL})")
    desktop = Desktop(version=AEDT_VERSION, non_graphical=NON_GRAPHICAL,
                      new_desktop=True, close_on_exit=CLOSE_ON_EXIT)

    try:
        # --- 1. HFSS build ---
        _stamp("opening HFSS design")
        hfss = Hfss(project=project_path, design=HFSS_DESIGN,
                    solution_type="DrivenModal")
        _stamp("building HFSS TSV array")
        n_signals, n_ports, top_names, bot_names = build_hfss_design(
            hfss, GEOM, ARRANGEMENT, PIN_W)
        _safe_save(hfss, "after HFSS build")

        # --- 2. Mechanical FULL: copy solids + Region + BCs (once) ---
        _stamp("opening Mechanical design")
        mech = Mechanical(project=project_path, design=MECH_FULL_DESIGN)
        build_mech_full(mech, hfss, run_tag)
        _safe_save(mech, "after Mech FULL build")

        # --- 3. Outer iteration loop ---
        T_drive = T_REF_CU         # iter 0 starts at ambient -> no Cu change
        T_mean_prev = None
        # Per-iter die power (mutated when DIE_ALPHA_W_PER_K > 0)
        P_die_iter_W = float(DIE_POWER_W)
        out_summary["leakage_model"] = {
            "die_power0_W": float(DIE_POWER_W),
            "die_alpha_W_per_K": float(DIE_ALPHA_W_PER_K),
            "die_t_ref_K":      float(DIE_T_REF_K),
            "die_p_max_W":      float(DIE_P_MAX_W),
        }
        print(
            f"  T-leakage model: P_die(T) = {DIE_POWER_W} W "
            f"+ {DIE_ALPHA_W_PER_K} W/K * (T_die - {DIE_T_REF_K} K)  "
            f"[cap {DIE_P_MAX_W} W]"
        )

        for k in range(MAX_ITER):
            _stamp(f"=== ITER {k} ===")
            iter_rec: Dict[str, Any] = {
                "iter": k, "T_drive_K": T_drive,
                "P_die_W": float(P_die_iter_W),
            }

            # 3a. update Cu sigma (iter 0: no-op since T_drive = T_ref)
            try:
                iter_rec["sigma_Cu"] = update_cu_sigma(hfss, T_drive)
            except Exception as e:
                iter_rec["error"] = f"Cu sigma update: {e}"
                out_summary["iters"].append(iter_rec)
                out_summary["aborted_reason"] = iter_rec["error"]
                break

            # 3b. solve HFSS with port_excited driven at PIN_W
            try:
                hfss_set_excitations_and_solve(hfss, n_ports, PIN_W,
                                                PORT_EXCITED,
                                                top_names, bot_names)
            except Exception as e:
                iter_rec["error"] = f"HFSS solve: {e}\n{traceback.format_exc()}"
                out_summary["iters"].append(iter_rec)
                out_summary["aborted_reason"] = "HFSS solve"
                break

            # 3c. export Touchstone snapshot + P_diss from S
            try:
                ts_path = os.path.join(OUTPUT_DIR,
                                       f"{TS_BASENAME}{k}.s{n_ports}p")
                hfss_export_touchstone(hfss, n_ports, ts_path)
                iter_rec["touchstone"] = ts_path
                S = parse_touchstone_S(ts_path, n_ports, GEOM["freq_hz"])
                p_diss_col    = p_dissipated_from_s(S, n_ports, PORT_EXCITED, PIN_W)
                p_diss_global = p_dissipated_global_balance(S, n_ports, PIN_W)
                iter_rec["P_diss_W"]            = p_diss_global   # primary: global balance
                iter_rec["P_diss_col_W"]        = p_diss_col      # diagnostic
                iter_rec["P_diss_global_W"]     = p_diss_global   # explicit copy for JSON readers
                iter_rec["max_abs_S"]           = float(np.abs(S).max())
                n_signals_loc = n_ports // 2
                print(f"  P_diss = {p_diss_global:.6g} W (global, all-tops drive; "
                      f"incident = {n_signals_loc * PIN_W:.2f} W), "
                      f"per-col = {p_diss_col:.4e} W, max|S| = {iter_rec['max_abs_S']:.4f}")
            except Exception as e:
                iter_rec["error"] = f"Touchstone / P_diss: {e}"
                out_summary["iters"].append(iter_rec)
                out_summary["aborted_reason"] = "Touchstone parse"
                break

            # 3d. Mechanical: re-map EM loss + solve (with this iter's die power)
            try:
                mech_ok, mf_used = mech_remap_and_solve(
                    mech, hfss, GEOM["freq_hz"], run_tag, k,
                    die_power_W=P_die_iter_W)
                iter_rec["mech_ok"] = mech_ok
                iter_rec["map_frequency"] = mf_used
            except Exception as e:
                iter_rec["error"] = f"Mech remap/solve: {e}\n{traceback.format_exc()}"
                out_summary["iters"].append(iter_rec)
                out_summary["aborted_reason"] = "Mech remap/solve"
                break

            if not mech_ok:
                iter_rec["error"] = "Mechanical analyse returned False"
                out_summary["iters"].append(iter_rec)
                out_summary["aborted_reason"] = "Mech analyse=False"
                break

            # 3e. Extract T from Mechanical
            try:
                t_info = extract_T_from_mech(mech, n_signals, ARRANGEMENT)
                iter_rec.update({
                    "T_mean_substrate_K": t_info["T_mean_substrate_K"],
                    "T_max_substrate_K":  t_info["T_max_substrate_K"],
                    "T_min_substrate_K":  t_info["T_min_substrate_K"],
                    "T_per_TSV_K":        t_info["T_per_TSV_K"],
                    "extraction_errors":  t_info["extraction_errors"],
                })
            except Exception as e:
                iter_rec["error"] = f"T extract: {e}\n{traceback.format_exc()}"
                out_summary["iters"].append(iter_rec)
                out_summary["aborted_reason"] = "T extract"
                break

            # 3f. T-dependent die-leakage update for the NEXT iter ----------
            # Proxy T_die by the substrate max (the die sits at the substrate
            # top corner and dominates the hotspot). Mirrors the Python side.
            T_die_proxy_K = float(iter_rec.get("T_max_substrate_K", float("nan")))
            if (DIE_ALPHA_W_PER_K > 0.0 and not math.isnan(T_die_proxy_K)):
                P_die_next_W = (
                    float(DIE_POWER_W)
                    + float(DIE_ALPHA_W_PER_K)
                      * (T_die_proxy_K - float(DIE_T_REF_K))
                )
                P_die_next_W = max(0.0, min(P_die_next_W, float(DIE_P_MAX_W)))
            else:
                P_die_next_W = float(DIE_POWER_W)
            iter_rec["T_die_proxy_K"] = T_die_proxy_K
            iter_rec["P_die_next_W"]  = P_die_next_W

            # 3g. Convergence check (use substrate T_mean if available)
            T_mean = iter_rec["T_mean_substrate_K"]
            if T_mean_prev is not None and not math.isnan(T_mean):
                dT = abs(T_mean - T_mean_prev)
                iter_rec["dT_mean_K"] = dT
                print(f"  dT_mean = {dT:.3f} K  "
                      f"T_die_proxy = {T_die_proxy_K:.2f} K  "
                      f"P_die_next = {P_die_next_W:.4f} W  (tol {TOL_T_K})")
                if dT < TOL_T_K:
                    out_summary["iters"].append(iter_rec)
                    print(f"  *** CONVERGED at iter {k} ***")
                    out_summary["converged_at_iter"] = k
                    break
            else:
                print(f"  T_die_proxy = {T_die_proxy_K:.2f} K  "
                      f"P_die_next = {P_die_next_W:.4f} W")

            # Apply new die power for the NEXT iter (no-op if alpha=0)
            P_die_iter_W = P_die_next_W

            # 3h. update T_drive for next iter
            if not math.isnan(T_mean):
                T_drive = T_mean
                T_mean_prev = T_mean
            else:
                # Fall back to mean of per-TSV temperatures
                vals = list(iter_rec["T_per_TSV_K"].values())
                if vals:
                    T_drive = float(np.mean(vals))
                    T_mean_prev = T_drive
                else:
                    print("  WARN no T extracted; keeping T_drive unchanged")

            out_summary["iters"].append(iter_rec)

            if SAVE_PER_ITER:
                with open(os.path.join(OUTPUT_DIR, f"iter{k}_summary.json"),
                          "w") as f:
                    json.dump(iter_rec, f, indent=2)
            _safe_save(mech, f"after iter {k}")

        # End-of-loop save
        _safe_save(hfss, "final HFSS")
        _safe_save(mech, "final Mech")

    except Exception as e:
        print(f"FATAL: {e}")
        traceback.print_exc()
        out_summary["aborted_reason"] = f"FATAL: {e}"

    finally:
        # Release the EXISTING desktop (do not call Desktop() with no args —
        # that would spawn a SECOND AEDT session, which was the bug last run).
        try:
            desktop.release_desktop(close_projects=False, close_on_exit=True)
            print("  released desktop session")
        except Exception as e:
            print(f"  WARN release_desktop: {e}")

    # Final summary JSON
    summary_path = os.path.join(OUTPUT_DIR, f"loop_summary_{run_tag}.json")
    with open(summary_path, "w") as f:
        json.dump(out_summary, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"\nfull summary -> {summary_path}")
    # Stable path so downstream tooling (tessera.signoff.run_signoff) can
    # locate the result without knowing the random run_tag.
    stable_path = os.path.join(OUTPUT_DIR, "signoff_result.json")
    with open(stable_path, "w") as f:
        json.dump(out_summary, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"stable result -> {stable_path}")
    return out_summary


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HFSS <-> Mechanical bidirectional electrothermal coupling (Linux)")
    print("=" * 70)
    res = run_bidirectional_loop()
    print()
    print(f"Iterations run: {len(res['iters'])}")
    if res.get("aborted_reason"):
        print(f"Aborted: {res['aborted_reason']}")
    elif "converged_at_iter" in res:
        print(f"Converged at iter {res['converged_at_iter']}")
    print(f"Output dir: {OUTPUT_DIR}")
