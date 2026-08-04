import math
import numpy as np
from scipy import integrate
from functools import lru_cache

# ==========================================
# CONSTANTS
# ==========================================
# Material thermal constants (W/mK)
K_METAL = 400.0    # Copper
K_INS = 1.5        # SiO2
K_SUBS = 148.0     # Silicon

# Geometry
PI = math.pi
FOUR_MINUS_PI = 4.0 - PI

@lru_cache(maxsize=1024)
def _compute_unit_cell_properties(r_via: float, t_ox: float) -> tuple[float, float, float]:
    """
    Computes thermal properties of a single TSV unit cell.
    Cached to avoid re-integrating for identical geometries.
    
    Returns:
        (K_vert, K_horiz, tsv_cell_area)
    """
    # 1. Geometry derived values
    r_outer = r_via + t_ox
    r_outer_sq = r_outer * r_outer
    r_via_sq = r_via * r_via
    
    # 2. Vertical Conductivity (Parallel Model / Rule of Mixtures)
    # Area definitions relative to r_outer_sq
    # Cu_area = PI * r_via_sq
    # ins_area = PI * (r_outer_sq - r_via_sq)
    # subs_area = (4.0 - PI) * r_outer_sq
    # unit_cell_area = 4.0 * r_outer_sq
    
    # Simplified algebra for K_vert to reduce operations
    area_cu = PI * r_via_sq
    area_ins = PI * (r_outer_sq - r_via_sq)
    area_subs = FOUR_MINUS_PI * r_outer_sq
    total_area = 4.0 * r_outer_sq
    
    K_vert = (K_METAL * area_cu + K_INS * area_ins + K_SUBS * area_subs) / total_area

    # 3. Horizontal Conductivity (Series-Parallel Integration)
    # Optimization: Use fixed Gaussian quadrature instead of adaptive quad.
    # The geometry is smooth (circles), so n=20 points is highly accurate and deterministic.
    
    # Pre-calculate inverse conductivities for speed in loop
    inv_k_metal = 1.0 / K_METAL
    inv_k_ins = 1.0 / K_INS
    inv_k_subs = 1.0 / K_SUBS

    def integrand(x):
        # Vectorized function for scipy integration
        x_sq = x * x
        # w_metal = sqrt(r_via^2 - x^2)
        # Using maximum to prevent sqrt domain error close to r_via
        w_metal = np.sqrt(np.maximum(r_via_sq - x_sq, 0.0))
        
        root_outer = np.sqrt(np.maximum(r_outer_sq - x_sq, 0.0))
        w_ins = root_outer - w_metal
        w_subs = r_outer - root_outer
        
        # Total thermal resistance of this slice
        r_total = (w_metal * inv_k_metal) + (w_ins * inv_k_ins) + (w_subs * inv_k_subs)
        return 1.0 / r_total

    # Fixed_quad is significantly faster than quad for smooth functions                                                                                                               
    K_horiz, _ = integrate.fixed_quad(integrand, 0.0, r_outer, n=200)
    
    # TSV unit cell side length and area
    l_tsv = 2.0 * r_outer
    tsv_cell_area = l_tsv * l_tsv

    return K_vert, K_horiz, tsv_cell_area

def thermal_conductivity(r_via: float, pitch: float, t_ox: float,
                         arrangement: np.ndarray) -> tuple[float, float, float]:
    """
    Compute equivalent thermal conductivities (Kx, Ky, Kz) for a TSV array.
    
    Args:
        r_via: Radius of the copper via (m)
        pitch: Center-to-center distance between TSVs (m)
        t_ox: Thickness of the oxide liner (m)
        arrangement: 2D numpy array where != 0 indicates a TSV.

    Returns:
        (K_x, K_y, K_z) in W/mK
    """
    # 1. Get Unit Cell Properties (Cached)
    # We cast to float to ensure cache hits work (avoid np.float64 vs float mismatches)
    K_vert, K_horiz, tsv_cell_area = _compute_unit_cell_properties(float(r_via), float(t_ox))
    l_tsv = math.sqrt(tsv_cell_area) # Side length of unit cell

    # 2. Analyze Array Occupancy
    # Fast vectorized check for occupied rows/cols
    # any(axis=1) checks if a row has any TSV. sum() counts them.
    # This replaces len(np.unique(nonzero)) which is slower and creates intermediate arrays.
    
    # If arrangement allows negative values (Ground), checking != 0 is correct.
    has_tsv_mask = arrangement != 0
    M = np.count_nonzero(has_tsv_mask.any(axis=1)) # Occupied rows
    N = np.count_nonzero(has_tsv_mask.any(axis=0)) # Occupied cols

    # If no TSVs, return pure substrate properties (Edge case safety)
    if M == 0 or N == 0:
        return K_SUBS, K_SUBS, K_SUBS

    # 3. Substrate Dimensions
    grid_h, grid_w = arrangement.shape
    l_subs = grid_h * pitch
    wid_subs = grid_w * pitch
    total_area = l_subs * wid_subs

    # 4. Array-Level Aggregation
    
    # Dimensions of the "Active Block" of TSVs
    M_l = M * l_tsv
    N_l = N * l_tsv

    # --- K_x Calculation ---
    # Heat flows in X direction.
    # Logic: The chip is split into two parallel regions:
    # 1. Pure silicon above/below the TSV block (Parallel path)
    # 2. The TSV block row, which contains the TSVs + Silicon to the left/right (Series path)
    
    # Term A: Conductance of the Silicon segments in the series path
    # Term B: Conductance of the TSV segments in the series path
    
    # nlw is a scaling factor common to both A and B in original code.
    # Original: nlw = N * l_subs * l_tsv
    # A = K_SUBS * nlw / ((l_subs - M_l) * wid_subs)
    # B = K_horiz * nlw / (M_l * wid_subs)
    # We can simplify (A*B)/(A+B) by cancelling common terms to reduce floating point ops
    
    # Let's keep strict algebraic equivalence to original for safety, just simplified variables.
    common_num = N * l_subs * l_tsv
    denom_A = (l_subs - M_l) * wid_subs
    denom_B = M_l * wid_subs
    
    # Handle edge case where TSVs fill the whole dimension (denom -> 0)
    if denom_A <= 1e-12: # TSVs fill height completely
        K_x_series = K_horiz * (common_num / denom_B) # Approximation
    else:
        A_x = K_SUBS * common_num / denom_A
        B_x = K_horiz * common_num / denom_B
        K_x_series = (A_x * B_x) / (A_x + B_x)

    # Parallel combination with the top/bottom silicon
    K_x_parallel = K_SUBS * (wid_subs - N_l) / wid_subs
    K_x = K_x_parallel + K_x_series

    # --- K_y Calculation ---
    # Heat flows in Y direction. Symmetric logic to X.
    common_num_y = M * l_tsv * wid_subs
    denom_A_y = (wid_subs - N_l) * l_subs
    denom_B_y = N * l_subs * l_tsv
    
    if denom_A_y <= 1e-12: # TSVs fill width completely
        K_y_series = K_horiz * (common_num_y / denom_B_y)
    else:
        A_y = K_SUBS * common_num_y / denom_A_y
        B_y = K_horiz * common_num_y / denom_B_y
        K_y_series = (A_y * B_y) / (A_y + B_y)

    K_y_parallel = K_SUBS * (l_subs - M_l) / l_subs
    K_y = K_y_parallel + K_y_series

    # --- K_z Calculation ---
    # Simple area-weighted average (Parallel model)
    # M * N is the number of "Active Unit Cells" in the bounding box
    # Note: This logic assumes a dense rectangular block of MxN. 
    # If the arrangement is sparse (e.g. diagonal), M*N might be > count_nonzero.
    # The original code used M*N (bounding box area), so we strictly preserve that.
    tsv_total_area = M * N * tsv_cell_area
    K_z = (K_vert * tsv_total_area + K_SUBS * (total_area - tsv_total_area)) / total_area

    return K_x, K_y, K_z