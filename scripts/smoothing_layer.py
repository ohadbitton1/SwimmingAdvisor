import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ==================== CONSTANTS & CONFIG ====================
SHOULDER_WIDTH_RATIO = 0.5
HIP_WIDTH_RATIO = 0.45
KNEE_WIDTH_RATIO = 0.35
ANKLE_WIDTH_RATIO = 0.25

# ==================== HELPER FUNCTIONS ====================
def any_nan(*pts):
    for p in pts:
        if p is None or np.any(np.isnan(np.asarray(p))): return True
    return False

def dist(a, b):
    if any_nan(a, b): return np.nan
    return float(np.sqrt(np.sum((np.asarray(a) - np.asarray(b))**2)))

def safe_midpoint(a, b):
    if any_nan(a, b): return np.array([np.nan, np.nan], dtype=np.float32)
    return (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)) / 2.0

# ==================== V10 GUARDS & LOGIC ====================
def global_antiflip(curr, prev):
    if prev is None: return curr
    LEFT_SET, RIGHT_SET = [1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]
    
    def get_cost(kpts, reference):
        total, used = 0.0, 0
        for i in range(1, 13):
            if not any_nan(kpts[i], reference[i]):
                total += dist(kpts[i], reference[i]); used += 1
        return total if used > 0 else np.nan

    c_keep = get_cost(curr, prev)
    swapped = curr.copy()
    for l, r in zip(LEFT_SET, RIGHT_SET):
        swapped[l], swapped[r] = curr[r].copy(), curr[l].copy()
    c_swap = get_cost(swapped, prev)

    return swapped if (not np.isnan(c_keep) and (np.isnan(c_swap) or c_swap < c_keep)) else curr

def torso_alignment_guard(curr):
    k = curr.copy()
    LS, RS, LH, RH = 1, 2, 7, 8
    if any_nan(k[LS], k[RS], k[LH], k[RH]): return k
    if (dist(k[LS], k[RH]) + dist(k[RS], k[LH])) < (dist(k[LS], k[LH]) + dist(k[RS], k[RH])):
        for l, r in [(1,2), (3,4), (5,6)]: k[l], k[r] = k[r].copy(), k[l].copy()
    return k

def leg_chain_guard(curr):
    k = curr.copy()
    LH, RH, LK, RK, LA, RA = 7, 8, 9, 10, 11, 12
    if any_nan(k[LH], k[RH]): return k
    if not any_nan(k[LK], k[RK]):
        if (dist(k[LK], k[RH]) + dist(k[RK], k[LH])) < (dist(k[LK], k[LH]) + dist(k[RK], k[RH])):
            k[LK], k[RK] = k[RK].copy(), k[LK].copy()
    if not any_nan(k[LA], k[RA], k[LK], k[RK]):
        if (dist(k[LA], k[RK]) + dist(k[RA], k[LK])) < (dist(k[LA], k[LK]) + dist(k[RA], k[RK])):
            k[LA], k[RA] = k[RA].copy(), k[LA].copy()
    return k

def enforce_min_width_by_torso(kpts, l_idx, r_idx, min_w):
    if any_nan(kpts[l_idx], kpts[r_idx]): return kpts
    d = dist(kpts[l_idx], kpts[r_idx])
    if d >= min_w or d < 1: return kpts
    center = (kpts[l_idx] + kpts[r_idx]) / 2.0
    dir_vec = (kpts[r_idx] - kpts[l_idx]) / d
    kpts[r_idx], kpts[l_idx] = center + dir_vec * (min_w/2), center - dir_vec * (min_w/2)
    return kpts

# ==================== TEMPORAL SMOOTHING ====================
def apply_temporal_smoothing(proc_data):
    """
    Applies Savitzky-Golay filter to the entire sequence.
    Using the updated pandas .ffill().bfill() logic.
    """
    for i in range(13):
        win = 13 if i in [1,2,3,4,5,6] else 9
        x = pd.Series(proc_data[:, i, 0]).interpolate(limit_direction="both").ffill().bfill().to_numpy()
        y = pd.Series(proc_data[:, i, 1]).interpolate(limit_direction="both").ffill().bfill().to_numpy()
        proc_data[:, i, 0] = savgol_filter(x, win, 2)
        proc_data[:, i, 1] = savgol_filter(y, win, 2)
    return proc_data