"""Hand-written process capability.

Cp  = (USL - LSL) / (6 * sigma_within)        sigma_within = MR-bar / d2
Cpk = min(USL - mean, mean - LSL) / (3 * sigma_within)
Pp  = (USL - LSL) / (6 * s)                   s = overall sample std (ddof=1)
Ppk = min(USL - mean, mean - LSL) / (3 * s)

Percentile (Clements / ISO 22514-2 style) for non-normal data:
Cp_pct  = (USL - LSL) / (P99.865 - P0.135)
Cpk_pct = min((USL - median) / (P99.865 - median), (median - LSL) / (median - P0.135))
"""
import numpy as np
from scipy import stats

from .spc import D2, moving_range


def cp_cpk(x, lsl: float, usl: float) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    mean = x.mean()
    sigma_w = moving_range(x).mean() / D2
    s = x.std(ddof=1)
    return {
        "n": len(x),
        "mean": mean,
        "sigma_within": sigma_w,
        "sigma_overall": s,
        "Cp": (usl - lsl) / (6 * sigma_w),
        "Cpk": min(usl - mean, mean - lsl) / (3 * sigma_w),
        "Pp": (usl - lsl) / (6 * s),
        "Ppk": min(usl - mean, mean - lsl) / (3 * s),
    }


def percentile_capability(x, lsl: float, usl: float) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    lo, med, hi = np.percentile(x, [0.135, 50, 99.865])
    return {
        "Cp_pct": (usl - lsl) / (hi - lo),
        "Cpk_pct": min((usl - med) / (hi - med), (med - lsl) / (med - lo)),
    }


def is_normal(x, level: float = 5.0) -> tuple[bool, float]:
    """Anderson-Darling test. Returns (passes_at_level, statistic)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    res = stats.anderson(x, dist="norm")
    crit = dict(zip(res.significance_level, res.critical_values))[level]
    return bool(res.statistic < crit), float(res.statistic)


def verdict(cpk: float) -> str:
    if cpk >= 1.67:
        return "excellent (>=1.67)"
    if cpk >= 1.33:
        return "acceptable (1.33-1.67)"
    if cpk >= 1.00:
        return "marginal (1.00-1.33)"
    return "not capable (<1.00)"
