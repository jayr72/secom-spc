"""Hand-written SPC: I-MR limits, p-chart limits, and run rules.

Constants for moving range of span 2 (AIAG SPC 2nd ed.):
    d2 = 1.128, D3 = 0, D4 = 3.267, so 3 / d2 = 2.66 (often called E2).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

D2 = 1.128
D4 = 3.267


@dataclass
class IMRLimits:
    center: float
    mr_bar: float
    sigma: float  # within-process sigma = MR-bar / d2
    ucl: float
    lcl: float
    mr_ucl: float


def moving_range(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.abs(np.diff(x))


def imr_limits(baseline) -> IMRLimits:
    """Individuals and moving-range limits from a Phase I baseline (NaNs removed)."""
    x = np.asarray(baseline, dtype=float)
    x = x[~np.isnan(x)]
    center = x.mean()
    mr_bar = moving_range(x).mean()
    sigma = mr_bar / D2
    return IMRLimits(center, mr_bar, sigma, center + 3 * sigma, center - 3 * sigma, D4 * mr_bar)


def run_rules(x, center: float, sigma: float) -> pd.DataFrame:
    """Flag points that are part of a run-rule violation.

    Numbering follows the spc-control-charts skill / AIAG list:
      R1: 1 point beyond 3 sigma
      R2: 9 points in a row on the same side of the center line
      R3: 6 points in a row steadily increasing or decreasing
      R4: 14 points in a row alternating up and down
    Returns a boolean DataFrame with columns R1..R4 and 'any'.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    flags = {k: np.zeros(n, dtype=bool) for k in ("R1", "R2", "R3", "R4")}

    flags["R1"] = np.abs(x - center) > 3 * sigma

    side = np.sign(x - center)
    _mark_runs(flags["R2"], side, 9, lambda w: abs(w.sum()) == len(w) and w[0] != 0)

    step = np.sign(np.diff(x))  # step[i] compares x[i+1] with x[i]
    # 6 points steadily trending = 5 consecutive steps in the same direction
    for i in range(len(step) - 4):
        w = step[i : i + 5]
        if w[0] != 0 and np.all(w == w[0]):
            flags["R3"][i : i + 6] = True

    # 14 points alternating = 13 steps, each opposite in sign to the previous
    for i in range(len(step) - 12):
        w = step[i : i + 13]
        if np.all(w != 0) and np.all(w[1:] == -w[:-1]):
            flags["R4"][i : i + 14] = True

    out = pd.DataFrame(flags)
    out["any"] = out.any(axis=1)
    return out


def _mark_runs(target: np.ndarray, vals: np.ndarray, length: int, pred) -> None:
    for i in range(len(vals) - length + 1):
        if pred(vals[i : i + length]):
            target[i : i + length] = True


def p_chart(fails: pd.Series, n: pd.Series) -> pd.DataFrame:
    """p-chart with variable subgroup size. fails and n are indexed by subgroup."""
    p_bar = fails.sum() / n.sum()
    se = np.sqrt(p_bar * (1 - p_bar) / n)
    df = pd.DataFrame({"n": n, "fails": fails, "p": fails / n})
    df["center"] = p_bar
    df["ucl"] = p_bar + 3 * se
    df["lcl"] = (p_bar - 3 * se).clip(lower=0)
    df["out_high"] = df["p"] > df["ucl"]
    df["out_low"] = df["p"] < df["lcl"]
    return df
