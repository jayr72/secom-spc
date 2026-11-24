"""Hand-calculated checks for the SPC and capability formulas."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secom.capability import cp_cpk, percentile_capability, verdict  # noqa: E402
from secom.spc import D2, D4, imr_limits, moving_range, p_chart, run_rules  # noqa: E402

# Worked example: x = 10, 12, 11, 13, 12, 14, 13, 12
# mean = 97/8 = 12.125
# moving ranges = 2,1,2,1,2,1,1 -> MR-bar = 10/7 = 1.428571
# sigma_within = 1.428571 / 1.128 = 1.266464
# UCL = 12.125 + 3*1.266464 = 15.924392 ; LCL = 8.325608 ; MR UCL = 3.267*1.428571 = 4.667143
X = np.array([10, 12, 11, 13, 12, 14, 13, 12], dtype=float)


def test_moving_range():
    assert moving_range(X).tolist() == [2, 1, 2, 1, 2, 1, 1]


def test_imr_limits_hand_calc():
    lim = imr_limits(X)
    assert lim.center == pytest.approx(12.125)
    assert lim.mr_bar == pytest.approx(10 / 7)
    assert lim.sigma == pytest.approx((10 / 7) / 1.128)
    assert lim.ucl == pytest.approx(15.924392, abs=1e-5)
    assert lim.lcl == pytest.approx(8.325608, abs=1e-5)
    assert lim.mr_ucl == pytest.approx(4.667143, abs=1e-5)


def test_imr_ignores_nan():
    assert imr_limits(np.r_[X, np.nan]).center == pytest.approx(12.125)


def test_constants():
    assert D2 == 1.128 and D4 == 3.267
    assert 3 / D2 == pytest.approx(2.66, abs=0.001)  # E2


def test_cp_cpk_hand_calc():
    # specs 8..16, mean 12.125, sigma_w 1.266464
    # Cp  = 8 / (6*1.266464) = 8 / 7.598784 = 1.052800
    # Cpk = min(3.875, 4.125) / (3*1.266464) = 3.875 / 3.799392 = 1.019900
    s = X.std(ddof=1)
    r = cp_cpk(X, 8, 16)
    assert r["Cp"] == pytest.approx(1.052800, abs=1e-5)
    assert r["Cpk"] == pytest.approx(1.019900, abs=1e-5)
    assert r["Pp"] == pytest.approx(8 / (6 * s))
    assert r["Ppk"] == pytest.approx(3.875 / (3 * s))


def test_cpk_le_cp_and_off_center_penalty():
    centered = cp_cpk(X, 12.125 - 4, 12.125 + 4)
    shifted = cp_cpk(X, 9, 17)
    assert centered["Cpk"] == pytest.approx(centered["Cp"])
    assert shifted["Cpk"] < shifted["Cp"]


def test_percentile_capability_on_normal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200_000)
    r = percentile_capability(x, -3, 3)
    assert r["Cp_pct"] == pytest.approx(1.0, abs=0.03)
    assert r["Cpk_pct"] == pytest.approx(1.0, abs=0.03)


def test_verdict_bands():
    assert verdict(1.7).startswith("excellent")
    assert verdict(1.4).startswith("acceptable")
    assert verdict(1.1).startswith("marginal")
    assert verdict(0.5).startswith("not capable")


def test_rule1_beyond_3_sigma():
    x = np.zeros(20)
    x[7] = 3.5
    f = run_rules(x, 0, 1)
    assert f["R1"].tolist() == [i == 7 for i in range(20)]


def test_rule2_nine_same_side():
    x = np.r_[np.full(8, 0.5), -0.5, np.full(9, 0.5)]
    f = run_rules(x, 0, 1)
    assert not f["R2"][:9].any()
    assert f["R2"][9:].all()


def test_rule3_six_trending():
    x = np.r_[0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0]
    f = run_rules(x, 0, 1)
    assert f["R3"][1:8].all() and not f["R3"][8:].any()
    five = np.r_[0, 0.1, 0.2, 0.3, 0.4, 0, 0]  # only 5 rising points: no signal
    assert not run_rules(five, 0, 1)["R3"].any()


def test_rule4_fourteen_alternating():
    x = np.array([0.5 if i % 2 else -0.5 for i in range(14)])
    assert run_rules(x, 0, 1)["R4"].all()
    assert not run_rules(x[:13], 0, 1)["R4"].any()


def test_p_chart_limits():
    fails = pd.Series([2, 5, 1])
    n = pd.Series([50, 50, 100])
    df = p_chart(fails, n)
    p_bar = 8 / 200
    assert df["center"].iloc[0] == pytest.approx(p_bar)
    assert df["ucl"].iloc[0] == pytest.approx(p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / 50))
    assert (df["lcl"] >= 0).all()
    assert df["ucl"].iloc[2] < df["ucl"].iloc[0]  # bigger subgroup -> tighter limits
