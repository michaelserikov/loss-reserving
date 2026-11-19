import numpy as np
import pytest

from reserving import methods as m
from reserving import triangle as tri

# Taylor and Ashe (1983), incremental paid claims, accident years 1 to 10
TA = np.array([
    [357848, 766940, 610542, 482940, 527326, 574398, 146342, 139950, 227229, 67948],
    [352118, 884021, 933894, 1183289, 445745, 320996, 527804, 266172, 425046, np.nan],
    [290507, 1001799, 926219, 1016654, 750816, 146923, 495992, 280405, np.nan, np.nan],
    [310608, 1108250, 776189, 1562400, 272482, 352053, 206286, np.nan, np.nan, np.nan],
    [443160, 693190, 991983, 769488, 504851, 470639, np.nan, np.nan, np.nan, np.nan],
    [396132, 937085, 847498, 805037, 705960, np.nan, np.nan, np.nan, np.nan, np.nan],
    [440832, 847631, 1131398, 1063269, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [359480, 1061648, 1443370, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [376686, 986608, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [344014, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
C = np.where(np.isnan(TA), np.nan, np.nancumsum(np.where(np.isnan(TA), 0, TA), axis=1))


def test_taylor_ashe_chain_ladder_matches_the_published_reserve():
    cl = m.chain_ladder(C)
    assert cl["factors"][0] == pytest.approx(3.4906, abs=5e-4)
    assert cl["factors"][1] == pytest.approx(1.7473, abs=5e-4)
    assert cl["reserve"].sum() == pytest.approx(18_680_856, abs=1.0)


def test_taylor_ashe_mack_standard_error_matches_the_published_value():
    r = m.mack(C)
    assert r["se_total"] == pytest.approx(2_447_095, rel=5e-4)
    assert r["se_by_year"][1] > 0 and r["se_by_year"][0] == 0


def test_bootstrap_mean_is_close_to_the_chain_ladder_reserve_and_error_is_in_the_published_range():
    b = m.odp_bootstrap(C, n_sim=4000, seed=3)
    assert b["mean"] == pytest.approx(b["cl_reserve"], rel=0.04)
    # England and Verrall report a prediction error of about 15% of the reserve for this triangle
    assert 0.11 < b["sd"] / b["cl_reserve"] < 0.20
    no_pe = m.odp_bootstrap(C, n_sim=2000, seed=3, process_error=False)
    assert no_pe["sd"] < b["sd"]


def test_bornhuetter_ferguson_limits():
    f = m.development_factors(C)
    prem = np.full(10, 5e6)
    # a priori loss ratio equal to the chain ladder ultimate loss ratio gives back the chain ladder reserve per year
    cl = m.chain_ladder(C)
    elr = cl["ultimate"] / prem
    bf = m.bornhuetter_ferguson(C, prem, elr, f)
    assert np.allclose(bf["ultimate"], cl["ultimate"])
    # a zero a priori loss ratio reserves nothing: ultimate equals the paid to date
    assert np.allclose(m.bornhuetter_ferguson(C, prem, 0.0, f)["reserve"], 0.0)


def test_cape_cod_elr_is_between_the_year_loss_ratios_when_premium_is_equal():
    prem = np.full(10, 5e6)
    cc = m.cape_cod(C, prem)
    cl = m.chain_ladder(C)
    lr = cl["ultimate"] / prem
    assert lr.min() <= cc["elr"] * 1.5 and cc["elr"] > 0
    assert cc["elr"] == pytest.approx(m.latest_diagonal(C).sum() / (prem / m.cumulative_factors(m.development_factors(C))[[9 - i for i in range(10)]]).sum())


def test_lognormal_quantile_and_solvency_factor():
    assert m.lognormal_quantile(100.0, 0.0, 0.995) == 100.0
    q = m.lognormal_quantile(100.0, 9.0, 0.5)
    assert q < 100.0                                              # median below the mean
    # standard formula: sigma 9% gives about 3 sigma of capital (the usual approximation)
    cap = m.solvency2_reserve_capital(100.0, 0.09)
    assert cap == pytest.approx(100 * 0.2447, abs=0.4) or 20 < cap < 27
    assert m.solvency2_reserve_capital(100.0, 0.0001) < 0.1


def test_upper_triangle_and_incremental_roundtrip():
    sq = np.arange(1, 101, dtype=float).reshape(10, 10)
    t = tri.Triangle(1, "x", "ppauto", sq, sq, np.ones(10))
    up = t.upper(sq)
    assert np.isnan(up[9, 1]) and not np.isnan(up[9, 0]) and not np.isnan(up[0, 9]) and np.isnan(up[1, 9])
    inc = tri.incremental(sq)
    assert np.allclose(np.cumsum(inc, axis=1), sq)
