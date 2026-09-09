"""The pure helpers behind the §3 detector, the §5 fit and the §6 protocol (export cell)."""
import math

import pytest


def test_parabola_vertex_known_parabola(nb):
    xv, ymin = nb.parabola_vertex([-1, 0, 1], [1, 0, 1])
    assert xv == 0 and ymin == 0
    xv, ymin = nb.parabola_vertex([-9, -8.5, -8], [1.5, 1.4, 1.6])      # y = 1.4 + 0.6*(x+8.583)^2-ish
    assert -9 < xv < -8 and ymin < 1.4
    assert nb.parabola_vertex([0, 1, 2], [0, 1, 0]) == (None, None)     # opens downward


def test_seed_vertex_edge_and_interior(nb):
    xv, ymin, kmin, edge = nb.seed_vertex({-10: 1.5, -9: 1.4, -8: 1.6})
    assert kmin == -9 and not edge and -10 < xv < -8
    xv, ymin, kmin, edge = nb.seed_vertex({-10: 1.3, -9: 1.4, -8: 1.6})
    assert edge and kmin == -10 and xv == -10


def test_regression_matches_closed_form(nb):
    xs = [8, 8, 9, 9, 10, 10]
    ys = [-8.8, -8.6, -9.9, -9.7, -10.5, -10.3]
    r = nb.regression(xs, ys, 12)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    assert abs(r["slope"] - slope) < 1e-12 and abs(r["pred"] - (my - slope * mx + slope * 12)) < 1e-12
    sigma = math.sqrt(sum((y - (my - slope * mx + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    assert abs(r["pi_half"] - nb.t975(n - 2) * sigma * math.sqrt(1 + 1 / n + (12 - mx) ** 2 / sxx)) < 1e-12
    assert r["dof"] == 4 and abs(nb.t975(4) - 2.776) < 1e-9 and abs(nb.t975(17) - 2.1105) < 1e-3


def test_detect_peak_synthetic_ramp(nb):
    for W in (50, 100, 200):
        syn = [min(1.0, (t + 1) / W) * 3e-3 for t in range(300)]
        assert nb.detect_peak(syn) == W


def test_slope_change_on_a_ramp(nb):
    series = [min(1.0, (t + 1) / 100) for t in range(300)]        # log slope ~1/t before W, 0 after
    s = nb.slope_change(series, 100, 10)
    assert 0.009 < s < 0.012


def _side(seeds, **kw):
    return {"argmin_log2": -9.0, "on_edge": False, "seeds": seeds, "config": {"steps": 300}, "stream_sha": ["a", "b", "c"], **kw}


def test_fair_compare_refusals_and_verdict(nb):
    assert nb.fair_compare("a", _side([1.50, 1.52, 1.54], on_edge=True), "b", _side([1.60, 1.62, 1.64]))["verdict"].startswith("REFUSED")
    assert nb.fair_compare("a", _side([1.50, 1.52, 1.54]), "b", _side([1.60, 1.62, 1.64], config={"steps": 999}))["verdict"].startswith("REFUSED")
    assert nb.fair_compare("a", _side([1.50, 1.52, 1.54]), "b", _side([1.60, 1.62, 1.64], stream_sha=["x", "b", "c"]))["verdict"].startswith("REFUSED")
    assert nb.fair_compare("a", _side([1.50, 1.52, 1.54]), "b", _side([1.51, 1.53, 1.55]))["verdict"].startswith("REFUSED")
    assert nb.fair_compare("a", _side([1.50, 1.52, 1.54]), "b", _side([1.60, 1.62, 1.64]))["verdict"].startswith("a wins")
    assert nb.fair_compare("a", _side([1.60, 1.62, 1.64]), "b", _side([1.50, 1.52, 1.54]))["verdict"].startswith("b wins")


def test_cache_key_covers_the_hyperparameters(nb):
    import hashlib, json
    def key(cfg):
        return hashlib.sha256(json.dumps({**cfg, "_cache_cfg": nb.CACHE_CFG}, sort_keys=True).encode()).hexdigest()[:16]
    base = {"section": "sweep", "d": 256, "lr": 1e-3, "seed": 1, "opt": "adamw"}
    assert key(base) != key({**base, "lr": 2e-3})
    assert "betas" in nb.CACHE_CFG and "weight_decay" in nb.CACHE_CFG and "corpus" in nb.CACHE_CFG
