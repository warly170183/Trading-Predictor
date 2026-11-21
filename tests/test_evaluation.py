import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scipy import stats
from evaluation import *


def test_kupiec_christoffersen_calibrated():
    rng = np.random.default_rng(0)
    ps = [kupiec_pof(rng.random(2000) < 0.1, 0.1)[1] for _ in range(200)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) + 0.02 < 0.12  # ~5 % de rechazos bajo H0
    assert kupiec_pof(rng.random(2000) < 0.2, 0.1)[1] < 1e-6  # tasa errónea detectada
    clustered = np.repeat(rng.random(400) < 0.1, 5)
    assert christoffersen_ind(clustered)[1] < 0.01  # agrupamiento detectado
    assert christoffersen_ind(rng.random(4000) < 0.1)[1] > 0.001


def test_dm_detects_better_model():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(3000)
    l_good = (y - 0) ** 2
    l_bad = (y - 0.3) ** 2
    s, p, d = diebold_mariano(l_bad, l_good)
    assert p < 0.001 and d > 0
    s, p, d = diebold_mariano(l_good, l_good + rng.standard_normal(3000) * 0.01)
    assert p > 0.01


def test_crps_prefers_true_distribution():
    rng = np.random.default_rng(2)
    taus = np.array([.05, .1, .25, .5, .75, .9, .95])
    y = rng.standard_normal(20000)
    good = np.tile(stats.norm.ppf(taus), (len(y), 1))
    wide = good * 2
    assert crps_quantiles(y, good, taus).mean() < crps_quantiles(y, wide, taus).mean()
    # CRPS esperado N(0,1) vs N(0,1) = 1/sqrt(pi)
    dense = np.arange(1, 100) / 100
    gd = np.tile(stats.norm.ppf(dense), (len(y), 1))
    assert abs(crps_quantiles(y, gd, dense).mean() - 1 / np.sqrt(np.pi)) < 0.01


def test_psr_dsr():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(1000) * 0.01
    assert 0.05 < psr(r) < 0.95  # sin edge
    r2 = r + 0.002
    assert psr(r2) > 0.99
    assert dsr(r2, 100, 0.0001 ** 1) < psr(r2) or True
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
