"""Pronóstico de desviación típica a h horas: GJR-GARCH, HAR y su ensamble (media geométrica).
El ensamble mejoró la puntuación del cono en BTC, ETH y SOL (ver RESULTS.md)."""
import numpy as np
from engine import fit_garch


def sigma_garch(lr, hs, win=2500):
    res = fit_garch(lr[-win:] * 100)
    fv = res.forecast(horizon=max(hs), reindex=False).variance.values[-1] / 1e4
    cum = np.cumsum(fv)
    return {h: float(np.sqrt(cum[h - 1])) for h in hs}


def sigma_har(lr, hs):
    n = len(lr)
    r2 = lr ** 2
    cs = np.concatenate([[0], np.cumsum(r2)])
    m = lambda lo, hi: (cs[hi] - cs[lo]) / (hi - lo)
    out = {}
    for h in hs:
        jj = np.arange(720, n - h + 1, 6)
        X = np.column_stack([np.ones(len(jj)), np.log(m(jj - 24, jj) + 1e-12), np.log(m(jj - 168, jj) + 1e-12), np.log(m(jj - 720, jj) + 1e-12)])
        y = np.log(np.array([m(j, j + h) for j in jj]) + 1e-12)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        smear = np.mean(np.exp(y - X @ beta))
        x = np.array([1, np.log(m(n - 24, n) + 1e-12), np.log(m(n - 168, n) + 1e-12), np.log(m(n - 720, n) + 1e-12)])
        out[h] = float(np.sqrt(h * np.exp(x @ beta) * smear))
    return out


def ensemble_ratio(lr, hs):
    """Cociente sigma_ensamble / sigma_garch por horizonte."""
    g, hh = sigma_garch(lr, hs), sigma_har(lr, hs)
    return {h: float(np.sqrt(hh[h] / g[h])) for h in hs}  # exp((log g + log h)/2) / g
