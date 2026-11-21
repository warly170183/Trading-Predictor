"""Evaluación estadística de predicciones probabilísticas y de estrategias (implementación propia, solo numpy/scipy).

- crps_quantiles: CRPS aproximado a partir de cuantiles (2 × pinball medio sobre una rejilla de niveles).
- diebold_mariano: contraste de igual precisión con varianza HAC (Newey-West) y corrección de Harvey-Leybourne-Newbold.
- kupiec_pof, christoffersen_ind, christoffersen_cc: cobertura incondicional, independencia de fallos y conjunta.
- psr, dsr: Sharpe probabilístico y Sharpe deflactado (Bailey & López de Prado)."""
import numpy as np
from scipy import stats

EULER = 0.5772156649015329


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def crps_quantiles(y, Q, taus):
    """y: (n,), Q: (n, k) cuantiles, taus: (k,). CRPS ≈ 2·media_k pinball (exacto en el límite de rejilla densa)."""
    y = np.asarray(y)[:, None]
    taus = np.asarray(taus)[None, :]
    return 2 * pinball(y, np.asarray(Q), taus).mean(axis=1)


def diebold_mariano(l1, l2, h=1, hln=True):
    """H0: E[l1 - l2] = 0. d>0 en favor de 2 (l2 menor). Devuelve (estadístico, p bilateral, d medio)."""
    d = np.asarray(l1) - np.asarray(l2)
    n = len(d)
    dm = d.mean()
    dc = d - dm
    L = max(h - 1, 0)
    g0 = dc @ dc / n
    var = g0
    for k in range(1, L + 1):
        gk = dc[k:] @ dc[:-k] / n
        var += 2 * (1 - k / (L + 1)) * gk
    var = max(var, 1e-18) / n
    stat = dm / np.sqrt(var)
    if hln:
        stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    else:
        p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p), float(dm)


def kupiec_pof(hits, p):
    """hits: 1 = observación fuera del intervalo; p: tasa nominal de fallos. LR ~ chi2(1)."""
    x = int(np.sum(hits)); n = len(hits)
    ph = min(max(x / n, 1e-12), 1 - 1e-12)
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - ph) + x * np.log(ph)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_ind(hits):
    """Independencia de fallos consecutivos (cadena de Markov de primer orden). LR ~ chi2(1)."""
    h = np.asarray(hits).astype(int)
    a, b = h[:-1], h[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    def ll(pp, nn, kk):
        return (nn * np.log(max(1 - pp, 1e-12)) if nn else 0) + (kk * np.log(max(pp, 1e-12)) if kk else 0)
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l1 = ll(p01, n00, n01) + ll(p11, n10, n11)
    l0 = ll(p, n00 + n10, n01 + n11)
    lr = max(-2 * (l0 - l1), 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_cc(hits, p):
    lr1, _ = kupiec_pof(hits, p)
    lr2, _ = christoffersen_ind(hits)
    lr = lr1 + lr2
    return float(lr), float(1 - stats.chi2.cdf(lr, 2))


def sharpe(r):
    r = np.asarray(r)
    return r.mean() / (r.std(ddof=1) + 1e-18)


def psr(r, sr_bench=0.0):
    """Probabilidad de que el Sharpe verdadero supere sr_bench (Sharpe por periodo, no anualizado)."""
    r = np.asarray(r); n = len(r)
    sr = sharpe(r)
    sk = stats.skew(r); ku = stats.kurtosis(r, fisher=False)
    den = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr_bench) * np.sqrt(n - 1) / den
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials, var_sr):
    """Sharpe máximo esperado entre n_trials estrategias sin edge (por periodo)."""
    if n_trials <= 1:
        return 0.0
    s = np.sqrt(var_sr)
    return float(s * ((1 - EULER) * stats.norm.ppf(1 - 1 / n_trials) + EULER * stats.norm.ppf(1 - 1 / (n_trials * np.e))))


def dsr(r, n_trials, var_sr_trials):
    """PSR frente al Sharpe máximo esperado por azar tras probar n_trials variantes."""
    return psr(r, expected_max_sharpe(n_trials, var_sr_trials))


def nw_tstat(x, lags=None):
    """t de la media con Newey-West."""
    x = np.asarray(x); n = len(x)
    lags = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    xc = x - x.mean()
    v = xc @ xc / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (xc[k:] @ xc[:-k] / n)
    return float(x.mean() / np.sqrt(v / n))
