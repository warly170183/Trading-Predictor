"""Conformal adaptativo (ACI, Gibbs & Candès 2021) con retroalimentación retrasada, sobre el cono de volatilidad.

Puntuación de no conformidad: s_t = |r_t| / sigma_t (retorno absoluto estandarizado). El intervalo simétrico al nivel
1-alpha_t es ±sigma·q, con q = cuantil empírico de las s pasadas ya conocidas. alpha_t se ajusta tras cada resultado
conocido: alpha ← alpha + gamma·(alpha_obj − fallo). Con horizonte h y paso STEP, el resultado de un origen solo se
conoce h/STEP orígenes después (retraso); el ajuste usa solo resultados ya conocidos."""
import numpy as np


def aci_intervals(ret, sig, h_lag, alpha=0.10, gamma=0.02, cal=300, min_cal=80):
    n = len(ret)
    a_t = alpha
    q_out = np.full(n, np.nan); hit = np.full(n, np.nan); a_hist = np.full(n, np.nan)
    s = np.abs(ret) / sig
    for k in range(n):
        known = np.arange(0, max(k - h_lag + 1, 0))
        known = known[np.isfinite(s[known])][-cal:]
        if len(known) < min_cal:
            continue
        # actualiza alpha con el resultado que se acaba de conocer (origen k - h_lag)
        j = k - h_lag
        if j >= 0 and np.isfinite(hit[j]):
            a_t = a_t + gamma * (alpha - (1 - hit[j]))
        a_eff = min(max(a_t, 0.005), 0.5)
        q = np.quantile(s[known], 1 - a_eff)
        q_out[k] = q * sig[k]
        hit[k] = abs(ret[k]) <= q_out[k]
        a_hist[k] = a_t
    return q_out, hit, a_hist
