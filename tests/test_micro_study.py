"""Prueba con datos sintéticos: el estudio debe detectar una señal plantada y no inventarla en ruido puro.
Ejecutar: python tests/test_micro_study.py"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from micro_study import study


def synth(plant, days=40, seed=5):
    rng = np.random.default_rng(seed)
    n = days * 1440
    imb = np.zeros(n)
    for t in range(1, n):
        imb[t] = 0.9 * imb[t - 1] + rng.normal(0, 0.2)
    imb = np.clip(imb, -1, 1)
    r = rng.normal(0, 0.0003, n)
    if plant:
        r[1:] += 0.00008 * imb[:-1]  # imb5 anticipa el retorno siguiente
    mid = 60000 * np.exp(np.cumsum(r))
    b = rng.gamma(2, 0.5, n); s = rng.gamma(2, 0.5, n)
    return pd.DataFrame({
        "ts": 1_700_000_000_000 + np.arange(n) * 60000, "mid": mid, "imb5": imb, "imb20": rng.normal(0, 0.3, n),
        "microprice_bps": rng.normal(0, 0.1, n), "depth_bid_btc": rng.gamma(5, 1, n), "depth_ask_btc": rng.gamma(5, 1, n),
        "basis_bps": -5 + rng.normal(0, 0.5, n).cumsum() * 0.01, "funding": np.full(n, 1e-5), "oi_btc": 30000 + rng.normal(0, 5, n).cumsum(),
        "tk_buy_btc": b, "tk_sell_btc": s, "missed": 0, "liq_long_btc": 0.0, "liq_short_btc": 0.0})


def hits(res):
    return {(r["feature"], r["h"]) for r in res["ic"] if r["p_bonf"] < 0.05}


null = hits(study(synth(False)))
plant = hits(study(synth(True)))
print("ruido puro, señales significativas:", null or "ninguna")
print("con señal plantada:", sorted(plant))
assert not null, "inventó señales en ruido puro"
assert ("imb5", 5) in plant, "no detectó la señal plantada"
print("OK")
