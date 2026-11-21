"""Estudio intradía con años de histórico de Binance (futuros USDT-M). Mismo protocolo que micro_study:
Spearman sin solape + Bonferroni + walk-forward logístico con coste. Señales disponibles en el histórico público:
flujo agresor (taker), base perp-índice, OI, ratios long/short y taker, y control de momentum.
No incluye libro de órdenes ni liquidaciones (ver micro_collect para eso).

Uso: python src/hist_study.py BTCUSDT [--cost 0.0010]"""
import argparse, json, os
import numpy as np
import pandas as pd
from paths import DATA
import micro_study as ms

HZ = (5, 15, 60, 240)


def builder(d):
    F = pd.DataFrame(index=d.index)
    vol = d.volume.where(d.close.notna())
    for n in (1, 5, 15, 60):
        b = d.taker_buy.rolling(n, min_periods=n).sum(); v = vol.rolling(n, min_periods=n).sum()
        F[f"flow_{n}m"] = (2 * b - v) / v
    if "premium" in d:
        F["premium"] = d.premium
        F["premium_chg_15m"] = d.premium.diff(15)
    if "oi" in d:
        F["oi_chg_1h"] = np.log(d.oi / d.oi.shift(60))
        F["oi_chg_4h"] = np.log(d.oi / d.oi.shift(240))
        F["ls_acc"] = np.log(d.ls_acc)
        F["ls_top"] = np.log(d.ls_top)
        F["taker_ratio"] = np.log(d.taker_ratio)
    F["ret_15m"] = np.log(d.close / d.close.shift(15)) * 1e4
    F["ret_60m"] = np.log(d.close / d.close.shift(60)) * 1e4
    F = F.replace([np.inf, -np.inf], np.nan)
    F = (F - F.rolling(1440, min_periods=360).mean()) / (F.rolling(1440, min_periods=360).std() + 1e-12)
    F[d.close.isna()] = np.nan
    return F, d.close


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="BTCUSDT")
    ap.add_argument("--cost", type=float, default=0.0010)
    a = ap.parse_args()
    d = pd.read_pickle(os.path.join(DATA, "binance", f"{a.symbol}_1m.pkl"))
    res = ms.study(d, a.cost, builder=builder, hz=HZ)
    res["symbol"] = a.symbol
    json.dump(res, open(os.path.join(DATA, f"hist_study_{a.symbol}.json"), "w"))
    ms.show(res)
