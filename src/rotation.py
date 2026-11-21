"""Rotación transversal entre ~40 perpetuos USDT de OKX (momentum relativo diario), con protocolo preregistrado.

PREREGISTRO (fijado antes de ver resultados):
  Universo: los 40 swaps USDT de OKX con más volumen 24 h (sin stablecoins), con ≥ 120 velas diarias propias.
  Primaria: momentum 30 d saltando el último día (ln P[t-1]/P[t-31]); comprar los 5 mejores a pesos iguales;
            rebalanceo semanal; resultado = exceso sobre la cartera de pesos iguales del universo; coste 0,05 % por
            unidad de rotación de un lado (0,10 % ida y vuelta) sobre la rotación de la estrategia.
  Variantes (cuentan como ensayos para el Sharpe deflactado): lookback {7,14,30,60,90} × top-k {3,5,8} × rebalanceo {3,7,14} d.
  Criterio: Sharpe deflactado (DSR) > 0,95 con n_ensayos = total de variantes, t de Newey-West > 3 y mismo signo en las
            dos mitades de la muestra.
SESGO DE SUPERVIVENCIA: el universo se toma HOY (monedas que siguen listadas); sobrestima los resultados de comprar
ganadoras. Es una cota optimista, no un resultado operable.

Uso: python src/rotation.py [--fetch]"""
import argparse, itertools, json, os, time
import numpy as np
import pandas as pd
import requests
from paths import DATA
import evaluation as ev

N_UNI = 40
STABLE = ("USDC", "USDG", "DAI", "TUSD", "FDUSD", "USDD", "PYUSD", "USTC")
LOOKS, KS, REBS = (7, 14, 30, 60, 90), (3, 5, 8), (3, 7, 14)
COST = 0.0005
OUT = os.path.join(DATA, "rotation")
os.makedirs(OUT, exist_ok=True)


def fetch_universe(n_bars=1100):
    tk = requests.get("https://www.okx.com/api/v5/market/tickers", params={"instType": "SWAP"}, timeout=30).json()["data"]
    rows = [(t["instId"], float(t["volCcy24h"]) * float(t["last"])) for t in tk if t["instId"].endswith("-USDT-SWAP")]
    rows = [r for r in rows if r[0].split("-")[0] not in STABLE]
    rows.sort(key=lambda x: -x[1])
    insts = [r[0] for r in rows[:N_UNI + 5]]
    series = {}
    for inst in insts:
        data, after = [], None
        while len(data) < n_bars:
            p = {"instId": inst, "bar": "1D", "limit": 100}
            if after:
                p["after"] = after
            d = requests.get("https://www.okx.com/api/v5/market/history-candles", params=p, timeout=20).json().get("data", [])
            if not d:
                break
            data += d; after = d[-1][0]; time.sleep(0.12)
        s = pd.Series({pd.to_datetime(int(x[0]), unit="ms", utc=True): float(x[4]) for x in data if x[8] == "1"}).sort_index()
        if len(s) >= 120:
            series[inst.replace("-USDT-SWAP", "")] = s
        if len(series) >= N_UNI:
            break
    P = pd.DataFrame(series).sort_index()
    P.to_pickle(os.path.join(OUT, "daily_close.pkl"))
    return P


def backtest(P, look, k, reb):
    R = P.pct_change()
    lp = np.log(P)
    sig = (lp.shift(1) - lp.shift(look + 1))  # ln P[t-1]/P[t-look-1]
    ew_ret = R.mean(axis=1)
    w = pd.DataFrame(0.0, index=P.index, columns=P.columns)
    cur = pd.Series(0.0, index=P.columns)
    for i, t in enumerate(P.index):
        if i > look + 1 and (i % reb == 0):
            s = sig.loc[t].dropna()
            if len(s) >= k + 5:
                cur = pd.Series(0.0, index=P.columns)
                cur[s.nlargest(k).index] = 1.0 / k
        w.loc[t] = cur
    held = w.shift(1).fillna(0.0)  # el peso fijado al cierre t se gana desde t+1
    turn = held.diff().abs().sum(axis=1) / 2 * 2  # rotación de un lado sumada (compras+ventas)
    gross = (held * R).sum(axis=1)
    net = gross - COST * held.diff().abs().sum(axis=1)
    start = look + 3
    exc = (net - ew_ret).iloc[start:].dropna()
    return exc, net.iloc[start:].dropna(), ew_ret.iloc[start:].dropna()


def stats_for(x):
    x = np.asarray(x)
    half = len(x) // 2
    return {"n_days": int(len(x)), "sharpe_ann": float(ev.sharpe(x) * np.sqrt(365)), "mean_bps_day": float(x.mean() * 1e4),
            "nw_t": ev.nw_tstat(x), "sharpe_h1": float(ev.sharpe(x[:half]) * np.sqrt(365)), "sharpe_h2": float(ev.sharpe(x[half:]) * np.sqrt(365))}


def run():
    P = pd.read_pickle(os.path.join(OUT, "daily_close.pkl"))
    res, series, srs = {}, {}, []
    for look, k, reb in itertools.product(LOOKS, KS, REBS):
        exc, net, ew = backtest(P, look, k, reb)
        key = f"L{look}_k{k}_r{reb}"
        res[key] = stats_for(exc); series[key] = exc
        srs.append(ev.sharpe(exc.values))
    ntr, var_sr = len(res), float(np.var(srs, ddof=1))
    prim = "L30_k5_r7"
    x = series[prim].values
    rng = np.random.default_rng(5)
    blk = 14
    nb = int(np.ceil(len(x) / blk)); idx = rng.integers(0, len(x) - blk + 1, size=(3000, nb))
    boots = np.array([x[(i[:, None] + np.arange(blk)).ravel()[:len(x)]].mean() for i in idx])
    prim_res = dict(res[prim], psr0=ev.psr(x), dsr=ev.dsr(x, ntr, var_sr), exp_max_sharpe_day=ev.expected_max_sharpe(ntr, var_sr),
                    mean_ci95_bps_day=[float(np.percentile(boots, 2.5) * 1e4), float(np.percentile(boots, 97.5) * 1e4)])
    exc, net, ew = backtest(P, 30, 5, 7)
    best = max(res, key=lambda k_: res[k_]["sharpe_ann"])
    out = {"universe": list(P.columns), "n_assets": P.shape[1], "from": str(P.index[0].date()), "to": str(P.index[-1].date()), "n_trials": ntr,
           "primary": prim, "primary_result": prim_res, "primary_abs_net": stats_for(net.values), "benchmark_ew": stats_for(ew.values),
           "best_variant": best, "best_variant_result": res[best], "best_dsr": ev.dsr(series[best].values, ntr, var_sr),
           "share_variants_sharpe_pos": float(np.mean([v["sharpe_ann"] > 0 for v in res.values()])), "variants": res}
    json.dump(out, open(os.path.join(DATA, "rotation.json"), "w"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch", action="store_true"); a = ap.parse_args()
    if a.fetch or not os.path.exists(os.path.join(OUT, "daily_close.pkl")):
        P = fetch_universe(); print("universo", P.shape)
    o = run()
    p = o["primary_result"]
    print(f"{o['n_assets']} monedas · {o['from']} → {o['to']} · {o['n_trials']} ensayos")
    print(f"PRIMARIA {o['primary']}: exceso sobre EW  Sharpe {p['sharpe_ann']:.2f} · t(NW) {p['nw_t']:.2f} · {p['mean_bps_day']:.1f} pb/día IC95 [{p['mean_ci95_bps_day'][0]:.1f},{p['mean_ci95_bps_day'][1]:.1f}] · mitades {p['sharpe_h1']:.2f}/{p['sharpe_h2']:.2f} · PSR {p['psr0']:.2f} · DSR {p['dsr']:.2f}")
    b = o["primary_abs_net"]; e = o["benchmark_ew"]
    print(f"  absoluto neto Sharpe {b['sharpe_ann']:.2f} · EW universo Sharpe {e['sharpe_ann']:.2f}")
    print(f"  mejor variante {o['best_variant']}: Sharpe {o['best_variant_result']['sharpe_ann']:.2f} DSR {o['best_dsr']:.2f} · {o['share_variants_sharpe_pos']*100:.0f} % de variantes con Sharpe>0")
