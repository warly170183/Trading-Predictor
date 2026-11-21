"""Estudio intradía: ¿anticipan el retorno a 5, 15 y 60 minutos el desequilibrio del libro, el flujo agresivo,
las liquidaciones y la base? Correlación de Spearman por señal con muestras sin solape (paso = horizonte), corrección de
Bonferroni, y un walk-forward logístico con coste de operación.

Uso: python src/micro_study.py [--cost 0.0010] [--min-rows 5000] [--force]
Lee data/micro/*.jsonl y journal/micro/*.csv.gz. Escribe data/micro_study.json."""
import argparse, glob, gzip, json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from paths import DATA, JOURNAL

warnings.filterwarnings("ignore")
HZ = (5, 15, 60)


def load():
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, "micro", "*.jsonl"))):
        frames.append(pd.read_json(f, lines=True))
    if not frames:  # sin jsonl locales: usa los CSV versionados
        for f in sorted(glob.glob(os.path.join(JOURNAL, "micro", "*.csv.gz"))):
            frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).drop_duplicates("ts").sort_values("ts")
    return df.reset_index(drop=True)


def features(df):
    """Rejilla de 1 minuto con señales estandarizadas (z-score móvil de 1 día, solo pasado)."""
    d = df.set_index(pd.to_datetime(df.ts, unit="ms", utc=True)).drop(columns="ts")
    d = d.asfreq("60s")
    bad = d["missed"].fillna(1).astype(bool) | d["mid"].isna()
    flow = (d.tk_buy_btc - d.tk_sell_btc) / (d.tk_buy_btc + d.tk_sell_btc).replace(0, np.nan)
    F = pd.DataFrame(index=d.index)
    F["imb5"] = d.imb5
    F["imb20"] = d.imb20
    F["microprice"] = d.microprice_bps
    F["depth_ratio"] = np.log(d.depth_bid_btc / d.depth_ask_btc)
    F["flow_1m"] = flow.where(~d.missed.fillna(1).astype(bool))
    for n in (5, 15):
        b = d.tk_buy_btc.where(~d.missed.fillna(1).astype(bool)).rolling(n, min_periods=n).sum()
        s = d.tk_sell_btc.where(~d.missed.fillna(1).astype(bool)).rolling(n, min_periods=n).sum()
        F[f"flow_{n}m"] = (b - s) / (b + s)
    ln = d.liq_long_btc.fillna(0).rolling(15).sum()
    sh = d.liq_short_btc.fillna(0).rolling(15).sum()
    F["liq_net_15m"] = np.sign(sh - ln) * np.log1p((sh - ln).abs())  # positivo: se liquidaron más cortos (presión compradora)
    F["basis"] = d.basis_bps
    F["basis_chg_15m"] = d.basis_bps.diff(15)
    F["funding"] = d.funding * 1e4
    F["oi_chg_15m"] = np.log(d.oi_btc / d.oi_btc.shift(15))
    F["ret_15m"] = np.log(d.mid / d.mid.shift(15)) * 1e4  # control: momentum a corto plazo
    F = F.replace([np.inf, -np.inf], np.nan)
    F = (F - F.rolling(1440, min_periods=360).mean()) / (F.rolling(1440, min_periods=360).std() + 1e-12)
    F[bad] = np.nan
    return F, d.mid


def fwd(mid, h):
    return np.log(mid.shift(-h) / mid)


def study(df, cost=0.0010):
    F, mid = features(df)
    names = list(F.columns)
    ntests = len(names) * len(HZ)
    res = {"n_rows": int(len(df)), "from": str(F.index[0]), "to": str(F.index[-1]), "n_tests": ntests, "cost": cost, "ic": [], "wf": {}}
    for h in HZ:
        y = fwd(mid, h)
        ok = F.notna().all(axis=1) & y.notna()
        idx = np.where(ok.values)[0]
        sel = idx[::h]  # sin solape
        for nme in names:
            m = F[nme].notna().values & y.notna().values
            ii = np.where(m)[0][::h]
            if len(ii) < 200:
                continue
            rho, p = stats.spearmanr(F[nme].values[ii], y.values[ii])
            res["ic"].append({"h": h, "feature": nme, "ic": float(rho), "p": float(p), "p_bonf": float(min(1, p * ntests)), "n": int(len(ii))})
        # walk-forward logístico
        if len(sel) < 600:
            continue
        P, Y, R = [], [], []
        start = int(len(sel) * 0.4)
        for k in range(start, len(sel), 20):
            tr = sel[:k]  # las etiquetas de entrenamiento terminan antes del primer origen del bloque (paso = horizonte)
            blk = sel[k:k + 20]
            if len(tr) < 300:
                continue
            X, yy = F.values[tr], (y.values[tr] > 0).astype(int)
            mu, sd = X.mean(0), X.std(0) + 1e-9
            mdl = LogisticRegression(C=0.02, max_iter=300).fit((X - mu) / sd, yy)
            P += list(mdl.predict_proba((F.values[blk] - mu) / sd)[:, 1])
            Y += list((y.values[blk] > 0).astype(int)); R += list(y.values[blk])
        P, Y, R = map(np.array, (P, Y, R))
        if len(P) < 100:
            continue
        pos = np.where(P > 0.5, 1.0, -1.0)
        gross = pos * R
        net = gross - cost
        rng = np.random.default_rng(3)
        bs = [net[rng.integers(0, len(net), len(net))].mean() for _ in range(2000)]
        res["wf"][str(h)] = {
            "n": int(len(P)), "acc": float(np.mean((P > 0.5) == (Y == 1))), "acc_naive": float(max(Y.mean(), 1 - Y.mean())),
            "auc": float(roc_auc_score(Y, P)) if len(set(Y)) > 1 else None,
            "gross_pct": float(gross.mean() * 100), "net_pct": float(net.mean() * 100),
            "net_ci_pct": [float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)],
            "mean_abs_move_pct": float(np.abs(R).mean() * 100),
        }
    res["ic"].sort(key=lambda r: r["p"])
    return res


def show(res):
    print(f"{res['n_rows']} filas · {res['from']} → {res['to']} · {res['n_tests']} pruebas · coste {res['cost']*100:.2f} %")
    for r in res["ic"][:10]:
        print(f"  IC {r['h']:>2} min {r['feature']:<13} {r['ic']:+.3f} p={r['p']:.4f} bonf={r['p_bonf']:.3f} n={r['n']}")
    for h, w in res["wf"].items():
        ci = w["net_ci_pct"]
        print(f"  WF {h:>2} min n={w['n']} acierto {w['acc']*100:.1f} % (ingenuo {w['acc_naive']*100:.1f} %) AUC {w['auc']:.3f} bruto {w['gross_pct']:+.4f} % neto {w['net_pct']:+.4f} % [{ci[0]:+.4f},{ci[1]:+.4f}] movimiento medio {w['mean_abs_move_pct']:.3f} %")
    sig = [r for r in res["ic"] if r["p_bonf"] < 0.05]
    print("Señales que superan Bonferroni:", [(r["feature"], r["h"]) for r in sig] or "ninguna")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.0010)
    ap.add_argument("--min-rows", type=int, default=5000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    df = load()
    if len(df) < a.min_rows and not a.force:
        print(f"Datos insuficientes: {len(df)} filas de {a.min_rows} necesarias (~{a.min_rows/1440:.1f} días). Sigue capturando.")
        raise SystemExit(0)
    res = study(df, a.cost)
    json.dump(res, open(os.path.join(DATA, "micro_study.json"), "w"))
    show(res)
