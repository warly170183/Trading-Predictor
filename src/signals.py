"""Banco de pruebas direccional: ¿alguna señal mejora la probabilidad de que BTC suba/baje?
Walk-forward expansivo, etiquetas sin solape (paso = horizonte), purga de etiquetas,
test por señal con bootstrap en bloques y corrección de Bonferroni."""
from paths import DATA
import json, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
COST = 0.0010  # 0,10 % ida y vuelta (comisión taker 2x0,05 %, sin deslizamiento extra)

a = np.array(json.load(open(f"{DATA}/btc_1h_long.json")))
df = pd.DataFrame(a, columns=["ts", "o", "h", "l", "c", "v"])
df["t"] = pd.to_datetime(df.ts, unit="ms", utc=True)
c = df.c
lr = np.log(c).diff()


def rsi(x, n=14):
    d = x.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / dn)


vol168 = lr.rolling(168).std()
F = pd.DataFrame(index=df.index)
for n in (6, 24, 72, 168):
    F[f"mom_{n}"] = np.log(c).diff(n) / (vol168 * np.sqrt(n))
for n in (50, 200):
    F[f"ema_dev_{n}"] = (c / c.ewm(span=n).mean() - 1) / vol168
F["rsi14"] = (rsi(c) - 50) / 25
sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
F["bb_z"] = (c - sma20) / sd20
F["vol_ratio"] = np.log(lr.rolling(24).std() / vol168)
F["vol_pct"] = vol168.rolling(2000, min_periods=500).rank(pct=True) - 0.5
mn, mx = c.rolling(168).min(), c.rolling(168).max()
F["range_pos"] = (c - mn) / (mx - mn) - 0.5
vz = (df.v - df.v.rolling(168).mean()) / df.v.rolling(168).std()
F["vol_z"] = vz
F["signed_vol"] = np.sign(lr) * vz
hr = df.t.dt.hour
F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
F = F.replace([np.inf, -np.inf], np.nan)
names = list(F.columns)
logc = np.log(c.values)


def fwd(h):
    return pd.Series(np.r_[logc[h:] - logc[:-h], [np.nan] * h], index=df.index)


def block_boot_mean(x, B=4000, blk=5):
    x = np.asarray(x, float)
    n = len(x)
    if n < 20:
        return (np.nan, np.nan)
    idx = np.arange(n)
    out = np.empty(B)
    nb = int(np.ceil(n / blk))
    for b in range(B):
        st = rng.integers(0, n, nb)
        ii = (st[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n
        out[b] = x[ii].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


res = {"n_candles": int(len(df)), "from": str(df.t.iloc[0]), "to": str(df.t.iloc[-1]), "cost": COST, "horizons": {}}
START = 6000  # primer origen OOS (≈ 8 meses de entrenamiento mínimo)
for h in (6, 24, 72):
    y_ret = fwd(h)
    ok = F.notna().all(axis=1) & y_ret.notna()
    origins = [i for i in range(START, len(df) - h, h) if ok.iloc[i]]
    P, Y, R, BASE = [], [], [], []
    coef_acc = np.zeros(len(names))
    nfit = 0
    for k, i in enumerate(origins):
        tr_idx = np.where(ok.values & (np.arange(len(df)) <= i - h))[0]
        tr_idx = tr_idx[::max(1, h // 3)]  # submuestreo para reducir autocorrelación
        Xtr, ytr = F.values[tr_idx], (y_ret.values[tr_idx] > 0).astype(int)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        m = LogisticRegression(C=0.02, max_iter=300).fit((Xtr - mu) / sd, ytr)
        p = m.predict_proba(((F.values[i] - mu) / sd)[None, :])[0, 1]
        P.append(p); Y.append(int(y_ret.values[i] > 0)); R.append(float(y_ret.values[i])); BASE.append(ytr.mean())
        coef_acc += m.coef_[0]; nfit += 1
    P, Y, R, BASE = map(np.array, (P, Y, R, BASE))
    n = len(P)
    acc = float(np.mean((P > 0.5) == (Y == 1)))
    acc_base = float(np.mean((BASE > 0.5) == (Y == 1)))  # siempre largo si la tasa base >50 %
    auc = float(roc_auc_score(Y, P)) if len(set(Y)) > 1 else float("nan")
    brier, brier_b = float(brier_score_loss(Y, P)), float(brier_score_loss(Y, BASE))
    # estrategia: largo si p>0.5, corto si p<0.5, con coste
    pos = np.where(P > 0.5, 1.0, -1.0)
    sret = pos * R - COST
    bh = R - COST / 1.0
    lo, hi = block_boot_mean(sret)
    # umbral de convicción
    thr_rows = []
    for th in (0.0, 0.02, 0.04, 0.06):
        sel = np.abs(P - 0.5) > th
        if sel.sum() >= 15:
            s = (np.where(P > 0.5, 1.0, -1.0) * R - COST)[sel]
            thr_rows.append({"th": th, "n": int(sel.sum()), "acc": float(np.mean(((P > 0.5) == (Y == 1))[sel])),
                             "mean_pct": float(s.mean() * 100), "sharpe_trade": float(s.mean() / s.std()) if s.std() > 0 else 0.0})
    # test binomial contra la tasa base de aciertos de "siempre largo"
    p_up = float(Y.mean())
    binom = stats.binomtest(int(round(acc * n)), n, max(p_up, 1 - p_up)).pvalue
    res["horizons"][str(h)] = {
        "n_oos": int(n), "p_up_real": p_up, "acc": acc, "acc_always_long": float(np.mean(Y == 1)), "acc_base": acc_base,
        "auc": auc, "brier": brier, "brier_base": brier_b,
        "strat_mean_pct": float(sret.mean() * 100), "strat_ci_pct": [lo * 100, hi * 100],
        "bh_mean_pct": float(bh.mean() * 100), "cum_strat_pct": float((np.exp(np.cumsum(np.log1p(np.clip(sret, -0.9, None)))[-1]) - 1) * 100),
        "cum_bh_pct": float((np.exp(np.cumsum(np.log1p(np.clip(bh, -0.9, None)))[-1]) - 1) * 100),
        "binom_p_vs_best_naive": float(binom), "thresholds": thr_rows,
        "coef": {names[j]: float(coef_acc[j] / nfit) for j in range(len(names))},
        "equity_strat": [float(v) for v in np.exp(np.cumsum(np.log1p(np.clip(sret, -0.9, None))))[::max(1, n // 80)]],
        "equity_bh": [float(v) for v in np.exp(np.cumsum(np.log1p(np.clip(bh, -0.9, None))))[::max(1, n // 80)]],
        "equity_t": [int(df.ts.iloc[origins[j]]) for j in range(0, n, max(1, n // 80))],
    }
    print(f"h={h:>2} n={n} acc={acc:.3f} base_long={np.mean(Y==1):.3f} AUC={auc:.3f} brier {brier:.4f} vs {brier_b:.4f} "
          f"strat {sret.mean()*100:+.3f}%/trade CI[{lo*100:+.3f},{hi*100:+.3f}] bh {bh.mean()*100:+.3f}% binom p={binom:.3f}", flush=True)

# IC por señal (Spearman) con t-stat por bloques, sobre muestras no solapadas
ic_rows = []
ntests = len(names) * 3
for h in (6, 24, 72):
    y_ret = fwd(h)
    ok = F.notna().all(axis=1) & y_ret.notna()
    idx = np.array([i for i in range(START, len(df) - h, h) if ok.iloc[i]])
    for nme in names:
        x, yv = F[nme].values[idx], y_ret.values[idx]
        rho, p = stats.spearmanr(x, yv)
        ic_rows.append({"h": h, "feature": nme, "ic": float(rho), "p": float(p), "p_bonf": float(min(1, p * ntests)), "n": int(len(idx))})
ic_rows.sort(key=lambda r: r["p"])
res["ic"] = ic_rows
res["n_tests"] = ntests
for r in ic_rows[:8]:
    print(f"IC h={r['h']:>2} {r['feature']:<12} {r['ic']:+.3f} p={r['p']:.3f} bonf={r['p_bonf']:.2f}")
json.dump(res, open(f"{DATA}/signals_output.json", "w"))
