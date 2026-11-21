"""Sesgo direccional diario: ¿funding, DVOL, OI, long/short, flujo taker y régimen HMM mejoran la predicción de dirección?
Walk-forward expansivo, etiquetas purgadas, comparación pareada contra el modelo de solo precio y corrección de pruebas múltiples."""
import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
rng = np.random.default_rng(23)
from paths import DATA as D
COST = 0.0010
DAY = 86400000

# ---------- tabla diaria ----------
c = np.array(json.load(open(f"{D}/btc_1d_long.json")))
df = pd.DataFrame(c, columns=["ts", "o", "h", "l", "c", "v"])
df["ts"] = df.ts.astype("int64")
close = df.c.values
lr1 = np.log(df.c).diff()

fund = np.array(json.load(open(f"{D}/deribit_funding.json")))
cum = np.cumsum(fund[:, 1])
ft = fund[:, 0].astype("int64")
def fsum(T):
    a, b = np.searchsorted(ft, T, "right"), np.searchsorted(ft, T + DAY, "right")
    return (cum[b - 1] - cum[a - 1]) if b > a and a > 0 else np.nan
df["fund"] = [fsum(T) * 1e4 for T in df.ts]  # puntos básicos por día

dv = np.array(json.load(open(f"{D}/deribit_dvol.json")))
dvt = dv[:, 0].astype("int64")
def dval(T):
    i = np.searchsorted(dvt, T + DAY, "right") - 1
    return dv[i, 4] if i >= 0 and abs(dvt[i] - (T + DAY)) <= 3600000 * 2 else np.nan
df["dvol"] = [dval(T) for T in df.ts]

def rub(name, col):
    a = np.array(json.load(open(f"{D}/{name}.json")))
    m = {int(r[0]): r[col] for r in a}
    return df.ts.map(m)
df["oi"] = rub("okx_oi_1d", 2)      # OI en BTC
df["ls"] = rub("okx_ls_1d", 1)      # ratio cuentas largas/cortas

# ---------- features ----------
vol30 = lr1.rolling(30).std()
F = pd.DataFrame(index=df.index)
for n in (1, 3, 7, 30):
    F[f"mom_{n}"] = np.log(df.c).diff(n) / (vol30 * np.sqrt(n))
for n in (20, 50):
    F[f"ema_dev_{n}"] = (df.c / df.c.ewm(span=n).mean() - 1) / vol30
def rsi(x, n=14):
    d = x.diff(); up = d.clip(lower=0).ewm(alpha=1 / n).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / dn)
F["rsi14"] = (rsi(df.c) - 50) / 25
F["bb_z"] = (df.c - df.c.rolling(20).mean()) / df.c.rolling(20).std()
F["vol_ratio"] = np.log(lr1.rolling(7).std() / vol30)
mn, mx = df.c.rolling(30).min(), df.c.rolling(30).max()
F["range_pos"] = (df.c - mn) / (mx - mn) - 0.5
BASE = list(F.columns)

zs = lambda s, n=90: (s - s.rolling(n).mean()) / s.rolling(n).std()
NEW = {}
NEW["fund_1d_z"] = zs(df.fund)
NEW["fund_7d_z"] = zs(df.fund.rolling(7).mean())
NEW["dvol_z"] = zs(df.dvol)
NEW["dvol_chg_1d"] = df.dvol.pct_change() * 100
rv7 = lr1.rolling(7).std() * np.sqrt(365) * 100
NEW["iv_rv"] = np.log(df.dvol / rv7)
NEW["oi_chg_1d"] = df.oi.pct_change() * 100
NEW["oi_chg_3d"] = df.oi.pct_change(3) * 100
NEW["oi_chg_7d"] = df.oi.pct_change(7) * 100
NEW["oi_z"] = zs(df.oi)
NEW["oi_signed"] = np.sign(lr1) * df.oi.pct_change() * 100   # OI creciendo en la dirección del movimiento
NEW["ls_z"] = zs(df.ls)
NEW["ls_chg_1d"] = df.ls.pct_change() * 100
for k, v in NEW.items():
    F[k] = v
F = F.replace([np.inf, -np.inf], np.nan)
NEWC = list(NEW.keys())
LAB = {"fund_1d_z": "Funding diario (z 90 d)", "fund_7d_z": "Funding 7 d (z 90 d)", "dvol_z": "DVOL (z 90 d)", "dvol_chg_1d": "Cambio diario del DVOL",
       "iv_rv": "Volatilidad implícita / realizada", "oi_chg_1d": "Cambio de OI 1 d", "oi_chg_3d": "Cambio de OI 3 d", "oi_chg_7d": "Cambio de OI 7 d",
       "oi_z": "OI (z 90 d)", "oi_signed": "OI con signo del movimiento", "ls_z": "Ratio largos/cortos (z 90 d)", "ls_chg_1d": "Cambio del ratio largos/cortos",
       }

logc = np.log(close)
def fwd(h):
    return np.r_[logc[h:] - logc[:-h], [np.nan] * h]


def block_ci(x, B=4000, blk=5):
    x = np.asarray(x, float); n = len(x); nb = int(np.ceil(n / blk)); out = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n, nb); ii = (st[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n
        out[b] = x[ii].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def wf(cols, h, valid, min_train=300, C=0.05):
    y = fwd(h)
    ok = F[cols].notna().all(axis=1).values & np.isfinite(y) & valid
    pos = np.where(valid & F[cols].notna().all(axis=1).values)[0]
    first = pos[min_train] if len(pos) > min_train else None
    out = {}
    if first is None:
        return out
    for i in range(first, len(df) - h, h):
        if not (valid[i] and np.isfinite(y[i])):
            continue
        tr = np.where(ok & (np.arange(len(df)) <= i - h))[0]
        Xtr, ytr = F[cols].values[tr], (y[tr] > 0).astype(int)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        m = LogisticRegression(C=C, max_iter=400).fit((Xtr - mu) / sd, ytr)
        p = m.predict_proba(((F[cols].values[i] - mu) / sd)[None, :])[0, 1]
        out[i] = (p, float(y[i]), float(ytr.mean()))
    return out


# filas donde todo (incluidas las nuevas features) existe: base común para comparar A y B
allcols = BASE + NEWC
valid_all = F[allcols].notna().all(axis=1).values
print("filas diarias:", len(df), "| con todas las features:", int(valid_all.sum()))

res = {"n_days": int(len(df)), "n_valid": int(valid_all.sum()), "cost": COST, "horizons": {}, "labels": LAB}
ev = {}
for h in (1, 3):
    A = wf(BASE, h, valid_all); B = wf(allcols, h, valid_all)
    idx = sorted(set(A) & set(B))
    pa = np.array([A[i][0] for i in idx]); pb = np.array([B[i][0] for i in idx])
    yr = np.array([A[i][1] for i in idx]); yb = (yr > 0).astype(int)
    n = len(idx)
    ll = lambda p: -(yb * np.log(np.clip(p, 1e-6, 1)) + (1 - yb) * np.log(np.clip(1 - p, 1e-6, 1)))
    dll = ll(pa) - ll(pb)   # >0 => B mejor
    lo, hi = block_ci(dll)
    row = {"n_oos": n, "p_up_real": float(yb.mean())}
    for nm, p in (("A", pa), ("B", pb)):
        sret = np.where(p > 0.5, 1.0, -1.0) * yr - COST
        l2, h2 = block_ci(sret)
        row[nm] = {"acc": float(np.mean((p > 0.5) == (yb == 1))), "auc": float(roc_auc_score(yb, p)), "net_pct": float(sret.mean() * 100), "ci": [l2 * 100, h2 * 100],
                   "binom_p": float(stats.binomtest(int(round(np.mean((p > 0.5) == (yb == 1)) * n)), n, max(yb.mean(), 1 - yb.mean())).pvalue)}
    row["dlogloss_mean"] = float(dll.mean()); row["dlogloss_ci"] = [lo, hi]
    row["naive_acc"] = float(max(yb.mean(), 1 - yb.mean()))
    # estrategia B con filtro de convicción
    thr = []
    for th in (0.0, 0.02, 0.04, 0.06):
        sel = np.abs(pb - 0.5) > th
        if sel.sum() >= 20:
            s = (np.where(pb > 0.5, 1.0, -1.0) * yr - COST)[sel]
            thr.append({"th": th, "n": int(sel.sum()), "acc": float(np.mean(((pb > 0.5) == (yb == 1))[sel])), "net_pct": float(s.mean() * 100)})
    row["thr"] = thr
    row["idx"] = [int(i) for i in idx]
    row["pB"] = [float(x) for x in pb]; row["yr"] = [float(x) for x in yr]
    res["horizons"][str(h)] = row
    print(f"h={h}d n={n} | A acc {row['A']['acc']:.3f} auc {row['A']['auc']:.3f} net {row['A']['net_pct']:+.3f}% | B acc {row['B']['acc']:.3f} auc {row['B']['auc']:.3f} net {row['B']['net_pct']:+.3f}% CI[{row['B']['ci'][0]:+.3f},{row['B']['ci'][1]:+.3f}] | naive {row['naive_acc']:.3f} | dLL {row['dlogloss_mean']:+.5f} CI[{lo:+.5f},{hi:+.5f}]", flush=True)

# ---------- IC univariante de las señales nuevas ----------
ic = []
nt = len(NEWC) * 2
for h in (1, 3):
    y = fwd(h)
    idxs = [i for i in range(0, len(df) - h, h) if valid_all[i] and np.isfinite(y[i])]
    idxs = np.array(idxs)
    for nm in NEWC:
        rho, p = stats.spearmanr(F[nm].values[idxs], y[idxs])
        ic.append({"h": h, "feature": nm, "label": LAB[nm], "ic": float(rho), "p": float(p), "p_bonf": float(min(1, p * nt)), "n": int(len(idxs))})
ic.sort(key=lambda r: r["p"])
res["ic"] = ic; res["n_tests"] = nt
for r_ in ic[:6]:
    print(f"IC {r_['h']}d {r_['feature']:<12} {r_['ic']:+.3f} p={r_['p']:.3f} bonf={r_['p_bonf']:.2f}")

# ---------- HMM: régimen walk-forward como filtro de confianza ----------
vol7 = lr1.rolling(7).std()
X = np.column_stack([lr1.values * 100, vol7.values * 100])
okx = np.isfinite(X).all(axis=1)
def fit_hmm(end):
    lo_ = max(1, end - 700)
    Xt = X[lo_:end][okx[lo_:end]]
    m = GaussianHMM(3, covariance_type="full", n_iter=150, random_state=7).fit(Xt)
    order = np.argsort(m.means_[:, 0])  # bajista, neutro, alcista por retorno medio
    return m, order
first = max(np.where(valid_all)[0][0] + 300, 400)
regs = {}
m, order, last_fit = None, None, -999
for i in range(first, len(df)):
    if i - last_fit >= 30:
        m, order = fit_hmm(i); last_fit = i
    w = X[max(1, i - 120):i + 1]
    w = w[np.isfinite(w).all(axis=1)]
    post = m.predict_proba(w)[-1]
    p3 = post[order]  # [bear, neutral, bull]
    regs[i] = p3
names = ["Bajista", "Neutro", "Alcista"]
hm = {}
for h in (1, 3):
    y = fwd(h)
    rows_ = []
    for k, nm in enumerate(names):
        sel = [i for i in range(first, len(df) - h, h) if np.isfinite(y[i]) and int(np.argmax(regs[i])) == k]
        if len(sel) >= 15:
            v = y[sel]
            t, p = stats.ttest_1samp(v, 0.0)
            rows_.append({"state": nm, "n": len(sel), "mean_pct": float(v.mean() * 100), "up_pct": float(np.mean(v > 0) * 100), "p": float(p), "p_bonf": float(min(1, p * 6))})
    hm[str(h)] = rows_
    for r_ in rows_:
        print(f"HMM {h}d {r_['state']:<8} n={r_['n']:>3} media {r_['mean_pct']:+.3f}% sube {r_['up_pct']:.1f}% p={r_['p']:.3f} bonf={r_['p_bonf']:.2f}")
res["hmm_oos"] = hm
# filtro de confianza: modelo B solo si el régimen coincide con la dirección
for h in (1, 3):
    row = res["horizons"][str(h)]
    idx, pb, yr = row["idx"], np.array(row["pB"]), np.array(row["yr"])
    agree, rets = [], []
    for j, i in enumerate(idx):
        if i not in regs:
            continue
        k = int(np.argmax(regs[i])); d = 1.0 if pb[j] > 0.5 else -1.0
        if (d > 0 and k == 2) or (d < 0 and k == 0):
            agree.append(j); rets.append(d * yr[j] - COST)
    if len(rets) >= 15:
        r_ = np.array(rets); lo_, hi_ = block_ci(r_)
        row["hmm_filter"] = {"n": len(r_), "net_pct": float(r_.mean() * 100), "ci": [lo_ * 100, hi_ * 100], "hit_pct": float(np.mean(r_ + COST > 0) * 100)}
        print(f"filtro HMM {h}d: n={len(r_)} neto {r_.mean()*100:+.3f}% CI[{lo_*100:+.3f},{hi_*100:+.3f}] acierto {np.mean(r_+COST>0)*100:.1f}%")
    else:
        row["hmm_filter"] = {"n": len(rets)}
        print(f"filtro HMM {h}d: n={len(rets)} (insuficiente)")

# ---------- lecturas actuales + P(up) de A y B con todos los datos ----------
last = len(df) - 1
now = {"as_of": int(df.ts.iloc[last] + DAY), "close": float(close[last])}
for k in ("fund", "dvol", "oi", "ls"):
    now[k] = float(df[k].iloc[last]) if np.isfinite(df[k].iloc[last]) else None
for k in ("fund_1d_z", "fund_7d_z", "dvol_z", "iv_rv", "oi_chg_1d", "oi_chg_7d", "oi_z", "ls_z"):
    v = F[k].iloc[last]; now[k] = float(v) if np.isfinite(v) else None
now["regime_probs"] = {names[k]: float(regs[last][k]) for k in range(3)}
now["regime"] = names[int(np.argmax(regs[last]))]
pu = {}
for h in (1, 3):
    y = fwd(h)
    for nm, cols in (("A", BASE), ("B", allcols)):
        ok = F[cols].notna().all(axis=1).values & np.isfinite(y) & valid_all
        ok[last - h + 1:] = False
        tr = np.where(ok)[0]
        Xtr, ytr = F[cols].values[tr], (y[tr] > 0).astype(int)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        mdl = LogisticRegression(C=0.05, max_iter=400).fit((Xtr - mu) / sd, ytr)
        pu[f"{nm}{h}"] = float(mdl.predict_proba(((F[cols].values[last] - mu) / sd)[None, :])[0, 1])
        pu[f"base{h}"] = float(ytr.mean())
now["p_up"] = pu
res["now"] = now
print("ahora:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in now.items() if k not in ("regime_probs", "p_up")})
print("régimen:", now["regime"], {k: round(v, 3) for k, v in now["regime_probs"].items()}, "| P(up):", {k: round(v, 3) for k, v in pu.items()})
# ---------- reglas de funding y OI definidas de antemano (8 pruebas) ----------
rules = [("Funding muy negativo (z < -2) -> largo", "fund_1d_z", lambda v: v < -2, 1.0),
         ("Funding muy positivo (z > +2) -> corto", "fund_1d_z", lambda v: v > 2, -1.0),
         ("Caída fuerte del OI (-5 % en 1 d) -> largo", "oi_chg_1d", lambda v: v < -5, 1.0),
         ("Subida fuerte del OI (+5 % en 1 d) -> corto", "oi_chg_1d", lambda v: v > 5, -1.0)]
ev2 = []
nt2 = len(rules) * 2
for nm, feat, cond, side in rules:
    for h in (1, 3):
        y = fwd(h); x = F[feat].values
        idx_, i = [], 0
        while i < len(df) - h:
            if valid_all[i] and np.isfinite(y[i]) and np.isfinite(x[i]) and cond(x[i]):
                idx_.append(i); i += h
            else:
                i += 1
        if len(idx_) < 10:
            ev2.append({"rule": nm, "h": h, "n": len(idx_)}); continue
        g = side * y[idx_]; net = g - COST
        t_, p_ = stats.ttest_1samp(net, 0.0)
        lo_, hi_ = block_ci(net, blk=2)
        ev2.append({"rule": nm, "h": h, "n": len(idx_), "hit_pct": float(np.mean(g > 0) * 100), "gross_pct": float(g.mean() * 100), "net_pct": float(net.mean() * 100),
                    "ci": [lo_ * 100, hi_ * 100], "p": float(p_), "p_bonf": float(min(1, p_ * nt2)),
                    "active_now": bool(cond(F[feat].iloc[last]))})
        print(f"REGLA {nm[:42]:<42} {h}d n={len(idx_):>3} acierto {np.mean(g>0)*100:.1f}% bruto {g.mean()*100:+.3f}% neto {net.mean()*100:+.3f}% p={p_:.3f} bonf={min(1,p_*nt2):.2f} activa={bool(cond(F[feat].iloc[last]))}")
res["rules"] = ev2
json.dump(res, open(f"{D}/bias_output.json", "w"))
