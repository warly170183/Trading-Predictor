"""Capa de operaciones: (1) estudio de eventos contrarian, (2) P(up) actual del modelo walk-forward,
(3) malla SL/TP long y short sobre las trayectorias GARCH con corrección de puente browniano,
(4) deriva necesaria para que el EV sea positivo."""
import json, importlib.util, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
rng = np.random.default_rng(11)
COST = 0.0010
from paths import ROOT as BASE

# ---------- features (misma definición que signals.py) ----------
a = np.array(json.load(open(f"{BASE}/data/btc_1h_long.json")))
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
logc = np.log(c.values)


def fwd(h):
    return np.r_[logc[h:] - logc[:-h], [np.nan] * h]


def block_ci(x, B=4000, blk=3):
    x = np.asarray(x, float); n = len(x)
    nb = int(np.ceil(n / blk)); out = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n, nb)
        ii = (st[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n
        out[b] = x[ii].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------- (1) estudio de eventos contrarian ----------
events = []
last = F.iloc[-1]
tests = [(f, th, h) for f in ("mom_24", "bb_z") for th in (1.5, 2.0, 2.5) for h in (6, 24)]
nt = len(tests)
for fnm, th, h in tests:
    yr = fwd(h)
    x = F[fnm].values
    idx, i = [], 400
    while i < len(df) - h:
        if np.isfinite(x[i]) and abs(x[i]) > th and np.isfinite(yr[i]):
            idx.append(i); i += h  # sin solape
        else:
            i += 1
    idx = np.array(idx)
    if len(idx) < 20:
        continue
    fade = -np.sign(x[idx]) * yr[idx] - COST
    gross = fade + COST
    t, p = stats.ttest_1samp(fade, 0.0)
    lo, hi = block_ci(fade)
    events.append({
        "rule": f"{'Momentum 24 h' if fnm=='mom_24' else 'Bollinger z'} extremo (|z| > {th}) → ir contra",
        "feature": fnm, "th": th, "h": h, "n": int(len(idx)),
        "hit_pct": float(np.mean(gross > 0) * 100), "gross_pct": float(gross.mean() * 100),
        "net_pct": float(fade.mean() * 100), "ci_lo": lo * 100, "ci_hi": hi * 100,
        "p": float(p), "p_bonf": float(min(1, p * nt)),
        "active_now": bool(abs(last[fnm]) > th), "dir_now": ("long" if last[fnm] < -th else "short" if last[fnm] > th else None),
    })
    print(f"EV {fnm:7} th={th} h={h:>2} n={len(idx):>4} hit={events[-1]['hit_pct']:.1f}% gross {gross.mean()*100:+.3f}% net {fade.mean()*100:+.3f}% p={p:.3f} bonf={min(1,p*nt):.2f} activo={events[-1]['active_now']}")

# ---------- (2) P(up) actual del modelo walk-forward ----------
now = []
for h in (6, 24, 72):
    yr = fwd(h)
    ok = F.notna().all(axis=1).values & np.isfinite(yr)
    ok[-h:] = False
    tr = np.where(ok)[0][::max(1, h // 3)]
    Xtr, ytr = F.values[tr], (yr[tr] > 0).astype(int)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = LogisticRegression(C=0.02, max_iter=300).fit((Xtr - mu) / sd, ytr)
    xnow = F.values[-1]
    pup = float(m.predict_proba(((xnow - mu) / sd)[None, :])[0, 1])
    now.append({"h": h, "p_up_pct": pup * 100, "base_pct": float(ytr.mean() * 100)})
    print(f"AHORA h={h}: P(up)={pup:.3f} (tasa base {ytr.mean():.3f})")
feat_now = {k: float(v) for k, v in last.items() if k in ("mom_6", "mom_24", "mom_72", "bb_z", "rsi14", "vol_ratio", "vol_pct", "ema_dev_50", "range_pos")}
print("features ahora:", {k: round(v, 2) for k, v in feat_now.items()})

# ---------- (3) malla SL/TP long y short ----------
spec = importlib.util.spec_from_file_location("engine", f"{BASE}/src/engine.py")
eng = importlib.util.module_from_spec(spec); spec.loader.exec_module(eng)
h1 = eng.load(f"{BASE}/data/btc_1h.json")
close = h1[:, 4]
res = eng.fit_garch(np.diff(np.log(close)) * 100)
H, N = 72, 40000
f = res.forecast(horizon=H, method="bootstrap", simulations=N, reindex=False)
rets = f.simulations.values[-1] / 100.0          # (N,H) retornos log por paso
var = f.simulations.residual_variances[-1] / 1e4  # (N,H) varianza condicional por paso
cum = np.cumsum(rets, axis=1)
# Escenario neutro: el precio es una martingala en cada paso (sin sesgo direccional heredado de la muestra)
adj = -np.log(np.exp(cum).mean(0))
cum = cum + adj[None, :]
x1 = cum
x0 = np.concatenate([np.zeros((N, 1)), cum[:, :-1]], axis=1)
spot = float(close[-1])
KK = np.arange(1, H + 1)[None, :]
ROW = np.arange(N)


def first(hit):
    return np.where(hit.any(1), hit.argmax(1), H + 5)


def outcome(direction, sl, tp, drift=0.0):
    """Niveles vigilados al cierre horario; salida al precio real de ese cierre (incluye saltos), sin recortar a SL/TP.
    Con precio martingala esto es insesgado: el EV antes de costes es ~0 por teorema de parada opcional."""
    long_ = direction == "long"
    pr = np.exp(x1 + drift * KK) - 1
    if long_:
        wh, lh, sg = pr >= tp, pr <= -sl, 1.0
    else:
        wh, lh, sg = pr <= -tp, pr >= sl, -1.0
    iw, il = first(wh), first(lh)
    ex = np.minimum(iw, il)
    none = ex >= H
    idx = np.where(none, H - 1, ex)
    win, loss = (iw < il) & ~none, (il < iw) & ~none
    out = sg * pr[ROW, idx] - COST
    return win.mean(), loss.mean(), none.mean(), out


SLs, TPs = [0.01, 0.015, 0.02, 0.03, 0.04], [0.01, 0.02, 0.03, 0.04, 0.06, 0.08]
grid = []
for d in ("long", "short"):
    for sl in SLs:
        for tp in TPs:
            pw, pl, pn, out = outcome(d, sl, tp)
            X = out / sl
            ev = float(out.mean())
            kel = float(max(0.0, X.mean() / np.mean(X ** 2))) if X.mean() > 0 else 0.0
            grid.append({"dir": d, "sl_pct": sl * 100, "tp_pct": tp * 100, "p_tp_pct": pw * 100, "p_sl_pct": pl * 100, "p_to_pct": pn * 100,
                         "ev_pct": ev * 100, "ev_se_pct": float(out.std() / np.sqrt(N) * 100),
                         "ev_r": float(X.mean()), "kelly_pct": kel * 100,
                         "be_pct": (sl + COST) / (sl + tp) * 100, "rr": tp / sl,
                         "sig_pos": bool(ev * 100 - 3 * float(out.std() / np.sqrt(N) * 100) > 0)})
gdf = pd.DataFrame(grid)
print(gdf.groupby("dir")[["ev_pct", "ev_r"]].agg(["max", "mean"]).round(3))


# deriva necesaria (por 24 h, %) para EV = 0 en la mejor combinación de cada dirección
def req_drift(d, sl, tp):
    """Deriva del precio (%/24 h) con la que el EV de la operación se anula (EV crece con la deriva en largos y decrece en cortos)."""
    s_ = 1 if d == "long" else -1
    lo, hi = -0.002, 0.002
    for _ in range(24):
        mid = (lo + hi) / 2
        if s_ * outcome(d, sl, tp, drift=mid)[3].mean() < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 * 24 * 100


top = []
for d in ("long", "short"):
    sub = gdf[gdf.dir == d].sort_values("ev_r", ascending=False).head(3)
    for j, (_, r) in enumerate(sub.iterrows(), 1):
        row = {k: (float(v) if not isinstance(v, str) else v) for k, v in r.items()}
        row["key"] = f"{'L' if d=='long' else 'S'}{j}"
        row["req_drift_24h_pct"] = float(req_drift(d, r.sl_pct / 100, r.tp_pct / 100)) if j == 1 else None
        top.append(row)
        print(row["key"], {k: round(v, 3) for k, v in row.items() if isinstance(v, float) and v is not None})

# ventaja bruta medida (del walk-forward) para comparar
sig = json.load(open(f"{BASE}/data/signals_output.json"))
meas = {}
for h in ("6", "24", "72"):
    s = sig["horizons"][h]
    meas[h] = {"gross_pct": s["strat_mean_pct"] + COST * 100, "lo": s["strat_ci_pct"][0] + COST * 100, "hi": s["strat_ci_pct"][1] + COST * 100}
print("ventaja bruta medida:", {k: {a: round(b, 3) for a, b in v.items()} for k, v in meas.items()})

n_sig = int(sum(1 for g in grid if g["sig_pos"]))
best_l = max((g for g in grid if g["dir"] == "long"), key=lambda g: g["ev_r"])
best_s = max((g for g in grid if g["dir"] == "short"), key=lambda g: g["ev_r"])
print("combinaciones con EV significativamente > 0:", n_sig, "de", len(grid))
json.dump({"n_sig": n_sig, "n_combos": len(grid), "best_long_ev_pct": best_l["ev_pct"], "best_short_ev_pct": best_s["ev_pct"],
           "events": events, "now": now, "feat_now": feat_now, "grid": grid, "top": top, "measured_gross": meas,
           "spot": spot, "cost_pct": COST * 100, "n_sims": N, "H": H}, open(f"{BASE}/data/trades_output.json", "w"))
print("ok")
