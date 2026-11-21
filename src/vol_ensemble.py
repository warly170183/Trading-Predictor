"""¿Se puede mejorar el cono con volatilidad implícita (DVOL) y un ensamble de modelos?
Walk-forward sobre velas 1H. Cada modelo da una desviación típica para el retorno a h horas; los cuantiles salen
de los residuos estandarizados pasados (calibración sin mirar al futuro). Se puntúa con pinball loss (5-95 %),
cobertura y Winkler, y se compara con el GJR-GARCH por bootstrap en bloques.
Uso: python src/vol_ensemble.py btc|eth|sol"""
import json, sys, time, warnings
import numpy as np
from paths import DATA
from engine import fit_garch

warnings.filterwarnings("ignore")
asset = sys.argv[1].lower()
HZ = (24, 72)
QS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
WIN, STEP, I0 = 2500, 24, 4000
rng = np.random.default_rng(11)

a = np.array(json.load(open(f"{DATA}/{asset}_1h_long.json")))
ts, close = a[:, 0].astype("int64"), a[:, 4]
lr = np.diff(np.log(close))
n = len(lr)
r2 = lr ** 2
cs = np.concatenate([[0], np.cumsum(r2)])
mean_r2 = lambda lo, hi: (cs[hi] - cs[lo]) / (hi - lo)  # media de r² en lr[lo:hi]

dv_ts = dv_c = None
try:
    d = np.array(json.load(open(f"{DATA}/{asset}_dvol.json")))
    dv_ts, dv_c = d[:, 0].astype("int64"), d[:, 4]
except FileNotFoundError:
    pass


def implied_hourly_var(i):
    """Varianza horaria implícita en el origen i, con la barra DVOL anterior a la hora de información (conservador)."""
    if dv_ts is None:
        return np.nan
    k = np.searchsorted(dv_ts, ts[i] - 3600000, "right") - 1
    return (dv_c[k] / 100) ** 2 / 8760 if k >= 0 and ts[i] - 3600000 - dv_ts[k] < 7200000 else np.nan


# --- HAR: características en cada instante j
J = np.arange(720, n)
FEAT = np.column_stack([np.ones(len(J)), np.log(mean_r2(J - 24, J) + 1e-12), np.log(mean_r2(J - 168, J) + 1e-12), np.log(mean_r2(J - 720, J) + 1e-12)])
IMP = np.array([implied_hourly_var(j) for j in J])


def target(j, h):
    return mean_r2(j, j + h)


origins = [i for i in range(I0, n - max(HZ), STEP)]
M = ["const", "ewma", "garch", "har", "iv", "ens", "ens_iv"]
sig = {h: {m: np.full(len(origins), np.nan) for m in M} for h in HZ}
ret = {h: np.array([lr[i:i + h].sum() for i in origins]) for h in HZ}
t0 = time.time()
lam = 0.97
for k, i in enumerate(origins):
    w = lr[i - WIN:i]
    v_const = w.var()
    e = r2[i - 2000:i]
    wts = lam ** np.arange(len(e) - 1, -1, -1)
    v_ewma = (wts * e).sum() / wts.sum()
    try:
        res = fit_garch(w * 100)
        fv = res.forecast(horizon=max(HZ), reindex=False).variance.values[-1] / 1e4
        cum_g = np.cumsum(fv)
    except Exception:
        cum_g = None
    for h in HZ:
        S = sig[h]
        S["const"][k] = np.sqrt(h * v_const)
        S["ewma"][k] = np.sqrt(h * v_ewma)
        if cum_g is not None:
            S["garch"][k] = np.sqrt(cum_g[h - 1])
        # HAR (entrenado solo con j <= i-h)
        jj = np.arange(720, i - h + 1, 6)
        if len(jj) > 500:
            X = FEAT[jj - 720]
            y = np.log(np.array([target(j, h) for j in jj]) + 1e-12)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            smear = np.mean(np.exp(y - X @ beta))
            x_now = np.array([1, np.log(mean_r2(i - 24, i) + 1e-12), np.log(mean_r2(i - 168, i) + 1e-12), np.log(mean_r2(i - 720, i) + 1e-12)])
            S["har"][k] = np.sqrt(h * np.exp(x_now @ beta) * smear)
        # Implícita escalada (prima de riesgo de varianza estimada en las últimas ~2000 h)
        iv_now = implied_hourly_var(i)
        if np.isfinite(iv_now):
            jj2 = jj[jj >= i - 2000 - h]
            im = IMP[jj2 - 720]
            ok = np.isfinite(im)
            if ok.sum() > 150:
                tgt = np.array([target(j, h) for j in jj2])[ok]
                scale = tgt.mean() / im[ok].mean()
                S["iv"][k] = np.sqrt(h * iv_now * scale)
    if k % 100 == 0:
        print(asset, k, len(origins), f"{time.time()-t0:.0f}s", flush=True)

for h in HZ:
    S = sig[h]
    S["ens"] = np.exp((np.log(S["garch"]) + np.log(S["har"])) / 2)
    S["ens_iv"] = np.exp((np.log(S["garch"]) + np.log(S["har"]) + np.log(S["iv"])) / 3)


def pinball(y, f):
    d = y[:, None] - f
    return np.maximum(QS * d, (QS - 1) * d).mean(axis=1) * 100  # en puntos porcentuales


def blockci(x, blk, B=4000):
    n_ = len(x)
    nb = int(np.ceil(n_ / blk))
    idx = rng.integers(0, n_ - blk + 1, size=(B, nb))
    m = np.array([x[(i[:, None] + np.arange(blk)).ravel()[:n_]].mean() for i in idx])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


out = {"asset": asset, "n_origins": len(origins), "has_iv": dv_ts is not None, "results": {}}
oi = np.array(origins)
for h in HZ:
    S = sig[h]
    y = ret[h]
    L = {m: np.full(len(origins), np.nan) for m in M}
    cov = {m: np.full((len(origins), 2), np.nan) for m in M}
    wk = {m: np.full(len(origins), np.nan) for m in M}
    for k in range(len(origins)):
        cal = np.where(oi + h <= oi[k])[0][-300:]
        if len(cal) < 80:
            continue
        for m in M:
            if not np.isfinite(S[m][k]):
                continue
            c_ok = cal[np.isfinite(S[m][cal])]
            if len(c_ok) < 80:
                continue
            z = y[c_ok] / S[m][c_ok]
            f = S[m][k] * np.quantile(z, QS)
            L[m][k] = pinball(np.array([y[k]]), f[None, :])[0]
            cov[m][k] = [f[2] <= y[k] <= f[4], f[0] <= y[k] <= f[6]]
            width = f[6] - f[0]
            wk[m][k] = (width + (20 * (f[0] - y[k]) if y[k] < f[0] else 20 * (y[k] - f[6]) if y[k] > f[6] else 0)) * 100
    res_h = {}
    for samp, models in (("completa", ["const", "ewma", "garch", "har", "ens"]), ("con_implicita", M)):
        ok = np.all([np.isfinite(L[m]) for m in models], axis=0)
        if ok.sum() < 60:
            continue
        blk = max(3, 2 * int(np.ceil(h / STEP)))
        rows = {}
        for m in models:
            d = (L["garch"][ok] - L[m][ok])
            rows[m] = {"pinball": float(L[m][ok].mean()), "mejora_vs_garch_pct": float(100 * d.mean() / L["garch"][ok].mean()),
                       "ic95_mejora_pct": [100 * v / float(L["garch"][ok].mean()) for v in blockci(d, blk)] if m != "garch" else None,
                       "cob50": float(cov[m][ok][:, 0].mean() * 100), "cob90": float(cov[m][ok][:, 1].mean() * 100), "winkler90": float(wk[m][ok].mean())}
        res_h[samp] = {"n": int(ok.sum()), "modelos": rows}
    out["results"][str(h)] = res_h
    for samp, v in res_h.items():
        print(f"\n{asset.upper()} {h} h · muestra {samp} (n={v['n']})")
        for m, r in v["modelos"].items():
            ic = r["ic95_mejora_pct"]
            print(f"  {m:8s} pinball {r['pinball']:.4f} · vs GARCH {r['mejora_vs_garch_pct']:+.2f} %" + (f" [{ic[0]:+.1f},{ic[1]:+.1f}]" if ic else "") + f" · cob 50/90 {r['cob50']:.0f}/{r['cob90']:.0f} · Winkler {r['winkler90']:.2f}")
json.dump(out, open(f"{DATA}/vol_ens_{asset}.json", "w"))
