from paths import DATA
import json, sys, time, warnings
import numpy as np
from arch import arch_model

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
QS = [5, 10, 25, 50, 75, 90, 95]


def load(path):
    a = np.array(json.load(open(path)))
    return a  # ts, o, h, l, c, v


def fit_garch(r_pct):
    am = arch_model(r_pct, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t")
    return am.fit(disp="off", show_warning=False)


def simulate(res, H, N):
    f = res.forecast(horizon=H, method="bootstrap", simulations=N, reindex=False)
    sims = f.simulations.values[-1]  # (N, H) returns in %
    return np.cumsum(sims, axis=1) / 100.0  # cumulative log returns


def baseline_cum(r, H, N):
    d = rng.choice(r, size=(N, H), replace=True)
    return np.cumsum(d, axis=1)


def bands(cum, h):
    x = cum[:, h - 1]
    return {q: np.percentile(x, q) for q in QS}


def winkler(lo, hi, y, alpha):
    w = hi - lo
    if y < lo:
        w += 2 / alpha * (lo - y)
    elif y > hi:
        w += 2 / alpha * (y - hi)
    return w


def walk_forward(close, window=2500, step=12, H=72, N=2000, hz=(1, 6, 24, 72)):
    lr = np.diff(np.log(close))
    out = {h: {"g": [], "b": [], "pit_g": [], "wg": [], "wb": []} for h in hz}
    origins = list(range(window, len(lr) - H, step))
    t0 = time.time()
    for k, i in enumerate(origins):
        win = lr[i - window:i]
        try:
            res = fit_garch(win * 100)
            cg = simulate(res, H, N)
        except Exception:
            continue
        cb = baseline_cum(win, H, N)
        for h in hz:
            y = np.sum(lr[i:i + h])
            xg, xb = cg[:, h - 1], cb[:, h - 1]
            out[h]["pit_g"].append(float(np.mean(xg < y)))
            rec = {}
            for lvl in (50, 80, 90):
                a = (100 - lvl) / 2
                lg, ug = np.percentile(xg, [a, 100 - a])
                lb, ub = np.percentile(xb, [a, 100 - a])
                rec[lvl] = (lg <= y <= ug, lb <= y <= ub)
                if lvl == 90:
                    out[h]["wg"].append(winkler(lg, ug, y, 0.10))
                    out[h]["wb"].append(winkler(lb, ub, y, 0.10))
            out[h]["g"].append({l: int(v[0]) for l, v in rec.items()})
            out[h]["b"].append({l: int(v[1]) for l, v in rec.items()})
        if k % 40 == 0:
            print(f"  wf {k}/{len(origins)}  {time.time()-t0:.0f}s", flush=True)
    summ = {}
    for h in hz:
        n = len(out[h]["g"])
        summ[str(h)] = {
            "n": n,
            "garch": {str(l): float(np.mean([r[l] for r in out[h]["g"]])) for l in (50, 80, 90)},
            "base": {str(l): float(np.mean([r[l] for r in out[h]["b"]])) for l in (50, 80, 90)},
            "winkler90_garch": float(np.mean(out[h]["wg"])),
            "winkler90_base": float(np.mean(out[h]["wb"])),
            "pit_hist": np.histogram(out[h]["pit_g"], bins=10, range=(0, 1))[0].tolist(),
        }
    return summ


def first_passage(paths_price, spot, up, dn):
    """P(touch up before dn) over paths (N,H)."""
    hi, lo = spot * (1 + up), spot * (1 - dn)
    hit_up = paths_price >= hi
    hit_dn = paths_price <= lo
    N, H = paths_price.shape
    iu = np.where(hit_up.any(1), hit_up.argmax(1), H + 1)
    idn = np.where(hit_dn.any(1), hit_dn.argmax(1), H + 1)
    return {
        "up_first": float(np.mean(iu < idn)),
        "dn_first": float(np.mean(idn < iu)),
        "none": float(np.mean((iu > H) & (idn > H))),
    }


def regime(close):
    from hmmlearn.hmm import GaussianHMM
    lr = np.diff(np.log(close))
    n = 24
    vol = np.array([lr[max(0, i - n + 1):i + 1].std() for i in range(len(lr))])
    X = np.column_stack([lr * 100, vol * 100])[n:]
    m = GaussianHMM(3, covariance_type="full", n_iter=200, random_state=7).fit(X)
    post = m.predict_proba(X)[-1]
    order = np.argsort(m.means_[:, 1])  # by vol
    names = ["Calma", "Normal", "Turbulento"]
    probs = {names[j]: float(post[order[j]]) for j in range(3)}
    cur = max(probs, key=probs.get)
    return cur, probs


def cone_block(ts, ohlc, H, step_ms, N, label, tail):
    close = ohlc[:, 4]
    lr = np.diff(np.log(close))
    res = fit_garch(lr * 100)
    cum = simulate(res, H, N)
    spot = close[-1]
    paths = spot * np.exp(cum)
    pct = {q: np.percentile(paths, q, axis=0) for q in QS}
    ann = {"1H": 24 * 365, "1D": 365}[label]
    cond_vol = float(res.conditional_volatility[-1]) / 100
    hist_vol = res.conditional_volatility / 100
    p = res.params
    block = {
        "label": label,
        "spot": float(spot),
        "last_ts": int(ts[-1]),
        "step_ms": step_ms,
        "H": H,
        "candles": [[int(a[0]), a[1], a[2], a[3], a[4]] for a in ohlc[-tail:]],
        "cone": {str(q): [float(spot)] + [float(v) for v in pct[q]] for q in QS},
        "sample_paths": [[float(spot)] + [float(v) for v in paths[i]] for i in range(0, 30)],
        "garch": {
            "omega": float(p["omega"]), "alpha": float(p["alpha[1]"]),
            "gamma": float(p["gamma[1]"]), "beta": float(p["beta[1]"]),
            "nu": float(p["nu"]),
            "persistence": float(p["alpha[1]"] + p["gamma[1]"] / 2 + p["beta[1]"]),
        },
        "vol_now_ann": cond_vol * np.sqrt(ann),
        "vol_pctile": float(np.mean(hist_vol <= hist_vol[-1]) * 100),
        "vol_hist": [float(v * np.sqrt(ann)) for v in hist_vol[-tail:]],
        "vol_hist_ts": [int(t) for t in ts[1:][-tail:]],
    }
    # probabilities
    probs = {}
    for h in ([1, 6, 24, 72] if label == "1H" else [1, 7, 14, 30]):
        if h <= H:
            probs[f"up_{h}"] = float(np.mean(cum[:, h - 1] > 0))
    block["p_close_up"] = probs
    rng_move = {}
    for h in ([24, 72] if label == "1H" else [7, 30]):
        x = cum[:, h - 1]
        rng_move[str(h)] = {
            "p5": float(np.exp(np.percentile(x, 5)) - 1), "p50": float(np.exp(np.percentile(x, 50)) - 1),
            "p95": float(np.exp(np.percentile(x, 95)) - 1),
        }
    block["move"] = rng_move
    levels = [(0.02, 0.02), (0.03, 0.03), (0.05, 0.05)] if label == "1H" else [(0.05, 0.05), (0.10, 0.10), (0.15, 0.15)]
    block["touch"] = [
        {"up": u, "dn": d, **first_passage(paths, spot, u, d)} for u, d in levels
    ]
    return block, res


if __name__ == "__main__":
    ASSET = sys.argv[sys.argv.index("--asset") + 1].lower() if "--asset" in sys.argv else "btc"
    h1 = load(f"{DATA}/{ASSET}_1h.json")
    d1 = load(f"{DATA}/{ASSET}_1d.json")
    t0 = time.time()
    b1, _ = cone_block(h1[:, 0], h1, 72, 3600_000, 5000, "1H", 168)
    bd, _ = cone_block(d1[:, 0], d1, 30, 86400_000, 5000, "1D", 90)
    print("cones", f"{time.time()-t0:.0f}s", flush=True)
    cur, probs = regime(h1[:, 4])
    print("regime", cur, probs, flush=True)
    wf = walk_forward(h1[:, 4]) if "--nowf" not in sys.argv else {}
    out = {
        "generated": int(time.time() * 1000),
        "inst": f"{ASSET.upper()}-USDT-SWAP",
        "n_hours": int(len(h1)),
        "n_days": int(len(d1)),
        "regime": {"current": cur, "probs": probs},
        "h1": b1, "d1": bd, "walk_forward": wf,
        "wf_cfg": {"window": 2500, "step": 12, "sims": 2000},
    }
    json.dump(out, open(f"{DATA}/cone_output.json" if ASSET == "btc" else f"{DATA}/cone_output_{ASSET}.json", "w"))
    print("done", f"{time.time()-t0:.0f}s")
