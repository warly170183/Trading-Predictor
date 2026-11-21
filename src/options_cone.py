"""Cono implícito de opciones (medida neutral al riesgo): cadena Deribit → SVI por vencimiento → precios Black-76 →
densidad de Breeden-Litzenberger → cuantiles del retorno a h horas.

Método: por vencimiento se ajusta SVI crudo a la varianza total w(k)=a+b(ρ(k−m)+√((k−m)²+σ²)), k=ln(K/F), con las
opciones OTM y mark_iv. Para un horizonte T se interpola la varianza total entre vencimientos que lo rodean (linealmente
en T, a k fijo); si T es menor que el primero se escala por T/T1. CDF: P(S_T<K)=1+dC/dK (tipo=0). Ojo: es la medida
neutral al riesgo; la prima de varianza hace que suela ser más ancha que la realizada; se mide en el diario con cobertura.

Uso: python src/options_cone.py btc|eth"""
import datetime as dt, json, re, sys
import numpy as np
import requests
from scipy.optimize import least_squares
from scipy.stats import norm

MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
QS = (5, 10, 25, 50, 75, 90, 95)


def fetch_chain(cur):
    r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                     params={"currency": cur.upper(), "kind": "option"}, timeout=30).json()["result"]
    now = dt.datetime.now(dt.timezone.utc)
    return r, now


def parse(name):
    m = re.match(r"\w+-(\d+)([A-Z]{3})(\d+)-(\d+)-([CP])", name)
    d, mo, y, k, cp = m.groups()
    exp = dt.datetime(2000 + int(y), MON[mo], int(d), 8, tzinfo=dt.timezone.utc)
    return exp, float(k), cp


def svi(p, k):
    a, b, rho, m, s = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def fit_svi(k, w):
    best = None
    for s0 in (0.05, 0.2):
        x0 = [max(w.min() * 0.8, 1e-6), 0.1, -0.2, 0.0, s0]
        lo = [-1, 1e-4, -0.999, -2, 1e-3]; hi = [5, 5, 0.999, 2, 3]
        try:
            r = least_squares(lambda p: svi(p, k) - w, x0, bounds=(lo, hi))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x, float(np.sqrt(2 * best.cost / len(k)))


def smiles(chain, now):
    """{T_años: (F, parámetros SVI, rmse, n)} por vencimiento con ≥6 strikes OTM válidos."""
    by = {}
    for o in chain:
        if not o.get("mark_iv") or o["mark_iv"] <= 0 or not o.get("open_interest"):
            continue
        exp, K, cp = parse(o["instrument_name"])
        T = (exp - now).total_seconds() / (365 * 86400)
        if T <= 0.5 / 365:  # descarta los que vencen en <12 h: ruido
            continue
        by.setdefault(exp, []).append((K, cp, o["mark_iv"] / 100, o["underlying_price"], T))
    out = {}
    for exp, L in sorted(by.items()):
        F = float(np.median([x[3] for x in L])); T = L[0][4]
        pts = [(np.log(K / F), iv ** 2 * T) for K, cp, iv, _, _ in L if (K < F and cp == "P") or (K >= F and cp == "C")]
        if len(pts) < 6:
            continue
        pts.sort(); k = np.array([p[0] for p in pts]); w = np.array([p[1] for p in pts])
        p, rmse = fit_svi(k, w)
        out[T] = {"F": F, "p": p, "rmse_vol_pts": rmse / np.sqrt(T) * 100 if False else float(np.sqrt(np.mean((np.sqrt(np.maximum(svi(p, k), 1e-12) / T) - np.sqrt(w / T)) ** 2)) * 100), "n": len(k), "exp": exp.isoformat(), "kmin": float(k.min()), "kmax": float(k.max())}
    return out


def w_at(sm, T, k):
    Ts = sorted(sm)
    lo = [t for t in Ts if t <= T]; hi = [t for t in Ts if t > T]
    if not hi:
        return svi(sm[Ts[-1]]["p"], k) * T / Ts[-1]
    if not lo:
        return svi(sm[hi[0]]["p"], k) * T / hi[0]
    t0, t1 = lo[-1], hi[0]
    wl, wh = svi(sm[t0]["p"], k), svi(sm[t1]["p"], k)
    return wl + (T - t0) / (t1 - t0) * (wh - wl)


def cone(sm, hours):
    """Cuantiles (en % de retorno simple) del precio a `hours` horas, medida neutral al riesgo."""
    T = hours / 8760
    k = np.linspace(-1.2, 1.2, 4801)  # log-moneyness respecto al forward
    w = np.maximum(w_at(sm, T, k), 1e-10)
    sd = np.sqrt(w)
    # call normalizado C(k)=N(d1)-e^k N(d2), d1=(-k+w/2)/sd
    d1 = (-k + w / 2) / sd; d2 = d1 - sd
    C = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    dC_dK = np.gradient(C, np.exp(k))  # K normalizado por F
    cdf = np.clip(1 + dC_dK, 0, 1)
    cdf = np.maximum.accumulate(cdf)
    ok = (cdf > 0.002) & (cdf < 0.998)
    out = {}
    for q in QS:
        out[str(q)] = float((np.interp(q / 100, cdf, np.exp(k)) - 1) * 100)  # F_T/F - 1 en %
    mono = bool(np.all(np.diff(cdf) >= -1e-9))
    return out, {"mass_in_grid": float(cdf[-1] - cdf[0]), "arbitrage_free_density": mono, "iv_atm_pct": float(np.sqrt(w[np.argmin(abs(k))] / T) * 100)}


def run(cur, hours=(24, 72)):
    chain, now = fetch_chain(cur)
    sm = smiles(chain, now)
    res = {"asset": cur, "ts": now.isoformat(), "expiries": len(sm), "fit": {f"{T*365:.1f}d": {"rmse_vol_pts": v["rmse_vol_pts"], "n": v["n"]} for T, v in sm.items()}, "cone": {}}
    for h in hours:
        q, diag = cone(sm, h)
        res["cone"][str(h)] = {"q": q, **diag}
    return res


if __name__ == "__main__":
    r = run(sys.argv[1])
    print(json.dumps(r, indent=1))
