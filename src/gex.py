"""GEX actual de BTC a partir de la cadena de opciones de Deribit (instantánea, sin historia: no validable).
Gamma de Black-Scholes por contrato, convención estándar: opciones call suman gamma y puts restan (supuesto sobre la posición de los dealers)."""
import json, re, datetime as dt
import numpy as np
from scipy.stats import norm

from paths import DATA as D
ch = json.load(open(f"{D}/deribit_chain.json"))
now = ch["ts"] / 1000
spot = float(ch["index"])
MON = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
rows = []
for o in ch["chain"]:
    m = re.match(r"BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])", o["instrument_name"])
    if not m or not o.get("mark_iv") or not o.get("open_interest"):
        continue
    d, mon, yy, K, cp = m.groups()
    exp = dt.datetime(2000 + int(yy), MON[mon], int(d), 8, 0, tzinfo=dt.timezone.utc).timestamp()
    T = (exp - now) / (365 * 86400)
    if T <= 0.0005:
        continue
    rows.append((float(K), T, float(o["mark_iv"]) / 100, float(o["open_interest"]), 1.0 if cp == "C" else -1.0, float(o["underlying_price"]) / spot, exp))
A = np.array(rows)
K, T, iv, oi, sg, fr, exp = A.T
print("contratos usados:", len(A), "| índice", spot)


def gex_at(S, mask=None):
    F = fr * S
    d1 = (np.log(F / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (F * iv * np.sqrt(T))
    g = sg * oi * gamma * S ** 2 * 0.01   # USD por cada 1 % de movimiento del spot
    return g if mask is None else g * mask


net = gex_at(spot).sum()
grid = np.linspace(spot * 0.75, spot * 1.25, 201)
prof = np.array([gex_at(s).sum() for s in grid])
flip = None
for i in range(len(grid) - 1):
    if prof[i] * prof[i + 1] < 0:
        flip = float(grid[i] - prof[i] * (grid[i + 1] - grid[i]) / (prof[i + 1] - prof[i]))
        break

# por strike (±15 %)
lo, hi = spot * 0.85, spot * 1.15
g = gex_at(spot)
strikes = sorted(set(K[(K >= lo) & (K <= hi)]))
by = []
for k in strikes:
    mk = K == k
    cg = float(g[mk & (sg > 0)].sum() / 1e6); pg = float(g[mk & (sg < 0)].sum() / 1e6)
    by.append({"strike": float(k), "call_gex_m": round(cg, 2), "put_gex_m": round(pg, 2), "net_gex_m": round(cg + pg, 2), "oi_btc": round(float(oi[mk].sum()), 1)})
call_wall = max(by, key=lambda r: r["call_gex_m"])["strike"]
put_wall = min(by, key=lambda r: r["put_gex_m"])["strike"]
# vencimientos próximos (<= 7 días) y total
near = (exp - now) <= 7 * 86400
res = {
    "spot": spot, "as_of": int(ch["ts"]), "n_contracts": int(len(A)),
    "net_gex_m": round(float(net / 1e6), 2), "net_gex_7d_m": round(float(g[near].sum() / 1e6), 2),
    "flip": flip, "call_wall": call_wall, "put_wall": put_wall,
    "regime": "gamma positiva (los dealers amortiguan los movimientos)" if net > 0 else "gamma negativa (los dealers amplifican los movimientos)",
    "by_strike": by,
    "profile": [{"spot": round(float(s), 1), "gex_m": round(float(p / 1e6), 2)} for s, p in zip(grid[::4], prof[::4])],
    "put_call_oi": round(float(oi[sg < 0].sum() / oi[sg > 0].sum()), 3),
}
json.dump(res, open(f"{D}/gex_output.json", "w"))
print({k: v for k, v in res.items() if k not in ("by_strike", "profile")})
print("strikes:", len(by), "| mayor |GEX| por strike:", sorted(by, key=lambda r: -abs(r["net_gex_m"]))[:4])
