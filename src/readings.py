"""Lecturas de derivados para ETH o SOL (equivalente ligero al 'now' de bias.py): funding (Deribit), DVOL (solo ETH),
open interest y ratio largos/cortos (OKX, 1D). z-score frente a los 90 días previos. Uso: python src/readings.py eth|sol
Escribe data/readings_<activo>.json con el último día completo."""
import json, sys, time
import numpy as np
import requests
from paths import DATA

asset = sys.argv[1].lower()
DAY = 86400000
now = int(time.time() * 1000)
OFF = 16 * 3600000                       # las velas diarias de OKX abren a las 00:00 UTC+8 (16:00 UTC); se usan los mismos días en todo
today = (now - OFF) // DAY * DAY + OFF   # inicio del día OKX en curso
S = requests.Session()
CUR = asset.upper()
DERIBIT_PERP = {"ETH": "ETH-PERPETUAL", "SOL": "SOL_USDC-PERPETUAL"}[CUR]


def zlast(s, n=90):
    s = np.asarray(s, float)
    prev = s[-n - 1:-1]
    prev = prev[np.isfinite(prev)]
    return float((s[-1] - prev.mean()) / prev.std()) if len(prev) > 30 and prev.std() > 0 and np.isfinite(s[-1]) else None


def rubik(path, extra=None):
    rows, end = [], None
    for _ in range(3):
        p = {"instId": f"{CUR}-USDT-SWAP", "period": "1D", "limit": 100, **(extra or {})}
        if end:
            p["end"] = end
        d = S.get(f"https://www.okx.com/api/v5/rubik/stat/{path}", params=p, timeout=20).json().get("data", [])
        if not d:
            break
        rows += d; end = d[-1][0]; time.sleep(0.15)
    return {int(r[0]): [float(x) for x in r] for r in rows}


out = {"asset": asset, "as_of": today - DAY}
days = np.arange(today - 130 * DAY, today, DAY)

# funding diario (puntos básicos por día)
try:
    fr, t = [], today - 131 * DAY
    while t < now:
        r = S.get("https://www.deribit.com/api/v2/public/get_funding_rate_history",
                  params={"instrument_name": DERIBIT_PERP, "start_timestamp": t, "end_timestamp": min(now, t + 30 * DAY)}, timeout=30).json()
        fr += [(x["timestamp"], x["interest_1h"]) for x in r.get("result", [])]
        t += 30 * DAY; time.sleep(0.15)
    fr = sorted(set(fr))
    ts_f = np.array([x[0] for x in fr]); v_f = np.array([x[1] for x in fr])
    fund = np.array([v_f[(ts_f >= d) & (ts_f < d + DAY)].sum() * 1e4 if ((ts_f >= d) & (ts_f < d + DAY)).sum() >= 20 else np.nan for d in days])
    out["fund"], out["fund_1d_z"] = (float(fund[-1]) if np.isfinite(fund[-1]) else None), zlast(fund)
except Exception as e:
    out["fund"] = out["fund_1d_z"] = None

# DVOL (ETH): cierre de la barra horaria que termina con cada día
if CUR == "ETH":
    try:
        r = S.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                  params={"currency": "ETH", "start_timestamp": today - 131 * DAY, "end_timestamp": now, "resolution": "3600"}, timeout=30).json()
        dv = {int(x[0]): x[4] for x in r["result"]["data"]}
        s = np.array([dv.get(d + DAY - 3600000, np.nan) for d in days])
        out["dvol"], out["dvol_z"] = (float(s[-1]) if np.isfinite(s[-1]) else None), zlast(s)
    except Exception:
        out["dvol"] = out["dvol_z"] = None
else:
    out["dvol"] = out["dvol_z"] = None

# OI y ratio largos/cortos
oi = rubik("contracts/open-interest-history")
ls = rubik("contracts/long-short-account-ratio-contract")
so = np.array([oi[d][2] if d in oi else np.nan for d in days])
sl = np.array([ls[d][1] if d in ls else np.nan for d in days])
out["oi_chg_1d"] = float((so[-1] / so[-2] - 1) * 100) if np.isfinite(so[-1]) and np.isfinite(so[-2]) else None
out["oi_chg_7d"] = float((so[-1] / so[-8] - 1) * 100) if np.isfinite(so[-1]) and np.isfinite(so[-8]) else None
out["ls"], out["ls_z"] = (float(sl[-1]) if np.isfinite(sl[-1]) else None), zlast(sl)
json.dump(out, open(f"{DATA}/readings_{asset}.json", "w"))
print(out)
