"""DVOL horario de Deribit para una moneda (BTC o ETH) desde dic-2023. Uso: python src/fetch_dvol.py ETH"""
import json, sys, time, requests
from paths import DATA

cur = sys.argv[1].upper()
now = int(time.time() * 1000)
t, dv = 1701388800000, []
S = requests.Session()
while t < now:
    r = S.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
              params={"currency": cur, "start_timestamp": t, "end_timestamp": min(now, t + 40 * 86400000), "resolution": "3600"}, timeout=30).json()
    dv += r.get("result", {}).get("data", [])
    t += 40 * 86400000
    time.sleep(0.15)
dv = sorted({r[0]: r for r in dv}.values())
json.dump(dv, open(f"{DATA}/{cur.lower()}_dvol.json", "w"))
print(cur, "dvol", len(dv), dv[0][0], dv[-1][0])
