"""Descarga derivados con historia larga: OKX (OI, long/short, taker 1D) y Deribit (funding horario, DVOL, cadena de opciones)."""
import json, time, requests

from paths import DATA as OUT
now = int(time.time() * 1000)
S = requests.Session()


def okx_rubik(path, extra=None, n=1200):
    rows, end = [], None
    for _ in range(40):
        p = {"instId": "BTC-USDT-SWAP", "period": "1D", "limit": 100}
        if extra:
            p.update(extra)
        if end:
            p["end"] = end
        d = S.get(f"https://www.okx.com/api/v5/rubik/stat/{path}", params=p, timeout=20).json().get("data", [])
        if not d:
            break
        rows += d
        end = d[-1][0]
        time.sleep(0.15)
        if len(rows) >= n:
            break
    return sorted({int(r[0]): [float(x) for x in r] for r in rows}.values())


oi = okx_rubik("contracts/open-interest-history")           # ts, oi(contratos), oiCcy, oiUsd
ls = okx_rubik("contracts/long-short-account-ratio-contract")  # ts, ratio
tk = okx_rubik("taker-volume-contract", {"unit": "2"})      # ts, sellVol, buyVol
for n, d in (("oi", oi), ("ls", ls), ("taker", tk)):
    json.dump(d, open(f"{OUT}/okx_{n}_1d.json", "w"))
    print(n, len(d), d[0][0], d[-1][0])

# Deribit funding horario desde dic-2023
start = 1701388800000  # 2023-12-01
fund, t = [], start
while t < now:
    r = S.get("https://www.deribit.com/api/v2/public/get_funding_rate_history",
              params={"instrument_name": "BTC-PERPETUAL", "start_timestamp": t, "end_timestamp": min(now, t + 30 * 24 * 3600 * 1000)}, timeout=30).json()
    fund += [[x["timestamp"], x["interest_1h"], x["interest_8h"]] for x in r.get("result", [])]
    t += 30 * 24 * 3600 * 1000
    time.sleep(0.15)
fund = sorted({r[0]: r for r in fund}.values())
json.dump(fund, open(f"{OUT}/deribit_funding.json", "w"))
print("funding", len(fund), fund[0][0], fund[-1][0])

# DVOL horario
dv, t = [], start
while t < now:
    r = S.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
              params={"currency": "BTC", "start_timestamp": t, "end_timestamp": min(now, t + 40 * 24 * 3600 * 1000), "resolution": "3600"}, timeout=30).json()
    dv += r.get("result", {}).get("data", [])
    t += 40 * 24 * 3600 * 1000
    time.sleep(0.15)
dv = sorted({r[0]: r for r in dv}.values())
json.dump(dv, open(f"{OUT}/deribit_dvol.json", "w"))
print("dvol", len(dv), dv[0][0], dv[-1][0])

# Cadena de opciones (snapshot) para el GEX actual
r = S.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency", params={"currency": "BTC", "kind": "option"}, timeout=30).json()
chain = r.get("result", [])
idx = S.get("https://www.deribit.com/api/v2/public/get_index_price", params={"index_name": "btc_usd"}, timeout=20).json().get("result", {})
json.dump({"ts": now, "index": idx.get("index_price"), "chain": chain}, open(f"{OUT}/deribit_chain.json", "w"))
print("opciones", len(chain), "índice", idx.get("index_price"))
