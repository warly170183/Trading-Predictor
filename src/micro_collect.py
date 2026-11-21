"""Capturador de microestructura (OKX, datos públicos, sin clave). Una fila por intervalo con:
libro de órdenes (spread, desequilibrio, profundidad, microprecio), flujo agresivo (compras y ventas a mercado),
liquidaciones, base mark/índice, funding y open interest.

Uso:
  python src/micro_collect.py                       # sin fin, una fila por minuto
  python src/micro_collect.py --duration 120 --interval 20   # prueba corta
Salida: data/micro/<instrumento>_<fecha UTC>.jsonl (una línea por fila).
Debe correr en una máquina encendida de forma continua; un cron de GitHub Actions no sirve para esto (mínimo de 5 min y sin garantías)."""
import argparse, json, os, time
import requests
from paths import DATA

p = argparse.ArgumentParser()
p.add_argument("--inst", default="BTC-USDT-SWAP")
p.add_argument("--interval", type=int, default=60, help="segundos entre filas")
p.add_argument("--tick", type=int, default=10, help="segundos entre lecturas de trades")
p.add_argument("--duration", type=int, default=0, help="segundos totales (0 = sin fin)")
a = p.parse_args()

API = "https://www.okx.com/api/v5"
S = requests.Session()
uly = a.inst.rsplit("-", 1)[0]          # BTC-USDT
spot_inst = uly
OUT = os.path.join(DATA, "micro")
os.makedirs(OUT, exist_ok=True)


def get(path, **params):
    try:
        r = S.get(f"{API}/{path}", params=params, timeout=10).json()
        return r.get("data") if r.get("code") == "0" else None
    except Exception:
        return None


ct = get("public/instruments", instType="SWAP", instId=a.inst)
CT = float(ct[0]["ctVal"]) if ct else 0.01   # BTC (o activo) por contrato
print("contrato", CT, flush=True)

last_tid, seen_liq = 0, set()
acc = {"buy": 0.0, "sell": 0.0, "n": 0, "notional": 0.0, "missed": 0}
liq = {"long": 0.0, "short": 0.0}


def poll_trades():
    global last_tid
    d = get("market/trades", instId=a.inst, limit=500)
    if not d:
        return
    new = [x for x in d if int(x["tradeId"]) > last_tid]
    if last_tid and new and min(int(x["tradeId"]) for x in new) > last_tid + 1:
        acc["missed"] = 1                # el lote no llegó hasta el último visto: se perdieron operaciones
    for x in new:
        q = float(x["sz"]) * CT
        acc["buy" if x["side"] == "buy" else "sell"] += q
        acc["n"] += 1
        acc["notional"] += q * float(x["px"])
    if new:
        last_tid = max(int(x["tradeId"]) for x in new)


def poll_liq():
    d = get("public/liquidation-orders", instType="SWAP", uly=uly, state="filled", limit=100)
    if not d:
        return
    for blk in d:
        for x in blk.get("details", []):
            k = (x["ts"], x["bkPx"], x["sz"], x["posSide"])
            if k in seen_liq:
                continue
            seen_liq.add(k)
            if len(seen_liq) > 5000:
                seen_liq.clear()
            if x["posSide"] in liq:
                liq[x["posSide"]] += float(x["sz"]) * CT


def snapshot(ts):
    row = {"ts": ts}
    b = get("market/books", instId=a.inst, sz=20)
    if b:
        bids = [(float(x[0]), float(x[1]) * CT) for x in b[0]["bids"]]
        asks = [(float(x[0]), float(x[1]) * CT) for x in b[0]["asks"]]
        bb, ba = bids[0], asks[0]
        mid = (bb[0] + ba[0]) / 2
        row.update({
            "mid": mid, "spread_bps": (ba[0] - bb[0]) / mid * 1e4,
            "microprice_bps": ((bb[0] * ba[1] + ba[0] * bb[1]) / (bb[1] + ba[1]) - mid) / mid * 1e4,
        })
        for n in (5, 20):
            sb, sa = sum(x[1] for x in bids[:n]), sum(x[1] for x in asks[:n])
            row[f"imb{n}"] = (sb - sa) / (sb + sa) if sb + sa else None
        row["depth_bid_btc"] = sum(x[1] for x in bids if x[0] >= mid * (1 - 0.0005))   # a 5 pb del precio medio
        row["depth_ask_btc"] = sum(x[1] for x in asks if x[0] <= mid * (1 + 0.0005))
    mk = get("public/mark-price", instType="SWAP", instId=a.inst)
    ix = get("market/index-tickers", instId=spot_inst)
    if mk and ix:
        row["mark"], row["index"] = float(mk[0]["markPx"]), float(ix[0]["idxPx"])
        row["basis_bps"] = (row["mark"] / row["index"] - 1) * 1e4
    fr = get("public/funding-rate", instId=a.inst)
    if fr:
        row["funding"] = float(fr[0]["fundingRate"])
    oi = get("public/open-interest", instId=a.inst)
    if oi:
        row["oi_btc"] = float(oi[0]["oiCcy"])
    poll_trades()
    poll_liq()
    row.update({"tk_buy_btc": acc["buy"], "tk_sell_btc": acc["sell"], "n_trades": acc["n"],
                "vwap": acc["notional"] / (acc["buy"] + acc["sell"]) if acc["buy"] + acc["sell"] else None,
                "missed": acc["missed"], "liq_long_btc": liq["long"], "liq_short_btc": liq["short"]})
    acc.update({"buy": 0.0, "sell": 0.0, "n": 0, "notional": 0.0, "missed": 0})
    liq.update({"long": 0.0, "short": 0.0})
    return row


def write(row):
    day = time.strftime("%Y-%m-%d", time.gmtime(row["ts"] / 1000))
    with open(os.path.join(OUT, f"{a.inst}_{day}.jsonl"), "a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


t_end = time.time() + a.duration if a.duration else None
next_row = time.time() // a.interval * a.interval + a.interval
poll_trades(); poll_liq()
liq.update({"long": 0.0, "short": 0.0}); acc.update({"buy": 0.0, "sell": 0.0, "n": 0, "notional": 0.0, "missed": 0})
while t_end is None or time.time() < t_end:
    time.sleep(max(0.0, min(a.tick, next_row - time.time())))
    if time.time() >= next_row:
        row = snapshot(int(next_row * 1000))
        write(row)
        print(time.strftime("%H:%M:%S", time.gmtime()), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k in ("mid", "spread_bps", "imb5", "basis_bps", "tk_buy_btc", "tk_sell_btc", "n_trades", "missed", "liq_long_btc", "liq_short_btc")}, flush=True)
        next_row += a.interval
    else:
        poll_trades()
