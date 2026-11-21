import json, os, importlib.util, datetime as dt
import numpy as np

from paths import ROOT as BASE
OUT = f"{BASE}/data/export"
os.makedirs(OUT, exist_ok=True)

spec = importlib.util.spec_from_file_location("engine", f"{BASE}/src/engine.py")
eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eng)

d = json.load(open(f"{BASE}/data/cone_output.json"))
h1, d1 = d["h1"], d["d1"]
iso = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
r1 = lambda x, n=1: round(float(x), n)

code_engine = open(f"{BASE}/src/engine.py").read()
code_fetch = open(f"{BASE}/src/fetch.py").read()
inp = "OKX BTC-USDT-SWAP, velas confirmadas 1H (5.999) y 1D (899) vía API pública history-candles, descargadas el 21-nov-2026."


def meta(with_code=False, extra=None):
    m = [{"label": "Datos de entrada", "value": inp}]
    if extra:
        m.append({"label": "Nota", "value": extra})
    if with_code:
        m.append({"label": "fetch.py", "value": code_fetch, "type": "code", "lang": "python"})
        m.append({"label": "engine.py", "value": code_engine, "type": "code", "lang": "python"})
    return m


def save(name, rows, m):
    json.dump({"rows": rows, "meta": m}, open(f"{OUT}/{name}.json", "w"), separators=(",", ":"))
    print(name, len(rows), os.path.getsize(f"{OUT}/{name}.json"), "bytes")


spot_t = h1["last_ts"] + h1["step_ms"]  # cierre de la última vela 1H
spot = h1["spot"]
scale_d = spot / d1["spot"]  # el cono diario se ancla al último precio 1H

# summary
g = h1["garch"]
reg = d["regime"]
save("summary", [{
    "spot": r1(spot, 1), "spot_time": iso(spot_t),
    "vol_now_ann_pct": r1(h1["vol_now_ann"] * 100, 1), "vol_pctile": r1(h1["vol_pctile"], 0),
    "regime": reg["current"], "p_regime_pct": r1(reg["probs"][reg["current"]] * 100, 0),
    "persistence": round(g["persistence"], 4), "nu": r1(g["nu"], 2),
    "omega": g["omega"], "alpha": round(g["alpha"], 4), "gamma": round(g["gamma"], 4), "beta": round(g["beta"], 4),
}], meta(True, "Fila única con el estado del modelo en el último cierre 1H."))

# candles
rows = []
for tf, b in (("1H", h1), ("1D", d1)):
    for c in b["candles"]:
        rows.append({"tf": tf, "t": iso(c[0]), "o": c[1], "h": c[2], "l": c[3], "c": c[4]})
save("candles", rows, meta(False, "1H: últimas 168 velas. 1D: últimas 90 velas."))

# cone + paths
rows, prow = [], []
for tf, b, sc in (("1H", h1, 1.0), ("1D", d1, scale_d)):
    for k in range(b["H"] + 1):
        t = spot_t + k * b["step_ms"]
        row = {"tf": tf, "step": k, "t": iso(t)}
        for q in (5, 10, 25, 50, 75, 90, 95):
            row[f"p{q}"] = r1(b["cone"][str(q)][k] * sc, 1)
        rows.append(row)
        pr = {"tf": tf, "step": k}
        for j, path in enumerate(b["sample_paths"][:25]):
            pr[f"p{j}"] = r1(path[k] * sc, 1)
        prow.append(pr)
note = "5.000 trayectorias simuladas por horizonte (GJR-GARCH t + bootstrap de residuos). El cono diario se reescala para anclarse al último precio 1H."
save("cone", rows, meta(True, note))
save("paths", prow, meta(False, "25 trayectorias de muestra de las 5.000 simuladas."))

# ranges
rows = []
for tf, b, hs in (("1H", h1, [("24h", 24), ("72h", 72)]), ("1D", d1, [("7d", 7), ("30d", 30)])):
    for lab, h in hs:
        mv = b["move"][str(h)]
        rows.append({"tf": tf, "horizon": lab, "p5_pct": r1(mv["p5"] * 100, 1), "p50_pct": r1(mv["p50"] * 100, 1),
                     "p95_pct": r1(mv["p95"] * 100, 1), "p_up_pct": r1(b["p_close_up"][f"up_{h}"] * 100, 1)})
save("ranges", rows, meta(False, "Movimiento relativo al último precio, percentiles 5/50/95 del cono; P(sube) = probabilidad de cerrar por encima del precio actual."))

# touch
rows = []
for tf, b in (("1H", h1), ("1D", d1)):
    for t in b["touch"]:
        rows.append({"tf": tf, "level": f"±{round(t['up']*100)} %", "up_first_pct": r1(t["up_first"] * 100, 1),
                     "dn_first_pct": r1(t["dn_first"] * 100, 1), "none_pct": r1(t["none"] * 100, 1)})
save("touch", rows, meta(False, "Probabilidad de tocar primero el nivel superior, el inferior o ninguno dentro del horizonte del cono."))

# calibration
rows = []
wf = d["walk_forward"]
for h, lab in (("1", "1h"), ("6", "6h"), ("24", "24h"), ("72", "72h")):
    s = wf[h]
    rows.append({
        "horizon": lab, "n": s["n"],
        "g50": r1(s["garch"]["50"] * 100, 1), "g80": r1(s["garch"]["80"] * 100, 1), "g90": r1(s["garch"]["90"] * 100, 1),
        "b50": r1(s["base"]["50"] * 100, 1), "b80": r1(s["base"]["80"] * 100, 1), "b90": r1(s["base"]["90"] * 100, 1),
        "w90_g": round(s["winkler90_garch"] * 100, 2), "w90_b": round(s["winkler90_base"] * 100, 2),
        "improve_pct": r1((s["winkler90_base"] - s["winkler90_garch"]) / s["winkler90_base"] * 100, 1),
    })
save("calibration", rows, meta(True, "Walk-forward: ventana de 2.500 h, reajuste del modelo cada 12 h, 2.000 simulaciones por origen. Base = bootstrap de retornos con volatilidad constante. Winkler en puntos porcentuales de retorno (menor es mejor). Los orígenes se solapan, así que n efectivo es menor que n."))

# vol history (último 720 h), mismo ajuste que el cono
h = eng.load(f"{BASE}/data/btc_1h.json")
res = eng.fit_garch(np.diff(np.log(h[:, 4])) * 100)
cv = res.conditional_volatility / 100 * np.sqrt(24 * 365) * 100
ts = h[1:, 0] + 3600_000
rows = [{"t": iso(t), "vol_ann_pct": r1(v, 1)} for t, v in zip(ts[-720:], cv[-720:])]
save("vol_hist", rows, meta(False, "Volatilidad condicional GJR-GARCH anualizada, últimas 720 horas."))

print("spot", spot, "scale_d", round(scale_d, 4), "pers", g["persistence"])
