"""Diario de predicciones: guarda cada día el cono y las lecturas de derivados, y puntúa las predicciones ya vencidas.

Uso:
  python src/journal.py run                        # BTC, ETH y SOL: descarga, calcula, guarda la predicción de hoy y puntúa
  python src/journal.py run --assets btc           # solo algunos activos
  python src/journal.py run --no-refresh           # reutiliza los datos de data/
  python src/journal.py score                      # solo puntúa y regenera el informe

Archivos (carpeta journal/):
  predictions.jsonl  una línea por día (se sobrescribe si se repite el mismo día); ETH y SOL: predictions_eth.jsonl, ...
  scores.csv         resultado de cada predicción vencida (24 h y 72 h); scores_eth.csv, ...
  REPORT.md          por activo: cobertura del cono y retornos posteriores por tipo de lectura

El sesgo aplicado al cono es 0. Las lecturas se guardan solo para validar hacia adelante
la hipótesis de continuación de funding y OI, sin sesgo retrospectivo."""
import csv, datetime as dt, json, os, subprocess, sys
import numpy as np
from paths import DATA, JOURNAL, SRC
from volmodels import ensemble_ratio

HZ = (24, 72)
QS = ("5", "10", "25", "50", "75", "90", "95")
FUND_Z, OI_FLUSH = 2.0, -5.0  # umbrales de las banderas
REPORT = os.path.join(JOURNAL, "REPORT.md")
ALL_ASSETS = ("btc", "eth", "sol")


def pred_path(asset):
    return os.path.join(JOURNAL, "predictions.jsonl" if asset == "btc" else f"predictions_{asset}.jsonl")


def score_path(asset):
    return os.path.join(JOURNAL, "scores.csv" if asset == "btc" else f"scores_{asset}.csv")
HOUR = 3600_000


def sh(*a):
    print("->", " ".join(a), flush=True)
    subprocess.run([sys.executable, *a], check=True, env={**os.environ, "TP_DATA": DATA})


def refresh(asset):
    s = lambda f: os.path.join(SRC, f)
    inst = f"{asset.upper()}-USDT-SWAP"
    sh(s("fetch.py"), inst, "1H", "5999", f"{DATA}/{asset}_1h.json")
    sh(s("fetch.py"), inst, "1D", "899", f"{DATA}/{asset}_1d.json")
    if asset == "btc":
        sh(s("fetch.py"), inst, "1D", "1199", f"{DATA}/btc_1d_long.json")
        sh(s("fetch_derivs.py"))
        sh(s("engine.py"), "--nowf")
        sh(s("gex.py"))
        sh(s("bias.py"))
    else:
        sh(s("readings.py"), asset)
        sh(s("engine.py"), "--nowf", "--asset", asset)


def build_record(asset="btc"):
    cone = json.load(open(f"{DATA}/cone_output.json" if asset == "btc" else f"{DATA}/cone_output_{asset}.json"))
    if asset == "btc":
        now = json.load(open(f"{DATA}/bias_output.json"))["now"]
        gex = json.load(open(f"{DATA}/gex_output.json"))
        hmm = {"state": now["regime"], "p": round(now["regime_probs"][now["regime"]], 3)}
    else:  # ETH y SOL: lecturas ligeras, sin HMM ni GEX
        now, gex, hmm = json.load(open(f"{DATA}/readings_{asset}.json")), {}, {"state": None, "p": None}
    b = cone["h1"]
    anchor = int(b["last_ts"])
    day = dt.datetime.fromtimestamp(anchor / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
    fz = now.get("fund_1d_z")
    flags = {
        "fund": ("neg" if fz < -FUND_Z else "pos" if fz > FUND_Z else "none") if fz is not None else "na",
        "oi_flush": bool(now["oi_chg_1d"] < OI_FLUSH) if now.get("oi_chg_1d") is not None else None,
    }
    h1 = np.array(json.load(open(f"{DATA}/{asset}_1h.json")))
    ratio = ensemble_ratio(np.diff(np.log(h1[:, 4])), HZ)
    spot = b["spot"]
    cone_ens = {str(h): {q: round(spot * float(np.exp(ratio[h] * np.log(b["cone"][q][h] / spot))), 2) for q in QS} for h in HZ}
    cone_opt = None
    if asset in ("btc", "eth"):  # cono implícito de opciones (neutral al riesgo), solo informativo
        try:
            import options_cone
            oc = options_cone.run(asset)
            cone_opt = {str(h): {q: round(spot * (1 + oc["cone"][str(h)]["q"][q] / 100), 2) for q in QS} for h in HZ}
        except Exception as e:
            print("cono de opciones no disponible:", e)
    return {
        "id": day,
        "asset": asset,
        "anchor_ts": anchor,
        "spot": b["spot"],
        "vol_ann_pct": round(b["vol_now_ann"] * 100, 1),
        "vol_pctile": round(b["vol_pctile"], 1),
        "regime_vol": cone["regime"]["current"],
        "hmm": hmm,
        "readings": {k: now.get(k) for k in ("fund", "fund_1d_z", "fund_7d_z", "dvol", "dvol_z", "iv_rv",
                                             "oi_chg_1d", "oi_chg_7d", "oi_z", "ls", "ls_z")},
        "gex": {k: gex.get(k) for k in ("net_gex_m", "flip", "call_wall", "put_wall", "put_call_oi")},
        "flags": flags,
        "bias_applied_pct": 0.0,
        "cone": {str(h): {q: round(b["cone"][q][h], 2) for q in QS} for h in HZ},
        "cone_ens": cone_ens,
        "cone_opt": cone_opt,
        "ens_ratio": {str(h): round(ratio[h], 4) for h in HZ},
        "p_up": {str(h): round(b["p_close_up"].get(f"up_{h}", float("nan")), 4) for h in HZ},
    }


def load_preds(asset="btc"):
    path = pred_path(asset)
    if not os.path.exists(path):
        return {}
    return {r["id"]: r for r in map(json.loads, filter(None, open(path).read().splitlines()))}


def save_preds(d, asset="btc"):
    os.makedirs(JOURNAL, exist_ok=True)
    with open(pred_path(asset), "w") as f:
        for k in sorted(d):
            f.write(json.dumps(d[k], separators=(",", ":")) + "\n")


def pit(x, q):
    lv = [int(k) / 100 for k in QS]
    pr = [q[k] for k in QS]
    return float(np.interp(x, pr, lv, left=0.0, right=1.0))  # aproximado: 0 o 1 fuera del 5-95


def score(preds, asset="btc"):
    path1h = f"{DATA}/{asset}_1h.json"
    if not os.path.exists(path1h):
        return []
    c = {int(r[0]): float(r[4]) for r in json.load(open(path1h))}
    rows = []
    for r in preds.values():
        for h in HZ:
            end = r["anchor_ts"] + h * HOUR
            if end not in c:
                continue
            q = r["cone"][str(h)]
            qe = r.get("cone_ens", {}).get(str(h))
            qo = (r.get("cone_opt") or {}).get(str(h))
            y = c[end]
            rows.append({
                "asset": asset, "id": r["id"], "h": h, "spot": r["spot"], "close": y,
                "ret_pct": round((y / r["spot"] - 1) * 100, 3),
                "in50": int(q["25"] <= y <= q["75"]), "in90": int(q["5"] <= y <= q["95"]),
                "pit": round(pit(y, q), 3),
                "in90_ens": int(qe["5"] <= y <= qe["95"]) if qe else "",
                "in90_opt": int(qo["5"] <= y <= qo["95"]) if qo else "", "fund": r["flags"]["fund"], "oi_flush": r["flags"]["oi_flush"],
                "hmm": r["hmm"]["state"],
            })
    rows.sort(key=lambda x: (x["id"], x["h"]))
    if rows:
        with open(score_path(asset), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return rows


def report_lines(asset, preds, rows):
    L = [f"## {asset.upper()}", "", f"Predicciones guardadas: **{len(preds)}**. Puntuadas (vencidas): **{len(rows)}**.", ""]
    L += ["### Cobertura del cono", "", "| Horizonte | n | Dentro del 50 % | Dentro del 90 % |", "|---|---|---|---|"]
    for h in HZ:
        r = [x for x in rows if x["h"] == h]
        n = len(r)
        L.append(f"| {h} h | {n} | " + (f"{100*np.mean([x['in50'] for x in r]):.0f} % | {100*np.mean([x['in90'] for x in r]):.0f} % |" if n else "— | — |"))
    L += ["", "Cono con ensamble GARCH+HAR (se guarda en paralelo para compararlo hacia adelante):", "", "| Horizonte | n | Dentro del 90 % (ensamble) |", "|---|---|---|"]
    for h in HZ:
        r = [x for x in rows if x["h"] == h and x["in90_ens"] != ""]
        L.append(f"| {h} h | {len(r)} | " + (f"{100*np.mean([x['in90_ens'] for x in r]):.0f} % |" if r else "— |"))
    if asset in ("btc", "eth"):
        L += ["", "Cono implícito de opciones (SVI + Breeden-Litzenberger, medida neutral al riesgo):", "", "| Horizonte | n | Dentro del 90 % (opciones) |", "|---|---|---|"]
        for h in HZ:
            r = [x for x in rows if x["h"] == h and x.get("in90_opt", "") != ""]
            L.append(f"| {h} h | {len(r)} | " + (f"{100*np.mean([x['in90_opt'] for x in r]):.0f} % |" if r else "— |"))
    L += ["", "Nominal: 50 % y 90 %. Con pocas predicciones la cobertura es ruido; las predicciones diarias consecutivas se solapan a 72 h.", ""]
    L += ["### Retorno posterior según la lectura", "", "| Lectura | Horizonte | n | Retorno medio | % de subidas |", "|---|---|---|---|---|"]
    groups = [("Funding muy negativo", lambda x: x["fund"] == "neg"), ("Funding muy positivo", lambda x: x["fund"] == "pos"),
              ("OI con caída fuerte en 1 d", lambda x: x["oi_flush"] is True), ("Sin lectura extrema", lambda x: x["fund"] == "none" and not x["oi_flush"])]
    for name, f in groups:
        for h in HZ:
            r = [x for x in rows if x["h"] == h and f(x)]
            L.append(f"| {name} | {h} h | {len(r)} | " + (f"{np.mean([x['ret_pct'] for x in r]):+.2f} % | {100*np.mean([x['ret_pct']>0 for x in r]):.0f} % |" if r else "— | — |"))
    L += ["", "Hipótesis a contrastar: tras extremos de funding y OI, el precio tiende a continuar y no a revertir.",
          "No se saca ninguna conclusión con menos de 30 eventos independientes por fila.", ""]
    return L


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    assets = sys.argv[sys.argv.index("--assets") + 1].lower().split(",") if "--assets" in sys.argv else list(ALL_ASSETS)
    lines = ["# Diario de predicciones", "", "Sesgo aplicado al cono: 0. Las lecturas son contexto y se validan aquí hacia adelante.", ""]
    for asset in assets:
        if cmd == "run" and "--no-refresh" not in sys.argv:
            refresh(asset)
        preds = load_preds(asset)
        if cmd == "run":
            rec = build_record(asset)
            preds[rec["id"]] = rec
            save_preds(preds, asset)
            print("guardada", asset, rec["id"], "spot", rec["spot"], "flags", rec["flags"])
        rows = score(preds, asset)
        lines += report_lines(asset, preds, rows)
        print(asset, "predicciones", len(preds), "puntuadas", len(rows))
    if cmd == "run" and os.path.isdir(os.path.join(DATA, "micro")):
        from micro_rollup import rollup
        print("microestructura:", rollup())
    os.makedirs(JOURNAL, exist_ok=True)
    open(REPORT, "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
