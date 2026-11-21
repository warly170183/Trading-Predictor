"""Datasets de seguimiento para el dashboard (Data de Claude): estado del diario, cobertura, lecturas y resultados de volatilidad."""
import csv, json, os
import numpy as np
from paths import DATA, JOURNAL, ROOT, SRC

OUT = f"{DATA}/export4"
os.makedirs(OUT, exist_ok=True)
ASSETS = ("btc", "eth", "sol")
HZ = (24, 72)
code = lambda n: {"label": n, "value": open(f"{SRC}/{n}").read(), "type": "code", "lang": "python"}


def save(name, rows, note, codes=("journal.py",)):
    meta = [{"label": "Nota", "value": note}] + [code(c) for c in codes]
    json.dump({"rows": rows, "meta": meta}, open(f"{OUT}/{name}.json", "w"), separators=(",", ":"))
    print(name, len(rows))


def preds(a):
    p = os.path.join(JOURNAL, "predictions.jsonl" if a == "btc" else f"predictions_{a}.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def scores(a):
    p = os.path.join(JOURNAL, "scores.csv" if a == "btc" else f"scores_{a}.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


# --- estado actual por activo (último registro del diario)
now = []
for a in ASSETS:
    P = preds(a)
    if not P:
        continue
    r = P[-1]
    rd, fl = r["readings"], r["flags"]
    txt = []
    if fl["fund"] == "neg": txt.append("funding muy negativo")
    if fl["fund"] == "pos": txt.append("funding muy positivo")
    if fl["oi_flush"]: txt.append("OI en caída fuerte")
    now.append({"asset": a.upper(), "n_pred": len(P), "day": r["id"], "spot": r["spot"], "vol_pctile": r["vol_pctile"], "regime_vol": r["regime_vol"],
                "fund_z": rd.get("fund_1d_z"), "oi_chg_1d": rd.get("oi_chg_1d"), "dvol_z": rd.get("dvol_z"), "ls_z": rd.get("ls_z"),
                "flags": ", ".join(txt) if txt else "ninguna", "ens_ratio_72": r.get("ens_ratio", {}).get("72")})
save("track_now", now, "Último registro del diario por activo. z frente a los 90 días previos. El cociente del ensamble es sigma(ensamble)/sigma(GARCH) a 72 h.")

# --- cobertura
cov = []
for a in ASSETS:
    R = scores(a)
    for h in HZ:
        r = [x for x in R if int(x["h"]) == h]
        re_ = [x for x in r if x.get("in90_ens") not in ("", None)]
        cov.append({"key": f"{a}_{h}", "asset": a.upper(), "horizon": f"{h} h", "n": len(r),
                    "c50": float(np.mean([int(x["in50"]) for x in r]) * 100) if r else None,
                    "c90": float(np.mean([int(x["in90"]) for x in r]) * 100) if r else None,
                    "c90e": float(np.mean([int(x["in90_ens"]) for x in re_]) * 100) if re_ else None})
save("track_cov", cov, "Porcentaje de predicciones vencidas cuyo precio real cayó dentro de la banda. Nominal: 50 % y 90 %.")

# --- retornos posteriores por lectura
G = [("Funding muy negativo", lambda x: x["fund"] == "neg"), ("Funding muy positivo", lambda x: x["fund"] == "pos"),
     ("OI con caída fuerte en 1 d", lambda x: x["oi_flush"] == "True"), ("Sin lectura extrema", lambda x: x["fund"] == "none" and x["oi_flush"] != "True")]
fl = []
for a in ASSETS:
    R = scores(a)
    for name, f in G:
        for h in HZ:
            r = [x for x in R if int(x["h"]) == h and f(x)]
            fl.append({"key": f"{a}_{name}_{h}", "asset": a.upper(), "reading": name, "horizon": f"{h} h", "n": len(r),
                       "mean_pct": float(np.mean([float(x["ret_pct"]) for x in r])) if r else None,
                       "up_pct": float(np.mean([float(x["ret_pct"]) > 0 for x in r]) * 100) if r else None})
save("track_flags", fl, "Retorno medio posterior a cada tipo de lectura, con las predicciones vencidas. No se interpreta con menos de 30 eventos independientes.")

# --- volatilidad: ensamble e implícita
LAB = {"har": "HAR", "ens": "Ensamble GARCH+HAR", "iv": "Solo implícita (DVOL)", "ens_iv": "Ensamble con implícita"}
ve = []
for a in ASSETS:
    p = f"{DATA}/vol_ens_{a}.json"
    if not os.path.exists(p):
        continue
    J = json.load(open(p))
    for h, d in J["results"].items():
        for samp, v in d.items():
            for m, r in v["modelos"].items():
                if m not in LAB or (m in ("iv", "ens_iv") and samp != "con_implicita") or (m in ("har", "ens") and samp != "completa"):
                    continue
                ic = r["ic95_mejora_pct"]
                ve.append({"key": f"{a}_{h}_{m}", "asset": a.upper(), "horizon": f"{h} h", "model": LAB[m], "n": v["n"], "gain": r["mejora_vs_garch_pct"],
                           "lo": ic[0], "hi": ic[1], "cov90": r["cob90"]})
save("vol_ens", ve, "Mejora de la puntuación del cono (pinball loss) frente al GJR-GARCH, en walk-forward sobre velas 1H. IC 95 % por bootstrap en bloques. Positivo es mejor.", ("vol_ensemble.py", "volmodels.py"))
