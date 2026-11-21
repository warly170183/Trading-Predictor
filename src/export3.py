import json, os, datetime as dt, math

from paths import ROOT as BASE
OUT = f"{BASE}/data/export3"
os.makedirs(OUT, exist_ok=True)
B = json.load(open(f"{BASE}/data/bias_output.json"))
G = json.load(open(f"{BASE}/data/gex_output.json"))
r = lambda x, n=2: None if x is None else round(float(x), n)
comma = lambda s: s.replace(".", ",")
fmt_ci = lambda lo, hi, d=2: f"{comma(f'{lo:+.{d}f}')} a {comma(f'{hi:+.{d}f}')}"

inp = ("OKX BTC-USDT-SWAP: velas 1D (1.199), open interest y ratio de cuentas largas/cortas diarios (1.057 días, desde dic-2023). "
       "Deribit: funding horario de BTC-PERPETUAL y DVOL desde dic-2023, y cadena de opciones BTC del 21-nov-2026. Costes: 0,10 % por operación.")
code = lambda name: {"label": name, "value": open(f"{BASE}/src/{name}").read(), "type": "code", "lang": "python"}


def meta(codes=(), note=None):
    m = [{"label": "Datos de entrada", "value": inp}]
    if note:
        m.append({"label": "Nota", "value": note})
    return m + [code(c) for c in codes]


def save(name, rows, m):
    json.dump({"rows": rows, "meta": m}, open(f"{OUT}/{name}.json", "w"), separators=(",", ":"))
    print(name, len(rows), os.path.getsize(f"{OUT}/{name}.json"))


H1, H3 = B["horizons"]["1"], B["horizons"]["3"]
now = B["now"]

# ---- veredicto ----
def fav(rule):
    return rule.get("p_bonf", 1) < 0.05 and rule.get("net_pct", -1) > 0
any_sig = (any(h["dlogloss_ci"][0] > 0 for h in (H1, H3)) or any(h["B"]["ci"][0] > 0 for h in (H1, H3))
           or any(x["p_bonf"] < 0.05 for x in B["ic"]) or any(x["p_bonf"] < 0.05 for hh in B["hmm_oos"].values() for x in hh)
           or any(fav(x) for x in B["rules"]))
rules_ok = [x for x in B["rules"] if x.get("n", 0) >= 10 and "gross_pct" in x]
n_neg = sum(1 for x in rules_ok if x["gross_pct"] < 0)
if not any_sig:
    verdict = ("El sesgo direccional no pasa la validación: con funding, OI, DVOL, ratio largos/cortos y régimen HMM, el modelo fuera de muestra no supera "
               "ni al modelo de solo precio ni al mejor ingenuo. Por eso el sesgo aplicado al cono y a las operaciones es 0.")
else:
    verdict = "Alguna señal supera la corrección por pruebas múltiples; revisa las tablas antes de aplicar un sesgo."
hyp = (f"En esta muestra, los extremos de funding y de OI tendieron a continuar y no a revertir: ir en contra de ellos perdió en bruto en {n_neg} de las {len(rules_ok)} reglas con eventos suficientes. "
       "Es una hipótesis para validar hacia adelante, no una señal.") if n_neg >= max(1, len(rules_ok) - 1) and rules_ok else "Ninguna regla de extremos mostró un patrón consistente."
save("bias_verdict", [{"verdict_text": verdict, "hypothesis_text": hyp, "bias_applied_pct": 0.0, "n_days": B["n_valid"], "n_tests": B["n_tests"] + 8 + 6}],
     meta(["bias.py", "gex.py", "fetch_derivs.py"], "Sesgo aplicado = 0 salvo que alguna señal supere la corrección de Bonferroni con resultado favorable tras costes."))

# ---- lecturas actuales ----
def zread(z, hi, lo):
    return "Sin dato" if z is None else (hi if z > 2 else lo if z < -2 else "Neutral")

rows = [
    {"key": "fund", "label": "Funding diario (Deribit)", "value": r(now["fund"], 1), "dec": 1, "unit": "pb/día", "z": r(now["fund_1d_z"], 1),
     "reading": zread(now["fund_1d_z"], "Extremo alto: pagan los largos", "Extremo bajo: pagan los cortos")},
    {"key": "oi1", "label": "Open interest, cambio en 1 d", "value": r(now["oi_chg_1d"], 1), "dec": 1, "unit": "%", "z": None,
     "reading": "Desapalancamiento fuerte" if now["oi_chg_1d"] < -5 else "Apalancamiento creciente" if now["oi_chg_1d"] > 5 else "Sin cambio relevante"},
    {"key": "oi7", "label": "Open interest, cambio en 7 d", "value": r(now["oi_chg_7d"], 1), "dec": 1, "unit": "%", "z": None,
     "reading": "Desapalancamiento" if now["oi_chg_7d"] < -10 else "Apalancamiento creciente" if now["oi_chg_7d"] > 10 else "Sin cambio relevante"},
    {"key": "dvol", "label": "DVOL (volatilidad implícita)", "value": r(now["dvol"], 1), "dec": 1, "unit": "puntos", "z": r(now["dvol_z"], 1),
     "reading": zread(now["dvol_z"], "Volatilidad implícita muy alta", "Volatilidad implícita muy baja")},
    {"key": "ivrv", "label": "Implícita frente a realizada (7 d)", "value": r((math.exp(now["iv_rv"]) - 1) * 100, 0), "dec": 0, "unit": "%", "z": None,
     "reading": "Las opciones descuentan más movimiento del realizado" if now["iv_rv"] > 0 else "Las opciones descuentan menos movimiento del realizado"},
    {"key": "ls", "label": "Ratio de cuentas largas/cortas (OKX)", "value": r(now["ls"], 2), "dec": 2, "unit": "largas por corta", "z": r(now["ls_z"], 1),
     "reading": zread(now["ls_z"], "Posicionamiento muy largo", "Posicionamiento muy corto")},
    {"key": "hmm", "label": f"Régimen HMM: {now['regime']}", "value": r(now["regime_probs"][now["regime"]] * 100, 0), "dec": 0, "unit": "% de probabilidad", "z": None,
     "reading": "Filtro de confianza, no señal"},
    {"key": "gex", "label": "GEX neto de opciones (Deribit)", "value": r(G["net_gex_m"], 0), "dec": 0, "unit": "M USD por 1 %", "z": None,
     "reading": "Gamma negativa: los dealers amplifican los movimientos" if G["net_gex_m"] < 0 else "Gamma positiva: los dealers amortiguan los movimientos"},
    {"key": "flip", "label": "Nivel de cambio de gamma", "value": r(G["flip"], 0), "dec": 0, "unit": "USD", "z": None,
     "reading": "Por encima del precio actual" if G["flip"] and G["flip"] > G["spot"] else "Por debajo del precio actual"},
    {"key": "cw", "label": "Strike con más gamma de calls", "value": r(G["call_wall"], 0), "dec": 0, "unit": "USD", "z": None, "reading": "Posible imán de precio, sin validar"},
    {"key": "pw", "label": "Strike con más gamma de puts", "value": r(G["put_wall"], 0), "dec": 0, "unit": "USD", "z": None, "reading": "Posible imán de precio, sin validar"},
]
save("bias_now", rows, meta(["bias.py", "gex.py"], "Lecturas en el último cierre diario (OKX y Deribit). El z compara con los 90 días previos. El GEX usa la convención estándar: calls suman gamma y puts restan; es una instantánea que no se puede validar con historia."))

# ---- modelos ----
mods = []
for hk, hz in (("1", "1d"), ("3", "3d")):
    h = B["horizons"][hk]
    mods.append({"horizon": hz, "n": h["n_oos"], "acc_a": r(h["A"]["acc"] * 100, 1), "acc_b": r(h["B"]["acc"] * 100, 1), "naive": r(h["naive_acc"] * 100, 1),
                 "auc_b": r(h["B"]["auc"], 3), "net_b": r(h["B"]["net_pct"], 3), "dll_k": r(h["dlogloss_mean"] * 1000, 2),
                 "dll_ci": fmt_ci(h["dlogloss_ci"][0] * 1000, h["dlogloss_ci"][1] * 1000)})
save("bias_models", mods, meta(["bias.py"], "Walk-forward expansivo con regresión logística L2. Solo precio = 8 señales de precio; Con derivados = precio + funding, DVOL, OI y ratio largos/cortos. Δ log-loss = log-loss de solo precio menos el de con derivados, por mil: positivo significa que los derivados mejoran. IC 95 % por bootstrap en bloques."))

# ---- IC ----
ic = [{"key": f"{x['feature']}_{x['h']}", "label": x["label"], "horizon": f"{x['h']}d", "ic": r(x["ic"], 3), "p": r(x["p"], 3), "p_bonf": r(x["p_bonf"], 3)} for x in B["ic"][:6]]
save("bias_ic", ic, meta(["bias.py"], f"Correlación de Spearman de cada señal con el retorno posterior, en muestras sin solape. Las 6 más fuertes de {B['n_tests']} pruebas; p corregido = Bonferroni."))

# ---- HMM ----
hm = []
for hk, hz in (("1", "1d"), ("3", "3d")):
    for x in B["hmm_oos"][hk]:
        hm.append({"key": f"{hz}_{x['state']}", "horizon": hz, "state": x["state"], "n": x["n"], "mean_pct": r(x["mean_pct"], 3), "up_pct": r(x["up_pct"], 1), "p_bonf": r(x["p_bonf"], 3)})
save("bias_hmm", hm, meta(["bias.py"], "Régimen HMM de 3 estados ajustado en walk-forward (reajuste cada 30 días). Retorno medio posterior por régimen, fuera de muestra y sin solape; p corregido sobre 6 pruebas."))

# ---- reglas ----
ru = []
for i, x in enumerate(B["rules"]):
    if x.get("n", 0) < 10 or "gross_pct" not in x:
        continue
    ru.append({"key": f"{i}", "rule": x["rule"].replace("->", "→"), "horizon": f"{x['h']}d", "n": x["n"], "hit_pct": r(x["hit_pct"], 1), "gross_pct": r(x["gross_pct"], 3),
               "net_pct": r(x["net_pct"], 3), "p_bonf": r(x["p_bonf"], 3), "active_now": "sí" if x["active_now"] else "no"})
save("bias_rules", ru, meta(["bias.py"], "Reglas de extremos definidas de antemano, con eventos sin solape. La regla de funding muy negativo no tuvo eventos suficientes (menos de 10). p corregido sobre 8 pruebas."))

# ---- GEX ----
save("gex_strikes", G["by_strike"], meta(["gex.py"], "GEX en millones de USD por cada 1 % de movimiento del spot, por strike (±15 % del precio). Instantánea de Deribit, 871 contratos."))
save("gex_summary", [{"spot": r(G["spot"], 0), "net_gex_m": G["net_gex_m"], "flip": r(G["flip"], 0), "call_wall": G["call_wall"], "put_wall": G["put_wall"], "put_call_oi": G["put_call_oi"]}],
     meta(["gex.py"]))
