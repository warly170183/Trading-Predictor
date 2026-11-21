import json, os, datetime as dt

from paths import ROOT as BASE
OUT = f"{BASE}/data/export2"
os.makedirs(OUT, exist_ok=True)
S = json.load(open(f"{BASE}/data/signals_output.json"))
T = json.load(open(f"{BASE}/data/trades_output.json"))
iso = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
r = lambda x, n=2: None if x is None else round(float(x), n)
comma = lambda s: s.replace(".", ",")

inp = "OKX BTC-USDT-SWAP, 29.999 velas 1H confirmadas (jun-2023 a nov-2026) vía API pública history-candles. Costes: 0,10 % por operación completa (2 x 0,05 % taker)."
code = lambda name: {"label": name, "value": open(f"{BASE}/src/{name}").read(), "type": "code", "lang": "python"}


def meta(codes=(), note=None):
    m = [{"label": "Datos de entrada", "value": inp}]
    if note:
        m.append({"label": "Nota", "value": note})
    m += [code(c) for c in codes]
    return m


def save(name, rows, m):
    json.dump({"rows": rows, "meta": m}, open(f"{OUT}/{name}.json", "w"), separators=(",", ":"))
    print(name, len(rows), os.path.getsize(f"{OUT}/{name}.json"))


LAB = {"mom_6": "Momentum 6 h", "mom_24": "Momentum 24 h", "mom_72": "Momentum 72 h", "mom_168": "Momentum 7 d",
       "ema_dev_50": "Distancia a EMA 50", "ema_dev_200": "Distancia a EMA 200", "rsi14": "RSI 14", "bb_z": "Bollinger z (20)",
       "vol_ratio": "Volatilidad 24 h / 7 d", "vol_pct": "Percentil de volatilidad", "range_pos": "Posición en el rango de 7 d",
       "vol_z": "Volumen anómalo (z)", "signed_vol": "Volumen con signo", "hour_sin": "Hora del día (seno)", "hour_cos": "Hora del día (coseno)"}
HZ = {6: "6h", 24: "24h", 72: "72h"}

# trade_top / trade_grid
top = []
for t in T["top"]:
    top.append({"key": t["key"], "dir_label": "Largo" if t["dir"] == "long" else "Corto", "sl_pct": r(t["sl_pct"], 1), "tp_pct": r(t["tp_pct"], 1),
                "p_tp_pct": r(t["p_tp_pct"], 1), "p_sl_pct": r(t["p_sl_pct"], 1), "p_to_pct": r(t["p_to_pct"], 1),
                "ev_pct": r(t["ev_pct"], 3), "ev_se_pct": r(t["ev_se_pct"], 3), "ev_r": r(t["ev_r"], 3),
                "be_pct": r(t["be_pct"], 1), "kelly_pct": r(t["kelly_pct"], 1)})
save("trade_top", top, meta(["trades.py"], "Las 3 mejores combinaciones de stop y objetivo por dirección según EV en R. Acierto necesario = porcentaje de aciertos con el que el EV se anula tras costes. Kelly = fracción del capital a arriesgar por operación (0 si el EV no es positivo)."))
grid = [{"dir": g["dir"], "sl_pct": r(g["sl_pct"], 1), "tp_pct": r(g["tp_pct"], 1), "ev_r": r(g["ev_r"], 3), "ev_pct": r(g["ev_pct"], 3),
         "ev_se_pct": r(g["ev_se_pct"], 3), "p_tp_pct": r(g["p_tp_pct"], 1), "sig_pos": bool(g["sig_pos"])} for g in T["grid"]]
save("trade_grid", grid, meta(["trades.py"], f"{T['n_sims']:,} trayectorias GJR-GARCH de 72 h, precio sin deriva (martingala), niveles vigilados al cierre horario y salida al precio real de ese cierre.".replace(",", ".")))

# signals
sig = []
for h in (6, 24, 72):
    s = S["horizons"][str(h)]
    sig.append({"horizon": HZ[h], "n": s["n_oos"], "acc_pct": r(s["acc"] * 100, 1), "naive_pct": r(max(s["p_up_real"], 1 - s["p_up_real"]) * 100, 1),
                "auc": r(s["auc"], 3), "net_pct": r(s["strat_mean_pct"], 3),
                "ic95": f"{comma(f'{s['strat_ci_pct'][0]:+.2f}')} a {comma(f'{s['strat_ci_pct'][1]:+.2f}')}",
                "p": r(s["binom_p_vs_best_naive"], 3)})
save("signals", sig, meta(["signals.py"], "Walk-forward expansivo con regresión logística L2 sobre 15 señales, etiquetas sin solape y purga. Rendimiento neto por operación en % tras costes; IC 95 % por bootstrap en bloques. p = prueba binomial del acierto frente al mejor ingenuo (siempre largo o siempre corto)."))

# equity
eq = []
for h in (6, 24, 72):
    s = S["horizons"][str(h)]
    for i, (t, a, b) in enumerate(zip(s["equity_t"], s["equity_strat"], s["equity_bh"])):
        eq.append({"horizon": HZ[h], "i": i, "t": iso(t), "strat": r(a, 4), "bh": r(b, 4)})
save("equity", eq, meta(["signals.py"], "Curva fuera de muestra con 1 unidad de capital y una operación por horizonte, tras costes. 'Siempre largo' paga el mismo coste en cada operación."))

# ic_top
ic = []
for x in S["ic"][:6]:
    ic.append({"key": f"{x['feature']}_{x['h']}", "feature_label": LAB[x["feature"]], "horizon": HZ[x["h"]], "ic": r(x["ic"], 3), "p": r(x["p"], 3), "p_bonf": r(x["p_bonf"], 3)})
save("ic_top", ic, meta(["signals.py"], f"Correlación de Spearman entre cada señal y el retorno posterior en muestras sin solape. Las 6 más fuertes de {S['n_tests']} pruebas. p corregido = Bonferroni."))

# events
ev = []
for e in sorted(T["events"], key=lambda e: e["p"]):
    nm = "Momentum 24 h" if e["feature"] == "mom_24" else "Bollinger z"
    ev.append({"key": f"{e['feature']}_{e['th']}_{e['h']}", "rule_short": f"{nm} > {comma(str(e['th']))}, ir en contra", "horizon": HZ[e["h"]], "n": e["n"],
               "hit_pct": r(e["hit_pct"], 1), "gross_pct": r(e["gross_pct"], 3), "net_pct": r(e["net_pct"], 3), "p_bonf": r(e["p_bonf"], 3),
               "active_now": "sí" if e["active_now"] else "no"})
save("events", ev, meta(["trades.py"], "Estudio de eventos sin solape sobre toda la muestra: 12 reglas contrarias definidas de antemano. p corregido = Bonferroni sobre las 12."))

# now_dir
now = [{"horizon": HZ[x["h"]], "p_up_pct": r(x["p_up_pct"], 1), "base_pct": r(x["base_pct"], 1)} for x in T["now"]]
save("now_dir", now, meta(["trades.py"], "Probabilidad de subir del modelo logístico ajustado con todos los datos, en el último cierre 1H, frente a la tasa histórica de subidas."))

# verdict
sig_ic = any(x["p_bonf"] < 0.05 and x["ic"] > -9 for x in S["ic"])
if T["n_sig"] == 0 and not sig_ic:
    txt = f"El modelo no encuentra ventaja: ninguna de las {T['n_combos']} combinaciones de stop y objetivo, largas o cortas, tiene un valor esperado positivo tras costes, y ninguna señal direccional supera al azar fuera de muestra."
else:
    txt = f"Hay {T['n_sig']} de {T['n_combos']} combinaciones con valor esperado significativamente positivo; revisa la tabla antes de actuar."
save("verdict", [{"n_combos": T["n_combos"], "n_sims": T["n_sims"], "cost_pct": r(T["cost_pct"], 2), "best_long_ev_pct": r(T["best_long_ev_pct"], 2),
                  "best_short_ev_pct": r(T["best_short_ev_pct"], 2), "n_sig": T["n_sig"], "verdict_text": txt}], meta(["trades.py", "signals.py"], "Resumen del veredicto: significativo = EV menos 3 errores estándar de Monte Carlo mayor que cero."))
print("sig_ic", sig_ic, "|", txt)
