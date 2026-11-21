"""¿Mejora el conformal adaptativo la cobertura del intervalo al 90 % frente a la calibración estática por cuantiles?
Usa data/vol_sig_{asset}.npz (de vol_ensemble.py). Compara, para ens y garch, el intervalo estático (cuantiles 5/95 de
residuos pasados, como en el cono) con ACI. Kupiec/Christoffersen sobre los fallos y Winkler 90.
Uso: python src/conformal_eval.py btc|eth|sol"""
import json, sys
import numpy as np
from paths import DATA
from conformal import aci_intervals
import evaluation as ev

STEP = 24

def run(asset):
    z = np.load(f"{DATA}/vol_sig_{asset}.npz")
    out = {}
    for h in (24, 72):
        y = z[f"ret{h}"]; lag = int(np.ceil(h / STEP))
        for m in ("garch", "ens"):
            sg = z[f"sig{h}_{m}"]
            ok = np.isfinite(sg)
            # estático: simétrico por cuantil 90 % de |z| pasado conocido (misma ventana que ACI con gamma=0)
            for tag, gam in (("estatico", 0.0), ("aci", 0.02)):
                q, hit, _ = aci_intervals(np.where(ok, y, np.nan), np.where(ok, sg, np.nan), lag, gamma=gam)
                v = np.isfinite(hit)
                hv = hit[v]
                w = 2 * q[v]
                miss_pen = 20 * np.maximum(np.abs(y[v]) - q[v], 0) * 2 / 2
                wink = float(np.mean(w + 20 * np.maximum(np.abs(y[v]) - q[v], 0)) * 100)
                out[f"{h}h_{m}_{tag}"] = {"n": int(v.sum()), "cob90": float(hv.mean() * 100), "winkler90": wink,
                                          "kupiec_p": ev.kupiec_pof(1 - hv, 0.10)[1], "christ_ind_p": ev.christoffersen_ind(1 - hv)[1],
                                          "ancho_medio_pct": float(w.mean() * 100)}
    return out

if __name__ == "__main__":
    a = sys.argv[1]
    r = run(a)
    json.dump(r, open(f"{DATA}/conformal_{a}.json", "w"))
    for k, v in r.items():
        print(f"{a.upper()} {k:<22} n={v['n']} cob90 {v['cob90']:.1f}% ancho {v['ancho_medio_pct']:.2f}% Winkler {v['winkler90']:.2f} Kupiec p={v['kupiec_p']:.2f} Christ p={v['christ_ind_p']:.2f}")
