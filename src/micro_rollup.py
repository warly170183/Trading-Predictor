"""Convierte los jsonl de data/micro/ en CSV comprimidos pequeños para versionar en journal/micro/.
Uso: python src/micro_rollup.py   (idempotente: reescribe los archivos de cada día)"""
import csv, glob, gzip, json, os
from paths import DATA, JOURNAL


def rollup():
    out = os.path.join(JOURNAL, "micro")
    done = []
    for f in sorted(glob.glob(os.path.join(DATA, "micro", "*.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if not rows:
            continue
        cols = list(dict.fromkeys(k for r in rows for k in r))
        os.makedirs(out, exist_ok=True)
        dst = os.path.join(out, os.path.basename(f).replace(".jsonl", ".csv.gz"))
        with gzip.open(dst, "wt", newline="") as g:
            w = csv.DictWriter(g, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        done.append((os.path.basename(dst), len(rows)))
    return done


if __name__ == "__main__":
    for n, k in rollup():
        print(n, k, "filas")
