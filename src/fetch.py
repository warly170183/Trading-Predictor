import sys, time, json, requests
inst, bar, n_target, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
url = "https://www.okx.com/api/v5/market/history-candles"
rows, after = [], None
while len(rows) < n_target:
    p = {"instId": inst, "bar": bar, "limit": 100}
    if after: p["after"] = after
    r = requests.get(url, params=p, timeout=20).json()
    d = r.get("data", [])
    if not d: break
    rows += d
    after = d[-1][0]
    time.sleep(0.12)
seen, clean = set(), []
for x in rows:
    if x[0] in seen or x[8] != "1": continue
    seen.add(x[0]); clean.append([int(x[0])] + [float(v) for v in x[1:6]])
clean.sort()
json.dump(clean, open(out, "w"))
print(out, len(clean), clean[0][0], clean[-1][0])
