import sys, time, json, requests
inst, out = sys.argv[1], sys.argv[2]
url = "https://www.okx.com/api/v5/public/funding-rate-history"
rows, after = [], None
for _ in range(400):
    p = {"instId": inst, "limit": 100}
    if after: p["after"] = after
    r = requests.get(url, params=p, timeout=20).json()
    d = r.get("data", [])
    if not d: break
    rows += [[int(x["fundingTime"]), float(x["realizedRate"] or x["fundingRate"])] for x in d]
    after = d[-1]["fundingTime"]
    time.sleep(0.12)
rows = sorted({r[0]: r for r in rows}.values())
json.dump(rows, open(out, "w"))
print(out, len(rows), rows[0][0] if rows else None, rows[-1][0] if rows else None)
