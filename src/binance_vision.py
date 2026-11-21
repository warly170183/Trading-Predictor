"""Descarga histórico público de Binance (data.binance.vision, futuros USDT-M) y lo une en una rejilla de 1 minuto.

Fuentes: klines 1m (con volumen agresor comprador), premiumIndexKlines 1m (base perp-índice) y metrics 5m
(OI, ratio long/short, ratio taker). Salida: data/binance/{SYM}_1m.pkl

Uso: python src/binance_vision.py --symbols BTCUSDT,ETHUSDT --months 24"""
import argparse, io, os, zipfile, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
from paths import DATA

BASE = "https://data.binance.vision/data/futures/um"
OUT = os.path.join(DATA, "binance")
os.makedirs(OUT, exist_ok=True)
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
         "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def get_zip_csv(url, retries=3):
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            return pd.read_csv(z.open(z.namelist()[0]), header=None if "klines" in url.lower() else 0)
        except Exception:
            continue
    return None


def month_list(n, today=None):
    today = today or dt.date.today()
    y, m = today.year, today.month
    out = []
    for _ in range(n):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        out.append(f"{y}-{m:02d}")
    return out[::-1]


def days_in_current_month(today=None):
    today = today or dt.date.today()
    return [(today.replace(day=1) + dt.timedelta(days=i)).isoformat()
            for i in range((today - dt.timedelta(days=1)).day) if (today.replace(day=1) + dt.timedelta(days=i)) < today]


def clean_klines(df):
    if df is None or df.empty:
        return None
    if str(df.iloc[0, 0]).strip().lower() == "open_time":  # algunos ficheros traen cabecera
        df = df.iloc[1:]
    df = df.iloc[:, :12].copy()
    df.columns = KCOLS
    return df.astype(float)


def klines(sym, kind, months, extra_days):
    urls = [f"{BASE}/monthly/{kind}/{sym}/1m/{sym}-1m-{m}.zip" for m in months]
    urls += [f"{BASE}/daily/{kind}/{sym}/1m/{sym}-1m-{d}.zip" for d in extra_days]
    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(get_zip_csv, urls))
    parts = [clean_klines(p) for p in parts]
    parts = [p for p in parts if p is not None]
    return pd.concat(parts).drop_duplicates("open_time").sort_values("open_time") if parts else None


def metrics(sym, days):
    urls = [f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{d}.zip" for d in days]
    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(get_zip_csv, urls))
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    m = pd.concat(parts)
    m["ts"] = pd.to_datetime(m["create_time"], utc=True)
    return m.drop_duplicates("ts").set_index("ts").sort_index()


def build(sym, n_months):
    months = month_list(n_months)
    extra = days_in_current_month()
    k = klines(sym, "klines", months, extra)
    p = klines(sym, "premiumIndexKlines", months, extra)
    if k is None:
        raise SystemExit(f"sin klines para {sym}")
    idx = pd.to_datetime(k.open_time, unit="ms", utc=True)
    d = pd.DataFrame({"close": k.close.values, "high": k.high.values, "low": k.low.values, "volume": k.volume.values,
                      "taker_buy": k.taker_buy_volume.values}, index=idx)
    d["flow"] = (2 * d.taker_buy - d.volume) / d.volume.replace(0, np.nan)
    if p is not None:
        pi = pd.DataFrame({"premium": p.close.values * 1e4}, index=pd.to_datetime(p.open_time, unit="ms", utc=True))
        d = d.join(pi)
    first = d.index[0].date()
    days = [(first + dt.timedelta(days=i)).isoformat() for i in range((dt.date.today() - first).days)]
    mt = metrics(sym, days)
    if mt is not None:
        mt = mt[["sum_open_interest", "count_long_short_ratio", "count_toptrader_long_short_ratio",
                 "sum_taker_long_short_vol_ratio"]].astype(float)
        mt.columns = ["oi", "ls_acc", "ls_top", "taker_ratio"]
        d = d.join(mt.reindex(d.index, method="ffill", tolerance=pd.Timedelta("10min")))
    d = d[~d.index.duplicated()].asfreq("60s")
    path = os.path.join(OUT, f"{sym}_1m.pkl")
    d.to_pickle(path)
    print(f"{sym}: {len(d)} min · {d.index[0]} → {d.index[-1]} · NaN close {d.close.isna().mean()*100:.2f}% -> {path}")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--months", type=int, default=24)
    a = ap.parse_args()
    for s in a.symbols.split(","):
        build(s, a.months)
