"""Download daily BTC/USD candles from Bitstamp's public API into data/. Run this first."""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA, OUTPUT, FIGURES = ROOT / "data", ROOT / "output", ROOT / "figures"
FILE = DATA / "btc_daily.csv"

START = "2015-01-10"  # after Bitstamp's outage in early January 2015 (flat candles with no volume on the 6th to 8th)
LAST = "2026-11-01"  # the README's numbers stop here; move it forward to extend the sample
URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400&limit=1000&start={}"


def check(df):
    """Stop on a broken file rather than fit models to it."""
    gaps = pd.date_range(df.index[0], df.index[-1]).difference(df.index)
    if len(gaps):
        raise ValueError(f"missing days: {list(gaps[:5])}")
    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("missing values")
    if (df[["open", "high", "low", "close", "volume"]] <= 0).any().any():
        raise ValueError("a price or volume is not positive")
    if (df.high < df[["open", "close"]].max(axis=1)).any() or (df.low > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("high or low outside the day's open and close")
    return df


def fetch():
    start, rows = int(pd.Timestamp(START, tz="UTC").timestamp()), []
    while True:
        request = urllib.request.Request(URL.format(start), headers={"User-Agent": "btc_forecast_check"})
        with urllib.request.urlopen(request, timeout=60) as response:
            page = json.load(response)["data"]["ohlc"]
        rows += page
        if len(page) < 1000:
            break
        start = int(page[-1]["timestamp"]) + 86400
        time.sleep(0.3)
    df = pd.DataFrame(rows).astype(float)
    df.index = pd.to_datetime(df.pop("timestamp"), unit="s").rename("date")
    today = pd.Timestamp.now("UTC").normalize().tz_localize(None)
    df = df[df.index < today]  # today's candle is still moving
    DATA.mkdir(exist_ok=True)
    check(df).to_csv(FILE)
    return df


def load(path=FILE):
    """The sample, or with a path, a Yahoo-format file such as the paper's btc.csv."""
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index.name = "date"
    df.columns = df.columns.str.lower()
    return check(df[["open", "high", "low", "close", "volume"]]) if path != FILE else check(df.loc[:LAST])


if __name__ == "__main__":
    df = fetch()
    print(f"{len(df)} days, {df.index[0].date()} to {df.index[-1].date()}", file=sys.stderr)
