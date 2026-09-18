"""
NSE 500 Momentum Ranking Screener
----------------------------------
Fully free data sources:
  - NSE500 constituent list : NSE's own CSV (nseindia.com)
  - Price history           : yfinance (Yahoo Finance), unofficial but free

Logic (per Kiru's description of the AQR/Asness 52-week-high momentum approach):
    ratio = latest month-end close / 52-week high AS OF THE PRIOR MONTH
Rank all NSE500 stocks by that ratio descending; top N = highest momentum.

Run this monthly (1st trading day of the month, after the previous month has
closed) e.g. via a GitHub Actions cron job. Takes roughly 5-10 minutes for
~500 tickers because of politeness delays against Yahoo Finance.

pip install pandas requests yfinance
"""

import io
import time
import datetime as dt

import pandas as pd
import requests
import yfinance as yf

NSE500_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"

# NSE blocks requests without a browser-like User-Agent / referer and without
# first "warming up" a session to pick up cookies.
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}


def get_nse500_symbols() -> list[str]:
    """Download today's NSE 500 constituent list directly from NSE.

    NOTE: this is always the *current* constituent list (NSE rebalances the
    index twice a year), so it's correct for live signal generation but
    introduces survivorship bias if you try to use it for a long historical
    backtest without also sourcing historical index membership separately.
    """
    with requests.Session() as s:
        s.headers.update(NSE_HEADERS)
        s.get("https://www.nseindia.com", timeout=10)  # sets cookies NSE expects
        resp = s.get(NSE500_CSV_URL, timeout=15)
        resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # NSE's CSV has a "Symbol" column; append .NS for yfinance's NSE suffix
    return [f"{sym.strip()}.NS" for sym in df["Symbol"]]


def month_close_and_prior_52w_high(ticker: str, asof: dt.date):
    """Return (latest_close, prior_52_week_high) for a ticker as of `asof`.

    prior_52_week_high looks at the 52 weeks BEFORE the current month began,
    matching "close (this month) / 52-week-high (as of end of last month)".
    """
    start = asof - dt.timedelta(days=400)  # buffer: 52 weeks + gaps/holidays
    hist = yf.download(
        ticker, start=start, end=asof + dt.timedelta(days=1),
        interval="1d", auto_adjust=True, progress=False,
    )
    if hist.empty:
        return None

    close_series = hist["Close"].dropna()
    latest_close = close_series.iloc[-1]

    prior_window_end = asof - pd.DateOffset(months=1)
    prior_52w = close_series[close_series.index <= prior_window_end].tail(252)
    if prior_52w.empty:
        return None
    prior_high = prior_52w.max()

    return float(latest_close), float(prior_high)


def rank_nse500(top_n: int = 10) -> pd.DataFrame:
    symbols = get_nse500_symbols()
    today = dt.date.today()
    rows = []

    for i, sym in enumerate(symbols, 1):
        try:
            result = month_close_and_prior_52w_high(sym, today)
            if result is None:
                continue
            close, prior_high = result
            rows.append({
                "symbol": sym,
                "close": close,
                "prior_52w_high": prior_high,
                "ratio": close / prior_high,
            })
        except Exception as e:
            print(f"[skip] {sym}: {e}")

        if i % 25 == 0:
            print(f"...processed {i}/{len(symbols)}")
        time.sleep(0.3)  # be polite to Yahoo Finance's unofficial endpoint

    ranked = pd.DataFrame(rows).sort_values("ratio", ascending=False)
    ranked.to_csv("nse500_momentum_ranked.csv", index=False)
    print("\nSaved full ranking to nse500_momentum_ranked.csv")
    return ranked.head(top_n)


if __name__ == "__main__":
    top = rank_nse500(top_n=10)
    print("\nTop momentum candidates:")
    print(top.to_string(index=False))
