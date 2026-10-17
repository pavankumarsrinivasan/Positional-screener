"""
NSE 500 Momentum Ranking Screener
----------------------------------
Fully free data sources:
  - NSE500 constituent list : NSE's own CSV (nseindia.com)
  - Price history           : yfinance (Yahoo Finance), unofficial but free

Logic (per Kiru's description of the AQR/Asness 52-week-high momentum approach):
    ratio = latest month-end close / 52-week high AS OF THE PRIOR MONTH
Rank all NSE500 stocks by that ratio descending; top N = highest momentum.
Stocks with under ~230 trading days of prior history (recent IPOs) are
excluded, since a short "52-week high" is trivially easy to beat and
would otherwise inflate their ratio unfairly.

Run this monthly (1st trading day of the month, after the previous month has
closed) e.g. via a GitHub Actions cron job. Takes roughly 5-10 minutes for
~500 tickers because of politeness delays against Yahoo Finance.

After ranking, the top 30 stocks are also written into the current calendar
month's column of the "Positional Screener Monthly Updater" Google Sheet,
using a service account whose credentials are read from the
GOOGLE_SERVICE_ACCOUNT_JSON environment variable (set as a GitHub Actions
secret — see the repo's workflow file for how it's wired in).

pip install pandas requests yfinance gspread google-auth
"""

import io
import json
import os
import time
import datetime as dt

import gspread
import pandas as pd
import requests
import yfinance as yf
from google.oauth2.service_account import Credentials

NSE500_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"

# --- Google Sheet the monthly top-30 list gets written into ---
GOOGLE_SHEET_ID = "1pQwWkZxsNodAvUvfkjxgZRu-yZV-s3LebkVo6IrMm_Q"
GOOGLE_WORKSHEET_NAME = "Sheet1"        # tab name in "Positional Screener Monthly Updater"
SHEET_FIRST_DATA_ROW = 2               # row where each month's tickers start (row 1 = month headers)
SHEET_STOCKS_PER_MONTH = 30            # matches the highlight_sell_column.gs script's assumptions

# Minimum trading days required in the prior-52-week window before a stock
# is eligible to be ranked. Recently-listed stocks (IPOs) haven't had a full
# year to trade, so their "52-week high" is really just "highest close since
# listing" — a much lower bar that artificially inflates their ratio. ~230
# trading days is roughly 11 months, filtering out anything without close to
# a full year of history while still allowing for holidays/gaps.
MIN_HISTORY_DAYS = 230

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

    # Recent yfinance versions return multi-level columns (ticker, field)
    # even for a single-ticker download — flatten so hist["Close"] is a
    # plain Series instead of a nested one-column frame.
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    close_series = hist["Close"].dropna()
    latest_close = close_series.iloc[-1]

    prior_window_end = asof - pd.DateOffset(months=1)
    prior_52w = close_series[close_series.index <= prior_window_end].tail(252)
    if len(prior_52w) < MIN_HISTORY_DAYS:
        # Not enough trading history yet (recent IPO, etc.) — skip rather
        # than let a short, easily-beaten "52-week high" inflate the ratio.
        print(f"[skip] {ticker}: only {len(prior_52w)} days of prior history "
              f"(need {MIN_HISTORY_DAYS})")
        return None
    prior_high = prior_52w.max()

    return float(latest_close), float(prior_high)


def rank_nse500() -> pd.DataFrame:
    """Return the FULL ranked table (all eligible NSE500 stocks), sorted by
    ratio descending, and also save it to nse500_momentum_ranked.csv."""
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

    ranked = pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)
    ranked.to_csv("nse500_momentum_ranked.csv", index=False)
    print("\nSaved full ranking to nse500_momentum_ranked.csv")
    return ranked


def write_top30_to_sheet(ranked: pd.DataFrame, asof: dt.date) -> None:
    """Write the top 30 tickers into the current month's column of the
    Google Sheet. Column A = January ... column L = December, matching
    asof.month directly (1-12).

    Requires GOOGLE_SERVICE_ACCOUNT_JSON (the service account's JSON key,
    as a single-line string) to be set as an environment variable, and that
    service account's email to have Editor access on the target sheet.
    """
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        print("[sheet] GOOGLE_SERVICE_ACCOUNT_JSON not set — skipping Google Sheet update.")
        return

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(GOOGLE_SHEET_ID)
    ws = sh.worksheet(GOOGLE_WORKSHEET_NAME)

    month_col = asof.month  # 1 = Jan ... 12 = Dec -> column A..L

    top30 = ranked.head(SHEET_STOCKS_PER_MONTH)["symbol"].str.replace(".NS", "", regex=False).tolist()
    top30 += [""] * (SHEET_STOCKS_PER_MONTH - len(top30))  # pad so stale rows don't linger

    cell_range = ws.range(
        SHEET_FIRST_DATA_ROW, month_col,
        SHEET_FIRST_DATA_ROW + SHEET_STOCKS_PER_MONTH - 1, month_col,
    )
    for cell, value in zip(cell_range, top30):
        cell.value = value
    ws.update_cells(cell_range)

    print(f"[sheet] Wrote top {SHEET_STOCKS_PER_MONTH} tickers into column {month_col} "
          f"of '{GOOGLE_WORKSHEET_NAME}'.")


if __name__ == "__main__":
    ranked = rank_nse500()
    print("\nTop 10 momentum candidates:")
    print(ranked.head(10).to_string(index=False))

    write_top30_to_sheet(ranked, dt.date.today())
