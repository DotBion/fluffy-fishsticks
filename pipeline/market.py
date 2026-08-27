"""Daily OHLCV for several tickers, from Alpha Vantage or yfinance.

Alpha Vantage stays the default because the project already provisions a key
for it. Its free tier allows 25 requests a day and 5 a minute, which is not a
constraint here: `outputsize=full` returns twenty years of history in one
call, so a five-ticker panel costs five requests total, not one per day of
data.

yfinance is the alternative when no key is available. It needs no
credentials and returns the same columns, at the cost of depending on an
unofficial endpoint that changes without notice.

    ALPHA_VANTAGE_API_KEY=... python -m pipeline.market AAPL MSFT --out panel/
    MARKET_PROVIDER=yfinance python -m pipeline.market AAPL MSFT --out panel/
"""

import os
import time

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]

# Alpha Vantage's free tier permits 5 calls a minute. One call per ticker
# with a 13-second gap keeps a panel fetch inside that without a retry loop.
AV_CALL_SPACING_SECONDS = float(os.getenv("AV_CALL_SPACING_SECONDS", "13"))


def _from_alpha_vantage(symbol):
    from alpha_vantage.timeseries import TimeSeries

    key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "ALPHA_VANTAGE_API_KEY is not set. Set it, or use "
            "MARKET_PROVIDER=yfinance which needs no credentials."
        )
    data, _ = TimeSeries(key=key, output_format="pandas").get_daily(
        symbol=symbol, outputsize="full"
    )
    data.index = pd.to_datetime(data.index)
    data = data.rename(columns={
        "1. open": "open", "2. high": "high", "3. low": "low",
        "4. close": "close", "5. volume": "volume",
    })
    data.index.name = "date"
    return data.reset_index()[["date"] + OHLCV]


def _from_yfinance(symbol):
    import yfinance as yf

    data = yf.Ticker(symbol).history(period="max", auto_adjust=False)
    if data.empty:
        raise ValueError(f"yfinance returned no rows for {symbol}")
    data = data.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    # yfinance timestamps carry an exchange timezone; the sentiment side is
    # plain calendar dates, so the join needs these naive.
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    data.index.name = "date"
    return data.reset_index()[["date"] + OHLCV]


PROVIDERS = {"alpha_vantage": _from_alpha_vantage, "yfinance": _from_yfinance}


def fetch_daily(symbol, start=None, end=None, provider=None):
    """One ticker's daily OHLCV, optionally clipped to a date window."""
    name = (provider or os.getenv("MARKET_PROVIDER", "alpha_vantage")).lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown MARKET_PROVIDER {name!r}; expected one of {sorted(PROVIDERS)}")

    df = PROVIDERS[name](symbol)
    df["date"] = pd.to_datetime(df["date"])
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError(f"No {symbol} market rows between {start} and {end}")
    return df.sort_values("date").reset_index(drop=True)


def fetch_panel(symbols, start=None, end=None, provider=None, spacing=None):
    """Daily OHLCV for several tickers as a ticker -> frame mapping.

    One symbol failing does not abandon the panel: the failure is reported
    and the rest are still returned, because a five-ticker fetch that dies on
    the fourth call has wasted the first three against the daily quota.
    """
    name = (provider or os.getenv("MARKET_PROVIDER", "alpha_vantage")).lower()
    gap = AV_CALL_SPACING_SECONDS if spacing is None else spacing

    frames, failures = {}, {}
    for i, symbol in enumerate(symbols):
        if i and name == "alpha_vantage" and gap:
            time.sleep(gap)
        try:
            frames[symbol] = fetch_daily(symbol, start, end, provider=name)
            print(f"  {symbol}: {len(frames[symbol])} rows")
        except Exception as e:
            failures[symbol] = str(e)
            print(f"  {symbol}: FAILED — {e}")

    if not frames:
        raise RuntimeError(f"Every symbol failed: {failures}")
    if failures:
        print(f"[warn] {len(failures)} of {len(symbols)} symbols failed: {sorted(failures)}")
    return frames


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2020-12-31")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--out", default=".", help="directory to write <SYMBOL>_market.csv into")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames = fetch_panel([s.upper() for s in args.symbols], args.start, args.end, args.provider)
    for symbol, frame in frames.items():
        path = os.path.join(args.out, f"{symbol}_market.csv")
        frame.to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
