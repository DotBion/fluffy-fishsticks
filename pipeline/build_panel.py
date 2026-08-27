"""Build the multi-ticker training panel from market data and tweet sentiment.

Produces one CSV with a `ticker` column that train/lstm_train_pytorch.py
consumes directly:

    date,open,high,low,close,volume,daily_avg_sentiment_score,ticker

The tweet corpus, not the market data, bounds what is possible. Market
history is available for any listed symbol and any year, but the Kaggle
corpus covers AAPL, AMZN, GOOG, GOOGL, MSFT and TSLA over 2015-2020, and a
row with no sentiment cannot feed a model whose sixth feature is sentiment.
Asking for a seventh ticker or a 2021 window yields an empty join, so the
defaults here are the corpus's actual extent.

    python -m pipeline.build_panel --out train/panel.csv \
        --tweet-csv /data/Tweet.csv --company-tweet-csv /data/Company_Tweet.csv

The market side can be skipped when the CSVs are already on disk:

    python -m pipeline.build_panel --out train/panel.csv --market-dir panel/ ...
"""

import argparse
import os

import pandas as pd

from pipeline.dataset import DEFAULT_TICKERS, build_panel, describe_panel
from pipeline.sentiment import join_market_and_sentiment, score_panel

# The corpus's extent. Not a preference — asking for more returns nothing.
CORPUS_START = "2015-01-01"
CORPUS_END = "2020-12-31"


def _market_frames(symbols, start, end, market_dir, provider):
    """Read <SYMBOL>_market.csv from disk, or fetch what is not there."""
    frames, to_fetch = {}, []
    for symbol in symbols:
        path = os.path.join(market_dir or "", f"{symbol}_market.csv")
        if market_dir and os.path.exists(path):
            frames[symbol] = pd.read_csv(path)
            print(f"  {symbol}: {len(frames[symbol])} rows from {path}")
        else:
            to_fetch.append(symbol)

    if to_fetch:
        from pipeline.market import fetch_panel

        print(f"Fetching market data for {', '.join(to_fetch)}")
        frames.update(fetch_panel(to_fetch, start, end, provider=provider))
        if market_dir:
            os.makedirs(market_dir, exist_ok=True)
            for symbol in to_fetch:
                frames[symbol].to_csv(os.path.join(market_dir, f"{symbol}_market.csv"), index=False)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--start", default=CORPUS_START)
    ap.add_argument("--end", default=CORPUS_END)
    ap.add_argument("--tweet-csv", default=os.getenv("TWEET_CSV", "Tweet.csv"))
    ap.add_argument("--company-tweet-csv",
                    default=os.getenv("COMPANY_TWEET_CSV", "Company_Tweet.csv"))
    ap.add_argument("--market-dir", default=None,
                    help="cache directory for <SYMBOL>_market.csv")
    ap.add_argument("--provider", default=None, help="alpha_vantage (default) or yfinance")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    symbols = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    for path in (args.tweet_csv, args.company_tweet_csv):
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} not found. The Kaggle 'Tweet Sentiment's Impact on Stock "
                "Returns' corpus is ~3M rows and is not committed to this repo; "
                "download it and pass --tweet-csv/--company-tweet-csv."
            )

    print(f"Market data for {', '.join(symbols)} ({args.start} .. {args.end})")
    market = _market_frames(symbols, args.start, args.end, args.market_dir, args.provider)

    print(f"\nScoring tweets from {args.tweet_csv}")
    sentiment = score_panel(args.tweet_csv, args.company_tweet_csv,
                            symbols, args.start, args.end)

    joined = {}
    for symbol in symbols:
        if symbol not in market or symbol not in sentiment:
            print(f"[warn] skipping {symbol}: "
                  f"{'no market data' if symbol not in market else 'no sentiment'}")
            continue
        merged = join_market_and_sentiment(market[symbol], sentiment[symbol])
        if merged.empty:
            print(f"[warn] skipping {symbol}: market and sentiment share no dates")
            continue
        joined[symbol] = merged

    if not joined:
        raise SystemExit("Nothing to write — every symbol lost its market/sentiment join.")

    panel = build_panel(joined)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    panel.to_csv(args.out, index=False)

    print(f"\n{describe_panel(panel).to_string()}")
    print(f"\nWrote {len(panel)} rows to {args.out}")


if __name__ == "__main__":
    main()
