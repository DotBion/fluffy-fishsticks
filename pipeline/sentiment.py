"""Daily tweet sentiment scoring with a finance-augmented VADER lexicon.

Ported from Sentiment_analysis_with_twitter_data_for_Apple.ipynb on the
dev-nc3610 branch, where this step existed only as a Colab notebook. That
made data_2018.csv - the training set - unreproducible. This module makes
the step runnable from the Airflow DAG and from the command line.
"""

import pandas as pd

# Plain VADER is tuned for general English and misreads market vocabulary:
# "sell", "short" and "miss" are not negative in everyday usage, and "bull"
# or "breakout" are not positive. These weights (+/-4 on VADER's -4..4 scale)
# push the analyzer toward finance-domain polarity.
POSITIVE_WORDS = (
    "high profit Growth Potential Opportunity Bullish Strong Valuable Success "
    "Promising Profitable Win Winner Outstanding Record Earnings Breakthrough "
    "buy bull long support undervalued underpriced cheap upward rising trend "
    "moon rocket hold breakout call beat support buying holding"
)
NEGATIVE_WORDS = (
    "resistance squeeze cover seller Risk Loss Decline Bearish Weak Declining "
    "Uncertain Troubling Downturn Struggle Unstable Volatile Slump Disaster "
    "Plunge sell bear bubble bearish short overvalued overbought overpriced "
    "expensive downward falling sold sell low put miss"
)

# Keys are lower-cased deliberately. VADER lower-cases each token before it
# looks it up, so an entry stored as "Bullish" can never match: thirty of the
# sixty-nine terms above are capitalised, and every one of them was inert
# until this line. The word lists are left exactly as the original notebook
# wrote them so the intended vocabulary stays readable.
FINANCIAL_LEXICON = {
    **{w.lower(): 4.0 for w in POSITIVE_WORDS.split()},
    **{w.lower(): -4.0 for w in NEGATIVE_WORDS.split()},
}


def lexicon_available():
    """True when VADER's own lexicon is already on disk."""
    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        return True
    except LookupError:
        return False


def build_analyzer():
    """VADER analyzer with the financial lexicon applied."""
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    if not lexicon_available():
        # The automatic download fails behind a proxy that will not let NLTK
        # pin the resolved address, and the resulting LookupError says
        # nothing about how to fix it. Name the one command that does.
        if not nltk.download("vader_lexicon", quiet=True) or not lexicon_available():
            raise RuntimeError(
                "VADER's lexicon is not installed and could not be downloaded. "
                "Run: python -m nltk.downloader vader_lexicon"
            )

    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(FINANCIAL_LEXICON)
    return sia


def load_tweets(tweet_csv, company_tweet_csv):
    """Join the Kaggle tweet corpus to its ticker mapping and add a date column."""
    tweets = pd.read_csv(tweet_csv)
    company = pd.read_csv(company_tweet_csv)
    tweets = tweets.merge(company, how="left", on="tweet_id")
    tweets["date"] = pd.to_datetime(
        pd.to_datetime(tweets["post_date"], unit="s").dt.date, errors="coerce"
    )
    return tweets


def score_tweets(tweets, ticker, start, end, analyzer=None):
    """Compound sentiment per tweet for one ticker over a date range."""
    sia = analyzer or build_analyzer()
    mask = (
        (tweets["ticker_symbol"] == ticker)
        & (tweets["date"] >= pd.Timestamp(start))
        & (tweets["date"] <= pd.Timestamp(end))
    )
    df = tweets.loc[mask, ["date", "body", "tweet_id"]].copy()
    if df.empty:
        raise ValueError(f"No tweets for {ticker} between {start} and {end}")
    df["score"] = df["body"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
    return df


def daily_average(scored):
    """Collapse per-tweet scores to one mean score per calendar day."""
    daily = scored.groupby(scored["date"].dt.date)["score"].mean()
    return pd.DataFrame({
        "date": pd.to_datetime(daily.index),
        "daily_avg_sentiment_score": daily.values,
    })


def join_market_and_sentiment(market, sentiment):
    """Inner-join OHLCV to daily sentiment on date.

    Inner join is deliberate: a trading day with no tweets would otherwise
    carry a NaN sentiment feature into training.
    """
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])
    # Drop any sentiment column already on the market frame (e.g. re-running
    # against an previously joined file); otherwise pandas silently produces
    # _x/_y suffixes and the training contract breaks downstream.
    market = market.drop(columns=["daily_avg_sentiment_score"], errors="ignore")
    merged = market.merge(sentiment, on="date", how="inner").sort_values("date")
    return merged[
        ["date", "open", "high", "low", "close", "volume", "daily_avg_sentiment_score"]
    ]


def score_panel(tweet_csv, company_tweet_csv, tickers, start, end,
                analyzer=None, chunksize=500_000):
    """Daily sentiment for several tickers: {ticker: frame}.

    Two things differ from calling score_tweets once per ticker.

    The corpus is scored before it is joined to the ticker mapping. A tweet
    naming three companies appears three times after the join, and VADER
    would otherwise score the identical text three times; over a five-ticker
    panel that is most of the runtime for no extra information.

    The date filter is applied while reading. Tweet.csv is several million
    rows, so the window is applied per chunk rather than after loading the
    whole file into memory.
    """
    sia = analyzer or build_analyzer()
    tickers = [t.upper() for t in tickers]
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)

    kept = []
    for chunk in pd.read_csv(tweet_csv, chunksize=chunksize):
        chunk["date"] = pd.to_datetime(
            pd.to_datetime(chunk["post_date"], unit="s").dt.date, errors="coerce"
        )
        kept.append(chunk.loc[chunk["date"].between(lo, hi), ["tweet_id", "body", "date"]])

    tweets = pd.concat(kept, ignore_index=True).drop_duplicates(subset="tweet_id")
    if tweets.empty:
        raise ValueError(f"No tweets between {start} and {end} in {tweet_csv}")

    tweets["score"] = tweets["body"].map(lambda t: sia.polarity_scores(str(t))["compound"])

    company = pd.read_csv(company_tweet_csv)
    company["ticker_symbol"] = company["ticker_symbol"].str.upper()
    tagged = tweets.merge(company[company["ticker_symbol"].isin(tickers)],
                          how="inner", on="tweet_id")

    out = {}
    for ticker, group in tagged.groupby("ticker_symbol"):
        out[ticker] = daily_average(group)
    missing = sorted(set(tickers) - set(out))
    if missing:
        print(f"[warn] no tweets in the window for: {missing}")
    return out
