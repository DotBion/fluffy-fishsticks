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

FINANCIAL_LEXICON = {
    **{w: 4.0 for w in POSITIVE_WORDS.split()},
    **{w: -4.0 for w in NEGATIVE_WORDS.split()},
}


def build_analyzer():
    """VADER analyzer with the financial lexicon applied."""
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

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
