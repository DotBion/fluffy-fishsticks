"""Multi-ticker sentiment scoring, against the committed tweet sample.

train/tweets_2018_limited.csv is a real slice of the Kaggle corpus: 10,000
rows, 7,990 distinct tweets, all six covered symbols, already exploded one
row per (tweet, ticker). Splitting it back into the corpus's own two-file
layout gives these tests real text to score rather than invented strings.
"""

import os

import pandas as pd
import pytest

from pipeline.sentiment import lexicon_available, score_panel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(REPO, "train", "tweets_2018_limited.csv")

# The lexicon is a runtime download, so a fresh clone that has not run
# `python -m nltk.downloader vader_lexicon` skips these rather than failing.
pytestmark = [
    pytest.mark.skipif(not os.path.exists(SAMPLE), reason="tweet sample missing"),
    pytest.mark.skipif(not lexicon_available(), reason="vader_lexicon not installed"),
]


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(SAMPLE)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, sample):
    """Rebuild the corpus's two-file layout: Tweet.csv and Company_Tweet.csv."""
    out = tmp_path_factory.mktemp("corpus")
    tweets = sample.drop_duplicates(subset="tweet_id")[
        ["tweet_id", "writer", "post_date", "body", "comment_num", "retweet_num", "like_num"]
    ]
    tweets.to_csv(out / "Tweet.csv", index=False)
    sample[["tweet_id", "ticker_symbol"]].drop_duplicates().to_csv(
        out / "Company_Tweet.csv", index=False
    )
    return str(out / "Tweet.csv"), str(out / "Company_Tweet.csv")


@pytest.fixture(scope="module")
def scored(corpus):
    tweet_csv, company_csv = corpus
    return score_panel(tweet_csv, company_csv,
                       ["AAPL", "AMZN", "MSFT", "TSLA"], "2018-01-01", "2018-01-06")


def test_every_requested_ticker_gets_a_frame(scored):
    assert sorted(scored) == ["AAPL", "AMZN", "MSFT", "TSLA"]


def test_each_frame_has_one_row_per_day(scored):
    for ticker, frame in scored.items():
        assert list(frame.columns) == ["date", "daily_avg_sentiment_score"]
        assert frame["date"].is_unique, f"{ticker} has a repeated day"
        assert frame["date"].is_monotonic_increasing


def test_scores_stay_in_vaders_range(scored):
    for ticker, frame in scored.items():
        s = frame["daily_avg_sentiment_score"]
        assert s.between(-1, 1).all(), f"{ticker} produced a score outside [-1, 1]"


def test_tickers_disagree_on_at_least_one_day(scored):
    """Different tweets per symbol, so the daily means must not be identical."""
    merged = None
    for ticker, frame in scored.items():
        col = frame.rename(columns={"daily_avg_sentiment_score": ticker})
        merged = col if merged is None else merged.merge(col, on="date", how="inner")
    values = merged.drop(columns="date")
    assert not values.nunique(axis=1).eq(1).all()


def test_the_window_is_respected(corpus):
    tweet_csv, company_csv = corpus
    scored = score_panel(tweet_csv, company_csv, ["AAPL"], "2018-01-03", "2018-01-04")
    assert scored["AAPL"]["date"].min() >= pd.Timestamp("2018-01-03")
    assert scored["AAPL"]["date"].max() <= pd.Timestamp("2018-01-04")


def test_an_empty_window_is_an_error_not_an_empty_frame(corpus):
    tweet_csv, company_csv = corpus
    with pytest.raises(ValueError, match="No tweets between"):
        score_panel(tweet_csv, company_csv, ["AAPL"], "2019-01-01", "2019-12-31")


def test_a_symbol_outside_the_corpus_is_reported_not_invented(corpus):
    tweet_csv, company_csv = corpus
    scored = score_panel(tweet_csv, company_csv, ["AAPL", "NVDA"], "2018-01-01", "2018-01-06")
    assert "NVDA" not in scored
    assert "AAPL" in scored


def test_chunked_reading_gives_the_same_answer(corpus, scored):
    """The chunk size is a memory knob, not a behaviour knob."""
    tweet_csv, company_csv = corpus
    chunked = score_panel(tweet_csv, company_csv, ["AAPL"],
                          "2018-01-01", "2018-01-06", chunksize=100)
    pd.testing.assert_frame_equal(chunked["AAPL"], scored["AAPL"])
