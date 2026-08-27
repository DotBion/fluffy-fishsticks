"""The finance-augmented VADER lexicon itself.

Plain VADER is tuned for general English and misreads market vocabulary.
These check that the override is actually reaching the analyzer - which for
thirty of its terms it was not.
"""

import pandas as pd
import pytest

from pipeline.sentiment import (
    FINANCIAL_LEXICON,
    build_analyzer,
    daily_average,
    lexicon_available,
)

pytestmark = pytest.mark.skipif(
    not lexicon_available(), reason="vader_lexicon not installed"
)


def test_the_finance_lexicon_overrides_plain_vader():
    """'sell' and 'short' are neutral in everyday English and negative here."""
    from nltk.sentiment import SentimentIntensityAnalyzer

    plain = SentimentIntensityAnalyzer()
    finance = build_analyzer()
    text = "time to sell, this is overvalued"
    assert finance.polarity_scores(text)["compound"] < plain.polarity_scores(text)["compound"]
    assert FINANCIAL_LEXICON["sell"] == -4.0
    assert FINANCIAL_LEXICON["bullish"] == 4.0


def test_every_lexicon_key_can_actually_match():
    """VADER lower-cases each token before lookup, so a capitalised key is dead.

    Thirty of the sixty-eight terms were stored capitalised and never fired.
    """
    assert [w for w in FINANCIAL_LEXICON if w != w.lower()] == []


def test_the_terms_vader_does_not_know_are_the_point():
    """Without the override these carry no polarity at all."""
    from nltk.sentiment import SentimentIntensityAnalyzer

    plain = SentimentIntensityAnalyzer()
    finance = build_analyzer()
    for word, expected in [("bullish", 1), ("plunge", -1), ("breakout", 1), ("overvalued", -1)]:
        text = f"the stock looks {word}"
        assert plain.polarity_scores(text)["compound"] == 0.0, f"vader already scores {word}"
        assert finance.polarity_scores(text)["compound"] * expected > 0, word


def test_positive_and_negative_lists_do_not_contradict():
    from pipeline.sentiment import NEGATIVE_WORDS, POSITIVE_WORDS

    pos = {w.lower() for w in POSITIVE_WORDS.split()}
    neg = {w.lower() for w in NEGATIVE_WORDS.split()}
    assert pos & neg == set(), f"terms scored both ways: {sorted(pos & neg)}"


def test_daily_average_collapses_to_one_row_per_day():
    scored = pd.DataFrame({
        "date": pd.to_datetime(["2018-01-02", "2018-01-02", "2018-01-03"]),
        "score": [0.5, -0.5, 0.2],
    })
    daily = daily_average(scored)
    assert len(daily) == 2
    assert daily.loc[0, "daily_avg_sentiment_score"] == pytest.approx(0.0)
