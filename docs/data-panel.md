# The training panel: what data, from where, and why that much

The model's headline weakness is sample size. `data_2018.csv` is 251 trading
days of one ticker; after windowing and a chronological split it leaves 40
held-out points, and every number the project reports rests on those 40. The
sentiment ablation's effect is smaller than the seed-to-seed spread, and the
model loses to predicting yesterday's close. Neither result can be moved by
tuning — only by more data.

This document is the plan for that.

## What bounds the panel

Not the market data. Alpha Vantage returns twenty years of daily OHLCV for
any listed symbol in a single call, and yfinance does the same for free.

The **tweet corpus** is the constraint. The model's sixth feature is daily
average sentiment, and a trading day with no tweets cannot be used — the join
is an inner join for exactly that reason. So the panel can only cover symbols
and dates the corpus covers.

| corpus | symbols | span | rows |
| --- | --- | --- | --- |
| Kaggle, *Tweet Sentiment's Impact on Stock Returns* | AAPL, AMZN, GOOG, GOOGL, MSFT, TSLA | 2015‑01‑01 → 2020‑12‑31 | ~3M tweets |
| `stockerbot-export.csv` (committed) | 453 | 2018‑02 → 2018‑07 | 28,875 |

`stockerbot-export.csv` looks like the wider option and is not: 453 symbols
across five months averages 64 tweets per symbol in total, which will not
produce a daily sentiment series for anything but the largest few names. The
Kaggle corpus is narrow and deep; that is the right trade for a daily model.

`train/tweets_2018_limited.csv` is a 10,000-row slice of the Kaggle corpus,
committed so the tests have real text to score. It spans six days, so it can
verify the pipeline but cannot train anything.

## The scope

Five tickers over six years.

| | |
| --- | --- |
| tickers | AAPL, AMZN, GOOGL, MSFT, TSLA |
| span | 2015‑01‑01 → 2020‑12‑31 |
| expected rows | ~1,510 trading days × 5 ≈ 7,500 |
| held-out sequences | ~1,500, against today's 40 |

GOOG is excluded by default. It and GOOGL are Alphabet's class C and class A
listings — the same company, tracking each other within a percent, and the
corpus tags most Alphabet tweets under both. Including both would let one
company contribute twice to the reported error while looking like
independent evidence. `pipeline.dataset.ALL_TICKERS` includes it for anyone
who wants the sixth series.

## Two rules the panel makes load-bearing

Both are enforced in `pipeline/dataset.py` and tested in
`tests/test_dataset.py`.

**Windows never span two companies.** A frame sorted by date interleaves
tickers, so a naive sliding window produces windows made of four days of AAPL
followed by six of AMZN. Sorting by (ticker, date) moves the problem rather
than solving it — the window straddling the last AAPL row and the first AMZN
row is still nonsense. Sequences are cut per ticker and concatenated. A
five-ticker panel therefore yields `n − 10` windows *per ticker*, forty fewer
than a panel-wide slide would claim.

**Each ticker gets its own scaler, fitted on the training rows only.** AMZN
traded near \$1,500 in 2018 and MSFT near \$95; one shared min–max would
compress MSFT into a sliver of [0, 1] and spend the model's capacity on price
level instead of shape. And fitting on the whole file before splitting — what
the original script did — leaks the held-out period's price range into the
normalisation. `scaler.pkl` is consequently a `{ticker: MinMaxScaler}`
mapping, which is why `/predict` accepts a `ticker` field.

## Splitting

Chronological, always. Adjacent trading days are highly correlated, so a
shuffled split lets the model interpolate between neighbours it has already
seen and reports an error far below what it would achieve on future data.

For the full panel:

    TRAIN_END=2018-12-31 VAL_END=2019-12-31

which gives train 2015–2018, validate 2019, test 2020. **Report 2020
separately.** It contains the COVID crash and the recovery; a model evaluated
across it is being asked a different question than one evaluated on 2019, and
averaging the two hides that.

For a single-year dataset, date boundaries cannot split anything — everything
lands in train. `VAL_FRACTION` splits each ticker's own series by position
instead, which is still chronological. That is the default.

## Running it

The corpus is ~3M rows and is not committed. Download it, then:

    python -m pipeline.build_panel \
        --out train/panel.csv \
        --market-dir train/market \
        --tweet-csv /path/to/Tweet.csv \
        --company-tweet-csv /path/to/Company_Tweet.csv

    cd train && DATA_CSV_PATH=panel.csv python lstm_train_pytorch.py

`--market-dir` caches `<SYMBOL>_market.csv`, so a second run costs no API
calls. Five tickers is five Alpha Vantage requests against a 25/day free
tier; `outputsize=full` means the span makes no difference to that count.
`MARKET_PROVIDER=yfinance` needs no key at all.

Scoring is the slow part — VADER over the windowed corpus, a few minutes.
`score_panel` scores each unique tweet once before joining it to the ticker
map, rather than once per ticker it mentions, which is most of the saving.

## What to expect

More data will not make this model good. Beating persistence on daily closes
is hard, and a 10-day window of OHLCV plus a daily sentiment average is a
weak signal for it. What the panel buys is the ability to *tell*: with ~1,500
held-out points instead of 40, the ablation's error bars shrink enough that
"sentiment helps" becomes a claim that can be confirmed or refuted rather
than one that is merely unfalsifiable.

If the honest answer turns out to be that sentiment does not help, that is a
result worth having, and the pipeline is the same either way.

## Beyond this corpus

Going wider than six symbols or later than 2020 means a different sentiment
source — a news API, or scraping under terms that permit it. The feature
contract does not change: anything that yields one sentiment score per symbol
per day drops into `join_market_and_sentiment` unchanged. That is the reason
sentiment scoring lives in `pipeline/sentiment.py` behind a function rather
than inline in the DAG.
