from flask import Flask, request, jsonify
from flask_cors import CORS
import pathlib
import textwrap
import google.generativeai as genai
import requests
import os
import csv
# import news_api

LSTM_API = os.getenv("LSTM_API", "http://localhost:9090/predict")
FINBERT_API = os.getenv("FINBERT_API", "http://localhost:5001/predict")
MARKET_CSV = os.getenv("MARKET_CSV", "../../train/data_2018.csv")
TICKER_CSV = os.getenv("TICKER_CSV", "../../stocks_cleaned.csv")
SEQ_LENGTH = 10
FEATURE_COLS = ["open", "high", "low", "close", "volume", "daily_avg_sentiment_score"]

app = Flask(__name__)
CORS(app)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
genai.configure(api_key=GEMINI_API_KEY)

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
      pass

model = genai.GenerativeModel('gemini-1.5-pro')

# query = "Should I invest in NVIDIA right now? And Why? write a social media post to convey the same"

# response = model.generate_content(query)

# print(response.text)

def _load_tickers():
    """Map lowercase company name and ticker -> ticker, from stocks_cleaned.csv."""
    lookup = {}
    try:
        with open(TICKER_CSV, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ticker = (row.get("ticker") or "").strip()
                name = (row.get(" name") or row.get("name") or "").strip()
                if ticker:
                    lookup[ticker.lower()] = ticker
                    if name:
                        lookup[name.lower()] = ticker
    except FileNotFoundError:
        app.logger.warning("Ticker list %s not found; company extraction degraded.", TICKER_CSV)
    return lookup


TICKERS = _load_tickers()


def extract_company_name(query):
    """Resolve a ticker from free text against the full ticker list."""
    lowered = query.lower()
    # Longest names first so "Berkshire Hathaway" wins over "Berkshire".
    for name in sorted(TICKERS, key=len, reverse=True):
        if len(name) > 2 and name in lowered:
            return TICKERS[name]
    for word in query.split():
        token = word.strip(".,!?$")
        cleaned = token.lower()
        if cleaned not in TICKERS:
            continue
        # Short tickers collide with ordinary words ("A" is Agilent), so a
        # lowercase token must be long enough to be unambiguous; an uppercase
        # token is taken as a deliberate ticker reference.
        if token.isupper() or len(cleaned) >= 4:
            return TICKERS[cleaned]
    return "UNKNOWN"

def get_finbert_sentiment(texts):
    """Classify recent headlines/tweets via the FinBERT service."""
    if not texts:
        return {"error": "no text supplied for sentiment analysis"}
    try:
        resp = requests.post(FINBERT_API, json={"input": texts}, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        labels = body.get("predictions", [])
        probs = body.get("probabilities", [])
        return {
            "sentiment": labels[0] if labels else None,
            "confidence": max(probs[0]) if probs else None,
            "per_text": labels,
        }
    except Exception as e:
        return {"error": f"FinBERT API error: {e}"}
def _latest_window():
    """Most recent SEQ_LENGTH rows of OHLCV + sentiment, oldest first."""
    try:
        with open(MARKET_CSV, newline="", encoding="utf-8") as fh:
            rows = sorted(csv.DictReader(fh), key=lambda r: r["date"])
    except FileNotFoundError:
        return None, f"Market data {MARKET_CSV} not found"
    if len(rows) < SEQ_LENGTH:
        return None, f"Need {SEQ_LENGTH} rows, found {len(rows)}"
    window = [[float(r[c]) for c in FEATURE_COLS] for r in rows[-SEQ_LENGTH:]]
    return window, None


def get_lstm_prediction(company_name):
    """Send a real market window to the LSTM service."""
    window, err = _latest_window()
    if err:
        return {"error": err}
    try:
        resp = requests.post(LSTM_API, json={"input": [window]}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"LSTM API error: {e}"}
    
def load_context_from_files(directory_path):
    context_data = ""
    p = pathlib.Path(directory_path)
    for file_path in p.glob('NVIDIA_*.txt'):  # Adjust the pattern if files have different extensions
        with open(file_path, "r", encoding="utf-8") as file:
            context_data += "\n" + file.read().strip()
    return context_data

def format_paragraphs(text):
    # split on blank line → wrap each in <p>
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    return ''.join(f'<p>{p}</p>' for p in paras)

# Simulate API call to Gemini LLM API (Hypothetical)
def call_gemini_llm_api(input, query, context):
    # Construct the payload including the context
    payload = {
        "parts": [
            {"text": query},    # Main query text
            # {"text": context}   # Additional context text
        ]
    }
    # print(context)
    
    # context = "Provide a detailed financial scenario here, including specific elements such as company performance indicators, market conditions, and economic forecasts."
    
    prompt_template = f"""
    You are a financial insight generation assistant. Given this context: {context} Use this to answer the following query: {query}.
    Here are the stock prediction from tommorow from our LSTM model, the company Name and sentiment score for this.
    {input}
    Structure your response as follows:
    1. Introduction: A brief introduction summarizing the context and the user's query.
    2. Detailed Analysis:
        a. Financial Health: Analyze the company’s financial stability, including key financial ratios.
        b. Market Trends: Describe current market trends affecting the scenario, including stock prices and economic indicators.
        c. Investment Risk: Identify potential risks with the proposed investment, categorized by type.
        d. Regulatory Impact: Discuss the impact of any recent or relevant financial regulations or policies.
        e. Strategic Recommendations: Provide suggestions for actions based on the analysis, tailored to the user’s goals.
    3. Conclusion: Offer concluding remarks that synthesize the analysis into actionable advice and forecasts.
    Answer in precise terms, provide concrete analysis based on these parameters. The advice should be easy to understand and include reasoning derived from the context. DO NOT give financial risk warning. Include numbers as possible.
    """

    try:
        response = model.generate_content(prompt_template)
    except Exception as e:
        return {"error": f"Gemini API error: {e}"}

    return {"response": format_paragraphs(response.text)}



comp_name =""
stock_ticker_name=""

@app.route("/")
def index():
    return "Wealth Wizardry API is running."


@app.route('/api/query', methods=['POST'])
def handle_query():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    company_name = extract_company_name(query)

    # Sentiment over any supplied headlines, falling back to the query itself.
    texts = data.get("texts") or [query]
    sentiment = get_finbert_sentiment(texts)
    lstm_output = get_lstm_prediction(company_name)

    signals = (
        f"Company: {company_name}\n"
        f"LSTM next-day close prediction: {lstm_output}\n"
        f"Sentiment: {sentiment}"
    )

    result = call_gemini_llm_api(signals, query, context="")
    if "error" in result:
        return jsonify(result), 502

    return jsonify({
        "response": result["response"],
        "company": company_name,
        "lstm": lstm_output,
        "sentiment": sentiment,
    })


@app.route('/api/com_name', methods=['POST'])
def get_comp_name():
    global comp_name,stock_ticker_name
    data = request.json
    comp_name = data['query']
    stock_ticker_prompt = f"what is the stock ticker name for {comp_name} in yfinance"
    response = model.generate_content(stock_ticker_prompt)
    stock_ticker_name = response.text
    return jsonify({'stockTicker':stock_ticker_name})
    


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)