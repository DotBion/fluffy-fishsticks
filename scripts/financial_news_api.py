import requests
import json
import os

top_companies = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'FB', 'TSLA', 'BRK.A', 'V', 'JNJ', 'WMT']

def fetch_news_sentiment(symbol, api_key):
    """Fetch news sentiment for a given stock symbol."""
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

def save_data_to_file(data, filename):
    """Save JSON data to a file."""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

api_key = os.environ["ALPHA_VANTAGE_API_KEY"]
output_dir = '../Data/NewsSentiment'
os.makedirs(output_dir, exist_ok=True)

for symbol in top_companies:
    data = fetch_news_sentiment(symbol, api_key)
    file_path = os.path.join(output_dir, f'{symbol}_news_sentiment.json')
    save_data_to_file(data, file_path)
    print(f"Data for {symbol} saved to {file_path}")

