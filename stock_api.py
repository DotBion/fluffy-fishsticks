import requests
import json
import os

def fetch_stock_data(symbol, api_key):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "compact"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    return data

def save_data_to_file(data, path, filename):
    os.makedirs(path, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(path, f"{filename}.json")
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  # Pretty print the JSON data
    print(f"Data saved to {file_path}")

api_key_alpha = os.environ["ALPHA_VANTAGE_API_KEY"]
stock_symbols = ['AMZN', 'GOOGL', 'MSFT', 'FB', 'TSLA', 'BRK.A', 'V', 'JNJ', 'WMT']
path = '/tmp/data/stock'

for symbol in stock_symbols:
    stock_data = fetch_stock_data(symbol, api_key_alpha)
    if stock_data:
        save_data_to_file(stock_data, path, symbol)
    else:
        print(f"Failed to fetch data for {symbol}")
