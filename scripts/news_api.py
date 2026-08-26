import requests
from newspaper import Article
import os

def fetch_news(api_key, query):
    """Fetch news URLs using NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    articles = []
    if response.status_code == 200:
        articles = response.json().get('articles', [])
    return articles

def scrape_news_article(url):
    """Scrape the full article content using newspaper3k."""
    article = Article(url)
    article.download()
    article.parse()
    return {
        "title": article.title,
        "text": article.text,
        "authors": article.authors,
        "publish_date": str(article.publish_date),
        "url": url
    }

def save_article_to_file(article, index, directory,query):
    """Save article content to a text file."""
    output_file_path = os.path.join(directory, f'{query}_{index}.txt')
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(f"Title: {article['title']}\n")
        file.write(f"Publish Date: {article['publish_date']}\n")
        file.write(f"Authors: {', '.join(article['authors'])}\n")
        file.write(f"URL: {article['url']}\n\n")
        file.write(article['text'])
    return output_file_path

# Main execution setup
api_key = '3d7606e7a31544a7a9d8ab61a72431a1'  # Replace with your actual NewsAPI key
query = 'HDFC BANK'
articles = fetch_news(api_key, query)

output_dir = 'Data/News'
os.makedirs(output_dir, exist_ok=True)

count = 15

for i, article_data in enumerate(articles, start=1):
    if i >= 15:
        break
    article_content = scrape_news_article(article_data['url'])
    if article_content:
        file_path = save_article_to_file(article_content, i, output_dir,query)
        print(f"Data for article {i} written to {file_path}")
    else:
        print(f"No content could be extracted from {article_data['url']}")


