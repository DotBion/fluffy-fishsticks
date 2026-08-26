import requests
from bs4 import BeautifulSoup
from newspaper import Article
from newspaper import Config
import os

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time
from selenium.webdriver.chrome.service import Service

service = Service(executable_path=r"C:\Users\lenovo\Documents\chromedriver-win64\chromedriver.exe")

def find_news_links_selenium(query, num_articles=5):
    options = Options()
    options.headless = True  # Uncomment if you do not need a GUI
    options.add_argument("window-size=1200x600")
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')

    driver = webdriver.Chrome(options=options, service=Service(r"C:\Users\lenovo\Documents\chromedriver-win64\chromedriver.exe"))
    url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    driver.get(url)
    
    wait = WebDriverWait(driver, 10)
    links = []

    try:
        # Wait for the articles to be visible
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'article h3 a')))
        articles = driver.find_elements(By.CSS_SELECTOR, 'article h3 a')
        # articles = driver.find_elements(By.CSS_SELECTOR, 'article a.news-link')


        # Collect links
        for article in articles[:num_articles]:
            links.append(article.get_attribute('href'))

        # Navigate to each link and extract content
        article_contents = []
        for link in links:
            driver.get(link)
            time.sleep(3)  # Wait for page to load content
            body_text = driver.find_element(By.TAG_NAME, 'body').text  # Assuming body tag encompasses all text
            article_contents.append(body_text[:1000])  # Collect only first 1000 characters or adjust as needed
            print(f"Content from {link}: {body_text[:500]}...")  # Print first 500 characters of the body text

    except TimeoutException:
        print("Timed out waiting for news articles to load")
    finally:
        driver.quit()
    
    return article_contents

# def find_news_links(query, num_articles=4):
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
#     }
#     url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
#     response = requests.get(url, headers=headers)
#     if response.status_code != 200:
#         print(f"Failed to fetch news links, status code: {response.status_code}")
#         return []

#     soup = BeautifulSoup(response.text, 'html.parser')
#     links = []
    
#     for a in soup.select('article a[href]')[:num_articles]:  # Adjust selector as needed
#         link = a['href']
#         if not link.startswith('http'):
#             link = f"https://news.google.com{link}"
#         links.append(link)
#         print(f"Found link: {link}")

#     return links

def scrape_news_article(url):
    """Scrape the news article and return its content."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    config = Config()
    config.browser_user_agent = headers['User-Agent']
    article = Article(url, config=config)
    article.download()
    if article.download_state != 2:
        print(f"Failed to download article: {url}")
        return None
    article.parse()
    return f"Title: {article.title}\n\nPublish Date: {article.publish_date}\n\nContent: {article.text[:500]}..."

# Usage
query = "NVIDIA"
links = find_news_links_selenium(query)
print("Extracted Links:", links)

# Directory for output files
output_dir = 'Data/News'
os.makedirs(output_dir, exist_ok=True)

for i, url in enumerate(links, start=1):
    content = scrape_news_article(url)
    if content:
        output_file_path = os.path.join(output_dir, f'sample_{i}.txt')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Data for article {i} written to {output_file_path}")
    else:
        print(f"No content could be extracted from {url}")
