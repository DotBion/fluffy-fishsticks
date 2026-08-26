import requests
from bs4 import BeautifulSoup
from lxml import etree
import json
import os

# Companies and their CIK numbers
companies = {
    'Apple Inc.': '0000320193',
    'Microsoft Corporation': '0000789019',
    'Amazon.com, Inc.': '0001018724',
    'Alphabet Inc.': '0001652044',
    'Facebook, Inc.': '0001326801',
    'Berkshire Hathaway Inc.': '0001067983',
    'Johnson & Johnson': '0000200406',
    'Exxon Mobil Corporation': '0000034088',
    'JPMorgan Chase & Co.': '0000019617',
    'Visa Inc.': '0001403161'
}

# Directory for output files
output_dir = 'SEC Filings'
os.makedirs(output_dir, exist_ok=True)

# User-Agent
user_agent = 'Your Company Name info@yourcompany.com'

def scrape_sec_filings(company_code):
    base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
    headers = {
        'User-Agent': 'Apple info@apple.com'  # Replace with your actual company name and contact email
    }
    params = {
        'action': 'getcompany',
        'CIK': company_code,
        'type': '10-k',
        'dateb': '',
        'owner': 'exclude',
        'start': '0',
        'count': '40',
        'output': 'xml'
    }

    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        soup = BeautifulSoup(response.content, 'lxml')

        # for filing in soup.find_all('filing'):
        #     print('Filing Type:', filing.type.text)
        #     print('Filing Date:', filing.datefiled.text)
        #     print('Filing URL:', filing.filinghref.text)
        for filing in soup.find_all('filing'):
            filing_date = filing.datefiled.text
            if '2024' in filing_date:  # Check if the filing is from the year 2024
                return filing.filinghref.text
        return None
    except requests.exceptions.HTTPError as e:
        print("HTTP Error:", e)
    except requests.exceptions.RequestException as e:
        print("Error during requests to EDGAR:", e)
    
    return None
        
        
def download_page(url):
    headers = {
        'User-Agent': 'Meta info@meta.com'  # Use a proper user-agent
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Checks if request was successful
    return response.text

def find_document_link(html_content, doc_type='10-K'):
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find_all('table')  # Assuming the document links are in a table
    if not table:
        return None

    # Inspecting rows in the document table to find the 10-K link
    for row in table[0].find_all('tr'):
        cells = row.find_all('td')
        if len(cells) > 2:  # Ensuring there are enough cells to check
            document_type = cells[3].get_text().strip()  # Adjusted cell index for document type
            if doc_type in document_type:
                doc_link = cells[2].find('a')['href']  # Adjusted cell index for the hyperlink
                full_link = 'https://www.sec.gov' + doc_link
                return full_link
    return None

def extract_10k_data(doc_url):
    page_content = download_page(doc_url)
    print(page_content)
    
def parse_xbrl_for_metrics(xbrl_content, metrics):
    root = etree.fromstring(xbrl_content)
    results = {}
    for metric, xpath in metrics.items():
        element = root.find(xpath, namespaces=root.nsmap)
        results[metric] = element.text if element is not None else 'Not reported'
    return results

def download_xbrl(url):
    headers = {'User-Agent': 'CompanyName info@company.com'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Ensure the request was successful
    return response.content

metrics_to_extract = {
    'Total Assets': './/us-gaap:Assets',
    'Total Revenue': './/us-gaap:Revenues',
    'Net Income': './/us-gaap:NetIncomeLoss',
    'EPS Basic': './/us-gaap:EarningsPerShareBasic',
    'Long-term Debt': './/us-gaap:LongTermDebt'
}



for company_name, cik in companies.items():
    try:
        index_url = scrape_sec_filings(cik)
        print(index_url)
        index_page_content = download_page(index_url)
        document_url = find_document_link(index_page_content)
        if document_url:
            print("10-K document link found:", document_url)
            a = 'https://www.sec.gov/'
            xbrl_url = a + document_url[len(a)+8:-4] + '_htm.xml'
            xbrl_content = download_xbrl(xbrl_url)
            financial_data = parse_xbrl_for_metrics(xbrl_content, metrics_to_extract)
        else:
            print("10-K document link not found")
            continue
        if financial_data:
            file_path = os.path.join(output_dir, f"{company_name.replace(' ', '_')}_Financials.json")
            with open(file_path, 'w') as file:
                json.dump(financial_data, file, indent=4)
            print(f"Data for {company_name} written to {file_path}")
        else:
            print(f"No data found for {company_name}")
    except Exception as e:
        print(f"Failed to process {company_name}: {e}")
