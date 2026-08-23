from flask import Flask, request, jsonify
from flask_cors import CORS
import pathlib
import textwrap
import google.generativeai as genai
import os
# import news_api

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
def call_gemini_llm_api(query, context):
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

    response = model.generate_content(prompt_template)

    
    # q=f"You are financial insight generatiohn assistant, give a query by the user output, financial advice and provide your reasoning {context} Given this context use this to answer the following query: {query}. Answer in precise terms, give concrete 4-5 parameters, the advcie should be easy to understand, state reasonings derived from the context"
    # response = model.generate_content(q)
    print(response)
    
    # Should I invest in NVIDIA right now?
    return {"response": (format_paragraphs("{}".format(response.text)))}



comp_name =""
stock_ticker_name=""

@app.route("/")
def index():
    return "Wealth Wizardry API is running."


@app.route('/api/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data['query']
    # Load context from a predefined directory
    context = load_context_from_files(r'/Users/deeptanshu./Desktop/Wealth-Wizardry/WealthyWizard/Data/News')
    response = call_gemini_llm_api(query, context)
    return jsonify(response)

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