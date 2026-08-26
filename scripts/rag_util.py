import faiss
import numpy as np
import json
import torch
from rank_bm25 import BM25Okapi
from collections import defaultdict


def create_embeddings(model, tokenizer, texts, device):
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}  # Ensure tensors are on the correct device

    with torch.no_grad():
        # Request the hidden states explicitly
        model_output = model(**encodings, output_hidden_states=True)
        embeddings = model_output.hidden_states[-1][:, 0, :]  # Access the last layer's hidden states and get the [CLS] token embeddings

    return embeddings

def build_faiss_index(embeddings):
    embeddings = embeddings.detach().cpu().numpy()  # Make sure embeddings are moved to CPU
    d = embeddings.shape[1]
    # index = faiss.IndexFlatL2(d)
    index = faiss.IndexHNSWFlat(d)
    index.add(embeddings)
    return index

def retrieve_documents(index, query_embedding, k=5):
    # Since query_embedding might be on GPU, ensure to move it to CPU
    query_embedding = query_embedding.cpu().numpy()  # Convert to NumPy array on CPU for FAISS
    distances, indices = index.search(query_embedding, k)  # Search in FAISS index
    return indices.squeeze().tolist()

def build_index_json(data):
    index = {}
    for entry in data:
        key = entry['name'].lower()  # Index by lowercase company name for case-insensitive matching
        if key not in index:
            index[key] = []
        index[key].append(entry)
    return index

def load_data_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['rows']

def retrieve_by_company_name(index, company_name):
    key = company_name.lower()
    return index.get(key, "No data found for company")


def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    documents = []
    for entry in data["rows"]:
        doc_text = f"Company: {entry['name']}, Symbol: {entry['symbol']}, EPS: {entry['eps']}, " \
                   f"Surprise: {entry['surprise']}, Market Cap: {entry['marketCap']}, " \
                   f"Fiscal Quarter Ending: {entry['fiscalQuarterEnding']}, " \
                   f"EPS Forecast: {entry['epsForecast']}, Number of Estimates: {entry['noOfEsts']}"
        documents.append(doc_text)
    return documents


class BM25Retriever:
    def __init__(self, tokenized_corpus):
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, tokenized_query, k=5):
        scores = self.bm25.get_scores(tokenized_query)
        # Get indices of documents with the highest scores
        best_indices = np.argsort(scores)[::-1][:k]
        return best_indices


def tokenize_documents(documents):
    # Simple tokenization based on whitespace and basic punctuation
    return [doc.lower().split() for doc in documents]


# Example of preprocessing data and creating a BM25 retriever
data = load_data('NVIDIA.json')
documents = preprocess_data(data)
tokenized_documents = tokenize_documents(documents)
bm25_retriever = BM25Retriever(tokenized_documents)

# Example of using the BM25 retriever
query = "NVIDIA"
tokenized_query = query.lower().split()
retrieved_indices = bm25_retriever.retrieve(tokenized_query, k=5)
print(retrieved_indices)  # Indices of the top-k documents that match the query
