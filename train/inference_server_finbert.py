"""Flask inference server for tweet sentiment classification.

Loads the fine-tuned FinBERT produced by train_finbert.py when it is present,
and otherwise falls back to the pretrained base model so the server always
starts. Which one is in use is reported by GET /health.
"""

import os

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

# train_finbert.py writes its fine-tuned output here.
FINETUNED_DIR = os.getenv("FINBERT_MODEL_DIR", "finbert_model")
BASE_MODEL = os.getenv("FINBERT_BASE_MODEL", "yiyanghkust/finbert-tone")

if os.path.isdir(FINETUNED_DIR):
    model_source = FINETUNED_DIR
    is_finetuned = True
else:
    # Falling back rather than crashing: the fine-tuned artifact is not in the
    # repo, so a fresh clone would otherwise be unable to start this server.
    model_source = BASE_MODEL
    is_finetuned = False
    print(
        f"[warn] {FINETUNED_DIR!r} not found — falling back to pretrained {BASE_MODEL!r}. "
        f"Run 'python train_finbert.py' to produce the fine-tuned model."
    )

tokenizer = AutoTokenizer.from_pretrained(model_source)
model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=3)
model.eval()

# Matches the ordering train_finbert.py assigns via score_to_label().
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


@app.route("/")
def home():
    return "FinBERT Inference Server is running!"


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_source": model_source,
        "fine_tuned": is_finetuned,
        "labels": list(LABEL_MAP.values()),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if "input" not in data:
            return jsonify({"error": "Missing 'input' key in JSON payload"}), 400

        texts = data["input"]
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return jsonify({"error": "'input' should be a list of strings."}), 400
        if not texts:
            return jsonify({"error": "'input' must contain at least one string."}), 400

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(**inputs).logits, dim=1)
            predicted = torch.argmax(probs, dim=1).tolist()

        return jsonify({
            "predictions": [LABEL_MAP[i] for i in predicted],
            "probabilities": probs.tolist(),
            "fine_tuned": is_finetuned,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")))
