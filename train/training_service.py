"""Training trigger service.

workflows/train-model.yaml POSTs to /trigger-training and reads
.new_model_version from the response to drive the build. That endpoint
previously existed only in the course's gourmetgram app, so the continuous
training chain had nothing to call.

Runs training in a background thread so the HTTP request does not block for
the length of a training run; the workflow polls /status.
"""

import os
import threading

from flask import Flask, jsonify

from lstm_train_pytorch import MODEL_PATH, SCALER_PATH, train

app = Flask(__name__)

_lock = threading.Lock()
_state = {"running": False, "last_version": None, "last_error": None, "last_metrics": None}


def _register(model, scalers, metrics):
    """Persist artifacts and log to MLflow. Returns the new registry version.

    `scalers` is the ticker -> MinMaxScaler mapping training produced; it is
    pickled whole so a multi-ticker model keeps every scaler it was fitted
    with, and serving.contract.ScalerBundle reads it back unchanged.
    """
    import torch
    from joblib import dump

    torch.save(model.state_dict(), MODEL_PATH)
    dump(scalers, SCALER_PATH)

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set; cannot register a model version.")

    import mlflow
    import mlflow.pytorch

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "finpulse-lstm"))
    model_name = os.getenv("MLFLOW_MODEL_NAME", "FinPulseLSTM")
    with mlflow.start_run():
        mlflow.log_param("tickers", ",".join(sorted(scalers)))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(SCALER_PATH)
        mlflow.log_artifact(MODEL_PATH)
        import torch as _torch

        from models import DEFAULTS, FEATURE_COLS

        example = _torch.randn(1, DEFAULTS["seq_length"], len(FEATURE_COLS)).numpy()
        mlflow.pytorch.log_model(
            model.cpu(), name="model", input_example=example,
            registered_model_name=model_name,
        )

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    return max(int(v.version) for v in versions)


def _run():
    try:
        from pipeline.data_source import resolve_training_csv

        data_path, _ = resolve_training_csv(local_fallback="data_2018.csv")
        tickers = [t for t in os.getenv("TICKERS", "").replace(",", " ").split() if t]
        model, scalers, metrics = train(
            data_path=data_path, verbose=False, tickers=tickers or None
        )
        version = _register(model, scalers, metrics)
        _state.update(last_version=version, last_metrics=metrics, last_error=None)
    except Exception as e:
        _state["last_error"] = str(e)
    finally:
        _state["running"] = False


@app.route("/health")
def health():
    return jsonify({"status": "ok", "training_in_progress": _state["running"]}), 200


@app.route("/status")
def status():
    return jsonify(_state), 200


@app.route("/trigger-training", methods=["POST"])
def trigger_training():
    """Kick off a training run.

    Responds 202 with new_model_version=null while running; the caller polls
    /status. When a previous run has completed, the finished version is
    returned so a synchronous caller still sees a usable value.
    """
    with _lock:
        if _state["running"]:
            return jsonify({"status": "already_running", "new_model_version": None}), 409
        _state.update(running=True, last_error=None)
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    if os.getenv("TRAIN_SYNC", "false").lower() == "true":
        # Blocking mode: the Argo step reads new_model_version directly.
        thread.join(timeout=float(os.getenv("TRAIN_TIMEOUT", "3600")))
        if thread.is_alive():
            return jsonify({"status": "timeout", "new_model_version": None}), 504
        if _state["last_error"]:
            return jsonify({"status": "failed", "error": _state["last_error"]}), 500
        return jsonify({"status": "completed", "new_model_version": _state["last_version"]}), 200

    return jsonify({"status": "started", "new_model_version": _state["last_version"]}), 202


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "9090")))
