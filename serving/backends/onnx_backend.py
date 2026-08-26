"""ONNX Runtime backend — the optimized path intended for deployment."""

import os

import numpy as np
import onnxruntime as ort


class OnnxBackend:
    name = "onnx"

    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv("ONNX_MODEL_PATH", "lstm_model.onnx")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "4"))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Request CUDA only when the installed onnxruntime actually offers it;
        # naming an unavailable provider makes newer versions raise.
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]

        self.session = ort.InferenceSession(self.model_path, opts, providers=providers)
        # Read the input name from the graph rather than assuming "input".
        self.input_name = self.session.get_inputs()[0].name

    def predict_scaled(self, scaled):
        outputs = self.session.run(None, {self.input_name: scaled.astype(np.float32)})
        return np.atleast_1d(outputs[0]).astype(np.float32)

    def describe(self):
        return {
            "backend": self.name,
            "model": self.model_path,
            "providers": self.session.get_providers(),
        }
