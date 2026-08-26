"""Inference backends. Selected at startup by the BACKEND env var."""

import os


def load_backend(name=None):
    """Return an initialized backend exposing .predict_scaled(array) and .describe()."""
    name = (name or os.getenv("BACKEND", "torch")).lower()
    if name == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend()
    if name == "onnx":
        from .onnx_backend import OnnxBackend

        return OnnxBackend()
    raise ValueError(f"Unknown BACKEND {name!r}; expected 'torch' or 'onnx'.")
