"""Model definition for training.

The feature contract lives in serving/contract.py so training and serving
cannot drift. This module re-exports it for backwards compatibility and
defines the network itself.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from serving.backends.torch_backend import LSTMModel  # noqa: F401,E402
from serving.contract import (  # noqa: F401,E402
    FEATURE_COLS,
    INPUT_SIZE,
    MODEL_DEFAULTS as DEFAULTS,
    SEQ_LENGTH,
    TARGET_COL,
    TARGET_IDX,
)
