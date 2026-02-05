"""NLP dependency parser wrapper (spaCy/Stanza)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def parse_text_to_graph(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return adjacency matrix and node features for text."""
    tokens = text.split()
    n = len(tokens)
    adjacency = np.zeros((n, n), dtype=np.float32)
    features = np.eye(n, dtype=np.float32)
    return adjacency, features
