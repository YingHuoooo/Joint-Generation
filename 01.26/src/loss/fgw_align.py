"""FGW alignment loss using POT (Python Optimal Transport)."""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import ot
except Exception:  # pragma: no cover
    ot = None


def fgw_align_loss(text_graph: Tuple[np.ndarray, np.ndarray], code_graph: Tuple[np.ndarray, np.ndarray]) -> float:
    """Compute FGW loss between text and code graphs.

    text_graph: (adjacency, node_features)
    code_graph: (adjacency, node_features)
    """
    if ot is None:
        raise ImportError("POT is required for FGW loss.")

    C1, X1 = text_graph
    C2, X2 = code_graph

    p = np.ones((C1.shape[0],)) / C1.shape[0]
    q = np.ones((C2.shape[0],)) / C2.shape[0]
    M = ot.dist(X1, X2)
    loss, _ = ot.gromov.fused_gromov_wasserstein2(M, C1, C2, p, q, alpha=0.5, log=True)
    return float(loss)
