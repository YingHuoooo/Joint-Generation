"""Symbolic and pointer-value heads."""
from __future__ import annotations

import torch


class SymbolicHead(torch.nn.Module):
    """Predict symbolic tokens such as workplane/box/hole."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class PointerValueHead(torch.nn.Module):
    """Pointer mechanism to select numeric spans from input text."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, input_embeddings: torch.Tensor) -> torch.Tensor:
        # simple dot-product attention
        query = self.query(hidden_states)
        scores = torch.matmul(query, input_embeddings.transpose(-1, -2))
        return scores
