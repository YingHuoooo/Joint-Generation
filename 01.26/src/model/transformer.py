"""Full model integrating backbone and two heads."""
from __future__ import annotations

import torch

from .backbone import QwenBackbone
from .heads import SymbolicHead, PointerValueHead


class NeuroSymbolicCompiler(torch.nn.Module):
    """Backbone + SymbolicHead + PointerValueHead."""

    def __init__(self, name_or_path: str, hidden_size: int, vocab_size: int):
        super().__init__()
        self.backbone = QwenBackbone(name_or_path)
        self.symbolic_head = SymbolicHead(hidden_size, vocab_size)
        self.pointer_head = PointerValueHead(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, input_embeddings: torch.Tensor):
        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_logits = self.symbolic_head(hidden_states)
        pointer_scores = self.pointer_head(hidden_states, input_embeddings)
        return token_logits, pointer_scores
