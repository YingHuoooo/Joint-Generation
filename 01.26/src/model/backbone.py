"""Backbone wrapper for Qwen2.5-Coder encoder-only hidden states."""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModel, AutoConfig


class QwenBackbone(torch.nn.Module):
    """Load pretrained Qwen2.5-Coder and expose hidden states only."""

    def __init__(self, name_or_path: str):
        super().__init__()
        config = AutoConfig.from_pretrained(name_or_path)
        config.output_hidden_states = True
        self.model = AutoModel.from_pretrained(name_or_path, config=config)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
