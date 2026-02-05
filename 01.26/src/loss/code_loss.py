"""Cross-entropy loss for code token prediction."""
from __future__ import annotations

import torch


def code_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
