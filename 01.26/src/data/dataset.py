"""PyTorch Dataset for multimodal samples."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch.utils.data import Dataset


class DeepCADDataset(Dataset):
    def __init__(self, items: list[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]
