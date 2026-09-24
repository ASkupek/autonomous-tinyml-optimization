"""Base model-agnostic PyTorch Dataset for the TinyML framework."""

import torch
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """Generic PyTorch Dataset that wraps numpy arrays into tensors 
    without making any assumptions about time-series or model architecture.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]