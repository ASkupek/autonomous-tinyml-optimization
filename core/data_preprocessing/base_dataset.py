"""
Core Data Structures for the TinyML Framework.

This module provides model-agnostic dataset wrappers to seamlessly bridge
NumPy preprocessing pipelines with PyTorch's data loading ecosystem.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """
        Generic, model-agnostic PyTorch Dataset wrapper.
        
        This class acts as a universal adapter that converts preprocessed NumPy 
        arrays (features and targets) into PyTorch tensors. It makes zero assumptions 
        about the underlying model architecture (e.g., MLP, CNN, RNN) or the data 
        modality (e.g., time-series, tabular, images), ensuring maximum reusability 
        across different TinyML use cases.

        Attributes:
            X (torch.Tensor): Feature tensor of shape (N, ...), cast to torch.float32.
            y (torch.Tensor): Target tensor of shape (N, ...), cast to torch.float32.

        Example:
            >>> X_train = np.random.rand(100, 10).astype(np.float32)
            >>> y_train = np.random.rand(100, 1).astype(np.float32)
            >>> dataset = BaseDataset(X_train, y_train)
            >>> len(dataset)
            100
            >>> x_sample, y_sample = dataset[0]
        """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initializes the dataset by converting NumPy arrays into PyTorch tensors.

        Args:
            X (np.ndarray): Input feature array from the preprocessing pipeline.
            y (np.ndarray): Target output array corresponding to the features.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples (batch dimension size).
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample (feature-target pair) at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature tensor 
            and target tensor for the given index.
        """
        return self.X[idx], self.y[idx]