"""Unit Tests for BaseDataset.

Verifies correct tensor conversion, data types, length calculation,
and indexing functionality of the BaseDataset wrapper.
"""

import numpy as np
import torch
import pytest

from core.data_preprocessing.base_dataset import BaseDataset


def test_base_dataset_initialization_and_types() -> None:
    """Test that numpy arrays are correctly converted to torch.float32 tensors."""
    X_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y_np = np.array([0.0, 1.0], dtype=np.float64)

    dataset = BaseDataset(X_np, y_np)

    assert isinstance(dataset.X, torch.Tensor)
    assert isinstance(dataset.y, torch.Tensor)
    assert dataset.X.dtype == torch.float32
    assert dataset.y.dtype == torch.float32


def test_base_dataset_length() -> None:
    """Test that __len__ returns the correct number of samples."""
    X_np = np.zeros((50, 5))
    y_np = np.zeros((50, 1))

    dataset = BaseDataset(X_np, y_np)
    assert len(dataset) == 50


def test_base_dataset_getitem() -> None:
    """Test that __getitem__ retrieves the correct feature-target tuple at a given index."""
    X_np = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    y_np = np.array([1.0, 2.0], dtype=np.float32)

    dataset = BaseDataset(X_np, y_np)
    
    x_sample, y_sample = dataset[0]
    
    assert torch.equal(x_sample, torch.tensor([10.0, 20.0]))
    assert torch.equal(y_sample, torch.tensor(1.0))