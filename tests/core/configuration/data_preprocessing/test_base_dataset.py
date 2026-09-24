"""Unit tests for the core BaseDataset and BaseDataProvider contract."""

import numpy as np
import torch
import pytest

from core.data_preprocessing.base_dataset import BaseDataset
from core.data_preprocessing.base_provider import BaseDataProvider


def test_base_dataset_initialization_and_length():
    """Test da BaseDataset pravilno pretvori numpy arraye v tensore in vrne pravo dolžino."""
    X_dummy = np.random.rand(10, 4).astype(np.float32)
    y_dummy = np.random.rand(10, 1).astype(np.float32)

    dataset = BaseDataset(X_dummy, y_dummy)

    assert len(dataset) == 10

    x_sample, y_sample = dataset[0]
    assert isinstance(x_sample, torch.Tensor)
    assert isinstance(y_sample, torch.Tensor)
    assert x_sample.shape == (4,)
    assert x_sample.dtype == torch.float32


def test_base_data_provider_abstract_enforcement():
    """Test da BaseDataProvider deluje kot abstraktni razred in prepreči instanciranje brez implementacije."""
    
    with pytest.raises(TypeError):
        BaseDataProvider(config=None)