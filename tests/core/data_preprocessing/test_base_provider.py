"""Unit Tests for BaseDataProvider.

Verifies abstract class enforcement and subclass contract compliance.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from core.data_preprocessing.base_provider import BaseDataProvider
from core.data_preprocessing.base_dataset import BaseDataset
from config import GlobalConfig


def test_base_data_provider_is_abstract() -> None:
    """Test that BaseDataProvider cannot be instantiated directly because it's abstract."""
    mock_config = MagicMock(spec=GlobalConfig)
    
    with pytest.raises(TypeError):
        _ = BaseDataProvider(config=mock_config)


def test_concrete_data_provider_implementation() -> None:
    """Test that a subclass implementing prepare_datasets can be instantiated and executed."""
    
    class DummyDataProvider(BaseDataProvider):
        def prepare_datasets(self):
            dummy_x = np.zeros((10, 2), dtype=np.float32)
            dummy_y = np.zeros((10, 1), dtype=np.float32)
            ds = BaseDataset(dummy_x, dummy_y)
            return ds, ds, ds

    mock_config = MagicMock(spec=GlobalConfig)
    provider = DummyDataProvider(config=mock_config)

    assert provider.config == mock_config

    train_ds, val_ds, test_ds = provider.prepare_datasets()
    assert isinstance(train_ds, BaseDataset)
    assert isinstance(val_ds, BaseDataset)
    assert isinstance(test_ds, BaseDataset)
    assert len(train_ds) == 10