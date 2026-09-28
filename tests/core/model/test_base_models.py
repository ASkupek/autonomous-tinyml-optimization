"""Unit Tests for BaseModel.

Verifies abstract constraints enforcement, activation function resolution,
and error handling for unsupported activation types.
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Any

from core.model.base_models import BaseModel


class DummyConcreteModel(BaseModel):
    """Concrete dummy subclass for testing BaseModel functionality."""
    
    def _build_network(self, layer_structure: List[int], **kwargs: Any) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_base_model_is_abstract() -> None:
    """Test that BaseModel cannot be instantiated directly due to abstract methods."""
    with pytest.raises(TypeError):
        _ = BaseModel()


def test_get_activation_function_valid() -> None:
    """Test that valid activation strings return correct PyTorch modules."""
    model = DummyConcreteModel()

    assert isinstance(model._get_activation_function("relu"), nn.ReLU)
    assert isinstance(model._get_activation_function("TANH"), nn.Tanh)  # Case-insensitive check
    assert isinstance(model._get_activation_function("sigmoid"), nn.Sigmoid)


def test_get_activation_function_invalid() -> None:
    """Test that an unsupported activation string raises a ValueError."""
    model = DummyConcreteModel()

    with pytest.raises(ValueError, match="Unsupported activation type"):
        model._get_activation_function("invalid_activation")