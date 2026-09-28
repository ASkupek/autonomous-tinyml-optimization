"""Unit Tests for ModelsTraining.

Verifies model training execution, loss injection, and training/validation loop completion.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, TensorDataset

from core.train import ModelsTraining


class SimpleTestModel(nn.Module):
    """Simple linear model for testing the trainer."""
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_models_training_execution() -> None:
    """Test that ModelsTraining successfully runs through epochs using default MSE loss."""
    # Setup mock configuration using MagicMock (trainer only needs ai.num_of_epochs & ai.learning_rate)
    mock_config = MagicMock()
    mock_config.ai.num_of_epochs = 2
    mock_config.ai.learning_rate = 1e-3

    # Setup synthetic data loaders
    X_data = torch.randn(10, 2)
    y_data = torch.randn(10, 1)
    dataset = TensorDataset(X_data, y_data)
    loader = DataLoader(dataset, batch_size=4)

    # Instantiate model and trainer
    model = SimpleTestModel()
    trainer = ModelsTraining(
        config=mock_config,
        train_loader=loader,
        validation_loader=loader,
        layer_structure=[4],
        model=model
    )

    # Run training
    trained_model = trainer.run_training()
    assert trained_model is not None
    assert trainer.epoch_validation_loss >= 0.0


def test_models_training_with_custom_loss() -> None:
    """Test that ModelsTraining correctly utilizes an injected custom loss function."""
    mock_config = MagicMock()
    mock_config.ai.num_of_epochs = 1
    mock_config.ai.learning_rate = 1e-3

    X_data = torch.randn(10, 2)
    y_data = torch.randn(10, 1)
    dataset = TensorDataset(X_data, y_data)
    loader = DataLoader(dataset, batch_size=5)

    # Custom loss function (e.g. Mean Absolute Error)
    def custom_mae_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(outputs - targets))

    model = SimpleTestModel()
    trainer = ModelsTraining(
        config=mock_config,
        train_loader=loader,
        validation_loader=loader,
        layer_structure=[4],
        model=model,
        loss_fn=custom_mae_loss
    )

    trained_model = trainer.run_training()
    assert trained_model is not None