"""Unit tests for Autonomous Driving TinyML Pipeline.

Executes baseline structural and shape assertions across configuration,
PyTorch model architecture, and factory pattern instantiations.
"""

from typing import List

import pytest
import torch
import torch.nn as nn

from config import GLOBAL_CONFIG
from ml_pipeline.models import AutonomousDriving
from ml_pipeline.train import ModelsTraining

def test_global_config_structure() -> None:
    """Verifies that key configuration parameters are defined and within valid bounds."""
    assert GLOBAL_CONFIG.ai.batch_size > 0
    assert GLOBAL_CONFIG.ai.learning_rate > 0.0
    assert len(GLOBAL_CONFIG.pipeline.input_features) == 5
    assert len(GLOBAL_CONFIG.pipeline.target_outputs) == 3

def test_autonomous_driving_model_shape() -> None:
    """Verifies forward pass tensor shapes and output bounds."""
    layer_structure = [64, 32]

    model = AutonomousDriving(
        layer_structure=layer_structure,
        activation_type="relu",
        layer_norm=True,
        config=GLOBAL_CONFIG
    )
    # 4 is num of data in batch, 5 is input_features
    dummy_input = torch.randn(size=(4, 5))
    output = model(dummy_input)
    # Output shape should be for each element in batch 3 outputs (target outputs)
    assert output.shape == (4, 3)
    # First two are throttle and break and should be between 0 and 1 sigmoid, last is steer and should be -1 to 1 tanh
    assert torch.all(output[:, 0:2] >= 0.0) and torch.all(output[:, 0:2] <= 1.0)
    assert torch.all(output[:, 2] >= -1.0) and torch.all(output[:, 2] <= 1.0)

def test_model_factory_pattern() -> None:
    """Verifies that dynamic model generation via model_factory returns valid instances."""
    def model_factory(layer_structure: List[int], activation_type: str, layer_norm: bool) -> AutonomousDriving:
        return AutonomousDriving(
            layer_structure=layer_structure,
            activation_type=activation_type,
            layer_norm=layer_norm,
            config=GLOBAL_CONFIG
        )

    model_instance = model_factory(layer_structure=[128, 64], activation_type="tanh", layer_norm=False)
    assert isinstance(model_instance, nn.Module)
    assert isinstance(model_instance, AutonomousDriving)