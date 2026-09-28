"""Core Neural Network Base Model.

This module defines the abstract base model for dynamic neural network architectures
within the TinyML optimization framework, standardizing activation function resolution
and layer construction contracts.
"""

from abc import ABC, abstractmethod
from typing import Any, List

import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """Abstract Base Neural Network Model for dynamic architectures.

    Handles generic tasks like activation function resolution and provides 
    a blueprint for dynamic layer construction.
    """

    def __init__(self) -> None:
        """Initializes the base neural network module."""
        super().__init__()

    def _get_activation_function(self, activation_type: str) -> nn.Module:
        """Returns the PyTorch activation function based on the specified type string.

        Args:
            activation_type (str): Type of activation function ('relu', 'tanh', 'sigmoid').

        Returns:
            nn.Module: Corresponding PyTorch activation function module.

        Raises:
            ValueError: If an unsupported activation string is provided.
        """
        activation_type = activation_type.lower()
        if activation_type == "relu":
            return nn.ReLU()
        elif activation_type == "tanh":
            return nn.Tanh()
        elif activation_type == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"[BaseModel Error] Unsupported activation type: {activation_type}")

    @abstractmethod
    def _build_network(self, layer_structure: List[int], **kwargs: Any) -> None:
        """Abstract hook for building domain-specific network layers.

        Args:
            layer_structure (List[int]): List defining the architecture dimensions.
            **kwargs: Additional parameters.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Abstract forward pass method that must be implemented by domain models.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output prediction tensor.
        """
        pass