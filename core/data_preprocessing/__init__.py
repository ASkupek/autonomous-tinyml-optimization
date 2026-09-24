"""Machine Learning Pipeline for CARLA Autonomous Driving.

This package encompasses the entire ML lifecycle, including data preprocessing,
neural network architecture definitions, training loops, and optimization .
"""

from .base_dataset import BaseDataset
from .base_provider import BaseDataProvider

__all__ = ["BaseDataset", "BaseDataProvider"]