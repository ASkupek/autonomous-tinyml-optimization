"""Base configuration for the TinyML optimization framework.

This module defines the default configuration parameters used by the
framework for neural network training, neural architecture search (NAS),
and data processing.

The configuration in this module is use-case independent. Concrete use
cases may override these defaults through their own configuration files.
"""

from dataclasses import dataclass, field


@dataclass
class AIConfig:
    """Default configuration for neural network training."""

    batch_size: int = 64
    """Batch size used during model training."""

    num_of_epochs: int = 7
    """Number of training epochs for each candidate architecture."""

    learning_rate: float = 1e-3
    """Initial learning rate used by the optimizer."""


@dataclass
class NASConfig:
    """Default configuration for neural architecture search."""

    population_size: int = 3
    """Number of candidate architectures in each population."""

    n_generations: int = 3
    """Number of generations used during the architecture search."""


@dataclass
class PipelineConfig:
    """Configuration for the data processing pipeline."""

    dataset_path: str
    """Path to the dataset used by the pipeline."""

    scaler_path: str
    """Path where the fitted data scaler is stored."""

    input_features: list[str]
    """Features used as inputs to the model."""

    target_outputs: list[str]
    """Target variables predicted by the model."""

    test_size: float = 0.2
    """Fraction of the dataset reserved for the test set."""

    validation_size: float = 0.15
    """Fraction of the dataset reserved for validation."""

    random_state: int = 42
    """Random seed used for reproducible data splitting and processing."""

@dataclass
class GlobalConfig:
    """Root global configuration container."""
    pipeline: PipelineConfig
    ai: AIConfig = field(default_factory=AIConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    
