"""
Core Data Provider Interfaces for the TinyML Framework.

This module defines abstract contracts and foundational patterns for data 
ingestion, preprocessing pipelines, and domain-specific dataset generation.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from core.configuration.base_config import GlobalConfig
from core.data_preprocessing.base_dataset import BaseDataset


class BaseDataProvider(ABC):
    """Abstract base class for domain-specific data providers.

    A data provider is responsible for obtaining and preparing data for a
    specific use case. The concrete implementation defines how the raw data
    is loaded, cleaned, transformed, split, and converted into datasets.

    The core framework only requires the provider to expose a common
    :meth:`prepare_datasets` contract. Domain-specific processing logic
    remains outside the core package.

    Args:
        config: Framework configuration containing the parameters required
            by the concrete data provider.

    Example:
        A domain-specific provider can implement the contract as follows::

            class CustomDataProvider(BaseDataProvider):

                def prepare_datasets(self):
                    # Domain-specific data preparation.
                    return train_dataset, validation_dataset, test_dataset
    """

    def __init__(self, config: GlobalConfig) -> None:
        """Initialize the data provider.

        Args:
            config: Framework configuration used by the data provider.
        """
        self.config = config

    @abstractmethod
    def prepare_datasets(self) -> Tuple[BaseDataset, BaseDataset, BaseDataset]: 
        """Prepare datasets required by the training pipeline.

        Concrete implementations are responsible for performing all
        use-case-specific data preparation required before training.

        The exact processing steps are intentionally left to the concrete
        implementation. Depending on the use case, these may include data
        loading, cleaning, filtering, splitting, scaling, windowing, or
        other transformations.

        Returns:
            A tuple containing the training, validation, and test datasets,
            respectively.
        """
        pass