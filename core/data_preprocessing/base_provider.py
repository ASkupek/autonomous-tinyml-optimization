"""
Core Data Provider Interfaces for the TinyML Framework.

This module defines abstract contracts and foundational patterns for data 
ingestion, preprocessing pipelines, and domain-specific dataset generation.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from core.data_preprocessing.base_dataset import BaseDataset


class BaseDataProvider(ABC):
    """
        Abstract Base Class (ABC) defining the mandatory contract for all data providers.
        
        Every domain-specific use case (e.g., CARLA autonomous driving, audio sensors, 
        industrial IoT) must inherit from this class and implement the `prepare_datasets` 
        method. This enforces a unified architecture across the entire framework while 
        granting full implementation freedom to individual domains.

        Attributes:
            config: Configuration object containing pipeline parameters, paths, and settings.

        Example:
            >>> class CustomDataProvider(BaseDataProvider):
            ...     def prepare_datasets(self):
            ...         # Custom loading and preprocessing logic here
            ...         return train_ds, val_ds, test_ds
        """

    def __init__(self, config) -> None:
        """
        Initializes the data provider with a given configuration framework.

        Args:
            config: Configuration object containing dataset paths, feature bounds, 
                    and pipeline hyper-parameters.
        """
        self.config = config

    @abstractmethod
    def prepare_datasets(self) -> Tuple[BaseDataset, BaseDataset, BaseDataset]: 
        """
        Abstract template method for executing the full data pipeline.
        
        Concrete implementations must handle:
            1. Loading raw data (e.g., CSV, binary streams).
            2. Cleaning, filtering (e.g., boundaries, NaNs), and splitting 
               to prevent data leakage.
            3. Scaling features (e.g., StandardScaler, RobustScaler) and saving state.
            4. Windowing or formatting for time-series / model requirements.
            5. Wrapping the resulting arrays into `BaseDataset` instances.

        Returns:
            Tuple[BaseDataset, BaseDataset, BaseDataset]: A tuple containing 
            the training, validation, and testing dataset objects.
        
        Raises:
            NotImplementedError: If a concrete subclass fails to implement this method.
        """
        pass