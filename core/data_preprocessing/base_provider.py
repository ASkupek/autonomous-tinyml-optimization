"""Base data provider interface for the TinyML framework."""

from abc import ABC, abstractmethod
from typing import Tuple
from core.data_preprocessing.base_dataset import BaseDataset


class BaseDataProvider(ABC):
    """Abstract base provider defining the mandatory contract for all data pipelines."""

    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def prepare_datasets(self) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
        """
        Must be implemented by concrete use-cases to load, process, 
        and return (train_dataset, test_dataset).
        """
        pass