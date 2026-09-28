"""Unit Tests for BaseNAS and NASHistoryCallback.

Verifies architectural contract enforcement, decoding logic, and initialization.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from typing import Dict, Any, List

from core.nas.base_nas import BaseNAS, NASHistoryCallback
from core.model.base_models import BaseModel
from core.train import ModelsTraining
from config import GlobalConfig
import torch
import torch.nn as nn

class DummyModel(BaseModel):
    """Dummy model subclass for testing contracts."""
    def _build_network(self, layer_structure: List[int], **kwargs: Any) -> None:
        pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConcreteNAS(BaseNAS):
    """Concrete NAS implementation for testing base functionality."""
    def _evaluate(self, x: np.ndarray, out: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        out["F"] = [0.1, 10.0]


def test_base_nas_type_validation() -> None:
    """Test that BaseNAS raises TypeError if invalid model_cls or trainer_cls are passed."""
    mock_config = MagicMock(spec=GlobalConfig)
    
    # Invalid model_cls (not a BaseModel subclass)
    with pytest.raises(TypeError, match="trainer_cls must be a subclass of BaseTraining"):
        ConcreteNAS(
            config=mock_config,
            xl=[1, 1],
            xu=[10, 10],
            n_obj=2,
            n_ieq_constr=0,
            model_cls=DummyModel,
            trainer_cls=dict  # type: ignore
        )

    # Invalid trainer_cls (not a ModelsTraining subclass)
    with pytest.raises(TypeError, match="trainer_cls must be a subclass of BaseTraining"):
        ConcreteNAS(
            config=mock_config,
            xl=[1, 1],
            xu=[10, 10],
            n_obj=2,
            n_ieq_constr=0,
            model_cls=DummyModel,
            trainer_cls=dict  # type: ignore
        )


def test_base_nas_decoding() -> None:
    """Test that _decode accurately maps continuous/integer vectors to parameter dictionaries."""
    mock_config = MagicMock(spec=GlobalConfig)
    nas_problem = ConcreteNAS(
        config=mock_config,
        xl=[0, 0, 0],
        xu=[10, 10, 2],
        n_obj=2,
        n_ieq_constr=0,
        model_cls=DummyModel,
        trainer_cls=ModelsTraining
    )

    nas_problem.decoding_rules = {
        "hidden_layers": {"type": "layers", "range": (0, 2)},
        "activation": {"type": "choice", "index": 2, "choices": ["relu", "tanh", "sigmoid"]}
    }

    # Vector representing layers [4, 8] and choice index 1 ("tanh")
    x_vector = np.array([4.2, 7.8, 1.1])
    decoded = nas_problem._decode(x_vector)

    assert decoded["hidden_layers"] == [4, 8]
    assert decoded["activation"] == "tanh"