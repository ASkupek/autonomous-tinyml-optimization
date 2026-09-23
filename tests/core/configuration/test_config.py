"""Unit tests for the framework configuration system."""

import pytest
from dataclasses import dataclass, field
from core.configuration.base_config import GlobalConfig, AIConfig, NASConfig, PipelineConfig


def test_default_ai_and_nas_configs():
    """Test that base AI and NAS configurations load correct default values."""
    ai = AIConfig()
    assert ai.batch_size == 64
    assert ai.num_of_epochs == 7
    assert ai.learning_rate == 1e-3

    nas = NASConfig()
    assert nas.population_size == 3
    assert nas.n_generations == 3


def test_pipeline_config_fail_fast():
    """Test that PipelineConfig raises a TypeError if mandatory fields are missing."""
    with pytest.raises(TypeError):
        # Namerno izpustimo obvezne argumente, da preverimo Fail-Fast obnašanje
        PipelineConfig()


def test_project_config_inheritance():
    """Test that a project-specific config successfully inherits and merges with GlobalConfig."""
    
    @dataclass
    class DummyProjectConfig(GlobalConfig):
        custom_param: str = "test_value"

    # Pravilna inicializacija z obveznimi pipeline podatki
    config = DummyProjectConfig(
        pipeline=PipelineConfig(
            dataset_path="dummy/path/data.csv",
            scaler_path="dummy/path/scaler.pkl",
            input_features=["feat_1", "feat_2"],
            target_outputs=["target_1"]
        ),
        ai=AIConfig(num_of_epochs=5)
    )

    # Preverimo, če so vrednosti pravilno nastavljene in dedovane
    assert config.custom_param == "test_value"
    assert config.ai.num_of_epochs == 5              # Preglašena vrednost
    assert config.ai.batch_size == 64                # Privzeta vrednost iz jedra
    assert config.pipeline.dataset_path == "dummy/path/data.csv"
    assert config.nas.population_size == 3           # Privzeta vrednost za NAS