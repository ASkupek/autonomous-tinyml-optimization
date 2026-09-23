"""CARLA specific project configuration extending the base framework config."""

from dataclasses import dataclass, field
from typing import List
from core.configuration.base_config import GlobalConfig, PipelineConfig, AIConfig, NASConfig


@dataclass
class CarlaServerConfig:
    """CARLA simulator connection and server settings."""
    carla_host: str = "localhost"
    carla_port: int = 2000
    carla_timeout: float = 30.0
    mcu_testing: bool = True


@dataclass
class KafkaConfig:
    """Apache Kafka telemetry streaming parameters."""
    kafka_server: str = "localhost:9092"
    kafka_topic: str = "vehicle-telemetry"


@dataclass
class CarlaProjectConfig(GlobalConfig):
    """Unified CARLA project configuration extending the core framework."""
    carla: CarlaServerConfig = field(default_factory=CarlaServerConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)


CONFIG = CarlaProjectConfig(
    pipeline=PipelineConfig(
        dataset_path="use_cases/carla_driving/data/raw/carla_dataset.csv",
        scaler_path="use_cases/carla_driving/models/scaler.pkl",
        input_features=["speed", "acceleration", "lane_offset", "heading_error"],
        target_outputs=["steer"]
    ),
    ai=AIConfig(
        num_of_epochs=10,
        batch_size=64,
        learning_rate=0.001
    ),
    nas=NASConfig(
        population_size=5
    )
)