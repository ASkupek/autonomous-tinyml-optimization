"""CARLA specific project configuration extending the base framework config."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from core.configuration.base_config import GlobalConfig, PipelineConfig, AIConfig, NASConfig


@dataclass
class CarlaServerConfig:
    """CARLA simulator connection and server settings."""
    carla_host: str = "localhost"
    carla_port: int = 2000
    carla_timeout: float = 30.0
    traffic_manager_port: int = 8000
    traffic_manager_distance: float = 3.0  
    carla_number_vehicles: int = 1
    carla_obstacle_sensor_distance: float = 50.0
    carla_obstacle_sensor_reach: float = 50.0
    carla_obstacle_sensor_hit_radius: float = 1.0
    carla_obstacle_sensor_only_dynamics: bool = True
    mcu_testing: bool = True
    time_for_town_train: float = 1200


@dataclass
class KafkaConfig:
    """Apache Kafka telemetry streaming parameters."""
    kafka_server: str = "localhost:9092"
    kafka_topic: str = "vehicle-telemetry"
    kafka_client_id: str = "carla-telemetry-streamer"
    kafka_linger_ms: int = 10
    kafka_batch_num_messages: int = 1000
    kafka_acks: str = "1"
    kafka_compression_type: str = "snappy"
    kafka_group_id: str = "carla-telemetry-group"
    kafka_offset_reset: str = "latest"
    kafka_auto_commit: bool = True
    kafka_commit_interval_ms: int = 5000
    kafka_session_timeout_ms: int = 6000
    kafka_buffer_size: int = 500

@dataclass
class PortConfiguration:
    """Parameters to configure port, for sending data to the MCXN947."""
    port_name: str = "COM3"
    baud_rate: int = 115200
    timeout: int = 2

@dataclass
class CarlaPipelineConfig(PipelineConfig):
    window_size: int = 10
    streaming_frequency_hz: int = 20
    path_to_downsampled_dataset: str = "use_cases/carla_driving/data/raw/carla_dataset_downsampled.csv"
    export_test_data: str = "use_cases/carla_driving/data/processed/testdata.npz"
    keep_ratio: float = 0.035
    feature_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = field(
        default_factory=lambda: {
            'speed': (0.0, None),
            'acceleration': (0.0, None),
            'distance': (0.0, 50.0),
            'throttle': (0.0, 1.0),
            'brake': (0.0, 1.0),
            'steer': (-1.0, 1.0),
        }
    )

@dataclass
class CarlaAIConfig(AIConfig):
    """CARLA-specifične AI in TinyML nastavitve."""
    num_of_hidden_layers: int = 4
    min_hidden_layer_size: int = 0
    max_hidden_layer_size: int = 128
    max_model_size_kb: float = 150.0
    latency_weight: float = 5.0
    exported_models_path: str = "use_cases/carla_driving/exported_models/"
    steer_penalty: float = 2.5

@dataclass
class CarlaProjectConfig(GlobalConfig):
    """Unified CARLA project configuration extending the core framework."""
    carla: CarlaServerConfig = field(default_factory=CarlaServerConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    port_configuration: PortConfiguration = field(default_factory=PortConfiguration)

    pipeline: CarlaPipelineConfig = field(default_factory=lambda: CarlaPipelineConfig(
        dataset_path="use_cases/carla_driving/data/raw/carla_dataset.csv",
        scaler_path="use_cases/carla_driving/models/scaler.pkl",
        input_features=["speed", "acceleration", "lane_offset", "heading_error"],
        target_outputs=["steer"]
    ))
    ai: CarlaAIConfig = field(default_factory=lambda: CarlaAIConfig(
        num_of_epochs=10,
        batch_size=64,
        learning_rate=0.001
    ))

CONFIG = CarlaProjectConfig()