"""Global Configuration Parameters for the CARLA-Kafka ML Pipeline.

This module acts as a single source of truth for all network settings, Kafka configurations,
data paths, and hyperparameters used across the streaming, consuming, and ML training pipelines.

Note:
    .. todo::
        **(v0.2.0)**: Create separated testing class and move needed flags there and add new ones
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# TODO(v0.2.0): Create separated testing class and move needed flags there and add new ones

@dataclass
class CarlaConfig:
    """Settings for the CARLA simulator connection, traffic manager, and sensors."""
    carla_host: str = "localhost"
    """Host address for the CARLA simulator server."""

    carla_port: int = 2000
    """Port for CARLA client-server communication."""

    carla_timeout: float = 30.0
    """Timeout limit (seconds) for waiting for responses from CARLA."""

    traffic_manager_port: int = 8000
    """Port assigned to the CARLA Traffic Manager service."""

    traffic_manager_distance: float = 3.0  
    """Safe distance maintained between spawned autonomous vehicles (m)."""

    carla_number_vehicles: int = 1
    """Total number of dynamic vehicles spawned in the CARLA simulation."""

    carla_obstacle_sensor_distance: float = 50.0
    """Max detection range for the vehicle obstacle sensor (m)."""

    carla_obstacle_sensor_reach: float = 50.0
    """Sensor beam width or reach length for detecting obstacles (m)."""

    carla_obstacle_sensor_hit_radius: float = 1.0
    """Detection bounding radius around target dynamic objects (m)."""

    carla_obstacle_sensor_only_dynamics: bool = True
    """If True, limits obstacle sensor tracking strictly to moving dynamic objects."""

    mcu_testing: bool = True
    """Execution mode flag:
    * **True**: Runs Hardware-in-the-Loop (HIL) closed-loop evaluation via direct UART stream to NXP MCU.
    * **False**: Runs standard baseline data collection / simulation pipeline.
    """
    time_for_town_train: float = 1200
    """Total duration (seconds) allocated to gather telemetry data per CARLA town."""

@dataclass
class KafkaConfig:
    """Broker connection, producer, consumer, and streaming buffer parameters for Apache Kafka."""
    kafka_server: str = "localhost:9092"
    """Kafka broker connection endpoint URL/port."""

    kafka_topic: str = "vehicle-telemetry"
    """Target Kafka topic name used for streaming telemetry messages."""

    kafka_client_id: str = "carla-telemetry-streamer"
    """Identifier sent to Kafka when making requests from the producer."""
    
    # Producer settings
    kafka_linger_ms: int = 10
    """Delay (ms) for producer to buffer and gather messages into batches."""
    kafka_batch_num_messages: int = 1000
    """Maximum number of telemetry records per batch payload."""
    kafka_acks: str = "1"
    """Acknowledgment level required by producer"""
    kafka_compression_type: str = "snappy"
    """Compression algorithm"""
    
    # Consumer settings
    kafka_group_id: str = "carla-telemetry-group"
    """Kafka consumer group identifier."""

    kafka_offset_reset: str = "latest"
    """Position to start reading messages if no initial offset is present."""

    kafka_auto_commit: bool = True
    """If True, automatically commits read message offsets periodically."""

    kafka_commit_interval_ms: int = 5000
    """Offset auto-commit frequency in milliseconds."""

    kafka_session_timeout_ms: int = 6000
    """Session heartbeat timeout (ms) before consumer is declared inactive."""

    kafka_buffer_size: int = 500
    """Maximum capacity of local message queue/buffer."""

@dataclass
class PipelineConfig:
    """Dataset features, storage paths, streaming frequency, and data splitting parameters."""

    csv_path: str = "data/raw/carla_dataset.csv"
    """Path to the raw CSV telemetry dataset file."""

    export_test_data: str = "data/processed/testdata.npz"
    """Destination path for exported NumPy preprocessed evaluation test data."""

    scaler_path: str = "data/artifacts/scaler.pkl"
    """Path to saved RobustScaler fitting parameters for dataset scaling."""

    #input_features: List[str] = field(default_factory=lambda: ["speed", "acceleration", "distance", "friction", "lane_offset", "heading_error"])
    #input_features: List[str] = field(default_factory=lambda: ["speed", "acceleration", "distance", "lane_offset", "heading_error"])
    
    input_features = ['speed', 'acceleration', 'lane_offset', 'heading_error']
    """List of telemetry input features fed into the neural network."""
    window_size: int = 10
    """Number of consecutive time steps used to form a single input sample window."""
    #target_outputs: List[str] = field(default_factory=lambda: ["throttle", "brake", "steer"])
    target_outputs: List[str] = field(default_factory=lambda: ["steer"])
    """Target output features predicted by the autonomous driving model."""

    streaming_frequency_hz: int = 20
    """Sampling frequency (Hz) for Telemetry streaming over Kafka."""

    test_size: float = 0.2
    """Fraction of dataset reserved for evaluation testing."""

    random_state: int = 42
    """Random seed applied for deterministic dataset shuffling and splits."""

    validation_size: float = 0.15
    """Fraction of dataset reserved for hyperparameter validation."""

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
    """Min/Max boundary constraints applied during dataset filtering (None = unconstrained)."""

    path_to_downsampled_dataset: str = "data/raw/carla_dataset_downsampled.csv"
    """Destination path for saving the downsampled dataset."""

    keep_ratio: float = 0.035
    """Fraction of straight-driving data samples (steer ≈ 0) retained during downsampling to prevent dataset bias."""

@dataclass
class AIConfig:
    """Hyperparameters for Neural Architecture Search (pymoo) and PyTorch model training."""

    batch_size: int = 64
    """Batch size used during PyTorch model training."""

    num_of_epochs: int = 7
    """Number of training epochs per candidate architecture."""

    num_of_hidden_layers: int = 4
    """Maximum number of hidden layers candidate networks can search over."""

    min_hidden_layer_size: int = 0
    """Minimum neuron count per hidden layer (0 indicates layer deactivation)."""

    max_hidden_layer_size: int = 128
    """Maximum neuron count permitted per hidden layer."""

    # TinyML limitations
    max_model_size_kb: float = 150.0
    """Strict TinyML constraint on Flash/RAM model footprint size in kilobytes."""

    latency_weight: float = 5.0
    """Objective weight multiplier assigned to model footprint/latency during multi-objective Pareto optimization."""
    # From 1e-6 we set to 1e-3 due to low epochs number
    learning_rate: float =  1e-3
    """Initial learning rate applied to Adam optimizer."""

    population_size: int = 3
    """Genetic algorithm population size for NAS (pymoo)."""

    n_generations: int = 3
    """Total generations for genetic search evolution."""

    exported_models_path: str = "ml_pipeline/exported_models/"
    """Directory where output models (.pt, .tflite, .h, .json) are exported."""
    steer_penalty: float = 2.5
    """DEPRECATED: Custom loss multiplier applied to steering prediction errors relative to throttle/brake loss."""

@dataclass
class PortConfiguration:
    """Parameters to configure port, for sending data to the MCXN947"""
    port_name: str = "COM3"
    """Serial COM port identifier for the physical NXP board."""

    baud_rate: int = 115200
    """Baud rate for UART serial binary packet communication."""

    timeout: int = 2
    """Serial read/write timeout limit in seconds."""

@dataclass
class GlobalConfig:
    """Root configuration holding nested sections for CARLA, Kafka, Data Pipeline, and AI/NAS."""
    carla: CarlaConfig = field(default_factory=CarlaConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    port_configuration: PortConfiguration = field(default_factory=PortConfiguration)

GLOBAL_CONFIG = GlobalConfig()