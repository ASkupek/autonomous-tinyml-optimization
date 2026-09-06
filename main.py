"""Main Entry Point for Autonomous Driving TinyML Pipeline.

Executes the end-to-end machine learning lifecycle for autonomous driving in three
modular execution stages:

1. Data Collection (`get-data`):
   Streams vehicle telemetry from the CARLA simulator over Apache Kafka 
   and persists records to disk for training.

2. NAS Optimization (`train`):
   Runs multi-objective evolutionary search (NSGA-II) via pymoo to explore neural 
   network architectures, retrains top candidates, and exports INT8 quantized 
   TFLite and C header artifacts for MCU deployment.

3. HIL Verification (`test`):
   Executes Hardware-in-the-Loop (HIL) evaluation by streaming telemetry over UART
   to an NXP MCXN947 microcontroller and measuring control prediction errors.

Example:
    Run default pipeline training:
        $ python main.py --step train

    Run MCU verification in closed-loop mode with CARLA:
        $ python main.py --step test --closed-loop
"""

import argparse
from datetime import datetime
from email import parser
import sys
import threading
import time
from typing import List, Optional
from xml.parsers.expat import model

import numpy as np
from ml_pipeline import preprocessing
from ml_pipeline.export_engine import ExportEngine
from ml_pipeline.preprocessing import DataPreprocessing
from ml_pipeline.train import ModelsTraining
import torch

from config import GlobalConfig, GLOBAL_CONFIG

# TODO(v0.2.0): [BUG] Make getting data in a loop for each town, at the moment it is breaking after x mins instead of going to use next town in simulation

def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for deterministic execution.

    Args:
        seed (int, optional): Random seed initializer. Defaults to 42.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def step_get_data() -> None:
    """"Streams data from CARLA simulator to Kafka and flushes records to CSV datasets.

    Spawns an asynchronous Kafka consumer thread to log incoming vehicle telemetry 
    while sequentially iterating over pre-configured CARLA towns.

    Raises:
        Exception: Re-raises execution errors encountered during streaming setup.
    """

    from data_ingestion.carla_streaming import CarlaDataStreaming
    from data_ingestion.telemetry_consumer import TelemetryConsumer

    def run_consumer(consumer_instance: TelemetryConsumer) -> None:
        """Helper target for background consumer thread execution.

        Args:
            consumer_instance (TelemetryConsumer): Initialized Kafka consumer instance.
        """
        try:
            consumer_instance.kafka_start_listening()
        except Exception as e:
            print(f"[Consumer Error]: {e}", file=sys.stderr)

    consumer = TelemetryConsumer(config=GLOBAL_CONFIG)
    consumer_thread = threading.Thread(target=run_consumer, args=(consumer,), daemon=True)
    consumer_thread.start()

    target_maps = ["Town01", "Town02", "Town03", "Town04"]
    for town in target_maps:
        print(f"\n[Data Collection] Spawning stream for {town}...")
        engine = CarlaDataStreaming(config=GLOBAL_CONFIG, target_map=town)
        try:
            engine.connect()
            engine.spawn_vehicle()
            engine.start_streaming()
        except KeyboardInterrupt:
            print("\n[Data Collection] User interrupted streaming.")
            break
        finally:
            engine.cleanup()
            time.sleep(3)

    consumer.stop_listening()
    print("\n Data collection finished successfully.")


def step_train() -> None:
    """Executes dataset preprocessing, Evolutionary NAS, and model export.

    Preprocesses raw CSV telemetry, initializes factory functions for dynamic architecture
    instantiation, runs NSGA-II optimization, and exports top-ranked Pareto models into PyTorch,
    TFLite, and microcontroller C header formats.
    """

    from ml_pipeline.preprocessing import DataPreprocessing
    from ml_pipeline.nas import NAS, NASHistoryCallback
    from ml_pipeline.export_engine import ExportEngine
    from ml_pipeline.train import ModelsTraining
    from ml_pipeline.models import AutonomousDriving
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    import torch.nn as nn

    set_seed(42)

    # Data preprocessing
    preprocessing = DataPreprocessing(config=GLOBAL_CONFIG)
    train_loader, validation_loader, test_loader = preprocessing.read_and_preprocess_dataset()

    def trainer_factory(
        layer_structure: List[int],
        activation_type: str,
        layer_norm: bool,
        model: Optional[nn.Module] = None,
        config: Optional[GlobalConfig] = None
    ) -> ModelsTraining:
        """Factory function to instantiate ModelsTraining dynamically during NAS search.

        Args:
            layer_structure (List[int]): Proposed hidden layer configuration.
            activation_type (str): Target activation function name.
            layer_norm (bool): Flag indicating if Layer Normalization is applied.
            model (Optional[nn.Module]): Optional pre-instantiated PyTorch model.

        Returns:
            ModelsTraining: Configured trainer instance.
        """
        used_config = config if config is not None else GLOBAL_CONFIG
        return ModelsTraining(
            train_loader=train_loader,
            validation_loader=validation_loader,
            layer_structure=layer_structure,
            activation_type=activation_type,
            layer_norm=layer_norm,
            config=used_config,
            model=model
        )

    def model_factory(
        layer_structure: List[int],
        activation_type: str,
        layer_norm: bool
    ) -> AutonomousDriving:
        """Factory function to instantiate PyTorch AutonomousDriving models dynamically.

        Args:
            layer_structure (List[int]): Proposed hidden layer configuration.
            activation_type (str): Target activation function name.
            layer_norm (bool): Flag indicating if Layer Normalization is applied.

        Returns:
            AutonomousDriving: Configured dynamic PyTorch neural network.
        """
        return AutonomousDriving(
            layer_structure=layer_structure,
            activation_type=activation_type,
            layer_norm=layer_norm,
            config=GLOBAL_CONFIG
        )

    print("\n[NAS Search] Initializing Multi-Objective Evolutionary Problem...")
    problem = NAS(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        trainer_factory=trainer_factory,
        model_factory=model_factory,
        config=GLOBAL_CONFIG
    )
    
    algorithm = NSGA2(pop_size=GLOBAL_CONFIG.ai.population_size)

    timestamp_str: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    history_filename: str = f"ml_pipeline/exported_models/nas_history_{timestamp_str}.json"
    history_callback = NASHistoryCallback(output_path=history_filename)

    res = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=('n_gen', GLOBAL_CONFIG.ai.n_generations),
        callback=history_callback,
        seed=42,
        verbose=True
    )
    print("\n Full NAS Optimization completed successfully.")

    # Export models
    print("\n[Export Engine] Exporting and Retraining Top Models...")
    exporter = ExportEngine(
        config=GLOBAL_CONFIG,
        export_dir="ml_pipeline/exported_models",
        trainer_factory=trainer_factory
    )

    top_profiles = exporter.export_top_models(
        res_X=res.X,
        res_F=res.F,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        top_n=5
    )


def step_test(close_loop: bool = False) -> None:
    """Executes Hardware-in-the-Loop (HIL) MCU model evaluation.

    Args:
        close_loop (bool, optional): If True, runs evaluation in CARLA live closed-loop mode.
            If False, runs offline batch dataset verification over UART. Defaults to False.
    """

    from tinyML_CPU_execution.execute_on_mcu import ExecuteOnMCU

    tester = ExecuteOnMCU(
        config=GLOBAL_CONFIG, 
        model_rank=1, 
        close_loop_test=close_loop
    )
    
    if not close_loop:
        tester.test_on_dataset()
    else:
        tester.close_loop_testing_CARLA()

def main() -> None:
    """Parses command-line arguments and routes execution to target pipeline step."""
    parser = argparse.ArgumentParser(
        description="Autonomous Driving TinyML Optimization Pipeline CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["get-data", "train", "test"],
        default="train",
        help=(
            "Specify the pipeline execution step:\n"
            "  train    - Run Neural Architecture Search (NAS) & model training (NSGA-II)\n"
            "  get-data - Run CARLA simulator data ingestion and preprocessing\n"
            "  test     - Run Hardware-in-the-Loop (HIL) verification & MCU tests"
        )
    )

    parser.add_argument(
        "--closed-loop",
        action="store_true", # If user set is it is True otherwise is False
        help="Run closed-loop evaluation inside the CARLA simulator (applicable with '--step test')."
    )

    args = parser.parse_args()

    try:
        if args.step == "get-data":
            step_get_data()
        elif args.step == "train":
            step_train()
        elif args.step == "test":
            step_test(close_loop=args.closed_loop)
    except Exception as e:
        import traceback
        print(f"\n[CRITICAL ERROR]: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()