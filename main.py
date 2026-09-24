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

from use_cases.carla_driving.carla_settings import CarlaProjectConfig, CONFIG
from use_cases.carla_driving.carla_provider import CarlaDataProvider
if __name__ == "__main__":

   config = CONFIG
   data_provider = CarlaDataProvider(config=CONFIG)
   train, validation, test = data_provider.prepare_datasets()

   




    
